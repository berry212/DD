from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PixArtAlphaPipeline,
    StableDiffusionPipeline,
    Transformer2DModel,
    UNet2DConditionModel,
)
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from .datasets import get_dataset_spec, supported_datasets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sample_indices(total_samples: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= total_samples:
        return np.arange(total_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.asarray(rng.choice(total_samples, size=max_samples, replace=False), dtype=np.int64)


def default_data_root() -> Path:
    root = os.getenv("HF_DATASETS_CACHE") or os.getenv("HF_HOME", "data")
    return Path(root).expanduser()


def default_data_npz_path(dataset_name: str) -> Path:
    return default_data_root() / f"{dataset_name}_224.npz"


def default_lora_output_dir(dataset_name: str) -> Path:
    return Path("outputs") / f"lora_{dataset_name}"


def compute_snr(noise_scheduler: DDPMScheduler, timesteps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    alpha = sqrt_alphas_cumprod[timesteps]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    return (alpha / sigma) ** 2


def create_class_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    bincount = np.bincount(labels.reshape(-1).astype(np.int64, copy=False))
    bincount = np.clip(bincount, a_min=1, a_max=None)
    sample_weights = 1.0 / bincount[labels.reshape(-1)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=int(labels.shape[0]),
        replacement=True,
    )


def create_lr_scheduler(
    optimizer: AdamW,
    total_optimizer_steps: int,
    warmup_steps: int,
    schedule: str,
) -> LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    total_optimizer_steps = max(1, int(total_optimizer_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if schedule == "cosine":
            progress = float(step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def strip_variance_channels(
    model_pred: torch.Tensor,
    target_channels: int,
    variance_type: str | None,
) -> torch.Tensor:
    if model_pred.shape[1] == target_channels:
        return model_pred

    if model_pred.shape[1] == 2 * target_channels:
        warnings.warn(
            "DiT output includes variance channels; dropping variance prediction for loss.",
            RuntimeWarning,
        )
        return model_pred[:, :target_channels]

    raise ValueError(
        "DiT output channels do not match latent channels: "
        f"pred={model_pred.shape[1]} target={target_channels} variance_type={variance_type}"
    )


def load_dit_transformer(model_id: str, dtype: torch.dtype) -> Transformer2DModel:
    try:
        return Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
        )
    except Exception:
        return Transformer2DModel.from_pretrained(model_id, torch_dtype=dtype)


def load_vae(model_id: str, dtype: torch.dtype) -> AutoencoderKL:
    try:
        return AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype)
    except Exception:
        return AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)


def load_scheduler(model_id: str) -> DDPMScheduler:
    try:
        return DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    except Exception:
        return DDPMScheduler.from_pretrained(model_id)


def save_lora_artifacts(output_dir: Path, transformer: nn.Module, lora_config: LoraConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_state_dict = get_peft_model_state_dict(transformer)
    torch.save(lora_state_dict, output_dir / "lora_transformer.pt")
    with open(output_dir / "lora_config.json", "w", encoding="utf-8") as handle:
        json.dump(lora_config.to_dict(), handle, indent=2)


class MedMNISTClassDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        split_data: Any,
        sample_indices: np.ndarray,
        resolution: int,
        class_label_offset: int,
    ) -> None:
        self.split_data = split_data
        self.sample_indices = np.ascontiguousarray(sample_indices, dtype=np.int64)
        self.class_label_offset = int(class_label_offset)

        if hasattr(split_data, "get_all_labels"):
            self.labels = np.asarray(split_data.get_all_labels()).reshape(-1).astype(np.int64, copy=False)
        else:
            self.labels = np.asarray(split_data.labels).reshape(-1).astype(np.int64, copy=False)

        self.images = split_data.imgs if hasattr(split_data, "imgs") else None
        self._split_accessor = split_data if hasattr(split_data, "get_image_and_label") else None

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_idx = int(self.sample_indices[index])
        if self._split_accessor is not None:
            image, label = self._split_accessor.get_image_and_label(source_idx)
        else:
            image = self.images[source_idx]
            label = int(self.labels[source_idx])

        image = np.asarray(image)
        label = int(label)
        pixel = self.transform(image)
        pixel = pixel * 2.0 - 1.0
        class_label = int(label) + self.class_label_offset
        return {
            "pixel_values": pixel,
            "class_labels": torch.tensor(class_label, dtype=torch.long),
        }


def resolve_backbone_type(model_id: str, backbone_type: str) -> str:
    if backbone_type and backbone_type != "auto":
        return backbone_type
    model_lower = model_id.lower()
    if any(kw in model_lower for kw in ("pixart", "dit", "transformer", "sd3", "flux")):
        return "dit"
    return "unet"


class MedMNISTTextDataset(Dataset[dict[str, torch.Tensor]]):
    """CLIP-tokenized dataset for Stable Diffusion / UNet path."""

    def __init__(
        self,
        split_data: Any,
        sample_indices: np.ndarray,
        tokenizer: CLIPTokenizer,
        resolution: int,
        prompt_dropout_prob: float,
        class_prompts: dict[int, str],
        default_prompt: str,
    ) -> None:
        self.split_data = split_data
        self.sample_indices = np.ascontiguousarray(sample_indices, dtype=np.int64)
        self.tokenizer = tokenizer
        self.prompts = dict(class_prompts)
        self.default_prompt = default_prompt
        self.prompt_dropout_prob = float(np.clip(prompt_dropout_prob, 0.0, 1.0))

        if hasattr(split_data, "get_all_labels"):
            self.labels = np.asarray(split_data.get_all_labels()).reshape(-1).astype(np.int64, copy=False)
        else:
            self.labels = np.asarray(split_data.labels).reshape(-1).astype(np.int64, copy=False)

        self.images = split_data.imgs if hasattr(split_data, "imgs") else None
        self._split_accessor = split_data if hasattr(split_data, "get_image_and_label") else None

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_idx = int(self.sample_indices[index])
        if self._split_accessor is not None:
            image, label = self._split_accessor.get_image_and_label(source_idx)
        else:
            image = self.images[source_idx]
            label = int(self.labels[source_idx])

        image = np.asarray(image)
        label = int(label)
        prompt = self.prompts.get(label, f"{self.default_prompt} class {label}")
        if self.prompt_dropout_prob > 0.0 and random.random() < self.prompt_dropout_prob:
            prompt = ""

        pixel = self.transform(image)
        pixel = pixel * 2.0 - 1.0

        # CLIP tokenization
        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel,
            "input_ids": text_inputs.input_ids[0],
        }


class MedMNISTTextDatasetDiT(Dataset[dict[str, torch.Tensor]]):
    """Like MedMNISTTextDataset but uses T5 tokenizer for PixArt-α / DiT."""

    def __init__(
        self,
        split_data: Any,
        sample_indices: np.ndarray,
        tokenizer: T5Tokenizer,
        resolution: int,
        prompt_dropout_prob: float,
        class_prompts: dict[int, str],
        default_prompt: str,
    ) -> None:
        self.split_data = split_data
        self.sample_indices = np.ascontiguousarray(sample_indices, dtype=np.int64)
        self.tokenizer = tokenizer
        self.prompts = dict(class_prompts)
        self.default_prompt = default_prompt
        self.prompt_dropout_prob = float(np.clip(prompt_dropout_prob, 0.0, 1.0))

        if hasattr(split_data, "get_all_labels"):
            self.labels = np.asarray(split_data.get_all_labels()).reshape(-1).astype(np.int64, copy=False)
        else:
            self.labels = np.asarray(split_data.labels).reshape(-1).astype(np.int64, copy=False)

        self.images = split_data.imgs if hasattr(split_data, "imgs") else None
        self._split_accessor = split_data if hasattr(split_data, "get_image_and_label") else None

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_idx = int(self.sample_indices[index])
        if self._split_accessor is not None:
            image, label = self._split_accessor.get_image_and_label(source_idx)
        else:
            image = self.images[source_idx]
            label = int(self.labels[source_idx])

        image = np.asarray(image)
        label = int(label)
        prompt = self.prompts.get(label, f"{self.default_prompt} class {label}")
        if self.prompt_dropout_prob > 0.0 and random.random() < self.prompt_dropout_prob:
            prompt = ""

        pixel = self.transform(image)
        pixel = pixel * 2.0 - 1.0

        # T5 tokenization (different from CLIP)
        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        return {
            "pixel_values": pixel,
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
        }


def _save_dit_lora_weights(
    transformer: Transformer2DModel,
    lora_config: LoraConfig,
    save_dir: Path,
) -> None:
    import json as _json
    from safetensors.torch import save_file

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    state_dict = get_peft_model_state_dict(transformer)
    save_file(state_dict, str(save_dir / "pytorch_lora_weights.safetensors"))

    # Save PEFT config so distillate.py can reconstruct the adapter
    config_dict = {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
        "lora_dropout": lora_config.lora_dropout,
        "bias": lora_config.bias,
        "task_type": "CAUSAL_LM",  # generic enough for transformer
        "peft_type": "LORA",
        "init_lora_weights": lora_config.init_lora_weights,
    }
    with open(save_dir / "lora_config.json", "w", encoding="utf-8") as f:
        _json.dump(config_dict, f, indent=2)
    print(f"[LoRA-Train] Saved DiT LoRA to {save_dir}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset_spec = get_dataset_spec(args.dataset)
    data_root = Path(args.data_root).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    if args.data_npz:
        print("[LoRA-Train] --data-npz is deprecated and ignored; loading data directly from dataset split.")

    if args.prompt_dropout_prob:
        warnings.warn("--prompt-dropout-prob is ignored for DiT training.", RuntimeWarning)

    output_dir = Path(args.output_dir) if args.output_dir else default_lora_output_dir(dataset_spec.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_bundle = dataset_spec.load_dataset_splits(data_root=str(data_root), image_size=args.resolution)
    train_split = split_bundle.train_set
    labels_all = np.asarray(
        train_split.get_all_labels() if hasattr(train_split, "get_all_labels") else train_split.labels,
        dtype=np.int64,
    ).reshape(-1)
    sample_indices = build_sample_indices(len(labels_all), args.max_train_samples, args.seed)
    labels = labels_all[sample_indices]

    print(f"[LoRA-Train] dataset={dataset_spec.name} data_root={data_root}")
    label_values, label_counts = np.unique(labels, return_counts=True)
    class_distribution = {int(k): int(v) for k, v in zip(label_values.tolist(), label_counts.tolist())}
    print(f"[LoRA-Train] class_distribution={class_distribution}")

    class_prompts = dataset_spec.build_class_prompts()
    default_prompt = dataset_spec.prompt_prefix
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    weight_dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    backbone_type = resolve_backbone_type(args.diffusion_model_id, args.backbone_type)
    print(f"[LoRA-Train] backbone_type={backbone_type} model_id={args.diffusion_model_id}")

    if backbone_type == "dit":
        # ---------- DiT / PixArt-α path ----------
        tokenizer = T5Tokenizer.from_pretrained(args.diffusion_model_id, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            args.diffusion_model_id, subfolder="text_encoder",
            torch_dtype=weight_dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            args.diffusion_model_id, subfolder="vae", torch_dtype=weight_dtype
        )
        transformer = Transformer2DModel.from_pretrained(
            args.diffusion_model_id, subfolder="transformer", torch_dtype=weight_dtype
        )
        noise_scheduler = DDPMScheduler.from_pretrained(args.diffusion_model_id, subfolder="scheduler")

        # Move all encoders to GPU for training speed
        vae.to(device)
        text_encoder.to(device)
        transformer.to(device)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # Enable gradient checkpointing on T5 to reduce activation memory
        if hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()
        transformer.requires_grad_(False)

        # Target both self-attn (attn1) and cross-attn (attn2) in DiT blocks
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        # Enable gradient checkpointing on base model before PEFT wrap
        if hasattr(transformer, "enable_gradient_checkpointing"):
            try:
                transformer.enable_gradient_checkpointing()
            except Exception:
                pass

        transformer = get_peft_model(transformer, lora_config)
        if weight_dtype == torch.float16 and device.type == "cuda":
            cast_training_params(transformer, dtype=torch.float32)
        transformer.train()

        trainable_params = [p for p in transformer.parameters() if p.requires_grad]
        trainable_param_count = int(sum(p.numel() for p in trainable_params))
        print(f"[LoRA-Train] trainable_params={trainable_param_count}")

        dataset = MedMNISTTextDatasetDiT(
            split_data=train_split,
            sample_indices=sample_indices,
            tokenizer=tokenizer,
            resolution=args.resolution,
            prompt_dropout_prob=args.prompt_dropout_prob,
            class_prompts=class_prompts,
            default_prompt=default_prompt,
        )
    else:
        # ---------- UNet / Stable Diffusion path ----------
        tokenizer = CLIPTokenizer.from_pretrained(args.diffusion_model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            args.diffusion_model_id, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        vae = AutoencoderKL.from_pretrained(
            args.diffusion_model_id, subfolder="vae", torch_dtype=weight_dtype
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.diffusion_model_id, subfolder="unet", torch_dtype=weight_dtype
        )
        noise_scheduler = DDPMScheduler.from_pretrained(args.diffusion_model_id, subfolder="scheduler")

        text_encoder.to(device)
        vae.to(device)
        unet.to(device)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        unet.add_adapter(lora_config)
        if weight_dtype == torch.float16 and device.type == "cuda":
            cast_training_params(unet, dtype=torch.float32)
        unet.train()

        trainable_params = [p for p in unet.parameters() if p.requires_grad]
        trainable_param_count = int(sum(p.numel() for p in trainable_params))
        print(f"[LoRA-Train] trainable_params={trainable_param_count}")

        dataset = MedMNISTTextDataset(
            split_data=train_split,
            sample_indices=sample_indices,
            tokenizer=tokenizer,
            resolution=args.resolution,
            prompt_dropout_prob=args.prompt_dropout_prob,
            class_prompts=class_prompts,
            default_prompt=default_prompt,
        )

    sampler = create_class_balanced_sampler(labels) if args.class_balance else None
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if sampler is not None:
        print("[LoRA-Train] using class-balanced sampling")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    grad_accum_steps = max(1, int(args.gradient_accumulation_steps))
    steps_per_epoch = math.ceil(len(data_loader) / grad_accum_steps)
    total_optimizer_steps = int(args.max_train_steps) if args.max_train_steps > 0 else int(args.epochs * steps_per_epoch)
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        total_optimizer_steps=total_optimizer_steps,
        warmup_steps=args.lr_warmup_steps,
        schedule=args.lr_schedule,
    )

    print(
        f"[LoRA-Train] grad_accum_steps={grad_accum_steps} "
        f"steps_per_epoch={steps_per_epoch} total_optimizer_steps={total_optimizer_steps}"
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=(weight_dtype == torch.float16 and device.type == "cuda"))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
        print(f"[LoRA-Train] GPU memory: allocated={allocated_mb:.0f} MiB  reserved={reserved_mb:.0f} MiB")

    denoiser = transformer if backbone_type == "dit" else unet

    global_step = 0
    running_loss = 0.0
    running_steps = 0
    optimizer.zero_grad(set_to_none=True)

    stop_training = False
    for epoch in range(1, args.epochs + 1):
        for batch_idx, batch in enumerate(data_loader, start=1):
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)

            with torch.inference_mode():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * float(getattr(vae.config, "scaling_factor", 0.18215))

                if backbone_type == "dit":
                    # T5 on GPU with gradient checkpointing; clean up inputs after encoding
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
                    encoder_hidden_states = text_encoder(
                        input_ids, attention_mask=attn_mask
                    ).last_hidden_state
                    del input_ids, attn_mask
                else:
                    input_ids = batch["input_ids"].to(device)
                    encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            if args.noise_offset > 0.0:
                noise = noise + args.noise_offset * torch.randn(
                    (bsz, latents.shape[1], 1, 1),
                    device=device,
                    dtype=latents.dtype,
                )

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )

            sampled_noise = noise
            if args.input_perturbation > 0.0:
                sampled_noise = noise + args.input_perturbation * torch.randn_like(noise)

            noisy_latents = noise_scheduler.add_noise(latents, sampled_noise, timesteps)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(weight_dtype == torch.float16)):
                if backbone_type == "dit":
                    # PixArt requires resolution / aspect-ratio conditioning
                    latent_h, latent_w = noisy_latents.shape[2], noisy_latents.shape[3]
                    img_h, img_w = latent_h * 8, latent_w * 8
                    bsz_dit = noisy_latents.shape[0]
                    added_cond_kwargs = {
                        "resolution": torch.tensor([img_h, img_w]).repeat(bsz_dit, 1).to(
                            device=device, dtype=weight_dtype
                        ),
                        "aspect_ratio": torch.tensor([float(img_h / img_w)]).repeat(bsz_dit, 1).to(
                            device=device, dtype=weight_dtype
                        ),
                    }
                    model_pred = denoiser(
                        hidden_states=noisy_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timesteps,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    # PixArt learned-sigma: output has 2× channels, 1st half is noise
                    if model_pred.shape[1] != latents.shape[1]:
                        model_pred = model_pred.chunk(2, dim=1)[0]
                else:
                    model_pred = denoiser(noisy_latents, timesteps, encoder_hidden_states).sample
                    # UNet learned-sigma: output has 2× channels, 1st half is noise
                    if model_pred.shape[1] != latents.shape[1]:
                        model_pred = model_pred.chunk(2, dim=1)[0]

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                model_pred = strip_variance_channels(
                    model_pred,
                    target_channels=target.shape[1],
                    variance_type=getattr(noise_scheduler.config, "variance_type", None),
                )

                if args.snr_gamma > 0.0:
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = torch.minimum(snr, args.snr_gamma * torch.ones_like(snr)) / snr
                    per_pixel_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    per_sample_loss = per_pixel_loss.mean(dim=tuple(range(1, per_pixel_loss.ndim)))
                    loss = (per_sample_loss * mse_loss_weights).mean()
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == len(data_loader))
            if not should_step:
                continue

            if args.max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            global_step += 1
            running_loss += float(loss.item()) * grad_accum_steps
            running_steps += 1

            if global_step % args.log_steps == 0:
                avg_loss = running_loss / max(running_steps, 1)
                print(
                    f"[LoRA-Train] epoch={epoch:03d}/{args.epochs} "
                    f"step={global_step:06d}/{total_optimizer_steps:06d} "
                    f"loss={avg_loss:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                running_loss = 0.0
                running_steps = 0

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop_training = True
                break

        if args.save_every_epoch:
            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if backbone_type == "dit":
                _save_dit_lora_weights(transformer=transformer, lora_config=lora_config, save_dir=ckpt_dir)
            else:
                unet_lora_state_dict = get_peft_model_state_dict(unet)
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=str(ckpt_dir),
                    unet_lora_layers=unet_lora_state_dict,
                    safe_serialization=True,
                )

        if stop_training:
            break

    if backbone_type == "dit":
        _save_dit_lora_weights(transformer=transformer, lora_config=lora_config, save_dir=output_dir)
    else:
        unet_lora_state_dict = get_peft_model_state_dict(unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(output_dir),
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

    summary = {
        "dataset": dataset_spec.name,
        "diffusion_model_id": args.diffusion_model_id,
        "backbone_type": backbone_type,
        "data_root": str(data_root),
        "data_npz": str(Path(args.data_npz).expanduser()) if args.data_npz else "",
        "train_samples_used": int(len(dataset)),
        "resolution": int(args.resolution),
        "rank": int(args.rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_target_modules": target_modules,
        "class_label_offset": int(args.class_label_offset),
        "epochs": int(args.epochs),
        "max_train_steps": int(args.max_train_steps),
        "global_step": int(global_step),
        "trainable_params": trainable_param_count,
        "effective_batch_size": int(args.batch_size * grad_accum_steps),
        "gradient_accumulation_steps": int(grad_accum_steps),
        "class_balance": bool(args.class_balance),
        "lr": float(args.lr),
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "max_grad_norm": float(args.max_grad_norm),
        "snr_gamma": float(args.snr_gamma),
        "noise_offset": float(args.noise_offset),
        "input_perturbation": float(args.input_perturbation),
        "device": device.type,
        "output_dir": str(output_dir),
    }

    with open(output_dir / "lora_train_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[LoRA-Train] Completed")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune DiT on supported ophthalmic/medical datasets"
    )
    parser.add_argument(
        "--dataset",
        default="dermamnist",
        choices=supported_datasets(),
    )
    parser.add_argument("--data-root", type=str, default=str(default_data_root()))
    parser.add_argument("--data-npz", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--diffusion-model-id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Diffusion model: SD1.5 for UNet, PixArt-alpha/PixArt-XL-2-1024-MS for DiT")
    parser.add_argument("--backbone-type", type=str, default="auto",
                        choices=["auto", "unet", "dit"],
                        help="'auto' detects from model_id; 'unet' for SD; 'dit' for PixArt/DiT")

    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=3000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="to_k,to_q,to_v,to_out.0",
        help="Comma-separated DiT module names for LoRA injection.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--noise-offset", type=float, default=0.05)
    parser.add_argument("--input-perturbation", type=float, default=0.0)
    parser.add_argument(
        "--prompt-dropout-prob",
        type=float,
        default=0.0,
        help="Deprecated; ignored for DiT training.",
    )
    parser.add_argument("--class-balance", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-steps", type=int, default=20)

    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--class-label-offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
