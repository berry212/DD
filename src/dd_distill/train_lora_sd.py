from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from medmnist import INFO
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

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


def load_medmnist_train(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(npz_path)
    expected = {"train_images", "train_labels"}
    missing = expected.difference(arrays.files)
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {sorted(missing)}")
    return arrays["train_images"], arrays["train_labels"].reshape(-1)


def maybe_subsample(images: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or max_samples >= len(images):
        return images, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=max_samples, replace=False)
    return images[idx], labels[idx]


def build_dataset_prompts(dataset_spec: Any) -> tuple[dict[int, str], str]:
    labels = INFO[dataset_spec.medmnist_key]["label"]
    class_names = {int(k): str(v) for k, v in labels.items()}
    prompts = dataset_spec.build_class_prompts(class_names)
    default_prompt = f"{dataset_spec.prompt_prefix} medical class"
    return prompts, default_prompt


def default_data_npz_path(dataset_name: str) -> Path:
    return Path("data") / f"{dataset_name}_224.npz"


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


class MedMNISTTextDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        tokenizer: CLIPTokenizer,
        resolution: int,
        prompt_dropout_prob: float,
        class_prompts: dict[int, str],
        default_prompt: str,
    ) -> None:
        self.images = np.ascontiguousarray(images)
        self.labels = labels.astype(np.int64, copy=False)
        self.tokenizer = tokenizer
        self.prompts = dict(class_prompts)
        self.default_prompt = default_prompt
        self.prompt_dropout_prob = float(np.clip(prompt_dropout_prob, 0.0, 1.0))
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
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self.images[index]
        label = int(self.labels[index])
        prompt = self.prompts.get(label, f"{self.default_prompt} class {label}")
        if self.prompt_dropout_prob > 0.0 and random.random() < self.prompt_dropout_prob:
            prompt = ""

        pixel = self.transform(image)
        pixel = pixel * 2.0 - 1.0

        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": pixel,
            "input_ids": input_ids,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset_spec = get_dataset_spec(args.dataset)
    data_npz_path = Path(args.data_npz) if args.data_npz else default_data_npz_path(dataset_spec.name)
    output_dir = Path(args.output_dir) if args.output_dir else default_lora_output_dir(dataset_spec.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, labels = load_medmnist_train(data_npz_path)
    images, labels = maybe_subsample(images, labels, args.max_train_samples, args.seed)
    class_prompts, default_prompt = build_dataset_prompts(dataset_spec)

    print(f"[LoRA-Train] dataset={dataset_spec.name} data_npz={data_npz_path}")
    label_values, label_counts = np.unique(labels, return_counts=True)
    class_distribution = {int(k): int(v) for k, v in zip(label_values.tolist(), label_counts.tolist())}
    print(f"[LoRA-Train] class_distribution={class_distribution}")

    weight_dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model_id, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.base_model_id, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.base_model_id, subfolder="unet", torch_dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model_id, subfolder="scheduler")

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
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    if weight_dtype == torch.float16 and device.type == "cuda":
        cast_training_params(unet, dtype=torch.float32)
    unet.train()

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    trainable_param_count = int(sum(p.numel() for p in trainable_params))
    print(f"[LoRA-Train] trainable_params={trainable_param_count}")

    dataset = MedMNISTTextDataset(
        images=images,
        labels=labels,
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

    global_step = 0
    running_loss = 0.0
    running_steps = 0
    optimizer.zero_grad(set_to_none=True)

    stop_training = False
    for epoch in range(1, args.epochs + 1):
        for batch_idx, batch in enumerate(data_loader, start=1):
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * float(getattr(vae.config, "scaling_factor", 0.18215))
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
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

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
            unet_lora_state_dict = get_peft_model_state_dict(unet)
            StableDiffusionPipeline.save_lora_weights(
                save_directory=str(ckpt_dir),
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

        if stop_training:
            break

    unet_lora_state_dict = get_peft_model_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )

    summary = {
        "dataset": dataset_spec.name,
        "base_model_id": args.base_model_id,
        "data_npz": str(data_npz_path),
        "train_samples_used": int(len(dataset)),
        "resolution": int(args.resolution),
        "rank": int(args.rank),
        "lora_alpha": int(args.lora_alpha),
        "epochs": int(args.epochs),
        "max_train_steps": int(args.max_train_steps),
        "global_step": int(global_step),
        "trainable_params": trainable_param_count,
        "effective_batch_size": int(args.batch_size * grad_accum_steps),
        "gradient_accumulation_steps": int(grad_accum_steps),
        "class_balance": bool(args.class_balance),
        "prompt_dropout_prob": float(args.prompt_dropout_prob),
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
    parser = argparse.ArgumentParser(description="LoRA fine-tune Stable Diffusion on MedMNIST train split")
    parser.add_argument("--dataset", type=str, default="dermamnist", choices=supported_datasets())
    parser.add_argument("--data-npz", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--base-model-id", type=str, default="runwayml/stable-diffusion-v1-5")

    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=3000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--noise-offset", type=float, default=0.05)
    parser.add_argument("--input-perturbation", type=float, default=0.0)
    parser.add_argument("--prompt-dropout-prob", type=float, default=0.1)
    parser.add_argument("--class-balance", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-steps", type=int, default=20)

    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
