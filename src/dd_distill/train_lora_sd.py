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
from torch import nn
from diffusers import AutoencoderKL, DDPMScheduler, DiTTransformer2DModel
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

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


def load_dit_transformer(model_id: str, dtype: torch.dtype) -> DiTTransformer2DModel:
    try:
        return DiTTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
        )
    except Exception:
        return DiTTransformer2DModel.from_pretrained(model_id, torch_dtype=dtype)


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

    weight_dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    if args.base_model_id and not args.dit_model_id:
        warnings.warn("--base-model-id is deprecated; use --dit-model-id.", RuntimeWarning)

    dit_model_id = args.dit_model_id or args.base_model_id
    vae_model_id = args.vae_model_id or dit_model_id
    scheduler_model_id = args.scheduler_model_id or dit_model_id

    transformer = load_dit_transformer(dit_model_id, dtype=weight_dtype)
    vae = load_vae(vae_model_id, dtype=weight_dtype)
    noise_scheduler = load_scheduler(scheduler_model_id)

    transformer.to(device)
    vae.to(device)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    target_modules = [v.strip() for v in args.lora_target_modules.split(",") if v.strip()]
    if not target_modules:
        raise ValueError("--lora-target-modules must provide at least one module name")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    if hasattr(transformer, "add_adapter"):
        transformer.add_adapter(lora_config)
    else:
        transformer = get_peft_model(transformer, lora_config)

    if weight_dtype == torch.float16 and device.type == "cuda":
        cast_training_params(transformer, dtype=torch.float32)
    transformer.train()

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    trainable_param_count = int(sum(p.numel() for p in trainable_params))
    print(f"[LoRA-Train] trainable_params={trainable_param_count}")

    num_classes = getattr(transformer.config, "num_classes", None)
    if num_classes is not None and int(num_classes) > 0:
        max_label = int(labels.max()) + int(args.class_label_offset) if labels.size > 0 else 0
        if max_label >= int(num_classes):
            raise ValueError(
                f"Dataset labels exceed DiT num_classes (max_label={max_label}, num_classes={num_classes}). "
                "Use --class-label-offset or a compatible DiT checkpoint."
            )

    dataset = MedMNISTClassDataset(
        split_data=train_split,
        sample_indices=sample_indices,
        resolution=args.resolution,
        class_label_offset=args.class_label_offset,
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
            class_labels = batch["class_labels"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * float(getattr(vae.config, "scaling_factor", 0.18215))

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
                model_pred = transformer(noisy_latents, timesteps, class_labels=class_labels).sample
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
            save_lora_artifacts(ckpt_dir, transformer, lora_config)

        if stop_training:
            break

    save_lora_artifacts(output_dir, transformer, lora_config)

    summary = {
        "dataset": dataset_spec.name,
        "dit_model_id": str(dit_model_id),
        "vae_model_id": str(vae_model_id),
        "scheduler_model_id": str(scheduler_model_id),
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
    parser.add_argument("--dit-model-id", type=str, default="facebook/DiT-XL-2-256")
    parser.add_argument("--vae-model-id", type=str, default="")
    parser.add_argument("--scheduler-model-id", type=str, default="")
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="",
        help="Deprecated alias for --dit-model-id.",
    )

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
