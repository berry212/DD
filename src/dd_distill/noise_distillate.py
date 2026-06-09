"""
Noise-only "distillation": generate random noise images as synthetic data,
then run teacher inference to produce soft labels.

This skips VAE encoding, clustering, and diffusion decoding entirely.
The output format matches that of the real distillation pipeline
so it can be consumed by run-train-distilled-student as-is.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .baseline_resnet18 import load_teacher_checkpoint
from .datasets import get_dataset_spec, supported_datasets
from .utils import (
    default_data_root,
    normalize_batch,
    save_distillation_artifacts,
    save_distilled_images,
    save_preview_grid,
    set_global_seed,
    resolve_device,
)
from .distillate import make_teacher_soft_labels
from .train_student import build_train_transform


def generate_noise_images(
    num_images: int,
    image_size: int,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 42,
) -> torch.Tensor:
    """Generate random Gaussian noise images in [0,1] range.

    Returns a tensor of shape (num_images, 3, image_size, image_size)
    with values clipped to [0, 1].
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    noise = torch.randn(num_images, 3, image_size, image_size, generator=gen)
    noise = noise * std + mean
    # Scale to [0, 1] for compatibility with normalization pipeline
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise.clamp(0.0, 1.0)


def run_noise_distillation(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.fp16 and device.type == "cuda")

    dataset_spec = get_dataset_spec(args.dataset)
    output_dir = Path(args.output_dir or f"outputs/{dataset_spec.name}_224_noise_ipc{args.clusters_per_class}")
    teacher_baseline_dir = Path(
        args.teacher_baseline_dir or f"outputs/{dataset_spec.name}_224_distill_baseline"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_baseline_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["output_dir"] = str(output_dir)
    run_config["teacher_baseline_dir"] = str(teacher_baseline_dir)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    split_bundle = dataset_spec.load_dataset_splits(
        data_root=args.data_root, image_size=args.image_size
    )
    val_set = split_bundle.val_set
    test_set = split_bundle.test_set
    num_classes = split_bundle.num_classes

    print(
        f"[Noise-Distill] dataset={dataset_spec.name} device={device} "
        f"classes={num_classes} baseline_dir={teacher_baseline_dir}"
    )

    # ── Load teacher ──
    teacher_ckpt_path = teacher_baseline_dir / "teacher_best.pt"
    if not teacher_ckpt_path.exists():
        raise FileNotFoundError(
            "Teacher checkpoint not found. Run `bash baseline.sh` first. "
            f"Missing path: {teacher_ckpt_path}"
        )

    teacher, _ = load_teacher_checkpoint(
        checkpoint_path=teacher_ckpt_path,
        num_classes=num_classes,
        backbone=args.teacher_backbone,
        imagenet_pretrained=args.imagenet_pretrained,
        device=device,
    )
    print(f"[Teacher] Loaded checkpoint: {teacher_ckpt_path}")

    # ── Compute total number of "distilled" samples ──
    ipc = max(1, int(args.clusters_per_class))
    num_distilled = ipc * num_classes

    # ── Assign labels (balanced per class) ──
    labels = torch.arange(num_classes, dtype=torch.long).repeat_interleave(ipc)
    counts = torch.ones(num_distilled, dtype=torch.long)  # placeholder, unused in uniform weighting
    weights = torch.ones(num_distilled, dtype=torch.float32)

    # ── Generate noise images ──
    print(f"[Noise-Distill] Generating {num_distilled} noise images (IPC={ipc}) ...")
    noise_images = generate_noise_images(
        num_images=num_distilled,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"[Noise-Distill] Noise shape: {tuple(noise_images.shape)}")

    # ── Teacher inference on noise → soft labels ──
    print("[Noise-Distill] Running teacher inference on noise images ...")
    soft_labels = make_teacher_soft_labels(
        teacher=teacher,
        distilled_images=noise_images,
        temperature=args.teacher_temperature,
        batch_size=args.eval_batch_size,
        device=device,
    )
    print(f"[Noise-Distill] Soft labels shape: {tuple(soft_labels.shape)}")

    # ── Save images as PNGs ──
    print("[Noise-Distill] Saving noise images to disk ...")
    saved_paths = save_distilled_images(
        images=noise_images,
        labels=labels,
        output_dir=output_dir,
        start_index=0,
        max_workers=args.save_async_workers,
    )

    # ── Preview ──
    preview_count = min(100, num_distilled)
    save_preview_grid(
        noise_images[:preview_count],
        output_dir / "distilled_preview.png",
    )

    # ── Optionally store images in pt ──
    store_images_in_pt = bool(args.store_images_in_pt or args.fkd_precompute_batches)
    distilled_images_for_pt = noise_images if store_images_in_pt else None

    # ── Optionally precompute FKD batches ──
    fkd_batch_path = ""
    fkd_batch_summary: dict[str, object] | None = None
    if bool(args.fkd_precompute_batches):
        print("[Noise-Distill] Precomputing FKD batches ...")
        # Reuse the existing FKD precomputation from distillate.py
        from .distillate import precompute_fkd_batch_cache

        fkd_batch_cache = precompute_fkd_batch_cache(
            teacher=teacher,
            distilled_images=noise_images,
            image_size=args.image_size,
            batch_size=args.fkd_batch_size,
            train_epochs=args.fkd_train_epochs,
            crop_min_scale=args.fkd_crop_min_scale,
            crop_max_scale=args.fkd_crop_max_scale,
            horizontal_flip_prob=args.fkd_horizontal_flip_prob,
            soft_label_temperature=args.teacher_temperature,
            eval_batch_size=args.eval_batch_size,
            device=device,
            seed=args.seed,
        )
        fkd_batch_path = "fkd_batches.pt"
        torch.save(fkd_batch_cache, output_dir / fkd_batch_path)
        fkd_batch_summary = {
            "batch_size": int(fkd_batch_cache["batch_size"]),
            "batches_per_epoch": int(fkd_batch_cache["batches_per_epoch"]),
            "train_epochs": int(fkd_batch_cache["train_epochs"]),
            "num_batches": int(fkd_batch_cache["num_batches"]),
            "image_size": int(fkd_batch_cache["image_size"]),
            "crop_min_scale": float(fkd_batch_cache["crop_min_scale"]),
            "crop_max_scale": float(fkd_batch_cache["crop_max_scale"]),
            "horizontal_flip_prob": float(fkd_batch_cache["horizontal_flip_prob"]),
            "seed": int(fkd_batch_cache["seed"]),
        }

    # ── Save artifacts in standard format ──
    save_distillation_artifacts(
        output_dir=output_dir,
        images=distilled_images_for_pt,
        weights=weights,
        soft_labels=soft_labels,
        center_labels=labels,
        counts=counts,
        saved_paths=saved_paths,
        dataset_name=dataset_spec.name,
        lora_path="",
        teacher_temperature=args.teacher_temperature,
        image_shards=[],
        store_images_in_pt=store_images_in_pt,
        fkd_batch_path=fkd_batch_path,
        fkd_batch_summary=fkd_batch_summary,
        distill_method="noise",
    )

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "dataset": dataset_spec.name,
        "method": "noise_baseline",
        "num_classes": int(num_classes),
        "clusters_per_class": int(ipc),
        "num_distilled": int(num_distilled),
        "teacher_backbone": str(args.teacher_backbone),
        "teacher_temperature": float(args.teacher_temperature),
        "teacher_checkpoint": str(teacher_ckpt_path),
        "noise_mean": 0.0,
        "noise_std": 1.0,
        "image_size": int(args.image_size),
        "fkd_precompute_batches": bool(args.fkd_precompute_batches),
        "fkd_batch_path": str(fkd_batch_path),
        "fkd_batch_summary": dict(fkd_batch_summary or {}),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Done] Noise distillation complete.")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Noise-only distillation baseline: generate random Gaussian noise images, "
            "run teacher inference for soft labels, and save in standard distilled format."
        )
    )
    parser.add_argument("--dataset", default="dermamnist", choices=supported_datasets())
    parser.add_argument("--data-root", type=str, default=default_data_root())
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--teacher-baseline-dir", type=str, default="")

    parser.add_argument("--clusters-per-class", type=float, default=100.0,
                        help="Number of noise images per class (analogous to IPC).")
    parser.add_argument("--teacher-backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--teacher-temperature", type=float, default=20.0)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--store-images-in-pt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fkd-precompute-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fkd-train-epochs", type=int, default=300)
    parser.add_argument("--fkd-batch-size", type=int, default=1024)
    parser.add_argument("--fkd-crop-min-scale", type=float, default=0.08)
    parser.add_argument("--fkd-crop-max-scale", type=float, default=1.0)
    parser.add_argument("--fkd-horizontal-flip-prob", type=float, default=0.5)

    parser.add_argument("--save-async-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_noise_distillation(args)


if __name__ == "__main__":
    main()
