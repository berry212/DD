from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from .baseline_resnet18 import (
    DermamnistNpzDataset,
    build_model,
    build_transforms,
    class_names_for_dermamnist,
    load_dermamnist_npz,
    maybe_subsample,
    resolve_device,
    set_seed,
)


def build_eval_loader(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, int]:
    arrays = load_dermamnist_npz(Path(args.data_npz))
    split = args.split.lower()
    image_key = f"{split}_images"
    label_key = f"{split}_labels"

    if image_key not in arrays or label_key not in arrays:
        raise KeyError(f"Unsupported split={split}, expected one of train/val/test")

    images, labels = maybe_subsample(arrays[image_key], arrays[label_key], args.max_samples, args.seed)
    labels = labels.reshape(-1)

    num_classes = int(
        max(
            arrays["train_labels"].reshape(-1).max(),
            arrays["val_labels"].reshape(-1).max(),
            arrays["test_labels"].reshape(-1).max(),
        )
        + 1
    )

    _, eval_transform = build_transforms(args.image_size)
    dataset = DermamnistNpzDataset(images, labels, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return loader, num_classes


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict: dict[str, Any]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[QuickEval] missing_keys={len(missing)}")
    if unexpected:
        print(f"[QuickEval] unexpected_keys={len(unexpected)}")


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader, num_classes = build_eval_loader(args, device)

    model = build_model(num_classes=num_classes, pretrained=args.imagenet_pretrained)
    model.to(device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint.strip() else None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        load_checkpoint_weights(model, checkpoint_path, device)
        print(f"[QuickEval] loaded checkpoint: {checkpoint_path}")
    elif not args.imagenet_pretrained:
        raise ValueError("Either provide --checkpoint or enable --imagenet-pretrained")
    else:
        print("[QuickEval] using torchvision ImageNet pretrained ResNet18 weights (no fine-tuned checkpoint)")
        print(
            "[QuickEval] note: the 7-class classification head is randomly initialized "
            "without a task-specific checkpoint."
        )

    amp_enabled = bool(args.amp and device.type == "cuda")
    model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        probs = torch.softmax(logits, dim=1)
        confidence, predictions = probs.max(dim=1)

        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)
        y_true_all.append(labels.cpu().numpy())
        y_pred_all.append(predictions.cpu().numpy())
        conf_all.append(confidence.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    confidence = np.concatenate(conf_all, axis=0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    class_names = class_names_for_dermamnist(num_classes)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    predictions_df = pd.DataFrame(
        {
            "sample_index": np.arange(y_true.shape[0], dtype=np.int64),
            "y_true": y_true.astype(np.int64, copy=False),
            "y_pred": y_pred.astype(np.int64, copy=False),
            "confidence": confidence.astype(np.float32, copy=False),
            "correct": (y_true == y_pred).astype(np.int64, copy=False),
        }
    )
    predictions_df.to_csv(output_dir / "quick_eval_predictions.csv", index=False)

    with open(output_dir / "quick_eval_report.txt", "w", encoding="utf-8") as handle:
        handle.write(report_text)

    summary = {
        "data_npz": args.data_npz,
        "split": args.split,
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(num_classes),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "imagenet_pretrained": bool(args.imagenet_pretrained),
        "device": device.type,
        "loss": float(avg_loss),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "output_dir": str(output_dir),
    }

    with open(output_dir / "quick_eval_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[QuickEval] classification report")
    print(report_text)
    print("[QuickEval] summary")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick ResNet18 validation (inference only, no training)")
    parser.add_argument("--data-npz", type=str, default="data/dermamnist_224.npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional fine-tuned checkpoint path. Leave empty to use only torchvision pretrained weights.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/resnet18_quick_eval")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()