from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import timm
from sklearn.metrics import f1_score, roc_auc_score
from timm.data import resolve_model_data_config
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from .datasets import TorchDataset, get_dataset_spec, _normalize_dataset_key, supported_datasets
from .utils import *


SUPPORTED_BACKBONES = ("resnet18", "resnet50", "vit_tiny_patch16_224")
BACKBONE_ALIASES = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vit": "vit_tiny_patch16_224",
    "vit_tiny": "vit_tiny_patch16_224",
    "vit-tiny": "vit_tiny_patch16_224",
    "vit_tiny_patch16_224": "vit_tiny_patch16_224",
}


def normalize_backbone_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    return BACKBONE_ALIASES.get(key, key)


def resolve_backbone_normalization(backbone: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    backbone_name = normalize_backbone_name(backbone)
    if backbone_name != "vit_tiny_patch16_224":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # ViT follows timm model data config (vit_tiny_patch16_224 defaults to mean/std=0.5).
    vit_probe = timm.create_model(backbone_name, pretrained=False, num_classes=1)
    try:
        data_config = resolve_model_data_config(vit_probe)
    finally:
        del vit_probe

    mean = tuple(float(v) for v in data_config.get("mean", (0.5, 0.5, 0.5)))
    std = tuple(float(v) for v in data_config.get("std", (0.5, 0.5, 0.5)))
    return mean, std


def build_teacher_transforms(image_size: int, backbone: str) -> tuple[transforms.Compose, transforms.Compose]:
    mean, std = resolve_backbone_normalization(backbone)
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_classifier(num_classes: int, backbone: str, imagenet_pretrained: bool) -> nn.Module:
    backbone_name = normalize_backbone_name(backbone)
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        model = resnet18(weights=weights)
    elif backbone_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
        model = resnet50(weights=weights)
    elif backbone_name == "vit_tiny_patch16_224":
        # vit_tiny_patch16_224 is selected for 12GB GPUs as a practical ViT baseline.
        return timm.create_model(backbone_name, pretrained=imagenet_pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. choices={SUPPORTED_BACKBONES}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
        probs = F.softmax(logits.float(), dim=1)

        y_true_chunks.append(labels.cpu().numpy())
        y_prob_chunks.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_chunks, axis=0)
    y_prob = np.concatenate(y_prob_chunks, axis=0)
    return y_true, y_prob


def compute_macro_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float | None:
    class_counts = np.bincount(y_true, minlength=num_classes)
    if int((class_counts > 0).sum()) < 2:
        return None

    y_true_one_hot = np.eye(num_classes, dtype=np.float32)[y_true]
    try:
        return float(roc_auc_score(y_true_one_hot, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def evaluate_teacher_baseline_metrics(
    model: nn.Module,
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
    num_classes: int,
) -> dict[str, Any]:
    test_loss, test_acc = evaluate_classifier(model, test_loader, device, amp_enabled=amp_enabled)
    y_true, y_prob = predict_probabilities(model, test_loader, device, amp_enabled=amp_enabled)
    y_pred = np.argmax(y_prob, axis=1)
    test_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    test_auc = compute_macro_auc(y_true, y_prob, num_classes=num_classes)

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_f1": test_macro_f1,
        "test_auc": test_auc,
        "baseline_metrics": {
            "ACC": float(test_acc),
            "AUC": test_auc,
            "MacroF1": test_macro_f1,
        },
    }


def load_teacher_checkpoint(
    checkpoint_path: Path,
    num_classes: int,
    backbone: str,
    imagenet_pretrained: bool,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    loaded = torch.load(checkpoint_path, map_location=device)
    if not isinstance(loaded, dict) or "model" not in loaded:
        raise KeyError(f"{checkpoint_path} missing required key: model")

    teacher = build_classifier(
        num_classes=num_classes,
        backbone=backbone,
        imagenet_pretrained=imagenet_pretrained,
    ).to(device)
    teacher.load_state_dict(loaded["model"])
    teacher.eval()
    return teacher, loaded


def summarize_teacher_model(
    teacher: nn.Module,
    test_set: Any,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    baseline_dir: Path,
    checkpoint_path: Path,
    best_val_acc: float | None,
) -> dict[str, Any]:
    _, eval_transform = build_teacher_transforms(args.image_size, args.teacher_backbone)
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        TorchDataset(test_set, transform=eval_transform),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    baseline_eval = evaluate_teacher_baseline_metrics(
        teacher,
        test_loader=test_loader,
        device=device,
        amp_enabled=amp_enabled,
        num_classes=num_classes,
    )

    teacher_summary = {
        "best_val_acc": best_val_acc,
        "test_loss": baseline_eval["test_loss"],
        "test_acc": baseline_eval["test_acc"],
        "test_auc": baseline_eval["test_auc"],
        "test_macro_f1": baseline_eval["test_macro_f1"],
        "epochs": int(args.teacher_epochs),
        "backbone": str(args.teacher_backbone),
        "checkpoint_path": str(checkpoint_path),
        "baseline_metrics": baseline_eval["baseline_metrics"],
    }

    with open(baseline_dir / "teacher_summary.json", "w", encoding="utf-8") as handle:
        json.dump(teacher_summary, handle, indent=2)

    with open(baseline_dir / "teacher_baseline_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(baseline_eval["baseline_metrics"], handle, indent=2)

    test_auc = teacher_summary["test_auc"]
    test_auc_text = f"{test_auc:.4f}" if test_auc is not None else "nan"
    print(
        f"[Teacher] Final metrics: test_acc={teacher_summary['test_acc']:.4f} "
        f"test_auc={test_auc_text} test_macro_f1={teacher_summary['test_macro_f1']:.4f}"
    )

    return teacher_summary


def train_teacher_baseline(
    train_set: Any,
    val_set: Any,
    test_set: Any,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    baseline_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    if args.teacher_epochs <= 0:
        raise ValueError("teacher_epochs must be positive.")

    train_transform, eval_transform = build_teacher_transforms(args.image_size, args.teacher_backbone)

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        TorchDataset(train_set, transform=train_transform),
        batch_size=args.teacher_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        TorchDataset(val_set, transform=eval_transform),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        TorchDataset(test_set, transform=eval_transform),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    teacher = build_classifier(
        num_classes=num_classes,
        backbone=args.teacher_backbone,
        imagenet_pretrained=args.imagenet_pretrained,
    ).to(device)

    optimizer = AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=args.teacher_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.teacher_epochs, 1))
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    best_ckpt_path = baseline_dir / "teacher_best.pt"
    best_val_acc = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.teacher_epochs + 1):
        teacher.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Training epoch {epoch}/{args.teacher_epochs}",
            leave=False,
            unit="batch",
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = teacher(images)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * images.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += images.size(0)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)
        val_loss, val_acc = evaluate_classifier(teacher, val_loader, device, amp_enabled=amp_enabled)
        test_loss, test_acc = evaluate_classifier(teacher, test_loader, device, amp_enabled=amp_enabled)
        y_true, y_prob = predict_probabilities(teacher, test_loader, device, amp_enabled=amp_enabled)
        y_pred = np.argmax(y_prob, axis=1)
        test_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        test_auc = compute_macro_auc(y_true, y_prob, num_classes=num_classes)
        scheduler.step()

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_macro_f1": test_macro_f1,
                "test_auc": test_auc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": teacher.state_dict(),
                    "best_val_acc": best_val_acc,
                    "epoch": int(epoch),
                },
                best_ckpt_path,
            )

        test_auc_text = f"{test_auc:.4f}" if test_auc is not None else "nan"
        print(
            f"[Teacher] epoch={epoch:03d}/{args.teacher_epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
            f"test_auc={test_auc_text} test_macro_f1={test_macro_f1:.4f}"
        )

    with open(baseline_dir / "teacher_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    teacher, loaded = load_teacher_checkpoint(
        checkpoint_path=best_ckpt_path,
        num_classes=num_classes,
        backbone=args.teacher_backbone,
        imagenet_pretrained=args.imagenet_pretrained,
        device=device,
    )
    best_val_acc_raw = loaded.get("best_val_acc")
    best_val_acc_final = float(best_val_acc_raw) if best_val_acc_raw is not None else None

    teacher_summary = summarize_teacher_model(
        teacher=teacher,
        test_set=test_set,
        num_classes=num_classes,
        args=args,
        device=device,
        amp_enabled=amp_enabled,
        baseline_dir=baseline_dir,
        checkpoint_path=best_ckpt_path,
        best_val_acc=best_val_acc_final,
    )

    return teacher, teacher_summary


def run_teacher_baseline(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    dataset_spec = get_dataset_spec(args.dataset)
    output_dir = Path(args.output_dir or f"outputs/{dataset_spec.name}_224_distill_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["output_dir"] = str(output_dir)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    split_bundle = dataset_spec.load_dataset_splits(data_root=args.data_root, image_size=args.image_size)
    train_set = split_bundle.train_set
    val_set = split_bundle.val_set
    test_set = split_bundle.test_set
    num_classes = split_bundle.num_classes

    _, teacher_summary = train_teacher_baseline(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        num_classes=num_classes,
        args=args,
        device=device,
        amp_enabled=amp_enabled,
        baseline_dir=output_dir,
    )
    print(json.dumps(teacher_summary, indent=2))
    return teacher_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train teacher baseline model for distillation datasets "
            "(DermaMNIST, BloodMNIST, NIH Chest X-ray14, ODIR-5K)."
        )
    )
    parser.add_argument("--dataset", type=_normalize_dataset_key, default="dermamnist", choices=supported_datasets())
    parser.add_argument("--data-root", type=str, default=default_data_root())
    parser.add_argument("--output-dir", type=str, default="")

    parser.add_argument(
        "--teacher-backbone",
        type=normalize_backbone_name,
        default="resnet18",
        choices=list(SUPPORTED_BACKBONES),
    )
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--teacher-batch-size", type=int, default=128)
    parser.add_argument("--teacher-lr", type=float, default=3e-4)
    parser.add_argument("--teacher-weight-decay", type=float, default=1e-4)

    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_teacher_baseline(args)


if __name__ == "__main__":
    main()
