from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, override
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import timm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from timm.data import resolve_model_data_config
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from .datasets import DistilledTripletDataset, TorchDataset, get_dataset_spec, supported_datasets
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

    # ViT models in timm can define their own data config. Use it for normalization.
    vit_probe = timm.create_model(backbone_name, pretrained=False, num_classes=1)
    try:
        data_config = resolve_model_data_config(vit_probe)
    finally:
        del vit_probe

    mean = tuple(float(v) for v in data_config.get("mean", (0.5, 0.5, 0.5)))
    std = tuple(float(v) for v in data_config.get("std", (0.5, 0.5, 0.5)))
    return mean, std


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_eval_transform(image_size: int, backbone: str) -> transforms.Compose:
    mean, std = resolve_backbone_normalization(backbone)
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_loaders(
    val_set: Any,
    test_set: Any,
    eval_transform: transforms.Compose,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    pin_memory = device.type == "cuda"
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        TorchDataset(val_set, transform=eval_transform),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        TorchDataset(test_set, transform=eval_transform),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return val_loader, test_loader


def load_distilled_triplet(
    distilled_data_path: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
    payload = torch.load(distilled_data_path, map_location="cpu")

    required = {"weights", "soft_labels"}
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Missing keys in {distilled_data_path}: {missing}")

    images_obj = payload.get("images")
    images: torch.Tensor
    if isinstance(images_obj, torch.Tensor):
        images = images_obj.float()
    else:
        image_shards = payload.get("image_shards")
        if isinstance(image_shards, list) and image_shards:
            image_chunks: list[torch.Tensor] = []
            for rel_shard in image_shards:
                shard_path = distilled_data_path.parent / str(rel_shard)
                shard_payload = torch.load(shard_path, map_location="cpu")
                shard_images = shard_payload.get("images")
                if not isinstance(shard_images, torch.Tensor):
                    raise ValueError(f"Missing tensor 'images' in shard: {shard_path}")
                shard_t = shard_images.float()
                if shard_t.max().item() > 1.0:
                    shard_t = shard_t / 255.0
                image_chunks.append(shard_t)
            images = torch.cat(image_chunks, dim=0)
        else:
            rel_paths = payload.get("image_relative_paths")
            if not isinstance(rel_paths, list) or not rel_paths:
                raise KeyError(
                    "Missing images in distilled payload. Expected one of: 'images', 'image_shards', 'image_relative_paths'."
                )

            image_chunks = []
            for rel_path in rel_paths:
                rel_str = str(rel_path)
                cand_a = distilled_data_path.parent / "distilled_images" / rel_str
                cand_b = distilled_data_path.parent / rel_str
                abs_path = cand_a if cand_a.exists() else cand_b
                if not abs_path.exists():
                    raise FileNotFoundError(f"Distilled image file not found: {cand_a} or {cand_b}")

                with Image.open(abs_path) as pil_img:
                    img_np = np.asarray(pil_img, dtype=np.float32)
                if img_np.ndim == 2:
                    img_np = img_np[:, :, None]
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous() / 255.0
                image_chunks.append(img_t)

            images = torch.stack(image_chunks, dim=0)

    weights = payload["weights"].float().view(-1)
    soft_labels = payload["soft_labels"].float()
    distilled_dataset = str(payload.get("dataset", "")).strip().lower()
    distilled_lora_path = str(payload.get("lora_path", "")).strip()

    if images.ndim != 4:
        raise ValueError(f"images must be 4D (N,C,H,W), got {tuple(images.shape)}")
    if soft_labels.ndim != 2:
        raise ValueError(f"soft_labels must be 2D (N,C), got {tuple(soft_labels.shape)}")

    n = images.size(0)
    if weights.numel() != n or soft_labels.size(0) != n:
        raise ValueError(
            "Mismatch among images/weights/soft_labels sizes: "
            f"N={n}, weights={weights.numel()}, soft_labels={soft_labels.size(0)}"
        )

    if not bool(torch.isfinite(images).all()):
        raise ValueError(f"images contain NaN/Inf: {distilled_data_path}")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError(f"weights contain NaN/Inf: {distilled_data_path}")
    if not bool(torch.isfinite(soft_labels).all()):
        raise ValueError(f"soft_labels contain NaN/Inf: {distilled_data_path}")

    min_val = float(images.min().item())
    max_val = float(images.max().item())
    if min_val >= 0.0 and max_val <= 1.0:
        pass
    elif min_val >= 0.0 and max_val <= 255.0 + 1e-6:
        images = images / 255.0
    elif min_val >= -1.0 - 1e-6 and max_val <= 1.0 + 1e-6:
        images = (images + 1.0) / 2.0
    else:
        images = images.clamp(0.0, 1.0)
    images = images.clamp(0.0, 1.0)

    weights = weights.clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-12)

    soft_labels = soft_labels.clamp_min(0.0)
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return images, weights, soft_labels, distilled_dataset, distilled_lora_path


def sharpen_soft_labels(soft_labels: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(max(temperature, 1e-6))
    if abs(t - 1.0) < 1e-8:
        return soft_labels

    logits = torch.log(soft_labels.clamp_min(1e-12))
    sharpened = F.softmax(logits / t, dim=1)
    return sharpened


def blend_sample_weights(
    weights: torch.Tensor,
    hard_labels: torch.Tensor,
    num_classes: int,
    balance_alpha: float,
) -> torch.Tensor:
    alpha = float(np.clip(balance_alpha, 0.0, 1.0))
    base = weights.clamp_min(0.0)
    base = base / base.sum().clamp_min(1e-12)

    if alpha <= 0.0:
        return base

    class_counts = torch.bincount(hard_labels.long(), minlength=num_classes).float()
    class_counts = class_counts.clamp_min(0.0)
    present = class_counts > 0
    inv = torch.zeros_like(class_counts)
    inv[present] = 1.0 / class_counts[present]

    balanced = inv[hard_labels.long()]
    balanced = balanced / balanced.sum().clamp_min(1e-12)

    mixed = (1.0 - alpha) * base + alpha * balanced
    mixed = mixed / mixed.sum().clamp_min(1e-12)
    return mixed


def build_classifier(
    num_classes: int,
    imagenet_pretrained: bool,
    backbone: str,
) -> nn.Module:
    name = normalize_backbone_name(backbone)
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        model = resnet18(weights=weights)
    elif name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
        model = resnet50(weights=weights)
    elif name == "vit_tiny_patch16_224":
        # vit_tiny_patch16_224 is selected for 12GB GPUs as a practical ViT baseline.
        model = timm.create_model(name, pretrained=imagenet_pretrained, num_classes=num_classes)
        return model
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

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[torch.Tensor] = []
    all_prob: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
        probs = F.softmax(logits.float(), dim=1)

        all_true.append(labels.cpu())
        all_prob.append(probs.cpu())

    y_true = torch.cat(all_true, dim=0).numpy()
    y_prob = torch.cat(all_prob, dim=0).numpy()
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


def train_student(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    amp_enabled: bool,
    output_dir: Path,
    kd_temperature: float,
    hard_label_alpha: float,
) -> dict[str, Any]:
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    best_val_acc = -1.0
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_soft_loss = 0.0
        epoch_hard_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        temp = float(max(kd_temperature, 1e-6))
        hard_alpha = float(np.clip(hard_label_alpha, 0.0, 1.0))

        for images, soft_labels, hard_labels, sample_weights in train_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device=device, dtype=torch.float32)
            hard_labels = hard_labels.to(device)
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                student_log_probs = F.log_softmax(logits.float() / temp, dim=1)
                per_sample_soft_ce = -(soft_labels * student_log_probs).sum(dim=1)
                soft_loss = (per_sample_soft_ce * sample_weights).sum() / (sample_weights.sum() + 1e-12)
                soft_loss = soft_loss * (temp * temp)

                per_sample_hard_ce = F.cross_entropy(logits.float(), hard_labels, reduction="none")
                hard_loss = (per_sample_hard_ce * sample_weights).sum() / (sample_weights.sum() + 1e-12)

                loss = (1.0 - hard_alpha) * soft_loss + hard_alpha * hard_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item()) * images.size(0)
            epoch_soft_loss += float(soft_loss.item()) * images.size(0)
            epoch_hard_loss += float(hard_loss.item()) * images.size(0)
            epoch_correct += int((logits.argmax(dim=1) == hard_labels).sum().item())
            epoch_samples += images.size(0)

        train_loss = epoch_loss / max(epoch_samples, 1)
        train_soft_loss = epoch_soft_loss / max(epoch_samples, 1)
        train_hard_loss = epoch_hard_loss / max(epoch_samples, 1)
        train_acc = epoch_correct / max(epoch_samples, 1)
        val_loss, val_acc = evaluate_classifier(model, val_loader, device, amp_enabled=amp_enabled)
        test_loss, test_acc = evaluate_classifier(model, test_loader, device, amp_enabled=amp_enabled)
        scheduler.step()

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_soft_loss": train_soft_loss,
                "train_hard_loss": train_hard_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                },
                output_dir / "student_best.pt",
            )

        print(
            f"[Student] epoch={epoch:03d}/{epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
        )

    torch.save({"model": model.state_dict(), "history": history}, output_dir / "student_last.pt")
    with open(output_dir / "student_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    best_test_acc = 0.0
    best_test_loss = 0.0
    for row in history:
        if int(row["epoch"]) == best_epoch:
            best_test_acc = float(row["test_acc"])
            best_test_loss = float(row["test_loss"])
            break

    return {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "test_acc_at_best_val": float(best_test_acc),
        "test_loss_at_best_val": float(best_test_loss),
        "final_test_acc": float(history[-1]["test_acc"]),
        "final_test_loss": float(history[-1]["test_loss"]),
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    dataset_spec = get_dataset_spec(args.dataset)
    output_dir = Path(args.output_dir or f"outputs/{dataset_spec.name}_224_student")
    distilled_data_path = Path(args.distilled_data or f"outputs/{dataset_spec.name}_224_distill/distilled_data.pt")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["output_dir"] = str(output_dir)
    run_config["distilled_data"] = str(distilled_data_path)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    split_bundle = dataset_spec.load_dataset_splits(data_root=args.data_root, image_size=args.image_size)
    val_set = split_bundle.val_set
    test_set = split_bundle.test_set
    num_classes = split_bundle.num_classes
    class_name_map = split_bundle.class_names
    class_names = [class_name_map.get(idx, f"class_{idx}") for idx in range(num_classes)]
    images, weights, soft_labels, distilled_dataset, distilled_lora_path = load_distilled_triplet(
        distilled_data_path,
    )

    soft_labels = sharpen_soft_labels(soft_labels, temperature=args.soft_label_sharpen)
    hard_labels = torch.argmax(soft_labels, dim=1).long()
    weights = blend_sample_weights(
        weights=weights,
        hard_labels=hard_labels,
        num_classes=num_classes,
        balance_alpha=args.weight_balance_alpha,
    )

    if distilled_dataset and distilled_dataset != dataset_spec.name:
        raise ValueError(
            "Distilled data dataset mismatch: "
            f"expected={dataset_spec.name} actual={distilled_dataset} path={distilled_data_path}"
        )
    print(
        f"[Setup] dataset={dataset_spec.name} device={device} N={images.size(0)} "
        f"weights_sum={weights.sum().item():.6f} soft_shape={tuple(soft_labels.shape)}"
    )

    eval_transform = build_eval_transform(args.image_size, backbone=args.student_backbone)
    val_loader, test_loader = build_eval_loaders(
        val_set=val_set,
        test_set=test_set,
        eval_transform=eval_transform,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    train_set = DistilledTripletDataset(
        images=images,
        weights=weights,
        soft_labels=soft_labels,
        transform=eval_transform,
    )
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = DataLoader(
        train_set,
        batch_size=min(args.train_batch_size, len(train_set)),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_classifier(
        num_classes=num_classes,
        imagenet_pretrained=args.imagenet_pretrained,
        backbone=args.student_backbone,
    ).to(device)

    training_summary = train_student(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.train_epochs,
        learning_rate=args.train_lr,
        weight_decay=args.weight_decay,
        device=device,
        amp_enabled=amp_enabled,
        output_dir=output_dir,
        kd_temperature=args.kd_temperature,
        hard_label_alpha=args.hard_label_alpha,
    )

    best_ckpt = torch.load(output_dir / "student_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    model.eval()

    y_true, y_prob = predict_probabilities(model, test_loader, device, amp_enabled=amp_enabled)
    y_pred = np.argmax(y_prob, axis=1)
    test_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    test_auc_macro = compute_macro_auc(y_true, y_prob, num_classes=num_classes)

    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    conf = confusion_matrix(y_true, y_pred).tolist()

    report_payload = {
        "classification_report": report_dict,
        "confusion_matrix": conf,
        "test_macro_f1": test_macro_f1,
        "test_auc_macro": test_auc_macro,
        "text": report_text,
    }
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    print("[Classification Report]")
    print(report_text)
    auc_text = f"{test_auc_macro:.4f}" if test_auc_macro is not None else "nan"
    print(f"[Student] test_auc_macro={auc_text} test_macro_f1={test_macro_f1:.4f}")

    summary = {
        "dataset": dataset_spec.name,
        "data_root": args.data_root,
        "distilled_data": str(distilled_data_path),
        "num_classes": int(num_classes),
        "num_distilled": int(images.size(0)),
        "distilled_dataset": distilled_dataset or dataset_spec.name,
        "distilled_lora_path": distilled_lora_path,
        "student_backbone": str(args.student_backbone),
        "amp_enabled": bool(amp_enabled),
        "kd_temperature": float(args.kd_temperature),
        "hard_label_alpha": float(args.hard_label_alpha),
        "weight_balance_alpha": float(args.weight_balance_alpha),
        "soft_label_sharpen": float(args.soft_label_sharpen),
        "best_epoch": int(training_summary["best_epoch"]),
        "best_val_acc": float(training_summary["best_val_acc"]),
        "test_acc_at_best_val": float(training_summary["test_acc_at_best_val"]),
        "test_loss_at_best_val": float(training_summary["test_loss_at_best_val"]),
        "final_test_acc": float(training_summary["final_test_acc"]),
        "final_test_loss": float(training_summary["final_test_loss"]),
        "auc_macro": test_auc_macro,
        "macro_f1": test_macro_f1,
        "weighted_f1": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Done] Student training complete.")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train student from distilled triplet data: {images, weights, soft_labels}")
    parser.add_argument("--dataset", default="dermamnist", choices=supported_datasets())
    parser.add_argument("--data-root", type=str, default=default_data_root())
    parser.add_argument("--distilled-data", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")

    parser.add_argument(
        "--student-backbone",
        type=normalize_backbone_name,
        default="resnet50",
        choices=list(SUPPORTED_BACKBONES),
    )
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--train-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--hard-label-alpha", type=float, default=0.0)
    parser.add_argument("--weight-balance-alpha", type=float, default=0.0)
    parser.add_argument("--soft-label-sharpen", type=float, default=1.0)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
