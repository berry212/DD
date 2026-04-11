from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
import torch.nn.functional as F
from medmnist import INFO, DermaMNIST
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class DermaMNISTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: DermaMNIST, transform: transforms.Compose) -> None:
        super().__init__()
        self.images = np.ascontiguousarray(data.imgs)
        self.labels = data.labels.reshape(-1).astype(np.int64, copy=False)
        self.transform = transform

    @override
    def __len__(self) -> int:
        return int(self.images.shape[0])

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


class DistilledTripletDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        images: torch.Tensor,
        weights: torch.Tensor,
        soft_labels: torch.Tensor,
        transform: transforms.Compose,
    ) -> None:
        super().__init__()
        self.images = images.float().cpu()
        self.weights = weights.float().cpu()
        self.soft_labels = soft_labels.float().cpu()
        self.hard_labels = torch.argmax(self.soft_labels, dim=1).long()
        self.transform = transform

    @override
    def __len__(self) -> int:
        return int(self.images.size(0))

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        soft = self.soft_labels[index]
        hard = self.hard_labels[index]
        weight = self.weights[index]
        return image, soft, hard, weight


def set_global_seed(seed: int) -> None:
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


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_dermamnist_eval_splits(data_root: str) -> tuple[DermaMNIST, DermaMNIST, int, list[str]]:
    val_set = DermaMNIST(split="val", download=True, root=data_root, size=224)
    test_set = DermaMNIST(split="test", download=True, root=data_root, size=224)
    num_classes = len(INFO["dermamnist"]["label"])
    class_names = [INFO["dermamnist"]["label"][str(i)] for i in range(num_classes)]
    return val_set, test_set, num_classes, class_names


def build_eval_loaders(
    val_set: DermaMNIST,
    test_set: DermaMNIST,
    eval_transform: transforms.Compose,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    pin_memory = device.type == "cuda"
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        DermaMNISTDataset(val_set, transform=eval_transform),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        DermaMNISTDataset(test_set, transform=eval_transform),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return val_loader, test_loader


def load_distilled_triplet(
    distilled_data_path: Path,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(distilled_data_path, map_location="cpu")

    required = {"images", "weights", "soft_labels"}
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Missing keys in {distilled_data_path}: {missing}")

    images = payload["images"].float()
    weights = payload["weights"].float().view(-1)
    soft_labels = payload["soft_labels"].float()

    if images.ndim != 4:
        raise ValueError(f"images must be 4D (N,C,H,W), got {tuple(images.shape)}")
    if soft_labels.ndim != 2:
        raise ValueError(f"soft_labels must be 2D (N,C), got {tuple(soft_labels.shape)}")
    if soft_labels.size(1) != num_classes:
        raise ValueError(
            f"soft_labels second dimension must be num_classes={num_classes}, got {soft_labels.size(1)}"
        )

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
    return images, weights, soft_labels


def build_classifier(
    num_classes: int,
    imagenet_pretrained: bool,
    backbone: str,
) -> nn.Module:
    name = backbone.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        model = resnet18(weights=weights)
    elif name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}.")

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
def predict_labels(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[torch.Tensor] = []
    all_pred: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu()

        all_true.append(labels.cpu())
        all_pred.append(preds)

    y_true = torch.cat(all_true, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()
    return y_true, y_pred


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
        epoch_correct = 0
        epoch_samples = 0

        for images, soft_labels, hard_labels, sample_weights in train_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device=device, dtype=torch.float32)
            hard_labels = hard_labels.to(device)
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                per_sample_soft_ce = -(soft_labels * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
                loss = (per_sample_soft_ce * sample_weights).sum() / (sample_weights.sum() + 1e-12)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item()) * images.size(0)
            epoch_correct += int((logits.argmax(dim=1) == hard_labels).sum().item())
            epoch_samples += images.size(0)

        train_loss = epoch_loss / max(epoch_samples, 1)
        train_acc = epoch_correct / max(epoch_samples, 1)
        val_loss, val_acc = evaluate_classifier(model, val_loader, device, amp_enabled=amp_enabled)
        test_loss, test_acc = evaluate_classifier(model, test_loader, device, amp_enabled=amp_enabled)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    val_set, test_set, num_classes, class_names = load_dermamnist_eval_splits(args.data_root)
    images, weights, soft_labels = load_distilled_triplet(Path(args.distilled_data), num_classes)
    print(
        f"[Setup] device={device} N={images.size(0)} "
        f"weights_sum={weights.sum().item():.6f} soft_shape={tuple(soft_labels.shape)}"
    )

    eval_transform = build_eval_transform(args.image_size)
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
    )

    best_ckpt = torch.load(output_dir / "student_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    model.eval()

    y_true, y_pred = predict_labels(model, test_loader, device, amp_enabled=amp_enabled)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    conf = confusion_matrix(y_true, y_pred).tolist()

    report_payload = {
        "classification_report": report_dict,
        "confusion_matrix": conf,
        "text": report_text,
    }
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    print("[Classification Report]")
    print(report_text)

    summary = {
        "data_root": args.data_root,
        "distilled_data": args.distilled_data,
        "num_classes": int(num_classes),
        "num_distilled": int(images.size(0)),
        "student_backbone": str(args.student_backbone),
        "amp_enabled": bool(amp_enabled),
        "best_epoch": int(training_summary["best_epoch"]),
        "best_val_acc": float(training_summary["best_val_acc"]),
        "test_acc_at_best_val": float(training_summary["test_acc_at_best_val"]),
        "test_loss_at_best_val": float(training_summary["test_loss_at_best_val"]),
        "final_test_acc": float(training_summary["final_test_acc"]),
        "final_test_loss": float(training_summary["final_test_loss"]),
        "macro_f1": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Done] Student training complete.")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train student from distilled triplet data: {images, weights, soft_labels}")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--distilled-data", type=str, default="outputs/dermamnist_224_distill/distilled_data.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/dermamnist_224_student")

    parser.add_argument("--student-backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--train-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

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
