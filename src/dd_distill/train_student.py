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
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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


class DermaMNISTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data: DermaMNIST,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.images = np.ascontiguousarray(data.imgs)
        self.labels = data.labels.reshape(-1).astype(np.int64, copy=False)
        self.transform = transform

    @override
    def __len__(self) -> int:
        return int(self.images.shape[0])

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_np = self.images[index]
        if self.transform is not None:
            image = self.transform(image_np)
        else:
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image = normalize_image(image)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


class DistilledSoftDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        soft_labels: torch.Tensor,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.images = images.float().cpu()
        self.labels = labels.long().cpu()
        self.weights = weights.float().cpu()
        self.soft_labels = soft_labels.float().cpu()
        self.transform = transform

    @override
    def __len__(self) -> int:
        return int(self.images.size(0))

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = normalize_image(image)
        soft = self.soft_labels[index]
        label = self.labels[index]
        weight = self.weights[index]
        return image, soft, label, weight


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


def load_dermamnist_splits(data_root: str) -> tuple[DermaMNIST, DermaMNIST, DermaMNIST, int]:
    train_set = DermaMNIST(split="train", download=True, root=data_root, size=224)
    val_set = DermaMNIST(split="val", download=True, root=data_root, size=224)
    test_set = DermaMNIST(split="test", download=True, root=data_root, size=224)
    num_classes = len(INFO["dermamnist"]["label"])
    return train_set, val_set, test_set, num_classes


def build_eval_loaders(
    val_set: DermaMNIST,
    test_set: DermaMNIST,
    eval_transform: transforms.Compose,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
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


def load_distilled_data(distilled_data_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(distilled_data_path, map_location="cpu")

    required = {"images", "labels", "weights"}
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Missing keys in {distilled_data_path}: {missing}")

    images = payload["images"].float()
    labels = payload["labels"].long()
    weights = payload["weights"].float()
    return images, labels, weights


def build_classifier(
    num_classes: int,
    imagenet_pretrained: bool,
    backbone: str,
) -> nn.Module:
    backbone_name = backbone.lower()
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        model = resnet18(weights=weights)
    elif backbone_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose from ['resnet18', 'resnet50'].")

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
def evaluate_classifier_detailed(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_classes: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_totals = torch.zeros(num_classes, dtype=torch.long)
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((preds == labels).sum().item())
        total_samples += images.size(0)

        labels_cpu = labels.detach().cpu()
        preds_cpu = preds.detach().cpu()
        class_totals += torch.bincount(labels_cpu, minlength=num_classes)

        correct_mask = preds_cpu == labels_cpu
        if bool(correct_mask.any()):
            class_correct += torch.bincount(labels_cpu[correct_mask], minlength=num_classes)

        confusion.index_put_(
            (labels_cpu, preds_cpu),
            torch.ones(labels_cpu.size(0), dtype=torch.long),
            accumulate=True,
        )

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    per_class_accuracy: dict[str, float] = {}
    for class_idx in range(num_classes):
        total = int(class_totals[class_idx].item())
        correct = int(class_correct[class_idx].item())
        per_class_accuracy[str(class_idx)] = 0.0 if total == 0 else correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def train_teacher(
    train_set: DermaMNIST,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    train_transform: transforms.Compose,
    num_classes: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_enabled: bool,
    imagenet_pretrained: bool,
    backbone: str,
    output_dir: Path,
) -> nn.Module:
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        DermaMNISTDataset(train_set, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    teacher = build_classifier(
        num_classes=num_classes,
        imagenet_pretrained=imagenet_pretrained,
        backbone=backbone,
    ).to(device)
    optimizer = AdamW(teacher.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    best_val_acc = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        teacher.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
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
            torch.save({"model": teacher.state_dict(), "best_val_acc": best_val_acc}, output_dir / "teacher_best.pt")

        print(
            f"[Teacher] epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
        )

    with open(output_dir / "teacher_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    best_ckpt = torch.load(output_dir / "teacher_best.pt", map_location=device)
    teacher.load_state_dict(best_ckpt["model"])
    teacher.eval()
    return teacher


@torch.no_grad()
def make_teacher_soft_labels(
    teacher: nn.Module,
    distilled_images: torch.Tensor,
    temperature: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    soft_targets: list[torch.Tensor] = []
    teacher.eval()

    for start in range(0, distilled_images.size(0), batch_size):
        end = min(distilled_images.size(0), start + batch_size)
        x = distilled_images[start:end].to(device=device)
        x = normalize_batch(x)
        logits = teacher(x)
        soft = F.softmax(logits / temperature, dim=1)
        soft_targets.append(soft.cpu())

    return torch.cat(soft_targets, dim=0)


def train_weighted_student(
    distilled_images: torch.Tensor,
    distilled_labels: torch.Tensor,
    distilled_weights: torch.Tensor,
    distilled_soft_labels: torch.Tensor,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    train_transform: transforms.Compose,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    amp_enabled: bool,
    output_dir: Path,
    imagenet_pretrained: bool,
    backbone: str,
) -> dict[str, Any]:
    train_set = DistilledSoftDataset(
        images=distilled_images,
        labels=distilled_labels,
        weights=distilled_weights,
        soft_labels=distilled_soft_labels,
        transform=train_transform,
    )
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = DataLoader(
        train_set,
        batch_size=min(batch_size, len(train_set)),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_classifier(
        num_classes=num_classes,
        imagenet_pretrained=imagenet_pretrained,
        backbone=backbone,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    best_val_acc = -1.0
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_weighted_loss = 0.0
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

            epoch_weighted_loss += float(loss.item()) * images.size(0)
            epoch_correct += int((logits.argmax(dim=1) == hard_labels).sum().item())
            epoch_samples += images.size(0)

        train_loss = epoch_weighted_loss / max(epoch_samples, 1)
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
            f"[Weighted-Training] epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
        )

    torch.save({"model": model.state_dict(), "history": history}, output_dir / "student_last.pt")

    with open(output_dir / "student_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    best_test_acc = 0.0
    for row in history:
        if int(row["epoch"]) == best_epoch:
            best_test_acc = row["test_acc"]
            break

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc_at_best_val": best_test_acc,
        "final_test_acc": history[-1]["test_acc"],
        "final_test_loss": history[-1]["test_loss"],
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    distilled_data_path = Path(args.distilled_data)
    print(f"[Setup] device={device}")
    print(f"[Setup] loading distilled data from {distilled_data_path}")

    distilled_images, distilled_labels, distilled_weights = load_distilled_data(distilled_data_path)

    train_set, val_set, test_set, num_classes = load_dermamnist_splits(args.data_root)
    train_transform, eval_transform = build_transforms(args.image_size)
    val_loader, test_loader = build_eval_loaders(
        val_set=val_set,
        test_set=test_set,
        eval_transform=eval_transform,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    teacher = train_teacher(
        train_set=train_set,
        val_loader=val_loader,
        test_loader=test_loader,
        train_transform=train_transform,
        num_classes=num_classes,
        epochs=args.teacher_epochs,
        learning_rate=args.teacher_lr,
        weight_decay=args.weight_decay,
        batch_size=args.teacher_batch_size,
        num_workers=args.num_workers,
        device=device,
        amp_enabled=amp_enabled,
        imagenet_pretrained=args.imagenet_pretrained,
        backbone=args.teacher_backbone,
        output_dir=output_dir,
    )

    teacher_test_detail = evaluate_classifier_detailed(
        model=teacher,
        loader=test_loader,
        device=device,
        num_classes=num_classes,
        amp_enabled=amp_enabled,
    )
    class_names = {str(i): INFO["dermamnist"]["label"][str(i)] for i in range(num_classes)}
    teacher_test_result = {
        "loss": float(teacher_test_detail["loss"]),
        "accuracy": float(teacher_test_detail["accuracy"]),
        "class_names": class_names,
        "per_class_accuracy": teacher_test_detail["per_class_accuracy"],
        "confusion_matrix": teacher_test_detail["confusion_matrix"],
    }

    with open(output_dir / "teacher_test_results.json", "w", encoding="utf-8") as handle:
        json.dump(teacher_test_result, handle, indent=2)

    print(
        "[Teacher-Test] "
        f"loss={teacher_test_result['loss']:.4f} "
        f"acc={teacher_test_result['accuracy']:.4f}"
    )

    soft_labels = make_teacher_soft_labels(
        teacher=teacher,
        distilled_images=distilled_images,
        temperature=args.temperature,
        batch_size=args.eval_batch_size,
        device=device,
    )

    training_summary = train_weighted_student(
        distilled_images=distilled_images,
        distilled_labels=distilled_labels,
        distilled_weights=distilled_weights,
        distilled_soft_labels=soft_labels,
        val_loader=val_loader,
        test_loader=test_loader,
        train_transform=train_transform,
        num_classes=num_classes,
        epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.train_lr,
        weight_decay=args.weight_decay,
        device=device,
        amp_enabled=amp_enabled,
        output_dir=output_dir,
        imagenet_pretrained=args.imagenet_pretrained,
        backbone=args.student_backbone,
    )

    summary = {
        "data_root": args.data_root,
        "distilled_data": str(distilled_data_path),
        "num_classes": num_classes,
        "num_distilled": int(distilled_images.size(0)),
        "amp_enabled": bool(amp_enabled),
        "image_size": int(args.image_size),
        "teacher_backbone": str(args.teacher_backbone),
        "student_backbone": str(args.student_backbone),
        "temperature": float(args.temperature),
        "teacher_epochs": int(args.teacher_epochs),
        "teacher_test_loss": float(teacher_test_result["loss"]),
        "teacher_test_acc": float(teacher_test_result["accuracy"]),
        "best_val_acc": float(training_summary["best_val_acc"]),
        "test_acc_at_best_val": float(training_summary["test_acc_at_best_val"]),
        "final_test_acc": float(training_summary["final_test_acc"]),
        "final_test_loss": float(training_summary["final_test_loss"]),
        "best_epoch": int(training_summary["best_epoch"]),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Done] Student training complete.")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train student model from pre-generated distilled images")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--distilled-data",
        type=str,
        default="outputs/dermamnist_224_distill/distilled_data.pt",
        help="Path to distilled_data.pt generated by run-distillation.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/dermamnist_224_student")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--teacher-backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone architecture for teacher model.",
    )
    parser.add_argument(
        "--student-backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone architecture for student model.",
    )

    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--teacher-lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=20.0)

    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--train-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
