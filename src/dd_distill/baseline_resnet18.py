from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


class DermamnistNpzDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose,
    ) -> None:
        self.images = np.ascontiguousarray(images)
        self.labels = labels.reshape(-1).astype(np.int64, copy=False)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = int(self.labels[index])
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


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


def load_dermamnist_npz(npz_path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(npz_path)
    expected_keys = {
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
    }
    missing = expected_keys.difference(arrays.files)
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {sorted(missing)}")

    return {
        "train_images": arrays["train_images"],
        "train_labels": arrays["train_labels"],
        "val_images": arrays["val_images"],
        "val_labels": arrays["val_labels"],
        "test_images": arrays["test_images"],
        "test_labels": arrays["test_labels"],
    }


def maybe_subsample(
    images: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or max_samples >= len(images):
        return images, labels

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=max_samples, replace=False)
    return images[indices], labels[indices]


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


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    arrays = load_dermamnist_npz(Path(args.data_npz))

    train_images, train_labels = maybe_subsample(
        arrays["train_images"], arrays["train_labels"], args.max_train_samples, args.seed
    )
    val_images, val_labels = maybe_subsample(arrays["val_images"], arrays["val_labels"], args.max_val_samples, args.seed)
    test_images, test_labels = maybe_subsample(
        arrays["test_images"], arrays["test_labels"], args.max_test_samples, args.seed
    )

    train_labels_1d = train_labels.reshape(-1)
    val_labels_1d = val_labels.reshape(-1)
    test_labels_1d = test_labels.reshape(-1)
    num_classes = int(max(train_labels_1d.max(), val_labels_1d.max(), test_labels_1d.max()) + 1)

    train_transform, eval_transform = build_transforms(args.image_size)

    train_set = DermamnistNpzDataset(train_images, train_labels, train_transform)
    val_set = DermamnistNpzDataset(val_images, val_labels, eval_transform)
    test_set = DermamnistNpzDataset(test_images, test_labels, eval_transform)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.device == "cuda",
    }

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, num_classes


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

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
        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc, y_true, y_pred


def class_names_for_dermamnist(num_classes: int) -> list[str]:
    names = [f"class_{i}" for i in range(num_classes)]
    try:
        from medmnist import INFO

        mapping = INFO["dermamnist"]["label"]
        names = [mapping.get(str(i), names[i]) for i in range(num_classes)]
    except Exception:
        pass
    return names


def write_results(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    summary: dict[str, Any],
) -> None:
    class_names = class_names_for_dermamnist(num_classes)

    labels = list(range(num_classes))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(report_text)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / "classification_report.csv", index=True)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=True)

    preds_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "correct": (y_true == y_pred).astype(np.int64),
        }
    )
    preds_df.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    with open(output_dir / "baseline_summary.json", "w", encoding="utf-8") as handle:
        json.dump({**summary, "test_metrics": metrics}, handle, indent=2)

    print("[Baseline] Test classification report")
    print(report_text)
    print(json.dumps(metrics, indent=2))


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.device = device.type

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, num_classes = create_dataloaders(args)

    model = build_model(num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    best_val_acc = -1.0
    best_epoch = -1

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
        )

        scheduler.step()

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                output_dir / "resnet18_best.pt",
            )

        print(
            f"[Train] epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    with open(output_dir / "train_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    best_ckpt = torch.load(output_dir / "resnet18_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=amp_enabled,
    )

    summary = {
        "data_npz": args.data_npz,
        "num_classes": num_classes,
        "pretrained": bool(args.pretrained),
        "device": device.type,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }

    write_results(
        output_dir=output_dir,
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        summary=summary,
    )

    print("[Baseline] Summary")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ResNet18 baseline training on Dermamnist dataset")
    parser.add_argument("--data-npz", type=str, default="data/dermamnist_224.npz")
    parser.add_argument("--output-dir", type=str, default="outputs/dermamnist_resnet18_baseline")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
