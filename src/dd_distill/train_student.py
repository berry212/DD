from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass
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

from .datasets import TorchDataset, get_dataset_spec, supported_datasets
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


def build_train_transform(
    image_size: int,
    backbone: str,
    min_scale: float,
    max_scale: float,
    horizontal_flip_prob: float,
) -> transforms.Compose:
    mean, std = resolve_backbone_normalization(backbone)
    min_s = float(np.clip(min_scale, 1e-4, 1.0))
    max_s = float(np.clip(max_scale, min_s, 1.0))
    hflip_p = float(np.clip(horizontal_flip_prob, 0.0, 1.0))
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((image_size, image_size), scale=(min_s, max_s)),
            transforms.RandomHorizontalFlip(p=hflip_p),
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
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        TorchDataset(val_set, transform=eval_transform),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        TorchDataset(test_set, transform=eval_transform),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return val_loader, test_loader


@dataclass
class DistilledDataBundle:
    num_samples: int
    images: torch.Tensor | None
    image_relative_paths: list[str]
    image_shards: list[str]
    weights: torch.Tensor
    soft_labels: torch.Tensor
    dataset: str
    lora_path: str
    teacher_temperature: float
    distill_method: str
    fkd_batch_path: str
    fkd_batch_summary: dict[str, Any]


def load_distilled_triplet(distilled_data_path: Path) -> DistilledDataBundle:
    payload = torch.load(distilled_data_path, map_location="cpu")

    required = {"weights", "soft_labels"}
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Missing keys in {distilled_data_path}: {missing}")

    images_obj = payload.get("images")
    images: torch.Tensor | None = None
    if isinstance(images_obj, torch.Tensor):
        images = images_obj.float()
    image_shards_obj = payload.get("image_shards")
    image_shards = [str(v) for v in image_shards_obj] if isinstance(image_shards_obj, list) else []
    rel_paths_obj = payload.get("image_relative_paths")
    image_relative_paths = [str(v) for v in rel_paths_obj] if isinstance(rel_paths_obj, list) else []

    weights = payload["weights"].float().view(-1)
    soft_labels = payload["soft_labels"].float()
    distilled_dataset = str(payload.get("dataset", "")).strip().lower()
    distilled_lora_path = str(payload.get("lora_path", "")).strip()
    teacher_temperature = float(payload.get("teacher_temperature", 0.0) or 0.0)
    distill_method = str(payload.get("distill_method", "clvq")).strip().lower()
    fkd_batch_path = str(payload.get("fkd_batch_path", "")).strip()
    fkd_batch_summary_obj = payload.get("fkd_batch_summary")
    fkd_batch_summary = dict(fkd_batch_summary_obj) if isinstance(fkd_batch_summary_obj, dict) else {}

    if soft_labels.ndim != 2:
        raise ValueError(f"soft_labels must be 2D (N,C), got {tuple(soft_labels.shape)}")

    n = int(weights.numel())
    if soft_labels.size(0) != n:
        raise ValueError(
            "Mismatch among images/weights/soft_labels sizes: "
            f"N={n}, weights={weights.numel()}, soft_labels={soft_labels.size(0)}"
        )

    if not bool(torch.isfinite(weights).all()):
        raise ValueError(f"weights contain NaN/Inf: {distilled_data_path}")
    if not bool(torch.isfinite(soft_labels).all()):
        raise ValueError(f"soft_labels contain NaN/Inf: {distilled_data_path}")

    if images is not None:
        if images.ndim != 4:
            raise ValueError(f"images must be 4D (N,C,H,W), got {tuple(images.shape)}")
        if images.size(0) != n:
            raise ValueError(
                f"Mismatch among images/weights/soft_labels sizes: N={images.size(0)} "
                f"weights={weights.numel()} soft_labels={soft_labels.size(0)}"
            )
        if not bool(torch.isfinite(images).all()):
            raise ValueError(f"images contain NaN/Inf: {distilled_data_path}")

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
    else:
        if image_relative_paths and len(image_relative_paths) != n:
            raise ValueError(
                f"image_relative_paths length mismatch: expected={n} actual={len(image_relative_paths)}"
            )
        if not image_relative_paths and not image_shards:
            raise KeyError(
                "Missing images in distilled payload. Expected one of: 'images', 'image_shards', 'image_relative_paths'."
            )

    weights = weights.clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        warnings.warn("Weights are all zero; fallback to uniform weights.", RuntimeWarning)
        weights = torch.ones_like(weights)

    soft_labels = soft_labels.clamp_min(0.0)
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return DistilledDataBundle(
        num_samples=n,
        images=images,
        image_relative_paths=image_relative_paths,
        image_shards=image_shards,
        weights=weights,
        soft_labels=soft_labels,
        dataset=distilled_dataset,
        lora_path=distilled_lora_path,
        teacher_temperature=teacher_temperature,
        distill_method=distill_method,
        fkd_batch_path=fkd_batch_path,
        fkd_batch_summary=fkd_batch_summary,
    )


def load_fkd_batch_payload(distilled_data_path: Path, fkd_batch_path: str) -> dict[str, Any]:
    if not fkd_batch_path:
        raise ValueError("FKD batch path is empty.")

    batch_path = distilled_data_path.parent / fkd_batch_path
    if not batch_path.exists():
        raise FileNotFoundError(f"FKD batch cache not found: {batch_path}")

    payload = torch.load(batch_path, map_location="cpu")
    required = {"indices", "crop_params", "flip_mask", "soft_labels", "batch_size", "batches_per_epoch", "train_epochs"}
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Missing FKD batch keys in {batch_path}: {missing}")

    return payload


class DistilledImageAccessor:
    def __init__(self, distilled_data_path: Path, bundle: DistilledDataBundle) -> None:
        self.distilled_data_path = distilled_data_path
        self.images = bundle.images
        self.image_relative_paths = list(bundle.image_relative_paths)
        self.image_shards = list(bundle.image_shards)
        self.num_samples = int(bundle.num_samples)

        self._cached_shard_rel = ""
        self._cached_shard_start = -1
        self._cached_shard_end = -1
        self._cached_shard_images: torch.Tensor | None = None
        self._shard_manifest: list[tuple[int, int, str]] | None = None

    def __len__(self) -> int:
        return self.num_samples

    def get_image(self, index: int) -> torch.Tensor:
        idx = int(index)
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"distilled image index out of range: {idx}")

        if self.images is not None:
            return self.images[idx].float().cpu()
        if self.image_relative_paths:
            return self._load_image_from_relative_path(self.image_relative_paths[idx])
        return self._load_image_from_shard(idx)

    def _resolve_relative_path(self, rel_path: str) -> Path:
        rel_str = str(rel_path)
        candidate_a = self.distilled_data_path.parent / "distilled_images" / rel_str
        candidate_b = self.distilled_data_path.parent / rel_str
        abs_path = candidate_a if candidate_a.exists() else candidate_b
        if not abs_path.exists():
            raise FileNotFoundError(f"Distilled image file not found: {candidate_a} or {candidate_b}")
        return abs_path

    def _load_image_from_relative_path(self, rel_path: str) -> torch.Tensor:
        abs_path = self._resolve_relative_path(rel_path)
        with Image.open(abs_path) as pil_img:
            img_np = np.asarray(pil_img, dtype=np.float32)
        if img_np.ndim == 2:
            img_np = img_np[:, :, None]
        image_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous() / 255.0
        return image_t.float().clamp(0.0, 1.0)

    def _ensure_shard_manifest(self) -> list[tuple[int, int, str]]:
        if self._shard_manifest is not None:
            return self._shard_manifest

        manifest: list[tuple[int, int, str]] = []
        cursor = 0
        for shard_rel in self.image_shards:
            shard_abs = self.distilled_data_path.parent / shard_rel
            shard_payload = torch.load(shard_abs, map_location="cpu")
            shard_images = shard_payload.get("images")
            if not isinstance(shard_images, torch.Tensor):
                raise ValueError(f"Missing tensor 'images' in shard: {shard_abs}")
            start = int(shard_payload.get("start_index", cursor))
            end = int(shard_payload.get("end_index", start + shard_images.size(0)))
            manifest.append((start, end, shard_rel))
            cursor = end
        self._shard_manifest = manifest
        return manifest

    def _load_image_from_shard(self, index: int) -> torch.Tensor:
        if self._cached_shard_images is None or not (self._cached_shard_start <= index < self._cached_shard_end):
            shard_rel = ""
            shard_start = -1
            shard_end = -1
            for start, end, rel in self._ensure_shard_manifest():
                if start <= index < end:
                    shard_start = start
                    shard_end = end
                    shard_rel = rel
                    break
            if not shard_rel:
                raise IndexError(f"Could not resolve shard for distilled image index: {index}")

            shard_abs = self.distilled_data_path.parent / shard_rel
            shard_payload = torch.load(shard_abs, map_location="cpu")
            shard_images = shard_payload.get("images")
            if not isinstance(shard_images, torch.Tensor):
                raise ValueError(f"Missing tensor 'images' in shard: {shard_abs}")

            shard_t = shard_images.float()
            if shard_t.max().item() > 1.0:
                shard_t = shard_t / 255.0
            self._cached_shard_rel = shard_rel
            self._cached_shard_start = shard_start
            self._cached_shard_end = shard_end
            self._cached_shard_images = shard_t.clamp(0.0, 1.0).cpu()

        local_index = index - self._cached_shard_start
        return self._cached_shard_images[local_index].float().cpu()


class LazyDistilledTripletDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        image_accessor: DistilledImageAccessor,
        weights: torch.Tensor,
        soft_labels: torch.Tensor,
        transform: transforms.Compose,
    ) -> None:
        super().__init__()
        self.image_accessor = image_accessor
        self.weights = weights.float().cpu()
        self.soft_labels = soft_labels.float().cpu()
        self.transform = transform

    @override
    def __len__(self) -> int:
        return len(self.image_accessor)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.transform(self.image_accessor.get_image(index))
        soft = self.soft_labels[index]
        weight = self.weights[index]
        return image, soft, weight


class DistilledTTMBatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        image_accessor: DistilledImageAccessor,
        weights: torch.Tensor,
        batch_indices: torch.Tensor,
        crop_params: torch.Tensor,
        flip_mask: torch.Tensor,
        batch_soft_labels: torch.Tensor,
        backbone: str,
        image_size: int,
    ) -> None:
        super().__init__()
        self.image_accessor = image_accessor
        self.weights = weights.float().cpu()
        self.batch_indices = batch_indices.long().cpu()
        self.crop_params = crop_params.to(dtype=torch.int16).cpu()
        self.flip_mask = flip_mask.bool().cpu()
        self.batch_soft_labels = batch_soft_labels.float().cpu()
        self.image_size = int(image_size)

        mean, std = resolve_backbone_normalization(backbone)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    @override
    def __len__(self) -> int:
        return int(self.batch_indices.size(0))

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(index)
        indices = self.batch_indices[idx]
        soft_labels = self.batch_soft_labels[idx]

        augmented_images: list[torch.Tensor] = []
        for pos, sample_idx in enumerate(indices.tolist()):
            crop = tuple(int(v) for v in self.crop_params[idx, pos].tolist())
            do_flip = bool(self.flip_mask[idx, pos].item())
            image = apply_resized_crop_with_flip(
                image=self.image_accessor.get_image(int(sample_idx)),
                crop_params=crop,
                output_size=self.image_size,
                horizontal_flip=do_flip,
            )
            augmented_images.append(image)

        batch_images = torch.stack(augmented_images, dim=0).float()
        mean = self.mean.to(dtype=batch_images.dtype)
        std = self.std.to(dtype=batch_images.dtype)
        batch_images = (batch_images - mean.unsqueeze(0)) / std.unsqueeze(0)

        sample_weights = self.weights[indices].float()
        return batch_images, soft_labels.float(), sample_weights


def unwrap_single_batch(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(samples) != 1:
        raise ValueError(f"Expected a single pre-batched sample, received {len(samples)}")
    return samples[0]


def sharpen_soft_labels(soft_labels: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(max(temperature, 1e-6))
    if abs(t - 1.0) < 1e-8:
        return soft_labels

    logits = torch.log(soft_labels.clamp_min(1e-12))
    sharpened = F.softmax(logits / t, dim=1)
    return sharpened


def blend_sample_weights(
    weights: torch.Tensor,
    pseudo_labels: torch.Tensor,
    num_classes: int,
    balance_alpha: float,
) -> torch.Tensor:
    alpha = float(np.clip(balance_alpha, 0.0, 1.0))
    base = weights.clamp_min(0.0)
    if float(base.sum().item()) <= 0.0:
        base = torch.ones_like(base)

    if alpha <= 0.0:
        return base / base.mean().clamp_min(1e-12)

    base = base / base.mean().clamp_min(1e-12)

    class_counts = torch.bincount(pseudo_labels.long(), minlength=num_classes).float()
    class_counts = class_counts.clamp_min(0.0)
    present = class_counts > 0
    inv = torch.zeros_like(class_counts)
    inv[present] = 1.0 / class_counts[present]

    balanced = inv[pseudo_labels.long()]
    balanced = balanced / balanced.mean().clamp_min(1e-12)

    mixed = (1.0 - alpha) * base + alpha * balanced
    mixed = mixed.clamp_min(0.0)
    return mixed / mixed.mean().clamp_min(1e-12)


def resolve_kd_temperature(requested_temperature: float, distilled_teacher_temperature: float) -> float:
    requested = float(requested_temperature)
    distilled = float(distilled_teacher_temperature)

    if requested > 0.0:
        if distilled > 0.0 and not np.isclose(requested, distilled, atol=1e-6):
            warnings.warn(
                "Student KD temperature does not match the distilled teacher temperature. "
                f"student={requested:.4f} teacher={distilled:.4f}",
                RuntimeWarning,
            )
        return requested

    if distilled > 0.0:
        return distilled
    return 1.0


def resolve_student_lr(backbone: str, train_lr: float) -> float:
    if float(train_lr) > 0.0:
        return float(train_lr)
    return 2e-3 if normalize_backbone_name(backbone) == "resnet18" else 1e-3


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
    train_loader: Any,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    amp_enabled: bool,
    output_dir: Path,
    kd_temperature: float,
) -> dict[str, Any]:
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    best_val_acc = -1.0
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        temp = float(max(kd_temperature, 1e-6))

        for images, soft_labels, sample_weights in train_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device=device, dtype=torch.float32)
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                student_log_probs = F.log_softmax(logits.float() / temp, dim=1)
                per_sample_ce = -(soft_labels * student_log_probs).sum(dim=1)
                loss = (per_sample_ce * sample_weights).sum() / float(max(images.size(0), 1))
                loss = loss * (temp * temp)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item()) * images.size(0)
            # Track accuracy using argmax of soft labels as pseudo-labels
            pseudo_labels = soft_labels.argmax(dim=1)
            epoch_correct += int((logits.argmax(dim=1) == pseudo_labels).sum().item())
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
    distilled_bundle = load_distilled_triplet(distilled_data_path)
    num_distilled = int(distilled_bundle.num_samples)
    weights = distilled_bundle.weights
    soft_labels = distilled_bundle.soft_labels
    distilled_dataset = distilled_bundle.dataset
    distilled_lora_path = distilled_bundle.lora_path
    image_accessor = DistilledImageAccessor(distilled_data_path, distilled_bundle)

    pseudo_labels = torch.argmax(soft_labels, dim=1).long()
    weights = blend_sample_weights(
        weights=weights,
        pseudo_labels=pseudo_labels,
        num_classes=num_classes,
        balance_alpha=args.weight_balance_alpha,
    )

    if distilled_dataset and distilled_dataset != dataset_spec.name:
        raise ValueError(
            "Distilled data dataset mismatch: "
            f"expected={dataset_spec.name} actual={distilled_dataset} path={distilled_data_path}"
        )
    print(
        f"[Setup] dataset={dataset_spec.name} device={device} N={num_distilled} "
        f"weights_mean={weights.mean().item():.6f} soft_shape={tuple(soft_labels.shape)}"
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

    train_mode = "samplewise"
    effective_train_epochs = max(1, int(args.train_epochs))
    effective_train_batch_size = max(1, int(min(args.train_batch_size, max(num_distilled, 1))))
    resolved_kd_temperature = resolve_kd_temperature(
        requested_temperature=args.kd_temperature,
        distilled_teacher_temperature=distilled_bundle.teacher_temperature,
    )

    fkd_batches_used = 0

    if bool(args.use_fkd_batches) and distilled_bundle.fkd_batch_path:
        fkd_payload = load_fkd_batch_payload(distilled_data_path, distilled_bundle.fkd_batch_path)
        precomputed_epochs = int(fkd_payload["train_epochs"])
        batches_per_epoch = int(fkd_payload["batches_per_epoch"])
        if effective_train_epochs > precomputed_epochs:
            raise ValueError(
                "Requested more student epochs than available FKD batches: "
                f"requested={effective_train_epochs} available={precomputed_epochs}"
            )

        effective_train_batch_size = int(fkd_payload["batch_size"])
        fkd_batches_used = effective_train_epochs * batches_per_epoch
        batch_soft_labels = fkd_payload["soft_labels"][:fkd_batches_used].float()
        original_shape = batch_soft_labels.shape
        batch_soft_labels = sharpen_soft_labels(
            batch_soft_labels.view(-1, original_shape[-1]),
            temperature=args.soft_label_sharpen,
        ).view(original_shape)

        if int(args.train_batch_size) != effective_train_batch_size:
            warnings.warn(
                "FKD cache provides a fixed batch size; ignoring train_batch_size in favor of the cached value "
                f"({effective_train_batch_size}).",
                RuntimeWarning,
            )

        train_set = DistilledTTMBatchDataset(
            image_accessor=image_accessor,
            weights=weights,
            batch_indices=fkd_payload["indices"][:fkd_batches_used],
            crop_params=fkd_payload["crop_params"][:fkd_batches_used],
            flip_mask=fkd_payload["flip_mask"][:fkd_batches_used],
            batch_soft_labels=batch_soft_labels,
            backbone=args.student_backbone,
            image_size=args.image_size,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=unwrap_single_batch,
        )
        train_mode = "fkd_batches"
        print(
            f"[FKD] using cached augmented batches: batches={fkd_batches_used} "
            f"per_epoch={batches_per_epoch} batch_size={effective_train_batch_size}"
        )
    else:
        if bool(args.use_fkd_batches) and not distilled_bundle.fkd_batch_path:
            warnings.warn(
                "FKD batches requested but not found in distilled_data.pt; falling back to sample-wise soft labels.",
                RuntimeWarning,
            )

        soft_labels = sharpen_soft_labels(soft_labels, temperature=args.soft_label_sharpen)
        train_transform = build_train_transform(
            image_size=args.image_size,
            backbone=args.student_backbone,
            min_scale=args.train_crop_min_scale,
            max_scale=args.train_crop_max_scale,
            horizontal_flip_prob=args.train_horizontal_flip_prob,
        )
        train_set = LazyDistilledTripletDataset(
            image_accessor=image_accessor,
            weights=weights,
            soft_labels=soft_labels,
            transform=train_transform,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=effective_train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    model = build_classifier(
        num_classes=num_classes,
        imagenet_pretrained=args.imagenet_pretrained,
        backbone=args.student_backbone,
    ).to(device)

    resolved_train_lr = resolve_student_lr(backbone=args.student_backbone, train_lr=args.train_lr)

    training_summary = train_student(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=effective_train_epochs,
        learning_rate=resolved_train_lr,
        weight_decay=args.weight_decay,
        device=device,
        amp_enabled=amp_enabled,
        output_dir=output_dir,
        kd_temperature=resolved_kd_temperature,
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
        "num_distilled": int(num_distilled),
        "distilled_dataset": distilled_dataset or dataset_spec.name,
        "distilled_lora_path": distilled_lora_path,
        "train_mode": train_mode,
        "fkd_batch_path": distilled_bundle.fkd_batch_path,
        "fkd_batches_used": int(fkd_batches_used),
        "student_backbone": str(args.student_backbone),
        "amp_enabled": bool(amp_enabled),
        "train_lr": float(resolved_train_lr),
        "train_epochs": int(effective_train_epochs),
        "train_batch_size": int(effective_train_batch_size),
        "kd_temperature": float(resolved_kd_temperature),
        "distilled_teacher_temperature": float(distilled_bundle.teacher_temperature),
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
        default="resnet18",
        choices=list(SUPPORTED_BACKBONES),
    )
    parser.add_argument("--train-epochs", type=int, default=300)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--train-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kd-temperature", type=float, default=0.0)
    parser.add_argument("--weight-balance-alpha", type=float, default=0.0)
    parser.add_argument("--soft-label-sharpen", type=float, default=1.0)
    parser.add_argument("--train-crop-min-scale", type=float, default=0.08)
    parser.add_argument("--train-crop-max-scale", type=float, default=1.0)
    parser.add_argument("--train-horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--use-fkd-batches", action=argparse.BooleanOptionalAction, default=True)

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
