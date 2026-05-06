from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DiTTransformer2DModel
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .baseline_resnet18 import load_teacher_checkpoint, train_teacher_baseline
from .datasets import TorchDataset, get_dataset_spec, supported_datasets
from .utils import *


@dataclass
class CLVQResult:
    centers: torch.Tensor
    center_labels: torch.Tensor
    counts: torch.Tensor
    weights: torch.Tensor


@dataclass
class SelectedSamplesResult:
    indices: torch.Tensor
    labels: torch.Tensor
    counts: torch.Tensor
    weights: torch.Tensor


@torch.no_grad()
def make_teacher_soft_labels(
    teacher: nn.Module,
    distilled_images: torch.Tensor,
    temperature: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    teacher.eval()
    soft_targets: list[torch.Tensor] = []

    for start in range(0, distilled_images.size(0), batch_size):
        end = min(distilled_images.size(0), start + batch_size)
        x = distilled_images[start:end].to(device=device)
        x = normalize_batch(x)
        logits = teacher(x)
        soft = F.softmax(logits / max(temperature, 1e-6), dim=1)
        soft_targets.append(soft.cpu())

    return torch.cat(soft_targets, dim=0)


def build_encode_loader(
    train_set: Any,
    image_size: int,
    encode_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    encode_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    return DataLoader(
        TorchDataset(train_set, transform=encode_transform),
        batch_size=encode_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def load_vae(model_id: str, device: torch.device, dtype: torch.dtype) -> tuple[AutoencoderKL, float]:
    print(f"[Encoding] loading AutoencoderKL: {model_id}")
    try:
        vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype)
    except Exception:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    return vae, scaling_factor


@torch.no_grad()
def encode_training_images(
    vae: AutoencoderKL,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    scaling_factor: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_latents: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    total_batches = len(data_loader)
    for batch_idx, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device=device, dtype=dtype)
        images = images * 2.0 - 1.0

        posterior = vae.encode(images).latent_dist
        latents = posterior.mean * scaling_factor

        all_latents.append(latents.float().cpu())
        all_labels.append(labels.cpu())

        if batch_idx % 20 == 0 or batch_idx == total_batches:
            print(f"[Encoding] batch {batch_idx}/{total_batches}")

    return torch.cat(all_latents, dim=0), torch.cat(all_labels, dim=0)


def initialize_clvq_centers(data: np.ndarray, num_centers: int, seed: int) -> np.ndarray:
    if num_centers <= 0:
        raise ValueError("num_centers must be positive.")

    n_samples = int(data.shape[0])
    if n_samples == 0:
        raise ValueError("Cannot initialize CLVQ centers from an empty dataset.")

    rng = np.random.default_rng(seed)
    replace = num_centers > n_samples
    chosen = rng.choice(n_samples, size=num_centers, replace=replace)
    return data[chosen].astype(np.float32, copy=True)


def target_count_for_class(class_samples: int, ipc: float) -> int:
    if class_samples <= 0:
        return 0

    ipc_is_integer = float(ipc).is_integer()
    if ipc_is_integer:
        target_k = int(ipc)
    else:
        target_k = int(round(class_samples * float(ipc)))

    return max(1, min(target_k, class_samples))


def compute_classwise_cluster_weights(
    counts_t: torch.Tensor,
    labels_t: torch.Tensor,
    num_classes: int,
    strategy: str,
) -> torch.Tensor:
    if counts_t.numel() != labels_t.numel():
        raise ValueError(
            f"counts/labels size mismatch: counts={counts_t.numel()} labels={labels_t.numel()}"
        )

    mode = str(strategy).strip().lower()
    if mode not in {"heuristic", "direct", "uniform"}:
        raise ValueError(f"Unsupported weighting strategy: {strategy}")

    labels_l = labels_t.long().view(-1)
    counts_f = counts_t.float().view(-1).clamp_min(0.0)
    weights = torch.zeros_like(counts_f)

    for class_id in range(num_classes):
        mask = labels_l == class_id
        class_k = int(mask.sum().item())
        if class_k <= 0:
            continue

        if mode == "uniform":
            weights[mask] = 1.0
            continue

        class_counts = counts_f[mask].clamp_min(1.0)
        class_mass = class_counts.sum().clamp_min(1e-12)
        if mode == "direct":
            # Direct cluster weights: normalized assignment counts in each class.
            weights[mask] = class_counts / class_mass
        else:
            # DDOQ Appendix H Eq.(34): w_k^(L) = K_L * v_k^(L) / sum_j v_j^(L).
            weights[mask] = float(class_k) * class_counts / class_mass

    return weights


def build_epoch_batch_indices(
    num_samples: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    sample_count = int(max(1, num_samples))
    batch_n = int(max(1, batch_size))
    num_batches = max(1, int(np.ceil(sample_count / batch_n)))

    epoch_indices = torch.randperm(sample_count, generator=generator)
    required = num_batches * batch_n
    if required > sample_count:
        extra = torch.randperm(sample_count, generator=generator)[: required - sample_count]
        epoch_indices = torch.cat([epoch_indices, extra], dim=0)

    return epoch_indices.view(num_batches, batch_n).long()


@torch.no_grad()
def precompute_fkd_batch_cache(
    teacher: nn.Module,
    distilled_images: torch.Tensor,
    image_size: int,
    batch_size: int,
    train_epochs: int,
    crop_min_scale: float,
    crop_max_scale: float,
    horizontal_flip_prob: float,
    soft_label_temperature: float,
    eval_batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    teacher.eval()

    images = distilled_images.float().cpu()
    num_samples = int(images.size(0))
    if num_samples <= 0:
        raise ValueError("FKD precomputation requires at least one distilled image.")

    batch_n = max(1, min(int(batch_size), num_samples))
    epoch_count = max(1, int(train_epochs))
    batches_per_epoch = max(1, int(np.ceil(num_samples / batch_n)))
    total_batches = epoch_count * batches_per_epoch

    min_scale = float(np.clip(crop_min_scale, 1e-4, 1.0))
    max_scale = float(np.clip(crop_max_scale, min_scale, 1.0))
    hflip_prob = float(np.clip(horizontal_flip_prob, 0.0, 1.0))

    generator = torch.Generator()
    generator.manual_seed(int(seed) + 7001)

    batch_indices_chunks: list[torch.Tensor] = []
    crop_param_chunks: list[torch.Tensor] = []
    flip_mask_chunks: list[torch.Tensor] = []
    soft_label_chunks: list[torch.Tensor] = []

    batch_cursor = 0
    for _epoch in range(epoch_count):
        epoch_batches = build_epoch_batch_indices(num_samples=num_samples, batch_size=batch_n, generator=generator)
        for local_batch_indices in epoch_batches:
            augmented_images: list[torch.Tensor] = []
            crop_params: list[tuple[int, int, int, int]] = []
            flip_mask: list[bool] = []

            for sample_idx in local_batch_indices.tolist():
                image = images[int(sample_idx)]
                crop = sample_random_resized_crop_params(
                    image_height=int(image.shape[-2]),
                    image_width=int(image.shape[-1]),
                    scale=(min_scale, max_scale),
                    generator=generator,
                )
                do_flip = bool(torch.rand(1, generator=generator).item() < hflip_prob)
                augmented = apply_resized_crop_with_flip(
                    image=image,
                    crop_params=crop,
                    output_size=image_size,
                    horizontal_flip=do_flip,
                )
                augmented_images.append(augmented)
                crop_params.append(crop)
                flip_mask.append(do_flip)

            batch_images = torch.stack(augmented_images, dim=0)
            batch_soft_labels = make_teacher_soft_labels(
                teacher=teacher,
                distilled_images=batch_images,
                temperature=soft_label_temperature,
                batch_size=eval_batch_size,
                device=device,
            )

            batch_indices_chunks.append(local_batch_indices.long().cpu())
            crop_param_chunks.append(torch.tensor(crop_params, dtype=torch.int16))
            flip_mask_chunks.append(torch.tensor(flip_mask, dtype=torch.bool))
            soft_label_chunks.append(batch_soft_labels.float().cpu())

            batch_cursor += 1
            if batch_cursor % 20 == 0 or batch_cursor == total_batches:
                print(f"[FKD] precomputed {batch_cursor}/{total_batches} augmented batches")

    return {
        "indices": torch.stack(batch_indices_chunks, dim=0).long(),
        "crop_params": torch.stack(crop_param_chunks, dim=0).to(dtype=torch.int16),
        "flip_mask": torch.stack(flip_mask_chunks, dim=0).bool(),
        "soft_labels": torch.stack(soft_label_chunks, dim=0).float(),
        "batch_size": int(batch_n),
        "batches_per_epoch": int(batches_per_epoch),
        "train_epochs": int(epoch_count),
        "num_batches": int(total_batches),
        "image_size": int(image_size),
        "crop_min_scale": float(min_scale),
        "crop_max_scale": float(max_scale),
        "horizontal_flip_prob": float(hflip_prob),
        "seed": int(seed),
    }


def assign_to_centers(data: np.ndarray, centers: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    if data.size == 0 or centers.size == 0:
        raise ValueError("assign_to_centers expects non-empty data and centers.")

    center_norm = np.sum(centers * centers, axis=1)
    assignments = np.empty(data.shape[0], dtype=np.int64)

    for start in range(0, data.shape[0], batch_size):
        end = min(data.shape[0], start + batch_size)
        chunk = data[start:end]
        chunk_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances = chunk_norm + center_norm[None, :] - 2.0 * (chunk @ centers.T)
        assignments[start:end] = np.argmin(distances, axis=1)

    return assignments


def anchor_centers_to_medoids(
    data: np.ndarray,
    centers: np.ndarray,
    anchor: float,
    batch_size: int = 2048,
) -> np.ndarray:
    if data.size == 0 or centers.size == 0:
        raise ValueError("anchor_centers_to_medoids expects non-empty data and centers.")

    alpha = float(np.clip(anchor, 0.0, 1.0))
    if alpha <= 0.0:
        return centers

    center_norm = np.sum(centers * centers, axis=1)
    best_dist = np.full((centers.shape[0],), fill_value=np.inf, dtype=np.float64)
    best_samples = centers.copy()

    for start in range(0, data.shape[0], batch_size):
        end = min(data.shape[0], start + batch_size)
        chunk = data[start:end]
        chunk_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances = chunk_norm + center_norm[None, :] - 2.0 * (chunk @ centers.T)

        winner_rows = np.argmin(distances, axis=0)
        winner_dist = distances[winner_rows, np.arange(centers.shape[0])]
        improved = winner_dist < best_dist

        if np.any(improved):
            best_dist[improved] = winner_dist[improved]
            best_samples[improved] = chunk[winner_rows[improved]]

    anchored = (1.0 - alpha) * centers + alpha * best_samples
    return anchored.astype(np.float32, copy=False)


def nearest_samples_to_centers_unique(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if data.size == 0 or centers.size == 0:
        raise ValueError("nearest_samples_to_centers_unique expects non-empty data and centers.")

    chosen: list[int] = []
    used: set[int] = set()
    for center in centers:
        distances = np.sum((data - center[None, :]) ** 2, axis=1)
        sorted_indices = np.argsort(distances)
        picked = int(sorted_indices[0])
        for candidate in sorted_indices:
            candidate_i = int(candidate)
            if candidate_i not in used:
                picked = candidate_i
                break
        used.add(picked)
        chosen.append(picked)

    return np.asarray(chosen, dtype=np.int64)


def classwise_random_selection(
    labels: torch.Tensor,
    clusters_per_class: float,
    num_classes: int,
    seed: int,
    weighting_strategy: str,
) -> SelectedSamplesResult:
    if clusters_per_class <= 0:
        raise ValueError("clusters_per_class must be positive.")

    ipc = float(clusters_per_class)
    labels_np = labels.long().view(-1).cpu().numpy().astype(np.int64, copy=False)

    index_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []

    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels_np == class_id)
        class_samples = int(class_indices.shape[0])
        if class_samples == 0:
            warnings.warn(f"Class {class_id} has no samples; skipping.", RuntimeWarning)
            continue

        class_k = target_count_for_class(class_samples, ipc)
        class_seed = seed + 1009 * (class_id + 1)
        rng = np.random.default_rng(class_seed)
        picked = np.asarray(rng.choice(class_indices, size=class_k, replace=False), dtype=np.int64)

        index_chunks.append(torch.from_numpy(picked))
        label_chunks.append(torch.full((class_k,), fill_value=class_id, dtype=torch.long))
        count_chunks.append(torch.ones((class_k,), dtype=torch.long))

        print(f"[Random-Class] class={class_id} samples={class_samples} selected={class_k}")

    if not index_chunks:
        raise RuntimeError("Random distillation failed: no samples selected.")

    indices_t = torch.cat(index_chunks, dim=0).long()
    labels_t = torch.cat(label_chunks, dim=0).long()
    counts_t = torch.cat(count_chunks, dim=0).long()
    weights_t = compute_classwise_cluster_weights(
        counts_t=counts_t,
        labels_t=labels_t,
        num_classes=num_classes,
        strategy=weighting_strategy,
    )

    return SelectedSamplesResult(indices=indices_t, labels=labels_t, counts=counts_t, weights=weights_t)


def classwise_kmeans_nearest_selection(
    latents: torch.Tensor,
    labels: torch.Tensor,
    clusters_per_class: float,
    num_classes: int,
    seed: int,
    weighting_strategy: str,
    kmeans_max_iter: int,
) -> SelectedSamplesResult:
    if clusters_per_class <= 0:
        raise ValueError("clusters_per_class must be positive.")

    ipc = float(clusters_per_class)
    kmeans_max_iter = max(10, int(kmeans_max_iter))
    flat_latents = latents.float().cpu().view(latents.size(0), -1).numpy().astype(np.float32, copy=False)
    labels_np = labels.long().view(-1).cpu().numpy().astype(np.int64, copy=False)

    index_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []

    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels_np == class_id)
        class_samples = int(class_indices.shape[0])
        if class_samples == 0:
            warnings.warn(f"Class {class_id} has no samples; skipping.", RuntimeWarning)
            continue

        class_k = target_count_for_class(class_samples, ipc)
        class_seed = seed + 1009 * (class_id + 1)
        class_latents = flat_latents[class_indices]

        kmeans = KMeans(n_clusters=class_k, random_state=class_seed, n_init=10, max_iter=kmeans_max_iter)
        assignments = kmeans.fit_predict(class_latents)
        nearest_local = nearest_samples_to_centers_unique(class_latents, kmeans.cluster_centers_)
        picked_global = class_indices[nearest_local]
        counts = np.bincount(assignments, minlength=class_k).astype(np.int64)

        index_chunks.append(torch.from_numpy(picked_global.astype(np.int64, copy=False)))
        label_chunks.append(torch.full((class_k,), fill_value=class_id, dtype=torch.long))
        count_chunks.append(torch.from_numpy(counts).long())

        print(
            f"[KMeans-Class] class={class_id} samples={class_samples} "
            f"selected={class_k} max_iter={kmeans_max_iter}"
        )

    if not index_chunks:
        raise RuntimeError("KMeans distillation failed: no samples selected.")

    indices_t = torch.cat(index_chunks, dim=0).long()
    labels_t = torch.cat(label_chunks, dim=0).long()
    counts_t = torch.cat(count_chunks, dim=0).long()
    weights_t = compute_classwise_cluster_weights(
        counts_t=counts_t,
        labels_t=labels_t,
        num_classes=num_classes,
        strategy=weighting_strategy,
    )

    return SelectedSamplesResult(indices=indices_t, labels=labels_t, counts=counts_t, weights=weights_t)


def gather_images_by_indices(
    train_set: Any,
    indices: torch.Tensor,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices_np = indices.long().cpu().numpy().astype(np.int64, copy=False)
    if indices_np.size == 0:
        raise ValueError("gather_images_by_indices expects at least one index.")

    image_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = TorchDataset(train_set, transform=image_transform)

    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for raw_idx in indices_np.tolist():
        image_t, label_t = dataset[int(raw_idx)]
        images.append(image_t)
        labels.append(label_t)

    return torch.stack(images, dim=0), torch.stack(labels, dim=0).long()


def iter_images_by_indices(
    train_set: Any,
    indices: torch.Tensor,
    image_size: int,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    indices_np = indices.long().cpu().numpy().astype(np.int64, copy=False)
    if indices_np.size == 0:
        raise ValueError("iter_images_by_indices expects at least one index.")

    chunk_size = max(1, int(batch_size))
    image_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = TorchDataset(train_set, transform=image_transform)

    for start in range(0, indices_np.size, chunk_size):
        end = min(indices_np.size, start + chunk_size)
        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for raw_idx in indices_np[start:end].tolist():
            image_t, label_t = dataset[int(raw_idx)]
            images.append(image_t)
            labels.append(label_t)
        yield torch.stack(images, dim=0), torch.stack(labels, dim=0).long()


def classwise_clvq(
    latents: torch.Tensor,
    labels: torch.Tensor,
    clusters_per_class: float,
    num_classes: int,
    seed: int,
    max_iter: int,
    tol: float,
    minibatch_size: int,
    medoid_anchor: float,
    weighting_strategy: str,
) -> CLVQResult:
    if clusters_per_class <= 0:
        raise ValueError("clusters_per_class must be positive.")

    print(
        "[CLVQ] Using the paper-aligned class-wise clustering path implemented with "
        "MiniBatchKMeans in latent space."
    )

    ipc = float(clusters_per_class)

    minibatch_size = max(32, int(minibatch_size))
    max_iter = max(10, int(max_iter))

    latents = latents.float().cpu()
    labels = labels.long().view(-1).cpu()

    flat_latents = latents.view(latents.size(0), -1).numpy().astype(np.float32, copy=False)
    labels_np = labels.numpy().astype(np.int64, copy=False)

    center_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []

    for class_id in range(num_classes):
        class_selector = labels_np == class_id
        class_latents = flat_latents[class_selector]
        class_samples = int(class_latents.shape[0])
        if class_samples == 0:
            warnings.warn(f"Class {class_id} has no samples; skipping.", RuntimeWarning)
            continue

        target_k = target_count_for_class(class_samples, ipc)
        class_k = target_k
        class_seed = seed + 1009 * (class_id + 1)
        kmeans = MiniBatchKMeans(
            n_clusters=class_k,
            random_state=class_seed,
            batch_size=minibatch_size,
            max_iter=max_iter,
            n_init=3,
            tol=float(max(tol, 1e-8)),
            reassignment_ratio=0.01,
        )
        assignments = kmeans.fit_predict(class_latents)
        centers = kmeans.cluster_centers_.astype(np.float32, copy=False)

        if medoid_anchor > 0.0:
            centers = anchor_centers_to_medoids(
                data=class_latents,
                centers=centers,
                anchor=medoid_anchor,
                batch_size=2048,
            )
            assignments = assign_to_centers(class_latents, centers, batch_size=2048)

        counts = np.bincount(assignments, minlength=class_k).astype(np.int64)

        non_empty_mask = counts > 0
        centers = centers[non_empty_mask]
        counts = counts[non_empty_mask]

        if centers.shape[0] == 0:
            continue

        center_chunks.append(torch.from_numpy(centers).view(centers.shape[0], *latents.shape[1:]).float())
        label_chunks.append(torch.full((centers.shape[0],), fill_value=class_id, dtype=torch.long))
        count_chunks.append(torch.from_numpy(counts))

        print(
            f"[CLVQ/MiniBatchKMeans-Class] class={class_id} samples={class_samples} "
            f"target_k={target_k} used={class_k} kept={centers.shape[0]} "
            f"batch_size={minibatch_size} max_iter={max_iter} medoid_anchor={medoid_anchor:.2f}"
        )

    if not center_chunks:
        raise RuntimeError("Class-wise MiniBatchKMeans failed: no centers produced.")

    centers_t = torch.cat(center_chunks, dim=0)
    labels_t = torch.cat(label_chunks, dim=0)
    counts_t = torch.cat(count_chunks, dim=0)
    weights_t = compute_classwise_cluster_weights(
        counts_t=counts_t,
        labels_t=labels_t,
        num_classes=num_classes,
        strategy=weighting_strategy,
    )

    return CLVQResult(
        centers=centers_t,
        center_labels=labels_t,
        counts=counts_t,
        weights=weights_t,
    )


class DiTDecoder:
    def __init__(
        self,
        model_id: str,
        vae: AutoencoderKL,
        device: torch.device,
        dtype: torch.dtype,
        num_inference_steps: int,
        noise_strength: float,
        lora_path: str,
        lora_scale: float,
        guidance_scale: float,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = max(2, int(num_inference_steps))
        self.noise_strength = float(np.clip(noise_strength, 0.01, 1.0))
        self.guidance_scale = float(max(0.0, guidance_scale))

        self.transformer = DiTTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self.transformer.to(device)
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae = vae
        self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        try:
            self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        except Exception:
            self.scheduler = DDIMScheduler.from_pretrained(model_id)

        self.transformer = self._load_lora_if_provided(
            transformer=self.transformer,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )

        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.null_class_label = self._resolve_null_class_label()
        self._warned_guidance = False

    def _resolve_null_class_label(self) -> int | None:
        embedding = getattr(self.transformer, "class_embedding", None)
        if embedding is None:
            embedding = getattr(self.transformer, "class_embed", None)
        if embedding is None or not hasattr(embedding, "num_embeddings"):
            return None
        num_embeddings = int(embedding.num_embeddings)
        if num_embeddings <= 0:
            return None
        return num_embeddings - 1

    def _load_lora_if_provided(
        self,
        transformer: nn.Module,
        lora_path: str,
        lora_scale: float,
    ) -> nn.Module:
        if not lora_path:
            return transformer

        adapter_dir = Path(lora_path)
        config_path = adapter_dir / "lora_config.json"
        weight_path = adapter_dir / "lora_transformer.pt"
        if not adapter_dir.exists() or not config_path.exists() or not weight_path.exists():
            warnings.warn(
                f"LoRA artifacts not found under {adapter_dir}; expected lora_config.json and lora_transformer.pt.",
                RuntimeWarning,
            )
            return transformer

        with open(config_path, "r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        allowed_keys = {
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "bias",
            "task_type",
            "init_lora_weights",
        }
        config_payload = {k: raw_config[k] for k in allowed_keys if k in raw_config}
        lora_config = LoraConfig(**config_payload)

        if hasattr(transformer, "add_adapter"):
            transformer.add_adapter(lora_config)
        else:
            transformer = get_peft_model(transformer, lora_config)

        state_dict = torch.load(weight_path, map_location="cpu")
        set_peft_model_state_dict(transformer, state_dict)
        self._apply_lora_scale(transformer, lora_scale)
        print(f"[LoRA] Loaded DiT adapter from {adapter_dir} with scale={lora_scale}")
        return transformer

    @staticmethod
    def _apply_lora_scale(model: nn.Module, lora_scale: float) -> None:
        scale = float(lora_scale)
        if scale == 1.0:
            return
        for module in model.modules():
            if hasattr(module, "scaling"):
                scaling = getattr(module, "scaling")
                if isinstance(scaling, torch.Tensor):
                    module.scaling = scaling * scale
                else:
                    module.scaling = float(scaling) * scale

    @torch.no_grad()
    def _decode_batch(self, centers: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        start_idx = int((1.0 - self.noise_strength) * (len(timesteps) - 1))
        start_idx = max(0, min(start_idx, len(timesteps) - 1))
        start_timestep = timesteps[start_idx]

        latents = centers.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, start_timestep.expand(latents.size(0)))

        for timestep in timesteps[start_idx:]:
            model_input = self.scheduler.scale_model_input(latents, timestep)

            if self.guidance_scale > 1.0 and self.null_class_label is not None:
                model_input = torch.cat([model_input, model_input], dim=0)
                null_labels = torch.full_like(labels, fill_value=int(self.null_class_label))
                class_labels = torch.cat([null_labels, labels], dim=0)
                noise_pred = self.transformer(
                    model_input,
                    timestep,
                    class_labels=class_labels,
                ).sample
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                if self.guidance_scale > 1.0 and self.null_class_label is None and not self._warned_guidance:
                    warnings.warn(
                        "Guidance scale > 1 requested but DiT model does not expose a null class label; "
                        "running without classifier-free guidance.",
                        RuntimeWarning,
                    )
                    self._warned_guidance = True
                noise_pred = self.transformer(
                    model_input,
                    timestep,
                    class_labels=labels,
                ).sample

            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

        images = self.vae.decode(latents / self.scaling_factor).sample
        return (images / 2.0 + 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def decode(self, centers: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        total = centers.size(0)
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            imgs = self._decode_batch(centers[start:end], labels[start:end])
            chunks.append(imgs.float().cpu())
            print(f"[Decoding] {end}/{total}")
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def decode_batches(
        self,
        centers: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        total = centers.size(0)
        chunk_size = max(1, int(batch_size))
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            imgs = self._decode_batch(centers[start:end], labels[start:end])
            print(f"[Decoding] {end}/{total}")
            yield imgs.float().cpu(), labels[start:end].long().cpu()

    def cleanup(self) -> None:
        del self.transformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_distillation(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.fp16 and device.type == "cuda")
    dataset_spec = get_dataset_spec(args.dataset)
    output_dir = Path(args.output_dir or f"outputs/{dataset_spec.name}_224_distill")
    teacher_baseline_dir = Path(args.teacher_baseline_dir or f"outputs/{dataset_spec.name}_224_distill_baseline")
    resolved_lora_path = resolve_lora_path(dataset_spec.name, args.lora_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_baseline_dir.mkdir(parents=True, exist_ok=True)

    resolved_vae_model_id = str(args.vae_model_id).strip() or str(args.diffusion_model_id)
    run_config = vars(args).copy()
    run_config["output_dir"] = str(output_dir)
    run_config["teacher_baseline_dir"] = str(teacher_baseline_dir)
    run_config["lora_path"] = resolved_lora_path
    run_config["resolved_vae_model_id"] = resolved_vae_model_id
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    split_bundle = dataset_spec.load_dataset_splits(data_root=args.data_root, image_size=args.image_size)
    train_set = split_bundle.train_set
    val_set = split_bundle.val_set
    test_set = split_bundle.test_set
    num_classes = split_bundle.num_classes
    print(
        f"[Setup] dataset={dataset_spec.name} device={device} "
        f"train={len(train_set)} classes={num_classes} baseline_dir={teacher_baseline_dir} "
        f"lora_path={resolved_lora_path}"
    )
    if str(args.teacher_backbone).strip().lower() != "resnet18":
        warnings.warn(
            "Paper-aligned soft-label protocol uses a ResNet-18 teacher. "
            f"Current teacher_backbone={args.teacher_backbone}",
            RuntimeWarning,
        )

    teacher_ckpt_path = teacher_baseline_dir / "teacher_best.pt"
    if not teacher_ckpt_path.exists():
        if not bool(args.auto_train_teacher_baseline):
            raise FileNotFoundError(
                "Teacher checkpoint not found. Run `bash baseline.sh` first or pass "
                f"`--auto-train-teacher-baseline`. Missing path: {teacher_ckpt_path}"
            )

        print(f"[Teacher] Missing checkpoint, training baseline explicitly: {teacher_ckpt_path}")
        teacher, _ = train_teacher_baseline(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            num_classes=num_classes,
            args=args,
            device=device,
            amp_enabled=amp_enabled,
            baseline_dir=teacher_baseline_dir,
        )
    else:
        try:
            teacher, _ = load_teacher_checkpoint(
                checkpoint_path=teacher_ckpt_path,
                num_classes=num_classes,
                backbone=args.teacher_backbone,
                imagenet_pretrained=args.imagenet_pretrained,
                device=device,
            )
            print(f"[Teacher] Loaded existing checkpoint: {teacher_ckpt_path}")
        except Exception as exc:
            if not bool(args.auto_train_teacher_baseline):
                raise RuntimeError(
                    "Failed to load teacher checkpoint. Re-run `bash baseline.sh` or pass "
                    f"`--auto-train-teacher-baseline`. path={teacher_ckpt_path}"
                ) from exc

            warnings.warn(
                f"Failed to load teacher checkpoint ({teacher_ckpt_path}), retraining explicitly. error={exc}",
                RuntimeWarning,
            )
            teacher, _ = train_teacher_baseline(
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                num_classes=num_classes,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                baseline_dir=teacher_baseline_dir,
            )

    # Latents are independent of IPC, so cache them under dataset baseline dir.
    latent_cache_path = teacher_baseline_dir / "train_latents.pt"
    vae_dtype = torch.float16 if amp_enabled else torch.float32
    vae_model_id = resolved_vae_model_id
    vae, scaling_factor = load_vae(vae_model_id, device=device, dtype=vae_dtype)

    cached_latents = load_latent_cache(
        cache_path=latent_cache_path,
        expected_num_samples=len(train_set),
        expected_dataset=dataset_spec.name,
        expected_vae_model_id=vae_model_id,
        expected_image_size=args.image_size,
    )

    latent_source = "cache"
    if cached_latents is None:
        encode_loader = build_encode_loader(
            train_set=train_set,
            image_size=args.image_size,
            encode_batch_size=args.encode_batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        latents, latent_labels = encode_training_images(
            vae=vae,
            data_loader=encode_loader,
            scaling_factor=scaling_factor,
            device=device,
            dtype=vae_dtype,
        )
        save_latent_cache(
            cache_path=latent_cache_path,
            latents=latents,
            labels=latent_labels,
            dataset_name=dataset_spec.name,
            vae_model_id=vae_model_id,
            image_size=args.image_size,
        )
        latent_source = "vae_encoder"
    else:
        latents, latent_labels = cached_latents

    stream_batch_size = max(1, int(args.save_batch_size))

    decoder: DiTDecoder | None = None

    if args.distill_method == "clvq":
        clvq = classwise_clvq(
            latents=latents,
            labels=latent_labels,
            clusters_per_class=args.clusters_per_class,
            num_classes=num_classes,
            seed=args.seed,
            max_iter=args.clvq_max_iter,
            tol=args.clvq_tol,
            minibatch_size=args.clvq_batch_size,
            medoid_anchor=args.clvq_medoid_anchor,
            weighting_strategy=args.weighting_strategy,
        )

        decoder = DiTDecoder(
            model_id=args.diffusion_model_id,
            vae=vae,
            device=device,
            dtype=vae_dtype,
            num_inference_steps=args.sde_steps,
            noise_strength=args.sde_noise_strength,
            lora_path=resolved_lora_path,
            lora_scale=args.lora_scale,
            guidance_scale=args.guidance_scale,
        )

        stream_iter = decoder.decode_batches(
            clvq.centers,
            clvq.center_labels,
            batch_size=min(stream_batch_size, max(1, int(args.decode_batch_size))),
        )

        distilled_labels = clvq.center_labels
        distilled_counts = clvq.counts
        distilled_weights = clvq.weights
    elif args.distill_method == "random":
        selected = classwise_random_selection(
            labels=latent_labels,
            clusters_per_class=args.clusters_per_class,
            num_classes=num_classes,
            seed=args.seed,
            weighting_strategy=args.weighting_strategy,
        )
        expected_labels = selected.labels.long().cpu()
        stream_iter = iter_images_by_indices(
            train_set=train_set,
            indices=selected.indices,
            image_size=args.image_size,
            batch_size=stream_batch_size,
        )
        distilled_labels = expected_labels
        distilled_counts = selected.counts
        distilled_weights = selected.weights
    elif args.distill_method == "kmeans":
        selected = classwise_kmeans_nearest_selection(
            latents=latents,
            labels=latent_labels,
            clusters_per_class=args.clusters_per_class,
            num_classes=num_classes,
            seed=args.seed,
            weighting_strategy=args.weighting_strategy,
            kmeans_max_iter=args.kmeans_max_iter,
        )
        expected_labels = selected.labels.long().cpu()
        stream_iter = iter_images_by_indices(
            train_set=train_set,
            indices=selected.indices,
            image_size=args.image_size,
            batch_size=stream_batch_size,
        )
        distilled_labels = expected_labels
        distilled_counts = selected.counts
        distilled_weights = selected.weights
    else:
        raise ValueError(f"Unsupported distill method: {args.distill_method}")

    saved_paths: list[str] = []
    shard_paths: list[str] = []
    soft_label_chunks: list[torch.Tensor] = []
    preview_chunks: list[torch.Tensor] = []
    image_chunks: list[torch.Tensor] = []
    collect_full_images = bool(args.store_images_in_pt or args.fkd_precompute_batches)
    preview_budget = 100
    cursor = 0
    shard_root = output_dir / "distilled_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    try:
        for batch_idx, (image_batch, label_batch) in enumerate(stream_iter, start=1):
            image_batch = image_batch.float().cpu()
            label_batch = label_batch.long().cpu()
            batch_n = int(image_batch.size(0))

            if args.distill_method in {"random", "kmeans"}:
                expected = expected_labels[cursor : cursor + batch_n]
                if not torch.equal(label_batch, expected):
                    warnings.warn(
                        f"Gathered labels mismatch selected labels in batch {batch_idx}; using gathered labels.",
                        RuntimeWarning,
                    )
                    distilled_labels[cursor : cursor + batch_n] = label_batch

            soft_batch = make_teacher_soft_labels(
                teacher=teacher,
                distilled_images=image_batch,
                temperature=args.teacher_temperature,
                batch_size=args.eval_batch_size,
                device=device,
            )
            soft_label_chunks.append(soft_batch)

            rel_paths = save_distilled_images(
                images=image_batch,
                labels=label_batch,
                output_dir=output_dir,
                start_index=cursor,
                max_workers=args.save_async_workers,
            )
            saved_paths.extend(rel_paths)

            shard_rel = str(Path("distilled_shards") / f"shard_{len(shard_paths):05d}.pt")
            shard_abs = output_dir / shard_rel
            torch.save(
                {
                    "images": (image_batch * 255.0).clamp(0.0, 255.0).to(torch.uint8),
                    "labels": label_batch,
                    "weights": distilled_weights[cursor : cursor + batch_n].float().cpu(),
                    "soft_labels": soft_batch.float().cpu(),
                    "start_index": int(cursor),
                    "end_index": int(cursor + batch_n),
                },
                shard_abs,
            )
            shard_paths.append(shard_rel)

            if preview_budget > 0:
                keep_n = min(preview_budget, batch_n)
                preview_chunks.append(image_batch[:keep_n])
                preview_budget -= keep_n

            if collect_full_images:
                image_chunks.append(image_batch)

            cursor += batch_n
            print(f"[Save] batch={batch_idx} saved={cursor}/{distilled_labels.size(0)}")
    finally:
        if decoder is not None:
            decoder.cleanup()

    if cursor != int(distilled_labels.size(0)):
        raise RuntimeError(
            f"Saved sample count mismatch: saved={cursor} expected={int(distilled_labels.size(0))}"
        )

    soft_labels = torch.cat(soft_label_chunks, dim=0)
    if preview_chunks:
        preview_images = torch.cat(preview_chunks, dim=0)
        save_preview_grid(preview_images, output_dir / "distilled_preview.png")

    distilled_images_full: torch.Tensor | None = None
    if collect_full_images:
        distilled_images_full = torch.cat(image_chunks, dim=0)

    distilled_images_for_pt: torch.Tensor | None = None
    if bool(args.store_images_in_pt):
        distilled_images_for_pt = distilled_images_full

    fkd_batch_path = ""
    fkd_batch_summary: dict[str, object] | None = None
    if bool(args.fkd_precompute_batches):
        if distilled_images_full is None:
            raise RuntimeError("FKD precomputation requested but distilled images were not retained in memory.")

        fkd_batch_cache = precompute_fkd_batch_cache(
            teacher=teacher,
            distilled_images=distilled_images_full,
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

    save_distillation_artifacts(
        output_dir=output_dir,
        images=distilled_images_for_pt,
        weights=distilled_weights,
        soft_labels=soft_labels,
        center_labels=distilled_labels,
        counts=distilled_counts,
        saved_paths=saved_paths,
        dataset_name=dataset_spec.name,
        lora_path=resolved_lora_path,
        teacher_temperature=args.teacher_temperature,
        image_shards=shard_paths,
        store_images_in_pt=bool(args.store_images_in_pt),
        fkd_batch_path=fkd_batch_path,
        fkd_batch_summary=fkd_batch_summary,
    )

    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "dataset": dataset_spec.name,
        "data_root": args.data_root,
        "diffusion_model_id": str(args.diffusion_model_id),
        "vae_model_id": str(vae_model_id),
        "teacher_baseline_dir": str(teacher_baseline_dir),
        "teacher_checkpoint": str(teacher_ckpt_path),
        "num_classes": int(num_classes),
        "distill_method": str(args.distill_method),
        "clustering_backend": "MiniBatchKMeans" if args.distill_method == "clvq" else str(args.distill_method),
        "clusters_per_class": float(args.clusters_per_class),
        "clvq_medoid_anchor": float(args.clvq_medoid_anchor),
        "clvq_batch_size": int(args.clvq_batch_size),
        "clvq_minibatch_size": int(args.clvq_batch_size),
        "kmeans_max_iter": int(args.kmeans_max_iter),
        "num_distilled": int(cursor),
        "teacher_backbone": str(args.teacher_backbone),
        "teacher_temperature": float(args.teacher_temperature),
        "weighting_strategy": str(args.weighting_strategy),
        "fkd_precompute_batches": bool(args.fkd_precompute_batches),
        "fkd_batch_path": str(fkd_batch_path),
        "fkd_batch_summary": dict(fkd_batch_summary or {}),
        "lora_path": resolved_lora_path,
        "latent_cache_path": str(latent_cache_path),
        "latent_source": latent_source,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Done] Distillation complete.")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset distillation: explicit teacher baseline + VAE encode + "
            "class-wise CLVQ (implemented via MiniBatchKMeans) / KMeans / Random selection "
            "+ optional DiT reverse-SDE decode"
        )
    )
    parser.add_argument("--dataset", default="dermamnist", choices=supported_datasets())
    parser.add_argument("--data-root", type=str, default=default_data_root())
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--teacher-baseline-dir", type=str, default="")

    parser.add_argument("--clusters-per-class", type=float, default=100.0)
    parser.add_argument("--distill-method", type=str, default="clvq", choices=["clvq", "random", "kmeans"])
    parser.add_argument("--clvq-max-iter", type=int, default=10000)
    parser.add_argument("--clvq-tol", type=float, default=1e-5)
    parser.add_argument("--clvq-batch-size", type=int, default=1024)
    parser.add_argument("--clvq-medoid-anchor", type=float, default=0.0)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument(
        "--weighting-strategy",
        type=str,
        default="heuristic",
        choices=["heuristic", "direct", "uniform"],
    )

    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--save-batch-size", type=int, default=64)
    parser.add_argument("--save-async-workers", type=int, default=0)
    parser.add_argument("--store-images-in-pt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fkd-precompute-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fkd-train-epochs", type=int, default=300)
    parser.add_argument("--fkd-batch-size", type=int, default=1024)
    parser.add_argument("--fkd-crop-min-scale", type=float, default=0.08)
    parser.add_argument("--fkd-crop-max-scale", type=float, default=1.0)
    parser.add_argument("--fkd-horizontal-flip-prob", type=float, default=0.5)

    parser.add_argument("--vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--diffusion-model-id", type=str, default="facebook/DiT-XL-2-256")
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--lora-scale", type=float, default=0.9)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--sde-steps", type=int, default=200)
    parser.add_argument("--sde-noise-strength", type=float, default=0.2)

    parser.add_argument("--teacher-backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--teacher-batch-size", type=int, default=128)
    parser.add_argument("--teacher-lr", type=float, default=3e-4)
    parser.add_argument("--teacher-weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-temperature", type=float, default=20.0)
    parser.add_argument("--auto-train-teacher-baseline", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_distillation(args)


if __name__ == "__main__":
    main()
