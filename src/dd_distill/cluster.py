from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from .utils import *

@dataclass
class ClusterResult:
    centers: torch.Tensor
    center_labels: torch.Tensor
    counts: torch.Tensor
    weights: torch.Tensor

def global_clvq(
    latents: torch.Tensor,
    labels: torch.Tensor,
    clusters_per_class: float,
    num_classes: int,
    seed: int,
    max_iter: int,
    tol: float,
    minibatch_size: int,
    weighting_strategy: str,
    kmeans_max_iter: int = 300,
    weight_smooth: float = 0.0,
) -> ClusterResult:
    latents = latents.float().cpu()
    labels = labels.long().view(-1).cpu()
    flat_latents = latents.view(latents.size(0), -1).numpy().astype(np.float32, copy=False)
    labels_np = labels.numpy().astype(np.int64, copy=False)
    n_samples = flat_latents.shape[0]

    total_k = min(int(clusters_per_class * num_classes), n_samples)

    if n_samples <= total_k:
        # Not enough samples — use all latent vectors directly as centers.
        centers = flat_latents.astype(np.float32, copy=True)
        # Assign each sample to itself (identity)
        assignments = np.arange(n_samples, dtype=np.int64)
        counts = np.ones(n_samples, dtype=np.int64)
        backend_label = "NoCluster"
    else:
        # Use MiniBatchKMeans exclusively for global clustering.
        kmeans_mb = MiniBatchKMeans(
            n_clusters=total_k,
            random_state=seed,
            batch_size=minibatch_size,
            max_iter=max_iter,
            n_init=3,
            tol=float(max(tol, 1e-8)),
            reassignment_ratio=0.1,
        )
        assignments = kmeans_mb.fit_predict(flat_latents)
        centers = kmeans_mb.cluster_centers_.astype(np.float32, copy=False)
        counts = np.bincount(assignments, minlength=total_k).astype(np.int64)
        backend_label = "MiniBatchKMeans"

    # Remove empty clusters
    non_empty_mask = counts > 0
    centers = centers[non_empty_mask]
    counts = counts[non_empty_mask]

    if centers.shape[0] == 0:
        raise RuntimeError("Global CLVQ failed: no centers produced.")

    # Recompute assignments for non-empty centers
    if not np.all(non_empty_mask):
        assignments = assign_to_centers(flat_latents, centers, batch_size=2048)
        counts = np.bincount(assignments, minlength=centers.shape[0]).astype(np.int64)

    # Compute majority label per cluster
    center_labels_arr = np.zeros(centers.shape[0], dtype=np.int64)
    for c in range(centers.shape[0]):
        mask = assignments == c
        if mask.any():
            cluster_sample_labels = labels_np[mask]
            majority = np.bincount(cluster_sample_labels).argmax()
            center_labels_arr[c] = majority

    centers_t = torch.from_numpy(centers).view(centers.shape[0], *latents.shape[1:]).float()
    labels_t = torch.from_numpy(center_labels_arr).long()
    counts_t = torch.from_numpy(counts).long()
    weights_t = compute_cluster_weights(
        counts_t=counts_t,
        labels_t=labels_t,
        num_classes=num_classes,
        strategy=weighting_strategy,
        weight_smooth=weight_smooth,
    )

    print(
        f"[CLVQ/{backend_label}-Global] n_samples={n_samples} total_k={total_k} "
        f"kept={centers.shape[0]} batch_size={minibatch_size} max_iter={max_iter}"
    )

    return ClusterResult(
        centers=centers_t,
        center_labels=labels_t,
        counts=counts_t,
        weights=weights_t,
    )


def classwise_clvq(
    latents: torch.Tensor,
    labels: torch.Tensor,
    clusters_per_class: float,
    num_classes: int,
    seed: int,
    max_iter: int,
    tol: float,
    minibatch_size: int,
    weighting_strategy: str,
    kmeans_max_iter: int = 300,
    weight_smooth: float = 0.0,
) -> ClusterResult:
    """Per-class (classwise) CLVQ: cluster within each class independently,
    then concatenate results across all classes."""
    latents = latents.float().cpu()
    labels = labels.long().view(-1).cpu()
    flat_latents = latents.view(latents.size(0), -1).numpy().astype(np.float32, copy=False)
    labels_np = labels.numpy().astype(np.int64, copy=False)
    n_samples = flat_latents.shape[0]

    ipc = max(1, int(clusters_per_class))
    all_centers: list[np.ndarray] = []
    all_center_labels: list[int] = []
    all_counts: list[int] = []

    for class_id in range(num_classes):
        class_mask = labels_np == class_id
        class_indices = np.where(class_mask)[0]
        class_n = int(class_indices.size)

        if class_n == 0:
            print(f"[CLVQ-Classwise] class={class_id} has 0 samples, skipping.")
            continue

        k = min(ipc, class_n)

        if class_n <= k:
            # Not enough samples in this class — use all directly
            class_centers = flat_latents[class_indices].astype(np.float32, copy=True)
            class_counts = np.ones(class_n, dtype=np.int64)
            backend_label = "NoCluster"
        else:
            kmeans_mb = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed + class_id,
                batch_size=min(minibatch_size, class_n),
                max_iter=max_iter,
                n_init=3,
                tol=float(max(tol, 1e-8)),
                reassignment_ratio=0.1,
            )
            class_centers = kmeans_mb.fit(flat_latents[class_indices]).cluster_centers_
            class_centers = class_centers.astype(np.float32, copy=False)

            # Assign class samples to centers for counting
            class_assignments = assign_to_centers(
                flat_latents[class_indices], class_centers, batch_size=2048
            )
            class_counts = np.bincount(class_assignments, minlength=k).astype(np.int64)
            backend_label = "MiniBatchKMeans"

        all_centers.append(class_centers)
        all_center_labels.extend([class_id] * class_centers.shape[0])
        all_counts.extend(class_counts.tolist())

    if len(all_centers) == 0:
        raise RuntimeError("Classwise CLVQ failed: no centers produced across any class.")

    centers_full = np.concatenate(all_centers, axis=0)
    center_labels_arr = np.array(all_center_labels, dtype=np.int64)
    counts_arr = np.array(all_counts, dtype=np.int64)

    centers_t = torch.from_numpy(centers_full).view(centers_full.shape[0], *latents.shape[1:]).float()
    labels_t = torch.from_numpy(center_labels_arr).long()
    counts_t = torch.from_numpy(counts_arr).long()
    weights_t = compute_cluster_weights(
        counts_t=counts_t,
        labels_t=labels_t,
        num_classes=num_classes,
        strategy=weighting_strategy,
        weight_smooth=weight_smooth,
    )

    print(
        f"[CLVQ/{backend_label}-Classwise] n_samples={n_samples} ipc={ipc} "
        f"total_centers={centers_full.shape[0]} classes_present={len(all_centers)} "
        f"batch_size={minibatch_size} max_iter={max_iter}"
    )

    return ClusterResult(
        centers=centers_t,
        center_labels=labels_t,
        counts=counts_t,
        weights=weights_t,
    )


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


def compute_cluster_weights(
    counts_t: torch.Tensor,
    labels_t: torch.Tensor,
    num_classes: int,
    strategy: str,
    weight_smooth: float = 0.0,
) -> torch.Tensor:
    if counts_t.numel() != labels_t.numel():
        raise ValueError(
            f"counts/labels size mismatch: counts={counts_t.numel()} labels={labels_t.numel()}"
        )

    mode = str(strategy).strip().lower()
    if mode not in {"heuristic", "direct", "uniform", "inverse"}:
        raise ValueError(f"Unsupported weighting strategy: {strategy}")

    smooth = float(max(0.0, min(1.0, weight_smooth)))
    labels_l = labels_t.long().view(-1)
    counts_f = counts_t.float().view(-1).clamp_min(0.0)
    weights = torch.zeros_like(counts_f)

    if mode == "inverse":
        # Per‑class inverse: w_i ∝ 1/c_i, normalised within each class.
        # Smaller clusters → larger weight → more attention during student training.
        for class_id in range(num_classes):
            mask = labels_l == class_id
            class_k = int(mask.sum().item())
            if class_k <= 0:
                continue
            class_counts = counts_f[mask].clamp_min(1.0)
            inv_c = 1.0 / class_counts
            total_c = inv_c.sum().clamp_min(1e-12)
            weights[mask] = inv_c / total_c
            if smooth > 0.0:
                weights[mask] = weights[mask].pow(smooth)
        return weights

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

        if smooth > 0.0 and mode in {"heuristic", "inverse"}:
            weights[mask] = weights[mask].pow(smooth)

    return weights