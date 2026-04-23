from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.models import ResNet50_Weights, resnet50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize distilled images with t-SNE based on ResNet50 penultimate-layer features. "
            "Each class is shown with a different color."
        )
    )
    parser.add_argument("--distilled-data", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--output-name", type=str, default="distilled_tsne.png")
    parser.add_argument("--label-source", type=str, choices=["auto", "metadata", "softmax"], default="auto")
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--point-size", type=float, default=12.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid distilled payload type: {type(payload)}")
    if "images" not in payload:
        raise KeyError(f"Missing 'images' in {path}")
    return payload


def _labels_from_metadata(metadata_path: Path, expected_size: int) -> np.ndarray | None:
    if not metadata_path.exists():
        return None
    meta = _load_json(metadata_path)
    raw = meta.get("center_labels")
    if not isinstance(raw, list) or len(raw) == 0:
        return None

    labels = np.asarray(raw, dtype=np.int64)
    if int(labels.shape[0]) != int(expected_size):
        return None
    return labels


def _labels_from_softmax(payload: dict[str, Any], expected_size: int) -> np.ndarray | None:
    soft = payload.get("soft_labels")
    if not isinstance(soft, torch.Tensor):
        return None

    soft_t = soft.detach().cpu().float()
    if soft_t.ndim != 2 or int(soft_t.shape[0]) != int(expected_size):
        return None
    return soft_t.argmax(dim=1).numpy().astype(np.int64, copy=False)


def _resolve_labels(
    payload: dict[str, Any],
    metadata_path: Path,
    label_source: str,
    expected_size: int,
) -> tuple[np.ndarray, str]:
    source = str(label_source).strip().lower()
    if source not in {"auto", "metadata", "softmax"}:
        raise ValueError(f"Unsupported label_source: {label_source}")

    if source in {"metadata", "auto"}:
        labels = _labels_from_metadata(metadata_path, expected_size)
        if labels is not None:
            return labels, "metadata"
        if source == "metadata":
            raise ValueError(
                f"Failed to use metadata labels from {metadata_path}. "
                "Expected center_labels with length matching sample count."
            )

    labels = _labels_from_softmax(payload, expected_size)
    if labels is not None:
        return labels, "softmax"

    raise ValueError(
        "Failed to resolve labels. metadata labels unavailable and soft_labels missing/invalid in distilled_data.pt"
    )


def _stratified_subsample_indices(
    labels: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    n = int(labels.shape[0])
    limit = int(max_points)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    chosen: list[int] = []

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    quotas: dict[int, int] = {}
    base_taken = 0
    for cls, cls_count in zip(unique_labels.tolist(), label_counts.tolist()):
        quota = max(1, int(round(limit * (cls_count / max(n, 1)))))
        quota = min(quota, int(cls_count))
        quotas[int(cls)] = quota
        base_taken += quota

    if base_taken > limit:
        overflow = base_taken - limit
        for cls in sorted(quotas.keys(), key=lambda k: quotas[k], reverse=True):
            if overflow <= 0:
                break
            reducible = max(0, quotas[cls] - 1)
            delta = min(reducible, overflow)
            quotas[cls] -= delta
            overflow -= delta
    elif base_taken < limit:
        remain = limit - base_taken
        for cls in sorted(quotas.keys(), key=lambda k: quotas[k]):
            if remain <= 0:
                break
            cls_count = int((labels == cls).sum())
            addable = max(0, cls_count - quotas[cls])
            delta = min(addable, remain)
            quotas[cls] += delta
            remain -= delta

    for cls in unique_labels.tolist():
        class_idx = np.where(labels == int(cls))[0]
        rng.shuffle(class_idx)
        take = quotas[int(cls)]
        chosen.extend(class_idx[:take].tolist())

    chosen_np = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen_np)
    return chosen_np


def _normalize_images_to_01(images: torch.Tensor) -> torch.Tensor:
    img = images.detach().cpu().float()
    if img.ndim != 4:
        raise ValueError(f"Expected images shape (N,C,H,W), got {tuple(img.shape)}")

    min_val = float(img.min().item())
    max_val = float(img.max().item())
    if min_val >= 0.0 and max_val <= 1.0:
        pass
    elif min_val >= 0.0 and max_val <= 255.0 + 1e-6:
        img = img / 255.0
    elif min_val >= -1.0 - 1e-6 and max_val <= 1.0 + 1e-6:
        img = (img + 1.0) / 2.0
    else:
        img = img.clamp(0.0, 1.0)
    return img.clamp(0.0, 1.0)


@torch.no_grad()
def _extract_resnet50_features(
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
    imagenet_pretrained: bool,
) -> np.ndarray:
    img = _normalize_images_to_01(images)

    if img.size(1) == 1:
        img = img.repeat(1, 3, 1, 1)
    elif img.size(1) != 3:
        raise ValueError(f"Expected image channels in {1,3}, got C={img.size(1)}")

    if img.size(2) != 224 or img.size(3) != 224:
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)

    weights = ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=torch.float32).view(1, 3, 1, 1)

    total = int(img.size(0))
    step = max(1, int(batch_size))
    chunks: list[torch.Tensor] = []
    total_batches = (total + step - 1) // step

    for batch_idx, start in enumerate(range(0, total, step), start=1):
        end = min(total, start + step)
        batch = img[start:end].to(device=device, dtype=torch.float32)
        batch = (batch - mean) / std

        feat = model(batch)
        if feat.ndim != 2:
            feat = feat.view(feat.size(0), -1)
        chunks.append(feat.float().cpu())

        if batch_idx % 20 == 0 or batch_idx == total_batches:
            print(f"[Feature] batch {batch_idx}/{total_batches}")

    features = torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return features


def _safe_perplexity(requested: float, n_samples: int) -> float:
    if n_samples <= 2:
        raise ValueError("t-SNE requires at least 3 samples.")

    p = float(max(requested, 1.0))
    max_allowed = float(n_samples - 1)
    if p >= max_allowed:
        p = max(1.0, max_allowed - 1e-3)
    return p


def _compute_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    pca_dim: int,
    perplexity: float,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(features, dtype=np.float32)
    n_samples, n_dims = int(x.shape[0]), int(x.shape[1])

    pca_used = 0
    if int(pca_dim) > 0 and n_dims > int(pca_dim) and n_samples > 2:
        pca_used = min(int(pca_dim), n_samples - 1, n_dims)
        if pca_used >= 2:
            x = PCA(n_components=pca_used, random_state=seed).fit_transform(x)

    p = _safe_perplexity(perplexity, n_samples)
    iter_value = max(250, int(n_iter))
    tsne_kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": p,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
        "metric": "euclidean",
    }

    # sklearn changed TSNE iteration arg from n_iter to max_iter in newer versions.
    tsne_signature = inspect.signature(TSNE.__init__)
    if "max_iter" in tsne_signature.parameters:
        tsne_kwargs["max_iter"] = iter_value
    else:
        tsne_kwargs["n_iter"] = iter_value

    tsne = TSNE(**tsne_kwargs)
    embedding = tsne.fit_transform(x)

    stats = {
        "num_points": n_samples,
        "input_dim": n_dims,
        "pca_dim_used": pca_used,
        "perplexity_used": p,
        "n_iter": max(250, int(n_iter)),
        "num_classes": int(np.unique(labels).shape[0]),
    }
    return embedding, stats


def _class_color_map(num_classes: int) -> Any:
    if num_classes <= 10:
        return plt.cm.get_cmap("tab10", num_classes)
    if num_classes <= 20:
        return plt.cm.get_cmap("tab20", num_classes)
    return plt.cm.get_cmap("gist_ncar", num_classes)


def _plot_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    point_size: float,
    alpha: float,
    dpi: int,
) -> None:
    unique_labels = np.unique(labels)
    cmap = _class_color_map(int(unique_labels.shape[0]))

    fig, ax = plt.subplots(figsize=(9, 7))
    for idx, class_id in enumerate(unique_labels.tolist()):
        mask = labels == int(class_id)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=float(point_size),
            alpha=float(alpha),
            color=cmap(idx),
            label=f"class_{class_id}",
            edgecolors="none",
        )

    ax.set_title("t-SNE of Distilled Images")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(alpha=0.2)

    legend_cols = 1 if unique_labels.shape[0] <= 10 else 2 if unique_labels.shape[0] <= 24 else 3
    ax.legend(loc="best", fontsize=8, ncol=legend_cols, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=max(80, int(dpi)))
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    distilled_data_path = Path(args.distilled_data).expanduser().resolve()
    if not distilled_data_path.exists():
        raise FileNotFoundError(f"distilled_data.pt not found: {distilled_data_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else distilled_data_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = (
        Path(args.metadata).expanduser().resolve()
        if args.metadata
        else distilled_data_path.parent / "distilled_metadata.json"
    )

    payload = _load_payload(distilled_data_path)
    images = payload["images"]
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"images must be torch.Tensor, got {type(images)}")

    total_samples = int(images.size(0))
    labels, label_source_used = _resolve_labels(
        payload=payload,
        metadata_path=metadata_path,
        label_source=args.label_source,
        expected_size=total_samples,
    )

    sampled_indices = _stratified_subsample_indices(
        labels=labels,
        max_points=args.max_points,
        seed=args.seed,
    )
    sampled_labels = labels[sampled_indices]
    sampled_images = images[sampled_indices]

    device = _resolve_device(args.device)
    features = _extract_resnet50_features(
        images=sampled_images,
        batch_size=args.feature_batch_size,
        device=device,
        imagenet_pretrained=bool(args.imagenet_pretrained),
    )

    embedding, stats = _compute_tsne(
        features=features,
        labels=sampled_labels,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        seed=args.seed,
    )

    image_path = output_dir / args.output_name
    npz_path = output_dir / "distilled_tsne_embedding.npz"
    summary_path = output_dir / "distilled_tsne_summary.json"

    _plot_embedding(
        embedding=embedding,
        labels=sampled_labels,
        output_path=image_path,
        point_size=args.point_size,
        alpha=args.alpha,
        dpi=args.dpi,
    )

    np.savez_compressed(
        npz_path,
        embedding=embedding.astype(np.float32),
        labels=sampled_labels.astype(np.int64),
    )

    summary: dict[str, Any] = {
        "distilled_data": str(distilled_data_path),
        "metadata": str(metadata_path),
        "label_source": label_source_used,
        "feature_extractor": "resnet50_penultimate",
        "feature_batch_size": int(max(1, int(args.feature_batch_size))),
        "device": str(device),
        "imagenet_pretrained": bool(args.imagenet_pretrained),
        "sampled_points": int(sampled_indices.shape[0]),
        "output_image": str(image_path),
        "output_embedding": str(npz_path),
        "stats": stats,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Done] t-SNE visualization generated:")
    print(f"- image: {image_path}")
    print(f"- embedding: {npz_path}")
    print(f"- summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
