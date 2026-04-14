from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read distilled_data.pt weights and generate visualization plots."
    )
    parser.add_argument(
        "--distilled-data",
        type=str,
        required=True,
        help="Path to distilled_data.pt (contains key: weights).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="",
        help="Optional path to distilled_metadata.json (for class-wise plot using center_labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for plots. Defaults to the distilled_data.pt parent folder.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Maximum number of bars shown in sorted weight plot.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=50,
        help="Bin count for histogram plot.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid distilled data payload type: {type(payload)}")
    if "weights" not in payload:
        raise KeyError(f"Missing 'weights' in {path}")
    return payload


def maybe_load_center_labels(metadata_path: Path) -> np.ndarray | None:
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    labels = meta.get("center_labels")
    if not isinstance(labels, list):
        return None

    if not labels:
        return None

    return np.asarray(labels, dtype=np.int64)


def save_sorted_plot(weights: np.ndarray, output_path: Path, top_k: int) -> None:
    sorted_weights = np.sort(weights)[::-1]
    n = min(len(sorted_weights), max(1, int(top_k)))
    y = sorted_weights[:n]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(n), y, width=0.9, color="#1f77b4")
    ax.set_title("Distilled Weights (Sorted)")
    ax.set_xlabel("Cluster Rank")
    ax.set_ylabel("Weight")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_hist_plot(weights: np.ndarray, output_path: Path, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(weights, bins=max(5, int(bins)), color="#ff7f0e", edgecolor="white")
    ax.set_title("Distilled Weights Histogram")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_class_plot(weights: np.ndarray, center_labels: np.ndarray, output_path: Path) -> None:
    if len(center_labels) != len(weights):
        raise ValueError(
            f"center_labels length mismatch: labels={len(center_labels)} weights={len(weights)}"
        )

    num_classes = int(center_labels.max()) + 1
    class_weight_sum = np.bincount(center_labels, weights=weights, minlength=num_classes)
    class_cluster_count = np.bincount(center_labels, minlength=num_classes)
    class_weight_mean = np.divide(
        class_weight_sum,
        np.maximum(class_cluster_count, 1),
        out=np.zeros_like(class_weight_sum),
        where=class_cluster_count > 0,
    )

    x = np.arange(num_classes)
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, class_weight_sum, width=width, label="Weight Sum", color="#2ca02c")
    ax.bar(x + width / 2, class_weight_mean, width=width, label="Weight Mean", color="#d62728")
    ax.set_title("Distilled Weights by Class")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Weight")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(weights: np.ndarray) -> dict[str, float]:
    w = weights.astype(np.float64)
    w_sum = float(w.sum())
    w_safe = w / max(w_sum, 1e-12)
    entropy = float(-(w_safe * np.log(w_safe + 1e-12)).sum())
    effective_clusters = float(1.0 / np.maximum((w_safe**2).sum(), 1e-12))

    return {
        "num_weights": float(len(w)),
        "sum": w_sum,
        "min": float(w.min(initial=np.inf)),
        "max": float(w.max(initial=-np.inf)),
        "mean": float(w.mean()),
        "std": float(w.std()),
        "entropy": entropy,
        "effective_clusters": effective_clusters,
    }


def main() -> None:
    args = parse_args()

    distilled_data_path = Path(args.distilled_data).expanduser().resolve()
    if not distilled_data_path.exists():
        raise FileNotFoundError(f"distilled data not found: {distilled_data_path}")

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

    payload = load_payload(distilled_data_path)
    weights_tensor = payload["weights"]
    weights = weights_tensor.detach().cpu().float().view(-1).numpy()
    if weights.size == 0:
        raise ValueError("weights is empty")
    if not np.isfinite(weights).all():
        raise ValueError("weights contains NaN/Inf")

    sorted_path = output_dir / "weights_sorted.png"
    hist_path = output_dir / "weights_hist.png"
    summary_path = output_dir / "weights_summary.json"

    save_sorted_plot(weights, sorted_path, top_k=args.top_k)
    save_hist_plot(weights, hist_path, bins=args.hist_bins)

    result_paths: dict[str, str] = {
        "weights_sorted": str(sorted_path),
        "weights_hist": str(hist_path),
    }

    center_labels = maybe_load_center_labels(metadata_path)
    if center_labels is not None:
        by_class_path = output_dir / "weights_by_class.png"
        save_class_plot(weights, center_labels, by_class_path)
        result_paths["weights_by_class"] = str(by_class_path)
    else:
        result_paths["weights_by_class"] = "skipped (center_labels not found)"

    summary = build_summary(weights)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    result_paths["weights_summary"] = str(summary_path)

    print("[Done] Weight visualization generated:")
    for k, v in result_paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
