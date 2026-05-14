#!/usr/bin/env python3
"""
教师模型推理蒸馏图片 → 概率分布可视化。

用法示例:
    # 自动检测教师模型（从 baseline 目录）
    python visualize_teacher_probs.py \
        --distilled-data outputs/dermamnist_224_distill_ipc10_inverse/distilled_data.pt \
        --output-dir outputs/dermamnist_224_distill_ipc10_inverse/teacher_probs

    # 显式指定教师模型
    python visualize_teacher_probs.py \
        --distilled-data outputs/dermamnist_224_distill_ipc10_inverse/distilled_data.pt \
        --teacher-checkpoint outputs/dermamnist_224_distill_baseline/teacher_best.pt \
        --output-dir outputs/dermamnist_224_distill_ipc10_inverse/teacher_probs \
        --dataset dermamnist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

# —— 导入项目内部模块 ——
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dd_distill.baseline_resnet18 import (
    build_classifier,
    build_teacher_transforms,
    load_teacher_checkpoint,
    normalize_backbone_name,
    SUPPORTED_BACKBONES,
)
from dd_distill.datasets import get_dataset_spec
from dd_distill.utils import normalize_batch


# ═══════════════════════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="教师模型推理蒸馏图片，输出概率分布可视化"
    )
    parser.add_argument("--distilled-data", type=str, required=True,
                        help="distilled_data.pt 路径")
    parser.add_argument("--teacher-checkpoint", type=str, default="",
                        help="教师模型 checkpoint 路径（留空则自动从 baseline 目录查找）")
    parser.add_argument("--output-dir", type=str, default="",
                        help="输出目录（默认在 distilled_data 同级目录下创建 teacher_probs/）")
    parser.add_argument("--dataset", type=str, default="",
                        help="数据集名称（dermamnist/bloodmnist/pathmnist/aptos-2019-blindness-detection）")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=SUPPORTED_BACKBONES, help="教师模型 backbone")
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3,
                        help="每个样本显示 top-K 预测类别")
    parser.add_argument("--max-gallery-samples", type=int, default=20,
                        help="Gallery 图中最多展示的样本数")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_distilled_payload(path: Path) -> dict[str, Any]:
    """加载 distilled_data.pt，返回字典。"""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"distilled_data.pt 内容应为 dict，实际为 {type(payload)}")
    return payload


def load_distilled_images(payload: dict[str, Any], data_dir: Path) -> torch.Tensor:
    """从 payload 中提取图片张量 (N, 3, H, W)，值域 [0, 1] 或 [0, 255]。"""
    # 方式 1: 直接在 payload 中
    images = payload.get("images")
    if isinstance(images, torch.Tensor) and images.numel() > 0:
        return images.float()

    # 方式 2: 分片存储
    image_shards = payload.get("image_shards")
    if isinstance(image_shards, list) and image_shards:
        chunks = []
        for rel_shard in image_shards:
            shard_path = data_dir / str(rel_shard)
            shard = torch.load(shard_path, map_location="cpu")
            shard_imgs = shard.get("images")
            if not isinstance(shard_imgs, torch.Tensor):
                raise ValueError(f"shard {shard_path} 中缺少 'images' 字段")
            chunks.append(shard_imgs.float())
        return torch.cat(chunks, dim=0)

    # 方式 3: 从 PNG 文件加载
    rel_paths = payload.get("image_relative_paths")
    if isinstance(rel_paths, list) and rel_paths:
        from PIL import Image
        from torchvision.transforms import functional as TF

        chunks = []
        for rel_path in rel_paths:
            for prefix in ("distilled_images", ""):
                cand = data_dir / prefix / str(rel_path)
                if cand.exists():
                    break
            else:
                raise FileNotFoundError(f"找不到图片: {rel_path}")
            img = Image.open(cand).convert("RGB")
            t = TF.to_tensor(img)  # [0, 1], (3, H, W)
            chunks.append(t)
        return torch.stack(chunks, dim=0)

    raise ValueError("无法从 distilled_data.pt 中加载图片（无 images / image_shards / image_relative_paths）")


def resolve_teacher_checkpoint(args: argparse.Namespace, data_dir: Path) -> Path:
    """解析教师模型 checkpoint 路径。"""
    if args.teacher_checkpoint:
        cp = Path(args.teacher_checkpoint)
        if cp.exists():
            return cp
        raise FileNotFoundError(f"指定的 teacher checkpoint 不存在: {cp}")

    # 自动从 data_dir 的父目录推断 baseline 目录
    # 例如: outputs/dermamnist_224_distill_ipc10_inverse/ → outputs/dermamnist_224_distill_baseline/
    parent_name = data_dir.name  # e.g. "dermamnist_224_distill_ipc10_inverse"
    parts = parent_name.split("_distill_")
    if len(parts) >= 2:
        baseline_name = parts[0] + "_distill_baseline"
        baseline_dir = data_dir.parent / baseline_name
        candidate = baseline_dir / "teacher_best.pt"
        if candidate.exists():
            return candidate

    # 最后尝试在 data_dir 下查找
    for cand in sorted(data_dir.rglob("teacher_best.pt")):
        return cand

    raise FileNotFoundError(
        "无法自动找到 teacher checkpoint。请用 --teacher-checkpoint 显式指定。\n"
        f"搜索路径: {data_dir}"
    )


class DistilledImageDataset(Dataset):
    """只返回图片的简单 Dataset，用于教师模型批量推理。"""

    def __init__(self, images: torch.Tensor, transform: transforms.Compose):
        self.images = images.float()
        self.transform = transform

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(self.images[idx])


@torch.no_grad()
def teacher_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    eval_transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """教师模型对蒸馏图片做推理，返回 softmax 概率 (N, num_classes)。"""
    dataset = DistilledImageDataset(images, eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_probs: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = F.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def get_class_names(dataset_name: str) -> dict[int, str]:
    """获取数据集的类别名称映射。"""
    try:
        spec = get_dataset_spec(dataset_name)
        return spec.class_names()
    except Exception:
        pass
    return {}


# ═══════════════════════════════════════════════════════════════════
# 可视化函数
# ═══════════════════════════════════════════════════════════════════

def set_style():
    """设置统一的 matplotlib 样式。"""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
    })


def _get_color_map(n: int) -> list[str]:
    """为 n 个类别生成颜色。"""
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    return [cmap(i % cmap.N) for i in range(n)]


def plot_per_class_mean_probs(
    probs: np.ndarray,
    class_names: dict[int, str],
    num_classes: int,
    output_dir: Path,
):
    """图 1: 各类别在所有蒸馏样本上的平均预测概率（柱状图）。"""
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.9), 4.5))

    mean_probs = probs.mean(axis=0)  # (num_classes,)
    std_probs = probs.std(axis=0)

    colors = _get_color_map(num_classes)
    x = np.arange(num_classes)

    bars = ax.bar(x, mean_probs, yerr=std_probs, color=colors, edgecolor="white",
                  linewidth=0.8, capsize=3, error_kw={"linewidth": 1})

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, mean_probs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    labels = [class_names.get(i, f"Class {i}") for i in range(num_classes)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Average Predicted Probability")
    ax.set_title("Per-Class Mean Teacher Probability on Distilled Images")
    ax.set_ylim(0, min(1.0, mean_probs.max() + 0.15))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "per_class_mean_probs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: per_class_mean_probs.png")


def plot_prediction_heatmap(
    probs: np.ndarray,
    class_names: dict[int, str],
    num_classes: int,
    output_dir: Path,
):
    """图 2: 所有蒸馏样本 × 类别的预测概率热力图。"""
    n_samples = probs.shape[0]
    fig_height = max(5, n_samples * 0.22)
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 1.2), fig_height))

    im = ax.imshow(probs, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    # 标注每个格子的数值（样本太多则按需采样标注）
    annotate = n_samples <= 30
    for i in range(n_samples):
        for j in range(num_classes):
            val = probs[i, j]
            if annotate:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.5 else "black")
            elif val > 0.5:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color="white")

    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels(
        [class_names.get(i, f"C{i}") for i in range(num_classes)],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(np.arange(n_samples))
    ax.set_yticklabels([f"#{i}" for i in range(n_samples)], fontsize=6)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Distilled Sample Index")
    ax.set_title(f"Teacher Prediction Heatmap ({n_samples} distilled samples)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Probability")

    fig.tight_layout()
    fig.savefig(output_dir / "prediction_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: prediction_heatmap.png")


def plot_topk_per_sample(
    probs: np.ndarray,
    class_names: dict[int, str],
    num_classes: int,
    top_k: int,
    output_dir: Path,
):
    """图 3: 每个蒸馏样本的 top-K 预测概率（分组柱状图）。"""
    n_samples = probs.shape[0]
    n_cols = min(5, n_samples)
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.2),
        squeeze=False,
    )

    for idx in range(n_samples):
        ax = axes[idx // n_cols][idx % n_cols]
        sample_probs = probs[idx]

        # 取 top-K
        top_indices = np.argsort(sample_probs)[::-1][:top_k]
        top_vals = sample_probs[top_indices]

        colors = plt.cm.RdYlGn(top_vals)
        bars = ax.bar(range(top_k), top_vals, color=colors, edgecolor="gray", linewidth=0.5)

        for j, (bar, val) in enumerate(zip(bars, top_vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=6)

        ax.set_xticks(range(top_k))
        ax.set_xticklabels(
            [class_names.get(int(top_indices[j]), f"C{top_indices[j]}") for j in range(top_k)],
            rotation=30, ha="right", fontsize=6
        )
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Sample #{idx}", fontsize=8)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    # 隐藏多余的 subplot
    for idx in range(n_samples, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f"Top-{top_k} Teacher Predictions per Distilled Sample", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "topk_per_sample.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: topk_per_sample.png")


def plot_confidence_distribution(probs: np.ndarray, output_dir: Path):
    """图 4: 教师模型对蒸馏图片的置信度分布（每样本最大概率的直方图）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    max_probs = probs.max(axis=1)  # (N,)

    # 左: 直方图
    ax = axes[0]
    ax.hist(max_probs, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(x=max_probs.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {max_probs.mean():.3f}")
    ax.axvline(x=np.median(max_probs), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(max_probs):.3f}")
    ax.set_xlabel("Max Probability (Confidence)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Confidence Distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 右: 累积分布
    ax = axes[1]
    sorted_probs = np.sort(max_probs)
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax.plot(sorted_probs, cumulative, color="#55A868", linewidth=2)
    ax.fill_between(sorted_probs, 0, cumulative, alpha=0.15, color="#55A868")
    ax.set_xlabel("Max Probability")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Cumulative Distribution of Confidence")
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: confidence_distribution.png")


def plot_prediction_gallery(
    images: torch.Tensor,
    probs: np.ndarray,
    class_names: dict[int, str],
    num_classes: int,
    max_samples: int,
    output_dir: Path,
):
    """图 5: 蒸馏图片 + 对应预测概率条（Gallery 视图）。"""
    n_samples = min(images.shape[0], max_samples)
    # 采样：均匀选取
    if images.shape[0] > n_samples:
        indices = np.linspace(0, images.shape[0] - 1, n_samples, dtype=int)
    else:
        indices = np.arange(n_samples)

    # 将图片 clamp 到 [0, 1]
    imgs = images[indices].float()
    if imgs.max() > 1.0:
        imgs = imgs / 255.0
    imgs = imgs.clamp(0, 1)

    n_cols = min(5, n_samples)
    n_rows = int(np.ceil(n_samples / n_cols))

    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.0))
    gs = GridSpec(n_rows * 2, n_cols, figure=fig, hspace=0.35, wspace=0.3)

    colors = _get_color_map(num_classes)

    for plot_idx, data_idx in enumerate(indices):
        row = (plot_idx // n_cols) * 2
        col = plot_idx % n_cols

        # 上方: 图片
        ax_img = fig.add_subplot(gs[row, col])
        img_np = imgs[plot_idx].permute(1, 2, 0).numpy()
        ax_img.imshow(img_np)
        pred_label = int(probs[data_idx].argmax())
        ax_img.set_title(f"#{data_idx}  Pred: {class_names.get(pred_label, f'C{pred_label}')}",
                         fontsize=7, fontweight="bold")
        ax_img.axis("off")

        # 下方: 概率条
        ax_bar = fig.add_subplot(gs[row + 1, col])
        sample_probs = probs[data_idx]
        ax_bar.bar(range(num_classes), sample_probs, color=colors, edgecolor="white", linewidth=0.3)
        ax_bar.set_xticks(range(num_classes))
        ax_bar.set_xticklabels(range(num_classes), fontsize=5)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_ylabel("Prob", fontsize=6)
        ax_bar.tick_params(axis="y", labelsize=5)
        ax_bar.grid(axis="y", alpha=0.2, linestyle="--")

    fig.suptitle("Distilled Images & Teacher Prediction Probabilities", fontsize=13, y=1.01)
    fig.savefig(output_dir / "prediction_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: prediction_gallery.png")


def plot_summary_dashboard(
    probs: np.ndarray,
    images: torch.Tensor,
    class_names: dict[int, str],
    num_classes: int,
    output_dir: Path,
):
    """图 6: 综合仪表板 —— 一张图汇总关键信息。"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    colors = _get_color_map(num_classes)
    labels = [class_names.get(i, f"C{i}") for i in range(num_classes)]

    # (0, 0): 各类别平均概率
    ax0 = fig.add_subplot(gs[0, 0])
    mean_probs = probs.mean(axis=0)
    ax0.bar(range(num_classes), mean_probs, color=colors, edgecolor="white", linewidth=0.8)
    for i, v in enumerate(mean_probs):
        ax0.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax0.set_xticks(range(num_classes))
    ax0.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax0.set_ylim(0, min(1.0, mean_probs.max() + 0.15))
    ax0.set_title("A. Mean Prob per Class", fontsize=10, fontweight="bold")
    ax0.grid(axis="y", alpha=0.3, linestyle="--")

    # (0, 1): 置信度分布
    ax1 = fig.add_subplot(gs[0, 1])
    max_probs = probs.max(axis=1)
    ax1.hist(max_probs, bins=15, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax1.axvline(max_probs.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"μ={max_probs.mean():.3f}")
    ax1.axvline(np.median(max_probs), color="orange", linestyle="--", linewidth=1.5,
                label=f"med={np.median(max_probs):.3f}")
    ax1.set_xlabel("Max Probability")
    ax1.set_ylabel("Count")
    ax1.set_title("B. Confidence Distribution", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7)

    # (0, 2): 预测类别分布（argmax）
    ax2 = fig.add_subplot(gs[0, 2])
    pred_labels = probs.argmax(axis=1)
    class_counts = np.bincount(pred_labels, minlength=num_classes)
    ax2.bar(range(num_classes), class_counts, color=colors, edgecolor="white", linewidth=0.8)
    for i, c in enumerate(class_counts):
        ax2.text(i, c + 0.3, str(c), ha="center", fontsize=8)
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax2.set_title("C. Predicted Class Distribution (argmax)", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # (1, 0:2): 热力图（缩小版）
    ax3 = fig.add_subplot(gs[1, :2])
    im = ax3.imshow(probs, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax3.set_xticks(range(num_classes))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax3.set_yticks(range(probs.shape[0]))
    ax3.set_yticklabels([f"#{i}" for i in range(probs.shape[0])], fontsize=5)
    ax3.set_title("D. Prediction Heatmap (rows=samples, cols=classes)", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax3, shrink=0.9)
    cbar.set_label("Prob")

    # (1, 2): 样本缩略图网格
    ax4 = fig.add_subplot(gs[1, 2])
    imgs_disp = images[:min(16, images.shape[0])].float()
    if imgs_disp.max() > 1.0:
        imgs_disp = imgs_disp / 255.0
    imgs_disp = imgs_disp.clamp(0, 1)
    grid = make_grid(imgs_disp, nrow=4, padding=2, normalize=False)
    ax4.imshow(grid.permute(1, 2, 0).numpy())
    ax4.set_title("E. Distilled Image Samples", fontsize=10, fontweight="bold")
    ax4.axis("off")

    fig.suptitle(
        f"Teacher Inference on Distilled Images — Summary Dashboard\n"
        f"({probs.shape[0]} samples, {num_classes} classes)",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.savefig(output_dir / "summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    set_style()

    device = resolve_device(args.device)
    print(f"📌 设备: {device}")

    # —— 1. 加载蒸馏数据 ——
    distilled_path = Path(args.distilled_data).resolve()
    if not distilled_path.exists():
        raise FileNotFoundError(f"distilled_data.pt 不存在: {distilled_path}")
    data_dir = distilled_path.parent

    payload = load_distilled_payload(distilled_path)
    images = load_distilled_images(payload, data_dir)
    num_classes = payload["soft_labels"].shape[1]  # 从 soft_labels 推断类别数
    dataset_name = args.dataset or payload.get("dataset", "unknown")
    print(f"📦 蒸馏数据: {images.shape[0]} 张图片, {num_classes} 个类别, 数据集={dataset_name}")

    # 确保图片在 [0, 1]
    if images.max() > 1.0:
        images = images / 255.0
    images = images.clamp(0, 1)

    # —— 2. 加载教师模型 ——
    teacher_path = resolve_teacher_checkpoint(args, data_dir)
    print(f"🧠 教师模型: {teacher_path}")
    num_classes_actual, backbone = num_classes, args.backbone
    teacher, _ = load_teacher_checkpoint(
        teacher_path, num_classes=num_classes_actual,
        backbone=backbone, imagenet_pretrained=args.imagenet_pretrained,
        device=device,
    )
    teacher.eval()
    print(f"    Backbone: {backbone}, 参数量: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M")

    # —— 3. 构建 eval transform ——
    _, eval_transform = build_teacher_transforms(224, backbone)

    # —— 4. 教师模型推理 ——
    print(f"🔍 教师模型正在推理 {images.shape[0]} 张蒸馏图片...")
    probs = teacher_inference(teacher, images, eval_transform, device, args.batch_size)
    print(f"   概率矩阵 shape: {probs.shape}")

    # —— 5. 获取类别名称 ——
    class_names = get_class_names(dataset_name)
    if not class_names:
        class_names = {i: f"Class_{i}" for i in range(num_classes)}
    print(f"🏷️  类别: {class_names}")

    # —— 6. 创建输出目录 ——
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "teacher_probs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")

    # —— 7. 保存概率矩阵 ——
    np.save(output_dir / "teacher_probs.npy", probs)
    # 保存 CSV
    header = ",".join([f"Class_{i}" for i in range(num_classes)])
    np.savetxt(output_dir / "teacher_probs.csv", probs, delimiter=",", header=header, comments="")
    print(f"  ✓ 保存: teacher_probs.npy / teacher_probs.csv")

    # —— 8. 生成可视化 ——
    print("🎨 生成可视化图表...")
    plot_per_class_mean_probs(probs, class_names, num_classes, output_dir)
    plot_prediction_heatmap(probs, class_names, num_classes, output_dir)
    plot_topk_per_sample(probs, class_names, num_classes, args.top_k, output_dir)
    plot_confidence_distribution(probs, output_dir)
    plot_prediction_gallery(images, probs, class_names, num_classes, args.max_gallery_samples, output_dir)
    plot_summary_dashboard(probs, images, class_names, num_classes, output_dir)

    # —— 9. 打印统计摘要 ——
    print("\n" + "=" * 55)
    print("📊 统计摘要")
    print("=" * 55)
    pred_labels = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    print(f"  平均置信度 (max prob):  {max_probs.mean():.4f} ± {max_probs.std():.4f}")
    print(f"  置信度中位数:            {np.median(max_probs):.4f}")
    print(f"  低置信度样本 (<0.4):     {(max_probs < 0.4).sum()} / {len(max_probs)}")
    print(f"  高置信度样本 (>0.9):     {(max_probs > 0.9).sum()} / {len(max_probs)}")

    # 各类别统计
    print(f"\n  各类别平均概率:")
    mean_probs = probs.mean(axis=0)
    for i in range(num_classes):
        label = class_names.get(i, f"Class {i}")
        print(f"    {label:40s}  {mean_probs[i]:.4f}")

    print(f"\n  argmax 预测分布:")
    class_counts = np.bincount(pred_labels, minlength=num_classes)
    for i in range(num_classes):
        label = class_names.get(i, f"Class {i}")
        print(f"    {label:40s}  {class_counts[i]:4d} 个样本")

    print(f"\n✨ 所有图表已保存到: {output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
