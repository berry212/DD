#!/usr/bin/env python3
"""将数据集通过 VAE encoder 转换到隐空间，再用 t-SNE 降维可视化。

不同类别的数据点用不同颜色标注，输出高清图片。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 必须在 import sklearn 之前设置，防止 PCA/KMeans 的 OpenBLAS/OpenMP 线程与 PyTorch 冲突
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

# — 父目录加入 sys.path，以便直接运行本脚本时也能 import dd_distill —
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from dd_distill.distillate import load_vae, build_encode_loader
from dd_distill.datasets import get_dataset_spec
from dd_distill.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAE latents → t-SNE 可视化（按类别着色）"
    )
    parser.add_argument("--dataset", type=str, default="dermamnist")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output", type=str, default="latent_tsne.png")
    parser.add_argument("--vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="降维最大样本数（t-SNE O(N²) 复杂度），0=不限制")
    parser.add_argument("--per-class-samples", type=int, default=0,
                        help="每类别抽样数（0=不使用，>0时按类别分层抽样，优先于--max-samples）")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iter", type=int, default=2000)
    parser.add_argument("--early-exagg", type=float, default=12.0,
                        help="t-SNE early exaggeration，高值拉大类间距")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="t-SNE learning rate，默认 auto（设为 N/12 附近通常更好）")
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=["cosine", "euclidean", "manhattan"],
                        help="t-SNE 距离度量。高维空间建议 cosine 或 euclidean")
    parser.add_argument("--target-class", type=int, default=-1,
                        help="只降维可视化指定类别（默认-1=全部类别，>=0时只显示该类）")
    parser.add_argument("--point-size", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 10])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    amp_enabled = args.fp16 and device.type == "cuda"
    vae_dtype = torch.float16 if amp_enabled else torch.float32

    # ── 1. 加载数据集 ──
    dataset_spec = get_dataset_spec(args.dataset)
    split_bundle = dataset_spec.load_dataset_splits(
        data_root=args.data_root, image_size=args.image_size
    )
    train_set = split_bundle.train_set
    num_classes = split_bundle.num_classes
    class_name_map = split_bundle.class_names
    class_names = [class_name_map.get(i, f"class_{i}") for i in range(num_classes)]

    print(f"[Dataset] name={dataset_spec.name} samples={len(train_set)} classes={num_classes}")

    # ── 2. DataLoader ──
    encode_loader = build_encode_loader(
        train_set=train_set,
        image_size=args.image_size,
        encode_batch_size=args.encode_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # ── 3. 编码到隐空间 ──
    vae, scaling_factor = load_vae(args.vae_model_id, device=device, dtype=vae_dtype)

    all_latents: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    total_batches = len(encode_loader)
    print(f"[Encoding] total batches: {total_batches}")
    for batch_idx, (images, labels) in enumerate(encode_loader, start=1):
        images = images.to(device=device, dtype=vae_dtype)
        images = images * 2.0 - 1.0

        with torch.no_grad():
            posterior = vae.encode(images).latent_dist
            latents = posterior.mean * scaling_factor

        latents_flat = latents.float().cpu().view(images.size(0), -1).numpy()
        all_latents.append(latents_flat)
        all_labels.append(labels.cpu().numpy())

        if batch_idx % 20 == 0 or batch_idx == total_batches:
            print(f"[Encoding] batch {batch_idx}/{total_batches}")

    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X = np.concatenate(all_latents, axis=0)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)

    latent_dim = X.shape[1]
    print(f"[Latents] shape={X.shape} dim={latent_dim}")
    print(f"[Stats] mean={X.mean():.4f} std={X.std():.4f} min={X.min():.4f} max={X.max():.4f}")

    # ── 4. 按类别过滤 ──
    if args.target_class >= 0:
        mask = y == args.target_class
        X = X[mask]
        y = y[mask]
        target_class_name = class_names[args.target_class]
        print(f"[Filter] keeping only class {args.target_class} ({target_class_name}), "
              f"samples={X.shape[0]}")
    else:
        print(f"[Filter] keeping all {num_classes} classes")

    # ── 5. 分层抽样 ──
    rng = np.random.default_rng(args.seed)
    if args.per_class_samples > 0:
        # 按类别分层抽样
        k = args.per_class_samples
        selected_indices: list[int] = []
        for class_id in range(num_classes):
            class_idx = np.where(y == class_id)[0]
            available = int(class_idx.shape[0])
            take = min(k, available)
            if take == 0:
                continue
            chosen = rng.choice(class_idx, size=take, replace=False)
            selected_indices.extend(chosen.tolist())
            print(f"[Stratified] class={class_id} available={available} sampled={take}")
        selected = np.array(selected_indices, dtype=np.int64)
        X = X[selected]
        y = y[selected]
        print(f"[Stratified] total={X.shape[0]}")
    elif args.max_samples > 0 and X.shape[0] > args.max_samples:
        selected = rng.choice(X.shape[0], size=args.max_samples, replace=False)
        X = X[selected]
        y = y[selected]
        print(f"[Subsample] random down to {X.shape[0]} samples")
    
    # ── 5.5 PCA 预降维 ──
    # 先降到 50-100 维（randomized SVD 避免线程冲突）
    pca_dim = min(100, X.shape[1], X.shape[0] - 1)
    print(f"[PCA] reducing from {X.shape[1]} to {pca_dim} dims...", flush=True)
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=args.seed)
    X = pca.fit_transform(X)
    print(f"[PCA] done. explained={pca.explained_variance_ratio_.sum():.4f}", flush=True)

    # ── 6. t-SNE 降维 ──
    lr = args.learning_rate if args.learning_rate is not None else max(X.shape[0] / 12., 200.)
    print(f"[t-SNE] metric={args.metric} perplexity={args.perplexity} "
          f"max_iter={args.tsne_iter} lr={lr:.0f} early_exagg={args.early_exagg} ...")
    tsne = TSNE(
        n_components=2,
        metric=args.metric,
        perplexity=min(args.perplexity, X.shape[0] - 1),
        max_iter=args.tsne_iter,
        learning_rate=lr,
        early_exaggeration=args.early_exagg,
        random_state=args.seed,
        verbose=1,
    )
    X_2d = tsne.fit_transform(X)
    print(f"[t-SNE] done. shape={X_2d.shape}")

    # ── 7. 可视化 ──
    n_colors = num_classes if args.target_class < 0 else 1
    cmap = plt.get_cmap("tab10", max(num_classes, 1))

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    if args.target_class >= 0:
        # 单类别：用单色绘制
        ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=[cmap(args.target_class % 10)],
            label=target_class_name,
            s=args.point_size,
            alpha=args.alpha,
            edgecolors="none",
        )
    else:
        for class_id in range(num_classes):
            mask = y == class_id
            if not mask.any():
                continue
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[cmap(class_id)],
                label=class_names[class_id],
                s=args.point_size,
                alpha=args.alpha,
                edgecolors="none",
            )

    title_class = f"class {args.target_class} ({target_class_name})" if args.target_class >= 0 else "all classes"
    ax.set_title(
        f"{dataset_spec.name} — VAE Latent Space t-SNE ({title_class})\n"
        f"(n={X.shape[0]}, perplexity={args.perplexity}, "
        f"latent_dim={latent_dim}, metric={args.metric})",
        fontsize=14,
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        markerscale=1.5,
        frameon=True,
    )
    fig.tight_layout()

    output_path = Path(args.output)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] saved → {output_path}")


if __name__ == "__main__":
    main()
