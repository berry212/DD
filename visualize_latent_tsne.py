#!/usr/bin/env python3
"""将数据集通过 VAE encoder 转换到隐空间，再用 t-SNE 降维可视化。

不同类别的数据点用不同颜色标注，输出高清图片。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
                        help="降维最大样本数（t-SNE O(N²) 复杂度）")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iter", type=int, default=2000)
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

    # ── 4. 下采样（t-SNE O(N²) 太贵）──
    if X.shape[0] > args.max_samples:
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(X.shape[0], size=args.max_samples, replace=False)
        X = X[selected]
        y = y[selected]
        print(f"[Subsample] down to {X.shape[0]} samples")

    # ── 5. t-SNE 降维 ──
    print(f"[t-SNE] perplexity={args.perplexity} max_iter={args.tsne_iter} ...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, X.shape[0] - 1),
        max_iter=args.tsne_iter,
        random_state=args.seed,
        verbose=1,
    )
    X_2d = tsne.fit_transform(X)
    print(f"[t-SNE] done. shape={X_2d.shape}")

    # ── 6. 可视化 ──
    n_colors = num_classes
    cmap = plt.cm.get_cmap("tab10", n_colors)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
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

    ax.set_title(
        f"{dataset_spec.name} — VAE Latent Space t-SNE\n"
        f"(n={X.shape[0]}, perplexity={args.perplexity}, latent_dim={latent_dim})",
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
