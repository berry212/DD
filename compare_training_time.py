#!/usr/bin/env python3
"""
全量数据集 Baseline vs 蒸馏数据集 (Ours/CLVQ) 训练时间对比。

在 dermamnist 上训练 ResNet-18，测量：
  1. 全量 baseline：7007 张训练图片
  2. 蒸馏数据 (Ours)：IPC=10/50/100/200
  3. 对比指标：训练时间、显存峰值、test accuracy、加速比

用法:
    python compare_training_time.py                        # 完整对比
    python compare_training_time.py --epochs 5              # 快速测试
    python compare_training_time.py --use-cached-baseline   # 跳过 baseline 训练
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# ── 项目路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dd_distill.baseline_resnet18 import (
    build_classifier as build_teacher_classifier,
    build_teacher_transforms,
    evaluate_classifier,
    load_teacher_checkpoint,
)
from dd_distill.datasets import TorchDataset, get_dataset_spec
from dd_distill.train_student import (
    load_distilled_triplet,
    load_fkd_batch_payload,
    run_training,
)

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

# ═══════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline vs 蒸馏数据训练时间对比")
    p.add_argument("--dataset", type=str, default="dermamnist")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="outputs/training_time_comparison")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--student-batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ipcs", type=int, nargs="+", default=[10, 50, 100, 200])
    p.add_argument("--use-cached-baseline", action="store_true",
                   help="使用已有 baseline 模型，跳过全量训练")
    p.add_argument("--baseline-model", type=str,
                   default="outputs/dermamnist_224_distill_baseline/teacher_best.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════

def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
    return 0.0


def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ═══════════════════════════════════════════════════
# 全量数据 Baseline 训练
# ═══════════════════════════════════════════════════

def train_baseline(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("🔵 全量数据 Baseline 训练")
    print("=" * 60)

    spec = get_dataset_spec(args.dataset)
    splits = spec.load_dataset_splits(args.data_root, 224)
    num_classes = splits.num_classes
    print(f"   训练图片: {len(splits.train_set)}, 验证: {len(splits.val_set)}, 测试: {len(splits.test_set)}")

    train_transform, eval_transform = build_teacher_transforms(224, "resnet18")

    train_loader = DataLoader(
        TorchDataset(splits.train_set, transform=train_transform),
        batch_size=args.student_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        TorchDataset(splits.val_set, transform=eval_transform),
        batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        TorchDataset(splits.test_set, transform=eval_transform),
        batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    model = build_teacher_classifier(num_classes, "resnet18", imagenet_pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = device.type == "cuda"
    # amp_enabled = False

    reset_memory_stats()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        train_acc = correct / total
        val_loss, val_acc = evaluate_classifier(model, val_loader, device, amp_enabled)
        test_loss, test_acc = evaluate_classifier(model, test_loader, device, amp_enabled)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"    epoch {epoch:3d}/{args.epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    peak_memory = get_gpu_memory_mb()

    test_loss, test_acc = evaluate_classifier(model, test_loader, device, amp_enabled)

    result = {
        "mode": "baseline",
        "num_train": len(splits.train_set), "num_epochs": args.epochs,
        "total_time_s": round(total_time, 2),
        "peak_gpu_memory_mb": round(peak_memory, 1),
        "test_acc": round(test_acc, 6), "test_loss": round(test_loss, 6),
        "batch_size": args.student_batch_size,
    }
    print(f"\n   ✅ 完成: {total_time:.1f}s  test_acc={test_acc:.4f}  mem={peak_memory:.0f}MB")
    return result


# ═══════════════════════════════════════════════════
# 蒸馏数据训练 (Ours / CLVQ, 软标签 KD)
# ═══════════════════════════════════════════════════

def train_on_distilled(
    args: argparse.Namespace, device: torch.device,
    distilled_data_path: Path, ipc: int,
) -> dict[str, Any]:
    print(f"\n  {'─' * 50}")
    print(f"  🟠 蒸馏数据训练  IPC={ipc}  (Ours / CLVQ, 软标签 KD)")
    print(f"  {'─' * 50}")

    distilled_data_path = distilled_data_path.resolve()
    bundle = load_distilled_triplet(distilled_data_path)
    num_distilled = bundle.num_samples

    # 构造与 train_student.sh 一致的参数
    student_args = argparse.Namespace(
        dataset=args.dataset,
        data_root=args.data_root,
        distilled_data=str(distilled_data_path),
        output_dir=f"{args.output_dir}/student_ipc{ipc}",
        student_backbone="resnet18",
        train_epochs=args.epochs,
        train_batch_size=args.student_batch_size,
        eval_batch_size=args.eval_batch_size,
        train_lr=4e-4,
        weight_decay=1e-4,
        kd_temperature=0,            # 0 = 自动从蒸馏数据解析 teacher_temperature（为 20.0）
        weight_balance_alpha=0.0,
        soft_label_sharpen=1.0,
        hard_label_alpha=0.0,
        train_crop_min_scale=0.08,
        train_crop_max_scale=1.0,
        train_horizontal_flip_prob=0.5,
        use_fkd_batches=True,        # 使用 FKD 预计算批次
        image_size=224,
        imagenet_pretrained=True,
        amp=True,
        num_workers=args.num_workers,
        pin_memory=False,
        device="auto",
        seed=args.seed,
    )

    # 预加载 FKD cache 使其进入 page cache，不计入训练时间
    if bundle.fkd_batch_path:
        _ = load_fkd_batch_payload(distilled_data_path, bundle.fkd_batch_path)
    if device.type == "cuda":
        torch.cuda.synchronize()

    reset_memory_stats()
    t_start = time.perf_counter()
    summary = run_training(student_args)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    peak_memory = get_gpu_memory_mb()

    result = {
        "mode": "Ours", "ipc": ipc,
        "num_distilled": num_distilled, "num_epochs": args.epochs,
        "total_time_s": round(total_time, 2),
        "peak_gpu_memory_mb": round(peak_memory, 1),
        "test_acc": round(summary["test_acc_at_best_val"], 6),
        "test_loss": round(summary["test_loss_at_best_val"], 6),
        "batch_size": summary.get("train_batch_size", args.student_batch_size),
    }
    print(f"     ✅ IPC={ipc}: {total_time:.1f}s  test_acc={result['test_acc']:.4f}  mem={peak_memory:.0f}MB")
    return result


# ═══════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════


def plot_comparison(baseline: dict, distilled: list[dict], args, output_dir: Path):
    """三栏图：训练时间 + Accuracy + 时间-准确率散点图。"""
    distilled.sort(key=lambda r: r["num_distilled"])

    ipcs = [r["ipc"] for r in distilled]
    n_samples = [r["num_distilled"] for r in distilled]
    times = [r["total_time_s"] for r in distilled]
    accs = [r["test_acc"] for r in distilled]

    base_time = baseline["total_time_s"]
    base_acc = baseline["test_acc"]
    base_n = baseline["num_train"]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(distilled)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # A: 训练时间
    ax = axes[0]
    x = np.arange(len(distilled))
    bars = ax.bar(x, times, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=base_time, color="#E74C3C", linestyle="--", linewidth=2.5,
               label=f"Baseline ({base_n} imgs) = {base_time:.0f}s")
    for bar, t in zip(bars, times):
        speedup = base_time / t if t > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + base_time * 0.03,
                f"×{speedup:.1f}", ha="center", fontsize=10, fontweight="bold")
    labels = [f"IPC={ipc}\n(n={n})" for ipc, n in zip(ipcs, n_samples)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total Training Time (s)")
    ax.set_title("A. 训练时间对比")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # B: Accuracy
    ax = axes[1]
    bars = ax.bar(x, accs, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=base_acc, color="#E74C3C", linestyle="--", linewidth=2.5,
               label=f"Baseline = {base_acc:.4f}")
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{a:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("B. 准确率对比")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # C: 时间-准确率散点图
    ax = axes[2]
    ax.scatter([base_time], [base_acc], c="#E74C3C", s=250, marker="s",
               edgecolors="white", linewidth=2, zorder=10,
               label=f"Baseline ({base_n} imgs)")
    for i in range(len(distilled)):
        ax.scatter([times[i]], [accs[i]], c=[colors[i]], s=150, edgecolors="white",
                   linewidth=1.5, zorder=5)
        ax.annotate(f"IPC={ipcs[i]}", (times[i], accs[i]),
                    textcoords="offset points", xytext=(10, -12), fontsize=9, color=colors[i])
    ax.set_xlabel("Total Training Time (s)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("C. 时间-准确率散点图")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"DermaMNIST: Baseline vs 蒸馏数据 (Ours/CLVQ) 训练时间 & 准确率\n"
        f"ResNet-18, {baseline['num_epochs']} epochs | "
        f"Baseline={baseline['num_train']} imgs, {base_time:.0f}s | "
        f"蒸馏={args.epochs} epochs, ⌀{np.mean(times):.0f}s",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "training_time_comparison.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ 保存: training_time_comparison.png")

    # 表格图
    fig2, ax2 = plt.subplots(figsize=(14, 2 + len(distilled) * 0.55))
    ax2.axis("off")

    header = ["Experiment", "Images", "Time (s)", "Speedup", "Test Acc", "GPU Mem (MB)"]
    rows = [header]
    rows.append([
        f"Baseline (全量数据)", str(base_n), f"{base_time:.1f}", "1.0×",
        f"{base_acc:.4f}", f"{baseline['peak_gpu_memory_mb']:.0f}",
    ])
    for r in distilled:
        sp = base_time / r["total_time_s"] if r["total_time_s"] > 0 else float("inf")
        rows.append([
            f"Ours (蒸馏) IPC={r['ipc']}", str(r["num_distilled"]),
            f"{r['total_time_s']:.1f}", f"×{sp:.1f}",
            f"{r['test_acc']:.4f}", f"{r['peak_gpu_memory_mb']:.0f}",
        ])

    table = ax2.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for j in range(len(header)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(len(header)):
        table[1, j].set_facecolor("#FADBD8")
    for i in range(len(distilled)):
        c = colors[i]
        for j in range(len(header)):
            table[i + 2, j].set_facecolor(c)
            table[i + 2, j].set_alpha(0.2)

    ax2.set_title(
        f"DermaMNIST — Baseline vs 蒸馏数据训练时间对比  ({baseline['num_epochs']} epochs, ResNet-18)",
        fontsize=13, fontweight="bold",
    )
    fig2.tight_layout()
    fig2.savefig(output_dir / "training_time_table.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ 保存: training_time_table.png")


# ═══════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    print(f"📌 设备: {device}  |  数据集: {args.dataset}  |  Epochs: {args.epochs}")
    print(f"    Batch size: {args.student_batch_size}  (Baseline & Student 统一)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    # ── 1. Baseline ──
    if args.use_cached_baseline:
        print("\n🔵 使用已有 baseline 模型 (跳过全量训练)")
        baseline_path = Path(args.baseline_model)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline 模型不存在: {baseline_path}")

        spec = get_dataset_spec(args.dataset)
        splits = spec.load_dataset_splits(args.data_root, 224)
        model, _ = load_teacher_checkpoint(baseline_path, splits.num_classes, "resnet18", True, device)
        _, eval_transform = build_teacher_transforms(224, "resnet18")
        test_loader = DataLoader(
            TorchDataset(splits.test_set, transform=eval_transform),
            batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=False,
        )
        test_loss, test_acc = evaluate_classifier(model, test_loader, device, device.type == "cuda")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        baseline_result = {
            "mode": "baseline",
            "num_train": len(splits.train_set), "num_epochs": args.epochs,
            "total_time_s": 0, "peak_gpu_memory_mb": 0,
            "test_acc": round(test_acc, 6), "test_loss": round(test_loss, 6),
            "batch_size": args.student_batch_size,
        }
        print(f"    使用缓存 baseline: test_acc={test_acc:.4f}")
    else:
        baseline_result = train_baseline(args, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_results.append(baseline_result)

    # ── 2. 蒸馏数据 (Ours = CLVQ) ──
    # 尝试新旧两种命名格式
    for ipc in args.ipcs:
        distilled_path = None
        for fmt in (
            f"outputs/{args.dataset}_224_distill_ipc{ipc}_method-clvq-softlabel/distilled_data.pt",
            f"outputs/{args.dataset}_224_distill_ipc{ipc}_method-clvq/distilled_data.pt",
            f"outputs/{args.dataset}_224_distill_ipc{ipc}/distilled_data.pt",
        ):
            if Path(fmt).exists():
                distilled_path = Path(fmt)
                break

        if distilled_path is None:
            print(f"  ⚠️  蒸馏数据不存在，跳过 IPC={ipc}")
            continue

        result = train_on_distilled(args, device, distilled_path, ipc)
        all_results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── 3. 保存 ──
    with open(output_dir / "training_time_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── 4. 打印汇总 ──
    baseline = all_results[0]
    distilled = all_results[1:]
    base_time = baseline["total_time_s"] if baseline["total_time_s"] > 0 else 1.0

    print("\n" + "=" * 78)
    print("📊 训练时间对比汇总")
    print("=" * 78)
    print(f"{'Experiment':<28} {'Images':>8} {'Time(s)':>10} {'Speedup':>10} "
          f"{'Acc':>10} {'Mem(MB)':>10}")
    print("-" * 78)
    for r in all_results:
        if r["mode"] == "baseline":
            exp = f"Baseline (全量 {r['num_train']})"
        else:
            exp = f"Ours (蒸馏) IPC={r['ipc']}"
        sp = base_time / r["total_time_s"] if r["total_time_s"] > 0 else float("inf")
        n = r.get("num_distilled", r.get("num_train", 0))
        print(f"{exp:<28} {n:>8} {r['total_time_s']:>10.1f} "
              f"{'×' + str(round(sp, 1)):>10} {r['test_acc']:>10.4f} "
              f"{r['peak_gpu_memory_mb']:>10.1f}")

    # ── 5. 可视化 ──
    if len(distilled) >= 1:
        print("\n🎨 生成可视化图表...")
        plot_comparison(baseline, distilled, args, output_dir)

    print(f"\n✨ 结果保存到: {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
