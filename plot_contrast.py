#!/usr/bin/env python3
"""绘制四种方法的 test_acc 和 test_loss 每 epoch 对比折线图。

方法：
  D4M:  no guidance, uniform weights, soft label, classwise cluster
  MGD³: guidance on,  uniform weights, soft label, classwise cluster
  DDOQ: no guidance, heuristic weights, soft label, classwise cluster
  Ours: no guidance, uniform weights, soft label, global cluster
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_STYLES = {
    "D4M": {"color": "#e41a1c", "marker": "s", "linestyle": "-", "label": "D⁴M"},
    "MGD³": {"color": "#377eb8", "marker": "o", "linestyle": "--", "label": "MGD³"},
    "DDOQ": {"color": "#4daf4a", "marker": "^", "linestyle": "-.", "label": "DDOQ"},
    "Ours": {"color": "#984ea3", "marker": "D", "linestyle": "-", "label": "Ours"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-epoch test_acc and test_loss for four contrast methods."
    )
    parser.add_argument("--dataset", type=str, default="dermamnist")
    parser.add_argument("--ipc", type=int, default=100)
    parser.add_argument("--d4m-json", type=str, required=True)
    parser.add_argument("--mgd3-json", type=str, required=True)
    parser.add_argument("--ddoq-json", type=str, required=True)
    parser.add_argument("--ours-json", type=str, required=True)
    parser.add_argument("--output-acc", type=str, default="contrast_test_acc.png")
    parser.add_argument("--output-loss", type=str, default="contrast_test_loss.png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 8])
    return parser.parse_args()


def load_history(json_path: str) -> dict[int, dict[str, float]]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    with open(path, "r") as f:
        rows = json.load(f)
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        out[epoch] = {k: float(v) for k, v in row.items()}
    return out


def plot_metric(
    data: dict[str, dict[int, dict[str, float]]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: str,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for method, history in data.items():
        style = METHOD_STYLES[method]
        epochs = sorted(history.keys())
        values = [history[e][metric] for e in epochs]
        ax.plot(
            epochs, values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
            markersize=6,
            linewidth=1.8,
            markevery=max(1, len(epochs) // 10),
        )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


def main() -> None:
    args = parse_args()

    data = {
        "D4M": load_history(args.d4m_json),
        "MGD³": load_history(args.mgd3_json),
        "DDOQ": load_history(args.ddoq_json),
        "Ours": load_history(args.ours_json),
    }

    n_epochs = max(len(h) for h in data.values())
    title_base = f"{args.dataset}  (IPC={args.ipc})"

    # ── test_acc ──
    plot_metric(
        data=data,
        metric="test_acc",
        ylabel="Accuracy",
        title=f"Test Accuracy — {title_base}",
        output_path=args.output_acc,
        dpi=args.dpi,
        figsize=args.figsize,
    )

    # ── test_loss ──
    plot_metric(
        data=data,
        metric="test_loss",
        ylabel="Loss",
        title=f"Test Loss — {title_base}",
        output_path=args.output_loss,
        dpi=args.dpi,
        figsize=args.figsize,
    )


if __name__ == "__main__":
    main()
