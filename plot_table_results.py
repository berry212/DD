"""绘制三个数据集在不同 IPC 下的分类准确率折线图"""
import matplotlib.pyplot as plt
import numpy as np

# ── 数据 ──
datasets = ["DermaMNIST", "BloodMNIST", "APTOS-2019"]
ipcs = [10, 50, 100, 200]

accs = {
    "DermaMNIST":  [0.7162, 0.7796, 0.8115, 0.8529],
    "BloodMNIST":  [0.9582, 0.9822, 0.9851, 0.9871],
    "APTOS-2019":  [0.7350, 0.7814, 0.8142, 0.8224],
}

full_accs = {
    "DermaMNIST": 0.8848,
    "BloodMNIST": 0.9901,
    "APTOS-2019": 0.8396,
}

# ── 颜色和标记 ──
colors = {
    "DermaMNIST": "#E74C3C",
    "BloodMNIST": "#3498DB",
    "APTOS-2019": "#2ECC71",
}
markers = {
    "DermaMNIST": "o",
    "BloodMNIST": "s",
    "APTOS-2019": "D",
}

# ── 绘图 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, dataset in zip(axes, datasets):
    # 折线
    ax.plot(ipcs, accs[dataset], marker=markers[dataset], color=colors[dataset],
            linewidth=2, markersize=8, label=f"{dataset} (Distillation)")
    # 全量横线
    ax.axhline(y=full_accs[dataset], color=colors[dataset], linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"{dataset} (Full)")

    ax.set_xlabel("IPC (Image Number Per Class)", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_title(dataset, fontsize=16)
    ax.set_xticks(ipcs)
    ax.set_ylim(0.50, 1.10)
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("outputs/ipc_acc_comparison.png", dpi=150, bbox_inches="tight")
print("已保存: outputs/ipc_acc_comparison.png")
plt.show()
