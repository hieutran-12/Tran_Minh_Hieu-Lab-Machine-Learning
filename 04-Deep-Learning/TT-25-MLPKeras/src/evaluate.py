"""
evaluate.py
-----------
Các hàm đánh giá & vẽ biểu đồ dùng chung cho mọi model (LightGBM, MLP các loại).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc_score(y_true, y_proba) -> float:
    return float(average_precision_score(y_true, y_proba))


def precision_at_k(y_true, y_proba, k: int = 3000) -> float:
    """Precision khi chỉ gọi điện cho k khách được xếp hạng cao nhất (đội telesales gọi k cuộc/ngày)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    k = min(k, len(y_proba))
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


def threshold_for_top_k(y_proba, k: int = 3000) -> float:
    """Ngưỡng xác suất tương ứng với việc chọn đúng top-k khách hàng."""
    y_proba = np.asarray(y_proba)
    k = min(k, len(y_proba))
    sorted_proba = np.sort(y_proba)[::-1]
    return float(sorted_proba[k - 1])


def plot_learning_curves(history, out_path: str, title: str = "Learning curves"):
    """Vẽ train/val loss và train/val PR-AUC theo epoch."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(hist["loss"], label="train_loss")
    axes[0].plot(hist["val_loss"], label="val_loss")
    axes[0].set_title("Loss theo epoch")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(hist["pr_auc"], label="train_pr_auc")
    axes[1].plot(hist["val_pr_auc"], label="val_pr_auc")
    axes[1].set_title("PR-AUC theo epoch")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pr_curve(y_true, y_proba, out_path: str, label: str = "model"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_bar_comparison(labels, values, out_path: str, ylabel: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color="#3E7CB1")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
