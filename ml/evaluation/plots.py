"""Matplotlib figures for the evaluation report (saved to files, no GUI)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def plot_confusion_matrix(matrix, class_names, path: Path, title: str) -> None:
    matrix = np.asarray(matrix)
    row_totals = matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(matrix / row_totals, cmap="Blues", vmin=0, vmax=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            share = matrix[i, j] / row_totals[i, 0]
            ax.text(j, i, f"{matrix[i, j]}\n({share:.0%})", ha="center", va="center",
                    color="white" if share > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (QUS T-score label)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_history(histories: dict[int, dict], path: Path) -> None:
    """Loss and accuracy curves for every fold; dashed line = start of fine-tuning."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for fold, h in sorted(histories.items()):
        epochs = np.arange(1, len(h["loss"]) + 1)
        line, = ax_loss.plot(epochs, h["loss"], alpha=0.6, label=f"fold {fold} train")
        ax_loss.plot(epochs, h["val_loss"], "--", color=line.get_color(), label=f"fold {fold} val")
        ax_acc.plot(epochs, h["accuracy"], alpha=0.6, color=line.get_color())
        ax_acc.plot(epochs, h["val_accuracy"], "--", color=line.get_color())
        for ax in (ax_loss, ax_acc):
            ax.axvline(h["phase_a_epochs"] + 0.5, color=line.get_color(), linestyle=":", alpha=0.4)
    ax_loss.set(title="Loss (solid = train, dashed = validation)", xlabel="epoch", ylabel="cross-entropy")
    ax_acc.set(title="Accuracy (dotted = fine-tuning starts)", xlabel="epoch", ylabel="accuracy")
    ax_loss.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_roc(y_true, y_score, class_names, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, name in enumerate(class_names):
        positives = (np.asarray(y_true) == i).astype(int)
        if 0 < positives.sum() < len(positives):
            fpr, tpr, _ = roc_curve(positives, np.asarray(y_score)[:, i])
            ax.plot(fpr, tpr, label=f"{name} vs rest")
    ax.plot([0, 1], [0, 1], "k:", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="One-vs-rest ROC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
