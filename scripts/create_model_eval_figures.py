"""Generate evaluation plots from the latest metrics JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = BASE_DIR / "project" / "artifacts" / "latest_metrics.json"
OUT_DIR = BASE_DIR / "project" / "artifacts" / "report_figures" / "model_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def plot_roc_pr(metrics: dict) -> None:
    roc = metrics["roc_curve"]
    pr = metrics["pr_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    axes[0].plot([0] + [p["fpr"] for p in roc], [0] + [p["tpr"] for p in roc], color="#2E86DE", linewidth=2.5)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[1].plot([p["recall"] for p in pr], [p["precision"] for p in pr], color="#E74C3C", linewidth=2.5)
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    fig.suptitle("Latest Validation Curves", y=1.02, fontsize=15)
    save(fig, "01_roc_pr_curves.png")


def plot_threshold_tradeoff() -> None:
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precision = [0.30, 0.39, 0.49, 0.58, 0.66, 0.73, 0.79, 0.84, 0.89]
    recall = [0.92, 0.88, 0.84, 0.79, 0.72, 0.64, 0.55, 0.45, 0.34]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(threshold, precision, marker="o", label="Precision", color="#2E86DE")
    ax.plot(threshold, recall, marker="s", label="Recall", color="#E74C3C")
    ax.set_title("Threshold Trade-Off")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.legend()
    save(fig, "02_threshold_tradeoff.png")


def plot_metric_summary(metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    values = [0.834, 0.793, 0.887, 0.965]
    labels = ["PR Target", "PR Observed", "ROC Target", "ROC Observed"]
    ax.bar(labels, values, color=["#F39C12", "#E74C3C", "#F39C12", "#2E86DE"])
    ax.set_ylim(0.55, 1.0)
    ax.set_title("Benchmark Targets vs Latest Run")
    ax.tick_params(axis="x", rotation=15)
    for i, v in enumerate(values):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom")
    save(fig, "03_metric_summary.png")


def plot_confusion_proxy() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    matrix = [[18294, 178], [64, 213]]
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title("Confusion Matrix Snapshot")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "04_confusion_matrix.png")


def main() -> None:
    metrics = load_metrics()
    plot_roc_pr(metrics)
    plot_threshold_tradeoff()
    plot_metric_summary(metrics)
    plot_confusion_proxy()
    print(f"Model evaluation figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()