import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for Docker/headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, precision_recall_curve, auc,
    roc_auc_score, confusion_matrix, roc_curve
)

def evaluate_model(y_test, probs, preds, project_root: Path = None, version: str = None, artifacts_dir: Path = None):
    """
    Evaluate model predictions, print metrics, save interactive JSON curves,
    and generate/save PNG graph files for the frontend to display.

    Args:
        y_test:       Ground truth labels
        probs:        Predicted probabilities for positive class
        preds:        Hard predictions (0 or 1)
        project_root: Absolute path to the project root directory.
                      Falls back to cwd if not provided (local dev mode).
    """
    if project_root is None:
        project_root = Path.cwd()

    if artifacts_dir is None:
        artifacts_dir = project_root / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    from datetime import datetime

    if version is None:
        version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create output directories anchored to project root
    graphs_dir = project_root / 'results' / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)

    def _save_versioned(fig, name: str):
        """Save figure as both a versioned copy and overwrite the fixed 'latest_' copy."""
        versioned = graphs_dir / f"{name}_{version}.png"
        latest    = graphs_dir / f"latest_{name}.png"
        fig.savefig(versioned, dpi=150, bbox_inches='tight')
        shutil.copy2(versioned, latest)   # atomic overwrite of the 'latest' slot
        plt.close(fig)
        print(f"Saved {name} → {versioned.name} + {latest.name}")
        return latest.name

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, preds))

    pr, rc, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(rc, pr)
    roc_auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)

    print(f"PR-AUC:  {pr_auc:.4f} (Paper XGBoost Benchmark: 0.834)")
    print(f"ROC-AUC: {roc_auc:.4f} (Paper XGBoost Benchmark: 0.887)")

    # ── Styling ──────────────────────────────────────────────────────────────
    plt.style.use('dark_background')
    PALETTE = {
        'bg':      '#0d1117',
        'card':    '#161b22',
        'cyan':    '#00d4ff',
        'violet':  '#0ea5e9',
        'rose':    '#fb7185',
        'emerald': '#10b981',
        'amber':   '#f59e0b',
        'muted':   '#6b7280',
        'border':  '#30363d',
        'text':    '#e6edf3',
    }
    FIGSIZE = (8, 6)

    def _style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(PALETTE['card'])
        ax.figure.set_facecolor(PALETTE['bg'])
        ax.set_title(title, color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
        ax.set_xlabel(xlabel, color=PALETTE['muted'], fontsize=11)
        ax.set_ylabel(ylabel, color=PALETTE['muted'], fontsize=11)
        ax.tick_params(colors=PALETTE['muted'])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE['border'])
        ax.grid(True, color=PALETTE['border'], linestyle='--', linewidth=0.6, alpha=0.6)

    # ── 1. ROC Curve ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(fpr, tpr, color=PALETTE['violet'], linewidth=2.5,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color=PALETTE['muted'], linestyle='--', linewidth=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE['violet'])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    _style_ax(ax, 'ROC Curve', 'False Positive Rate', 'True Positive Rate')
    ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'],
              labelcolor=PALETTE['text'], fontsize=10)
    plt.tight_layout()
    _save_versioned(fig, 'roc_curve')

    # ── 2. Precision-Recall Curve ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(rc, pr, color=PALETTE['cyan'], linewidth=2.5,
            label=f'PR Curve (AUC = {pr_auc:.4f})')
    baseline = y_test.mean()
    ax.axhline(y=baseline, color=PALETTE['muted'], linestyle='--', linewidth=1,
               label=f'Baseline (prevalence = {baseline:.3f})')
    ax.fill_between(rc, pr, alpha=0.12, color=PALETTE['cyan'])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    _style_ax(ax, 'Precision-Recall Curve', 'Recall', 'Precision')
    ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'],
              labelcolor=PALETTE['text'], fontsize=10)
    plt.tight_layout()
    _save_versioned(fig, 'pr_curve')

    # ── 3. Confusion Matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Custom high-contrast colormap for dark theme
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("fraud_cmap", [PALETTE['card'], PALETTE['cyan']])
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=custom_cmap,
        linewidths=2, linecolor=PALETTE['bg'],
        annot_kws={'size': 22, 'weight': 'bold'}, # Removed fixed color for auto-contrast
        xticklabels=['Safe', 'Fraud'],
        yticklabels=['Safe', 'Fraud'],
        ax=ax, cbar=False
    )
    ax.set_title('Confusion Matrix', color=PALETTE['text'], fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', color=PALETTE['muted'], fontsize=12)
    ax.set_ylabel('True Label', color=PALETTE['muted'], fontsize=12)
    ax.tick_params(colors=PALETTE['text'], labelsize=11)
    plt.tight_layout()
    _save_versioned(fig, 'confusion_matrix')

    # ── 4. Metric Bar Chart ───────────────────────────────────────────────────
    from sklearn.metrics import precision_score, recall_score, f1_score
    metric_names  = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    metric_values = [
        precision_score(y_test, preds, zero_division=0),
        recall_score(y_test, preds, zero_division=0),
        f1_score(y_test, preds, zero_division=0),
        roc_auc,
        pr_auc,
    ]
    colors = [PALETTE['cyan'], PALETTE['emerald'], PALETTE['amber'],
              PALETTE['violet'], PALETTE['rose']]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor=PALETTE['border'],
                  linewidth=0.8, width=0.6)
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f'{val:.3f}', ha='center', va='bottom', color=PALETTE['text'],
                fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)
    _style_ax(ax, 'Model Performance Summary', 'Metric', 'Score')
    plt.tight_layout()
    _save_versioned(fig, 'metric_summary')

    # ── 5. Interactive JSON for Recharts frontend ──────────────────────────────
    def subsample(x, y, num=100):
        if len(x) <= num:
            return x.tolist(), y.tolist()
        idx = np.linspace(0, len(x) - 1, num, dtype=int)
        return x[idx].tolist(), y[idx].tolist()

    fpr_sub, tpr_sub = subsample(fpr, tpr)
    rc_sub,  pr_sub  = subsample(rc, pr)

    metrics_data = {
        "version":    version,
        # cache_bust changes every run so the browser doesn't serve stale PNGs
        "cache_bust": datetime.now().isoformat(),
        "roc_curve":  [{"fpr": round(f, 4), "tpr": round(t, 4)}        for f, t in zip(fpr_sub, tpr_sub)],
        "pr_curve":   [{"recall": round(r, 4), "precision": round(p, 4)} for r, p in zip(rc_sub,  pr_sub)],
        # These are the stable "latest" filenames — versioned copies also exist
        "graph_files": {
            "roc_curve":        "latest_roc_curve.png",
            "pr_curve":         "latest_pr_curve.png",
            "confusion_matrix": "latest_confusion_matrix.png",
            "metric_summary":   "latest_metric_summary.png",
        }
    }

    metrics_path = artifacts_dir / 'latest_metrics.json'
    versioned_metrics_path = artifacts_dir / f'metrics_{version}.json'
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f)
    
    with open(versioned_metrics_path, 'w') as f:
        json.dump(metrics_data, f)

    print(f"Saved interactive curve data → {metrics_path}")
    print(f"All graphs saved to → {graphs_dir}/")
    return metrics_data
