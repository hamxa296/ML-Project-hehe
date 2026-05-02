"""
dimensionality_reduction.py — PCA Fraud Visualization
------------------------------------------------------
Task: Compress the 175-feature processed space down to 2D with PCA.
      Visualize whether fraud transactions form separable clusters.

This answers: "Does fraud cluster visibly in feature space?"
It JUSTIFIES the XGBoost approach by showing non-linear separation exists.

Outputs:
  - artifacts/pca_results.json (variance explained + 2D coords sample)
  - results/graphs/latest_pca_scatter.png
  - results/graphs/latest_pca_variance.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_dimensionality_reduction(
    X_processed: np.ndarray,
    y: pd.Series,
    feature_names: list,
    artifacts_dir: Path,
    graphs_dir: Path,
    n_components: int = 10,
    sample_size: int = 5000,
) -> dict:
    """
    PCA on the post-pipeline processed features:
      1. StandardScale (PCA is variance-sensitive)
      2. Fit PCA with n_components
      3. Extract 2D projection for scatter plot
      4. Record variance explained per component
      5. Identify top contributing features per PC

    Args:
        X_processed:  numpy array of the processed features (post sklearn pipeline)
        y:            fraud labels (Series)
        feature_names: list of feature names matching X_processed columns
        sample_size:  max points to plot (scatter plot performance)
    """
    artifacts_dir = Path(artifacts_dir)
    graphs_dir    = Path(graphs_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> [PCA] Input shape: {X_processed.shape} | Fitting PCA(n={n_components})...")

    # Replace any inf/nan that may have slipped through
    X_clean = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    pca = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_.tolist()
    cumulative    = np.cumsum(var_explained).tolist()
    print(f">>> [PCA] PC1={var_explained[0]:.3f} | PC2={var_explained[1]:.3f} | "
          f"Top-{n_components} cumulative={cumulative[-1]:.3f}")

    # Top contributing features per component
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names[:X_clean.shape[1]],
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    top_features = {}
    for pc in loadings.columns:
        top_features[pc] = loadings[pc].abs().nlargest(5).index.tolist()

    # Subsample for scatter (stratified by class)
    y_arr = np.array(y)
    idx_fraud = np.where(y_arr == 1)[0]
    idx_safe  = np.where(y_arr == 0)[0]
    n_fraud   = min(len(idx_fraud), sample_size // 5)   # oversample fraud for visibility
    n_safe    = min(len(idx_safe),  sample_size - n_fraud)
    rng = np.random.default_rng(42)
    sample_idx = np.concatenate([
        rng.choice(idx_fraud, n_fraud, replace=False),
        rng.choice(idx_safe,  n_safe,  replace=False),
    ])
    X_2d     = X_pca[sample_idx, :2]
    y_sample = y_arr[sample_idx]

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "task":          "Dimensionality Reduction (PCA)",
        "n_components":  pca.n_components_,
        "total_features": int(X_clean.shape[1]),
        "variance_explained": [round(v, 5) for v in var_explained],
        "cumulative_variance": [round(v, 5) for v in cumulative],
        "top_features_per_pc": top_features,
        "scatter_sample": [
            {"pc1": round(float(x[0]), 4), "pc2": round(float(x[1]), 4), "is_fraud": int(label)}
            for x, label in zip(X_2d, y_sample)
        ],
    }
    out_path = artifacts_dir / 'pca_results.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f">>> [PCA] Results saved → {out_path}")

    _save_scatter(X_2d, y_sample, var_explained, graphs_dir)
    _save_variance_plot(var_explained, cumulative, graphs_dir)

    return result


def _save_scatter(X_2d, y_sample, var_explained, graphs_dir: Path):
    PALETTE = {'bg': '#0d1117', 'card': '#161b22', 'cyan': '#00d4ff', 'rose': '#fb7185',
               'muted': '#6b7280', 'border': '#30363d', 'text': '#e6edf3'}
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.set_facecolor(PALETTE['bg']); ax.set_facecolor(PALETTE['card'])

    safe_idx  = y_sample == 0
    fraud_idx = y_sample == 1
    ax.scatter(X_2d[safe_idx, 0],  X_2d[safe_idx, 1],  c=PALETTE['cyan'],
               s=4, alpha=0.25, label=f'Safe ({safe_idx.sum():,})', rasterized=True)
    ax.scatter(X_2d[fraud_idx, 0], X_2d[fraud_idx, 1], c=PALETTE['rose'],
               s=10, alpha=0.6, label=f'Fraud ({fraud_idx.sum():,})', rasterized=True)

    ax.set_title('PCA 2D Projection — Fraud vs Safe Transactions',
                 color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)', color=PALETTE['muted'], fontsize=11)
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)', color=PALETTE['muted'], fontsize=11)
    ax.tick_params(colors=PALETTE['muted'])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE['border'])
    ax.grid(True, color=PALETTE['border'], linestyle='--', linewidth=0.5, alpha=0.4)
    ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'],
              labelcolor=PALETTE['text'], fontsize=11, markerscale=3)

    plt.tight_layout()
    out = graphs_dir / 'latest_pca_scatter.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f">>> [PCA] Scatter plot → {out}")


def _save_variance_plot(var_explained, cumulative, graphs_dir: Path):
    PALETTE = {'bg': '#0d1117', 'card': '#161b22', 'cyan': '#00d4ff', 'amber': '#f59e0b',
               'muted': '#6b7280', 'border': '#30363d', 'text': '#e6edf3'}
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.set_facecolor(PALETTE['bg']); ax.set_facecolor(PALETTE['card'])
    x = np.arange(1, len(var_explained) + 1)
    ax.bar(x, [v * 100 for v in var_explained], color=PALETTE['cyan'],
           alpha=0.75, label='Individual')
    ax2 = ax.twinx()
    ax2.plot(x, [v * 100 for v in cumulative], color=PALETTE['amber'],
             marker='o', linewidth=2, markersize=5, label='Cumulative')
    ax2.set_ylabel('Cumulative Variance (%)', color=PALETTE['amber'], fontsize=11)
    ax2.tick_params(colors=PALETTE['amber'])
    ax2.set_ylim(0, 105)
    ax.set_title('PCA Explained Variance by Component',
                 color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Principal Component', color=PALETTE['muted'], fontsize=11)
    ax.set_ylabel('Individual Variance (%)', color=PALETTE['cyan'], fontsize=11)
    ax.tick_params(colors=PALETTE['muted'])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE['border'])
    ax.grid(True, color=PALETTE['border'], linestyle='--', linewidth=0.5, alpha=0.4)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, facecolor=PALETTE['card'],
              edgecolor=PALETTE['border'], labelcolor=PALETTE['text'], fontsize=10)
    plt.tight_layout()
    out = graphs_dir / 'latest_pca_variance.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f">>> [PCA] Variance plot → {out}")
