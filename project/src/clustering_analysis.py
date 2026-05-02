"""
clustering_analysis.py — KMeans Cluster Profile Analysis
---------------------------------------------------------
Promotes the KMeans hidden inside the sklearn pipeline into a standalone
interpretable analytical task showing per-cluster fraud rates.

Outputs:
  - artifacts/cluster_profiles.json
  - results/graphs/latest_cluster_fraud_rate.png
  - results/graphs/latest_cluster_size.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path


def run_clustering_analysis(fitted_pipeline, X_train, y_train, artifacts_dir, graphs_dir):
    artifacts_dir = Path(artifacts_dir)
    graphs_dir = Path(graphs_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("\n>>> [Clustering] Extracting cluster assignments from fitted pipeline...")
    steps = fitted_pipeline.named_steps
    X_t = steps['prune'].transform(X_train)
    X_t = steps['fe'].transform(X_t)
    X_t = steps['clustering'].transform(X_t)

    labels = X_t['cluster_label'].values
    y_arr = np.array(y_train)
    n_clusters = int(labels.max()) + 1
    overall_rate = float(y_arr.mean())
    print(f">>> [Clustering] {n_clusters} clusters | {len(labels)} samples")

    key_cols = [c for c in ['TransactionAmt','hour','dow','user_count','Amt_to_Median_User']
                if c in X_t.columns]
    profiles = []
    for c in range(n_clusters):
        mask = labels == c
        cy = y_arr[mask]
        size = int(mask.sum())
        fraud_count = int(cy.sum())
        fraud_rate = float(fraud_count / size) if size > 0 else 0.0
        feat_means = {col: round(float(X_t.loc[X_t.index[mask], col].mean()), 4) for col in key_cols}
        profiles.append({"cluster": c, "size": size, "fraud_count": fraud_count,
                         "fraud_rate": round(fraud_rate, 5),
                         "pct_of_data": round(size / len(labels) * 100, 2),
                         "feature_means": feat_means})

    best = max(profiles, key=lambda p: p['fraud_rate'])
    result = {"task": "Cluster Behaviour Profiling (KMeans)", "n_clusters": n_clusters,
              "overall_fraud_rate": round(overall_rate, 5), "profiles": profiles,
              "insight": f"Cluster {best['cluster']} has highest fraud rate: {best['fraud_rate']*100:.2f}%"}

    out_path = artifacts_dir / 'cluster_profiles.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f">>> [Clustering] Saved → {out_path} | {result['insight']}")

    # --- Bar chart: fraud rate per cluster
    PALETTE = {'bg':'#0d1117','card':'#161b22','violet':'#0ea5e9','rose':'#fb7185',
               'amber':'#f59e0b','muted':'#6b7280','border':'#30363d','text':'#e6edf3'}
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.set_facecolor(PALETTE['bg']); ax.set_facecolor(PALETTE['card'])
    clabs = [f"Cluster {p['cluster']}" for p in profiles]
    rates = [p['fraud_rate'] * 100 for p in profiles]
    colors = [PALETTE['rose'] if r > overall_rate * 100 else PALETTE['violet'] for r in rates]
    bars = ax.bar(clabs, rates, color=colors, edgecolor=PALETTE['border'], linewidth=0.8)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05,
                f'{val:.2f}%', ha='center', va='bottom', color=PALETTE['text'], fontsize=11, fontweight='bold')
    ax.axhline(overall_rate*100, color=PALETTE['amber'], linestyle='--', linewidth=1.5,
               label=f'Overall Rate ({overall_rate*100:.2f}%)')
    ax.set_title('Fraud Rate by KMeans Cluster', color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Cluster', color=PALETTE['muted'], fontsize=11)
    ax.set_ylabel('Fraud Rate (%)', color=PALETTE['muted'], fontsize=11)
    ax.tick_params(colors=PALETTE['muted'])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE['border'])
    ax.grid(True, axis='y', color=PALETTE['border'], linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'], fontsize=10)
    plt.tight_layout()
    out = graphs_dir / 'latest_cluster_fraud_rate.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f">>> [Clustering] Fraud rate chart → {out}")

    # --- Pie chart: cluster sizes
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.set_facecolor(PALETTE['bg'])
    pie_labels = [f"C{p['cluster']}\n({p['pct_of_data']:.1f}%)" for p in profiles]
    sizes = [p['size'] for p in profiles]
    pie_colors = ['#00d4ff','#8b5cf6','#fb7185','#f59e0b','#10b981']
    ax.pie(sizes, labels=pie_labels, colors=pie_colors[:len(profiles)], startangle=140,
           wedgeprops=dict(edgecolor=PALETTE['border'], linewidth=1.2),
           textprops=dict(color=PALETTE['text'], fontsize=11))
    ax.set_title('Cluster Size Distribution', color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
    plt.tight_layout()
    out2 = graphs_dir / 'latest_cluster_size.png'
    fig.savefig(out2, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f">>> [Clustering] Size chart → {out2}")

    return result
