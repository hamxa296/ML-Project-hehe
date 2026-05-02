"""
Comprehensive Processed Data EDA
Runs after the sklearn pipeline transforms the data.
Saves all plots to artifacts/eda_processed/
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA

PALETTE = {
    'bg': '#0d1117', 'card': '#161b22', 'cyan': '#00d4ff',
    'violet': '#0ea5e9', 'rose': '#fb7185', 'emerald': '#10b981',
    'amber': '#f59e0b', 'muted': '#6b7280', 'border': '#30363d', 'text': '#e6edf3',
}
plt.style.use('dark_background')

CLUSTER_COLORS = [PALETTE['cyan'], PALETTE['violet'], PALETTE['rose'],
                  PALETTE['emerald'], PALETTE['amber']]

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PALETTE['card'])
    ax.figure.set_facecolor(PALETTE['bg'])
    if title:  ax.set_title(title, color=PALETTE['text'], fontsize=13, fontweight='bold', pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=PALETTE['muted'], fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=PALETTE['muted'], fontsize=10)
    ax.tick_params(colors=PALETTE['muted'], labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE['border'])
    ax.grid(True, color=PALETTE['border'], linestyle='--', linewidth=0.5, alpha=0.5)

def savefig(fig, name, out_dir):
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    print(f"  Saved -> {path.name}")


def run_processed_eda(X_transformed: np.ndarray, y: pd.Series,
                      feature_names: list, out_dir: Path,
                      raw_shape: tuple = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  PROCESSED DATA EDA")
    print("="*60)

    df = pd.DataFrame(X_transformed, columns=feature_names)
    df['isFraud'] = y.values

    n, p = df.shape
    p_feat = p - 1  # exclude isFraud

    # ── [1] Feature Reduction Summary ────────────────────────
    print("\n[1] Feature Reduction Summary...")
    if raw_shape:
        print(f"  Raw features      : {raw_shape[1]}")
        print(f"  Processed features: {p_feat}")
        print(f"  Reduction         : {raw_shape[1] - p_feat} features removed")
        print(f"  Kept (%)          : {p_feat/raw_shape[1]*100:.1f}%")

    print(f"  Processed rows    : {n:,}")
    print(f"  Missing values    : {df.drop(columns='isFraud').isnull().sum().sum()}")
    print(f"  Dtypes            : {df.dtypes.value_counts().to_dict()}")

    # ── [2] Engineered Feature Analysis ──────────────────────
    print("\n[2] Engineered Features Analysis...")
    eng_feats = [c for c in ['hour', 'dow', 'Amt_to_Median_User', 'user_count'] if c in df.columns]
    print(f"  Engineered features found: {eng_feats}")

    if eng_feats:
        fig, axes = plt.subplots(2, len(eng_feats), figsize=(5*len(eng_feats), 10))
        fig.set_facecolor(PALETTE['bg'])
        fig.suptitle('Engineered Feature Analysis', color=PALETTE['text'], fontsize=14, fontweight='bold')
        if len(eng_feats) == 1: axes = axes.reshape(2, 1)

        for i, col in enumerate(eng_feats):
            # Distribution by class
            ax = axes[0, i]
            safe  = df.loc[df['isFraud']==0, col].dropna()
            fraud = df.loc[df['isFraud']==1, col].dropna()
            ax.hist(safe,  bins=40, alpha=0.7, color=PALETTE['emerald'], density=True, label='Safe')
            ax.hist(fraud, bins=40, alpha=0.7, color=PALETTE['rose'],    density=True, label='Fraud')
            ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'],
                      labelcolor=PALETTE['text'], fontsize=8)
            style_ax(ax, f'{col} — Distribution by Class', col, 'Density')

            # Fraud rate across bins
            ax2 = axes[1, i]
            if col in ['hour', 'dow']:
                grp = df.groupby(col)['isFraud'].mean() * 100
                ax2.bar(grp.index, grp.values, color=PALETTE['violet'], edgecolor=PALETTE['border'])
            else:
                bins_cut = pd.cut(df[col], bins=20)
                grp = df.groupby(bins_cut)['isFraud'].mean() * 100
                ax2.bar(range(len(grp)), grp.values, color=PALETTE['amber'], edgecolor=PALETTE['border'])
            style_ax(ax2, f'{col} — Fraud Rate (%)', col, 'Fraud Rate (%)')
        plt.tight_layout()
        savefig(fig, 'p01_engineered_features', out_dir)

    # ── [3] Cluster Analysis ──────────────────────────────────
    print("\n[3] Cluster Label Analysis...")
    if 'cluster_label' in df.columns:
        cluster_counts = df['cluster_label'].value_counts().sort_index()
        cluster_fraud  = df.groupby('cluster_label')['isFraud'].agg(['mean','sum','count'])
        cluster_fraud.columns = ['fraud_rate','fraud_count','total']

        print(f"  Cluster distribution:")
        for cl, row in cluster_fraud.iterrows():
            print(f"    Cluster {cl}: {int(row['total']):>7,} samples  |  "
                  f"Fraud rate: {row['fraud_rate']*100:.3f}%  |  "
                  f"Fraud count: {int(row['fraud_count']):>5,}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.set_facecolor(PALETTE['bg'])
        fig.suptitle('KMeans Cluster Analysis', color=PALETTE['text'], fontsize=14, fontweight='bold')

        # Cluster sizes
        bars = axes[0].bar(cluster_counts.index, cluster_counts.values,
                           color=CLUSTER_COLORS[:len(cluster_counts)], edgecolor=PALETTE['border'])
        for b, v in zip(bars, cluster_counts.values):
            axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+200, f'{v:,}',
                         ha='center', color=PALETTE['text'], fontsize=9, fontweight='bold')
        style_ax(axes[0], 'Samples per Cluster', 'Cluster', 'Count')

        # Fraud rate per cluster
        bars2 = axes[1].bar(cluster_fraud.index, cluster_fraud['fraud_rate']*100,
                             color=CLUSTER_COLORS[:len(cluster_fraud)], edgecolor=PALETTE['border'])
        for b, v in zip(bars2, cluster_fraud['fraud_rate']):
            axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{v*100:.2f}%',
                         ha='center', color=PALETTE['text'], fontsize=10, fontweight='bold')
        style_ax(axes[1], 'Fraud Rate (%) per Cluster', 'Cluster', 'Fraud Rate (%)')

        # Fraud count vs Safe count stacked
        fraud_c = cluster_fraud['fraud_count'].values
        safe_c  = cluster_fraud['total'].values - fraud_c
        x = np.arange(len(cluster_fraud))
        axes[2].bar(x, safe_c,  color=PALETTE['emerald']+'88', edgecolor=PALETTE['border'], label='Safe')
        axes[2].bar(x, fraud_c, bottom=safe_c, color=PALETTE['rose']+'88', edgecolor=PALETTE['border'], label='Fraud')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f'Cluster {i}' for i in cluster_fraud.index], color=PALETTE['muted'])
        axes[2].legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'])
        style_ax(axes[2], 'Safe vs Fraud Count per Cluster', 'Cluster', 'Count')
        plt.tight_layout()
        savefig(fig, 'p02_cluster_analysis', out_dir)

        # TransactionAmt by cluster
        if 'TransactionAmt' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            cluster_data = [df.loc[df['cluster_label']==cl, 'TransactionAmt'].dropna().values
                            for cl in sorted(df['cluster_label'].unique())]
            bp = ax.boxplot(cluster_data,
                            labels=[f'Cluster {i}' for i in sorted(df['cluster_label'].unique())],
                            patch_artist=True, notch=True,
                            medianprops={'color': PALETTE['amber'], 'linewidth': 2})
            for patch, color in zip(bp['boxes'], CLUSTER_COLORS):
                patch.set_facecolor(color + '55')
            style_ax(ax, 'TransactionAmt Distribution per Cluster', 'Cluster', 'TransactionAmt')
            plt.tight_layout()
            savefig(fig, 'p03_cluster_amounts', out_dir)
    else:
        print("  'cluster_label' not found in processed data.")

    # ── [4] PCA Visualisation ─────────────────────────────────
    print("\n[4] PCA 2D Visualisation...")
    feat_cols = [c for c in df.columns if c != 'isFraud']
    X_pca_in = df[feat_cols].fillna(0).values

    sample_n = min(20000, n)
    idx = np.random.RandomState(42).choice(n, sample_n, replace=False)
    X_s = X_pca_in[idx]
    y_s = df['isFraud'].values[idx]

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_s)
    var_explained = pca.explained_variance_ratio_
    print(f"  PC1 variance explained: {var_explained[0]*100:.2f}%")
    print(f"  PC2 variance explained: {var_explained[1]*100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.set_facecolor(PALETTE['bg'])

    # Coloured by class
    for cls, col, lbl in [(0, PALETTE['emerald'], 'Safe'), (1, PALETTE['rose'], 'Fraud')]:
        mask = y_s == cls
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, alpha=0.3, s=6, label=lbl)
    axes[0].legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'])
    style_ax(axes[0],
             f'PCA — Class (PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%)',
             'PC1', 'PC2')

    # Coloured by cluster (if available)
    if 'cluster_label' in df.columns:
        clusters_s = df['cluster_label'].values[idx]
        for cl in sorted(np.unique(clusters_s)):
            m = clusters_s == cl
            axes[1].scatter(X_2d[m, 0], X_2d[m, 1],
                            c=CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)],
                            alpha=0.3, s=6, label=f'Cluster {cl}')
        axes[1].legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'],
                       labelcolor=PALETTE['text'], markerscale=3)
        style_ax(axes[1], 'PCA — KMeans Cluster Assignment', 'PC1', 'PC2')
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    savefig(fig, 'p04_pca_visualisation', out_dir)

    # ── [5] Feature Distributions (Processed) ────────────────
    print("\n[5] Processed Feature Distributions (top 20 by corr)...")
    feat_corr = df[feat_cols + ['isFraud']].corr()['isFraud'].drop('isFraud').abs().sort_values(ascending=False)
    top20_feats = feat_corr.head(20).index.tolist()

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.set_facecolor(PALETTE['bg'])
    fig.suptitle('Top 20 Processed Features — Distribution by Class',
                 color=PALETTE['text'], fontsize=14, fontweight='bold')
    for i, col in enumerate(top20_feats):
        ax = axes[i//5][i%5]
        s = df.loc[df['isFraud']==0, col].dropna()
        f = df.loc[df['isFraud']==1, col].dropna()
        ax.hist(s, bins=40, alpha=0.7, color=PALETTE['emerald'], density=True, label='Safe')
        ax.hist(f, bins=40, alpha=0.7, color=PALETTE['rose'],    density=True, label='Fraud')
        style_ax(ax, col, '', '')
        if i == 0:
            ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'],
                      labelcolor=PALETTE['text'], fontsize=7)
    plt.tight_layout()
    savefig(fig, 'p05_top_feature_distributions', out_dir)

    # ── [6] Processed Correlation Heatmap ────────────────────
    print("\n[6] Processed Feature Correlation Heatmap (top 25)...")
    top25 = feat_corr.head(24).index.tolist() + ['isFraud']
    corr_mat = df[top25].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdPu',
                linewidths=0.4, linecolor=PALETTE['border'],
                annot_kws={'size': 7}, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_facecolor(PALETTE['bg']); ax.figure.set_facecolor(PALETTE['bg'])
    ax.set_title('Processed Feature Correlations (Top 25)',
                 color=PALETTE['text'], fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors=PALETTE['muted'], labelsize=8)
    plt.tight_layout()
    savefig(fig, 'p06_processed_correlation', out_dir)

    # ── [7] Mutual Information ────────────────────────────────
    print("\n[7] Mutual Information with isFraud (top 30)...")
    from sklearn.feature_selection import mutual_info_classif
    sample_idx = np.random.RandomState(42).choice(n, min(50000, n), replace=False)
    X_mi = df[feat_cols].fillna(0).values[sample_idx]
    y_mi = df['isFraud'].values[sample_idx]
    mi = mutual_info_classif(X_mi, y_mi, random_state=42)
    mi_series = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
    top30_mi = mi_series.head(30)
    print(top30_mi.to_string())

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(top30_mi.index[::-1], top30_mi.values[::-1],
            color=PALETTE['cyan'], edgecolor=PALETTE['border'])
    style_ax(ax, 'Mutual Information with isFraud — Top 30 Processed Features',
             'Mutual Information Score', 'Feature')
    plt.tight_layout()
    savefig(fig, 'p07_mutual_information', out_dir)

    # ── [8] Statistical Tests on Engineered Features ─────────
    print("\n[8] Mann-Whitney U Tests on Engineered + Top Features...")
    test_feats = eng_feats + feat_corr.head(10).index.tolist()
    test_feats = list(dict.fromkeys(test_feats))
    mw_proc = []
    print(f"  {'Feature':<25} {'U-stat':>14} {'p-value':>12} {'Significant':>12}")
    for col in test_feats:
        if col not in df.columns: continue
        s = df.loc[df['isFraud']==0, col].dropna()
        f = df.loc[df['isFraud']==1, col].dropna()
        u, pv = stats.mannwhitneyu(f, s, alternative='two-sided')
        print(f"  {col:<25} {u:>14.0f} {pv:>12.2e} {'YES' if pv<0.05 else 'no':>12}")
        mw_proc.append({'feature': col, 'p_value': float(pv), 'significant': bool(pv < 0.05)})

    # ── JSON Export ────────────────────────────────────────────
    print("\n[9] Exporting JSON data for frontend...")
    import json

    cluster_export = []
    if 'cluster_label' in df.columns:
        for cl, row in cluster_fraud.iterrows():
            cluster_export.append({
                'cluster': int(cl),
                'total': int(row['total']),
                'fraud_count': int(row['fraud_count']),
                'safe_count': int(row['total'] - row['fraud_count']),
                'fraud_rate': round(float(row['fraud_rate'])*100, 4),
            })

    mi_export = [{'feature': col, 'mi_score': round(float(v), 6)}
                 for col, v in top30_mi.items()]

    top_proc_corr = [{'feature': col, 'correlation': round(float(v), 6)}
                     for col, v in feat_corr.head(30).items()]

    # PCA scatter — export a sample (2000 pts) for the frontend
    pca_sample_n = min(2000, len(X_2d))
    pca_idx = np.random.RandomState(0).choice(len(X_2d), pca_sample_n, replace=False)
    pca_export = [{'pc1': round(float(X_2d[i,0]), 4), 'pc2': round(float(X_2d[i,1]), 4),
                   'label': int(y_s[i]),
                   'cluster': int(df['cluster_label'].values[idx[i]]) if 'cluster_label' in df.columns else 0}
                  for i in pca_idx]

    export = {
        'summary': {
            'raw_features': int(raw_shape[1]) if raw_shape else None,
            'processed_features': int(p_feat),
            'features_removed': int(raw_shape[1] - p_feat) if raw_shape else None,
            'total_rows': int(n),
            'missing_values': int(df.drop(columns='isFraud').isnull().sum().sum()),
            'n_clusters': int(df['cluster_label'].nunique()) if 'cluster_label' in df.columns else 0,
            'pca_pc1_variance': round(float(var_explained[0])*100, 2),
            'pca_pc2_variance': round(float(var_explained[1])*100, 2),
        },
        'cluster_analysis': cluster_export,
        'mutual_information': mi_export,
        'top_feature_correlations': top_proc_corr,
        'pca_scatter': pca_export,
        'mann_whitney_tests': mw_proc,
        'plots': sorted(f.name for f in out_dir.iterdir() if f.suffix == '.png'),
    }

    json_path = out_dir / 'processed_eda_data.json'
    with open(json_path, 'w') as jf:
        json.dump(export, jf)
    print(f"  JSON exported -> {json_path.name}")

    print("\n" + "="*60)
    print(f"  PROCESSED EDA COMPLETE — plots saved to {out_dir}")
    print("="*60)

