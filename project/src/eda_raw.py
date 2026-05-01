"""
Comprehensive Raw Data EDA
Saves all plots to artifacts/eda_raw/
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats

PALETTE = {
    'bg': '#0d1117', 'card': '#161b22', 'cyan': '#00d4ff',
    'violet': '#8b5cf6', 'rose': '#fb7185', 'emerald': '#10b981',
    'amber': '#f59e0b', 'muted': '#6b7280', 'border': '#30363d', 'text': '#e6edf3',
}
plt.style.use('dark_background')

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


def run_raw_eda(train_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "="*60)
    print("  RAW DATA EDA")
    print("="*60)

    # ── Load ──────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df = pd.read_csv(train_path)
    n, p = df.shape
    print(f"  Shape            : {n:,} rows x {p} cols")
    print(f"  Memory           : {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"  Dtypes           : {df.dtypes.value_counts().to_dict()}")

    # ── Target ───────────────────────────────────────────────
    print("\n[2] Target Distribution...")
    vc   = df['isFraud'].value_counts()
    frate = df['isFraud'].mean()
    ratio = vc[0] / vc[1]
    print(f"  Fraud     : {vc[1]:,}  ({frate*100:.4f}%)")
    print(f"  Safe      : {vc[0]:,}  ({(1-frate)*100:.4f}%)")
    print(f"  Imbalance : {ratio:.1f}:1  (Safe:Fraud)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.set_facecolor(PALETTE['bg'])
    bars = axes[0].bar(['Safe', 'Fraud'], [vc[0], vc[1]],
                       color=[PALETTE['emerald'], PALETTE['rose']], edgecolor=PALETTE['border'], width=0.5)
    for b, v in zip(bars, [vc[0], vc[1]]):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+500, f'{v:,}',
                     ha='center', color=PALETTE['text'], fontsize=10, fontweight='bold')
    style_ax(axes[0], 'Class Distribution (Raw Counts)', 'Class', 'Count')

    wedges, texts, autotexts = axes[1].pie(
        [vc[0], vc[1]], labels=['Safe', 'Fraud'],
        autopct='%1.2f%%', startangle=90,
        colors=[PALETTE['emerald'], PALETTE['rose']],
        wedgeprops={'edgecolor': PALETTE['border'], 'linewidth': 1.5})
    for t in texts + autotexts:
        t.set_color(PALETTE['text'])
    axes[1].set_facecolor(PALETTE['bg'])
    axes[1].set_title('Class Split (%)', color=PALETTE['text'], fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig, '01_target_distribution', out_dir)

    # ── Missing Values ───────────────────────────────────────
    print("\n[3] Missing Value Analysis...")
    missing = (df.isnull().mean()*100).sort_values(ascending=False)
    nonzero_missing = missing[missing > 0]
    print(f"  Cols with any missing : {len(nonzero_missing)}")
    print(f"  Cols with 0% missing  : {(missing == 0).sum()}")
    print(f"  Cols with >50% missing: {(missing > 50).sum()}")
    print(f"  Cols with >95% missing: {(missing > 95).sum()}  <- would be pruned")

    if len(nonzero_missing) > 0:
        top_miss = nonzero_missing.head(40)
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = [PALETTE['rose'] if v > 95 else PALETTE['amber'] if v > 50 else PALETTE['cyan']
                  for v in top_miss.values]
        ax.barh(top_miss.index[::-1], top_miss.values[::-1], color=colors[::-1], edgecolor=PALETTE['border'])
        ax.axvline(95, color=PALETTE['rose'], linestyle='--', linewidth=1.5, label='>95% pruned')
        ax.axvline(50, color=PALETTE['amber'], linestyle='--', linewidth=1.5, label='>50%')
        ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'])
        style_ax(ax, 'Missing Values (%) — Top 40 Columns', 'Missing %', 'Feature')
        plt.tight_layout()
        savefig(fig, '02_missing_values', out_dir)
    else:
        print("  No missing values found (data is pre-encoded).")

    # ── TransactionAmt ───────────────────────────────────────
    print("\n[4] TransactionAmt Analysis...")
    amt    = df['TransactionAmt']
    amt_f  = df.loc[df['isFraud']==1, 'TransactionAmt']
    amt_s  = df.loc[df['isFraud']==0, 'TransactionAmt']
    print(f"  Mean   : {amt.mean():.4f}")
    print(f"  Median : {amt.median():.4f}")
    print(f"  Std    : {amt.std():.4f}")
    print(f"  Skew   : {amt.skew():.4f}")
    print(f"  Fraud mean   : {amt_f.mean():.4f}  |  Safe mean  : {amt_s.mean():.4f}")
    print(f"  Fraud std    : {amt_f.std():.4f}   |  Safe std   : {amt_s.std():.4f}")
    stat, pval = stats.mannwhitneyu(amt_f, amt_s, alternative='two-sided')
    print(f"  Mann-Whitney U p-value: {pval:.6f}  ({'SIGNIFICANT' if pval<0.05 else 'not significant'})")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.set_facecolor(PALETTE['bg'])
    fig.suptitle('TransactionAmt Analysis', color=PALETTE['text'], fontsize=15, fontweight='bold', y=1.01)

    axes[0,0].hist(df['TransactionAmt'], bins=80, color=PALETTE['cyan'], edgecolor=PALETTE['border'], alpha=0.8)
    style_ax(axes[0,0], 'TransactionAmt Distribution', 'Amount', 'Count')

    log_amt = np.log1p(df['TransactionAmt'])
    axes[0,1].hist(log_amt, bins=80, color=PALETTE['violet'], edgecolor=PALETTE['border'], alpha=0.8)
    style_ax(axes[0,1], 'log(1+TransactionAmt) Distribution', 'log(1+Amount)', 'Count')

    axes[1,0].hist(amt_s, bins=60, alpha=0.7, label='Safe', color=PALETTE['emerald'], density=True)
    axes[1,0].hist(amt_f, bins=60, alpha=0.7, label='Fraud', color=PALETTE['rose'], density=True)
    axes[1,0].legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'])
    style_ax(axes[1,0], 'Amount Distribution by Class (Density)', 'Amount', 'Density')

    bp = axes[1,1].boxplot([amt_s, amt_f], labels=['Safe', 'Fraud'],
                            patch_artist=True, notch=True,
                            medianprops={'color': PALETTE['amber'], 'linewidth':2})
    bp['boxes'][0].set_facecolor(PALETTE['emerald'] + '55')
    bp['boxes'][1].set_facecolor(PALETTE['rose'] + '55')
    style_ax(axes[1,1], 'Amount Boxplot by Class', 'Class', 'Amount')
    plt.tight_layout()
    savefig(fig, '03_transaction_amount', out_dir)

    # ── Time Analysis ─────────────────────────────────────────
    print("\n[5] Time-Based Analysis...")
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['dow']  = (df['TransactionDT'] // (3600*24)) % 7

    hour_fraud = df.groupby('hour')['isFraud'].agg(['mean','sum','count'])
    dow_fraud  = df.groupby('dow')['isFraud'].agg(['mean','sum','count'])
    print(f"  Peak fraud hour (rate): {hour_fraud['mean'].idxmax()} ({hour_fraud['mean'].max()*100:.2f}%)")
    print(f"  Peak fraud dow  (rate): {dow_fraud['mean'].idxmax()} ({dow_fraud['mean'].max()*100:.2f}%)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.set_facecolor(PALETTE['bg'])
    fig.suptitle('Time-Based Fraud Patterns', color=PALETTE['text'], fontsize=15, fontweight='bold')

    axes[0,0].bar(hour_fraud.index, hour_fraud['count'], color=PALETTE['cyan']+'88', edgecolor=PALETTE['border'])
    style_ax(axes[0,0], 'Transactions per Hour of Day', 'Hour', 'Transaction Count')

    axes[0,1].bar(hour_fraud.index, hour_fraud['mean']*100, color=PALETTE['rose'], edgecolor=PALETTE['border'])
    style_ax(axes[0,1], 'Fraud Rate (%) by Hour of Day', 'Hour', 'Fraud Rate (%)')

    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    axes[1,0].bar(range(7), dow_fraud['count'], color=PALETTE['violet']+'88', edgecolor=PALETTE['border'])
    axes[1,0].set_xticks(range(7)); axes[1,0].set_xticklabels(days, color=PALETTE['muted'])
    style_ax(axes[1,0], 'Transactions per Day of Week', 'Day', 'Count')

    axes[1,1].bar(range(7), dow_fraud['mean']*100, color=PALETTE['amber'], edgecolor=PALETTE['border'])
    axes[1,1].set_xticks(range(7)); axes[1,1].set_xticklabels(days, color=PALETTE['muted'])
    style_ax(axes[1,1], 'Fraud Rate (%) by Day of Week', 'Day', 'Fraud Rate (%)')
    plt.tight_layout()
    savefig(fig, '04_time_analysis', out_dir)

    # ── C Features ───────────────────────────────────────────
    print("\n[6] C-Features (Count Features)...")
    c_cols = [c for c in df.columns if c.startswith('C')]
    print(f"  Count: {len(c_cols)}")
    c_stats = df[c_cols].describe().T[['mean','std','50%','max']]
    print(c_stats.to_string())

    c_corr = df[c_cols + ['isFraud']].corr()['isFraud'].drop('isFraud').sort_values(key=abs, ascending=False)
    print(f"\n  C-features corr with isFraud:")
    print(c_corr.to_string())

    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    fig.set_facecolor(PALETTE['bg'])
    fig.suptitle('C-Feature Distributions by Class', color=PALETTE['text'], fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()
    for i, col in enumerate(c_cols):
        ax = axes_flat[i]
        c_s = df.loc[df['isFraud']==0, col].dropna()
        c_f = df.loc[df['isFraud']==1, col].dropna()
        c_s = c_s[c_s >= 0]
        c_f = c_f[c_f >= 0]
        ax.hist(np.log1p(c_s), bins=40, alpha=0.7,
                color=PALETTE['emerald'], label='Safe', density=True)
        ax.hist(np.log1p(c_f), bins=40, alpha=0.7,
                color=PALETTE['rose'], label='Fraud', density=True)
        style_ax(ax, f'{col} (log)', '', '')
        if i == 0:
            ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'],
                      labelcolor=PALETTE['text'], fontsize=8)
    for j in range(len(c_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    savefig(fig, '05_c_features', out_dir)

    # ── D Features ───────────────────────────────────────────
    print("\n[7] D-Features (Timedelta Features)...")
    d_cols = [c for c in df.columns if c.startswith('D')]
    print(f"  Count: {len(d_cols)}")
    d_miss = (df[d_cols].isnull().mean()*100)
    print(f"  Missing %:\n{d_miss.to_string()}")

    d_corr = df[d_cols + ['isFraud']].corr()['isFraud'].drop('isFraud').sort_values(key=abs, ascending=False)
    print(f"  Corr with isFraud:\n{d_corr.to_string()}")

    n_d = len(d_cols)
    cols_plot = 4
    rows_plot = (n_d + cols_plot - 1) // cols_plot
    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(16, rows_plot*3+1))
    fig.set_facecolor(PALETTE['bg'])
    fig.suptitle('D-Feature Distributions by Class', color=PALETTE['text'], fontsize=14, fontweight='bold')
    axes_flat = axes.flatten() if n_d > 1 else [axes]
    for i, col in enumerate(d_cols):
        ax = axes_flat[i]
        clean_s = df.loc[df['isFraud']==0, col].dropna()
        clean_f = df.loc[df['isFraud']==1, col].dropna()
        clean_s = clean_s[clean_s >= 0]
        clean_f = clean_f[clean_f >= 0]
        ax.hist(np.log1p(clean_s), bins=40, alpha=0.7, color=PALETTE['emerald'], density=True, label='Safe')
        ax.hist(np.log1p(clean_f), bins=40, alpha=0.7, color=PALETTE['rose'], density=True, label='Fraud')
        style_ax(ax, f'{col} (log)', '', '')
    for j in range(len(d_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    savefig(fig, '06_d_features', out_dir)

    # ── V Features ────────────────────────────────────────────
    print("\n[8] V-Features Analysis...")
    v_cols = [c for c in df.columns if c.startswith('V')]
    print(f"  Count: {len(v_cols)}")
    v_miss = (df[v_cols].isnull().mean()*100)
    print(f"  Missing > 50%: {(v_miss > 50).sum()} features")
    print(f"  Missing > 95%: {(v_miss > 95).sum()} features (would be pruned)")

    v_corr = df[v_cols + ['isFraud']].corr()['isFraud'].drop('isFraud').abs().sort_values(ascending=False)
    print(f"\n  Top 20 V-features by |corr| with isFraud:")
    print(v_corr.head(20).to_string())

    top_v = v_corr.head(20).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_v[::-1], v_corr[top_v[::-1]],
                   color=PALETTE['violet'], edgecolor=PALETTE['border'])
    for b, val in zip(bars, v_corr[top_v[::-1]]):
        ax.text(val+0.002, b.get_y()+b.get_height()/2, f'{val:.4f}',
                va='center', color=PALETTE['text'], fontsize=9)
    style_ax(ax, 'Top 20 V-Features — |Correlation| with isFraud', '|Correlation|', 'Feature')
    plt.tight_layout()
    savefig(fig, '07_v_features_corr', out_dir)

    # V-feature missing heatmap
    v_miss_df = v_miss.reset_index()
    v_miss_df.columns = ['feature','missing_pct']
    v_miss_df = v_miss_df.sort_values('missing_pct', ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(v_miss_df)), v_miss_df['missing_pct'],
           color=[PALETTE['rose'] if x>95 else PALETTE['amber'] if x>50 else PALETTE['cyan']
                  for x in v_miss_df['missing_pct']],
           edgecolor=PALETTE['border'], width=1.0)
    ax.axhline(95, color=PALETTE['rose'], linestyle='--', linewidth=1.5, label='>95% pruned')
    ax.axhline(50, color=PALETTE['amber'], linestyle='--', linewidth=1.5, label='>50%')
    ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'])
    style_ax(ax, 'V-Feature Missing Values (%)', 'V-Feature Index', 'Missing %')
    plt.tight_layout()
    savefig(fig, '08_v_features_missing', out_dir)

    # ── Correlation Matrix (top features) ────────────────────
    print("\n[9] Correlation Matrix (Top 25 features with isFraud)...")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    all_corr = df[num_cols].corr()['isFraud'].drop('isFraud').abs().sort_values(ascending=False)
    top25 = all_corr.head(25).index.tolist() + ['isFraud']
    corr_mat = df[top25].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.zeros_like(corr_mat, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdPu',
                linewidths=0.5, linecolor=PALETTE['border'],
                annot_kws={'size': 7}, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_facecolor(PALETTE['bg'])
    ax.figure.set_facecolor(PALETTE['bg'])
    ax.set_title('Correlation Matrix — Top 25 Features + isFraud',
                 color=PALETTE['text'], fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors=PALETTE['muted'], labelsize=8)
    plt.tight_layout()
    savefig(fig, '09_correlation_matrix', out_dir)

    # ── High-Correlation Pairs ────────────────────────────────
    print("\n[10] Redundant Feature Pairs (r > 0.98)...")
    sample_df = df[num_cols].sample(n=min(50000, n), random_state=42)
    corr_m = sample_df.corr().abs()
    upper  = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    pairs  = [(c, r, round(upper.loc[r, c], 4))
              for c in upper.columns for r in upper.index
              if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > 0.98]
    print(f"  Total pairs with r > 0.98: {len(pairs)}")
    for pair in sorted(pairs, key=lambda x: -x[2])[:15]:
        print(f"    {pair[0]:12s} <-> {pair[1]:12s}  r={pair[2]}")

    # ── ProductCD & Card Analysis ─────────────────────────────
    print("\n[11] Category Fraud Rate Analysis...")
    for col in ['ProductCD', 'card4', 'card6']:
        if col in df.columns:
            grp = df.groupby(col)['isFraud'].agg(['mean','count'])
            grp.columns = ['fraud_rate','count']
            grp = grp.sort_values('fraud_rate', ascending=False)
            print(f"\n  {col}:")
            print(grp.to_string())

    cat_cols = ['ProductCD', 'card4', 'card6']
    existing = [c for c in cat_cols if c in df.columns]
    fig, axes = plt.subplots(1, len(existing), figsize=(6*len(existing), 6))
    fig.set_facecolor(PALETTE['bg'])
    if len(existing) == 1: axes = [axes]
    for ax, col in zip(axes, existing):
        grp = df.groupby(col)['isFraud'].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(grp)), grp.values * 100,
                      color=PALETTE['rose'], edgecolor=PALETTE['border'], width=0.6)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels([f'Cat {i}' for i in range(len(grp))], color=PALETTE['muted'], fontsize=9)
        for b, val in zip(bars, grp.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{val*100:.1f}%',
                    ha='center', color=PALETTE['text'], fontsize=9, fontweight='bold')
        style_ax(ax, f'Fraud Rate (%) by {col}', col, 'Fraud Rate (%)')
    plt.tight_layout()
    savefig(fig, '10_category_fraud_rates', out_dir)

    # ── Outlier Analysis ──────────────────────────────────────
    print("\n[12] Outlier Analysis (IQR method on key features)...")
    key_num = ['TransactionAmt'] + c_cols[:5]
    outlier_summary = []
    for col in key_num:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask_out = (df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)
        out_fraud_rate = df.loc[mask_out, 'isFraud'].mean()
        outlier_summary.append({'feature': col, 'n_outliers': mask_out.sum(),
                                 'pct': mask_out.mean()*100, 'fraud_rate_in_outliers': out_fraud_rate})
    out_df = pd.DataFrame(outlier_summary)
    print(out_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(out_df))
    ax.bar(x, out_df['pct'], color=PALETTE['cyan'], edgecolor=PALETTE['border'], width=0.4, label='Outlier %')
    ax2 = ax.twinx()
    ax2.plot(x, out_df['fraud_rate_in_outliers']*100, 'o--',
             color=PALETTE['rose'], linewidth=2, markersize=8, label='Fraud rate in outliers')
    ax.set_xticks(list(x))
    ax.set_xticklabels(out_df['feature'].tolist(), color=PALETTE['muted'], fontsize=9)
    ax.set_ylabel('Outlier %', color=PALETTE['muted'])
    ax2.set_ylabel('Fraud Rate in Outliers (%)', color=PALETTE['rose'])
    ax.set_title('Outlier Analysis — Count vs Fraud Rate', color=PALETTE['text'], fontsize=13, fontweight='bold')
    ax.set_facecolor(PALETTE['card']); ax.figure.set_facecolor(PALETTE['bg'])
    ax.grid(True, color=PALETTE['border'], linestyle='--', linewidth=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, facecolor=PALETTE['card'],
              edgecolor=PALETTE['border'], labelcolor=PALETTE['text'])
    plt.tight_layout()
    savefig(fig, '11_outlier_analysis', out_dir)

    # ── Statistical Summary ───────────────────────────────────
    print("\n[13] Mann-Whitney U Tests (Top 10 V-features)...")
    top10_v = v_corr.head(10).index.tolist()
    mw_results = []
    print(f"  {'Feature':<10} {'U-stat':>12} {'p-value':>12} {'Significant':>12}")
    for col in top10_v:
        s = df.loc[df['isFraud']==0, col].dropna()
        f = df.loc[df['isFraud']==1, col].dropna()
        u, pv = stats.mannwhitneyu(f, s, alternative='two-sided')
        print(f"  {col:<10} {u:>12.0f} {pv:>12.2e} {'YES' if pv<0.05 else 'no':>12}")
        mw_results.append({'feature': col, 'u_stat': round(float(u), 0), 'p_value': float(pv), 'significant': bool(pv < 0.05)})

    # ── JSON Export for interactive frontend ──────────────────
    print("\n[14] Exporting JSON data for frontend...")
    import json

    # Hourly data
    hourly = [{'hour': int(h), 'count': int(row['count']),
                'fraud_count': int(row['sum']), 'fraud_rate': round(float(row['mean'])*100, 4)}
              for h, row in hour_fraud.iterrows()]

    # Day of week data
    dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    dow_data = [{'day': dow_names[int(d)], 'dow': int(d), 'count': int(row['count']),
                 'fraud_count': int(row['sum']), 'fraud_rate': round(float(row['mean'])*100, 4)}
                for d, row in dow_fraud.iterrows()]

    # TransactionAmt histogram (50 bins)
    hist_s, bin_edges = np.histogram(np.log1p(amt_s), bins=50)
    hist_f, _         = np.histogram(np.log1p(amt_f), bins=bin_edges)
    amt_hist = [{'bin': round(float((bin_edges[i]+bin_edges[i+1])/2), 4),
                 'safe': int(hist_s[i]), 'fraud': int(hist_f[i])}
                for i in range(len(hist_s))]

    # Top feature correlations
    top_corr = [{'feature': col, 'correlation': round(float(v), 6)}
                for col, v in all_corr.head(30).items()]

    # C-feature correlations
    c_corr_data = [{'feature': col, 'correlation': round(float(v), 6)}
                   for col, v in c_corr.items()]

    # Outlier data
    outlier_data = [{'feature': r['feature'], 'outlier_pct': round(r['pct'], 4),
                     'fraud_rate_in_outliers': round(float(r['fraud_rate_in_outliers'])*100, 4)}
                    for _, r in out_df.iterrows()]

    # V-feature missing
    v_missing_data = [{'feature': row['feature'], 'missing_pct': round(float(row['missing_pct']), 4)}
                      for _, row in v_miss_df.head(50).iterrows()]

    export = {
        'summary': {
            'total_rows': int(n), 'total_features': int(p),
            'fraud_count': int(vc[1]), 'safe_count': int(vc[0]),
            'fraud_rate': round(float(frate)*100, 4),
            'imbalance_ratio': round(float(ratio), 2),
            'high_corr_pairs': len(pairs),
        },
        'transaction_amt': {
            'mean': round(float(amt.mean()), 4), 'median': round(float(amt.median()), 4),
            'std': round(float(amt.std()), 4), 'skew': round(float(amt.skew()), 4),
            'fraud_mean': round(float(amt_f.mean()), 4), 'safe_mean': round(float(amt_s.mean()), 4),
            'fraud_std': round(float(amt_f.std()), 4), 'safe_std': round(float(amt_s.std()), 4),
        },
        'amt_histogram': amt_hist,
        'hourly_fraud': hourly,
        'dow_fraud': dow_data,
        'top_feature_correlations': top_corr,
        'c_feature_correlations': c_corr_data,
        'outlier_analysis': outlier_data,
        'v_feature_missing': v_missing_data,
        'mann_whitney_tests': mw_results,
        'plots': sorted(f.name for f in out_dir.iterdir() if f.suffix == '.png'),
    }

    json_path = out_dir / 'raw_eda_data.json'
    with open(json_path, 'w') as jf:
        json.dump(export, jf)
    print(f"  JSON exported -> {json_path.name}")

    print("\n" + "="*60)
    print(f"  RAW EDA COMPLETE — plots saved to {out_dir}")
    print("="*60)
