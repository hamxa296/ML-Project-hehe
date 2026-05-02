"""
timeseries.py — Fraud Rate Time Series Analysis (Descriptive)
--------------------------------------------------------------
Distinct from regression.py: NO model is trained, no prediction target.
This is purely descriptive — rolling averages, anomaly detection, heatmap.

Outputs:
  - artifacts/timeseries_fraud_rate.json
  - results/graphs/latest_timeseries_line.png
  - results/graphs/latest_timeseries_heatmap.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path


def _aggregate_by_window(df: pd.DataFrame, window_hours: int = 1) -> pd.DataFrame:
    dt = pd.to_numeric(df['TransactionDT'], errors='coerce').fillna(0)
    window_sec = window_hours * 3600
    df = df.copy()
    df['window'] = (dt // window_sec).astype(int)
    df['hour_of_day'] = (df['window'] % 24).astype(int)
    df['day_of_week'] = ((df['window'] // 24) % 7).astype(int)
    agg = df.groupby('window').agg(
        total_txn=('TransactionDT', 'count'),
        fraud_count=('isFraud', 'sum'),
        hour_of_day=('hour_of_day', 'first'),
        day_of_week=('day_of_week', 'first'),
    ).reset_index().sort_values('window')
    agg['fraud_rate'] = (agg['fraud_count'] / agg['total_txn'].clip(lower=1)).round(4)
    agg['rolling_avg'] = agg['fraud_rate'].rolling(window=6, min_periods=1).mean().round(4)
    return agg


def _detect_anomalies(agg: pd.DataFrame, z_threshold: float = 2.5):
    mu = agg['fraud_rate'].mean()
    std = agg['fraud_rate'].std()
    agg = agg.copy()
    agg['z_score'] = ((agg['fraud_rate'] - mu) / (std + 1e-9)).round(4)
    agg['is_anomaly'] = (agg['z_score'] > z_threshold).astype(int)
    return agg, mu, std


def run_timeseries(train_df: pd.DataFrame, artifacts_dir: Path, graphs_dir: Path,
                   window_hours: int = 1, z_threshold: float = 2.5) -> dict:
    artifacts_dir = Path(artifacts_dir)
    graphs_dir = Path(graphs_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("\n>>> [TimeSeries] Aggregating by hourly window...")
    agg = _aggregate_by_window(train_df, window_hours=window_hours)
    agg, mu, std = _detect_anomalies(agg, z_threshold=z_threshold)

    n_anomalies = int(agg['is_anomaly'].sum())
    peak_rate = float(agg['fraud_rate'].max())
    overall_rate = float(train_df['isFraud'].mean())
    print(f">>> [TimeSeries] Windows={len(agg)} | Anomalies={n_anomalies} | Peak={peak_rate:.4f}")

    # Heatmap: mean fraud rate by hour x day
    dt = pd.to_numeric(train_df['TransactionDT'], errors='coerce').fillna(0)
    df_h = train_df.copy()
    df_h['hod'] = ((dt // 3600) % 24).astype(int)
    df_h['dow'] = ((dt // 86400) % 7).astype(int)
    pivot = df_h.groupby(['dow', 'hod'])['isFraud'].mean().unstack(fill_value=0)
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0.0
    pivot = pivot.sort_index(axis=1)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    heatmap_data = [
        {"day": day_names[int(d) % 7], "hour": int(h), "rate": round(float(pivot.loc[d, h]), 5)}
        for d in pivot.index for h in pivot.columns
    ]

    agg_sub = agg.iloc[::max(1, len(agg) // 500)].copy()
    result = {
        "task": "Fraud Rate Time Series Analysis",
        "method": "Descriptive (rolling mean + z-score anomaly detection)",
        "window_hours": window_hours,
        "total_windows": int(len(agg)),
        "overall_fraud_rate": round(overall_rate, 5),
        "peak_fraud_rate": round(peak_rate, 5),
        "anomalous_windows": n_anomalies,
        "timeseries": [
            {"window": int(r['window']), "fraud_rate": float(r['fraud_rate']),
             "rolling_avg": float(r['rolling_avg']), "is_anomaly": int(r['is_anomaly'])}
            for _, r in agg_sub.iterrows()
        ],
        "heatmap": heatmap_data,
    }
    out_path = artifacts_dir / 'timeseries_fraud_rate.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f">>> [TimeSeries] JSON saved → {out_path}")

    _save_line_chart(agg_sub, mu, std, z_threshold, graphs_dir)
    _save_heatmap(pivot, graphs_dir)
    return result


def _save_line_chart(agg_sub, mu, std, z_threshold, graphs_dir: Path):
    PALETTE = {'bg': '#0d1117', 'card': '#161b22', 'cyan': '#00d4ff', 'rose': '#fb7185',
               'amber': '#f59e0b', 'muted': '#6b7280', 'border': '#30363d',
               'text': '#e6edf3', 'emerald': '#10b981'}
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.set_facecolor(PALETTE['bg']); ax.set_facecolor(PALETTE['card'])
    x = np.arange(len(agg_sub))
    rates = agg_sub['fraud_rate'].values
    roll = agg_sub['rolling_avg'].values
    anoms = agg_sub['is_anomaly'].values
    ax.plot(x, rates, color=PALETTE['cyan'], linewidth=1.2, alpha=0.6, label='Fraud Rate (hourly)')
    ax.plot(x, roll, color=PALETTE['emerald'], linewidth=2.0, label='6-Window Rolling Average')
    thresh = mu + z_threshold * std
    ax.axhline(thresh, color=PALETTE['amber'], linestyle='--', linewidth=1.2,
               label=f'Anomaly Threshold (z={z_threshold})')
    anom_x = x[anoms == 1]; anom_y = rates[anoms == 1]
    if len(anom_x):
        ax.scatter(anom_x, anom_y, color=PALETTE['rose'], zorder=5, s=40, label='Anomalous Window')
    ax.set_title('Fraud Rate Over Time — Anomaly Detection', color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Time Window Index (hourly)', color=PALETTE['muted'], fontsize=11)
    ax.set_ylabel('Fraud Rate', color=PALETTE['muted'], fontsize=11)
    ax.tick_params(colors=PALETTE['muted'])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE['border'])
    ax.grid(True, color=PALETTE['border'], linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'], fontsize=10)
    plt.tight_layout()
    out = graphs_dir / 'latest_timeseries_line.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f">>> [TimeSeries] Line chart → {out}")


def _save_heatmap(pivot, graphs_dir: Path):
    import seaborn as sns
    PALETTE = {'bg': '#0d1117', 'card': '#161b22', 'muted': '#6b7280',
               'border': '#30363d', 'text': '#e6edf3'}
    plt.style.use('dark_background')
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot.index = [day_labels[i % 7] for i in pivot.index]
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.set_facecolor(PALETTE['bg']); ax.set_facecolor(PALETTE['card'])
    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', linewidths=0.3, linecolor=PALETTE['border'],
                cbar_kws={'label': 'Mean Fraud Rate'}, annot=False)
    ax.set_title('Fraud Rate Heatmap — Hour of Day × Day of Week',
                 color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Hour of Day', color=PALETTE['muted'], fontsize=11)
    ax.set_ylabel('Day of Week', color=PALETTE['muted'], fontsize=11)
    ax.tick_params(colors=PALETTE['muted'])
    plt.tight_layout()
    out = graphs_dir / 'latest_timeseries_heatmap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f">>> [TimeSeries] Heatmap → {out}")
