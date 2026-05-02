"""
regression.py — Transaction Velocity Forecasting
-------------------------------------------------
Task: Given the fraud counts in the last N time windows (lag features),
      predict how many fraud transactions will occur in the NEXT window.

This is a supervised regression task (Ridge) that is DISTINCT from the
time series analysis task:
  - Regression  → trains a model, outputs RMSE / R², predicts future counts
  - Time Series  → descriptive analysis, no model, outputs rate patterns

Input:  Raw training DataFrame (needs TransactionDT + isFraud columns)
Output: Trained Ridge model, metrics JSON, forecast plot PNG
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_time_series(df: pd.DataFrame, window_hours: int = 1) -> pd.DataFrame:
    """
    Aggregate raw transactions into fixed-size time windows.
    Returns a DataFrame with columns:
        window_start, total_txn, fraud_count, fraud_rate
    """
    df = df.copy()
    # TransactionDT is seconds from a reference point
    dt = pd.to_numeric(df['TransactionDT'], errors='coerce').fillna(0)
    window_sec = window_hours * 3600
    df['window'] = (dt // window_sec).astype(int)

    agg = df.groupby('window').agg(
        total_txn=('TransactionDT', 'count'),
        fraud_count=('isFraud', 'sum')
    ).reset_index().sort_values('window')

    agg['fraud_rate'] = agg['fraud_count'] / agg['total_txn'].clip(lower=1)
    agg['window_start'] = agg['window'] * window_sec
    return agg


def _build_lag_features(agg: pd.DataFrame, n_lags: int = 6) -> pd.DataFrame:
    """
    Build a supervised dataset from the aggregated time series.
    Features: [lag_1, ..., lag_n, hour_sin, hour_cos, dow_sin, dow_cos]
    Target:   fraud_count at next window
    """
    df = agg.copy()

    # Lag features (previous window fraud counts)
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['fraud_count'].shift(lag)

    # Cyclic time encodings to capture daily / weekly periodicity
    # Window index → approximate hour-of-day and day-of-week
    df['hour_of_day'] = (df['window'] % 24)
    df['day_of_week'] = (df['window'] // 24) % 7
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Target: next window's fraud count
    df['target'] = df['fraud_count'].shift(-1)

    # Drop rows with NaN (from lags or shifted target)
    df = df.dropna(subset=[f'lag_{i}' for i in range(1, n_lags + 1)] + ['target'])
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def run_regression(
    train_df: pd.DataFrame,
    artifacts_dir: Path,
    graphs_dir: Path,
    window_hours: int = 1,
    n_lags: int = 6,
    train_ratio: float = 0.8,
) -> dict:
    """
    Full regression pipeline:
      1. Aggregate transactions into hourly windows
      2. Build lag features (supervised framing)
      3. Chronological train/test split (no shuffling — time series integrity)
      4. Train Ridge regression pipeline (StandardScaler + Ridge)
      5. Evaluate: RMSE, MAE, R²
      6. Save forecast chart PNG + metrics JSON

    Returns dict with metrics.
    """
    artifacts_dir = Path(artifacts_dir)
    graphs_dir    = Path(graphs_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("\n>>> [Regression] Building time-window aggregation...")
    agg = _build_time_series(train_df, window_hours=window_hours)

    print(f">>> [Regression] {len(agg)} hourly windows found. Building lag features...")
    feat_df = _build_lag_features(agg, n_lags=n_lags)

    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + \
                   ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

    X = feat_df[feature_cols].values
    y = feat_df['target'].values
    window_ids = feat_df['window'].values

    # ── Chronological split ───────────────────────────────────────────────────
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    win_test        = window_ids[split_idx:]

    print(f">>> [Regression] Train windows: {split_idx} | Test windows: {len(X_test)}")

    # ── Model: StandardScaler + Ridge ────────────────────────────────────────
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  Ridge(alpha=1.0))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0, None)  # fraud counts can't be negative

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_clipped)))
    r2   = float(r2_score(y_test, y_pred_clipped))
    mae  = float(np.mean(np.abs(y_test - y_pred_clipped)))

    print(f">>> [Regression] RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics = {
        "task":          "Transaction Velocity Forecasting",
        "model":         "Ridge Regression",
        "window_hours":  window_hours,
        "n_lags":        n_lags,
        "train_windows": int(split_idx),
        "test_windows":  int(len(X_test)),
        "rmse":          round(rmse, 4),
        "mae":           round(mae, 4),
        "r2":            round(r2, 4),
        "forecast": [
            {"window": int(w), "actual": float(a), "predicted": float(p)}
            for w, a, p in zip(win_test[:100], y_test[:100], y_pred_clipped[:100])
        ]
    }
    out_path = artifacts_dir / 'regression_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f">>> [Regression] Metrics saved → {out_path}")

    # ── Save forecast chart ───────────────────────────────────────────────────
    _save_forecast_plot(win_test, y_test, y_pred_clipped, rmse, r2, graphs_dir)

    return metrics


def _save_forecast_plot(win_test, y_test, y_pred, rmse, r2, graphs_dir: Path):
    PALETTE = {
        'bg': '#0d1117', 'card': '#161b22', 'cyan': '#00d4ff',
        'rose': '#fb7185', 'muted': '#6b7280', 'border': '#30363d', 'text': '#e6edf3',
    }
    plt.style.use('dark_background')

    # Show at most 200 windows for readability
    n = min(200, len(win_test))
    x_axis = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['card'])

    ax.plot(x_axis, y_test[:n],  color=PALETTE['cyan'], linewidth=1.8,
            label='Actual Fraud Count', alpha=0.9)
    ax.plot(x_axis, y_pred[:n],  color=PALETTE['rose'], linewidth=1.8,
            linestyle='--', label=f'Ridge Forecast (RMSE={rmse:.2f}, R²={r2:.3f})', alpha=0.9)
    ax.fill_between(x_axis, y_test[:n], y_pred[:n], alpha=0.08, color=PALETTE['rose'])

    ax.set_title('Transaction Velocity Forecasting — Ridge Regression',
                 color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Time Window (hourly)', color=PALETTE['muted'], fontsize=11)
    ax.set_ylabel('Fraud Transaction Count', color=PALETTE['muted'], fontsize=11)
    ax.tick_params(colors=PALETTE['muted'])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE['border'])
    ax.grid(True, color=PALETTE['border'], linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'],
              labelcolor=PALETTE['text'], fontsize=10)

    plt.tight_layout()
    out = graphs_dir / 'latest_regression_forecast.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f">>> [Regression] Forecast plot saved → {out}")
