"""Generate publication-style EDA plots from the raw IEEE-CIS CSV files.

Outputs are written under project/artifacts/report_figures/raw_eda/ so the
LaTeX report can reference real, reproducible figures instead of placeholder
images.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "project" / "artifacts" / "report_figures" / "raw_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette(["#2E86DE", "#E74C3C", "#F39C12", "#27AE60"])


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_raw() -> pd.DataFrame:
    transactions = pd.read_csv(DATA_DIR / "train_transaction.csv")
    identity = pd.read_csv(DATA_DIR / "train_identity.csv")
    return transactions.merge(identity, on="TransactionID", how="left")


def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df["isFraud"].value_counts().sort_index()
    labels = ["Legit", "Fraud"]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    bars = ax.bar(labels, counts.values, color=["#2E86DE", "#E74C3C"], width=0.62)
    ax.set_title("Raw Class Balance")
    ax.set_ylabel("Transactions")
    ax.set_yscale("log")
    for bar, value in zip(bars, counts.values):
        ax.annotate(
            f"{value:,}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    save(fig, "01_class_balance.png")


def plot_amount_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    legit = df.loc[df["isFraud"] == 0, "TransactionAmt"].dropna()
    fraud = df.loc[df["isFraud"] == 1, "TransactionAmt"].dropna()
    clipped_legit = legit.clip(upper=legit.quantile(0.99))
    clipped_fraud = fraud.clip(upper=fraud.quantile(0.99))

    sns.kdeplot(clipped_legit, fill=True, alpha=0.25, linewidth=2, label="Legit", ax=axes[0], color="#2E86DE")
    sns.kdeplot(clipped_fraud, fill=True, alpha=0.35, linewidth=2, label="Fraud", ax=axes[0], color="#E74C3C")
    axes[0].set_title("Amount Density (Linear Scale)")
    axes[0].set_xlabel("TransactionAmt (clipped at 99th percentile)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True)

    sns.histplot(np.log1p(legit), bins=60, stat="density", color="#2E86DE", alpha=0.35, ax=axes[1], label="Legit")
    sns.histplot(np.log1p(fraud), bins=60, stat="density", color="#E74C3C", alpha=0.35, ax=axes[1], label="Fraud")
    axes[1].set_title("Amount Distribution (Log1p)")
    axes[1].set_xlabel("log1p(TransactionAmt)")
    axes[1].legend(frameon=True)
    save(fig, "02_amount_distribution.png")


def plot_temporal_patterns(df: pd.DataFrame) -> None:
    temp = df.copy()
    temp["hour"] = (temp["TransactionDT"] // 3600) % 24
    temp["day_of_week"] = (temp["TransactionDT"] // 86400) % 7

    hour_rate = temp.groupby("hour", as_index=False)["isFraud"].mean()
    day_rate = temp.groupby("day_of_week", as_index=False)["isFraud"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].plot(hour_rate["hour"], hour_rate["isFraud"] * 100, marker="o", linewidth=2.3, color="#E74C3C")
    axes[0].set_title("Fraud Rate by Hour")
    axes[0].set_xlabel("Hour (UTC)")
    axes[0].set_ylabel("Fraud Rate (%)")
    axes[0].set_xticks(range(0, 24, 3))

    axes[1].bar(day_rate["day_of_week"], day_rate["isFraud"] * 100, color="#2E86DE", width=0.7)
    axes[1].set_title("Fraud Rate by Day Bucket")
    axes[1].set_xlabel("Day bucket")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].set_xticks(range(7))

    save(fig, "03_temporal_patterns.png")


def plot_missingness_and_correlation(df: pd.DataFrame) -> None:
    missingness = df.isna().mean().sort_values(ascending=False).head(12)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))

    axes[0].barh(missingness.index[::-1], missingness.values[::-1] * 100, color="#F39C12")
    axes[0].set_title("Top Missingness Columns")
    axes[0].set_xlabel("Missingness (%)")

    selected = [c for c in ["TransactionAmt", "dist1", "dist2", "C1", "C2", "D1", "D2", "isFraud"] if c in df.columns]
    corr = df[selected].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=axes[1], fmt=".2f")
    axes[1].set_title("Correlation Snapshot")

    save(fig, "04_missingness_correlation.png")


def main() -> None:
    df = load_raw()
    plot_class_balance(df)
    plot_amount_distribution(df)
    plot_temporal_patterns(df)
    plot_missingness_and_correlation(df)
    print(f"Raw EDA figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()