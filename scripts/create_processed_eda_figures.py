"""Generate EDA plots from the cleaned and feature-engineered dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "project" / "artifacts" / "report_figures" / "processed_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette(["#2E86DE", "#E74C3C", "#27AE60", "#F39C12"])


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_processed() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "processed_train.csv")


def plot_feature_density(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=[np.number]).columns
    non_target = [c for c in numeric if c != "isFraud"]
    sample_cols = non_target[:8]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    for ax, col in zip(axes, sample_cols):
        sns.histplot(df[col].dropna(), bins=40, kde=True, ax=ax, color="#2E86DE")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20)
    for ax in axes[len(sample_cols):]:
        ax.axis("off")
    fig.suptitle("Cleaned Feature Distributions", y=1.02, fontsize=15)
    save(fig, "01_feature_density.png")


def plot_correlation(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if "TransactionID" in numeric.columns:
        numeric = numeric.drop(columns=["TransactionID"])
    corr = numeric.corr(numeric_only=True)
    top = corr["isFraud"].drop("isFraud").abs().sort_values(ascending=False).head(15).index
    subcorr = df[list(top) + ["isFraud"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(subcorr, cmap="coolwarm", center=0, annot=False, ax=ax, linewidths=0.4)
    ax.set_title("Top Feature Correlations After Cleaning")
    save(fig, "02_processed_correlation.png")


def plot_pca(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=[np.number]).copy()
    features = [c for c in numeric.columns if c != "isFraud"][:50]
    subset = numeric[features].fillna(numeric[features].median())
    scaled = StandardScaler().fit_transform(subset)
    comps = PCA(n_components=2, random_state=42).fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    scatter = ax.scatter(comps[:, 0], comps[:, 1], c=df["isFraud"], cmap="coolwarm", s=4, alpha=0.18)
    ax.set_title("PCA Projection of Cleaned Feature Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="isFraud")
    save(fig, "03_processed_pca.png")


def plot_mutual_information_proxy(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=[np.number]).copy().drop(columns=[c for c in ["TransactionID"] if c in df.columns], errors="ignore")
    target = numeric.pop("isFraud")
    corr = numeric.apply(lambda s: abs(s.corr(target)))
    corr = corr.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.barh(corr.index[::-1], corr.values[::-1], color="#27AE60")
    ax.set_title("Proxy for Feature Relevance After Cleaning")
    ax.set_xlabel("Absolute correlation with isFraud")
    save(fig, "04_feature_relevance_proxy.png")


def main() -> None:
    df = load_processed()
    plot_feature_density(df)
    plot_correlation(df)
    plot_pca(df)
    plot_mutual_information_proxy(df)
    print(f"Processed EDA figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()