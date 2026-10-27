"""
visualize.py — Generate EDA plots for the Dry Bean Dataset.

Generates:
1. class_distribution.png — Frequency of each bean variety
2. correlation_heatmap.png — Correlation matrix across shape measurements
3. feature_boxplots.png — Box plots of key morphologic metrics per variety
4. feature_histograms.png — Distribution of features across the dataset
5. pairplot.png — Multi-feature scatter relationships
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_preprocessing import (
    download_dataset,
    load_data,
    clean_data,
    engineer_features,
    ORIGINAL_FEATURES,
    TARGET_COLUMN,
)

IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")


def plot_class_distribution(df):
    """Bar chart showing the distribution of bean varieties."""
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df[TARGET_COLUMN].value_counts().sort_values(ascending=True)
    colors = sns.color_palette("viridis", len(counts))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{count} ({count/len(df)*100:.1f}%)",
                va="center", fontsize=10)

    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_title("Distribution of Dry Bean Varieties (Total: 13,543)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_heatmap(df):
    """Correlation heatmap of numeric features."""
    numeric_cols = [c for c in ORIGINAL_FEATURES if c in df.columns]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap (Original Morphological Attributes)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_boxplots(df):
    """Box plots of key features grouped by bean class."""
    key_features = ["Area", "Perimeter", "Roundness", "Compactness",
                    "AspectRatio", "Eccentricity"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=feat, ax=axes[i],
                    palette="viridis", flierprops={"markersize": 2})
        axes[i].set_title(f"{feat} by Bean Variety", fontsize=12, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_xlabel("")

    plt.suptitle("Feature Distributions by Bean Variety", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, "feature_boxplots.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_histograms(df):
    """Histograms of all numeric features."""
    numeric_cols = [c for c in ORIGINAL_FEATURES if c in df.columns]
    n = len(numeric_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=40, color="steelblue", edgecolor="black", linewidth=0.5, alpha=0.8)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions — Dry Bean Dataset", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, "feature_histograms.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pairplot(df):
    """Pair plot of selected features, colored by class."""
    selected = ["Area", "Roundness", "Compactness", "AspectRatio", TARGET_COLUMN]
    subset = df[selected].copy()

    g = sns.pairplot(subset, hue=TARGET_COLUMN, palette="viridis",
                     diag_kind="kde", plot_kws={"alpha": 0.5, "s": 10},
                     height=2.5)
    g.figure.suptitle("Pair Plot — Morphological Features by Bean Variety",
                      fontsize=14, fontweight="bold", y=1.02)
    save_path = os.path.join(IMAGES_DIR, "pairplot.png")
    g.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("GENERATING EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("=" * 60)

    download_dataset()
    df = load_data()
    df = clean_data(df)

    print("\nGenerating plots...")
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    plot_histograms(df)
    plot_pairplot(df)

    print(f"\nAll plots successfully generated in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
