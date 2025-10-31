# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
INPUT_FILE = "data/linkedin_posts_clean.csv"
OUTPUT_DIR = "data/"

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_eda():
    print("📂 Loading data...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["post_date"])
    print(f"✅ Data loaded: {df.shape[0]} posts, {df.shape[1]} columns")

    # -------------------------
    # Summary Statistics
    # -------------------------
    print("\n📊 Summary Statistics (first 5 rows):")
    print(df.head())
    print("\n🔢 Numeric summary:")
    print(df.describe())

    # -------------------------
    # Plot 1: Histogram of Impressions
    # -------------------------
    print("📈 Saving impressions histogram...")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["impressions"], bins=30, kde=True)
    plt.title("Distribution of Impressions")
    plt.xlabel("Impressions")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "impressions_hist.png"))
    plt.close()

    # -------------------------
    # Plot 2: Engagement Rate Distribution
    # -------------------------
    print("📈 Saving engagement rate histogram...")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["engagement_rate"], bins=30, kde=True, color="green")
    plt.title("Distribution of Engagement Rate")
    plt.xlabel("Engagement Rate")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "engagement_rate_hist.png"))
    plt.close()

    # -------------------------
    # Plot 3: Correlation Heatmap
    # -------------------------
    print("📈 Saving correlation heatmap...")
    plt.figure(figsize=(8, 6))
    corr = df[["impressions", "reactions", "comments", "reposts", "saves", "engagement_rate"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlations_heatmap.png"))
    plt.close()

    # -------------------------
    # Plot 4: Impressions vs Reactions Scatter
    # -------------------------
    print("📈 Saving impressions vs reactions scatterplot...")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="impressions", y="reactions", alpha=0.6)
    plt.title("Impressions vs. Reactions")
    plt.xlabel("Impressions")
    plt.ylabel("Reactions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "impressions_vs_reactions.png"))
    plt.close()

    # -------------------------
    # Plot 5: Boxplot Engagement by Weekday
    # -------------------------
    if "weekday" in df.columns:
        print("📈 Saving engagement rate by weekday boxplot...")
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="weekday", y="engagement_rate")
        plt.title("Engagement Rate by Weekday")
        plt.xlabel("Weekday (0=Mon, 6=Sun)")
        plt.ylabel("Engagement Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "engagement_by_weekday.png"))
        plt.close()

    print(f"\n🎉 EDA complete. Plots saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    run_eda()

