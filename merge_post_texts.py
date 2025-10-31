#!/usr/bin/env python3
"""
Merge collected post texts into the main LinkedIn dataset.

Input:
  - data/linkedin_posts_clean.csv
  - data/post_texts.csv

Output:
  - data/linkedin_posts_with_text.csv
"""

import pandas as pd
from pathlib import Path

# --- Configuration ---
MAIN_CSV = "data/linkedin_posts_clean.csv"
TEXTS_CSV = "data/post_texts.csv"
OUTPUT_CSV = "data/linkedin_posts_with_text.csv"


def main():
    print("📂 Loading main dataset:", MAIN_CSV)
    df_main = pd.read_csv(MAIN_CSV)

    print("📂 Loading post texts:", TEXTS_CSV)
    df_texts = pd.read_csv(TEXTS_CSV)

    # Basic sanity checks
    if "post_url" not in df_main.columns or "post_url" not in df_texts.columns:
        raise ValueError("❌ Both CSV files must contain a 'post_url' column!")

    if "post_text" not in df_texts.columns:
        raise ValueError("❌ The post_texts.csv file must have a 'post_text' column!")

    # Normalize URLs (remove trailing slashes or whitespace)
    df_main["post_url"] = df_main["post_url"].astype(str).str.strip().str.rstrip("/")
    df_texts["post_url"] = df_texts["post_url"].astype(str).str.strip().str.rstrip("/")

    # Merge
    df_merged = pd.merge(df_main, df_texts, on="post_url", how="left")

    # Count stats
    total_posts = len(df_merged)
    with_text = df_merged["post_text"].notna().sum()
    missing_text = total_posts - with_text

    print(f"✅ Merged {with_text}/{total_posts} posts with text.")
    if missing_text > 0:
        print(f"⚠️ {missing_text} posts are still missing text.")

    # Save
    Path("data").mkdir(exist_ok=True)
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Saved merged dataset to {OUTPUT_CSV}")

    # Quick preview
    print("\n🔍 Preview of merged data:")
    print(df_merged[["post_url", "post_text"]].head())


if __name__ == "__main__":
    main()

