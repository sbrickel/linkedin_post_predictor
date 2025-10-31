#!/usr/bin/env python3
"""
Module 4 — Text-Aware Modeling for LinkedIn Post Performance Prediction
----------------------------------------------------------------------
This script uses TF-IDF to convert post text into numeric features and
combines them with structured metadata (weekday, hour, etc.) to predict
engagement_rate using Ridge Regression.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
INPUT_CSV = "data/linkedin_posts_with_text.csv"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    print("📂 Loading dataset:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)

    # --- Basic data checks ---
    if "post_text" not in df.columns:
        raise ValueError("❌ No 'post_text' column found! Please merge texts first.")

    # Drop posts without text or engagement rate
    df = df.dropna(subset=["post_text", "engagement_rate"])
    print(f"✅ Using {len(df)} posts with text.")

    # --- Feature selection ---
    text_data = df["post_text"].astype(str)
    numeric_features = ["weekday", "hour", "month", "reactions", "comments", "reposts", "saves"]
    X_num = df[numeric_features].fillna(0).values
    y = df["engagement_rate"].values

    # --- Train/test split ---
    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        text_data, X_num, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --- TF-IDF Vectorization ---
    print("🔠 Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    print(f"✅ TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")

    # --- Combine numeric + text features ---
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    X_train_combined = hstack([X_train_tfidf, X_train_num_scaled])
    X_test_combined = hstack([X_test_tfidf, X_test_num_scaled])

    # --- Train model ---
    print("🤖 Training Ridge Regression model...")
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_combined, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test_combined)

    mae = mean_absolute_error(y_test, y_pred)
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Performance (TF-IDF + Ridge):")
    print(f"   MAE : {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   R²  : {r2:.4f}")

    # --- Save metrics ---
    metrics_df = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2}])
    metrics_df.to_csv(OUTPUT_DIR / "model_tfidf_metrics.csv", index=False)

    # --- Plot predictions ---
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("True Engagement Rate")
    plt.ylabel("Predicted Engagement Rate")
    plt.title("TF-IDF + Ridge Regression Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_tfidf_predictions.png")
    plt.close()
    print("📈 Saved prediction plot to data/model_tfidf_predictions.png")

    # --- Feature importance (optional, most informative words) ---
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = model.coef_[:len(feature_names)]  # only TF-IDF part
    top_idx = np.argsort(coefs)[-20:]
    top_features = feature_names[top_idx]
    top_values = coefs[top_idx]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_values, y=top_features, palette="viridis")
    plt.title("Top 20 Words Correlated with Higher Engagement")
    plt.xlabel("Model Coefficient (importance)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_tfidf_top_words.png")
    plt.close()
    print("📈 Saved top-word importance plot to data/model_tfidf_top_words.png")

    print("\n🎉 Text-aware modeling complete! Metrics and plots saved to 'data/'.")


if __name__ == "__main__":
    main()

