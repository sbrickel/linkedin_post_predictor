#!/usr/bin/env python3
"""
Module 5: Advanced Modeling — Transformer Embeddings
----------------------------------------------------
This script builds a semantic-level LinkedIn Post Performance Predictor
using pretrained transformer embeddings from the 'sentence-transformers' library.

Steps:
1. Load the processed dataset with text.
2. Compute contextual embeddings using a pretrained model.
3. Train a Ridge Regression model to predict engagement rate.
4. Evaluate performance and visualize predictions.

Dependencies:
    pip install sentence-transformers scikit-learn pandas numpy matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ========== CONFIG ==========
DATA_PATH = "data/linkedin_posts_with_text.csv"
OUT_DIR = "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42
TEST_SIZE = 0.2
# =============================


def main():
    print(f"📂 Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if "post_text" not in df.columns:
        raise ValueError("❌ 'post_text' column not found in dataset. Please merge text data first.")

    # Drop rows without text or engagement data
    df = df.dropna(subset=["post_text", "engagement_rate"]).reset_index(drop=True)
    print(f"✅ Using {len(df)} posts with text.")

    # ========== EMBEDDING ==========
    print(f"🔠 Loading transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("⚙️  Computing embeddings (this may take a minute)...")
    embeddings = []
    for text in tqdm(df["post_text"], desc="Encoding posts"):
        embeddings.append(model.encode(str(text), show_progress_bar=False))
    X = np.vstack(embeddings)
    y = df["engagement_rate"].values

    print(f"✅ Embeddings shape: {X.shape}")

    # ========== MODELING ==========
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print("🤖 Training Ridge Regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ========== EVALUATION ==========
    mae = mean_absolute_error(y_test, y_pred)
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Performance (Transformer Embeddings + Ridge):")
    print(f"   MAE : {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   R²  : {r2:.4f}")

    # ========== VISUALIZATION ==========
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.8)
    plt.xlabel("True Engagement Rate")
    plt.ylabel("Predicted Engagement Rate")
    plt.title("Predicted vs True Engagement (Transformer Embeddings)")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    out_path = os.path.join(OUT_DIR, "model_transformer_predictions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Saved prediction plot to {out_path}")

    # Save metrics
    metrics_path = os.path.join(OUT_DIR, "model_transformer_metrics.csv")
    pd.DataFrame([{
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }]).to_csv(metrics_path, index=False)
    print(f"✅ Metrics saved to {metrics_path}")

    print("\n🎉 Transformer-based modeling complete! You’re now using semantic understanding 🚀")


if __name__ == "__main__":
    main()

