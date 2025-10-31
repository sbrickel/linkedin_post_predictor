#!/usr/bin/env python3
"""
Model tuning and validation for LinkedIn post performance prediction.

This script:
- Loads dataset with post texts and metadata
- Generates transformer embeddings
- Combines text embeddings with numeric features
- Trains and tunes multiple regression models with cross-validation
- Saves metrics and comparison plots to 'data/'
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/linkedin_posts_with_text.csv"
OUTPUT_METRICS = "data/model_tuning_results.csv"
OUTPUT_PLOT = "data/model_tuning_comparison.png"

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# ---------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

if "post_text" not in df.columns:
    raise ValueError("❌ Missing 'post_text' column. Please merge texts first.")

# Drop rows without post text
df = df.dropna(subset=["post_text"])
print(f"✅ Using {len(df)} posts with text.")

# Target variable
y = df["engagement_rate"].values

# ---------------------------------------------------
# 2. Generate text embeddings
# ---------------------------------------------------
print("🔠 Loading transformer model (MiniLM)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("⚙️  Computing embeddings...")
embeddings = model.encode(df["post_text"].tolist(), show_progress_bar=True)
print(f"✅ Embeddings shape: {embeddings.shape}")

# ---------------------------------------------------
# 3. Combine text embeddings + numeric features
# ---------------------------------------------------
numeric_features = ["weekday", "month", "hour", "impressions", "reactions", "comments", "reposts", "saves"]
X_numeric = df[numeric_features].fillna(0).values

X = np.hstack([embeddings, X_numeric])
print(f"✅ Combined feature matrix shape: {X.shape}")

# ---------------------------------------------------
# 4. Define models & hyperparameter grids
# ---------------------------------------------------
models = {
    "Ridge": (Ridge(), {"ridge__alpha": [0.1, 1.0, 10.0]}),

    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {
            "randomforest__n_estimators": [100, 300],
            "randomforest__max_depth": [5, 10, None],
        },
    ),

    "XGBoost": (
        XGBRegressor(random_state=42, eval_metric="rmse"),
        {
            "xgboost__learning_rate": [0.05, 0.1],
            "xgboost__max_depth": [3, 6, 10],
            "xgboost__n_estimators": [100, 300],
        },
    ),

    "MLP": (
        MLPRegressor(max_iter=500, random_state=42),
        {
            "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "mlp__alpha": [0.0001, 0.001, 0.01],
        },
    ),
}

# ---------------------------------------------------
# 5. Cross-validation & grid search
# ---------------------------------------------------
results = []

for name, (model, param_grid) in models.items():
    print(f"\n🤖 Tuning {name} model...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        (name.lower(), model),
    ])

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    mean_r2 = grid.best_score_

    # Cross-validation metrics
    mae = -np.mean(cross_val_score(best_model, X, y, cv=5, scoring="neg_mean_absolute_error"))
    rmse = np.sqrt(-np.mean(cross_val_score(best_model, X, y, cv=5, scoring="neg_mean_squared_error")))

    results.append({
        "model": name,
        "best_params": grid.best_params_,
        "R²": mean_r2,
        "MAE": mae,
        "RMSE": rmse,
    })

    print(f"✅ Best {name} params: {grid.best_params_}")
    print(f"   R²={mean_r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

# ---------------------------------------------------
# 6. Save results
# ---------------------------------------------------
results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False)
results_df.to_csv(OUTPUT_METRICS, index=False)
print(f"\n✅ Results saved to {OUTPUT_METRICS}")

# ---------------------------------------------------
# 7. Plot model comparison
# ---------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="model", y="R²", palette="viridis")
plt.title("Model R² Comparison (Cross-Validation)")
plt.ylabel("Mean R² (5-fold CV)")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.close()

print(f"📈 Comparison plot saved to {OUTPUT_PLOT}")
print("\n🎉 Model tuning and validation complete!")

