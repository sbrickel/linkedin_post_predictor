# model_basics.py
"""
Module 3: Modeling Basics
Baseline models for predicting LinkedIn engagement rate.

Steps:
1. Load preprocessed CSV (from process_linkedin_data.py).
2. Prepare features and target.
3. Split into train/test sets.
4. Train Linear Regression and Decision Tree baselines.
5. Evaluate performance and visualize results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Ensure output folder exists
os.makedirs("data", exist_ok=True)

# -----------------------------------------------------------------------------
# 1️⃣ Load data
# -----------------------------------------------------------------------------
print("📂 Loading cleaned data...")
df = pd.read_csv("data/linkedin_posts_clean.csv")

# drop rows missing target or key features
df = df.dropna(subset=["engagement_rate"])

print(f"✅ Data loaded: {len(df)} posts, {df.shape[1]} columns")

# -----------------------------------------------------------------------------
# 2️⃣ Feature selection
# -----------------------------------------------------------------------------
# Basic numeric and temporal features for baseline model
feature_cols = ["weekday", "hour", "month", "impressions"]
X = df[feature_cols]
y = df["engagement_rate"]

# Fill missing or invalid numeric values with 0
X = X.fillna(0)

# -----------------------------------------------------------------------------
# 3️⃣ Train/Test Split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"🧩 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# -----------------------------------------------------------------------------
# 4️⃣ Train Baseline Models
# -----------------------------------------------------------------------------
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

# -----------------------------------------------------------------------------
# 5️⃣ Evaluation Function
# -----------------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {name} Performance:")
    print(f"   MAE : {mae:.5f}")
    print(f"   RMSE: {rmse:.5f}")
    print(f"   R²  : {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

lr_metrics = evaluate_model("Linear Regression", y_test, lr_preds)
tree_metrics = evaluate_model("Decision Tree", y_test, tree_preds)

# -----------------------------------------------------------------------------
# 6️⃣ Visualization
# -----------------------------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, lr_preds, alpha=0.7, label="Linear Regression", color="steelblue")
plt.scatter(y_test, tree_preds, alpha=0.7, label="Decision Tree", color="tomato")
plt.plot([0, max(y_test)], [0, max(y_test)], "k--", lw=1)
plt.xlabel("Actual Engagement Rate")
plt.ylabel("Predicted Engagement Rate")
plt.title("Actual vs Predicted Engagement Rate")
plt.legend()
plt.tight_layout()
plot_path = "data/model_baseline_predictions.png"
plt.savefig(plot_path)
plt.close()
print(f"📈 Saved prediction plot to {plot_path}")

# -----------------------------------------------------------------------------
# 7️⃣ Summary Table
# -----------------------------------------------------------------------------
summary = pd.DataFrame([lr_metrics, tree_metrics], index=["LinearRegression", "DecisionTree"])
print("\n🔍 Summary of model performance:")
print(summary)

summary.to_csv("data/model_baseline_metrics.csv")
print("✅ Metrics saved to data/model_baseline_metrics.csv")

print("\n🎉 Baseline modeling complete.")

