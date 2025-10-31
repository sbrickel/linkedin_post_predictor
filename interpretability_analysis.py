# interpretability_analysis.py
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1. Load model and data
# ----------------------------
model = joblib.load("data/best_xgboost_model.pkl")
df = pd.read_csv("data/linkedin_posts_with_text.csv")

# ----------------------------
# 2. Prepare features
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(df["post_text"].fillna("").tolist(), show_progress_bar=True)
numeric = df[["reactions", "comments", "reposts", "saves", "impressions"]].fillna(0).to_numpy()
X = np.hstack([embeddings, numeric])

# ----------------------------
# 3. Compute SHAP values
# ----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ----------------------------
# 4. Global summary plot
# ----------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False, plot_type="bar")
plt.title("Global Feature Importance (SHAP Values)")
plt.tight_layout()
plt.savefig("data/global_shap_importance.png")
plt.show()

print("✅ Global SHAP plot saved to data/global_shap_importance.png")

