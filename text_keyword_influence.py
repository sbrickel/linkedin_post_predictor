# text_keyword_influence.py
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

# ----------------------------
# 1. Load data
# ----------------------------
df = pd.read_csv("data/linkedin_posts_with_text.csv")
df = df[df["post_text"].notna() & (df["engagement_rate"] > 0)]

texts = df["post_text"].tolist()
y = df["engagement_rate"].values

# ----------------------------
# 2. Vectorize text
# ----------------------------
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words="english"
)
X = tfidf.fit_transform(texts)

# ----------------------------
# 3. Train simple interpretable model
# ----------------------------
model = Ridge(alpha=1.0)
model.fit(X, y)

# ----------------------------
# 4. Get feature importances
# ----------------------------
feature_names = np.array(tfidf.get_feature_names_out())
coefficients = model.coef_

# Sort and select top/bottom 20
top_n = 20
top_positive = np.argsort(coefficients)[-top_n:]
top_negative = np.argsort(coefficients)[:top_n]

# ----------------------------
# 5. Plot keyword influence
# ----------------------------
plt.figure(figsize=(10, 6))
plt.barh(feature_names[top_positive], coefficients[top_positive], color="green")
plt.barh(feature_names[top_negative], coefficients[top_negative], color="red")
plt.title("Keyword Influence on Engagement Rate")
plt.xlabel("Effect on predicted engagement")
plt.tight_layout()
plt.savefig("data/keyword_influence_plot.png")
plt.show()

print("✅ Keyword influence plot saved to data/keyword_influence_plot.png")

# Save data for Streamlit display
pd.DataFrame({
    "keyword": feature_names,
    "influence": coefficients
}).to_csv("data/keyword_influence.csv", index=False)

print("✅ Keyword influence CSV saved to data/keyword_influence.csv")

