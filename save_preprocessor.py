import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = "data/linkedin_posts_with_text.csv"
SAVE_PATH = "data/preprocessor.pkl"
os.makedirs("data", exist_ok=True)

print(f"📂 Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ dataset shape: {df.shape}")
print(f"columns: {list(df.columns)}")

TEXT_COL = "post_text" if "post_text" in df.columns else "text"
CATEGORICAL_FEATURE = "post_type"

possible_numeric_features = ["impressions", "reactions", "comments", "reposts", "saves", "likes"]
numeric_features = [f for f in possible_numeric_features if f in df.columns]

if CATEGORICAL_FEATURE not in df.columns:
    print(f"⚠️ '{CATEGORICAL_FEATURE}' column not found — defaulting all to 'Text'")
    df[CATEGORICAL_FEATURE] = "Text"

allowed_post_types = ["Text", "Image", "Video", "Poll", "Link", "Article", "Document"]
df[CATEGORICAL_FEATURE] = df[CATEGORICAL_FEATURE].apply(lambda x: x if x in allowed_post_types else "Text")

print(f"⚙️ Fitting StandardScaler on numeric features: {numeric_features}")
scaler = StandardScaler().fit(df[numeric_features])

print(f"⚙️ Fitting OneHotEncoder on categorical feature '{CATEGORICAL_FEATURE}' with allowed values: {allowed_post_types}")
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(pd.DataFrame({CATEGORICAL_FEATURE: allowed_post_types}))

encoder_feature_names = list(encoder.get_feature_names_out([CATEGORICAL_FEATURE]))
expected_dim = len(numeric_features) + len(encoder_feature_names) + 384

preprocessor = {
    "scaler": scaler,
    "encoder": encoder,
    "numeric_features": numeric_features,
    "categorical_feature": CATEGORICAL_FEATURE,
    "encoder_feature_names": encoder_feature_names,
    "allowed_post_types": allowed_post_types,
    "embedding_dim": 384,
    "expected_dim": expected_dim,
}

joblib.dump(preprocessor, SAVE_PATH)
print(f"✅ Saved preprocessor to {SAVE_PATH}")
print(f"🔢 Expected total feature dimension: {expected_dim}")

