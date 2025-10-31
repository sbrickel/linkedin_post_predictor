import os
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_PATH = "data/linkedin_posts_with_text.csv"
PREPROC_PATH = "data/preprocessor.pkl"
MODEL_PATH = "data/best_xgb_model.pkl"

print(f"📂 Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ dataset shape: {df.shape}")
print(f"columns: {list(df.columns)}")

TEXT_COL = "post_text" if "post_text" in df.columns else "text"
if TEXT_COL not in df.columns:
    raise ValueError(f"❌ No text column found. Columns: {list(df.columns)}")
print(f"✅ Using text column: {TEXT_COL}")

print(f"📂 Loading preprocessor: {PREPROC_PATH}")
preprocessor = joblib.load(PREPROC_PATH)

scaler = preprocessor["scaler"]
encoder = preprocessor["encoder"]
numeric_features = preprocessor["numeric_features"]
categorical_feature = preprocessor["categorical_feature"]

if categorical_feature not in df.columns:
    print(f"⚠️ '{categorical_feature}' not found — defaulting to 'Text'")
    df[categorical_feature] = "Text"

df[categorical_feature] = df[categorical_feature].apply(lambda x: x if x in preprocessor["allowed_post_types"] else "Text")

print("⚙️ Scaling numeric features...")
numeric_scaled = scaler.transform(df[numeric_features])

print("⚙️ Encoding categorical feature...")
categorical_encoded = encoder.transform(df[[categorical_feature]])

print("🔠 Computing text embeddings (may take a minute)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = embedder.encode(df[TEXT_COL].fillna(""), show_progress_bar=True)

X = np.hstack([numeric_scaled, categorical_encoded, text_embeddings])
y = df["engagement_rate"].values

print(f"✅ Final feature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training XGBoost model...")
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"✅ Model trained. RMSE={rmse:.4f}, R²={r2:.3f}")

joblib.dump(model, MODEL_PATH)
print(f"✅ Saved model to {MODEL_PATH}")

