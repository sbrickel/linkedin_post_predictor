import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
from sentence_transformers import SentenceTransformer

# ===========================
# 1️⃣ LOAD MODEL & PREPROCESSOR
# ===========================
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("data/preprocessor.pkl")
    model = joblib.load("data/best_xgb_model.pkl")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return preprocessor, model, embedder

preprocessor, model, embedder = load_assets()

numeric_features = preprocessor["numeric_features"]
categorical_feature = preprocessor["categorical_feature"]
expected_dim = preprocessor["expected_dim"]

# ===========================
# 2️⃣ STREAMLIT UI
# ===========================
st.set_page_config(page_title="LinkedIn Post Engagement Predictor", layout="centered")
st.title("📈 LinkedIn Post Engagement Predictor")

st.markdown(
    """
    Paste your LinkedIn post below to predict its **engagement rate** 🔮  
    You'll see which **post attributes and words** drive your prediction.
    """
)

post_text = st.text_area("✍️ Paste your LinkedIn post text here:", height=200)

post_type = st.selectbox("🗂️ Post type:", ["Text", "Image", "Video", "Link", "Poll"])

st.markdown("### 📊 Post context (numeric features)")
numeric_inputs = {}
cols = st.columns(len(numeric_features))
for i, feature in enumerate(numeric_features):
    with cols[i]:
        numeric_inputs[feature] = st.number_input(f"{feature.title()}", min_value=0.0, value=0.0)

# ===========================
# 3️⃣ PREPROCESS INPUT
# ===========================
def preprocess_input(post_text, numeric_inputs, post_type, preprocessor, embedder):
    numeric_df = pd.DataFrame([numeric_inputs])
    scaled = preprocessor["scaler"].transform(numeric_df)

    encoder = preprocessor["encoder"]
    encoded = encoder.transform(np.array([[post_type]]))

    text_emb = embedder.encode([post_text])
    X = np.concatenate([scaled, encoded, text_emb], axis=1)
    return X

# ===========================
# 4️⃣ PREDICT
# ===========================
if st.button("🚀 Predict Engagement"):
    if not post_text.strip():
        st.warning("Please paste your post text first.")
        st.stop()

    try:
        features = preprocess_input(post_text, numeric_inputs, post_type, preprocessor, embedder)
        if features.shape[1] != expected_dim:
            st.error(f"❌ Feature dimension mismatch: expected {expected_dim}, got {features.shape[1]}")
            st.stop()

        prediction = model.predict(features)[0]
        st.success(f"🎯 Predicted engagement rate: **{prediction:.2f}%**")

        # ===========================
        # 5️⃣ SHAP EXPLANATION (NUMERIC + CATEGORY)
        # ===========================
        st.markdown("---")
        st.markdown("### 🧩 Feature Impact on Prediction")

        encoder = preprocessor["encoder"]
        cat_names = encoder.get_feature_names_out([categorical_feature])
        all_feature_names = numeric_features + list(cat_names)

        # Compute SHAP values for tabular part
        explainer = shap.Explainer(model)
        shap_values = explainer(features)

        feature_shap_df = pd.DataFrame({
            "Feature": all_feature_names,
            "SHAP Value": shap_values.values[0][:len(all_feature_names)]
        }).sort_values("SHAP Value", key=abs, ascending=False)

        st.write(feature_shap_df.style.background_gradient(cmap="RdYlGn"))

        # ===========================
        # 6️⃣ WORD-LEVEL INFLUENCE ANALYSIS
        # ===========================
        st.markdown("---")
        st.markdown("### 🧠 Word-Level Influence Analysis")

        words = post_text.split()
        if len(words) < 3:
            st.info("Write a longer post to analyze individual word effects.")
            st.stop()

        # Build word-level embeddings
        word_embs = embedder.encode(words)
        base_tabular = np.concatenate(
            [preprocessor["scaler"].transform(pd.DataFrame([numeric_inputs])),
             encoder.transform(np.array([[post_type]]))],
            axis=1
        )

        repeated_tabular = np.repeat(base_tabular, len(words), axis=0)
        X_word_level = np.concatenate([repeated_tabular, word_embs], axis=1)

        word_preds = model.predict(X_word_level)
        avg_pred = np.mean(word_preds)
        contributions = word_preds - avg_pred

        max_abs = np.max(np.abs(contributions)) or 1
        colored_text = ""
        for word, contrib in zip(words, contributions):
            color = "green" if contrib > 0 else "red"
            opacity = 0.25 + 0.75 * (abs(contrib) / max_abs)
            colored_text += f"<span style='background-color:{color};opacity:{opacity};padding:2px 4px;border-radius:3px'>{word}</span> "

        st.markdown(colored_text, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

