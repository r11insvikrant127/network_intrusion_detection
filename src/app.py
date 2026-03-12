import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_features, scale_features
from feature_engineering import apply_pca

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🔐 Network Intrusion Detection Dashboard",
    layout="wide"
)

st.title("🔐 Network Intrusion Detection System")
st.caption("Model Used: SVM_RBF | Dataset: UNSW-NB15")

st.markdown("""
### 🔎 Overview

This dashboard detects malicious network traffic using a trained **Machine Learning model**.

The model analyzes network flow features and classifies traffic as:

- 🟢 **Normal Traffic**
- 🔴 **Attack Traffic**

📌 **How to use this dashboard**

1. Upload a **network traffic CSV file**
2. The model will classify each record
3. View prediction summaries and dataset insights
""")

# -----------------------------
# Sidebar Upload
# -----------------------------
st.sidebar.header("📤 Upload Network Traffic Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/best_model.pkl")

# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # X = preprocess_features_for_prediction(df)
    # X = apply_pca_for_prediction(X)

   # If label exists, separate it
    if "label" in df.columns:
        X = df.drop("label", axis=1)
        y = df["label"]
    else:
        X = df

    try:

        # -----------------------------
        # Preprocessing
        # -----------------------------
        X_train, X_test, preprocessor = preprocess_features(X, X)

        # -----------------------------
        # Scaling
        # -----------------------------
        X_train, X_test, scaler = scale_features(X_train, X_test)

        # -----------------------------
        # PCA
        # -----------------------------
        X_train_pca, X_test_pca, pca = apply_pca(
            X_train,
            X_test,
            n_components=20
        )

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = model.predict(X_test_pca)

        results = df.copy()
        results["Prediction"] = predictions
        results["Prediction"] = results["Prediction"].map(
            {0: 0, 1: 1}
        )

        st.subheader("🧠 Prediction Results")
        st.dataframe(results)

        st.caption("* 0 = Normal, 1 = Attack")

        # Summary
        attack_count = (results["Prediction"] == 1).sum()
        normal_count = (results["Prediction"] == 0).sum()

        col1, col2 = st.columns(2)

        col1.metric("Normal Traffic", normal_count)
        col2.metric("Attack Traffic", attack_count)

    except Exception as e:
        st.error(f"Prediction Error: {e}")


# -----------------------------
# Dataset Analysis
# -----------------------------
st.markdown("---")
st.header("📊 Dataset & Model Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Class Distribution Before SMOTE")
    st.image("outputs/before_smote.png")

with col2:
    st.subheader("Class Distribution After SMOTE")
    st.image("outputs/after_smote.png")

st.subheader("Correlation Heatmap")
st.image("outputs/correlation_heatmap.png")

st.subheader("Feature Importance")
st.image("outputs/feature_importance.png")

# -----------------------------
# Model Evaluation
# -----------------------------
st.subheader("📊 Model Performance Comparison")

st.markdown("""
The chart below compares different machine learning models used for
network intrusion detection.

Metric used:

- **F1 Score** — balances precision and recall
- Higher values indicate better performance.

📌 This comparison helps select the best model for deployment.
""")

comparison_img = Image.open("outputs/model_comparison.png")

st.image(
    comparison_img,
    caption="Model Comparison (F1 Score)",
    width="stretch"
)

# -----------------------------
# Best Model Selected
# -----------------------------
st.subheader("🏆 Best Model Selected")

st.markdown("""
Based on the evaluation of multiple machine learning models using the **F1 Score**,  
the model with the best performance on the dataset is:

### 🧠 **Support Vector Machine (RBF Kernel)**

This model achieved the **highest F1 Score**, meaning it provides the best balance between:

- **Precision** – correctly identifying attacks
- **Recall** – detecting as many attacks as possible

📌 **Why SVM with RBF Kernel works well:**
- Handles **non-linear decision boundaries**
- Performs well on **high-dimensional datasets**
- Effective for **network intrusion detection patterns**

Therefore, the deployed model used in this dashboard is:

### 🚀 **SVM_RBF**
""")

st.success("✔ Active Detection Model: SVM_RBF")