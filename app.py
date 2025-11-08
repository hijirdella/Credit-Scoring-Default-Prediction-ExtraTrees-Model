# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from credit_scoring_utils import build_customer_features_from_combined


@st.cache_resource
def load_model(model_path: str = "best_credit_scoring_extratrees.pkl"):
    return joblib.load(model_path)


st.title("Credit Default Prediction Dashboard")
st.write("Upload combined loan–payment–customer data (CSV) to score default risk at customer level.")


# File uploader
uploaded_file = st.file_uploader("Upload combined_df CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV (preserve IDs as strings if columns exist)
    try:
        df_raw = pd.read_csv(
            uploaded_file,
            dtype={
                "application_id": str,
                "customer_id": str,
                "loan_id": str,
                "payment_id": str,
            },
        )
    except Exception:
        df_raw = pd.read_csv(uploaded_file)

    st.subheader("Raw Input Preview")
    st.dataframe(df_raw.head())

    # Build customer-level features
    df_features = build_customer_features_from_combined(df_raw)

    st.subheader("Customer-Level Feature Preview")
    st.dataframe(df_features.head())

    # Load ExtraTrees model pipeline
    model = load_model()

    # Prepare X for model (drop ID and target if present)
    X = df_features.drop(columns=["customer_id", "default_flag_customer"], errors="ignore").copy()

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64", "Int64", "float32"]).columns
    cat_cols = X.columns.difference(num_cols)

    # Simple imputation – must be consistent with training preprocessing pipeline
    # (This is a safety layer; main preprocessing is inside the pipeline.)
    for col in num_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    for col in cat_cols:
        X[col] = X[col].fillna("missing")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict PD (probability of default)
    if hasattr(model.named_steps["model"], "predict_proba"):
        pd_score = model.predict_proba(X)[:, 1]
    else:
        s = model.decision_function(X)
        pd_score = (s - s.min()) / (s.max() - s.min() + 1e-9)

    # Threshold slider
    st.subheader("Scoring Parameters")
    threshold = st.slider(
        "Default classification threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    df_features["pd_score"] = pd_score
    df_features["pred_default_flag"] = (df_features["pd_score"] >= threshold).astype(int)

    # Summary
    total = len(df_features)
    n_default = int(df_features["pred_default_flag"].sum())
    pct_default = round(n_default / total * 100, 2) if total > 0 else 0.0

    st.subheader("Scoring Summary")
    st.write(f"Total customers scored: **{total}**")
    st.write(f"Predicted defaults: **{n_default}**")
    st.write(f"Predicted default rate: **{pct_default}%**")

    # Output for business / download
    cols_out = ["customer_id", "default_flag_customer", "pd_score", "pred_default_flag"]
    cols_out = [c for c in cols_out if c in df_features.columns]
    scored_df = df_features[cols_out].copy()

    st.subheader("Scored Customers (Sample)")
    st.dataframe(scored_df.head())

    # Download button
    csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored customers CSV",
        data=csv_bytes,
        file_name="scored_customers_extratrees.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload a combined_df-style CSV to start scoring.")

