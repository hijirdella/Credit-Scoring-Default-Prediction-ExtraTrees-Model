import io
from io import BytesIO
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from credit_scoring_utils import (
    build_customer_features_from_combined,
    score_customers,
    build_decile_table,
    find_acceptance_cut_for_target_default_rate,
    extract_feature_importance,
)

MODEL_PATH = "best_credit_scoring_extratrees.pkl"
ORANGE = "#FF9800"
DARK_GREEN = "#2E7D32"
TABLE_HEIGHT = 300
FIG_W = 4
FIG_H = 3
HEATMAP_CMAP = LinearSegmentedColormap.from_list("green_orange", [DARK_GREEN, "white", ORANGE])

REQUIRED_COLUMNS = [
    "cdate", "application_id", "customer_id", "loan_purpose",
    "loan_purpose_desc", "dob", "address_provinsi", "marital_status",
    "dependent", "job_type", "job_industry", "loan_id", "loan_amount",
    "loan_duration", "installment_amount", "fund_transfer_ts", "payment_id",
    "due_date", "paid_date", "paid_amount", "dpd"
]

st.set_page_config(
    page_title="Credit Default Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def load_model(model_path: str):
    model = joblib.load(model_path)
    return model

def cast_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["application_id", "customer_id", "loan_id", "payment_id"]
    for col in id_cols:
        if col in df.columns:
            def _to_str(x):
                if pd.isna(x):
                    return ""
                try:
                    return str(int(x))
                except Exception:
                    return str(x)
            df[col] = df[col].apply(_to_str).astype("string")
    return df

def add_bar_labels(ax, fmt="{:.0f}", skip_zero=True, eps=1e-9):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h) or (skip_zero and abs(h) < eps):
            continue
        ax.annotate(fmt.format(h),
                    (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8)

def plot_target_distribution(df):
    if "default_flag_customer" not in df.columns:
        st.info("No target column 'default_flag_customer'.")
        return
    counts = df["default_flag_customer"].value_counts().sort_index()
    labels = ["Non-default", "Default"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(x=labels, y=values, ax=ax, palette=[DARK_GREEN, ORANGE])
    ax.set_title("Customer Default Distribution")
    ax.set_xlabel("Default Flag")
    ax.set_ylabel("Count")
    add_bar_labels(ax)
    st.pyplot(fig)

def plot_default_rate_by_category(df, col, max_categories=8):
    if col not in df.columns or "default_flag_customer" not in df.columns:
        return
    top = df[col].value_counts().head(max_categories).index
    tmp = df[df[col].isin(top)].copy()
    rate_df = (
        tmp.groupby(col)["default_flag_customer"]
        .mean()
        .reset_index()
        .rename(columns={"default_flag_customer": "default_rate"})
        .sort_values("default_rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(x=col, y="default_rate", data=rate_df, color=ORANGE, ax=ax)
    ax.set_title(f"Default Rate by {col}")
    ax.set_ylabel("Default Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    add_bar_labels(ax, fmt="{:.1%}")
    st.pyplot(fig)

def plot_pd_histogram(scored_df):
    if "pd" not in scored_df.columns:
        return
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    counts, _, patches = ax.hist(scored_df["pd"], bins=20, color=ORANGE, edgecolor="k")
    ax.set_title("Predicted PD Distribution")
    ax.set_xlabel("PD")
    ax.set_ylabel("Count")
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, count,
                    f"{int(count)}", ha="center", va="bottom", fontsize=7)
    st.pyplot(fig)

def plot_correlation_heatmap(df_fe, max_features=15):
    num_cols = [c for c in df_fe.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]
    if len(num_cols) < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return
    if len(num_cols) > max_features:
        num_cols = num_cols[:max_features]
    corr = df_fe[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, cmap=HEATMAP_CMAP, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap (Customer-level numeric features)")
    st.pyplot(fig)

def main():
    st.title("Credit Default Prediction Dashboard")
    st.markdown(
        "This interactive dashboard demonstrates a **machine learning–based credit risk model** "
        "built using **loan, payment, and customer data**."
    )

    st.markdown("### Input Data")
    uploaded = st.file_uploader(
        "Upload input CSV (same structure as combined_df.csv)",
        type=["csv"],
        help="Any CSV filename is allowed as long as its structure matches combined_df.csv.",
    )

    with st.expander("App Flow", expanded=False):
        st.markdown(
            """
1. Upload your **CSV file** containing raw loan–payment–customer data  
   (any filename is accepted as long as the structure matches **combined_df.csv**).
2. The app performs **Exploratory Data Analysis (EDA)** on the uploaded dataset.
3. Data is aggregated automatically to **customer level** using predefined logic.
4. The pre-trained **ExtraTrees model** predicts the **Probability of Default (PD)** for each customer.
5. You can explore **feature importance, correlation heatmap, business insights**,  
   and **download the scored customers** (CSV / Excel).
            """
        )

    with st.expander("Important Note & Data Assumptions", expanded=False):
        st.markdown(
            """
This dashboard runs on **raw, uncleaned data** to test model performance on  
real-world, unprocessed inputs.  

All data cleaning, outlier handling, and feature engineering were done  
during **training, validation, and testing** in Python (offline).  

Hence, this dashboard is designed for **model inference and business insight simulation**,  
not for data preprocessing or retraining.
            """
        )
        st.markdown(
            "**Required columns in the input CSV:**  \n"
            + "`" + ", ".join(REQUIRED_COLUMNS) + "`"
        )

    with st.expander("Created by: Hijir Della Wirasti", expanded=False):
        st.markdown(
            """
[Website](https://www.hijirdata.com/)  
[Email](mailto:hijirdw@gmail.com)  
[LinkedIn](https://www.linkedin.com/in/hijirdella/)
            """
        )

    if uploaded is None:
        st.info("Please upload a CSV file first to begin.")
        return

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    raw_df = cast_id_columns(raw_df)
    st.subheader("1. Raw Input Preview")
    st.write(f"Rows: {raw_df.shape[0]}, Columns: {raw_df.shape[1]}")
    st.dataframe(raw_df.head(20))

    missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing:
        st.warning("Missing expected columns: " + ", ".join(missing))

    st.subheader("2. Exploratory Data Analysis (EDA)")
    num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = raw_df.select_dtypes(include=["object", "string"]).columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric Overview**")
        if num_cols:
            st.dataframe(raw_df[num_cols].describe().T, height=TABLE_HEIGHT)
            sel_num = st.selectbox("Numeric column for histogram", num_cols)
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            ax.hist(raw_df[sel_num].dropna(), bins=30, color=ORANGE, edgecolor="k")
            ax.set_title(f"Histogram of {sel_num}")
            ax.set_xlabel(sel_num)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No numeric columns detected.")
    with col2:
        st.markdown("**Categorical Overview**")
        if cat_cols:
            rows = []
            for c in cat_cols:
                vc = raw_df[c].value_counts(dropna=False)
                rows.append({
                    "Column": c,
                    "Unique Values": len(vc),
                    "Top Category": vc.index[0] if len(vc) > 0 else None,
                    "Count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                })
            st.dataframe(pd.DataFrame(rows), height=TABLE_HEIGHT)
        else:
            st.info("No categorical columns detected.")

    st.subheader("3. Customer-level Feature Engineering")
    try:
        df_fe = build_customer_features_from_combined(raw_df)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        return
    st.write(f"Customer-level shape: {df_fe.shape}")
    st.dataframe(df_fe.head(20))

    st.subheader("4. Correlation Heatmap")
    plot_correlation_heatmap(df_fe)

    st.subheader("5. Model Prediction")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    run_pred = st.button("Run Prediction", type="primary")
    if not run_pred:
        st.info("Click **Run Prediction** to generate PD scores.")
        return

    scored_df = score_customers(model, df_fe)
    scored_full = df_fe.merge(scored_df[["customer_id", "pd"]], on="customer_id", how="left")
    st.write("Sample of scored customers:")
    st.dataframe(scored_full.head(20))
    plot_pd_histogram(scored_df)

    dec_table = build_decile_table(scored_full[["customer_id", "pd"]], n_deciles=10)
    st.subheader("Decile Summary (1 = lowest PD, 10 = highest PD)")
    st.dataframe(dec_table)

    out_cols = ["customer_id", "pd"]
    out_df = scored_full[out_cols].copy()
    base = st.text_input("Output file name (without extension)", value="scored_customers").strip() or "scored_customers"
    fmt = st.selectbox("Output format", ["CSV", "Excel (.xlsx)"])

    if fmt == "CSV":
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        st.download_button(
            "Download CSV",
            buf.getvalue().encode("utf-8"),
            f"{base}.csv",
            "text/csv",
        )
    else:
        xbuf = BytesIO()
        with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
            out_df.to_excel(writer, index=False, sheet_name="scored_customers")
        st.download_button(
            "Download Excel",
            xbuf.getvalue(),
            f"{base}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

if __name__ == "__main__":
    main()
