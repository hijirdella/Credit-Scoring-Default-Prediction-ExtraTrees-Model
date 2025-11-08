import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from credit_scoring_utils import (
    build_customer_features_from_combined,
    score_customers,
    build_decile_table,
    find_acceptance_cut_for_target_default_rate,
    extract_feature_importance,
)

MODEL_PATH = "best_credit_scoring_extratrees.pkl"

st.set_page_config(
    page_title="Credit Default Prediction Dashboard",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: str):
    """
    Load trained model pipeline (ExtraTrees + preprocessing + SMOTE) from disk.
    """
    model = joblib.load(model_path)
    return model


def plot_target_distribution(df_fe: pd.DataFrame):
    if "default_flag_customer" not in df_fe.columns:
        st.info("Target column 'default_flag_customer' is not available for this dataset.")
        return

    target_counts = df_fe["default_flag_customer"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        x=target_counts.index.map({0: "Non-default", 1: "Default"}),
        y=target_counts.values,
        ax=ax,
    )
    ax.set_title("Customer-level Default Distribution")
    ax.set_xlabel("Default Flag")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)


def plot_default_rate_by_category(df_fe: pd.DataFrame, col: str, max_categories: int = 8):
    if col not in df_fe.columns:
        st.warning(f"Column '{col}' not found.")
        return
    if "default_flag_customer" not in df_fe.columns:
        st.info("Target column 'default_flag_customer' is not available for this dataset.")
        return

    # Take top categories by count
    top_cats = df_fe[col].value_counts().head(max_categories).index
    tmp = df_fe[df_fe[col].isin(top_cats)].copy()

    rate_df = (
        tmp.groupby(col)["default_flag_customer"]
        .mean()
        .reset_index()
        .rename(columns={"default_flag_customer": "default_rate"})
        .sort_values("default_rate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(data=rate_df, x=col, y="default_rate", ax=ax)
    ax.set_title(f"Default Rate by {col}")
    ax.set_ylabel("Default Rate")
    ax.set_xlabel(col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)


def plot_pd_histogram(scored_df: pd.DataFrame):
    if "pd" not in scored_df.columns:
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(scored_df["pd"], bins=20, edgecolor="k")
    ax.set_title("Predicted PD Distribution")
    ax.set_xlabel("PD")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)


def render_business_insights(scored_df: pd.DataFrame, dec_table: pd.DataFrame):
    st.subheader("Business Questions")

    has_target = "default_flag_customer" in scored_df.columns

    # 1. Which customers/loans should we avoid? Why?
    st.markdown("**1. Which customers/loans should we avoid? Why?**")

    if has_target and "cum_default_rate" in dec_table.columns:
        # Define high-risk segment as worst 2 deciles (9–10)
        high_risk_deciles = dec_table["decile"].max() - 1
        worst_deciles = dec_table[dec_table["decile"] >= high_risk_deciles]

        # Aggregate information for explanation
        total_customers = scored_df.shape[0]
        high_risk_customers = scored_df[
            scored_df["decile"].isin(worst_deciles["decile"])
        ]
        share_portfolio = len(high_risk_customers) / total_customers

        if has_target:
            share_defaults = (
                high_risk_customers["default_flag_customer"].sum()
                / scored_df["default_flag_customer"].sum()
            )
        else:
            share_defaults = np.nan

        st.write(
            "- Customers in the **highest PD deciles** (e.g., deciles 9–10) "
            "should be avoided or priced very conservatively."
        )
        st.write(
            f"- These segments represent around "
            f"{share_portfolio:.1%} of the portfolio but capture approximately "
            f"{share_defaults:.1%} of historical defaults (based on the training data)."
        )
        st.write(
            "- Their predicted PD is significantly higher than the portfolio average, "
            "indicating poor risk–return trade-off."
        )
    else:
        st.write(
            "- Customers with the **highest predicted PDs** should be avoided or "
            "priced conservatively, as they contribute disproportionately to risk."
        )

    # 2. If the business targets a 2% cumulative default rate, which loans should we accept?
    st.markdown(
        "**2. If the business would like to achieve a 2% cumulative default rate, "
        "which loans should we accept?**"
    )

    if has_target and "cum_default_rate" in dec_table.columns:
        cut_decile = find_acceptance_cut_for_target_default_rate(
            decile_table=dec_table,
            target_cum_default_rate=0.02,
        )

        if cut_decile is None:
            st.write(
                "- With the current portfolio and model, a 2% cumulative default "
                "target cannot be achieved even if we only approve the very best customers."
            )
        else:
            acc_row = dec_table[dec_table["decile"] == cut_decile].iloc[0]
            accept_rate = acc_row["cum_accept_rate"]
            cum_def_rate = acc_row["cum_default_rate"]

            st.write(
                f"- Sort customers by **ascending PD** (best risk first), then approve sequentially."
            )
            st.write(
                f"- To keep cumulative default rate around **2%**, "
                f"accept up to **decile {cut_decile}**."
            )
            st.write(
                f"- This corresponds to approving roughly **{accept_rate:.1%}** "
                f"of customers with an estimated portfolio default rate of "
                f"about **{cum_def_rate:.2%}** on the historical data."
            )
    else:
        st.write(
            "- Sort customers by ascending PD, approve the best-risk customers first, "
            "and determine the cut-off once the simulated cumulative default rate hits 2% "
            "(requires historical default labels)."
        )

    # 3. What are the characteristics of a defaulter, and how important are they?
    st.markdown("**3. What are the characteristics of a defaulter, and how important are they?**")

    if has_target:
        # Compare top-risk vs low-risk groups
        scored_with_decile = scored_df.copy()
        if "decile" not in scored_with_decile.columns and "pd" in scored_with_decile.columns:
            scored_with_decile = scored_with_decile.sort_values("pd").reset_index(drop=True)
            scored_with_decile["rank"] = np.arange(1, len(scored_with_decile) + 1)
            scored_with_decile["decile"] = pd.qcut(
                scored_with_decile["rank"],
                10,
                labels=list(range(1, 11)),
            )

        top_decile = scored_with_decile[scored_with_decile["decile"] == 10]
        bottom_decile = scored_with_decile[scored_with_decile["decile"] == 1]

        st.write(
            "- Comparing the **highest-risk decile** (decile 10) with the **lowest-risk decile** (decile 1):"
        )
        st.write(
            f"  - Average PD in decile 10: {top_decile['pd'].mean():.2%} "
            f"vs decile 1: {bottom_decile['pd'].mean():.2%}"
        )

        # Only describe some key behavioral features if present
        def avg_if_col(col):
            return top_decile[col].mean() if col in top_decile.columns else np.nan

        char_lines = []
        if "worst_slik_score" in scored_with_decile.columns:
            char_lines.append(
                f"  - Higher **worst_slik_score** (more severe delinquency history)"
            )
        if "pay_ratio_total" in scored_with_decile.columns:
            char_lines.append(
                f"  - Lower **pay_ratio_total** (paid a smaller fraction of total due)"
            )
        if "late_ratio" in scored_with_decile.columns:
            char_lines.append(
                f"  - Higher **late_ratio** (more installments paid late)"
            )
        if "n_defaulted_loans" in scored_with_decile.columns:
            char_lines.append(
                f"  - More **n_defaulted_loans** in the past"
            )

        if char_lines:
            st.write("- Typical defaulters tend to have:")
            for line in char_lines:
                st.write(line)
        else:
            st.write(
                "- High-risk customers generally have worse historical repayment behavior "
                "than low-risk customers (e.g., more days past due, more late payments)."
            )
    else:
        st.write(
            "- Defaulter characteristics are usually derived by comparing default vs "
            "non-default segments on behavioral and demographic features. "
            "This requires historical default labels in the data."
        )


def main():
    st.title("Credit Default Prediction Dashboard")

    st.sidebar.header("Input Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload combined_df CSV",
        type=["csv"],
        help="Upload the raw combined loan–payment–customer dataset",
    )

    if uploaded_file is None:
        st.info("Please upload a combined_df-style CSV file to start.")
        return

    # 1. Load raw data
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    st.subheader("1. Raw Input Preview")
    st.write(f"Rows: {raw_df.shape[0]}, Columns: {raw_df.shape[1]}")
    st.dataframe(raw_df.head(20))

    # 2. Build customer-level features
    st.subheader("2. Customer-level Feature Engineering")
    try:
        df_fe = build_customer_features_from_combined(raw_df)
    except Exception as e:
        st.error(f"Error during feature engineering: {e}")
        return

    st.write(f"Customer-level table shape: {df_fe.shape[0]} customers, {df_fe.shape[1]} columns.")
    st.dataframe(df_fe.head(20))

    # 3. Exploratory Data Analysis
    st.subheader("3. Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        plot_target_distribution(df_fe)

    with col2:
        plot_default_rate_by_category(df_fe, "worst_slik_score")

    col3, col4 = st.columns(2)

    with col3:
        if "main_loan_purpose" in df_fe.columns:
            plot_default_rate_by_category(df_fe, "main_loan_purpose")
        else:
            st.info("Column 'main_loan_purpose' is not available.")

    with col4:
        if "age_bucket" in df_fe.columns:
            plot_default_rate_by_category(df_fe, "age_bucket")
        else:
            st.info("Column 'age_bucket' is not available.")

    # 4. Model Prediction and Business Insights
    st.subheader("4. Model Prediction and Business Insights")

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(
            "Failed to load model from "
            f"'{MODEL_PATH}'. Please ensure the file exists and sklearn/numpy versions match. "
            f"Error: {e}"
        )
        return

    # Score customers
    scored_df = score_customers(model, df_fe)

    # Merge features into scored_df for later analysis / insight
    scored_full = df_fe.merge(
        scored_df[["customer_id", "pd"]],
        on="customer_id",
        how="left",
    )

    st.write("Sample of scored customers:")
    st.dataframe(scored_full.head(20))

    # PD distribution
    plot_pd_histogram(scored_df)

    # Build decile table using customers with known target (if available)
    if "default_flag_customer" in scored_full.columns:
        scored_for_decile = scored_full[["customer_id", "default_flag_customer", "pd"]].copy()
    else:
        scored_for_decile = scored_full[["customer_id", "pd"]].copy()

    dec_table = build_decile_table(scored_for_decile, n_deciles=10)
    st.markdown("**Decile Table (1 = lowest PD, 10 = highest PD)**")
    st.dataframe(dec_table)

    # Attach decile info back to scored_full for further analysis
    scored_for_decile = scored_for_decile.sort_values("pd").reset_index(drop=True)
    scored_for_decile["rank"] = np.arange(1, len(scored_for_decile) + 1)
    scored_for_decile["decile"] = pd.qcut(
        scored_for_decile["rank"],
        10,
        labels=list(range(1, 11)),
    )
    scored_full = scored_full.merge(
        scored_for_decile[["customer_id", "decile"]],
        on="customer_id",
        how="left",
    )

    # Feature importance
    st.subheader("5. Feature Importance")
    try:
        importance_df = extract_feature_importance(model)
        st.dataframe(importance_df.head(20))
    except Exception as e:
        st.warning(f"Could not extract feature importance: {e}")

    # Business Q&A using PD + deciles
    render_business_insights(scored_full, dec_table)

    # Download scored results
    st.subheader("6. Download Scored Customers")

    output_cols = ["customer_id", "pd"]
    if "default_flag_customer" in scored_full.columns:
        output_cols.append("default_flag_customer")
    if "decile" in scored_full.columns:
        output_cols.append("decile")

    output_df = scored_full[output_cols].copy()
    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="Download scored_customers.csv",
        data=csv_bytes,
        file_name="scored_customers.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
