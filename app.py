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

# Path to trained model pipeline
MODEL_PATH = "best_credit_scoring_extratrees.pkl"

# Colors
ORANGE = "#FF9800"
DARK_GREEN = "#2E7D32"

# Colormap for correlation heatmap
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "green_orange", [DARK_GREEN, "white", ORANGE]
)

# Expected input columns for combined_df.csv
REQUIRED_COLUMNS = [
    "cdate",
    "application_id",
    "customer_id",
    "loan_purpose",
    "loan_purpose_desc",
    "dob",
    "address_provinsi",
    "marital_status",
    "dependent",
    "job_type",
    "job_industry",
    "loan_id",
    "loan_amount",
    "loan_duration",
    "installment_amount",
    "fund_transfer_ts",
    "payment_id",
    "due_date",
    "paid_date",
    "paid_amount",
    "dpd",
]

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


def cast_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast ID-like columns to string so they do not show thousand separators.
    """
    id_cols = ["application_id", "customer_id", "loan_id", "payment_id"]
    for col in id_cols:
        if col in df.columns:
            def _to_str(x):
                if pd.isna(x):
                    return ""
                if isinstance(x, (int, np.integer)):
                    return str(x)
                if isinstance(x, (float, np.floating)):
                    return str(int(x))
                return str(x)
            df[col] = df[col].apply(_to_str).astype("string")
    return df


def add_bar_labels(ax, fmt="{:.0f}"):
    """
    Add value labels on top of bars in a bar chart.
    """
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def plot_target_distribution(df_fe: pd.DataFrame):
    """
    Default vs non-default bar chart at customer level.
    """
    if "default_flag_customer" not in df_fe.columns:
        st.info("Target column 'default_flag_customer' is not available for this dataset.")
        return

    target_counts = df_fe["default_flag_customer"].value_counts().sort_index()
    labels = ["Non-default", "Default"]
    values = [target_counts.get(0, 0), target_counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        x=labels,
        y=values,
        ax=ax,
        palette=[DARK_GREEN, ORANGE],
    )
    ax.set_title("Customer-level Default Distribution")
    ax.set_xlabel("Default Flag")
    ax.set_ylabel("Number of Customers")
    add_bar_labels(ax, fmt="{:.0f}")
    st.pyplot(fig)


def plot_default_rate_by_category(df_fe: pd.DataFrame, col: str, max_categories: int = 8):
    """
    Default rate by selected categorical feature (customer level).
    """
    if col not in df_fe.columns:
        st.warning(f"Column '{col}' not found.")
        return
    if "default_flag_customer" not in df_fe.columns:
        st.info("Target column 'default_flag_customer' is not available for this dataset.")
        return

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
    sns.barplot(data=rate_df, x=col, y="default_rate", ax=ax, color=ORANGE)
    ax.set_title(f"Default Rate by {col}")
    ax.set_ylabel("Default Rate")
    ax.set_xlabel(col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    add_bar_labels(ax, fmt="{:.1%}")
    st.pyplot(fig)


def plot_pd_histogram(scored_df: pd.DataFrame):
    """
    Histogram for predicted PDs.
    """
    if "pd" not in scored_df.columns:
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    counts, bins, patches = ax.hist(
        scored_df["pd"], bins=20, edgecolor="k", color=ORANGE
    )
    ax.set_title("Predicted PD Distribution")
    ax.set_xlabel("PD")
    ax.set_ylabel("Number of Customers")

    for count, patch in zip(counts, patches):
        if count <= 0:
            continue
        x = patch.get_x() + patch.get_width() / 2
        ax.annotate(
            f"{int(count)}",
            (x, count),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    st.pyplot(fig)


def plot_default_rate_deciles(dec_table: pd.DataFrame):
    """
    Visual for default rate by decile + cumulative default rate.
    """
    if "default_rate" not in dec_table.columns:
        st.info("Default rate is not available in decile table.")
        return

    d = dec_table.sort_values("decile")
    x = d["decile"].astype(str)

    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.bar(x, d["default_rate"], alpha=0.8, color=ORANGE)
    ax1.set_xlabel("Decile (1 = lowest PD, 10 = highest PD)")
    ax1.set_ylabel("Default Rate")
    add_bar_labels(ax1, fmt="{:.1%}")

    if "cum_default_rate" in d.columns:
        ax2 = ax1.twinx()
        ax2.plot(x, d["cum_default_rate"], marker="o", color=DARK_GREEN)
        ax2.set_ylabel("Cumulative Default Rate")

    ax1.set_title("Default Rate and Cumulative Default Rate by Decile")
    st.pyplot(fig)


def plot_top_bottom_decile_feature_means(scored_full: pd.DataFrame):
    """
    Compare key behavioural features between decile 1 and decile 10.
    """
    if "decile" not in scored_full.columns:
        st.info("Decile information is not available.")
        return

    features = ["worst_slik_score", "pay_ratio_total", "late_ratio", "n_defaulted_loans"]
    features = [f for f in features if f in scored_full.columns]
    if not features:
        st.info("Key behavioural features are not available for comparison.")
        return

    top = scored_full[scored_full["decile"] == 10]
    bottom = scored_full[scored_full["decile"] == 1]

    if top.empty or bottom.empty:
        st.info("Cannot compute comparison: decile 1 or 10 is empty.")
        return

    mean_top = top[features].mean()
    mean_bottom = bottom[features].mean()

    plot_df = pd.DataFrame(
        {
            "feature": features,
            "Decile 1 (lowest PD)": mean_bottom.values,
            "Decile 10 (highest PD)": mean_top.values,
        }
    )

    plot_df = plot_df.melt(id_vars="feature", var_name="group", value_name="value")

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(
        data=plot_df,
        x="feature",
        y="value",
        hue="group",
        ax=ax,
        palette={"Decile 1 (lowest PD)": DARK_GREEN, "Decile 10 (highest PD)": ORANGE},
    )
    ax.set_title("Key Behavioural Features: Decile 1 vs Decile 10")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean Value")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    add_bar_labels(ax, fmt="{:.2f}")
    st.pyplot(fig)


def plot_actual_default_after_prediction(scored_full: pd.DataFrame):
    """
    After prediction, show bar + pie for default vs non-default
    using historical default_flag_customer (if available).
    """
    if "default_flag_customer" not in scored_full.columns:
        st.info("Historical 'default_flag_customer' is not available for this dataset.")
        return

    counts = scored_full["default_flag_customer"].value_counts().sort_index()
    labels = ["Non-default", "Default"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(
            x=labels,
            y=values,
            ax=ax,
            palette=[DARK_GREEN, ORANGE],
        )
        ax.set_title("Number of Customers by Default Status")
        ax.set_xlabel("Status")
        ax.set_ylabel("Count")
        add_bar_labels(ax, fmt="{:.0f}")
        st.pyplot(fig)

    with col2:
        total = sum(values)
        if total == 0:
            st.info("No default labels available for pie chart.")
            return
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(
            values,
            labels=labels,
            colors=[DARK_GREEN, ORANGE],
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
        )
        ax.set_title("Percentage of Customers by Default Status")
        ax.axis("equal")
        st.pyplot(fig)


def plot_correlation_heatmap(df_fe: pd.DataFrame, max_features: int = 15):
    """
    Correlation heatmap for customer-level numeric features.
    ID-like numeric columns are excluded from the matrix.
    """
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude id-like numeric columns from correlation matrix
    num_cols = [c for c in num_cols if "id" not in c.lower()]

    if len(num_cols) < 2:
        st.info("Not enough numeric features to compute correlation heatmap.")
        return

    if len(num_cols) > max_features:
        num_cols = num_cols[:max_features]

    corr = df_fe[num_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr,
        cmap=HEATMAP_CMAP,
        annot=True,
        fmt=".2f",
        square=True,
        cbar=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Customer-level numeric features)")
    st.pyplot(fig)


def render_business_insights(scored_full: pd.DataFrame, dec_table: pd.DataFrame):
    """
    Text + visual answers for business questions.
    """
    st.subheader("Business Questions")

    has_target = "default_flag_customer" in scored_full.columns
    has_decile = "decile" in scored_full.columns

    # 1. Which customers/loans should we avoid? Why?
    st.markdown("**1. Which customers/loans should we avoid? Why?**")

    if has_target and has_decile:
        high_risk = scored_full[scored_full["decile"].isin([9, 10])]
        total_customers = scored_full.shape[0]

        portfolio_share = len(high_risk) / total_customers if total_customers > 0 else np.nan

        total_defaults = scored_full["default_flag_customer"].sum()
        defaults_share = (
            high_risk["default_flag_customer"].sum() / total_defaults
            if total_defaults > 0
            else np.nan
        )

        st.write(
            "Customers in the highest PD deciles (e.g., deciles 9–10) "
            "should be avoided or priced very conservatively."
        )
        st.write(
            f"These segments represent around **{portfolio_share:.1%}** of the portfolio "
            f"but capture approximately **{defaults_share:.1%}** of historical defaults "
            "(based on the training data)."
        )
        st.write(
            "Their predicted PD is significantly higher than the portfolio average, "
            "indicating poor risk–return trade-off."
        )

        plot_default_rate_deciles(dec_table)

    else:
        st.write(
            "Customers with the highest predicted PDs (top risk deciles) should be "
            "avoided or priced conservatively, as they contribute disproportionately to portfolio risk."
        )

    # 2. If the business would like to achieve a 2% cumulative default rate, which loans should we accept?
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
                "With the current portfolio and model, a 2% cumulative default "
                "target cannot be achieved even if we only approve the very best customers."
            )
        else:
            d_sorted = dec_table.sort_values("decile")
            row = d_sorted[d_sorted["decile"] == cut_decile].iloc[0]
            accept_rate = row["cum_accept_rate"]
            cum_def_rate = row["cum_default_rate"]

            st.write("Sort customers by ascending PD (best risk first), then approve sequentially.")
            st.write(
                f"To keep cumulative default rate around **2%**, accept up to **decile {cut_decile}**."
            )
            st.write(
                f"This corresponds to approving roughly **{accept_rate:.1%}** of customers "
                f"with an estimated portfolio default rate of about **{cum_def_rate:.2%}** "
                "on the historical data."
            )

    else:
        st.write(
            "Sort customers by ascending PD, approve the best-risk customers first, "
            "and place the cut-off where simulated cumulative default rate reaches 2% "
            "(requires historical default labels)."
        )

    # 3. What are the characteristics of a defaulter, and how important are they?
    st.markdown("**3. What are the characteristics of a defaulter, and how important are they?**")

    if has_target and has_decile:
        top = scored_full[scored_full["decile"] == 10]
        bottom = scored_full[scored_full["decile"] == 1]

        if not top.empty and not bottom.empty:
            avg_pd_top = top["pd"].mean()
            avg_pd_bottom = bottom["pd"].mean()

            st.write(
                "Comparing the highest-risk decile (decile 10) with the lowest-risk decile (decile 1):"
            )
            st.write(
                f"- Average PD in decile 10: **{avg_pd_top:.2%}** "
                f"vs decile 1: **{avg_pd_bottom:.2%}**"
            )

            st.write("Typical defaulters tend to have:")
            if "worst_slik_score" in scored_full.columns:
                st.write("- Higher **worst_slik_score** (more severe delinquency history)")
            if "pay_ratio_total" in scored_full.columns:
                st.write("- Lower **pay_ratio_total** (paid a smaller fraction of total due)")
            if "late_ratio" in scored_full.columns:
                st.write("- Higher **late_ratio** (more installments paid late)")
            if "n_defaulted_loans" in scored_full.columns:
                st.write("- More **n_defaulted_loans** in the past")

            plot_top_bottom_decile_feature_means(scored_full)
        else:
            st.write(
                "Defaulter characteristics are usually derived by comparing highest-risk "
                "and lowest-risk segments, but decile 1 or decile 10 is empty in this dataset."
            )
    else:
        st.write(
            "Defaulter characteristics can be analysed by comparing default vs non-default customers "
            "on behavioural and demographic features. This requires historical default labels."
        )


def main():
    st.title("Credit Default Prediction Dashboard")

    st.markdown(
        """
This application scores a portfolio of consumer loans and visualises credit risk at **customer level**.

**High-level flow**

1. Upload the raw `combined_df.csv` file (loan–payment–customer level).
2. The app performs basic EDA on the raw data (numeric and categorical).
3. The app aggregates to customer-level features and shows customer-level EDA.
4. A pre-trained ExtraTrees-based credit scoring model predicts **Probability of Default (PD)** for each customer.
5. The app builds deciles, shows feature importance, and answers key **business questions**:
   - Which customers/loans should we avoid?
   - Which loans can be accepted to target a 2% cumulative default rate?
   - What are typical characteristics of defaulters?
"""
    )

    st.markdown(
        """
**Expected input file (combined_df.csv)**

The CSV should contain at least the following columns (case-sensitive):

- `cdate` (application timestamp)
- `application_id`
- `customer_id`
- `loan_purpose`, `loan_purpose_desc`
- `dob`
- `address_provinsi`
- `marital_status`
- `dependent`
- `job_type`, `job_industry`
- `loan_id`
- `loan_amount`, `loan_duration`, `installment_amount`
- `fund_transfer_ts`
- `payment_id`
- `due_date`, `paid_date`
- `paid_amount`
- `dpd` (days past due per installment)
"""
    )

    st.sidebar.header("Input Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload combined_df CSV",
        type=["csv"],
        help="Upload the raw combined loan–payment–customer dataset with the columns listed in the description.",
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

    raw_df = cast_id_columns(raw_df)

    st.subheader("1. Raw Input Preview")
    st.write(f"Rows: {raw_df.shape[0]}, Columns: {raw_df.shape[1]}")
    st.dataframe(raw_df.head(20))

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        st.warning(
            "Some expected columns are missing from the input file:\n\n"
            + ", ".join(missing_cols)
        )

    # 2. Raw Data EDA (numeric and categorical)
    st.subheader("2. Raw Data EDA (combined_df)")

    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()

    col1, col2 = st.columns(2)

    # Numeric EDA
    with col1:
        st.markdown("**Numeric overview**")
        if numeric_cols:
            st.write(raw_df[numeric_cols].describe().T)

            selected_num = st.selectbox(
                "Numeric column for histogram",
                options=numeric_cols,
                index=0,
            )
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(raw_df[selected_num].dropna(), bins=30, edgecolor="k", color=ORANGE)
            ax.set_title(f"Histogram of {selected_num}")
            ax.set_xlabel(selected_num)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found in the raw dataset.")

    # Categorical EDA
    with col2:
        st.markdown("**Categorical overview**")

        all_cat_cols = raw_df.select_dtypes(include=["object", "string"]).columns.tolist()
        if all_cat_cols:
            date_like_cols = [
                c for c in all_cat_cols if any(k in c.lower() for k in ["date", "time", "ts"])
            ]
            id_like_cols = [c for c in all_cat_cols if "id" in c.lower()]

            main_cat = [c for c in all_cat_cols if c not in id_like_cols + date_like_cols]
            ordered_cols = main_cat + [
                c for c in all_cat_cols if c in id_like_cols + date_like_cols
            ]

            overview_rows = []
            for col in ordered_cols:
                vc = raw_df[col].value_counts(dropna=False)
                n_unique = len(vc)
                top_cat = vc.index[0] if n_unique > 0 else None
                top_cnt = int(vc.iloc[0]) if n_unique > 0 else 0
                overview_rows.append(
                    {
                        "column": col,
                        "n_unique": n_unique,
                        "top_category": top_cat,
                        "top_count": top_cnt,
                    }
                )
            overview_df = pd.DataFrame(overview_rows)
            st.write("Categorical columns summary (ID and Date columns are listed last):")
            st.dataframe(overview_df)

            selected_cat = st.selectbox(
                "Categorical column for bar chart",
                options=ordered_cols,
                index=0,
            )

            vc = raw_df[selected_cat].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x=vc.index, y=vc.values, ax=ax, color=DARK_GREEN)
            ax.set_title(f"Top categories of {selected_cat}")
            ax.set_xlabel(selected_cat)
            ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.info("No categorical columns found in the raw dataset.")

    # 3. Customer-level Feature Engineering
    st.subheader("3. Customer-level Feature Engineering")
    try:
        df_fe = build_customer_features_from_combined(raw_df)
    except Exception as e:
        st.error(f"Error during feature engineering: {e}")
        return

    st.write(f"Customer-level table shape: {df_fe.shape[0]} customers, {df_fe.shape[1]} columns.")
    st.dataframe(df_fe.head(20))

    # 4. Customer-level EDA
    st.subheader("4. Customer-level EDA")

    col3, col4 = st.columns(2)

    with col3:
        plot_target_distribution(df_fe)

    with col4:
        plot_default_rate_by_category(df_fe, "worst_slik_score")

    col5, col6 = st.columns(2)

    with col5:
        if "main_loan_purpose" in df_fe.columns:
            plot_default_rate_by_category(df_fe, "main_loan_purpose")
        else:
            st.info("Column 'main_loan_purpose' is not available.")

    with col6:
        if "age_bucket" in df_fe.columns:
            plot_default_rate_by_category(df_fe, "age_bucket")
        else:
            st.info("Column 'age_bucket' is not available.")

    # 5. Correlation Heatmap (customer-level)
    st.subheader("5. Correlation Heatmap (Customer-level numeric features)")
    plot_correlation_heatmap(df_fe, max_features=15)

    # 6. Model Prediction and Business Insights
    st.subheader("6. Model Prediction and Business Insights")

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(
            "Failed to load model from "
            f"'{MODEL_PATH}'. Please ensure the file exists and sklearn/numpy versions match. "
            f"Error: {e}"
        )
        return

    run_pred = st.button("Run prediction", type="primary")

    if not run_pred:
        st.info("Click 'Run prediction' to score customers and see business insights.")
        return

    scored_df = score_customers(model, df_fe)

    scored_full = df_fe.merge(
        scored_df[["customer_id", "pd"]],
        on="customer_id",
        how="left",
    )

    st.write("Sample of scored customers:")
    st.dataframe(scored_full.head(20))

    plot_pd_histogram(scored_df)

    if "default_flag_customer" in scored_full.columns:
        scored_for_decile = scored_full[["customer_id", "default_flag_customer", "pd"]].copy()
    else:
        scored_for_decile = scored_full[["customer_id", "pd"]].copy()

    dec_table = build_decile_table(scored_for_decile, n_deciles=10)
    st.markdown("**Decile Table (1 = lowest PD, 10 = highest PD)**")
    st.dataframe(dec_table)

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

    # 7. Feature Importance
    st.subheader("7. Feature Importance")
    try:
        importance_df = extract_feature_importance(model)
        st.dataframe(importance_df.head(20))
    except Exception as e:
        st.warning(f"Could not extract feature importance: {e}")

    # Business Q&A
    render_business_insights(scored_full, dec_table)

    # 8. Default vs Non-default
    st.subheader("8. Default vs Non-default (Actual Labels)")
    plot_actual_default_after_prediction(scored_full)

    # 9. Download scored results
    st.subheader("9. Download Scored Customers")

    output_cols = ["customer_id", "pd"]
    if "default_flag_customer" in scored_full.columns:
        output_cols.append("default_flag_customer")
    if "decile" in scored_full.columns:
        output_cols.append("decile")

    output_df = scored_full[output_cols].copy()

    default_name = "scored_customers"
    base_name = st.text_input(
        "Output file name (without extension)",
        value=default_name,
    ).strip() or default_name

    file_format = st.selectbox(
        "Output format",
        ["CSV", "Excel (.xlsx)"],
        index=0,
    )

    if file_format == "CSV":
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)
        data_bytes = csv_buffer.getvalue().encode("utf-8")
        file_name = f"{base_name}.csv"
        mime = "text/csv"
    else:
        bytes_buf = BytesIO()
        with pd.ExcelWriter(bytes_buf, engine="xlsxwriter") as writer:
            output_df.to_excel(writer, index=False, sheet_name="scored_customers")
        data_bytes = bytes_buf.getvalue()
        file_name = f"{base_name}.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    st.download_button(
        label="Download file",
        data=data_bytes,
        file_name=file_name,
        mime=mime,
    )


if __name__ == "__main__":
    main()
