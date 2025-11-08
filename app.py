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

TABLE_HEIGHT = 260
FIG_W = 3.2
FIG_H = 2.2

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "green_orange", [DARK_GREEN, "white", ORANGE]
)

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
        ax.annotate(
            fmt.format(h),
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=4,
        )


def plot_target_distribution(df):
    if "default_flag_customer" not in df.columns:
        return
    counts = df["default_flag_customer"].value_counts().sort_index()
    labels = ["Non-default", "Default"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(x=labels, y=values, ax=ax, palette=[DARK_GREEN, ORANGE])
    ax.set_title("Customer Default Distribution", fontsize=7)
    ax.set_xlabel("Default Flag", fontsize=5)
    ax.set_ylabel("Count", fontsize=5)
    ax.tick_params(axis="both", labelsize=4)
    add_bar_labels(ax)
    plt.tight_layout()
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
    ax.set_title(f"Default Rate by {col}", fontsize=7)
    ax.set_ylabel("Default Rate", fontsize=5)
    ax.set_xlabel(col, fontsize=5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=4)
    ax.tick_params(axis="y", labelsize=4)
    add_bar_labels(ax, fmt="{:.1%}")
    plt.tight_layout()
    st.pyplot(fig)


def plot_pd_histogram(scored_df):
    if "pd" not in scored_df.columns:
        return
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    counts, _, patches = ax.hist(
        scored_df["pd"], bins=15, color=ORANGE, edgecolor="k"
    )
    ax.set_title("Predicted PD Distribution", fontsize=7)
    ax.set_xlabel("PD", fontsize=5)
    ax.set_ylabel("Count", fontsize=5)
    ax.tick_params(axis="both", labelsize=4)
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count + 0.5,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=4,
            )
    plt.tight_layout()
    st.pyplot(fig)


def plot_correlation_heatmap(df_fe, max_features=7):
    num_cols = [
        c for c in df_fe.select_dtypes(include=[np.number]).columns
        if "id" not in c.lower()
    ]
    if len(num_cols) < 2:
        return

    if len(num_cols) > max_features:
        num_cols = num_cols[:max_features]

    corr = df_fe[num_cols].corr()

    fig, ax = plt.subplots(figsize=(3, 2.0))
    sns.heatmap(
        corr,
        cmap=HEATMAP_CMAP,
        square=False,
        linewidths=0.3,
        cbar_kws={"shrink": 0.4},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap", fontsize=7, pad=4)
    ax.tick_params(axis="both", labelsize=4)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)


def plot_default_rate_deciles(dec_table):
    if "default_rate" not in dec_table.columns:
        return
    d = dec_table.sort_values("decile")
    x = d["decile"].astype(str)
    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax1.bar(x, d["default_rate"], color=ORANGE)
    ax1.set_xlabel("Decile", fontsize=5)
    ax1.set_ylabel("Default Rate", fontsize=5)
    ax1.tick_params(axis="both", labelsize=4)
    for p in bars:
        h = p.get_height()
        if not np.isnan(h):
            ax1.text(
                p.get_x() + p.get_width() / 2,
                h + 0.005,
                f"{h:.1%}",
                ha="center",
                va="bottom",
                fontsize=4,
            )
    if "cum_default_rate" in d.columns:
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            d["cum_default_rate"],
            marker="o",
            color=DARK_GREEN,
            linewidth=1.0,
            markersize=2,
        )
        ax2.set_ylabel("Cumulative Default Rate", fontsize=5)
        ax2.tick_params(axis="y", labelsize=4)
    ax1.set_title("Default Rate and Cumulative Default Rate by Decile", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)


def plot_top_bottom_decile_feature_means(scored_full):
    if "decile" not in scored_full.columns:
        return
    features = [
        "worst_slik_score",
        "pay_ratio_total",
        "late_ratio",
        "n_defaulted_loans",
    ]
    features = [f for f in features if f in scored_full.columns]
    if not features:
        return

    top = scored_full[scored_full["decile"] == 10]
    bottom = scored_full[scored_full["decile"] == 1]
    if top.empty or bottom.empty:
        return

    mean_top = top[features].mean()
    mean_bottom = bottom[features].mean()
    plot_df = pd.DataFrame(
        {
            "Feature": features,
            "Decile 1 (lowest PD)": mean_bottom.values,
            "Decile 10 (highest PD)": mean_top.values,
        }
    ).melt(id_vars="Feature", var_name="Group", value_name="Value")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(
        data=plot_df,
        x="Feature",
        y="Value",
        hue="Group",
        ax=ax,
        palette={
            "Decile 1 (lowest PD)": DARK_GREEN,
            "Decile 10 (highest PD)": ORANGE,
        },
    )
    ax.set_title("Key Behavioural Features: Decile 1 vs Decile 10", fontsize=7)
    ax.set_xlabel("Feature", fontsize=5)
    ax.set_ylabel("Mean Value", fontsize=5)
    ax.tick_params(axis="x", labelsize=4, rotation=45)
    ax.tick_params(axis="y", labelsize=4)
    ax.legend(title="Group", fontsize=4, title_fontsize=4.5)
    add_bar_labels(ax, fmt="{:.2f}")
    plt.tight_layout()
    st.pyplot(fig)


def plot_default_vs_nondefault(df):
    if "default_flag_customer" not in df.columns:
        st.info("No 'default_flag_customer' available.")
        return

    counts = df["default_flag_customer"].value_counts().sort_index()
    labels = ["Non-default", "Default"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        sns.barplot(x=labels, y=values, ax=ax, palette=[DARK_GREEN, ORANGE])
        ax.set_title("Customer Count by Default Status", fontsize=7)
        ax.set_xlabel("Status", fontsize=5)
        ax.set_ylabel("Count", fontsize=5)
        ax.tick_params(axis="both", labelsize=4)
        add_bar_labels(ax)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        total = sum(values)
        if total > 0:
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            ax.pie(
                values,
                labels=labels,
                colors=[DARK_GREEN, ORANGE],
                autopct=lambda p: f"{p:.1f}%",
                startangle=90,
                textprops={"fontsize": 4},
            )
            ax.set_title("Customer % by Default Status", fontsize=7)
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)


def render_business_insights(scored_full: pd.DataFrame, dec_table: pd.DataFrame):
    st.subheader("7. Business Questions")

    has_target = "default_flag_customer" in scored_full.columns
    has_decile = "decile" in scored_full.columns

    st.markdown("**1. Which customers/loans should we avoid? Why?**")
    if has_target and has_decile:
        total_customers = len(scored_full)
        total_defaults = scored_full["default_flag_customer"].sum()
        high_risk = scored_full[scored_full["decile"].isin([9, 10])]
        n_high = len(high_risk)
        defaults_high = high_risk["default_flag_customer"].sum()
        portfolio_share = n_high / total_customers if total_customers > 0 else np.nan
        defaults_share = (
            defaults_high / total_defaults if total_defaults > 0 else np.nan
        )
        avg_pd_portfolio = scored_full["pd"].mean()
        avg_pd_high = high_risk["pd"].mean()

        st.write(
            "Customers in the highest PD deciles (9–10) should be avoided or "
            "priced very conservatively."
        )
        st.write(
            f"These segments represent about **{portfolio_share:.1%}** of the portfolio "
            f"but capture around **{defaults_share:.1%}** of historical defaults."
        )
        st.write(
            f"Their predicted PD is around **{avg_pd_high:.2%}**, compared with a "
            f"portfolio average of **{avg_pd_portfolio:.2%}**, indicating a poor "
            "risk–return trade-off."
        )

        plot_default_rate_deciles(dec_table)
    else:
        st.write(
            "High-PD customers (top risk deciles) should generally be avoided or priced "
            "conservatively, as they contribute disproportionately to portfolio risk."
        )

    st.markdown(
        "**2. If the business would like to achieve a 2% cumulative default rate, "
        "which loans should we accept?**"
    )
    if has_target and "cum_default_rate" in dec_table.columns:
        cut_decile = find_acceptance_cut_for_target_default_rate(dec_table, 0.02)
        if cut_decile is None:
            st.write(
                "With the current portfolio and model, a 2% cumulative default target "
                "cannot be achieved even if only the very best customers are approved."
            )
        else:
            d_sorted = dec_table.sort_values("decile")
            row = d_sorted[d_sorted["decile"] == cut_decile].iloc[0]
            accept_rate = row["cum_accept_rate"]
            cum_def_rate = row["cum_default_rate"]
            st.write(
                "Sort customers by ascending PD (best risk first) and approve "
                "sequentially."
            )
            st.write(
                f"To keep cumulative default rate near **2%**, accept up to **decile "
                f"{int(cut_decile)}**."
            )
            st.write(
                f"This corresponds to approving roughly **{accept_rate:.1%}** of "
                f"customers with an estimated portfolio default rate of about "
                f"**{cum_def_rate:.2%}** on the historical data."
            )
    else:
        st.write(
            "A 2% cumulative default cut-off can be simulated once historical "
            "default labels are available; it is based on the decile where "
            "cumulative default rate first reaches 2%."
        )

    st.markdown(
        "**3. What are the characteristics of a defaulter, and how important are they?**"
    )
    if has_decile:
        top = scored_full[scored_full["decile"] == 10]
        bottom = scored_full[scored_full["decile"] == 1]
        if not top.empty and not bottom.empty:
            avg_pd_top = top["pd"].mean()
            avg_pd_bottom = bottom["pd"].mean()
            st.write(
                "Comparing the highest-risk decile (10) with the lowest-risk decile (1):"
            )
            st.write(
                f"- Average PD in decile 10: **{avg_pd_top:.2%}** "
                f"vs decile 1: **{avg_pd_bottom:.2%}**"
            )
            st.write("Typical defaulters tend to have:")
            if "worst_slik_score" in scored_full.columns:
                st.write("- Higher **worst_slik_score** (worse delinquency history)")
            if "pay_ratio_total" in scored_full.columns:
                st.write("- Lower **pay_ratio_total** (pays a smaller share of total due)")
            if "late_ratio" in scored_full.columns:
                st.write("- Higher **late_ratio** (more late instalments)")
            if "n_defaulted_loans" in scored_full.columns:
                st.write("- More **n_defaulted_loans** historically")
            plot_top_bottom_decile_feature_means(scored_full)
        else:
            st.write(
                "Both decile 1 and decile 10 must contain customers to compare "
                "defaulter characteristics; one of them is empty in this dataset."
            )
    else:
        st.write(
            "Defaulter characteristics can be profiled by comparing high-PD vs "
            "low-PD segments on behavioural and demographic features once deciles "
            "are available."
        )


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

All data cleaning, outlier handling, and feature engineering were completed  
during **training, validation, and testing** in Python (offline).  

The dashboard is designed for **model inference and business insight simulation**,  
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
            ax.set_title(f"Histogram of {sel_num}", fontsize=7)
            ax.set_xlabel(sel_num, fontsize=5)
            ax.set_ylabel("Frequency", fontsize=5)
            ax.tick_params(axis="both", labelsize=4)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No numeric columns detected.")

    with col2:
        st.markdown("**Categorical Overview**")
        if cat_cols:
            date_cols = [
                c for c in cat_cols
                if any(x in c.lower() for x in ["date", "time", "ts"])
            ]
            id_cols = [c for c in cat_cols if "id" in c.lower()]
            main_cat = [c for c in cat_cols if c not in id_cols + date_cols]
            ordered = main_cat + date_cols + id_cols

            rows = []
            for c in ordered:
                vc = raw_df[c].value_counts(dropna=False)
                rows.append(
                    {
                        "Column": c,
                        "Unique Values": len(vc),
                        "Top Category": vc.index[0] if len(vc) > 0 else None,
                        "Count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                    }
                )
            st.dataframe(pd.DataFrame(rows), height=TABLE_HEIGHT)

            sel_cat = st.selectbox("Categorical column for bar chart", ordered)
            vc = raw_df[sel_cat].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            sns.barplot(x=vc.index, y=vc.values, ax=ax, color=DARK_GREEN)
            ax.set_title(f"Top categories of {sel_cat}", fontsize=7)
            ax.set_xlabel(sel_cat, fontsize=5)
            ax.set_ylabel("Count", fontsize=5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=4)
            ax.tick_params(axis="y", labelsize=4)
            plt.tight_layout()
            st.pyplot(fig)
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

    st.subheader("4. Customer-level EDA")
    c1, c2 = st.columns(2)
    with c1:
        plot_target_distribution(df_fe)
    with c2:
        plot_default_rate_by_category(df_fe, "worst_slik_score")

    c3, c4 = st.columns(2)
    with c3:
        if "main_loan_purpose" in df_fe.columns:
            plot_default_rate_by_category(df_fe, "main_loan_purpose")
    with c4:
        if "age_bucket" in df_fe.columns:
            plot_default_rate_by_category(df_fe, "age_bucket")

    st.subheader("5. Correlation Heatmap")
    plot_correlation_heatmap(df_fe)

    st.subheader("6. Model Prediction")
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
    scored_full = df_fe.merge(
        scored_df[["customer_id", "pd"]], on="customer_id", how="left"
    )
    st.write("Sample of scored customers:")
    st.dataframe(scored_full.head(20))
    plot_pd_histogram(scored_df)

    if "default_flag_customer" in scored_full.columns:
        dec_input = scored_full[["customer_id", "default_flag_customer", "pd"]].copy()
    else:
        dec_input = scored_full[["customer_id", "pd"]].copy()

    dec_table = build_decile_table(dec_input, n_deciles=10)

    dec_table_display = dec_table.copy()
    if "default_flag_customer" in dec_table_display.columns:
        dec_table_display = dec_table_display.rename(
            columns={"default_flag_customer": "n_defaults"}
        )

    st.subheader("Decile Summary (1 = lowest PD, 10 = highest PD)")
    st.dataframe(dec_table_display)

    dec_input = dec_input.sort_values("pd").reset_index(drop=True)
    dec_input["rank"] = np.arange(1, len(dec_input) + 1)
    dec_input["decile"] = pd.qcut(
        dec_input["rank"],
        10,
        labels=list(range(1, 11)),
    )
    scored_full = scored_full.merge(
        dec_input[["customer_id", "decile"]],
        on="customer_id",
        how="left",
    )

    st.subheader("7 Feature Importance")
    try:
        imp = extract_feature_importance(model)
        st.dataframe(imp.head(20))
    except Exception as e:
        st.warning(f"Could not extract feature importance: {e}")

    render_business_insights(scored_full, dec_table)

    st.subheader("8. Default vs Non-default (Actual Labels)")
    plot_default_vs_nondefault(scored_full)

    st.subheader("9. Download Scored Customers")
    out_cols = ["customer_id", "pd"]
    for c in ["default_flag_customer", "decile"]:
        if c in scored_full.columns:
            out_cols.append(c)
    out_df = scored_full[out_cols].copy()

    base = (
        st.text_input(
            "Output file name (without extension)", value="scored_customers"
        ).strip()
        or "scored_customers"
    )
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

