import pandas as pd
import numpy as np


def build_customer_features_from_combined(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw combined_df (loan + payment + customer info)
    into customer-level features for credit scoring.

    This implementation mirrors the SQL feature engineering:

    - DPD → SLIK score:
        ≤ 0       → 1
        1–90     → 2
        91–120   → 3
        121–180  → 4
        > 180    → 5

    - Loan-level default definition:
        • total_paid is NULL or 0
        • OR max_dpd > 360 (≈ > 12 months past due)
        • OR total_paid < 0.2 * total_due

    - Customer-level aggregation:
        n_loans, n_defaulted_loans, default_flag_customer,
        loan amount stats, DPD stats, repayment ratios,
        plus demographics and age bucket.
    """

    df = combined_df.copy()

    # ------------------------------------------------------------------
    # 1. Basic type handling (IDs, numerics, dates)
    # ------------------------------------------------------------------
    id_cols = ["application_id", "customer_id", "loan_id", "payment_id"]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    numeric_cols = [
        "loan_amount",
        "loan_duration",
        "installment_amount",
        "paid_amount",
        "dependent",
        "dpd",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # ------------------------------------------------------------------
    # 2. DPD → SLIK score mapping (same as in your SQL)
    # ------------------------------------------------------------------
    if "dpd" not in df.columns:
        df["dpd"] = np.nan

    cond1 = df["dpd"].isna() | (df["dpd"] <= 0)
    cond2 = df["dpd"].between(1, 90, inclusive="both")
    cond3 = df["dpd"].between(91, 120, inclusive="both")
    cond4 = df["dpd"].between(121, 180, inclusive="both")
    cond5 = df["dpd"] > 180

    df["slik_score"] = np.select(
        [cond1, cond2, cond3, cond4, cond5],
        [1, 2, 3, 4, 5],
        default=1,
    )

    # ------------------------------------------------------------------
    # 3. Loan-level aggregation (loan_level CTE)
    # ------------------------------------------------------------------
    loan_level = (
        df.groupby(["customer_id", "loan_id"], dropna=False)
        .agg(
            avg_loan_amount=("loan_amount", "mean"),
            avg_loan_duration=("loan_duration", "mean"),
            avg_installment=("installment_amount", "mean"),
            total_paid=("paid_amount", "sum"),
            total_due=("installment_amount", "sum"),
            n_payments=("installment_amount", "count"),
            n_late=("dpd", lambda x: (pd.to_numeric(x, errors="coerce") > 0).sum()),
            avg_dpd=("dpd", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            max_dpd=("dpd", lambda x: pd.to_numeric(x, errors="coerce").max()),
            worst_slik_score=("slik_score", "max"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 4. Loan-level default flag (loan_flag CTE, adjusted definition)
    #     • total_paid IS NULL OR total_paid = 0
    #     • OR max_dpd > 360
    #     • OR total_paid < 0.2 * total_due
    # ------------------------------------------------------------------
    loan_level["default_flag_loan"] = np.where(
        (loan_level["total_paid"].isna())
        | (loan_level["total_paid"] == 0)
        | (loan_level["max_dpd"] > 360)
        | (loan_level["total_paid"] < 0.2 * loan_level["total_due"]),
        1,
        0,
    )

    # ------------------------------------------------------------------
    # 5. Customer-level behavioural aggregation (cust_behavior CTE)
    # ------------------------------------------------------------------
    cust_behavior = (
        loan_level.groupby("customer_id", dropna=False)
        .agg(
            n_loans=("loan_id", "nunique"),
            n_defaulted_loans=("default_flag_loan", "sum"),
            default_flag_customer=("default_flag_loan", "max"),
            avg_loan_amount=("avg_loan_amount", lambda x: round(np.mean(x), 0)),
            max_loan_amount=("avg_loan_amount", lambda x: round(np.max(x), 0)),
            min_loan_amount=("avg_loan_amount", lambda x: round(np.min(x), 0)),
            avg_loan_duration=("avg_loan_duration", lambda x: round(np.mean(x), 1)),
            avg_dpd=("avg_dpd", lambda x: round(np.mean(x), 1)),
            worst_dpd=("max_dpd", "max"),
            worst_slik_score=("worst_slik_score", "max"),
            sum_total_paid=("total_paid", "sum"),
            sum_total_due=("total_due", "sum"),
            sum_n_late=("n_late", "sum"),
            sum_n_payments=("n_payments", "sum"),
        )
        .reset_index()
    )

    # Repayment ratios (pay_ratio_total, late_ratio, ontime_ratio)
    cust_behavior["pay_ratio_total"] = (
        cust_behavior["sum_total_paid"]
        / cust_behavior["sum_total_due"].replace(0, np.nan)
    )
    cust_behavior["late_ratio"] = (
        cust_behavior["sum_n_late"]
        / cust_behavior["sum_n_payments"].replace(0, np.nan)
    )
    cust_behavior["ontime_ratio"] = 1 - cust_behavior["late_ratio"]

    for col in ["pay_ratio_total", "late_ratio", "ontime_ratio"]:
        cust_behavior[col] = (
            cust_behavior[col]
            .replace([np.inf, -np.inf], np.nan)
            .round(3)
            .fillna(0)
        )

    # Drop intermediate sums to match SQL output
    cust_behavior = cust_behavior.drop(
        columns=["sum_total_paid", "sum_total_due", "sum_n_late", "sum_n_payments"]
    )

    # ------------------------------------------------------------------
    # 6. Customer-level demographics (cust_demo CTE)
    # ------------------------------------------------------------------
    def mode_or_unknown(x):
        x = x.dropna()
        if x.empty:
            return "unknown"
        m = x.mode()
        return m.iloc[0] if not m.empty else "unknown"

    cust_demo = (
        df.groupby("customer_id", dropna=False)
        .agg(
            marital_status=("marital_status", mode_or_unknown),
            job_type=("job_type", mode_or_unknown),
            job_industry=("job_industry", mode_or_unknown),
            address_provinsi=("address_provinsi", mode_or_unknown),
            main_loan_purpose=("loan_purpose", mode_or_unknown),
            avg_dependent=("dependent", "mean"),
            dob=("dob", lambda x: pd.to_datetime(x, errors="coerce").max()),
        )
        .reset_index()
    )

    # Age (using 2022-12-31 as reference, same as JULIANDAY logic)
    cust_demo["age"] = (
        (pd.Timestamp("2022-12-31") - cust_demo["dob"]).dt.days / 365.25
    )
    cust_demo["age"] = cust_demo["age"].replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 7. Age bucket (same as CASE expression in SQL)
    #     <25, 25–34.999, 35–44.999, 45–54.999, 55+
    # ------------------------------------------------------------------
    cust_demo["age_bucket"] = pd.cut(
        cust_demo["age"],
        bins=[-np.inf, 25, 35, 45, 55, np.inf],
        labels=["<25", "25-34", "35-44", "45-54", "55+"],
    )

    cust_demo["age_bucket"] = (
        cust_demo["age_bucket"]
        .astype(str)
        .fillna("unknown")
    )

    # Round age to 1 decimal place to match SQL ROUND(d.age, 1)
    cust_demo["age"] = cust_demo["age"].round(1)

    # ------------------------------------------------------------------
    # 8. Final merge to customer-level feature table
    # ------------------------------------------------------------------
    df_features = pd.merge(
        cust_behavior,
        cust_demo[
            [
                "customer_id",
                "marital_status",
                "job_type",
                "job_industry",
                "address_provinsi",
                "main_loan_purpose",
                "avg_dependent",
                "age",
                "age_bucket",
            ]
        ],
        on="customer_id",
        how="left",
    )

    # Ensure avg_dependent numeric-ish behaviour
    df_features["avg_dependent"] = df_features["avg_dependent"].astype(float)

    return df_features
