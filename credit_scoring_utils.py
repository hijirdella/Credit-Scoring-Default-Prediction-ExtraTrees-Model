import io
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def build_customer_features_from_combined(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level features and target (default_flag_customer)
    from the raw combined_df (loan–payment–customer level).
    
    This reproduces the SQL feature engineering logic from the notebook:
    - base          : raw records
    - dpd_slik      : map dpd to SLIK-style score 1–5
    - loan_level    : aggregate to loan-level metrics
    - loan_flag     : define loan-level default_flag_loan
    - cust_behavior : aggregate to customer-level behavior features
    - cust_demo     : aggregate demographics per customer
    - final join    : combine behavior + demo and create age_bucket
    """
    df = raw_df.copy()

    # Basic datetime parsing (safe)
    date_cols = ["paid_date", "due_date", "dob", "cdate", "fund_transfer_ts"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Base table (like the SQL `base` CTE)
    base_cols = [
        "customer_id",
        "loan_id",
        "loan_amount",
        "loan_duration",
        "installment_amount",
        "paid_amount",
        "paid_date",
        "due_date",
        "dpd",
        "marital_status",
        "job_type",
        "job_industry",
        "address_provinsi",
        "loan_purpose",
        "dependent",
        "dob",
    ]
    missing_cols = [c for c in base_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for feature engineering: {missing_cols}")

    base = df[base_cols].copy()

    # dpd_slik: add SLIK-score-style bucket
    dpd_slik = base.copy()
    dpd = dpd_slik["dpd"]

    dpd_slik["slik_score"] = np.select(
        [
            dpd.isna() | (dpd <= 0),
            (dpd >= 1) & (dpd <= 90),
            (dpd >= 91) & (dpd <= 120),
            (dpd >= 121) & (dpd <= 180),
            dpd > 180,
        ],
        [1, 2, 3, 4, 5],
        default=np.nan,
    )

    # loan_level aggregations (per customer_id, loan_id)
    grp_loan = dpd_slik.groupby(["customer_id", "loan_id"], dropna=False)

    loan_level = grp_loan.agg(
        avg_loan_amount=("loan_amount", "mean"),
        avg_loan_duration=("loan_duration", "mean"),
        avg_installment=("installment_amount", "mean"),
        total_paid=("paid_amount", "sum"),
        total_due=("installment_amount", "sum"),
        n_payments=("installment_amount", "size"),
        n_late=("dpd", lambda x: np.sum((x.fillna(0) > 0).astype(int))),
        avg_dpd=("dpd", "mean"),
        max_dpd=("dpd", "max"),
        worst_slik_score=("slik_score", "max"),
    ).reset_index()

    # loan_flag: define default_flag_loan (loan-level target)
    lf = loan_level.copy()

    cond_never_paid = lf["total_paid"].isna() | (lf["total_paid"] == 0)
    cond_severe_dpd = lf["max_dpd"] > 360
    cond_deep_loss = lf["total_paid"] < 0.2 * lf["total_due"]

    lf["default_flag_loan"] = np.where(
        cond_never_paid | cond_severe_dpd | cond_deep_loss,
        1,
        0,
    )

    # cust_behavior (aggregate to customer-level)
    gb_cust = lf.groupby("customer_id", dropna=False)

    cust_behavior = gb_cust.agg(
        n_loans=("loan_id", "nunique"),
        n_defaulted_loans=("default_flag_loan", "sum"),
        default_flag_customer=("default_flag_loan", "max"),
        avg_loan_amount=("avg_loan_amount", lambda x: round(x.mean(), 0) if len(x) else np.nan),
        max_loan_amount=("avg_loan_amount", lambda x: round(x.max(), 0) if len(x) else np.nan),
        min_loan_amount=("avg_loan_amount", lambda x: round(x.min(), 0) if len(x) else np.nan),
        avg_loan_duration=("avg_loan_duration", lambda x: round(x.mean(), 1) if len(x) else np.nan),
        avg_dpd=("avg_dpd", lambda x: round(x.mean(), 1) if len(x) else np.nan),
        worst_dpd=("max_dpd", "max"),
        worst_slik_score=("worst_slik_score", "max"),
        total_paid=("total_paid", "sum"),
        total_due=("total_due", "sum"),
        n_late=("n_late", "sum"),
        n_payments=("n_payments", "sum"),
    ).reset_index()

    # Ratios: pay_ratio_total, late_ratio, ontime_ratio
    cust_behavior["pay_ratio_total"] = np.where(
        cust_behavior["total_due"].fillna(0) == 0,
        np.nan,
        (cust_behavior["total_paid"] / cust_behavior["total_due"]).round(3),
    )

    cust_behavior["late_ratio"] = np.where(
        cust_behavior["n_payments"].fillna(0) == 0,
        np.nan,
        (cust_behavior["n_late"] / cust_behavior["n_payments"]).round(3),
    )

    cust_behavior["ontime_ratio"] = np.where(
        cust_behavior["n_payments"].fillna(0) == 0,
        np.nan,
        (
            (cust_behavior["n_payments"] - cust_behavior["n_late"])
            / cust_behavior["n_payments"]
        ).round(3),
    )

    # Drop intermediate sums we do not want as final features
    cust_behavior = cust_behavior.drop(
        columns=["total_paid", "total_due", "n_late", "n_payments"]
    )

    # cust_demo: demographics per customer
    base_demo = base.copy()

    ref_date = pd.Timestamp("2022-12-31")
    base_demo["age"] = (ref_date - base_demo["dob"]).dt.days / 365.25

    demo_grp = base_demo.groupby("customer_id", dropna=False)

    def last_non_null(series: pd.Series):
        s = series.dropna()
        return s.iloc[-1] if len(s) else np.nan

    cust_demo = demo_grp.agg(
        marital_status=("marital_status", last_non_null),
        job_type=("job_type", last_non_null),
        job_industry=("job_industry", last_non_null),
        address_provinsi=("address_provinsi", last_non_null),
        main_loan_purpose=("loan_purpose", last_non_null),
        avg_dependent=("dependent", "mean"),
        age=("age", "mean"),
    ).reset_index()

    # Final feature table: join behavior + demo
    df_fe = cust_behavior.merge(cust_demo, on="customer_id", how="left")

    # Age bucket
    def age_bucket_func(a: float) -> str:
        if pd.isna(a):
            return "unknown"
        if a < 25:
            return "<25"
        if 25 <= a < 35:
            return "25-34"
        if 35 <= a < 45:
            return "35-44"
        if 45 <= a < 55:
            return "45-54"
        return "55+"

    df_fe["age_bucket"] = df_fe["age"].apply(age_bucket_func)

    return df_fe


def score_customers(
    model, df_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply trained model pipeline (ExtraTrees + preprocessing + SMOTE)
    to customer-level features.
    
    Returns a DataFrame with:
    - customer_id
    - default_flag_customer (if available)
    - pd : predicted probability of default
    """
    if "customer_id" not in df_features.columns:
        raise ValueError("df_features must contain 'customer_id'.")

    X = df_features.copy()

    # Drop target column if present (model was trained excluding it)
    if "default_flag_customer" in X.columns:
        y_true = X["default_flag_customer"].astype(int)
        X = X.drop(columns=["default_flag_customer"])
    else:
        y_true = None

    # Model output
    if hasattr(model, "predict_proba"):
        pd_hat = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalize to 0-1 via min-max as a fallback
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            pd_hat = np.ones_like(scores) * 0.5
        else:
            pd_hat = (scores - s_min) / (s_max - s_min)
    else:
        preds = model.predict(X)
        pd_hat = preds.astype(float)

    result = pd.DataFrame(
        {
            "customer_id": df_features["customer_id"].values,
            "pd": pd_hat,
        }
    )

    if y_true is not None:
        result["default_flag_customer"] = y_true.values

    return result


def build_decile_table(
    scored_df: pd.DataFrame,
    n_deciles: int = 10,
    target_col: str = "default_flag_customer",
    pd_col: str = "pd",
) -> pd.DataFrame:
    """
    Build a decile table based on predicted PD:
    - Sort ascending PD (best risk first)
    - Split into deciles
    - Compute n_customers, n_defaults, default_rate, avg_pd
    - Compute cumulative acceptance rate and cumulative default rate
    """
    if pd_col not in scored_df.columns:
        raise ValueError(f"{pd_col} column is required for decile table.")

    df = scored_df.dropna(subset=[pd_col]).copy()
    df = df.sort_values(pd_col, ascending=True).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # Assign deciles 1–n_deciles (1 = best, lowest PD)
    df["decile"] = pd.qcut(
        df["rank"],
        q=n_deciles,
        labels=list(range(1, n_deciles + 1)),
    )

    agg_dict = {
        "customer_id": "size",
        pd_col: "mean",
    }
    if target_col in df.columns:
        agg_dict[target_col] = "sum"

    dec = (
        df.groupby("decile")
        .agg(agg_dict)
        .rename(columns={"customer_id": "n_customers", pd_col: "avg_pd"})
        .reset_index()
    )

    total_customers = dec["n_customers"].sum()

    if target_col in df.columns:
        dec["default_rate"] = dec[target_col] / dec["n_customers"]
        dec["cum_customers"] = dec["n_customers"].cumsum()
        dec["cum_defaults"] = dec[target_col].cumsum()
        dec["cum_accept_rate"] = dec["cum_customers"] / total_customers
        dec["cum_default_rate"] = dec["cum_defaults"] / dec["cum_customers"]
    else:
        dec["default_rate"] = np.nan
        dec["cum_customers"] = dec["n_customers"].cumsum()
        dec["cum_defaults"] = np.nan
        dec["cum_accept_rate"] = dec["cum_customers"] / total_customers
        dec["cum_default_rate"] = np.nan

    return dec


def find_acceptance_cut_for_target_default_rate(
    decile_table: pd.DataFrame,
    target_cum_default_rate: float = 0.02,
) -> Optional[int]:
    """
    Given a decile table with `cum_default_rate`, find the highest decile
    where cumulative default rate is still <= target_cum_default_rate.
    
    Returns the decile number (int) or None if target cannot be achieved.
    """
    if "cum_default_rate" not in decile_table.columns:
        return None

    # Ensure deciles are properly ordered
    d = decile_table.sort_values("decile").copy()
    ok = d[d["cum_default_rate"] <= target_cum_default_rate]

    if ok.empty:
        return None

    return int(ok["decile"].max())


def extract_feature_importance(model) -> pd.DataFrame:
    """
    Extract feature importance from the trained pipeline:
    - Use transformed feature names from ColumnTransformer (numeric + OHE categorical)
    - For tree-based models (ExtraTrees, RF, etc.), use feature_importances_.
    
    Returns DataFrame with columns: feature, importance (sorted desc).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    # Expect an imblearn Pipeline: ('preprocess', preprocessor), ('smote', SMOTE), ('model', estimator)
    if not hasattr(model, "named_steps"):
        raise ValueError("Model must be a Pipeline with named_steps.")

    preprocessor = model.named_steps.get("preprocess", None)
    estimator = model.named_steps.get("model", None)

    if preprocessor is None or estimator is None:
        raise ValueError("Pipeline must contain 'preprocess' and 'model' steps.")

    if not isinstance(preprocessor, ColumnTransformer):
        raise ValueError("'preprocess' step must be a ColumnTransformer.")

    # Get feature names after preprocessing
    num_features = []
    cat_features = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            num_features = list(cols)
        elif name == "cat":
            cat_features = list(cols)

    # Get one-hot encoded categorical names
    ohe = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )
    if not isinstance(ohe, OneHotEncoder):
        raise ValueError("Categorical pipeline must contain OneHotEncoder as 'onehot' step.")

    ohe_feature_names = list(ohe.get_feature_names_out(cat_features))

    all_feature_names = num_features + ohe_feature_names

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        # For linear models, use absolute coefficients
        coef = estimator.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)
    else:
        importances = np.zeros(len(all_feature_names))

    importance_df = (
        pd.DataFrame({"feature": all_feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df
