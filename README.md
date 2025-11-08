# Credit Scoring Default Prediction – ExtraTrees Model (Streamlit App)

This project provides a Streamlit web application that predicts the probability of default (PD) for individual customers using an ExtraTrees classifier.  
It performs automated customer-level feature engineering, applies SLIK-style credit scoring, and generates portfolio-level insights.

---

## Live Application

- Streamlit App: []()
- GitHub Repository: []()

---

## Project Overview

**Objective**  
Develop an interpretable and robust machine learning model to predict the likelihood of customer default using behavioral and demographic data.

**Model**  
ExtraTrees Classifier — selected for its generalization strength, interpretability, and high AUC (~0.91 mean cross-validation).

**Data Source**  
Historical loan–payment–customer dataset (`combined_df.csv`), processed into customer-level features consistent with SLIK credit scoring definitions.

---

## Key Features

### 1. Automated Feature Engineering
- Aggregates loan, payment, and customer demographic data.
- Creates behavioral metrics such as on-time ratio, late ratio, pay ratio, and DPD averages.
- Maps DPD values into **SLIK credit score categories (1–5)**:

| SLIK Score | Description |
|-------------|-------------|
| 1 | Credit Current (Lancar) |
| 2 | Under Supervision (DPK, 1–90 DPD) |
| 3 | Substandard (91–120 DPD) |
| 4 | Doubtful (121–180 DPD) |
| 5 | Default / Loss (>180 DPD) |

### 2. Machine Learning Model
- Model: ExtraTrees Classifier  
- Cross-validated AUC: 0.9089  
- Strong generalization and balanced performance across folds.  
- Key predictors: `n_defaulted_loans`, `pay_ratio_total`, `ontime_ratio`, `late_ratio`, `main_loan_purpose`, `min_loan_amount`, `avg_loan_duration`.

### 3. Streamlit Web Interface
- Upload combined loan–payment–customer CSV.
- Automated feature processing and scoring.
- Adjustable PD threshold for decision control.
- Downloadable results in CSV format.

---

## Input Format

Expected columns in the uploaded CSV file:

| Category | Example Columns |
|-----------|----------------|
| Identifiers | `application_id`, `customer_id`, `loan_id`, `payment_id` |
| Loan Details | `loan_amount`, `loan_duration`, `installment_amount` |
| Payment Details | `paid_amount`, `paid_date`, `due_date`, `dpd` |
| Customer Info | `marital_status`, `job_type`, `job_industry`, `address_provinsi`, `loan_purpose`, `dependent`, `dob` |

The app automatically aggregates data to one row per `customer_id`.

---

## Model Performance Summary

| Metric | Value |
|---------|-------|
| Validation AUC | 0.9662 |
| Cross-Validation Mean AUC | 0.9089 |
| F1-Score | 0.92 |
| KS Statistic | 0.85 |
| Accuracy | 0.91 |


