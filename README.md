# Credit Card Behaviour Score Prediction

> A financially interpretable classification model that predicts whether a credit card customer will default on their payment next month — built for Finance Club's Open Project Summer 2025.

---

## Overview

Bank A wants to move from reactive to proactive credit risk management by building a forward-looking **Behaviour Score**: a binary classifier that predicts whether a credit card customer will default in the next billing cycle.

Using anonymized historical behavioral data from **over 30,000 customers** — payment history, bill amounts, repayment patterns, credit limits, and demographics — this project:

1. Conducts in-depth **exploratory & financial analysis** of what drives default risk
2. Engineers **financially meaningful features** (utilization ratios, delinquency streaks, repayment trends)
3. Trains and compares multiple **classification models**, handling class imbalance along the way
4. Selects a **business-aligned evaluation metric and threshold** to balance missed defaulters against false alarms
5. Produces **production-style predictions** on an unlabeled validation set

---

## Objectives

- Build a binary classifier for `next_month_default` (1 = Default, 0 = No Default)
- Handle class imbalance (SMOTE / class weighting / downsampling)
- Go beyond basic EDA — analyze payment delays, repayment consistency, and utilization trends
- Engineer predictive, financially meaningful features
- Compare Logistic Regression, Decision Trees, and ensemble methods (XGBoost, LightGBM)
- Choose and justify evaluation metrics that reflect real-world credit risk trade-offs
- Set a classification threshold aligned with the bank's risk appetite
- Generate final predictions on the validation dataset

---

## Dataset

| File | Description |
|---|---|
| `train_dataset_final1.csv` | ~25,000 labeled records used for training/validation |
| `validate_dataset_final.csv` | ~5,000 unlabeled records for final prediction |
| `submission_22322002.csv` | Final predictions: `Customer`, `next_month_default` |

### Key Features

| Column | Description |
|---|---|
| `Customer_ID` | Unique customer identifier |
| `LIMIT_BAL` | Assigned credit limit |
| `age`, `sex`, `education`, `marriage` | Demographics |
| `pay_0`, `pay_2`–`pay_6` | Monthly repayment status (-2 = no consumption, -1 = fully paid, 0 = partial/minimum payment, ≥1 = months overdue) |
| `Bill_amt1`–`Bill_amt6` | Monthly bill amounts (last 6 months) |
| `pay_amt1`–`pay_amt6` | Monthly payment amounts |
| `AVG_Bill_amt` | Average bill amount over 6 months |
| `PAY_TO_BILL_ratio` | Ratio of total payments to total bills over 6 months |
| `next_month_default` | **Target** — 1 if customer defaults next month, 0 otherwise |

Dataset Link - https://drive.google.com/file/d/1-cuatRE_9yRNhD8MXfgQf1eiAp6VkBbr/view
---

## Methodology

```
Data Cleaning → EDA & Financial Analysis → Feature Engineering →
Class Imbalance Handling → Model Comparison → Threshold Tuning → Prediction
```

1. **Data Cleaning** — checked missing values/dtypes (minimal missingness, ~126 nulls in `age`); imputed numerical features with median, categorical with mode.
2. **EDA & Financial Analysis** — univariate distributions (age, credit limit, bill/payment amounts), repayment-status countplots, correlation heatmap, and bivariate analysis of `LIMIT_BAL`, `PAY_TO_BILL_ratio`, and bill amounts against default status.
3. **Feature Engineering:**
   - **Average Credit Utilization Ratio** — mean of monthly `Bill_amt / LIMIT_BAL` across 6 months
   - **Delinquency Streak** — longest consecutive run of overdue months (`pay_0`–`pay_6` ≥ 1)
   - **Repayment Variance** — variance of `pay_amt1`–`pay_amt6`
   - **Rolling Averages/Trends** — 3-month average bill, 6-month bill trend
   - **Log-transformed features** — for highly skewed variables
   - **Interaction term** — utilization × delinquency streak (captures compounding risk)
4. **Class Imbalance Handling** — class weighting (and SMOTE/downsampling explored) to address the skew toward non-defaulters.
5. **Model Comparison** — Logistic Regression, Decision Tree, XGBoost, and LightGBM evaluated via 5-fold stratified cross-validation, optimizing for **F2-score** (`fbeta_score`, beta=2).
6. **Threshold Tuning** — optimal classification threshold selected by maximizing F2-score on the precision-recall curve.
7. **Final Prediction** — best model applied to the validation set to generate `submission_<enrollment_number>.csv`.

---

## Key EDA & Financial Findings

- The target variable `next_month_default` is **imbalanced** — most customers do not default.
- **Recent repayment status (`pay_0`) is the strongest risk signal** — positively correlated with default.
- **`PAY_TO_BILL_ratio` is inversely related to default risk** — customers who repay a higher share of their bill are far less likely to default.
- **Credit utilization** is a key driver: defaulters consistently show higher average utilization of their credit limit.
- **Delinquency streaks** are strongly linked to default — defaulters typically have much longer consecutive overdue-payment runs, while non-defaulters mostly have none.
- Demographic factors (age, education, marital status, sex) show only subtle trends and are far less predictive than behavioral/financial variables.

---

## Model Performance

### Cross-Validated Comparison (F2-optimized)

| Model | Best CV Score (F2) | Key Hyperparameters |
|---|---|---|
| **Decision Tree** ⭐ | **0.79** | `criterion='entropy'`, `max_depth=5` |
| LightGBM | 0.78 | `learning_rate=0.1`, `max_depth=5`, `n_estimators=50` |
| XGBoost | 0.78 | similar to LightGBM |
| Logistic Regression | 0.64 | `class_weight='balanced'` |

**Final model: Decision Tree Classifier**, chosen for its top F2-score *and* strong interpretability — a key requirement for banking risk management and regulatory transparency.

### Training Set Metrics (at optimal threshold = 0.10)

| Metric | Score |
|---|---|
| Accuracy | 0.82 |
| F1-score | 0.82 |
| Recall | 0.82 |
| F2-score | 0.8294 |

### Evaluation Strategy

- **F2-score** (β = 2) — weights recall twice as heavily as precision, since missing an actual defaulter (false negative) is costlier to the bank than a false alarm.
- **ROC-AUC** — measures overall class separability across thresholds.
- The classification **cutoff was tuned** by scanning thresholds against the precision-recall curve and selecting the one that maximized F2-score.

---

## Business Implications

- **Minimizes financial losses** by catching more true defaulters before they default
- **Improves risk management** through data-driven, proactive lending decisions
- **Optimizes loan portfolio quality** and long-term portfolio health
- **Boosts operational efficiency** via faster/automated credit assessment
- **Enables strategic resource allocation** — focus manual review on high-risk cases

---

## Repository Structure

```
credit-card-behaviour-score/
├── data/
│   ├── raw/                              # train_dataset_final1.csv, validate_dataset_final.csv (not tracked)
│   └── processed/                        # Cleaned/engineered data (generated)
├── notebooks/
│   └── credit_card_default_prediction.ipynb   # Full analysis: EDA → Feature Eng. → Modeling → Prediction
├── docs/
│   ├── project_brief.pdf                 # Finance Club Open Project problem statement
│   └── financial_report.pdf              # Full EDA, financial analysis & findings report
├── outputs/
│   └── submission_22322002.csv           # Final predictions (Customer, next_month_default)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Tech Stack

- **Language:** Python 3.10+
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (Logistic Regression, Decision Tree), XGBoost, LightGBM
- **Imbalance Handling:** imbalanced-learn (SMOTE), class weighting
- **Environment:** Jupyter Notebook

---

## Limitations & Future Work

- Evaluation metrics were reported on the training set; a held-out test split or nested cross-validation would give a more robust estimate of generalization.
- Explore SHAP/LIME explainability (optional in the brief) to make individual-level default explanations available to risk officers.
- Test cost-sensitive learning or Bayesian threshold optimization tied to actual loss-given-default figures instead of a pure F2 proxy.

---

## 👤 Author

**Akanksha Sahni**

---
