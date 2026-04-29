# MEPS-Healthcare-Cost-Project

## Overview

A three-stage sequential machine learning pipeline built on 272,568 patient records
from the Medical Expenditure Panel Survey (MEPS) 2013–2022. The pipeline predicts
which patients will fall in the top 20% of annual healthcare expenditure and estimates
individual spending magnitude — enabling insurance companies to proactively identify
high-cost patients and pre-position care management interventions before costs are
incurred.

Each stage produces outputs that feed directly into the next, creating genuine
sequential dependency rather than three isolated models.

---

## Research Question

Can patient-level behavioral, demographic, and clinical features derived from MEPS
accurately predict high-cost healthcare utilization — and what is the expected
expenditure magnitude across the full patient population — enabling healthcare
organizations to proactively target interventions that optimize resource allocation
and revenue stability?

---

## Pipeline Architecture

Data → Segmentation (K-Means) → Classification (XGBoost) → Regression (XGBoost) → Business Action

### Stage 1 — K-Means Segmentation
Segments patients into latent risk profiles based on demographics, chronic condition
burden, and utilization patterns. K=2 selected via silhouette score across K=2–8
(silhouette = 0.31). Cluster labels passed downstream as features into Stages 2 and 3.

### Stage 2 — XGBoost Classifier
Predicts whether a patient falls in the top 20% of annual healthcare expenditure
(high_cost = 1) for their survey year. The 20% threshold is computed per year to
account for healthcare cost inflation across the 2013–2022 window.

- **Test AUC: 0.95**
- Validated with 5-fold stratified cross-validation
- Class imbalance handled with scale_pos_weight (80/20 split)
- Risk score output passed as feature to Stage 3

### Stage 3 — XGBoost Regressor
Estimates total healthcare expenditure magnitude across all patients using
log-transformed total expenditure as the target. Regression runs on the full
population — not just high-cost patients — to avoid selection bias and leverage
the complete expenditure distribution.

- **Test R²: 0.82 (log space)**
- Validated with 5-fold cross-validation
- SHAP applied for feature-level explainability
- Predictions back-transformed via exp(ŷ) − 1 for dollar estimates

---

## Key Results

| Stage | Method | Primary Metric | Result |
|-------|--------|---------------|--------|
| Segmentation | K-Means (K=2) | Silhouette Score | 0.31 |
| Classification | XGBoost | Test AUC | 0.95 |
| Regression | XGBoost | Test R² (log space) | 0.82 |

Both hypotheses supported:
- H₁ AUC > 0.80 → Actual: 0.95 ✔
- H₁ R² > 0.30 → Actual: 0.82 ✔

---

## Top Cost Drivers (SHAP)

SHAP analysis on the regression model identified the strongest predictors of
healthcare expenditure magnitude:

- Utilization index (weighted composite of office visits, ER visits, inpatient
  nights, Rx fills)
- Chronic burden score (count of diagnosed chronic conditions)
- Age
- Insurance type
- Poverty category

---

## Business Output — Intervention Routing

Each patient exits the pipeline with two actionable outputs: a risk score
(probability of being high-cost) and a predicted expenditure in dollars.

| Risk Score | Predicted Spend | Intervention |
|------------|----------------|--------------|
| High (> 0.6) | High (> $15K) | Intensive case management |
| High (> 0.6) | Moderate ($5K–$15K) | Preventive outreach + chronic care management |
| Low (< 0.3) | Low (< $5K) | Routine wellness + digital touchpoints |

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| MEPS (2013–2022) | Patient-level expenditure, utilization, demographics | meps.ahrq.gov |

**Note:** 2020 excluded due to COVID-19 causing atypical utilization and spending
patterns that would corrupt the model's understanding of normal cost drivers.

---

## Dataset

- 272,568 patients across 9 survey years (2013–2022)
- Features: age, sex, race/ethnicity, poverty category, insurance type, chronic
  condition burden, utilization index, health status, office visits, ER visits,
  inpatient nights, Rx fills
- Target (classifier): high_cost = 1 if total expenditure ≥ 80th percentile
  for that survey year
- Target (regressor): log(total_expenditure + 1)

---

## Tools & Technologies

- Python
- pandas, NumPy
- scikit-learn (K-Means, StandardScaler, cross-validation)
- XGBoost
- SHAP
- statsmodels (OLS benchmark)
- matplotlib, seaborn

---

## Methodology Notes

**Why log-transform expenditure?** Healthcare spending is extremely right-skewed —
a small number of patients incur catastrophically high costs. Log transformation
corrects this skew and produces better-calibrated regression estimates. The R² of
0.82 is measured in log space; back-transformed dollar predictions are evaluated
using RMSE.

**Why regression on all patients?** Restricting regression to high-cost patients
only removes 80% of training data and introduces selection bias — the model never
sees the full expenditure distribution and systematically underestimates borderline
patients.

**Why per-year high-cost threshold?** A fixed dollar cutoff becomes stale across
a 9-year window due to healthcare cost inflation. Computing the 80th percentile
per survey year ensures a consistent 20% positive class rate annually.
