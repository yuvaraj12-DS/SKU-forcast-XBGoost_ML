# SKU-forcast-XBGoost_ML
SKU-level sales forecasting using Machine learning (log-transformed XGBoost model).
# SKU-Level Sales Forecasting using Log-Tuned XGBoost

This repository contains a complete pipeline for forecasting product-level sales using machine learning. The project is built around a cleaned retail dataset containing over 430K rows of sales information. Forecasting is done using a log-transformed target variable and tuned XGBoost model, designed to meet market-level performance standards.

## Dataset Overview
- **Total rows:** 436,927
- **Store code:** Single store (fixed)
- **No missing or duplicate values**
- **Wide range in `ACTUAL` quantity and `AVG_UNIT_PRICE`, with notable outliers**

## Preprocessing
- Capped extreme values in `ACTUAL` and `AVG_UNIT_PRICE` using 99th percentile
- Applied log transformation to target variable (`ACTUAL_capped → ACTUAL_log`)
- Engineered robust features using time signals and lag patterns

## Feature Engineering
- Temporal signals: `WEEK`, `MONTH`, `QUARTER`, `YEAR`
- Lag features: `LAG_1`, `LAG_2`, `LAG_MEAN_2`, `ROLL_MEAN_3`
- Price segment flag and product frequency counts
- One-hot encoded category variables

## Modeling Pipeline
- Stratified sampling via volume bins (`Low`, `Mid`, `High`) for balanced training/test split
- Initial trials with Random Forest and XGBoost
- Log-transformed target used with XGBoost
- Hyperparameter tuning done with `RandomizedSearchCV`

## Final Model (Log-Tuned XGBoost)

| Metric             | Value       |
|--------------------|-------------|
| MAE                | 1.52        |
| MAPE               | 19.07%      |
| R² Score           | 0.92        |

## SHAP Interpretability
- Feature importance led by `ROLL_MEAN_3`, followed by `LAG_MEAN_2` and `PRODUCT_FREQ`
- SHAP Summary plot used to explain individual SKU behavior and driver features

## Visualization
- Forecast vs Actual plots validate model's alignment with real sales trends
- Feature importance plots using gain and SHAP values
- Volume binning for stratified sampling overview

## Final outcome
This model demonstrates production-grade accuracy and generalization for SKU-level demand forecasting. Further refinements could include deeper target transformations, stacking models, or segment-specific forecasting.

---
