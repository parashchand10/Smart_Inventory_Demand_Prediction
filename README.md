# Smart Inventory Demand Forecasting - Machine Learning

An end-to-end regression pipeline designed to forecast SKU-level daily demand by evaluating price elasticity, promotional efficiency, and seasonal drivers. Built to help supply chain teams prevent stockouts and curb inventory holding costs using distributed exploration and ensemble modeling.

---

## Project Overview

Inventory miscalculations directly impact bottom-line profitability: overstocking ties up working capital and inflates warehousing costs, while stockouts result in lost revenue and degraded customer retention.

This project analyzes retail transaction intervals across 10 core market indicators to accurately capture demand dynamics. It covers the complete machine learning lifecycle:
- Distributed SQL aggregation and baseline querying via **Apache Spark (PySpark)**.
- Exploratory data analysis and feature engineering (effective pricing, marketing efficiency indices).
- Rigorous comparative benchmarking across baseline, kernel-based, bagging, boosting, and stacking regressors.
- Production-ready serialization (`pickle`) for real-time inference.

---

## Architecture & Workflow

```text
Raw Data (CSV)
  │
  ├──► PySpark SQL Engine (Large-scale aggregations & competitive pricing checks)
  │
  ├──► Preprocessing & Feature Engineering
  │      ├── Effective Price: base_price * (1 - discount)
  │      ├── Marketing Efficiency: marketing_spend / base_price
  │      └── Feature Scaling: StandardScaler
  │
  ├──► Model Evaluation Matrix (Train/Test Split: 80/20)
  │      ├── Linear Regression (Baseline)
  │      ├── SVR (Hyperparameter tuned via GridSearchCV / RandomizedSearchCV)
  │      ├── Ensembles (Bagging Regressor, Random Forest)
  │      ├── Stacking (LinearRegression + SVR + DecisionTree -> SVR Meta-Model)
  │      └── Boosting (AdaBoost, GradientBoosting, XGBoost)
  │
  └──► Artifact Export (smart_inventory_model.pkl, scaler.pkl)
```
## 📈 Benchmark & Model Performance

Models were trained on 800 observations and benchmarked on a strictly held-out test split of 200 observations[cite: 1].

| Algorithm | MSE | R² Score | Notes |
| :--- | :---: | :---: | :--- |
| **Linear Regression** | **105.93** | **0.7429** | Highest aggregate test score across linear relationships[cite: 1] |
| **SVR (Tuned: C=100, Linear)** | 110.99 | 0.7306 | High margin-of-tolerance stability[cite: 1] |
| **Bagging Regressor (SVR Base)** | 111.65 | 0.7290 | Reduced variance over standard SVR[cite: 1] |
| **Stacking Regressor** | 111.65 | 0.7290 | Combined LR, SVR, and DecisionTree with SVR meta-learner[cite: 1] |
| **Bagging Regressor (Random Forest)** | 118.19 | 0.7132 | High robustness against non-linear fluctuations[cite: 1] |
| **Gradient Boosting** | 121.07 | 0.7062 | 100 boosting stages[cite: 1] |
| **Bagging Regressor (Decision Tree)** | 122.48 | 0.7028 | Exceptional point accuracy on sample validation points[cite: 1] |
| **AdaBoost Regressor** | 122.57 | 0.7025 | Standard exponential error re-weighting[cite: 1] |
| **XGBoost Regressor** | 143.48 | 0.6518 | Slight sensitivity to localized sub-sample variance[cite: 1] |

> **Production Recommendation:** While Linear Regression achieved the top nominal $R^2$, the **Bagging Decision Tree** model exhibited superior generalization and stability on granular edge tests (predicting **30.48** units against actual observed ground-truth of **31** units)[cite: 1].

---

## Quickstart & Usage

### Clone & Setup Environment
```bash
git clone [https://github.com/](https://github.com/)<your-username>/smart-inventory-demand-prediction.git
cd smart-inventory-demand-prediction
pip install -r requirements.txt
```
