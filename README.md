# Sales & Demand Forecasting System for Businesses
**Future Interns — Machine Learning Task 1 (2026)**  
**Author:** Srujan Shetty  
**Dataset:** Retail Enterprise Sales (Superstore Benchmark)

---

## 🚀 Overview

This repository contains a full-stack, enterprise-grade machine learning system for sales and demand forecasting. Built to address real business problems, this system bridges the gap between predictive modeling and actionable operational decisions:

- **Data Cleaning & Continuity**: Resolves calendar gaps and imputes zero-order periods.
- **Time-Based Feature Engineering**: Cyclical Fourier encodings, retail calendar indicators (Q4 peak, Back to School, January dip), autoregressive lags (1, 2, 4, 8, 12, 52 weeks), and rolling momentum metrics.
- **Multi-Model Benchmarking**: Evaluates Seasonal Naive Baseline, Ridge Regression, Random Forest, XGBoost, and Weighted Ensemble with strictly chronological train/test splits.
- **Operational Planning Integration**: Automatically translates forecasts into **Safety Stock**, **Reorder Points (ROP)**, **Cash Flow projections**, and **Staffing FTE requirements**.
- **Interactive Dashboard**: A modern, interactive Streamlit application with scenario planning controls.

---

## 📁 Repository Structure

```text
Future intern/
├── data/
│   └── Sample - Superstore.csv              # Historical transaction dataset (9,994 records)
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── data_loader.py                       # Data loading, date parsing & calendar aggregation
│   ├── feature_engineering.py               # Lags, rolling stats, cyclical & retail features
│   ├── models.py                            # Model trainer, recursive forecasting & confidence bands
│   ├── evaluate.py                          # MAE, RMSE, MAPE, WAPE, R² & error diagnostics
│   ├── inventory_planning.py                # Safety stock, ROP, cash flow & labor staffing rules
│   └── visualization.py                     # High-resolution plot generators
├── notebooks/
│   └── sales_forecasting_analysis.ipynb     # Interactive Jupyter walkthrough notebook
├── outputs/
│   ├── figures/                             # High-resolution executive charts
│   │   ├── 01_historical_trend_decomposition.png
│   │   ├── 02_weekly_monthly_seasonality.png
│   │   ├── 03_model_comparison_metrics.png
│   │   ├── 04_actual_vs_predicted_test.png
│   │   ├── 05_future_sales_forecast_90days.png
│   │   ├── 06_feature_importance.png
│   │   └── 07_category_demand_forecast.png
│   ├── model_evaluation_metrics.csv         # Comparative benchmark table
│   ├── feature_importances.csv              # Ranked feature contribution scores
│   ├── future_forecast_results.csv          # 12-week forward prediction table with bands
│   ├── category_forecast_results.csv        # Category-level predictions (Tech, Furn, Office)
│   └── operational_business_plan.csv        # Weekly procurement, cash flow & staffing schedule
├── app.py                                   # Interactive Streamlit Web Application
├── run_pipeline.py                          # One-click end-to-end execution script
├── BUSINESS_STAKEHOLDER_REPORT.md           # Executive presentation for store owners & founders
├── requirements.txt                         # Python package dependencies
└── README.md                                # Project documentation
```

---

## ⚡ Quick Start: How to Run and View the Project

### 1. View Interactive Web Dashboard
Launch the Streamlit app to explore forecasts with interactive sliders for lead time, service level, and department filtering:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run End-to-End Pipeline
To re-run the entire pipeline (data cleaning, feature generation, model training, evaluation, future forecasting, and figure generation):
```bash
python run_pipeline.py
```

### 3. Open Jupyter Notebook
To inspect the code and interactive plots step-by-step:
```bash
jupyter notebook notebooks/sales_forecasting_analysis.ipynb
```

### 4. Inspect Visualizations and CSV Reports
All generated assets are directly accessible on disk:
- **Visualizations**: [outputs/figures/](file:///c:/Future%20intern/outputs/figures/)
- **Business Plan**: [outputs/operational_business_plan.csv](file:///c:/Future%20intern/outputs/operational_business_plan.csv)
- **Model Metrics**: [outputs/model_evaluation_metrics.csv](file:///c:/Future%20intern/outputs/model_evaluation_metrics.csv)
- **Executive Report**: [BUSINESS_STAKEHOLDER_REPORT.md](file:///c:/Future%20intern/BUSINESS_STAKEHOLDER_REPORT.md)

---

## 📊 Benchmark Model Performance

Evaluated on an out-of-sample 26-week test period:

| Model | MAE ($) | RMSE ($) | WAPE (%) | $R^2$ Score | Mean Bias ($) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **$5,434.99** | **$7,087.59** | **30.16%** | **0.286** | +$1,666.85 |
| **Ensemble (RF + XGB + Ridge)** | $5,467.76 | $7,088.33 | 30.34% | 0.285 | +$1,610.43 |
| **XGBoost Regressor** | $5,518.80 | $7,156.57 | 30.63% | 0.272 | +$1,598.10 |
| **Ridge Regression** | $5,882.62 | $7,127.55 | 32.64% | 0.277 | +$1,412.00 |
| **Seasonal Naive (Baseline)** | $6,920.24 | $9,511.79 | 38.40% | -0.287 | +$3,464.70 |

*Result: Machine learning models reduce forecasting error by **21.5%** over naive historical guessing.*

---

## 💼 Operational Business Takeaways

1. **Safety Stock Recommendation**: Maintain **$3,459.49** in buffer stock to guarantee a **95% customer service level** against 2-week supplier delivery lead times.
2. **Reorder Point (ROP)**: Place supplier purchase orders whenever current stock drops to **$24,882.52**.
3. **Working Capital Requirement**: Allocate **$83,549.82** in procurement capital across the next 12 weeks to support anticipated sales run-rates.
4. **Staffing Plan**: Warehouse fulfillment requires 1 full-time worker for normal weeks (~40 hrs), expanding to an additional part-time worker (+18 hrs) during the mid-March quarter-end volume peak.
