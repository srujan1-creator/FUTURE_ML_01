"""
run_pipeline.py
===============
End-to-end execution pipeline for Sales and Demand Forecasting.
Executes data loading, feature engineering, model training, benchmarking,
error analysis, future 12-week forecasting, inventory/cash flow planning,
and generates publication-ready figures.
"""

import os
import numpy as np
import pandas as pd

from src.data_loader import load_raw_data, aggregate_daily, aggregate_weekly, get_category_breakdown
from src.feature_engineering import build_feature_pipeline
from src.models import ModelTrainer
from src.evaluate import evaluate_all_models, analyze_residuals
from src.inventory_planning import calculate_inventory_policy, calculate_cash_flow_projections, calculate_staffing_requirements
from src.visualization import (
    plot_historical_sales_decomposition,
    plot_seasonality_patterns,
    plot_model_comparison,
    plot_test_actual_vs_predicted,
    plot_future_forecast,
    plot_feature_importance,
    plot_category_breakdown_forecast
)


def run_pipeline():
    print("=" * 75)
    print("      FUTURE INTERNS - SALES & DEMAND FORECASTING SYSTEM (2026)")
    print("=" * 75)

    # 1. Load and Clean Data
    print("\n[Step 1/7] Loading and cleaning historical sales data...")
    raw_df = load_raw_data("data/Sample - Superstore.csv")
    print(f" -> Loaded {len(raw_df):,} transaction records.")
    print(f" -> Time Span: {raw_df['Order Date'].min().strftime('%Y-%m-%d')} to {raw_df['Order Date'].max().strftime('%Y-%m-%d')}")
    print(f" -> Total Historical Sales: ${raw_df['Sales'].sum():,.2f}")

    # 2. Resample Time Series
    print("\n[Step 2/7] Aggregating continuous calendar time series...")
    daily_df = aggregate_daily(raw_df)
    weekly_df = aggregate_weekly(raw_df)
    cat_breakdown = get_category_breakdown(raw_df)
    print(f" -> Continuous Daily Records: {len(daily_df)} days")
    print(f" -> Weekly Aggregated Records: {len(weekly_df)} weeks (Mean: ${weekly_df['Sales'].mean():,.2f}/week)")

    # 3. Feature Engineering
    print("\n[Step 3/7] Engineering temporal, cyclical, lag, and momentum features...")
    featured_df, feature_cols = build_feature_pipeline(weekly_df, target_col="Sales", drop_na=True)
    print(f" -> Engineered {len(feature_cols)} features across {len(featured_df)} clean historical weeks.")
    print(f" -> Features: {', '.join(feature_cols[:8])}... (+ {len(feature_cols)-8} more)")

    # 4. Train / Test Split (Time-based chronological split)
    print("\n[Step 4/7] Splitting data chronologically into Train and Out-of-Sample Test...")
    test_horizon = 26  # Last 26 weeks (~6 months)
    train_df = featured_df.iloc[:-test_horizon]
    test_df = featured_df.iloc[-test_horizon:]

    X_train = train_df[feature_cols]
    y_train = train_df["Sales"]
    X_test = test_df[feature_cols]
    y_test = test_df["Sales"]

    print(f" -> Train Period: {train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')} ({len(train_df)} weeks)")
    print(f" -> Test Period:  {test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')} ({len(test_df)} weeks)")

    # 5. Model Training & Evaluation
    print("\n[Step 5/7] Training multiple models and evaluating out-of-sample performance...")
    trainer = ModelTrainer(random_state=42)
    trainer.fit_all(X_train, y_train)

    predictions = trainer.predict_all(X_test)
    metrics_df = evaluate_all_models(y_test, predictions)

    print("\n" + "-" * 75)
    print("                    MODEL BENCHMARK RESULTS")
    print("-" * 75)
    print(metrics_df.to_string(index=False))
    print("-" * 75)

    os.makedirs("outputs", exist_ok=True)
    metrics_df.to_csv("outputs/model_evaluation_metrics.csv", index=False)
    print(" -> Saved metrics to 'outputs/model_evaluation_metrics.csv'")

    # Select best model (lowest MAE)
    best_model_name = metrics_df.iloc[0]["Model"]
    best_mae = metrics_df.iloc[0]["MAE ($)"]
    print(f" -> Best Performing Model: '{best_model_name}' (MAE: ${best_mae:,.2f})")

    # Feature importances
    imp_df = trainer.get_feature_importances(feature_cols)
    imp_df.to_csv("outputs/feature_importances.csv", index=False)

    # 6. Future Multi-Step Forecast (Next 12 Weeks / Q1)
    print("\n[Step 6/7] Generating 12-week forward recursive forecast & prediction intervals...")
    future_forecast = trainer.recursive_forecast(
        model_name=best_model_name,
        historical_series=weekly_df["Sales"],
        horizon_weeks=12
    )

    future_forecast.to_csv("outputs/future_forecast_results.csv")
    total_proj_sales = future_forecast["Forecast"].sum()
    avg_proj_sales = future_forecast["Forecast"].mean()
    print(f" -> Total 12-Week Projected Revenue: ${total_proj_sales:,.2f}")
    print(f" -> Projected Weekly Run-Rate: ${avg_proj_sales:,.2f}/week")

    # Category-level forecasts
    cat_forecasts = {}
    for cat in ["Technology", "Furniture", "Office Supplies"]:
        if cat in cat_breakdown.columns:
            cat_series = cat_breakdown[cat]
            cat_feat_df, cat_cols = build_feature_pipeline(cat_series.to_frame(name="Sales"), target_col="Sales", drop_na=True)
            cat_trainer = ModelTrainer(random_state=42)
            cat_trainer.fit_all(cat_feat_df[cat_cols], cat_feat_df["Sales"])
            cat_f = cat_trainer.recursive_forecast("Random Forest", cat_series, horizon_weeks=12)
            cat_forecasts[cat] = cat_f

    cat_f_summary = pd.DataFrame({
        cat: cat_forecasts[cat]["Forecast"] for cat in cat_forecasts
    })
    cat_f_summary.to_csv("outputs/category_forecast_results.csv")

    # Operational Planning calculations
    print("\n[Step 7/7] Computing inventory policies, cash flows, and staffing needs...")
    inv_policy = calculate_inventory_policy(future_forecast["Forecast"], lead_time_weeks=2, service_level=0.95)
    cash_flow_df = calculate_cash_flow_projections(future_forecast)
    staffing_df = calculate_staffing_requirements(future_forecast)

    # Combine operational plan
    op_plan = future_forecast[["Forecast", "Lower_80", "Upper_80"]].copy()
    op_plan["COGS_Requirement"] = cash_flow_df["COGS_Procurement_Expense"]
    op_plan["Net_Cash_Contribution"] = cash_flow_df["Gross_Profit_Contribution"]
    op_plan["Labor_Hours_Needed"] = staffing_df["Required_Labor_Hours"]
    op_plan["FTE_Headcount"] = staffing_df["Recommended_FTE_Headcount"]
    op_plan.to_csv("outputs/operational_business_plan.csv")

    print(f" -> Operational Inventory Parameters (95% Service Level, 2-Wk Lead Time):")
    print(f"    - Lead Time Demand:        ${inv_policy['lead_time_demand']:,.2f}")
    print(f"    - Safety Buffer Stock:     ${inv_policy['safety_stock']:,.2f}")
    print(f"    - Target Reorder Point:    ${inv_policy['reorder_point']:,.2f}")
    print(f"    - Working Capital Reserve: ${inv_policy['recommended_capital_buffer']:,.2f}")

    # Visualizations
    print("\n[Generating Visualizations] Rendering executive figures...")
    plot_historical_sales_decomposition(weekly_df["Sales"])
    plot_seasonality_patterns(raw_df)
    plot_model_comparison(metrics_df)
    plot_test_actual_vs_predicted(
        y_test=y_test,
        y_pred=predictions[best_model_name],
        model_name=best_model_name,
        residual_std=trainer.residual_std.get(best_model_name, 2500.0)
    )
    plot_future_forecast(weekly_df["Sales"], future_forecast, model_name=best_model_name)
    plot_feature_importance(imp_df, top_n=10)
    plot_category_breakdown_forecast(cat_breakdown, cat_forecasts)

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("All outputs, tables, and figures saved to 'outputs/' directory.")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
