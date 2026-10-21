"""
visualization.py
================
Generates high-resolution, business-ready figures and charts for sales forecasting,
model evaluation, error analysis, and operational planning.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Visual styling setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.titlesize"] = 14
plt.rcParams["figure.titleweight"] = "bold"


def plot_historical_sales_decomposition(
    weekly_series: pd.Series,
    output_path: str = "outputs/figures/01_historical_trend_decomposition.png"
):
    """
    Plots the full historical weekly sales trajectory, moving averages,
    and key business seasonal milestones.
    """
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

    # 12-week moving average
    ma_12 = weekly_series.rolling(12).mean()
    ma_4 = weekly_series.rolling(4).mean()

    ax.plot(weekly_series.index, weekly_series.values, color="#94a3b8", alpha=0.6, linewidth=1.2, label="Actual Weekly Sales")
    ax.plot(ma_4.index, ma_4.values, color="#3b82f6", linewidth=1.8, label="4-Week Momentum Average")
    ax.plot(ma_12.index, ma_12.values, color="#1e3a8a", linewidth=2.5, label="12-Week Underlying Trend")

    # Annotate peak periods (e.g. Q4 surges)
    q4_2016_peak_date = weekly_series.loc["2016-10-01":"2016-12-31"].idxmax()
    q4_2016_peak_val = weekly_series.loc[q4_2016_peak_date]
    ax.annotate(
        f"2016 Q4 Peak\n${q4_2016_peak_val:,.0f}",
        xy=(q4_2016_peak_date, q4_2016_peak_val),
        xytext=(q4_2016_peak_date - pd.Timedelta(days=80), q4_2016_peak_val + 2000),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", color="#dc2626"
    )

    q4_2017_peak_date = weekly_series.loc["2017-10-01":"2017-12-31"].idxmax()
    q4_2017_peak_val = weekly_series.loc[q4_2017_peak_date]
    ax.annotate(
        f"2017 Q4 Surge\n${q4_2017_peak_val:,.0f}",
        xy=(q4_2017_peak_date, q4_2017_peak_val),
        xytext=(q4_2017_peak_date - pd.Timedelta(days=80), q4_2017_peak_val + 2000),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", color="#dc2626"
    )

    ax.set_title("Historical Weekly Sales Trajectory & Trend (2014 - 2017)")
    ax.set_xlabel("Order Week")
    ax.set_ylabel("Weekly Sales ($ USD)")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_seasonality_patterns(
    raw_df: pd.DataFrame,
    output_path: str = "outputs/figures/02_weekly_monthly_seasonality.png"
):
    """
    Plots two-panel seasonality breakdown:
    Left: Monthly sales volume pattern across all years (identifying the Q4 surge).
    Right: Day-of-week sales and order distribution (for operational staffing).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Monthly breakdown
    raw_df_copy = raw_df.copy()
    raw_df_copy["Month_Name"] = raw_df_copy["Order Date"].dt.strftime("%b")
    raw_df_copy["Month_Num"] = raw_df_copy["Order Date"].dt.month
    monthly_sales = raw_df_copy.groupby(["Month_Num", "Month_Name"])["Sales"].sum().reset_index()

    colors = ["#3b82f6" if m < 10 else "#dc2626" for m in monthly_sales["Month_Num"]]
    bars1 = ax1.bar(monthly_sales["Month_Name"], monthly_sales["Sales"], color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_title("Total Sales Volume by Calendar Month\n(Highlighting Q4 Holiday Surge in Red)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Cumulative Sales ($ USD)")
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 5000, f"${yval/1e3:.0f}K", ha="center", va="bottom", fontsize=8)

    # Day of Week breakdown
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    raw_df_copy["Day_Name"] = raw_df_copy["Order Date"].dt.day_name()
    dow_sales = raw_df_copy.groupby("Day_Name")["Sales"].sum().reindex(day_order).fillna(0)

    bars2 = ax2.bar(dow_sales.index, dow_sales.values, color="#10b981", edgecolor="black", linewidth=0.5)
    ax2.set_title("Sales Distribution by Day of Week\n(Operational Staffing & Order Processing Demand)")
    ax2.set_xlabel("Day of Week")
    ax2.set_ylabel("Cumulative Sales ($ USD)")
    ax2.yaxis.set_major_formatter("${x:,.0f}")
    ax2.tick_params(axis='x', rotation=30)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 5000, f"${yval/1e3:.0f}K", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: str = "outputs/figures/03_model_comparison_metrics.png"
):
    """
    Renders comparative bar charts of MAE and WAPE across models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # MAE Comparison
    bars1 = ax1.barh(metrics_df["Model"], metrics_df["MAE ($)"], color="#6366f1", edgecolor="black", linewidth=0.5)
    ax1.set_title("Mean Absolute Error (MAE in $) - Lower is Better")
    ax1.set_xlabel("MAE ($)")
    ax1.xaxis.set_major_formatter("${x:,.0f}")
    ax1.invert_yaxis()
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 100, bar.get_y() + bar.get_height()/2, f"${w:,.0f}", ha="left", va="center", fontsize=9, fontweight="bold")

    # WAPE / MAPE Comparison
    bars2 = ax2.barh(metrics_df["Model"], metrics_df["WAPE (%)"], color="#0ea5e9", edgecolor="black", linewidth=0.5)
    ax2.set_title("Weighted Absolute % Error (WAPE %) - Lower is Better")
    ax2.set_xlabel("WAPE (%)")
    ax2.xaxis.set_major_formatter("{x:.1f}%")
    ax2.invert_yaxis()
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_test_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    residual_std: float,
    output_path: str = "outputs/figures/04_actual_vs_predicted_test.png"
):
    """
    Plots the out-of-sample test evaluation: Actual vs Predicted with 80% and 95% error bands.
    """
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

    dates = y_test.index
    lower_80 = np.maximum(0, y_pred - 1.28 * residual_std)
    upper_80 = y_pred + 1.28 * residual_std
    lower_95 = np.maximum(0, y_pred - 1.96 * residual_std)
    upper_95 = y_pred + 1.96 * residual_std

    ax.fill_between(dates, lower_95, upper_95, color="#3b82f6", alpha=0.15, label="95% Confidence Interval")
    ax.fill_between(dates, lower_80, upper_80, color="#3b82f6", alpha=0.25, label="80% Confidence Interval")
    ax.plot(dates, y_test.values, marker="o", color="#0f172a", linewidth=2.0, label="Actual Observed Sales")
    ax.plot(dates, y_pred, marker="s", color="#2563eb", linestyle="--", linewidth=2.0, label=f"Model Forecast ({model_name})")

    ax.set_title(f"Out-of-Sample Test Evaluation: Actual vs Forecasted Sales\nModel: {model_name}")
    ax.set_xlabel("Order Week")
    ax.set_ylabel("Weekly Sales ($ USD)")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_future_forecast(
    historical_series: pd.Series,
    forecast_df: pd.DataFrame,
    model_name: str,
    output_path: str = "outputs/figures/05_future_sales_forecast_90days.png"
):
    """
    Plots the upcoming 12-week / 90-day future sales forecast with uncertainty bands
    alongside recent historical context.
    """
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

    recent_history = historical_series.iloc[-26:]
    ax.plot(recent_history.index, recent_history.values, marker="o", color="#334155", linewidth=2.0, label="Recent Historical Sales (Last 26 Weeks)")

    f_dates = forecast_df.index
    ax.fill_between(f_dates, forecast_df["Lower_95"], forecast_df["Upper_95"], color="#10b981", alpha=0.15, label="95% Forecast Range (Worst to Best Case)")
    ax.fill_between(f_dates, forecast_df["Lower_80"], forecast_df["Upper_80"], color="#10b981", alpha=0.25, label="80% Forecast Range (Expected Operating Range)")
    ax.plot(f_dates, forecast_df["Forecast"], marker="D", color="#059669", linewidth=2.5, linestyle="-", label=f"Recommended Baseline Forecast ({model_name})")

    # Connect last historical point to first forecast point
    ax.plot([recent_history.index[-1], f_dates[0]], [recent_history.values[-1], forecast_df["Forecast"].iloc[0]], color="#059669", linestyle=":")

    ax.set_title(f"Executive 12-Week Forward Demand Forecast\nNext Quarter Projected Revenue: ${forecast_df['Forecast'].sum():,.0f}")
    ax.set_xlabel("Week Start Date")
    ax.set_ylabel("Projected Weekly Sales ($ USD)")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 12,
    output_path: str = "outputs/figures/06_feature_importance.png"
):
    """
    Plots top driving features influencing sales forecasts.
    """
    top_df = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(top_df["Feature"], top_df["Mean_Importance"], color="#8b5cf6", edgecolor="black", linewidth=0.5)

    ax.set_title(f"Top {top_n} Factors Driving Sales Forecasts\n(Tree-based Feature Importance)")
    ax.set_xlabel("Relative Feature Importance Score")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}", ha="left", va="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_category_breakdown_forecast(
    category_history: pd.DataFrame,
    category_forecasts: Dict[str, pd.DataFrame],
    output_path: str = "outputs/figures/07_category_demand_forecast.png"
):
    """
    Plots category demand trajectory and upcoming forecast by department.
    """
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

    palette = {"Technology": "#3b82f6", "Furniture": "#f59e0b", "Office Supplies": "#10b981"}

    for cat, col in palette.items():
        if cat in category_history.columns:
            recent_hist = category_history[cat].iloc[-26:]
            ax.plot(recent_hist.index, recent_hist.values, color=col, alpha=0.5, linewidth=1.5, linestyle="--", label=f"{cat} (Recent History)")
        if cat in category_forecasts:
            f_df = category_forecasts[cat]
            ax.plot(f_df.index, f_df["Forecast"], color=col, marker="o", linewidth=2.5, label=f"{cat} (12-Wk Forecast)")

    ax.set_title("Department-Level Forward Demand Forecast (Technology vs Furniture vs Office Supplies)")
    ax.set_xlabel("Order Week")
    ax.set_ylabel("Weekly Sales ($ USD)")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
