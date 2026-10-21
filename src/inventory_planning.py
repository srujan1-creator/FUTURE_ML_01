"""
inventory_planning.py
=====================
Translates machine learning demand forecasts into operational business decisions:
Inventory levels (Safety Stock, Reorder Points), Cash Flow projections, and Staffing schedules.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def calculate_inventory_policy(
    forecast_series: pd.Series,
    lead_time_weeks: int = 2,
    service_level: float = 0.95,
    unit_cost_estimate: float = 50.0
) -> Dict[str, Any]:
    """
    Calculates operational inventory parameters based on forecast demand:
    - Expected Lead Time Demand
    - Safety Stock
    - Reorder Point (ROP)
    - Working Capital Required for Buffer Stock

    Parameters:
        forecast_series: Weekly forecasted demand (in $ or units)
        lead_time_weeks: Supplier delivery lead time in weeks
        service_level: Desired cycle service level (0.90, 0.95, 0.99)
        unit_cost_estimate: Average cost per unit (for dollar-to-unit calculations)
    """
    # Z-scores for standard normal distribution
    z_table = {0.90: 1.282, 0.95: 1.645, 0.98: 2.054, 0.99: 2.326}
    z_score = z_table.get(service_level, 1.645)

    mean_weekly_demand = float(forecast_series.mean())
    std_weekly_demand = float(forecast_series.std(ddof=1)) if len(forecast_series) > 1 else (mean_weekly_demand * 0.25)

    # Lead time demand
    lead_time_demand = mean_weekly_demand * lead_time_weeks

    # Safety stock = Z * sigma * sqrt(L)
    safety_stock = z_score * std_weekly_demand * np.sqrt(lead_time_weeks)

    # Reorder point = Lead time demand + Safety Stock
    reorder_point = lead_time_demand + safety_stock

    # Maximum stock target (Reorder Point + 2 weeks cycle stock)
    max_stock_level = reorder_point + (2 * mean_weekly_demand)

    # Estimated inventory capital buffer
    capital_buffer = safety_stock * 0.65  # assuming wholesale COGS is ~65% of retail price

    return {
        "service_level_pct": service_level * 100,
        "lead_time_weeks": lead_time_weeks,
        "mean_weekly_demand": round(mean_weekly_demand, 2),
        "lead_time_demand": round(lead_time_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "max_stock_level": round(max_stock_level, 2),
        "recommended_capital_buffer": round(capital_buffer, 2)
    }


def calculate_cash_flow_projections(
    forecast_df: pd.DataFrame,
    gross_margin_pct: float = 0.35,
    cogs_pct: float = 0.65,
    payment_collection_delay_weeks: int = 1
) -> pd.DataFrame:
    """
    Translates weekly sales forecast into cash inflow, procurement outflow,
    and net operational cash flow.
    """
    cf = pd.DataFrame(index=forecast_df.index)
    cf["Forecasted_Revenue"] = forecast_df["Forecast"]
    cf["COGS_Procurement_Expense"] = forecast_df["Forecast"] * cogs_pct
    cf["Gross_Profit_Contribution"] = forecast_df["Forecast"] * gross_margin_pct

    # Cash inflow delayed by credit card / invoice settlement
    cf["Projected_Cash_Inflow"] = cf["Forecasted_Revenue"].shift(payment_collection_delay_weeks).fillna(cf["Forecasted_Revenue"].iloc[0])
    cf["Net_Weekly_Cashflow"] = cf["Projected_Cash_Inflow"] - cf["COGS_Procurement_Expense"]

    return cf


def calculate_staffing_requirements(
    forecast_df: pd.DataFrame,
    revenue_per_labor_hour: float = 250.0,
    hours_per_full_time_worker: float = 40.0
) -> pd.DataFrame:
    """
    Translates sales volume into labor hour requirements and required headcount
    for warehouse fulfillment and customer service.
    """
    staff = pd.DataFrame(index=forecast_df.index)
    staff["Forecasted_Sales"] = forecast_df["Forecast"]
    staff["Required_Labor_Hours"] = (staff["Forecasted_Sales"] / revenue_per_labor_hour).round(1)
    staff["Recommended_FTE_Headcount"] = np.ceil(staff["Required_Labor_Hours"] / hours_per_full_time_worker).astype(int)
    staff["Part_Time_Buffer_Hours"] = (staff["Required_Labor_Hours"] % hours_per_full_time_worker).round(1)

    return staff
