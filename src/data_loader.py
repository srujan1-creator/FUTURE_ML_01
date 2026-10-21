"""
data_loader.py
==============
Module for loading, cleaning, validating, and resampling historical sales data.
Designed for Superstore and general transactional sales data.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np


def load_raw_data(filepath: str = "data/Sample - Superstore.csv") -> pd.DataFrame:
    """
    Loads raw CSV transaction data, cleans columns, parses dates, and validates data hygiene.
    """
    # Superstore is typically encoded in windows-1252 or utf-8
    try:
        df = pd.read_csv(filepath, encoding="windows-1252")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="utf-8")

    # Clean whitespace in column names
    df.columns = [col.strip() for col in df.columns]

    # Parse dates
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    # Ensure numerical types
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0.0)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
    df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0.0)
    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0.0)

    # Sort chronologically
    df = df.sort_values("Order Date").reset_index(drop=True)

    return df


def aggregate_daily(df: pd.DataFrame, category: Optional[str] = None) -> pd.DataFrame:
    """
    Aggregates transactions into continuous daily time series.
    Missing calendar dates are filled with 0 sales and orders.
    """
    filtered_df = df if category is None else df[df["Category"] == category]

    daily = filtered_df.groupby("Order Date").agg(
        Sales=("Sales", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique"),
        Profit=("Profit", "sum"),
        Avg_Discount=("Discount", "mean")
    )

    # Reindex to a complete continuous calendar range
    full_date_range = pd.date_range(start=df["Order Date"].min(), end=df["Order Date"].max(), freq="D")
    daily = daily.reindex(full_date_range)

    # Fill NaNs for zero-volume days
    daily["Sales"] = daily["Sales"].fillna(0.0)
    daily["Quantity"] = daily["Quantity"].fillna(0)
    daily["Orders"] = daily["Orders"].fillna(0)
    daily["Profit"] = daily["Profit"].fillna(0.0)
    daily["Avg_Discount"] = daily["Avg_Discount"].fillna(0.0)
    daily.index.name = "Date"

    return daily


def aggregate_weekly(df: pd.DataFrame, category: Optional[str] = None) -> pd.DataFrame:
    """
    Aggregates daily time series into weekly totals (Monday-anchored).
    Weekly aggregation smooths daily zero-sales noise and aligns with
    business procurement, inventory replenishment, and staffing schedules.
    """
    daily = aggregate_daily(df, category=category)
    weekly = daily.resample("W-MON").agg({
        "Sales": "sum",
        "Quantity": "sum",
        "Orders": "sum",
        "Profit": "sum",
        "Avg_Discount": "mean"
    })
    weekly.index.name = "Week_Start"
    return weekly


def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provides weekly sales breakdown across product categories (Furniture, Office Supplies, Technology).
    """
    cat_daily = df.pivot_table(index="Order Date", columns="Category", values="Sales", aggfunc="sum").fillna(0)
    full_idx = pd.date_range(start=df["Order Date"].min(), end=df["Order Date"].max(), freq="D")
    cat_daily = cat_daily.reindex(full_idx, fill_value=0.0)
    cat_weekly = cat_daily.resample("W-MON").sum()
    cat_weekly.index.name = "Week_Start"
    return cat_weekly


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Data loaded successfully: {len(df)} records from {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
    daily = aggregate_daily(df)
    print(f"Daily series: {len(daily)} days, Total Sales: ${daily['Sales'].sum():,.2f}")
    weekly = aggregate_weekly(df)
    print(f"Weekly series: {len(weekly)} weeks, Mean Weekly Sales: ${weekly['Sales'].mean():,.2f}")
