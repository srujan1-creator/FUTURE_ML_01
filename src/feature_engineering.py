"""
feature_engineering.py
======================
Module for generating time-based, cyclical, trend, lag, and rolling window features
for sales and demand forecasting.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates calendar, cyclical, trend, and retail seasonal indicator features.
    """
    data = df.copy()
    idx = data.index

    # 1. Trend indicators
    data["trend"] = np.arange(len(data))
    data["trend_sq"] = (data["trend"] / len(data)) ** 2

    # 2. Calendar attributes
    data["month"] = idx.month
    data["quarter"] = idx.quarter
    data["weekofyear"] = idx.isocalendar().week.astype(int)
    data["year"] = idx.year

    # 3. Cyclical encoding (smooth periodic representation)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["week_sin"] = np.sin(2 * np.pi * data["weekofyear"] / 52)
    data["week_cos"] = np.cos(2 * np.pi * data["weekofyear"] / 52)

    # 4. Business and Retail Calendar Events
    data["is_q4"] = (data["quarter"] == 4).astype(int)
    data["is_back_to_school"] = data["weekofyear"].isin([32, 33, 34, 35, 36]).astype(int)
    data["is_jan_dip"] = (data["month"] == 1).astype(int)
    data["is_quarter_end"] = data["weekofyear"].isin([13, 26, 39, 52]).astype(int)

    return data


def generate_lag_features(df: pd.DataFrame, target_col: str = "Sales", lags: List[int] = None) -> pd.DataFrame:
    """
    Generates autoregressive lag features for the target variable.
    """
    if lags is None:
        lags = [1, 2, 4, 8, 12, 52]

    data = df.copy()
    for lag in lags:
        data[f"lag_{lag}"] = data[target_col].shift(lag)

    return data


def generate_rolling_features(df: pd.DataFrame, target_col: str = "Sales") -> pd.DataFrame:
    """
    Generates rolling window statistics (mean, std, min, max, momentum) using
    lagged target values to strictly prevent future data leakage.
    """
    data = df.copy()
    shifted_target = data[target_col].shift(1)

    # 4-week window (~1 month)
    data["rolling_mean_4"] = shifted_target.rolling(window=4).mean()
    data["rolling_std_4"] = shifted_target.rolling(window=4).std()
    data["rolling_min_4"] = shifted_target.rolling(window=4).min()
    data["rolling_max_4"] = shifted_target.rolling(window=4).max()

    # 12-week window (~1 quarter)
    data["rolling_mean_12"] = shifted_target.rolling(window=12).mean()
    data["rolling_std_12"] = shifted_target.rolling(window=12).std()

    # Momentum indicator: Ratio of last week to 4-week moving average
    data["momentum_4"] = data[f"lag_1"] / (data["rolling_mean_4"] + 1e-5)

    return data


def build_feature_pipeline(
    df: pd.DataFrame,
    target_col: str = "Sales",
    drop_na: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete end-to-end feature pipeline.
    Returns:
        featured_df: DataFrame with all generated features.
        feature_names: List of independent feature column names.
    """
    data = generate_time_features(df)
    data = generate_lag_features(data, target_col=target_col)
    data = generate_rolling_features(data, target_col=target_col)

    if drop_na:
        data = data.dropna().copy()

    feature_cols = [
        col for col in data.columns
        if col not in [target_col, "Quantity", "Orders", "Profit", "Avg_Discount", "Date", "Week_Start"]
    ]

    return data, feature_cols


def create_future_feature_row(
    history_series: pd.Series,
    future_date: pd.Timestamp,
    base_trend_idx: int,
    total_len_for_trend: int
) -> dict:
    """
    Constructs a feature dictionary for a single future forecasting step
    using historical and recursively generated predictions.
    """
    row = {}

    # Trend
    row["trend"] = base_trend_idx
    row["trend_sq"] = (base_trend_idx / total_len_for_trend) ** 2

    # Calendar
    month = future_date.month
    quarter = future_date.quarter
    week = int(future_date.isocalendar().week)
    year = future_date.year

    row["month"] = month
    row["quarter"] = quarter
    row["weekofyear"] = week
    row["year"] = year

    # Cyclical
    row["month_sin"] = np.sin(2 * np.pi * month / 12)
    row["month_cos"] = np.cos(2 * np.pi * month / 12)
    row["week_sin"] = np.sin(2 * np.pi * week / 52)
    row["week_cos"] = np.cos(2 * np.pi * week / 52)

    # Seasonal indicators
    row["is_q4"] = int(quarter == 4)
    row["is_back_to_school"] = int(week in [32, 33, 34, 35, 36])
    row["is_jan_dip"] = int(month == 1)
    row["is_quarter_end"] = int(week in [13, 26, 39, 52])

    # Lags
    for l in [1, 2, 4, 8, 12, 52]:
        if len(history_series) >= l:
            row[f"lag_{l}"] = history_series.iloc[-l]
        else:
            row[f"lag_{l}"] = history_series.mean()

    # Rolling stats
    last_4 = history_series.iloc[-4:] if len(history_series) >= 4 else history_series
    last_12 = history_series.iloc[-12:] if len(history_series) >= 12 else history_series

    row["rolling_mean_4"] = last_4.mean()
    row["rolling_std_4"] = last_4.std(ddof=1) if len(last_4) > 1 else 0.0
    row["rolling_min_4"] = last_4.min()
    row["rolling_max_4"] = last_4.max()

    row["rolling_mean_12"] = last_12.mean()
    row["rolling_std_12"] = last_12.std(ddof=1) if len(last_12) > 1 else 0.0

    row["momentum_4"] = row["lag_1"] / (row["rolling_mean_4"] + 1e-5)

    return row
