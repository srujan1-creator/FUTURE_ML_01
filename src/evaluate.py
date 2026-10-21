"""
evaluate.py
===========
Module for computing forecasting accuracy metrics, residual diagnostics,
and producing clear comparison reports.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> dict:
    """
    Computes standard business forecasting metrics:
    MAE, RMSE, MAPE, WAPE, R2, and Error Bias.
    """
    y_true_vals = y_true.values
    errors = y_true_vals - y_pred

    mae = mean_absolute_error(y_true_vals, y_pred)
    rmse = root_mean_squared_error(y_true_vals, y_pred)

    # Avoid zero division in MAPE
    nonzero_mask = y_true_vals > 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs(errors[nonzero_mask]) / y_true_vals[nonzero_mask]) * 100.0
    else:
        mape = np.nan

    # WAPE (Weighted Absolute Percentage Error - retail industry benchmark)
    wape = (np.sum(np.abs(errors)) / np.sum(y_true_vals)) * 100.0
    r2 = r2_score(y_true_vals, y_pred)
    mean_bias = np.mean(errors)  # Positive = under-predicting, Negative = over-predicting

    return {
        "Model": model_name,
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "WAPE (%)": round(wape, 2),
        "R2": round(r2, 3),
        "Mean Bias ($)": round(mean_bias, 2)
    }


def evaluate_all_models(
    y_true: pd.Series,
    predictions_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Evaluates a dictionary of model predictions and outputs a ranked comparison DataFrame.
    """
    records = []
    for model_name, y_pred in predictions_dict.items():
        metrics = compute_metrics(y_true, y_pred, model_name)
        records.append(metrics)

    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.sort_values("MAE ($)", ascending=True).reset_index(drop=True)
    return metrics_df


def analyze_residuals(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Performs residual error diagnostics.
    """
    residuals = y_true.values - y_pred
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "skewness": float(pd.Series(residuals).skew()),
        "under_predict_count": int(np.sum(residuals > 0)),
        "over_predict_count": int(np.sum(residuals < 0))
    }
