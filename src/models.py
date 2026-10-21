"""
models.py
=========
Collection of forecasting models (Baseline, Ridge, Random Forest, XGBoost, and Ensemble)
with recursive multi-step future forecasting and prediction intervals.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import create_future_feature_row


class BaselineSeasonalNaive:
    """
    Baseline model that forecasts the target value from 52 weeks ago (annual seasonality),
    or falls back to the most recent week if 52-week lag is unavailable.
    """
    def __init__(self, lag: int = 52):
        self.lag = lag
        self.history = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.history = y.copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if f"lag_{self.lag}" in X.columns:
            preds = X[f"lag_{self.lag}"].fillna(X["lag_1"] if "lag_1" in X.columns else self.history.iloc[-1])
            return preds.values
        return np.full(len(X), self.history.iloc[-1])


class ModelTrainer:
    """
    Orchestrates training, evaluation, comparison, and recursive forecasting
    for multiple regression algorithms.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {
            "Seasonal Naive": BaselineSeasonalNaive(lag=52),
            "Ridge Regression": Ridge(alpha=10.0, random_state=random_state),
            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ),
            "XGBoost": XGBRegressor(
                n_estimators=120,
                learning_rate=0.04,
                max_depth=3,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state
            )
        }
        self.fitted_models: Dict[str, Any] = {}
        self.residual_std: Dict[str, float] = {}

    def fit_all(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits all registered models on training data.
        """
        # Fit scaler for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)

        for name, model in self.models.items():
            if name == "Seasonal Naive":
                model.fit(X_train, y_train)
                self.fitted_models[name] = model
            elif name == "Ridge Regression":
                model.fit(X_train_scaled, y_train)
                self.fitted_models[name] = model
            else:
                model.fit(X_train, y_train)
                self.fitted_models[name] = model

            # In-sample residual standard deviation
            preds_train = self.predict_single(name, X_train)
            residuals = y_train.values - preds_train
            self.residual_std[name] = float(np.std(residuals, ddof=1))

    def predict_single(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for a single named model.
        """
        model = self.fitted_models[model_name]
        if model_name == "Seasonal Naive":
            return model.predict(X)
        elif model_name == "Ridge Regression":
            X_scaled = self.scaler.transform(X)
            return model.predict(X_scaled)
        elif model_name == "Ensemble":
            rf_pred = self.predict_single("Random Forest", X)
            xgb_pred = self.predict_single("XGBoost", X)
            ridge_pred = self.predict_single("Ridge Regression", X)
            return 0.45 * rf_pred + 0.45 * xgb_pred + 0.10 * ridge_pred
        else:
            return model.predict(X)

    def predict_all(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predicts with all models plus Ensemble.
        """
        results = {}
        for name in self.fitted_models:
            results[name] = self.predict_single(name, X_test)

        # Compute Ensemble
        rf_p = results["Random Forest"]
        xgb_p = results["XGBoost"]
        ridge_p = results["Ridge Regression"]
        results["Ensemble"] = 0.45 * rf_p + 0.45 * xgb_p + 0.10 * ridge_p
        self.residual_std["Ensemble"] = 0.5 * (self.residual_std["Random Forest"] + self.residual_std["XGBoost"])

        return results

    def recursive_forecast(
        self,
        model_name: str,
        historical_series: pd.Series,
        horizon_weeks: int = 12,
        freq: str = "W-MON"
    ) -> pd.DataFrame:
        """
        Performs recursive multi-step forecasting into the future.
        Returns a DataFrame with columns:
        ['Forecast', 'Lower_80', 'Upper_80', 'Lower_95', 'Upper_95']
        """
        history = historical_series.copy()
        last_date = history.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=horizon_weeks, freq=freq)

        base_trend = len(history)
        total_len = base_trend + horizon_weeks
        sigma = self.residual_std.get(model_name, self.residual_std.get("Random Forest", 2500.0))

        forecast_records = []

        for step, f_date in enumerate(future_dates):
            feat_dict = create_future_feature_row(
                history_series=history,
                future_date=f_date,
                base_trend_idx=base_trend + step,
                total_len_for_trend=total_len
            )
            feat_df = pd.DataFrame([feat_dict])

            # Predict next step
            y_pred = self.predict_single(model_name, feat_df)[0]
            # Retail sales cannot be negative
            y_pred = max(float(y_pred), 0.0)

            # Uncertainty expands slightly with horizon step h
            h_factor = np.sqrt(1.0 + 0.08 * step)
            step_sigma = sigma * h_factor

            lower_80 = max(0.0, y_pred - 1.28 * step_sigma)
            upper_80 = y_pred + 1.28 * step_sigma
            lower_95 = max(0.0, y_pred - 1.96 * step_sigma)
            upper_95 = y_pred + 1.96 * step_sigma

            forecast_records.append({
                "Date": f_date,
                "Forecast": y_pred,
                "Lower_80": lower_80,
                "Upper_80": upper_80,
                "Lower_95": lower_95,
                "Upper_95": upper_95
            })

            # Append prediction to history for next recursive lag calculation
            history = pd.concat([history, pd.Series([y_pred], index=[f_date])])

        res_df = pd.DataFrame(forecast_records).set_index("Date")
        return res_df

    def get_feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Extracts feature importances from Random Forest and XGBoost.
        """
        rf = self.fitted_models.get("Random Forest")
        xgb = self.fitted_models.get("XGBoost")

        rf_imp = rf.feature_importances_ if rf is not None else np.zeros(len(feature_names))
        xgb_imp = xgb.feature_importances_ if xgb is not None else np.zeros(len(feature_names))

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "RF_Importance": rf_imp,
            "XGB_Importance": xgb_imp
        })
        imp_df["Mean_Importance"] = (imp_df["RF_Importance"] + imp_df["XGB_Importance"]) / 2
        imp_df = imp_df.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)
        return imp_df
