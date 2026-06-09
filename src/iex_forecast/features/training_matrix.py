"""Build supervised training matrices for direct multi-horizon forecasting."""

import numpy as np
import pandas as pd

from iex_forecast.domain.constants import FEATURE_COLUMNS, FORECAST_HORIZON, TARGET_COLUMN
from iex_forecast.features.builder import FeatureBuilder


class TrainingMatrixBuilder:
    """
    Creates (X, y) samples where each row predicts MCP at origin+t+horizon.

    Uses direct strategy: `horizon` is a feature; one model set per horizon
    is trained via filtered matrices in the trainer.
    """

    def __init__(self) -> None:
        self.feature_builder = FeatureBuilder()
        self._cache = {}

    def build_for_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        df_id = id(df)
        if df_id not in self._cache:
            self._cache[df_id] = self.feature_builder.transform(df)
        enriched = self._cache[df_id]
        enriched = enriched.dropna(subset=["mcp_lag_1", "mcp_lag_96"]).reset_index(drop=True)

        feature_cols = [c for c in FEATURE_COLUMNS if c in enriched.columns and c != "horizon"]
        targets = enriched[TARGET_COLUMN].shift(-horizon)
        features = enriched[feature_cols].copy()
        features["horizon"] = horizon

        valid = targets.notna()
        X = features.loc[valid].reset_index(drop=True)
        y = targets.loc[valid].reset_index(drop=True)
        return X, y

    def build_all_horizons(
        self,
        df: pd.DataFrame,
    ) -> dict[int, tuple[pd.DataFrame, pd.Series]]:
        return {h: self.build_for_horizon(df, h) for h in range(1, FORECAST_HORIZON + 1)}

    def temporal_split(
        self,
        df: pd.DataFrame,
        split_date: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_ts = pd.Timestamp(split_date, tz="Asia/Kolkata")
        ts = pd.to_datetime(df["block_timestamp"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("Asia/Kolkata")
        train = df.loc[ts < split_ts].copy()
        test = df.loc[ts >= split_ts].copy()
        return train, test

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        errors = y_true - y_pred
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        mape = float(np.mean(np.abs(errors) / np.maximum(np.abs(y_true), 1.0)) * 100)
        return {"mae": mae, "rmse": rmse, "mape": mape}
