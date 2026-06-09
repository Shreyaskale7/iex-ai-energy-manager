"""Regression metrics for MCP forecasting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error (%), stabilized for near-zero MCP."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Symmetric MAPE (%), stabilized when actual and forecast are near zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def percentage_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAPE and sMAPE for a single prediction vector."""
    return {"mape": mape(y_true, y_pred), "smape": smape(y_true, y_pred)}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }
