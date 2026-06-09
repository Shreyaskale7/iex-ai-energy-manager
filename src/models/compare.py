"""Compare LightGBM vs XGBoost model performance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from models.metrics import regression_metrics


def load_metrics_json(path: Path) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def load_metrics_from_bundle(path: Path) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None
    bundle = joblib.load(path)
    return bundle.get("metrics")


def load_model_metrics(path: Path, json_path: Path | None = None) -> dict[str, dict[str, float]] | None:
    if json_path and json_path.exists():
        metrics = load_metrics_json(json_path)
        if metrics:
            return metrics
    return load_metrics_from_bundle(path)


def compare_models(
    xgb_metrics: dict[str, dict[str, float]],
    lgbm_metrics: dict[str, dict[str, float]],
    split: str = "test",
) -> dict[str, Any]:
    """Compare metrics on a given split; lower is better for error metrics."""
    metric_names = ["mae", "rmse", "mape"]
    comparison: dict[str, Any] = {"split": split, "metrics": {}, "overall_winner": None}

    xgb_wins = 0
    lgbm_wins = 0

    for name in metric_names:
        x_val = xgb_metrics[split][name]
        l_val = lgbm_metrics[split][name]
        if x_val < l_val:
            winner = "xgboost"
            xgb_wins += 1
        elif l_val < x_val:
            winner = "lightgbm"
            lgbm_wins += 1
        else:
            winner = "tie"
        comparison["metrics"][name] = {
            "xgboost": x_val,
            "lightgbm": l_val,
            "delta_lgbm_minus_xgb": l_val - x_val,
            "winner": winner,
        }

    x_r2 = xgb_metrics[split]["r2"]
    l_r2 = lgbm_metrics[split]["r2"]
    if x_r2 > l_r2:
        r2_winner = "xgboost"
        xgb_wins += 1
    elif l_r2 > x_r2:
        r2_winner = "lightgbm"
        lgbm_wins += 1
    else:
        r2_winner = "tie"
    comparison["metrics"]["r2"] = {
        "xgboost": x_r2,
        "lightgbm": l_r2,
        "delta_lgbm_minus_xgb": l_r2 - x_r2,
        "winner": r2_winner,
    }

    if lgbm_wins > xgb_wins:
        comparison["overall_winner"] = "lightgbm"
    elif xgb_wins > lgbm_wins:
        comparison["overall_winner"] = "xgboost"
    else:
        comparison["overall_winner"] = "tie"

    comparison["score"] = {"xgboost": xgb_wins, "lightgbm": lgbm_wins}
    return comparison


def compare_predictions(
    y_true: np.ndarray,
    xgb_pred: np.ndarray,
    lgbm_pred: np.ndarray,
) -> dict[str, Any]:
    return {
        "xgboost": regression_metrics(y_true, xgb_pred),
        "lightgbm": regression_metrics(y_true, lgbm_pred),
    }
