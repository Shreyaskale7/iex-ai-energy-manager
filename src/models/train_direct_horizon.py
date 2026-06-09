"""Lightweight direct-horizon LightGBM trainer (no Optuna — uses fixed params)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from models.dataset import chronological_split, load_multihorizon_dataset
from models.metrics import regression_metrics, smape

logger = logging.getLogger(__name__)

# Sensible defaults — taken from tuned t+1 LightGBM run
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}


def train_direct_horizon(
    features_path: Path,
    horizon: int,
    output_dir: Path = Path("models"),
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    early_stopping_rounds: int = 50,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train a LightGBM model to directly predict MCP at ``horizon`` blocks ahead.

    Returns a dict with model_path, test metrics, and horizon info.
    """
    X, y, timestamps, feature_names = load_multihorizon_dataset(features_path, horizon)
    splits = chronological_split(X, y, timestamps, val_ratio=val_ratio, test_ratio=test_ratio)

    lgb_params = {**DEFAULT_PARAMS, **(params or {})}

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        splits["X_train"],
        splits["y_train"],
        eval_set=[(splits["X_val"], splits["y_val"])],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Predictions
    y_test = splits["y_test"].to_numpy()
    y_pred = model.predict(splits["X_test"])

    metrics = regression_metrics(y_test, y_pred)
    metrics["smape"] = smape(y_test, y_pred)

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"direct_h{horizon}.pkl"
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "horizon": horizon,
        "metrics": {"test": metrics},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, model_path)

    logger.info(
        "Direct h=%d | MAE=%.1f RMSE=%.1f R²=%.4f sMAPE=%.2f%%",
        horizon,
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
        metrics["smape"],
    )

    return {
        "horizon": horizon,
        "model_path": str(model_path),
        "test_metrics": metrics,
        "train_rows": len(splits["X_train"]),
        "test_rows": len(splits["X_test"]),
        "best_iteration": getattr(model, "best_iteration_", model.n_estimators),
    }
