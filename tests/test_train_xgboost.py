"""Smoke tests for XGBoost training pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.build_features import ENGINEERED_FEATURES, MARKET_INPUT_COLUMNS, TARGET_COLUMN
from models.metrics import regression_metrics
from models.dataset import FEATURE_COLUMNS, NEXT_TARGET_COLUMN
from models.train_xgboost import XGBoostMCPTrainer


def _build_features_parquet(path: Path, n_days: int = 12) -> None:
    rows = []
    rng = np.random.default_rng(7)
    for day in range(1, n_days + 1):
        for hour in range(1, 25):
            for tb in range(1, 5):
                ts = pd.Timestamp(f"2024-02-{day:02d}", tz="Asia/Kolkata") + pd.Timedelta(
                    minutes=(hour - 1) * 60 + (tb - 1) * 15
                )
                mcp = float(rng.uniform(3000, 4000))
                row = {
                    "block_timestamp": ts,
                    "trade_date": ts.date(),
                    "session_id": 1,
                    "source_file": "t.xlsx",
                    TARGET_COLUMN: mcp,
                }
                for col in MARKET_INPUT_COLUMNS:
                    row[col] = float(rng.uniform(1000, 12000))
                for col in ENGINEERED_FEATURES:
                    if col.startswith("mcp_lag"):
                        row[col] = mcp - float(rng.uniform(0, 50))
                    elif col.startswith("rolling"):
                        row[col] = mcp
                    elif col == "volume_change":
                        row[col] = float(rng.uniform(-100, 100))
                    elif col in ("sin_hour", "cos_hour", "sin_block", "cos_block"):
                        row[col] = float(rng.uniform(-1, 1))
                    else:
                        row[col] = float(rng.integers(0, 10))
                rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_regression_metrics():
    y = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 280.0])
    m = regression_metrics(y, pred)
    assert "mae" in m and "rmse" in m and "mape" in m and "r2" in m


def test_xgboost_train_smoke(tmp_path: Path):
    features = tmp_path / "features.parquet"
    _build_features_parquet(features)

    trainer = XGBoostMCPTrainer(
        features_path=features,
        model_path=tmp_path / "xgboost.pkl",
        report_path=tmp_path / "report.html",
        optuna_trials=2,
        n_splits=2,
        early_stopping_rounds=10,
        shap_sample_size=100,
    )
    result = trainer.run()

    assert result.model_path.exists()
    assert result.report_path.exists()
    assert result.metrics["test"]["rmse"] > 0
    assert len(result.feature_names) == len(FEATURE_COLUMNS)
