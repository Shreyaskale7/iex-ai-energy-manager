"""Smoke tests for LightGBM training."""

from pathlib import Path

import numpy as np
import pandas as pd

from features.build_features import ENGINEERED_FEATURES, MARKET_INPUT_COLUMNS, TARGET_COLUMN
from models.compare import compare_models
from models.train_lightgbm import LightGBMMCPTrainer


def _features_parquet(path: Path, days: int = 14) -> None:
    rows = []
    rng = np.random.default_rng(1)
    for day in range(1, days + 1):
        for hour in range(1, 25):
            for tb in range(1, 5):
                ts = pd.Timestamp(f"2024-03-{day:02d}", tz="Asia/Kolkata") + pd.Timedelta(
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
                    row[col] = float(rng.uniform(5000, 12000))
                for col in ENGINEERED_FEATURES:
                    row[col] = float(rng.uniform(-1, 1)) if "sin" in col or "cos" in col else mcp
                rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_compare_models():
    xgb = {"test": {"mae": 100, "rmse": 120, "mape": 3.0, "r2": 0.9}}
    lgbm = {"test": {"mae": 95, "rmse": 125, "mape": 2.8, "r2": 0.91}}
    result = compare_models(xgb, lgbm, split="test")
    assert result["overall_winner"] in {"xgboost", "lightgbm", "tie"}


def test_lightgbm_train_smoke(tmp_path: Path):
    features = tmp_path / "features.parquet"
    _features_parquet(features)

    result = LightGBMMCPTrainer(
        features_path=features,
        model_path=tmp_path / "lightgbm.pkl",
        report_path=tmp_path / "lgbm.html",
        comparison_report_path=tmp_path / "compare.html",
        xgb_model_path=tmp_path / "missing_xgb.pkl",
        optuna_trials=2,
        n_splits=2,
        early_stopping_rounds=5,
        shap_sample_size=50,
    ).run()

    assert result.model_path.exists()
    assert result.report_path.exists()
    assert result.metrics["test"]["rmse"] > 0
    assert result.xgb_comparison is None
