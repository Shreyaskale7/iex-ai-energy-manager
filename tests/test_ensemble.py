"""Tests for ensemble weight optimization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.build_features import ENGINEERED_FEATURES, MARKET_INPUT_COLUMNS, TARGET_COLUMN
from models.ensemble import BASE_MODEL_NAMES, EnsembleTrainer, WeightedEnsemble


def _write_fake_bundle(path: Path, n_features: int = 5) -> list[str]:
    from sklearn.linear_model import Ridge

    feature_names = list(MARKET_INPUT_COLUMNS[:2]) + list(ENGINEERED_FEATURES[: n_features - 2])
    model = Ridge().fit(np.random.randn(50, len(feature_names)), np.random.randn(50))
    import joblib

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "metrics": {"test": {"mae": 1, "rmse": 1, "mape": 1, "r2": 0.9}},
        },
        path,
    )
    return feature_names


def _features(path: Path, n_rows: int = 800, feature_names: list[str] | None = None) -> None:
    rng = np.random.default_rng(0)
    cols = feature_names or (MARKET_INPUT_COLUMNS + ENGINEERED_FEATURES)
    data = {c: rng.uniform(0, 1, n_rows) for c in cols}
    data[TARGET_COLUMN] = rng.uniform(3000, 4000, n_rows)
    data["block_timestamp"] = pd.date_range("2024-01-01", periods=n_rows, freq="15min", tz="Asia/Kolkata")
    data["trade_date"] = pd.to_datetime(data["block_timestamp"]).date
    data["session_id"] = 1
    data["source_file"] = "t.xlsx"
    pd.DataFrame(data).to_parquet(path, index=False)


def test_weight_optimization_sums_to_one(tmp_path: Path):
    features = tmp_path / "features.parquet"
    fn = _write_fake_bundle(tmp_path / "xgboost.pkl")
    _write_fake_bundle(tmp_path / "lightgbm.pkl")
    _write_fake_bundle(tmp_path / "catboost.pkl")
    _features(features, feature_names=fn)

    result = EnsembleTrainer(
        features_path=features,
        model_path=tmp_path / "ensemble.pkl",
        report_path=tmp_path / "report.html",
        xgb_path=tmp_path / "xgboost.pkl",
        lgbm_path=tmp_path / "lightgbm.pkl",
        catboost_path=tmp_path / "catboost.pkl",
    ).run()

    assert abs(sum(result.weights.values()) - 1.0) < 1e-6
    assert all(w >= 0 for w in result.weights.values())
    assert set(result.weights.keys()) == set(BASE_MODEL_NAMES)
    assert result.model_path.exists()
    assert "ensemble" in result.metrics["test"]


def test_weighted_ensemble_predict(tmp_path: Path):
    fn = _write_fake_bundle(tmp_path / "m.pkl", n_features=6)
    ens = WeightedEnsemble(
        weights={"xgboost": 1.0, "lightgbm": 0.0, "catboost": 0.0},
        model_paths={"xgboost": str(tmp_path / "m.pkl"), "lightgbm": str(tmp_path / "m.pkl"), "catboost": str(tmp_path / "m.pkl")},
        feature_names=fn,
    )
    X = pd.DataFrame({c: [1.0, 2.0] for c in fn})
    pred = ens.predict(X)
    assert len(pred) == 2
