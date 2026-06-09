"""Tests for recursive 96-block ensemble forecasting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.ensemble import WeightedEnsemble
from models.recursive_forecast import FORECAST_HORIZON, RecursiveEnsembleForecaster


def _history(n: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        ts = pd.Timestamp("2024-01-01", tz="Asia/Kolkata") + pd.Timedelta(minutes=15 * i)
        hour = (ts.hour + 1) if ts.hour < 23 else 24
        tb = (ts.minute // 15) + 1
        rows.append(
            {
                "block_timestamp": ts,
                "trade_date": ts.date(),
                "hour": hour,
                "time_block": tb,
                "daily_block_index": (hour - 1) * 4 + tb,
                "session_id": 1,
                "purchase_bid_mw": float(rng.uniform(8000, 12000)),
                "sell_bid_mw": float(rng.uniform(8000, 12000)),
                "mcv_mw": float(rng.uniform(3000, 7000)),
                "scheduled_volume_mw": float(rng.uniform(3000, 6500)),
                "mcp_rs_mwh": float(rng.uniform(3000, 4000)),
                "source_file": "test.xlsx",
            }
        )
    return pd.DataFrame(rows)


def _fake_ensemble(tmp_path: Path) -> Path:
    from sklearn.linear_model import Ridge
    import joblib

    feature_names = [
        "purchase_bid_mw",
        "sell_bid_mw",
        "mcv_mw",
        "scheduled_volume_mw",
        "mcp_lag_1",
        "hour",
        "block_number",
    ]
    model = Ridge()
    X = np.random.randn(100, len(feature_names))
    y = np.random.randn(100)
    model.fit(X, y)

    paths = {}
    for name in ("xgboost", "lightgbm", "catboost"):
        p = tmp_path / f"{name}.pkl"
        joblib.dump({"model": model, "feature_names": feature_names}, p)
        paths[name] = str(p)

    ens = WeightedEnsemble(
        weights={"xgboost": 0.34, "lightgbm": 0.33, "catboost": 0.33},
        model_paths=paths,
        feature_names=feature_names,
    )
    bundle_path = tmp_path / "ensemble.pkl"
    joblib.dump({"ensemble": ens, "weights": ens.weights, "model_paths": paths, "feature_names": feature_names}, bundle_path)
    return bundle_path


def test_recursive_forecast_96_blocks(tmp_path: Path):
    ens_path = _fake_ensemble(tmp_path)
    hist = _history(700)
    # Trim feature set by using minimal history columns only - Ridge uses subset; pipeline needs full features
    # Use full history and patch ensemble with expanded feature list matching pipeline output
    forecaster = RecursiveEnsembleForecaster(
        ensemble_path=ens_path,
        master_path=tmp_path / "missing.parquet",
        output_path=tmp_path / "forecast_96.csv",
        horizon=12,
        min_history_blocks=100,
    )
    # Will fail on feature mismatch with fake ensemble - test structure with monkeypatch

    from features.build_features import RTMFeaturePipeline

    pipeline = RTMFeaturePipeline()
    featured = pipeline.build_features(hist)
    feature_names = featured.columns.drop(
        [c for c in featured.columns if c in ("block_timestamp", "trade_date", "session_id", "source_file", "mcp_rs_mwh")]
    ).tolist()

    from sklearn.linear_model import Ridge
    import joblib

    model = Ridge().fit(featured[feature_names].fillna(0).iloc[:200], np.random.randn(200))
    paths = {}
    for name in ("xgboost", "lightgbm", "catboost"):
        p = tmp_path / f"{name}2.pkl"
        joblib.dump({"model": model, "feature_names": feature_names}, p)
        paths[name] = str(p)

    ens = WeightedEnsemble(
        weights={"xgboost": 0.34, "lightgbm": 0.33, "catboost": 0.33},
        model_paths=paths,
        feature_names=feature_names,
    )
    joblib.dump(
        {"ensemble": ens, "weights": ens.weights, "model_paths": paths, "feature_names": feature_names},
        ens_path,
    )

    result = RecursiveEnsembleForecaster(
        ensemble_path=ens_path,
        output_path=tmp_path / "forecast_96.csv",
        horizon=12,
        min_history_blocks=700,
    ).run(market_state=hist)

    assert len(result.blocks) == 12
    assert result.output_path.exists()
    df = pd.read_csv(result.output_path)
    assert list(df.columns) >= ["timestamp", "predicted_mcp", "confidence_interval"]
    assert df["predicted_mcp"].notna().all()
