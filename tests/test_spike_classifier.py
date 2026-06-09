"""Tests for MCP spike classifier."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.build_features import ENGINEERED_FEATURES, MARKET_INPUT_COLUMNS, TARGET_COLUMN
from models.classification_metrics import classification_metrics
from models.spike_classifier import RTMSpikeClassifierTrainer, SpikeClassifierBundle


def _features(path: Path, n: int = 1200) -> None:
    rng = np.random.default_rng(1)
    cols = MARKET_INPUT_COLUMNS + ENGINEERED_FEATURES
    data = {c: rng.uniform(0, 1, n) for c in cols}
    base = np.linspace(2500, 8000, n)
    noise = rng.normal(0, 200, n)
    data[TARGET_COLUMN] = base + noise
    data["block_timestamp"] = pd.date_range("2024-01-01", periods=n, freq="15min", tz="Asia/Kolkata")
    data["trade_date"] = pd.to_datetime(data["block_timestamp"]).date
    data["session_id"] = 1
    data["source_file"] = "t.xlsx"
    pd.DataFrame(data).to_parquet(path, index=False)


def test_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.6, 0.9, 0.2])
    m = classification_metrics(y_true, y_pred, y_prob)
    assert "confusion_matrix" in m
    assert m["precision"] >= 0
    assert m["recall"] >= 0
    assert m["f1"] >= 0
    assert not np.isnan(m["roc_auc"])


def test_spike_classifier_train(tmp_path: Path):
    features = tmp_path / "features.parquet"
    _features(features)

    result = RTMSpikeClassifierTrainer(
        features_path=features,
        model_path=tmp_path / "spike_classifier.pkl",
        report_path=tmp_path / "report.html",
    ).run()

    assert result.model_path.exists()
    assert result.spike_threshold > 0
    test = result.metrics["test"]
    assert "confusion_matrix" in test
    assert "roc_auc" in test

    bundle = SpikeClassifierBundle(
        model=__import__("joblib").load(result.model_path)["model"],
        feature_names=result.feature_names,
        spike_threshold=result.spike_threshold,
    )
    X = pd.read_parquet(features).drop(columns=[TARGET_COLUMN], errors="ignore")
    X = X[result.feature_names].iloc[:10]
    probs = bundle.spike_probability(X)
    assert len(probs) == 10
    assert np.all((probs >= 0) & (probs <= 1))
