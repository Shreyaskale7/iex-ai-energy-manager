"""XGBoost classifier for RTM MCP price spike prediction."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from models.classification_metrics import classification_metrics
from models.dataset import FEATURE_COLUMNS, chronological_split, load_forecast_dataset
from models.report import generate_spike_classifier_report

logger = logging.getLogger(__name__)

SPIKE_PERCENTILE = 75
SPIKE_LABEL_COLUMN = "is_spike"
PROBABILITY_COLUMN = "spike_probability"


@dataclass
class SpikeClassifierResult:
    model_path: Path
    report_path: Path
    metrics_path: Path
    metrics: dict[str, dict[str, Any]]
    spike_threshold: float
    feature_names: list[str]


class SpikeClassifierBundle:
    """Serializable wrapper for inference."""

    def __init__(
        self,
        model: xgb.XGBClassifier,
        feature_names: list[str],
        spike_threshold: float,
        percentile: float = SPIKE_PERCENTILE,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.spike_threshold = spike_threshold
        self.percentile = percentile

    def spike_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(spike) for each row."""
        features = X[self.feature_names]
        proba = self.model.predict_proba(features)
        return proba[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.spike_probability(X) >= threshold).astype(int)


class RTMSpikeClassifierTrainer:
    """
    Predicts whether the next 15-minute MCP will exceed the 75th percentile.

    Spike definition: next_mcp_rs_mwh > percentile_75 computed on the training split only.
    """

    def __init__(
        self,
        features_path: Path | str = "data/features/features.parquet",
        model_path: Path | str = "models/spike_classifier.pkl",
        report_path: Path | str = "reports/spike_classifier_report.html",
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        spike_percentile: float = SPIKE_PERCENTILE,
        n_splits: int = 5,
        early_stopping_rounds: int = 30,
        random_state: int = 42,
    ) -> None:
        self.features_path = Path(features_path)
        self.model_path = Path(model_path)
        self.report_path = Path(report_path)
        self.metrics_path = self.model_path.parent / "spike_classifier_metrics.json"
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.spike_percentile = spike_percentile
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._feature_names: list[str] = []
        self._spike_threshold: float = 0.0

    def run(self) -> SpikeClassifierResult:
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Features not found: {self.features_path}. Run scripts/build_features.py first."
            )

        X, y_reg, timestamps, self._feature_names = load_forecast_dataset(self.features_path)
        splits = chronological_split(
            X, y_reg, timestamps, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )

        self._spike_threshold = float(
            np.percentile(splits["y_train"].to_numpy(), self.spike_percentile)
        )
        logger.info(
            "Spike threshold (P%d on train next-MCP): %.2f Rs/MWh",
            int(self.spike_percentile),
            self._spike_threshold,
        )

        labels = self._build_labels(y_reg)
        y_train = labels.loc[splits["X_train"].index]
        y_val = labels.loc[splits["X_val"].index]
        y_test = labels.loc[splits["X_test"].index]

        self._log_class_balance("train", y_train)
        self._log_class_balance("validation", y_val)
        self._log_class_balance("test", y_test)

        model = self._fit_classifier(
            splits["X_train"],
            y_train,
            splits["X_val"],
            y_val,
        )

        metrics = self._evaluate_splits(model, splits, labels)
        bundle = SpikeClassifierBundle(
            model=model,
            feature_names=self._feature_names,
            spike_threshold=self._spike_threshold,
            percentile=self.spike_percentile,
        )

        payload = {
            "bundle": bundle,
            "model": model,
            "feature_names": self._feature_names,
            "spike_threshold": self._spike_threshold,
            "spike_percentile": self.spike_percentile,
            "metrics": metrics,
            "target": "next_block_spike",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._save_model(payload)
        self._save_metrics(metrics)

        test_probs = model.predict_proba(splits["X_test"])[:, 1]
        generate_spike_classifier_report(
            output_path=self.report_path,
            metrics=metrics,
            spike_threshold=self._spike_threshold,
            percentile=self.spike_percentile,
            y_test=y_test.to_numpy(),
            y_prob=test_probs,
            y_pred=(test_probs >= 0.5).astype(int),
        )

        logger.info(
            "Spike classifier test F1=%.4f ROC-AUC=%.4f",
            metrics["test"]["f1"],
            metrics["test"]["roc_auc"],
        )

        return SpikeClassifierResult(
            model_path=self.model_path,
            report_path=self.report_path,
            metrics_path=self.metrics_path,
            metrics=metrics,
            spike_threshold=self._spike_threshold,
            feature_names=self._feature_names,
        )

    def _build_labels(self, y_reg: pd.Series) -> pd.Series:
        return (y_reg > self._spike_threshold).astype(int).rename(SPIKE_LABEL_COLUMN)

    def _fit_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> xgb.XGBClassifier:
        pos = max(int(y_train.sum()), 1)
        neg = max(int((1 - y_train).sum()), 1)
        scale_pos_weight = neg / pos

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            max_depth=8,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model

    def _evaluate_splits(
        self,
        model: xgb.XGBClassifier,
        splits: dict[str, Any],
        labels: pd.Series,
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for split_name, x_key in [
            ("train", "X_train"),
            ("validation", "X_val"),
            ("test", "X_test"),
        ]:
            X = splits[x_key]
            y_true = labels.loc[X.index].to_numpy()
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            split_metrics = classification_metrics(y_true, y_pred, y_prob)
            split_metrics[PROBABILITY_COLUMN] = {
                "mean": float(y_prob.mean()),
                "max": float(y_prob.max()),
                "min": float(y_prob.min()),
            }
            results[split_name] = split_metrics
        return results

    @staticmethod
    def _log_class_balance(name: str, y: pd.Series) -> None:
        rate = y.mean() * 100
        logger.info("%s spike rate: %.2f%% (%d / %d)", name, rate, int(y.sum()), len(y))

    def _save_model(self, payload: dict[str, Any]) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.model_path)
        logger.info("Spike classifier saved to %s", self.model_path)

    def _save_metrics(self, metrics: dict[str, Any]) -> None:
        body = {
            "metrics": metrics,
            "spike_threshold": self._spike_threshold,
            "spike_percentile": self.spike_percentile,
            "feature_count": len(self._feature_names),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.metrics_path.write_text(json.dumps(body, indent=2), encoding="utf-8")


def load_spike_classifier(path: Path | str = "models/spike_classifier.pkl") -> SpikeClassifierBundle:
    payload = joblib.load(path)
    if isinstance(payload.get("bundle"), SpikeClassifierBundle):
        return payload["bundle"]
    return SpikeClassifierBundle(
        model=payload["model"],
        feature_names=payload["feature_names"],
        spike_threshold=payload["spike_threshold"],
        percentile=payload.get("spike_percentile", SPIKE_PERCENTILE),
    )


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> int:
    configure_logging()
    result = RTMSpikeClassifierTrainer().run()
    print(f"Model:     {result.model_path}")
    print(f"Threshold: {result.spike_threshold:.2f} Rs/MWh (P75 train)")
    print(f"Test metrics: {result.metrics['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
