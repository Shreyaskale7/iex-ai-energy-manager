"""Weighted ensemble of XGBoost, LightGBM, and CatBoost MCP forecasts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.dataset import NEXT_TARGET_COLUMN, chronological_split, load_forecast_dataset
from models.metrics import regression_metrics
from models.report import generate_ensemble_comparison_report

logger = logging.getLogger(__name__)

BASE_MODEL_NAMES = ("xgboost", "lightgbm", "catboost")


@dataclass
class WeightedEnsemble:
    """Loads base models and produces a weighted MCP forecast."""

    weights: dict[str, float]
    model_paths: dict[str, str]
    feature_names: list[str]
    target: str = NEXT_TARGET_COLUMN
    horizon_minutes: int = 15
    _models: dict[str, Any] = field(default_factory=dict, repr=False)

    def _ensure_models(self) -> None:
        if self._models:
            return
        for name, path in self.model_paths.items():
            bundle = joblib.load(path)
            self._models[name] = bundle["model"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_models()
        X_features = X[self.feature_names]
        total = np.zeros(len(X_features), dtype=float)
        for name in BASE_MODEL_NAMES:
            weight = self.weights.get(name, 0.0)
            if weight <= 0:
                continue
            total += weight * self._models[name].predict(X_features)
        return total

    def predict_components(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._ensure_models()
        X_features = X[self.feature_names]
        return {name: self._models[name].predict(X_features) for name in BASE_MODEL_NAMES}

    def predict_with_interval(
        self,
        X: pd.DataFrame,
        z_score: float = 1.96,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ensemble mean and approximate confidence bounds from base-model spread.
        """
        components = self.predict_components(X)
        matrix = np.column_stack([components[name] for name in BASE_MODEL_NAMES])
        mean = matrix.mean(axis=1)
        std = matrix.std(axis=1)
        lower = mean - z_score * std
        upper = mean + z_score * std
        return mean, lower, upper


@dataclass
class EnsembleTrainResult:
    model_path: Path
    report_path: Path
    metrics_path: Path
    weights: dict[str, float]
    metrics: dict[str, dict[str, dict[str, float]]]
    overall_winner: str


class EnsembleTrainer:
    """
    Optimizes non-negative ensemble weights on the validation split (sum to 1),
    then evaluates weighted predictions vs each base model.
    """

    def __init__(
        self,
        features_path: Path | str = "data/features/features.parquet",
        model_path: Path | str = "models/ensemble.pkl",
        report_path: Path | str = "reports/ensemble_comparison_report.html",
        xgb_path: Path | str = "models/xgboost.pkl",
        lgbm_path: Path | str = "models/lightgbm.pkl",
        catboost_path: Path | str = "models/catboost.pkl",
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        optimization_metric: str = "rmse",
    ) -> None:
        self.features_path = Path(features_path)
        self.model_path = Path(model_path)
        self.report_path = Path(report_path)
        self.metrics_path = self.model_path.parent / "ensemble_metrics.json"
        self.model_paths = {
            "xgboost": Path(xgb_path),
            "lightgbm": Path(lgbm_path),
            "catboost": Path(catboost_path),
        }
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.optimization_metric = optimization_metric

    def run(self) -> EnsembleTrainResult:
        self._verify_base_models()

        X, y, timestamps, feature_names = load_forecast_dataset(self.features_path)
        splits = chronological_split(
            X, y, timestamps, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )

        val_preds = self._collect_predictions(splits["X_val"])
        test_preds = self._collect_predictions(splits["X_test"])

        y_val = splits["y_val"].to_numpy()
        y_test = splits["y_test"].to_numpy()

        weights = self._optimize_weights(val_preds, y_val)
        logger.info("Optimized ensemble weights: %s", weights)

        val_matrix = self._stack_predictions(val_preds)
        test_matrix = self._stack_predictions(test_preds)
        val_ensemble = val_matrix @ np.array([weights[n] for n in BASE_MODEL_NAMES])
        test_ensemble = test_matrix @ np.array([weights[n] for n in BASE_MODEL_NAMES])

        metrics: dict[str, dict[str, dict[str, float]]] = {
            "validation": {},
            "test": {},
        }
        for name in BASE_MODEL_NAMES:
            metrics["validation"][name] = regression_metrics(y_val, val_preds[name])
            metrics["test"][name] = regression_metrics(y_test, test_preds[name])
        metrics["validation"]["ensemble"] = regression_metrics(y_val, val_ensemble)
        metrics["test"]["ensemble"] = regression_metrics(y_test, test_ensemble)

        ensemble = WeightedEnsemble(
            weights=weights,
            model_paths={k: str(v) for k, v in self.model_paths.items()},
            feature_names=feature_names,
        )

        bundle = {
            "ensemble": ensemble,
            "weights": weights,
            "model_paths": {k: str(v) for k, v in self.model_paths.items()},
            "feature_names": feature_names,
            "metrics": metrics,
            "optimization": {
                "split": "validation",
                "metric": self.optimization_metric,
                "constraint": "weights >= 0, sum(weights) = 1",
            },
            "target": NEXT_TARGET_COLUMN,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._save_bundle(bundle)
        self._save_metrics(metrics, weights)

        overall_winner = self._pick_overall_winner(metrics["test"])
        generate_ensemble_comparison_report(
            output_path=self.report_path,
            weights=weights,
            metrics=metrics,
            y_test=y_test,
            test_preds=test_preds,
            test_ensemble=test_ensemble,
            overall_winner=overall_winner,
        )

        logger.info("Ensemble test RMSE=%.4f (winner: %s)", metrics["test"]["ensemble"]["rmse"], overall_winner)
        return EnsembleTrainResult(
            model_path=self.model_path,
            report_path=self.report_path,
            metrics_path=self.metrics_path,
            weights=weights,
            metrics=metrics,
            overall_winner=overall_winner,
        )

    def _verify_base_models(self) -> None:
        missing = [name for name, path in self.model_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing base models: {missing}. Train with:\n"
                "  python scripts/train_xgboost.py\n"
                "  python scripts/train_lightgbm.py\n"
                "  python scripts/train_catboost.py"
            )

    def _collect_predictions(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        preds: dict[str, np.ndarray] = {}
        for name, path in self.model_paths.items():
            bundle = joblib.load(path)
            model = bundle["model"]
            features = bundle["feature_names"]
            preds[name] = model.predict(X[features])
        return preds

    @staticmethod
    def _stack_predictions(preds: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([preds[name] for name in BASE_MODEL_NAMES])

    def _optimize_weights(self, preds: dict[str, np.ndarray], y_true: np.ndarray) -> dict[str, float]:
        matrix = self._stack_predictions(preds)
        n_models = len(BASE_MODEL_NAMES)
        metric_key = self.optimization_metric

        def objective(weights: np.ndarray) -> float:
            w = weights / weights.sum()
            y_pred = matrix @ w
            return regression_metrics(y_true, y_pred)[metric_key]

        w0 = np.ones(n_models) / n_models
        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        bounds = [(0.0, 1.0)] * n_models

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if not result.success:
            logger.warning("Weight optimization did not converge: %s", result.message)

        optimal = result.x / result.x.sum()
        return {name: float(optimal[i]) for i, name in enumerate(BASE_MODEL_NAMES)}

    @staticmethod
    def _pick_overall_winner(test_metrics: dict[str, dict[str, float]]) -> str:
        models = list(BASE_MODEL_NAMES) + ["ensemble"]
        scores: dict[str, int] = {m: 0 for m in models}
        for metric in ("mae", "rmse", "mape"):
            best = min(models, key=lambda m: test_metrics[m][metric])
            scores[best] += 1
        best_r2 = max(models, key=lambda m: test_metrics[m]["r2"])
        scores[best_r2] += 1
        return max(scores, key=scores.get)

    def _save_bundle(self, bundle: dict[str, Any]) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.model_path)
        logger.info("Ensemble saved to %s", self.model_path)

    def _save_metrics(self, metrics: dict[str, Any], weights: dict[str, float]) -> None:
        payload = {
            "weights": weights,
            "metrics": metrics,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")


def main() -> int:
    configure_logging()
    result = EnsembleTrainer().run()
    print(f"Ensemble: {result.model_path}")
    print(f"Weights:  {result.weights}")
    print(f"Report:   {result.report_path}")
    print(f"Test ensemble metrics: {result.metrics['test']['ensemble']}")
    print(f"Overall winner (test): {result.overall_winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
