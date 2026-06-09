"""Production CatBoost trainer for next 15-minute MCP forecasting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

from models.dataset import NEXT_TARGET_COLUMN, chronological_split, load_forecast_dataset
from models.metrics import regression_metrics
from models.report import generate_catboost_html_report

logger = logging.getLogger(__name__)


@dataclass
class CatBoostTrainResult:
    model_path: Path
    report_path: Path
    metrics_path: Path
    metrics: dict[str, dict[str, float]]
    best_params: dict[str, Any]
    feature_names: list[str]


class CatBoostMCPTrainer:
    """CatBoost regressor with Optuna, TimeSeriesSplit, and early stopping."""

    def __init__(
        self,
        features_path: Path | str = "data/features/features.parquet",
        model_path: Path | str = "models/catboost.pkl",
        report_path: Path | str = "reports/catboost_report.html",
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        n_splits: int = 5,
        optuna_trials: int = 30,
        early_stopping_rounds: int = 50,
        shap_sample_size: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.features_path = Path(features_path)
        self.model_path = Path(model_path)
        self.report_path = Path(report_path)
        self.metrics_path = self.model_path.parent / "catboost_metrics.json"
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.n_splits = n_splits
        self.optuna_trials = optuna_trials
        self.early_stopping_rounds = early_stopping_rounds
        self.shap_sample_size = shap_sample_size
        self.random_state = random_state
        self._feature_names: list[str] = []
        self._best_params: dict[str, Any] = {}
        self._optuna_study: optuna.Study | None = None

    def run(self) -> CatBoostTrainResult:
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features not found: {self.features_path}")

        X, y, timestamps, self._feature_names = load_forecast_dataset(self.features_path)
        splits = chronological_split(
            X, y, timestamps, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )

        self._best_params = self._tune_hyperparameters(splits["X_train"], splits["y_train"])
        model, best_iteration = self._fit_final_model(
            splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"]
        )

        metrics = {
            "train": regression_metrics(splits["y_train"], model.predict(splits["X_train"])),
            "validation": regression_metrics(splits["y_val"], model.predict(splits["X_val"])),
            "test": regression_metrics(splits["y_test"], model.predict(splits["X_test"])),
        }

        importance = self._feature_importance_frame(model, self._feature_names)
        shap_values, _ = self._compute_shap(model, splits["X_test"])

        bundle = {
            "model": model,
            "feature_names": self._feature_names,
            "best_params": self._best_params,
            "best_iteration": best_iteration,
            "metrics": metrics,
            "target": NEXT_TARGET_COLUMN,
            "horizon_minutes": 15,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._save_model(bundle)
        self._save_metrics(metrics)

        generate_catboost_html_report(
            output_path=self.report_path,
            metrics=metrics,
            best_params=self._best_params,
            optuna_summary={
                "n_trials": len(self._optuna_study.trials) if self._optuna_study else 0,
                "best_value": float(self._optuna_study.best_value) if self._optuna_study else None,
            },
            importance=importance,
            y_true=splits["y_test"].to_numpy(),
            y_pred=model.predict(splits["X_test"]),
            shap_values=shap_values,
            feature_names=self._feature_names,
            split_info={
                "train_rows": len(splits["X_train"]),
                "val_rows": len(splits["X_val"]),
                "test_rows": len(splits["X_test"]),
            },
        )

        logger.info("CatBoost test RMSE=%.4f", metrics["test"]["rmse"])
        return CatBoostTrainResult(
            model_path=self.model_path,
            report_path=self.report_path,
            metrics_path=self.metrics_path,
            metrics=metrics,
            best_params=self._best_params,
            feature_names=self._feature_names,
        )

    def _base_params(self, trial_params: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "loss_function": "RMSE",
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
        }
        if trial_params:
            params.update(trial_params)
        return params

    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                "iterations": trial.suggest_int("iterations", 200, 1200, step=100),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
            }
            scores: list[float] = []
            for train_idx, val_idx in tscv.split(X_train):
                model = CatBoostRegressor(
                    **self._base_params(trial_params),
                    early_stopping_rounds=self.early_stopping_rounds,
                )
                model.fit(
                    X_train.iloc[train_idx],
                    y_train.iloc[train_idx],
                    eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                    verbose=False,
                )
                pred = model.predict(X_train.iloc[val_idx])
                scores.append(regression_metrics(y_train.iloc[val_idx].to_numpy(), pred)["rmse"])
            return float(np.mean(scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", study_name="catboost_mcp")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
        self._optuna_study = study
        return study.best_params

    def _fit_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[CatBoostRegressor, int]:
        model = CatBoostRegressor(
            **self._base_params(self._best_params),
            early_stopping_rounds=self.early_stopping_rounds,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        best_iteration = int(model.get_best_iteration() or model.tree_count_)
        return model, best_iteration

    @staticmethod
    def _feature_importance_frame(model: CatBoostRegressor, feature_names: list[str]) -> pd.DataFrame:
        scores = model.get_feature_importance()
        return (
            pd.DataFrame({"feature": feature_names, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _compute_shap(self, model: CatBoostRegressor, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        sample_n = min(self.shap_sample_size, len(X_test))
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X_test), size=sample_n, replace=False)
        X_sample = X_test.iloc[idx]
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_sample)
        if isinstance(values, list):
            values = values[0]
        return np.asarray(values), idx

    def _save_model(self, bundle: dict[str, Any]) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.model_path)

    def _save_metrics(self, metrics: dict[str, dict[str, float]]) -> None:
        self.metrics_path.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "best_params": self._best_params,
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")


def main() -> int:
    configure_logging()
    result = CatBoostMCPTrainer().run()
    print(f"Model: {result.model_path}")
    print(f"Test:  {result.metrics['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
