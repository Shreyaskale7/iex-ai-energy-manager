"""Production XGBoost trainer for next 15-minute MCP forecasting."""

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
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from models.dataset import NEXT_TARGET_COLUMN, chronological_split, load_forecast_dataset
from models.metrics import regression_metrics
from models.report import generate_xgb_html_report

logger = logging.getLogger(__name__)


@dataclass
class XGBoostTrainResult:
    model_path: Path
    report_path: Path
    metrics_path: Path
    metrics: dict[str, dict[str, float]]
    best_params: dict[str, Any]
    feature_names: list[str]


class XGBoostMCPTrainer:
    """
    Trains an XGBoost regressor to predict MCP at the next 15-minute block.

    Validation strategy:
        - Chronological train / validation / holdout test split
        - Optuna hyperparameter search with TimeSeriesSplit on the train slice
        - Early stopping on the validation slice for final model fitting
    """

    def __init__(
        self,
        features_path: Path | str = "data/features/features.parquet",
        model_path: Path | str = "models/xgboost.pkl",
        report_path: Path | str = "reports/xgb_report.html",
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        n_splits: int = 5,
        optuna_trials: int = 40,
        early_stopping_rounds: int = 50,
        shap_sample_size: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.features_path = Path(features_path)
        self.model_path = Path(model_path)
        self.report_path = Path(report_path)
        self.metrics_path = self.model_path.parent / "xgboost_metrics.json"
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

    def run(self) -> XGBoostTrainResult:
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Features not found: {self.features_path}. Run scripts/build_features.py first."
            )

        X, y, timestamps, self._feature_names = load_forecast_dataset(self.features_path)
        splits = chronological_split(
            X, y, timestamps, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )

        logger.info("Starting Optuna tuning (%d trials, %d-fold TimeSeriesSplit)", self.optuna_trials, self.n_splits)
        self._best_params = self._tune_hyperparameters(splits["X_train"], splits["y_train"])

        logger.info("Training final model with early stopping")
        model, best_iteration = self._fit_final_model(
            splits["X_train"],
            splits["y_train"],
            splits["X_val"],
            splits["y_val"],
        )

        metrics = {
            "validation": regression_metrics(splits["y_val"], model.predict(splits["X_val"])),
            "test": regression_metrics(splits["y_test"], model.predict(splits["X_test"])),
        }
        metrics["train"] = regression_metrics(splits["y_train"], model.predict(splits["X_train"]))

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

        optuna_summary = {
            "n_trials": len(self._optuna_study.trials) if self._optuna_study else 0,
            "best_value": float(self._optuna_study.best_value) if self._optuna_study else None,
            "direction": "minimize",
            "objective": "mean_cv_rmse",
        }

        generate_xgb_html_report(
            output_path=self.report_path,
            metrics=metrics,
            best_params=self._best_params,
            optuna_summary=optuna_summary,
            importance=importance,
            y_true=splits["y_test"].to_numpy(),
            y_pred=model.predict(splits["X_test"]),
            shap_values=shap_values,
            feature_names=self._feature_names,
            split_info={
                "train_rows": len(splits["X_train"]),
                "val_rows": len(splits["X_val"]),
                "test_rows": len(splits["X_test"]),
                "train_start": str(splits["ts_train"].iloc[0]),
                "test_end": str(splits["ts_test"].iloc[-1]),
            },
        )

        logger.info(
            "Training complete | test MAE=%.3f RMSE=%.3f MAPE=%.2f%% R2=%.4f",
            metrics["test"]["mae"],
            metrics["test"]["rmse"],
            metrics["test"]["mape"],
            metrics["test"]["r2"],
        )

        return XGBoostTrainResult(
            model_path=self.model_path,
            report_path=self.report_path,
            metrics_path=self.metrics_path,
            metrics=metrics,
            best_params=self._best_params,
            feature_names=self._feature_names,
        )

    def _base_params(self, trial_params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        if trial_params:
            params.update(trial_params)
        return params

    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }
            fold_scores: list[float] = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = xgb.XGBRegressor(
                    **self._base_params(trial_params),
                    early_stopping_rounds=self.early_stopping_rounds,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                fold_scores.append(regression_metrics(y_va.to_numpy(), model.predict(X_va))["rmse"])

            return float(np.mean(fold_scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", study_name="xgb_mcp_next_block")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
        self._optuna_study = study
        logger.info("Optuna best CV RMSE=%.4f", study.best_value)
        return study.best_params

    def _fit_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[xgb.XGBRegressor, int]:
        model = xgb.XGBRegressor(
            **self._base_params(self._best_params),
            early_stopping_rounds=self.early_stopping_rounds,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_iteration = int(model.best_iteration) if model.best_iteration is not None else model.n_estimators
        return model, best_iteration

    @staticmethod
    def _feature_importance_frame(model: xgb.XGBRegressor, feature_names: list[str]) -> pd.DataFrame:
        return (
            pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _compute_shap(
        self,
        model: xgb.XGBRegressor,
        X_test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_n = min(self.shap_sample_size, len(X_test))
        rng = np.random.default_rng(self.random_state)
        sample_idx = rng.choice(len(X_test), size=sample_n, replace=False)
        X_sample = X_test.iloc[sample_idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return np.asarray(shap_values), sample_idx

    def _save_model(self, bundle: dict[str, Any]) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.model_path)
        logger.info("Model saved to %s", self.model_path)

    def _save_metrics(self, metrics: dict[str, dict[str, float]]) -> None:
        payload = {
            "metrics": metrics,
            "best_params": self._best_params,
            "feature_count": len(self._feature_names),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> int:
    configure_logging()
    result = XGBoostMCPTrainer().run()
    print(f"Model:  {result.model_path}")
    print(f"Report: {result.report_path}")
    print(f"Test metrics: {result.metrics['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
