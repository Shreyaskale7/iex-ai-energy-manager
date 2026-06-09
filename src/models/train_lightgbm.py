"""Production LightGBM trainer for next 15-minute MCP forecasting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.model_selection import TimeSeriesSplit

from models.compare import compare_models, load_model_metrics
from models.dataset import NEXT_TARGET_COLUMN, load_forecast_dataset, chronological_split
from models.metrics import regression_metrics
from models.report import generate_comparison_report, generate_lgbm_html_report

logger = logging.getLogger(__name__)


@dataclass
class LightGBMTrainResult:
    model_path: Path
    report_path: Path
    comparison_report_path: Path
    metrics_path: Path
    metrics: dict[str, dict[str, float]]
    best_params: dict[str, Any]
    feature_names: list[str]
    xgb_comparison: dict[str, Any] | None


class LightGBMMCPTrainer:
    """LightGBM regressor with Optuna + TimeSeriesSplit, SHAP, and XGBoost comparison."""

    def __init__(
        self,
        features_path: Path | str = "data/features/features.parquet",
        model_path: Path | str = "models/lightgbm.pkl",
        report_path: Path | str = "reports/lgbm_report.html",
        comparison_report_path: Path | str = "reports/model_comparison_report.html",
        xgb_model_path: Path | str = "models/xgboost.pkl",
        xgb_metrics_path: Path | str = "models/xgboost_metrics.json",
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
        self.comparison_report_path = Path(comparison_report_path)
        self.xgb_model_path = Path(xgb_model_path)
        self.xgb_metrics_path = Path(xgb_metrics_path)
        self.metrics_path = self.model_path.parent / "lightgbm_metrics.json"
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

    def run(self) -> LightGBMTrainResult:
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Features not found: {self.features_path}. Run scripts/build_features.py first."
            )

        X, y, timestamps, self._feature_names = load_forecast_dataset(self.features_path)
        splits = chronological_split(
            X, y, timestamps, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )
        logger.info("Prepared %d rows, %d features", len(X), len(self._feature_names))

        logger.info("Optuna tuning LightGBM (%d trials, %d-fold TimeSeriesSplit)", self.optuna_trials, self.n_splits)
        self._best_params = self._tune_hyperparameters(splits["X_train"], splits["y_train"])

        logger.info("Training final LightGBM with early stopping")
        model, best_iteration = self._fit_final_model(
            splits["X_train"],
            splits["y_train"],
            splits["X_val"],
            splits["y_val"],
        )

        metrics = {
            "train": regression_metrics(splits["y_train"], model.predict(splits["X_train"])),
            "validation": regression_metrics(splits["y_val"], model.predict(splits["X_val"])),
            "test": regression_metrics(splits["y_test"], model.predict(splits["X_test"])),
        }

        importance = self._feature_importance_frame(model, self._feature_names)
        shap_values, _ = self._compute_shap(model, splits["X_test"])
        test_pred = model.predict(splits["X_test"])

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

        generate_lgbm_html_report(
            output_path=self.report_path,
            metrics=metrics,
            best_params=self._best_params,
            optuna_summary=optuna_summary,
            importance=importance,
            y_true=splits["y_test"].to_numpy(),
            y_pred=test_pred,
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

        xgb_comparison = self._build_xgboost_comparison(metrics, splits, test_pred)

        logger.info(
            "LightGBM training complete | test MAE=%.3f RMSE=%.3f MAPE=%.2f%% R2=%.4f",
            metrics["test"]["mae"],
            metrics["test"]["rmse"],
            metrics["test"]["mape"],
            metrics["test"]["r2"],
        )

        return LightGBMTrainResult(
            model_path=self.model_path,
            report_path=self.report_path,
            comparison_report_path=self.comparison_report_path,
            metrics_path=self.metrics_path,
            metrics=metrics,
            best_params=self._best_params,
            feature_names=self._feature_names,
            xgb_comparison=xgb_comparison,
        )

    def _build_xgboost_comparison(
        self,
        lgbm_metrics: dict[str, dict[str, float]],
        splits: dict[str, Any],
        lgbm_pred: np.ndarray,
    ) -> dict[str, Any] | None:
        xgb_metrics = load_model_metrics(self.xgb_model_path, self.xgb_metrics_path)
        xgb_pred = None

        if self.xgb_model_path.exists():
            xgb_bundle = joblib.load(self.xgb_model_path)
            xgb_model = xgb_bundle["model"]
            xgb_features = xgb_bundle["feature_names"]
            xgb_pred = xgb_model.predict(splits["X_test"][xgb_features])
            if xgb_metrics is None:
                xgb_metrics = xgb_bundle.get("metrics")

        if xgb_metrics is None:
            logger.warning("XGBoost metrics/model not found; skipping comparison report")
            return None

        comparison = compare_models(xgb_metrics, lgbm_metrics, split="test")
        y_test = splits["y_test"].to_numpy()
        pred_comparison = None
        if xgb_pred is not None:
            pred_comparison = {
                "xgboost": regression_metrics(y_test, xgb_pred),
                "lightgbm": regression_metrics(y_test, lgbm_pred),
            }
            generate_comparison_report(
                output_path=self.comparison_report_path,
                xgb_metrics=xgb_metrics,
                lgbm_metrics=lgbm_metrics,
                comparison=comparison,
                y_true=y_test,
                xgb_pred=xgb_pred,
                lgbm_pred=lgbm_pred,
            )

        payload = {
            "comparison": comparison,
            "prediction_metrics": pred_comparison,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        comparison_json = self.comparison_report_path.with_suffix(".json")
        comparison_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Model comparison | overall winner: %s | report: %s",
            comparison["overall_winner"],
            self.comparison_report_path,
        )
        return payload

    def _base_params(self, trial_params: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
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
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "max_depth": trial.suggest_int("max_depth", 4, 14),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            fold_scores: list[float] = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMRegressor(**self._base_params(trial_params))
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                preds = model.predict(X_va)
                fold_scores.append(regression_metrics(y_va.to_numpy(), preds)["rmse"])

            return float(np.mean(fold_scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", study_name="lgbm_mcp_next_block")
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
    ) -> tuple[lgb.LGBMRegressor, int]:
        model = lgb.LGBMRegressor(**self._base_params(self._best_params))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = int(model.best_iteration_) if hasattr(model, "best_iteration_") else model.n_estimators
        return model, best_iteration

    @staticmethod
    def _feature_importance_frame(model: lgb.LGBMRegressor, feature_names: list[str]) -> pd.DataFrame:
        return (
            pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _compute_shap(
        self,
        model: lgb.LGBMRegressor,
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
    result = LightGBMMCPTrainer().run()
    print(f"Model:      {result.model_path}")
    print(f"LGBM report: {result.report_path}")
    print(f"Comparison: {result.comparison_report_path}")
    print(f"Test metrics: {result.metrics['test']}")
    if result.xgb_comparison:
        print(f"Overall winner (test): {result.xgb_comparison['comparison']['overall_winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
