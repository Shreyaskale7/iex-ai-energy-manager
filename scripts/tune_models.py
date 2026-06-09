#!/usr/bin/env python3
"""CLI: tune hyperparameters using Optuna."""

import sys
import json
import argparse
from pathlib import Path

import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iex_forecast.core.logging import configure_logging, get_logger
from iex_forecast.features.training_matrix import TrainingMatrixBuilder
from iex_forecast.config.settings import get_settings

import lightgbm as lgb
import xgboost as xgb

logger = get_logger(__name__)

def objective(trial, X_train, y_train, X_test, y_test, model_type="lightgbm"):
    if model_type == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "regression",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
    elif model_type == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgb.XGBRegressor(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

def main() -> int:
    parser = argparse.ArgumentParser(description="Tune Models")
    parser.add_argument("--trials", type=int, default=10, help="Number of optuna trials")
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    
    logger.info("Loading dataset...")
    from iex_forecast.data.pipeline import DataPipeline
    pipeline = DataPipeline(settings)
    df = pipeline.load_from_db()
    if df.empty:
        df = pipeline.load_processed_parquet()

    if df.empty:
        logger.error("No data found")
        return 1
    
    matrix_builder = TrainingMatrixBuilder()
    train_df, test_df = matrix_builder.temporal_split(df, settings.train_test_split_date)
    
    # We tune on horizon 1 as a proxy for hyperparams
    horizon = 1
    X_train, y_train = matrix_builder.build_for_horizon(train_df, horizon)
    X_test, y_test = matrix_builder.build_for_horizon(test_df, horizon)
    
    if X_train.empty or X_test.empty:
        logger.error("Empty training matrix")
        return 1

    tuned_params = {}

    # Tune LightGBM
    logger.info("Tuning LightGBM...")
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, "lightgbm"), n_trials=args.trials)
    logger.info(f"Best LightGBM params: {study_lgb.best_params}")
    tuned_params["lightgbm"] = study_lgb.best_params

    # Tune XGBoost
    logger.info("Tuning XGBoost...")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, "xgboost"), n_trials=args.trials)
    logger.info(f"Best XGBoost params: {study_xgb.best_params}")
    tuned_params["xgboost"] = study_xgb.best_params
    
    config_dir = ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    params_file = config_dir / "model_params.json"
    
    with open(params_file, "w") as f:
        json.dump(tuned_params, f, indent=4)
        
    logger.info(f"Saved tuned parameters to {params_file}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
