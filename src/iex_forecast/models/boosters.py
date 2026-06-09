"""Gradient boosting model factories."""

from typing import Any

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator


def build_xgboost(**kwargs) -> BaseEstimator:
    params = {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


def build_lightgbm(**kwargs) -> BaseEstimator:
    params = {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "regression",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(kwargs)
    return lgb.LGBMRegressor(**params)


def build_lightgbm_quantile(alpha: float = 0.5, **kwargs) -> BaseEstimator:
    params = {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "quantile",
        "alpha": alpha,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(kwargs)
    return lgb.LGBMRegressor(**params)


def build_catboost(**kwargs) -> BaseEstimator:
    params = {
        "iterations": 400,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(kwargs)
    return CatBoostRegressor(**params)


def build_catboost_quantile(alpha: float = 0.5, **kwargs) -> BaseEstimator:
    params = {
        "iterations": 400,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": f"Quantile:alpha={alpha}",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(kwargs)
    return CatBoostRegressor(**params)


from iex_forecast.models.pytorch_models import PyTorchLSTMModel

def build_pytorch_lstm(**kwargs) -> BaseEstimator:
    params = {
        "hidden_dim": 64,
        "num_layers": 2,
        "epochs": 10,
        "lr": 0.005,
    }
    params.update(kwargs)
    return PyTorchLSTMModel(**params)


BOOSTER_FACTORIES: dict[str, Any] = {
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "catboost": build_catboost,
    "pytorch_lstm": build_pytorch_lstm,
}

QUANTILE_FACTORIES: dict[str, Any] = {
    "lightgbm": build_lightgbm_quantile,
    "catboost": build_catboost_quantile,
}
