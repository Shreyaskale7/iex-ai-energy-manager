"""Weighted ensemble of XGBoost, LightGBM, and CatBoost per horizon."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from iex_forecast.models.boosters import BOOSTER_FACTORIES, QUANTILE_FACTORIES


class HorizonEnsemble:
    def __init__(self, weights: dict[str, float], model_params: dict[str, dict] = None) -> None:
        self.weights = weights
        self.model_params = model_params or {}
        self.models: dict[str, BaseEstimator] = {}
        self.quantile_models_p10: dict[str, BaseEstimator] = {}
        self.quantile_models_p90: dict[str, BaseEstimator] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HorizonEnsemble":
        self.models = {}
        self.quantile_models_p10 = {}
        self.quantile_models_p90 = {}
        
        # Fit Mean models
        for name, factory in BOOSTER_FACTORIES.items():
            if name not in self.weights:
                continue
            params = self.model_params.get(name, {})
            model = factory(**params)
            model.fit(X, y)
            self.models[name] = model

        # Fit Quantile models
        for name, factory in QUANTILE_FACTORIES.items():
            if name not in self.weights:
                continue
            params = self.model_params.get(name, {})
            
            p10_model = factory(alpha=0.1, **params)
            p10_model.fit(X, y)
            self.quantile_models_p10[name] = p10_model
            
            p90_model = factory(alpha=0.9, **params)
            p90_model.fit(X, y)
            self.quantile_models_p90[name] = p90_model
            
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("Ensemble is not fitted")

        predictions = []
        total_weight = 0.0
        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            if weight <= 0:
                continue
            pred = model.predict(X)
            predictions.append(pred * weight)
            total_weight += weight

        if total_weight == 0:
            raise ValueError("No models with positive weight available for prediction")

        stacked = np.vstack(predictions)
        return stacked.sum(axis=0) / total_weight

    def predict_with_interval(
        self,
        X: pd.DataFrame,
        z_score: float = 1.96,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mean prediction and bounds from native quantile models or fallback."""
        mean_pred = self.predict(X)
        
        if self.quantile_models_p10 and self.quantile_models_p90:
            p10_preds = []
            p90_preds = []
            total_weight = 0.0
            
            for name in self.quantile_models_p10.keys():
                weight = self.weights.get(name, 0.0)
                if weight <= 0:
                    continue
                p10_preds.append(self.quantile_models_p10[name].predict(X) * weight)
                p90_preds.append(self.quantile_models_p90[name].predict(X) * weight)
                total_weight += weight
                
            lower = np.vstack(p10_preds).sum(axis=0) / total_weight
            upper = np.vstack(p90_preds).sum(axis=0) / total_weight
            return mean_pred, lower, upper
            
        # Fallback to standard deviation
        member_preds = []
        for model in self.models.values():
            member_preds.append(model.predict(X))
        matrix = np.vstack(member_preds)
        std = matrix.std(axis=0)
        return mean_pred, mean_pred - z_score * std, mean_pred + z_score * std
