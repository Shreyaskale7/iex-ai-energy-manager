"""Generate 96-block (24h) MCP forecasts."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from iex_forecast.config.settings import Settings
from iex_forecast.core.exceptions import ForecastError, ModelNotFoundError
from iex_forecast.core.logging import get_logger
from iex_forecast.domain.constants import BLOCK_MINUTES, FORECAST_HORIZON
from iex_forecast.domain.entities import ForecastPoint
from iex_forecast.features.builder import FeatureBuilder
from iex_forecast.infrastructure.repositories import (
    PostgresForecastRepository,
    PostgresRTMRepository,
)
from iex_forecast.models.registry import ModelRegistry

logger = get_logger(__name__)


class RTMForecastPredictor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.feature_builder = FeatureBuilder()
        self.registry = ModelRegistry(settings.model_registry_dir, version="v1")
        self.rtm_repo = PostgresRTMRepository()
        self.forecast_repo = PostgresForecastRepository()
        self._models = None

    def _load_models(self) -> dict:
        if self._models is None:
            if not self.registry.is_complete():
                raise ModelNotFoundError(
                    "Model registry incomplete. Run training before inference."
                )
            self._models = self.registry.load_all()
        return self._models

    def predict(
        self,
        history: pd.DataFrame | None = None,
        persist: bool = True,
    ) -> tuple[datetime, list[ForecastPoint]]:
        if history is None:
            history = self.rtm_repo.load_blocks()

        if history.empty or len(history) < 200:
            raise ForecastError(
                "Insufficient historical blocks for feature generation (need >= 200)"
            )

        history = history.sort_values("block_timestamp").reset_index(drop=True)
        origin_ts = pd.to_datetime(history["block_timestamp"].iloc[-1])
        if origin_ts.tzinfo is None:
            origin_ts = origin_ts.tz_localize("Asia/Kolkata")

        models = self._load_models()
        points: list[ForecastPoint] = []
        working_history = history.copy()

        for horizon in range(1, FORECAST_HORIZON + 1):
            X = self.feature_builder.transform_latest_row(working_history, horizon)
            if X.empty or X.isna().any().any():
                raise ForecastError(f"Feature row contains NaN at horizon {horizon}")

            ensemble = models[horizon]
            mean, lower, upper = ensemble.predict_with_interval(X)
            forecast_value = float(mean[0])
            forecast_ts = origin_ts + timedelta(minutes=BLOCK_MINUTES * horizon)

            points.append(
                ForecastPoint(
                    horizon=horizon,
                    forecast_timestamp=forecast_ts.to_pydatetime(),
                    mcp_forecast_rs_mwh=round(forecast_value, 2),
                    lower_bound=round(float(lower[0]), 2),
                    upper_bound=round(float(upper[0]), 2),
                )
            )

            synthetic = working_history.iloc[[-1]].copy()
            synthetic["block_timestamp"] = forecast_ts
            synthetic["mcp_rs_mwh"] = forecast_value
            minutes_from_midnight = forecast_ts.hour * 60 + forecast_ts.minute
            synthetic["hour"] = minutes_from_midnight // 60 + 1
            synthetic["time_block"] = (minutes_from_midnight % 60) // 15 + 1
            working_history = pd.concat([working_history, synthetic], ignore_index=True)

        if persist:
            run_id = self.forecast_repo.save_forecast(
                origin_timestamp=origin_ts.to_pydatetime(),
                points=points,
                model_version=self.registry.version,
            )
            logger.info("forecast_persisted", run_id=run_id, points=len(points))

        return origin_ts.to_pydatetime(), points

    def latest_forecast_dataframe(self) -> pd.DataFrame:
        return self.forecast_repo.get_latest_forecast()
