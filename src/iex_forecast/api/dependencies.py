"""FastAPI dependency injection."""

from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status

from iex_forecast.application.forecast_service import ForecastService
from iex_forecast.config.settings import Settings, get_settings
from iex_forecast.inference.predictor import RTMForecastPredictor
from iex_forecast.infrastructure.repositories import (
    PostgresForecastRepository,
    PostgresTrainingRunRepository,
    PostgresRTMRepository,
)


@lru_cache
def get_rtm_repository() -> PostgresRTMRepository:
    return PostgresRTMRepository()


@lru_cache
def get_forecast_repository() -> PostgresForecastRepository:
    return PostgresForecastRepository()


@lru_cache
def get_forecast_service() -> ForecastService:
    settings = get_settings()
    return ForecastService(
        repository=get_forecast_repository(),
        csv_backup_dir=settings.forecast_csv_backup_dir,
    )


@lru_cache
def get_predictor() -> RTMForecastPredictor:
    return RTMForecastPredictor(get_settings())


@lru_cache
def get_training_repository() -> PostgresTrainingRunRepository:
    return PostgresTrainingRunRepository()


def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
