"""Abstract interfaces for infrastructure adapters."""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from iex_forecast.domain.entities import ForecastAccuracy, ForecastPoint


class RTMRepository(ABC):
    @abstractmethod
    def upsert_blocks(self, df: pd.DataFrame) -> int:
        ...

    @abstractmethod
    def load_blocks(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        order_desc: bool = False,
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def latest_block_timestamp(self) -> datetime | None:
        ...


class ForecastRepository(ABC):
    @abstractmethod
    def save_forecast(
        self,
        origin_timestamp: datetime,
        points: list[ForecastPoint],
        model_version: str,
        horizon_blocks: int = 2880,
        csv_path: str | None = None,
    ) -> str:
        """Persist a full forecast run and return the run_id."""
        ...

    @abstractmethod
    def get_latest_forecast(self, forecast_type: str | None = None) -> pd.DataFrame:
        """Return all points from the most recent forecast run."""
        ...

    @abstractmethod
    def get_forecast_by_run(self, run_id: str) -> pd.DataFrame:
        """Return all points for a specific run_id."""
        ...

    @abstractmethod
    def get_forecasts_by_date_range(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return forecast points whose forecast_timestamp falls in [start, end]."""
        ...

    @abstractmethod
    def get_forecasts_by_zone(
        self,
        zone: str,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """Return forecast points filtered by zone."""
        ...

    @abstractmethod
    def list_runs(self, limit: int = 20) -> pd.DataFrame:
        """Return metadata for recent forecast runs."""
        ...

    @abstractmethod
    def save_accuracy(self, records: list[ForecastAccuracy]) -> int:
        """Persist forecast-vs-actual accuracy records. Return rows inserted."""
        ...

    @abstractmethod
    def get_accuracy_by_run(self, run_id: str) -> pd.DataFrame:
        """Return accuracy records for a forecast run."""
        ...
