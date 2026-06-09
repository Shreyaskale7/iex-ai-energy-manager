"""Domain entities for RTM blocks and forecasts."""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class RTMBlock:
    trade_date: date
    hour: int
    session_id: int
    time_block: int
    purchase_bid_mw: float
    sell_bid_mw: float
    mcv_mw: float
    scheduled_volume_mw: float
    mcp_rs_mwh: float
    block_timestamp: datetime | None = None

    @property
    def global_block_index(self) -> int:
        """Zero-based index of 15-min block within the calendar day (0–95)."""
        return (self.hour - 1) * 4 + (self.time_block - 1)


@dataclass(frozen=True)
class ForecastPoint:
    horizon: int
    forecast_timestamp: datetime
    mcp_forecast_rs_mwh: float
    zone: str = "GREEN"
    confidence: float = 0.0
    spike_probability: float = 0.0
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class ForecastAccuracy:
    """Tracks accuracy of a single forecast block once actuals arrive."""
    run_id: str
    horizon: int
    forecast_timestamp: datetime
    predicted_mcp: float
    actual_mcp: float
    predicted_zone: str
    actual_zone: str
    absolute_error: float
    percentage_error: float | None = None
    evaluated_at: datetime | None = None


@dataclass
class TrainingRunMetadata:
    run_id: str
    started_at: datetime
    finished_at: datetime | None
    horizons_trained: int
    train_rows: int
    test_rows: int
    mean_mae: float
    mean_rmse: float
    model_version: str
    status: str
