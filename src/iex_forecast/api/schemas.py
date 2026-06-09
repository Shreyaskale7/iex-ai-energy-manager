"""Pydantic request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class ForecastPointSchema(BaseModel):
    horizon: int = Field(..., ge=1, le=2880)
    forecast_timestamp: datetime
    predicted_mcp: float = Field(..., alias="mcp_forecast_rs_mwh", description="Predicted MCP (Rs/MWh)")
    zone: str = Field("GREEN", description="GREEN / YELLOW / RED")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    spike_probability: float = Field(0.0, ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class ForecastResponse(BaseModel):
    run_id: str | None = None
    origin_timestamp: datetime | None = None
    model_version: str
    horizon_blocks: int = 2880
    total_points: int = 0
    zone_summary: dict[str, int] | None = None
    points: list[ForecastPointSchema]


class PaginatedForecastResponse(BaseModel):
    run_id: str | None = None
    origin_timestamp: datetime | None = None
    model_version: str
    total_points: int = 0
    skip: int = 0
    limit: int = 100
    points: list[ForecastPointSchema]


class MarketStatusSchema(BaseModel):
    trade_date: str
    time_block: int
    block_timestamp: datetime
    purchase_bid_mw: float
    sell_bid_mw: float
    mcv_mw: float
    mcp_rs_mwh: float


class MarketStatusResponse(BaseModel):
    latest_blocks: list[MarketStatusSchema]


class ZoneSummaryResponse(BaseModel):
    run_id: str
    generated_at: datetime
    green_blocks: int
    yellow_blocks: int
    red_blocks: int
    total_blocks: int


# ── Forecast Runs ────────────────────────────────────────────────────
class ForecastRunSchema(BaseModel):
    run_id: str
    generated_at: datetime
    model_version: str
    feature_version: str
    forecast_type: str
    status: str
    csv_path: str | None = None


class ForecastRunListResponse(BaseModel):
    runs: list[ForecastRunSchema]
    total: int


# ── Accuracy ─────────────────────────────────────────────────────────
class AccuracyPointSchema(BaseModel):
    horizon: int
    forecast_timestamp: datetime
    predicted_mcp: float
    actual_mcp: float
    predicted_zone: str
    actual_zone: str
    absolute_error: float
    percentage_error: float | None = None


class AccuracySummaryResponse(BaseModel):
    run_id: str
    records_evaluated: int
    mae: float
    rmse: float
    zone_accuracy_pct: float
    zone_breakdown: dict[str, int]


class AccuracyDetailResponse(BaseModel):
    run_id: str
    records: list[AccuracyPointSchema]
    total: int


# ── Training ─────────────────────────────────────────────────────────
class TrainingMetricsResponse(BaseModel):
    id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    horizons_trained: int | None = None
    train_rows: int | None = None
    test_rows: int | None = None
    mean_mae: float | None = None
    mean_rmse: float | None = None
    model_version: str | None = None
    status: str | None = None


# ── Errors ───────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    detail: str


# ── Decision Engine ──────────────────────────────────────────────────
from iex_forecast.domain.devices import DeviceProfile, DeviceState
from iex_forecast.application.decision_engine import CustomThresholds

class DecisionEngineRequest(BaseModel):
    forecast_type: str = Field("24-Hour", description="Forecast horizon to base decisions on")
    devices: list[DeviceProfile]
    thresholds: CustomThresholds | None = None

class DeviceStateRecommendation(BaseModel):
    device_id: str
    name: str
    category: str
    recommended_state: str

class ScheduleBlock(BaseModel):
    forecast_timestamp: datetime
    predicted_mcp: float
    effective_zone: str
    total_load_kw: float
    device_states: list[DeviceStateRecommendation]

class DecisionEngineResponse(BaseModel):
    baseline_cost_rs: float
    optimized_cost_rs: float
    expected_savings_rs: float
    savings_percentage: float
    recommended_schedule: list[ScheduleBlock]
