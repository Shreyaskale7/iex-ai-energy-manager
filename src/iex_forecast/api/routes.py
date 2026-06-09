"""API route handlers."""

import sys
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
import pandas as pd

from iex_forecast.api.dependencies import (
    get_forecast_service,
    get_rtm_repository,
    verify_api_key,
)
from iex_forecast.api.schemas import (
    ForecastPointSchema,
    ForecastResponse,
    HealthResponse,
    MarketStatusResponse,
    MarketStatusSchema,
    PaginatedForecastResponse,
    ZoneSummaryResponse,
)
from iex_forecast.application.forecast_engine import ForecastGenerationService
from iex_forecast.application.forecast_service import ForecastService
from iex_forecast.config.settings import Settings, get_settings
from iex_forecast.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
    )


# ── Market Status ────────────────────────────────────────────────────
@router.get("/market-status", response_model=MarketStatusResponse)
def market_status(
    limit: int = Query(96, ge=1, le=1000),
    rtm_repo = Depends(get_rtm_repository),
) -> MarketStatusResponse:
    df = rtm_repo.load_blocks(limit=limit, order_desc=True)
    if df.empty:
        raise HTTPException(status_code=404, detail="No market data available.")
    
    blocks = [
        MarketStatusSchema(
            trade_date=str(row["trade_date"]),
            time_block=row["time_block"],
            block_timestamp=row["block_timestamp"],
            purchase_bid_mw=float(row["purchase_bid_mw"]),
            sell_bid_mw=float(row["sell_bid_mw"]),
            mcv_mw=float(row["mcv_mw"]),
            mcp_rs_mwh=float(row["mcp_rs_mwh"])
        )
        for _, row in df.iterrows()
    ]
    return MarketStatusResponse(latest_blocks=blocks)


# ── Forecast: Latest ──────────────────────────────────────────────────
@router.get("/forecast/latest", response_model=ForecastResponse)
def get_latest_forecast(
    forecast_type: str = Query("24-Hour", description="24-Hour, 7-Day, or 30-Day"),
    service: ForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    df = service.get_latest_forecast(forecast_type=forecast_type)
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {forecast_type} forecast available.",
        )
    return _dataframe_to_response(df)


# ── Forecast: Day ─────────────────────────────────────────────────────
@router.get("/forecast/day", response_model=ForecastResponse)
def get_forecast_day(
    service: ForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    df = service.get_latest_forecast(forecast_type="24-Hour")
    if df.empty:
        raise HTTPException(status_code=404, detail="No 24-Hour forecast available.")
    df = df.head(96)
    return _dataframe_to_response(df)


# ── Forecast: Week ────────────────────────────────────────────────────
@router.get("/forecast/week", response_model=ForecastResponse)
def get_forecast_week(
    service: ForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    df = service.get_latest_forecast(forecast_type="7-Day")
    if df.empty:
        raise HTTPException(status_code=404, detail="No 7-Day forecast available.")
    df = df.head(672)
    return _dataframe_to_response(df)


# ── Forecast: Month ───────────────────────────────────────────────────
@router.get("/forecast/month", response_model=PaginatedForecastResponse)
def get_forecast_month(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=2880),
    service: ForecastService = Depends(get_forecast_service),
) -> PaginatedForecastResponse:
    df = service.get_latest_forecast(forecast_type="30-Day")
    if df.empty:
        raise HTTPException(status_code=404, detail="No 30-Day forecast available.")
    
    total = len(df)
    df_slice = df.iloc[skip : skip + limit]
    
    return _dataframe_to_paginated_response(df_slice, skip=skip, limit=limit, total=total)


# ── Forecast: By Date ─────────────────────────────────────────────────
@router.get("/forecast/date/{date}", response_model=ForecastResponse)
def get_forecast_by_date(
    date: str,
    service: ForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    try:
        dt = pd.to_datetime(date).date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
    df = service.get_latest_forecast(forecast_type="30-Day")
    if df.empty:
        raise HTTPException(status_code=404, detail="No forecasts available.")
        
    df["date_only"] = df["forecast_timestamp"].dt.date
    df_filtered = df[df["date_only"] == dt]
    
    if df_filtered.empty:
        raise HTTPException(status_code=404, detail=f"No forecast data for date {date}.")
        
    return _dataframe_to_response(df_filtered)


# ── Zones ────────────────────────────────────────────────────────────
@router.get("/zones", response_model=ZoneSummaryResponse)
def get_zones_summary(
    forecast_type: str = Query("24-Hour", description="24-Hour, 7-Day, or 30-Day"),
    service: ForecastService = Depends(get_forecast_service),
) -> ZoneSummaryResponse:
    df = service.get_latest_forecast(forecast_type=forecast_type)
    if df.empty:
        raise HTTPException(status_code=404, detail="No forecasts available.")
        
    green = len(df[df["zone"] == "GREEN"])
    yellow = len(df[df["zone"] == "YELLOW"])
    red = len(df[df["zone"] == "RED"])
    
    return ZoneSummaryResponse(
        run_id=str(df["run_id"].iloc[0]),
        generated_at=df["generated_at"].iloc[0],
        green_blocks=green,
        yellow_blocks=yellow,
        red_blocks=red,
        total_blocks=len(df),
    )


# ── Alerts ───────────────────────────────────────────────────────────
@router.get("/alerts", response_model=PaginatedForecastResponse)
def get_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    service: ForecastService = Depends(get_forecast_service),
) -> PaginatedForecastResponse:
    df = service.get_latest_forecast(forecast_type="30-Day")
    if df.empty:
        raise HTTPException(status_code=404, detail="No forecasts available.")
        
    # Condition: RED zone or spike prob > 0.5
    alerts_df = df[(df["zone"] == "RED") | (df["spike_probability"] > 0.5)]
    total = len(alerts_df)
    
    df_slice = alerts_df.iloc[skip : skip + limit]
    
    return _dataframe_to_paginated_response(df_slice, skip=skip, limit=limit, total=total)


# ── Generate Forecasts ────────────────────────────────────────────────
from pydantic import BaseModel

class GenerateForecastRequest(BaseModel):
    forecast_type: str = "24h" # 24h, 7d, 30d, all

class GenerateForecastResultSchema(BaseModel):
    forecast_type: str
    run_id: str
    total_blocks: int
    generated_at: str
    model_version: str
    csv_latest_path: str
    csv_archive_path: str
    zone_summary: dict

@router.post(
    "/forecast/generate",
    response_model=List[GenerateForecastResultSchema],
    dependencies=[Depends(verify_api_key)],
)
def generate_forecasts(
    request: GenerateForecastRequest,
) -> List[GenerateForecastResultSchema]:
    ROOT = Path(sys.modules['iex_forecast'].__file__).parents[2]
    
    engine = ForecastGenerationService(
        ensemble_path=ROOT / "models" / "ensemble.pkl",
        direct_model_dir=ROOT / "models" / "direct",
        spike_classifier_path=ROOT / "models" / "spike_classifier.pkl",
        master_path=ROOT / "data" / "processed" / "rtm_master.parquet",
        features_path=ROOT / "data" / "features" / "features.parquet",
        output_dir=ROOT / "forecasts"
    )
    
    results = []
    if request.forecast_type == "all":
        results = engine.generate_all()
    elif request.forecast_type == "24h":
        results = [engine.generate_24h()]
    elif request.forecast_type == "7d":
        results = [engine.generate_7d()]
    elif request.forecast_type == "30d":
        results = [engine.generate_30d()]
    else:
        raise HTTPException(status_code=400, detail="Invalid forecast_type")
        
    return [
        GenerateForecastResultSchema(
            forecast_type=r.forecast_type,
            run_id=r.run_id,
            total_blocks=r.total_blocks,
            generated_at=r.generated_at,
            model_version=r.model_version,
            csv_latest_path=r.csv_latest_path,
            csv_archive_path=r.csv_archive_path,
            zone_summary=r.zone_summary,
        ) for r in results
    ]


# ── Decision Engine ───────────────────────────────────────────────────
from iex_forecast.api.schemas import DecisionEngineRequest, DecisionEngineResponse
from iex_forecast.application.decision_engine import DecisionEngineService

@router.post("/decision/schedule", response_model=DecisionEngineResponse)
def get_decision_schedule(
    request: DecisionEngineRequest,
    service: ForecastService = Depends(get_forecast_service),
) -> DecisionEngineResponse:
    df = service.get_latest_forecast(forecast_type=request.forecast_type)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No {request.forecast_type} forecast available.")
        
    engine = DecisionEngineService()
    
    # Convert df to list of dicts
    points = df.to_dict('records')
    
    result = engine.calculate_schedule(points, request.devices, request.thresholds)
    return DecisionEngineResponse(**result)


# ── Helpers ───────────────────────────────────────────────────────────

def _dataframe_to_response(df: pd.DataFrame) -> ForecastResponse:
    """Convert a repository DataFrame to a ForecastResponse."""
    if df.empty:
        return ForecastResponse(model_version="unknown", points=[])
        
    mcp_col = "predicted_mcp" if "predicted_mcp" in df.columns else "mcp_forecast_rs_mwh"

    points = [
        ForecastPointSchema(
            horizon=int(row["forecast_horizon"] if "forecast_horizon" in row else row["horizon"]),
            forecast_timestamp=row["forecast_timestamp"],
            predicted_mcp=float(row[mcp_col]),
            zone=row.get("zone", "GREEN"),
            confidence=float(row.get("confidence", 0.0)),
            spike_probability=float(row.get("spike_probability", 0.0)),
            lower_bound=row.get("lower_bound"),
            upper_bound=row.get("upper_bound"),
        )
        for _, row in df.iterrows()
    ]

    zone_summary = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    if "zone" in df.columns:
        for z in df["zone"]:
            zone_summary[z] = zone_summary.get(z, 0) + 1

    origin = df["generated_at"].iloc[0] if "generated_at" in df.columns else None
    run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns else None
    
    return ForecastResponse(
        run_id=run_id,
        origin_timestamp=origin,
        model_version=str(df["model_version"].iloc[0]) if "model_version" in df.columns else "v1",
        horizon_blocks=len(points),
        total_points=len(points),
        zone_summary=zone_summary,
        points=points,
    )


def _dataframe_to_paginated_response(
    df: pd.DataFrame, skip: int, limit: int, total: int
) -> PaginatedForecastResponse:
    """Convert a DataFrame slice to a PaginatedForecastResponse."""
    if df.empty:
        return PaginatedForecastResponse(model_version="unknown", points=[], total_points=total, skip=skip, limit=limit)
        
    mcp_col = "predicted_mcp" if "predicted_mcp" in df.columns else "mcp_forecast_rs_mwh"

    points = [
        ForecastPointSchema(
            horizon=int(row["forecast_horizon"] if "forecast_horizon" in row else row["horizon"]),
            forecast_timestamp=row["forecast_timestamp"],
            predicted_mcp=float(row[mcp_col]),
            zone=row.get("zone", "GREEN"),
            confidence=float(row.get("confidence", 0.0)),
            spike_probability=float(row.get("spike_probability", 0.0)),
            lower_bound=row.get("lower_bound"),
            upper_bound=row.get("upper_bound"),
        )
        for _, row in df.iterrows()
    ]

    origin = df["generated_at"].iloc[0] if "generated_at" in df.columns else None
    run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns else None

    return PaginatedForecastResponse(
        run_id=run_id,
        origin_timestamp=origin,
        model_version=str(df["model_version"].iloc[0]) if "model_version" in df.columns else "v1",
        total_points=total,
        skip=skip,
        limit=limit,
        points=points,
    )
