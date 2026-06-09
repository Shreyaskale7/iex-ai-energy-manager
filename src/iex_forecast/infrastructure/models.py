"""SQLAlchemy ORM models."""

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ── RTM Market Blocks ────────────────────────────────────────────────
class RTMBlockRow(Base):
    __tablename__ = "rtm_blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False)
    time_block: Mapped[int] = mapped_column(Integer, nullable=False)
    purchase_bid_mw: Mapped[float] = mapped_column(Float, nullable=False)
    sell_bid_mw: Mapped[float] = mapped_column(Float, nullable=False)
    mcv_mw: Mapped[float] = mapped_column(Float, nullable=False)
    scheduled_volume_mw: Mapped[float] = mapped_column(Float, nullable=False)
    mcp_rs_mwh: Mapped[float] = mapped_column(Float, nullable=False)
    block_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "trade_date",
            "hour",
            "session_id",
            "time_block",
            name="uq_rtm_block_identity",
        ),
        Index("ix_rtm_blocks_block_timestamp", "block_timestamp"),
    )


# ── Forecast Runs ────────────────────────────────────────────────────
class ForecastRunRow(Base):
    __tablename__ = "forecast_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False, server_default="v1")
    forecast_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    csv_path: Mapped[str | None] = mapped_column(String(512), nullable=True)


# ── Forecasts (individual predicted blocks) ──────────────────────────
class ForecastRow(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    forecast_run_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    forecast_horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    forecast_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    block_number: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_mcp: Mapped[float] = mapped_column(Float, nullable=False)
    zone: Mapped[str] = mapped_column(String(10), default="GREEN", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    spike_probability: Mapped[float] = mapped_column(Float, default=0.0)
    lower_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("forecast_run_id", "forecast_horizon", name="uq_forecast_horizon"),
        Index("ix_forecasts_timestamp", "forecast_timestamp"),
        Index("ix_forecasts_zone", "zone"),
    )


# ── Forecast Accuracy (populated when actuals arrive) ────────────────
class ForecastAccuracyRow(Base):
    __tablename__ = "forecast_accuracy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    forecast_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    predicted_mcp: Mapped[float] = mapped_column(Float, nullable=False)
    actual_mcp: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_zone: Mapped[str] = mapped_column(String(10), nullable=False)
    actual_zone: Mapped[str] = mapped_column(String(10), nullable=False)
    absolute_error: Mapped[float] = mapped_column(Float, nullable=False)
    percentage_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("run_id", "horizon", name="uq_accuracy_run_horizon"),
        Index("ix_accuracy_run_id", "run_id"),
    )


# ── Training Runs ────────────────────────────────────────────────────
class TrainingRunRow(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    horizons_trained: Mapped[int] = mapped_column(Integer, nullable=False)
    train_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    test_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_mae: Mapped[float] = mapped_column(Float, nullable=False)
    mean_rmse: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)


# Keep legacy alias so existing imports don't break
ForecastPointRow = ForecastRow
