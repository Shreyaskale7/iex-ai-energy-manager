"""PostgreSQL repository implementations."""

import uuid
from datetime import datetime

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from iex_forecast.domain.entities import ForecastAccuracy, ForecastPoint
from iex_forecast.domain.interfaces import ForecastRepository, RTMRepository
from iex_forecast.infrastructure.database import get_session_factory
from iex_forecast.infrastructure.models import (
    ForecastAccuracyRow,
    ForecastRow,
    ForecastRunRow,
    RTMBlockRow,
    TrainingRunRow,
)


# ── RTM Market Data ──────────────────────────────────────────────────
class PostgresRTMRepository(RTMRepository):
    def upsert_blocks(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        records = df.to_dict(orient="records")
        session = get_session_factory()()
        try:
            stmt = insert(RTMBlockRow).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_rtm_block_identity",
                set_={
                    "purchase_bid_mw": stmt.excluded.purchase_bid_mw,
                    "sell_bid_mw": stmt.excluded.sell_bid_mw,
                    "mcv_mw": stmt.excluded.mcv_mw,
                    "scheduled_volume_mw": stmt.excluded.scheduled_volume_mw,
                    "mcp_rs_mwh": stmt.excluded.mcp_rs_mwh,
                    "block_timestamp": stmt.excluded.block_timestamp,
                    "source_file": stmt.excluded.source_file,
                },
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount or len(records)
        finally:
            session.close()

    def load_blocks(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        order_desc: bool = False,
    ) -> pd.DataFrame:
        session = get_session_factory()()
        try:
            if order_desc:
                query = select(RTMBlockRow).order_by(RTMBlockRow.block_timestamp.desc())
            else:
                query = select(RTMBlockRow).order_by(RTMBlockRow.block_timestamp)
                
            if start is not None:
                query = query.where(RTMBlockRow.block_timestamp >= start)
            if end is not None:
                query = query.where(RTMBlockRow.block_timestamp <= end)
            if limit is not None:
                query = query.limit(limit)
                
            rows = session.execute(query).scalars().all()
            if not rows:
                return pd.DataFrame()
            data = [
                {
                    "trade_date": r.trade_date,
                    "hour": r.hour,
                    "session_id": r.session_id,
                    "time_block": r.time_block,
                    "purchase_bid_mw": r.purchase_bid_mw,
                    "sell_bid_mw": r.sell_bid_mw,
                    "mcv_mw": r.mcv_mw,
                    "scheduled_volume_mw": r.scheduled_volume_mw,
                    "mcp_rs_mwh": r.mcp_rs_mwh,
                    "block_timestamp": r.block_timestamp,
                    "source_file": r.source_file,
                }
                for r in rows
            ]
            return pd.DataFrame(data)
        finally:
            session.close()

    def latest_block_timestamp(self) -> datetime | None:
        session = get_session_factory()()
        try:
            return session.execute(select(func.max(RTMBlockRow.block_timestamp))).scalar()
        finally:
            session.close()


# ── Forecast Storage ─────────────────────────────────────────────────
class PostgresForecastRepository(ForecastRepository):

    # ── Write ─────────────────────────────────────────────────────────

    def save_forecast(
        self,
        origin_timestamp: datetime,
        points: list[ForecastPoint],
        model_version: str,
        horizon_blocks: int = 2880,
        csv_path: str | None = None,
        feature_version: str = "v1",
        forecast_type: str = "30-Day",
    ) -> str:
        run_id = str(uuid.uuid4())
        session = get_session_factory()()
        try:
            session.add(
                ForecastRunRow(
                    id=run_id,
                    generated_at=origin_timestamp,
                    model_version=model_version,
                    feature_version=feature_version,
                    forecast_type=forecast_type,
                    status="completed",
                    csv_path=csv_path,
                )
            )
            for point in points:
                # Calculate daily block index (1-96)
                block_num = ((point.forecast_timestamp.hour * 60 + point.forecast_timestamp.minute) // 15) + 1
                session.add(
                    ForecastRow(
                        forecast_run_id=run_id,
                        forecast_horizon=point.horizon,
                        forecast_timestamp=point.forecast_timestamp,
                        block_number=block_num,
                        predicted_mcp=point.mcp_forecast_rs_mwh,
                        zone=point.zone,
                        confidence=point.confidence,
                        spike_probability=point.spike_probability,
                        lower_bound=point.lower_bound,
                        upper_bound=point.upper_bound,
                    )
                )
            session.commit()
            return run_id
        finally:
            session.close()

    def save_accuracy(self, records: list[ForecastAccuracy]) -> int:
        if not records:
            return 0
        session = get_session_factory()()
        try:
            for rec in records:
                session.add(
                    ForecastAccuracyRow(
                        run_id=rec.run_id,
                        horizon=rec.horizon,
                        forecast_timestamp=rec.forecast_timestamp,
                        predicted_mcp=rec.predicted_mcp,
                        actual_mcp=rec.actual_mcp,
                        predicted_zone=rec.predicted_zone,
                        actual_zone=rec.actual_zone,
                        absolute_error=rec.absolute_error,
                        percentage_error=rec.percentage_error,
                        evaluated_at=rec.evaluated_at or datetime.utcnow(),
                    )
                )
            session.commit()
            return len(records)
        finally:
            session.close()

    # ── Read ──────────────────────────────────────────────────────────

    def get_latest_forecast(self, forecast_type: str | None = None) -> pd.DataFrame:
        session = get_session_factory()()
        try:
            stmt = select(ForecastRunRow.id).order_by(ForecastRunRow.generated_at.desc())
            if forecast_type:
                stmt = stmt.where(ForecastRunRow.forecast_type == forecast_type)
                
            latest_run_id = session.execute(stmt.limit(1)).scalar_one_or_none()
            if latest_run_id is None:
                return pd.DataFrame()
            return self._load_run_points(session, latest_run_id)
        finally:
            session.close()

    def get_forecast_by_run(self, run_id: str) -> pd.DataFrame:
        session = get_session_factory()()
        try:
            return self._load_run_points(session, run_id)
        finally:
            session.close()

    def get_forecasts_by_date_range(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        session = get_session_factory()()
        try:
            # Use the latest run that overlaps the window
            latest_run_id = session.execute(
                select(ForecastRunRow.id)
                .order_by(ForecastRunRow.generated_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if latest_run_id is None:
                return pd.DataFrame()

            points = (
                session.execute(
                    select(ForecastRow)
                    .where(ForecastRow.forecast_run_id == latest_run_id)
                    .where(ForecastRow.forecast_timestamp >= start)
                    .where(ForecastRow.forecast_timestamp <= end)
                    .order_by(ForecastRow.forecast_horizon)
                )
                .scalars()
                .all()
            )
            return self._points_to_dataframe(points, latest_run_id, session)
        finally:
            session.close()

    def get_forecasts_by_zone(
        self,
        zone: str,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        session = get_session_factory()()
        try:
            if run_id is None:
                run_id = session.execute(
                    select(ForecastRunRow.id)
                    .order_by(ForecastRunRow.generated_at.desc())
                    .limit(1)
                ).scalar_one_or_none()
            if run_id is None:
                return pd.DataFrame()

            points = (
                session.execute(
                    select(ForecastRow)
                    .where(ForecastRow.forecast_run_id == run_id)
                    .where(ForecastRow.zone == zone.upper())
                    .order_by(ForecastRow.forecast_horizon)
                )
                .scalars()
                .all()
            )
            return self._points_to_dataframe(points, run_id, session)
        finally:
            session.close()

    def list_runs(self, limit: int = 20) -> pd.DataFrame:
        session = get_session_factory()()
        try:
            runs = (
                session.execute(
                    select(ForecastRunRow)
                    .order_by(ForecastRunRow.generated_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            if not runs:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {
                        "run_id": r.id,
                        "generated_at": r.generated_at,
                        "model_version": r.model_version,
                        "feature_version": r.feature_version,
                        "forecast_type": r.forecast_type,
                        "status": r.status,
                        "csv_path": r.csv_path,
                    }
                    for r in runs
                ]
            )
        finally:
            session.close()

    def get_accuracy_by_run(self, run_id: str) -> pd.DataFrame:
        session = get_session_factory()()
        try:
            rows = (
                session.execute(
                    select(ForecastAccuracyRow)
                    .where(ForecastAccuracyRow.run_id == run_id)
                    .order_by(ForecastAccuracyRow.horizon)
                )
                .scalars()
                .all()
            )
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {
                        "run_id": r.run_id,
                        "horizon": r.horizon,
                        "forecast_timestamp": r.forecast_timestamp,
                        "predicted_mcp": r.predicted_mcp,
                        "actual_mcp": r.actual_mcp,
                        "predicted_zone": r.predicted_zone,
                        "actual_zone": r.actual_zone,
                        "absolute_error": r.absolute_error,
                        "percentage_error": r.percentage_error,
                        "evaluated_at": r.evaluated_at,
                    }
                    for r in rows
                ]
            )
        finally:
            session.close()

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_run_points(session, run_id: str) -> pd.DataFrame:
        points = (
            session.execute(
                select(ForecastRow)
                .where(ForecastRow.forecast_run_id == run_id)
                .order_by(ForecastRow.forecast_horizon)
            )
            .scalars()
            .all()
        )
        run = session.get(ForecastRunRow, run_id)
        return PostgresForecastRepository._points_to_dataframe(points, run_id, session, run)

    @staticmethod
    def _points_to_dataframe(points, run_id, session, run=None) -> pd.DataFrame:
        if not points:
            return pd.DataFrame()
        if run is None:
            run = session.get(ForecastRunRow, run_id)
        return pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "generated_at": run.generated_at if run else None,
                    "model_version": run.model_version if run else None,
                    "feature_version": run.feature_version if run else None,
                    "forecast_type": run.forecast_type if run else None,
                    "forecast_horizon": p.forecast_horizon,
                    "forecast_timestamp": p.forecast_timestamp,
                    "block_number": p.block_number,
                    "predicted_mcp": p.predicted_mcp,
                    "zone": p.zone,
                    "confidence": p.confidence,
                    "spike_probability": p.spike_probability,
                    "lower_bound": p.lower_bound,
                    "upper_bound": p.upper_bound,
                }
                for p in points
            ]
        )


# ── Training Runs ────────────────────────────────────────────────────
class PostgresTrainingRunRepository:
    def save(self, metadata: dict) -> None:
        session = get_session_factory()()
        try:
            session.add(TrainingRunRow(**metadata))
            session.commit()
        finally:
            session.close()

    def latest(self) -> dict | None:
        session = get_session_factory()()
        try:
            row = session.execute(
                select(TrainingRunRow).order_by(TrainingRunRow.started_at.desc()).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "id": row.id,
                "started_at": row.started_at,
                "finished_at": row.finished_at,
                "horizons_trained": row.horizons_trained,
                "train_rows": row.train_rows,
                "test_rows": row.test_rows,
                "mean_mae": row.mean_mae,
                "mean_rmse": row.mean_rmse,
                "model_version": row.model_version,
                "status": row.status,
                "metrics_json": row.metrics_json,
            }
        finally:
            session.close()

    def delete_all_forecasts(self) -> None:
        session = get_session_factory()()
        try:
            session.execute(delete(ForecastAccuracyRow))
            session.execute(delete(ForecastRow))
            session.execute(delete(ForecastRunRow))
            session.commit()
        finally:
            session.close()
