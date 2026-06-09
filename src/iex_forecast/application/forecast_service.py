"""Forecast persistence service — DB + CSV backup + retrieval."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from iex_forecast.domain.constants import (
    ZONE_GREEN_MAX,
    ZONE_YELLOW_MAX,
    classify_zone,
)
from iex_forecast.domain.entities import ForecastAccuracy, ForecastPoint
from iex_forecast.domain.interfaces import ForecastRepository

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "horizon",
    "forecast_timestamp",
    "predicted_mcp",
    "zone",
    "confidence",
    "spike_probability",
    "lower_bound",
    "upper_bound",
]


class ForecastService:
    """
    High-level service that orchestrates forecast persistence.

    Responsibilities:
        1. Save forecast points to PostgreSQL via ForecastRepository
        2. Write CSV backup to disk
        3. Retrieve forecasts by various criteria
        4. Evaluate forecast accuracy when actuals arrive
    """

    def __init__(
        self,
        repository: ForecastRepository,
        csv_backup_dir: Path | str = "forecasts",
    ) -> None:
        self.repository = repository
        self.csv_backup_dir = Path(csv_backup_dir)
        self.csv_backup_dir.mkdir(parents=True, exist_ok=True)

    # ── Persist ───────────────────────────────────────────────────────

    def save_forecast_run(
        self,
        origin_timestamp: datetime,
        points: list[ForecastPoint],
        model_version: str,
        feature_version: str = "v1",
    ) -> dict[str, Any]:
        """
        Persist a forecast run to both PostgreSQL and a CSV backup file.

        Returns a summary dict with run_id, csv_path, and block counts.
        """
        # 1. CSV backup
        csv_path = self._write_csv_backup(origin_timestamp, points)
        logger.info(
            "CSV backup written: %s (%d blocks)", csv_path, len(points)
        )

        # Compute forecast_type
        total_blocks = len(points)
        if total_blocks <= 96:
            forecast_type = "24-Hour"
        elif total_blocks <= 672:
            forecast_type = "7-Day"
        else:
            forecast_type = "30-Day"

        # 2. Database
        run_id = self.repository.save_forecast(
            origin_timestamp=origin_timestamp,
            points=points,
            model_version=model_version,
            csv_path=str(csv_path),
            feature_version=feature_version,
            forecast_type=forecast_type,
        )
        logger.info(
            "Forecast run persisted: run_id=%s blocks=%d type=%s", run_id, len(points), forecast_type
        )

        # 3. Zone summary
        zone_counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
        for p in points:
            zone_counts[p.zone] = zone_counts.get(p.zone, 0) + 1

        return {
            "run_id": run_id,
            "generated_at": origin_timestamp.isoformat(),
            "model_version": model_version,
            "feature_version": feature_version,
            "forecast_type": forecast_type,
            "total_blocks": len(points),
            "csv_path": str(csv_path),
            "zone_summary": zone_counts,
        }

    def _write_csv_backup(
        self,
        origin_timestamp: datetime,
        points: list[ForecastPoint],
    ) -> Path:
        """Write a CSV backup file with a timestamped filename."""
        ts_str = origin_timestamp.strftime("%Y%m%d_%H%M%S")
        csv_path = self.csv_backup_dir / f"forecast_{ts_str}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for p in points:
                writer.writerow(
                    {
                        "horizon": p.horizon,
                        "forecast_timestamp": p.forecast_timestamp.isoformat(),
                        "predicted_mcp": round(p.mcp_forecast_rs_mwh, 2),
                        "zone": p.zone,
                        "confidence": round(p.confidence, 4),
                        "spike_probability": round(p.spike_probability, 4),
                        "lower_bound": round(p.lower_bound, 2) if p.lower_bound is not None else "",
                        "upper_bound": round(p.upper_bound, 2) if p.upper_bound is not None else "",
                    }
                )
        return csv_path

    # ── Retrieve ──────────────────────────────────────────────────────

    def get_latest_forecast(self, forecast_type: str | None = None) -> pd.DataFrame:
        """Return the most recent forecast run as a DataFrame."""
        return self.repository.get_latest_forecast(forecast_type)

    def get_forecast_by_run(self, run_id: str) -> pd.DataFrame:
        """Return all forecast points for a specific run_id."""
        return self.repository.get_forecast_by_run(run_id)

    def get_forecasts_by_date_range(
        self, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Return forecast points whose timestamps fall in [start, end]."""
        return self.repository.get_forecasts_by_date_range(start, end)

    def get_forecasts_by_zone(
        self, zone: str, run_id: str | None = None
    ) -> pd.DataFrame:
        """Return forecast points filtered by zone (GREEN/YELLOW/RED)."""
        return self.repository.get_forecasts_by_zone(zone, run_id)

    def list_runs(self, limit: int = 20) -> pd.DataFrame:
        """Return metadata for recent forecast runs."""
        return self.repository.list_runs(limit)

    # ── Accuracy Evaluation ───────────────────────────────────────────

    def evaluate_accuracy(
        self,
        run_id: str,
        actuals: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Compare a forecast run against actual MCP values and persist results.

        Parameters
        ----------
        run_id : str
            The forecast run to evaluate.
        actuals : pd.DataFrame
            Must contain columns: ``forecast_timestamp`` and ``actual_mcp``.

        Returns
        -------
        dict with summary metrics (MAE, RMSE, zone_accuracy, records_evaluated).
        """
        forecast_df = self.repository.get_forecast_by_run(run_id)
        if forecast_df.empty:
            raise ValueError(f"No forecast found for run_id={run_id}")

        # Merge on timestamp
        actuals = actuals.copy()
        actuals["forecast_timestamp"] = pd.to_datetime(actuals["forecast_timestamp"])
        forecast_df["forecast_timestamp"] = pd.to_datetime(forecast_df["forecast_timestamp"])

        merged = forecast_df.merge(
            actuals[["forecast_timestamp", "actual_mcp"]],
            on="forecast_timestamp",
            how="inner",
        )
        if merged.empty:
            raise ValueError("No matching timestamps between forecast and actuals")

        records: list[ForecastAccuracy] = []
        for _, row in merged.iterrows():
            predicted = float(row["predicted_mcp"])
            actual = float(row["actual_mcp"])
            abs_err = abs(predicted - actual)
            denom = max(abs(actual), 1.0)
            pct_err = (abs_err / denom) * 100.0

            records.append(
                ForecastAccuracy(
                    run_id=run_id,
                    horizon=int(row["horizon"]),
                    forecast_timestamp=row["forecast_timestamp"].to_pydatetime(),
                    predicted_mcp=predicted,
                    actual_mcp=actual,
                    predicted_zone=row.get("zone", classify_zone(predicted)),
                    actual_zone=classify_zone(actual),
                    absolute_error=abs_err,
                    percentage_error=pct_err,
                )
            )

        saved = self.repository.save_accuracy(records)

        # Summary metrics
        errors = [r.absolute_error for r in records]
        mae = sum(errors) / len(errors)
        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        zone_correct = sum(
            1 for r in records if r.predicted_zone == r.actual_zone
        )
        zone_accuracy = zone_correct / len(records) * 100.0

        logger.info(
            "Accuracy evaluated: run_id=%s blocks=%d MAE=%.2f zone_acc=%.1f%%",
            run_id,
            saved,
            mae,
            zone_accuracy,
        )

        return {
            "run_id": run_id,
            "records_evaluated": saved,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "zone_accuracy_pct": round(zone_accuracy, 2),
            "zone_breakdown": {
                "correct": zone_correct,
                "total": len(records),
            },
        }

    def get_accuracy(self, run_id: str) -> pd.DataFrame:
        """Return accuracy records for a run."""
        return self.repository.get_accuracy_by_run(run_id)

    # ── Utility: Build ForecastPoints from raw predictions ────────────

    @staticmethod
    def build_forecast_points(
        timestamps: list[datetime],
        predictions: list[float],
        lower_bounds: list[float] | None = None,
        upper_bounds: list[float] | None = None,
        spike_probabilities: list[float] | None = None,
    ) -> list[ForecastPoint]:
        """
        Factory to build ForecastPoint entities from raw model output,
        automatically computing zone and confidence.
        """
        n = len(predictions)
        lowers = lower_bounds or [0.0] * n
        uppers = upper_bounds or [0.0] * n
        spikes = spike_probabilities or [0.0] * n

        points: list[ForecastPoint] = []
        for i in range(n):
            mcp = predictions[i]
            lb = lowers[i]
            ub = uppers[i]
            ci_width = ub - lb if (ub and lb) else 0.0
            # Confidence: inverse of relative CI width (narrower band → higher confidence)
            confidence = max(0.0, min(1.0, 1.0 - ci_width / max(abs(mcp), 1.0)))

            points.append(
                ForecastPoint(
                    horizon=i + 1,
                    forecast_timestamp=timestamps[i],
                    mcp_forecast_rs_mwh=round(mcp, 2),
                    zone=classify_zone(mcp),
                    confidence=round(confidence, 4),
                    spike_probability=round(spikes[i], 4),
                    lower_bound=round(lb, 2) if lb else None,
                    upper_bound=round(ub, 2) if ub else None,
                )
            )
        return points
