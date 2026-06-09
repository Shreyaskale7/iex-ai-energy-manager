"""Data quality validation and reporting for RTM master datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.schema import (
    BLOCKS_PER_DAY,
    DEDUP_KEYS,
    NUMERIC_COLUMNS,
    REQUIRED_FOR_MODEL,
    REQUIRED_FOR_TIMESTAMP,
)

REPORT_VERSION = "1.0.0"


@dataclass
class ColumnMissingStats:
    column: str
    missing_count: int
    missing_pct: float
    dtype: str


@dataclass
class DataQualityReport:
    """Structured output from RTM master validation."""

    report_version: str
    generated_at_utc: str
    row_count: int
    column_count: int
    source_files: list[str]
    date_range_start: str | None
    date_range_end: str | None
    duplicate_rows_removed: int
    duplicate_rows_remaining: int
    missing_by_column: list[ColumnMissingStats]
    rows_missing_required_timestamp: int
    rows_missing_required_model: int
    incomplete_days: dict[str, int]
    incomplete_day_count: int
    complete_days: int
    mcp_negative_count: int
    mcp_zero_count: int
    mcp_stats: dict[str, float]
    files_ingested: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    status: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_by_column"] = [asdict(m) for m in self.missing_by_column]
        return payload

    def to_text(self) -> str:
        lines = [
            "=" * 72,
            "IEX RTM DATA QUALITY REPORT",
            "=" * 72,
            f"Generated (UTC): {self.generated_at_utc}",
            f"Report version:  {self.report_version}",
            f"Status:          {self.status.upper()}",
            "",
            "DATASET SUMMARY",
            "-" * 72,
            f"Rows:            {self.row_count:,}",
            f"Columns:         {self.column_count}",
            f"Date range:      {self.date_range_start} -> {self.date_range_end}",
            f"Source files:    {len(self.source_files)}",
            "",
            "DEDUPLICATION",
            "-" * 72,
            f"Duplicates removed (pre-save): {self.duplicate_rows_removed:,}",
            f"Duplicate keys remaining:      {self.duplicate_rows_remaining:,}",
            "",
            "MISSING VALUES BY COLUMN",
            "-" * 72,
        ]
        for stat in self.missing_by_column:
            lines.append(
                f"  {stat.column:<28} {stat.missing_count:>8,}  ({stat.missing_pct:>6.2f}%)"
            )
        lines.extend(
            [
                "",
                "REQUIRED FIELD GAPS",
                "-" * 72,
                f"Rows missing timestamp fields: {self.rows_missing_required_timestamp:,}",
                f"Rows missing model fields:   {self.rows_missing_required_model:,}",
                "",
                "DAILY BLOCK COVERAGE (expect 96 blocks/day)",
                "-" * 72,
                f"Complete days (96 blocks):   {self.complete_days:,}",
                f"Incomplete days:             {self.incomplete_day_count:,}",
            ]
        )
        if self.incomplete_days:
            sample = list(self.incomplete_days.items())[:10]
            for day, count in sample:
                lines.append(f"  {day}: {count} blocks")
            if len(self.incomplete_days) > 10:
                lines.append(f"  ... and {len(self.incomplete_days) - 10} more days")

        lines.extend(
            [
                "",
                "MCP (Rs/MWh) CHECKS",
                "-" * 72,
                f"Negative MCP values: {self.mcp_negative_count:,}",
                f"Zero MCP values:     {self.mcp_zero_count:,}",
                f"Min:                 {self.mcp_stats.get('min', float('nan')):.2f}",
                f"Max:                 {self.mcp_stats.get('max', float('nan')):.2f}",
                f"Mean:                {self.mcp_stats.get('mean', float('nan')):.2f}",
                f"Median:              {self.mcp_stats.get('median', float('nan')):.2f}",
                f"Std:                 {self.mcp_stats.get('std', float('nan')):.2f}",
            ]
        )
        if self.warnings:
            lines.extend(["", "WARNINGS", "-" * 72])
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        lines.append("=" * 72)
        return "\n".join(lines)

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "data_quality_report.json"
        txt_path = output_dir / "data_quality_report.txt"
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        txt_path.write_text(self.to_text(), encoding="utf-8")
        return json_path, txt_path


class DataQualityValidator:
    """Runs validation checks and builds a DataQualityReport."""

    def __init__(
        self,
        duplicate_rows_removed: int = 0,
        files_ingested: list[dict[str, Any]] | None = None,
    ) -> None:
        self.duplicate_rows_removed = duplicate_rows_removed
        self.files_ingested = files_ingested or []

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        if df.empty:
            return DataQualityReport(
                report_version=REPORT_VERSION,
                generated_at_utc=datetime.now(timezone.utc).isoformat(),
                row_count=0,
                column_count=0,
                source_files=[],
                date_range_start=None,
                date_range_end=None,
                duplicate_rows_removed=self.duplicate_rows_removed,
                duplicate_rows_remaining=0,
                missing_by_column=[],
                rows_missing_required_timestamp=0,
                rows_missing_required_model=0,
                incomplete_days={},
                incomplete_day_count=0,
                complete_days=0,
                mcp_negative_count=0,
                mcp_zero_count=0,
                mcp_stats={},
                files_ingested=self.files_ingested,
                warnings=["Dataset is empty after ingestion."],
                status="fail",
            )

        warnings: list[str] = []
        missing_stats = self._missing_by_column(df)
        dup_remaining = self._count_duplicate_keys(df)
        if dup_remaining > 0:
            warnings.append(f"{dup_remaining} duplicate keys still present after deduplication.")

        ts_missing = self._rows_missing_required(df, REQUIRED_FOR_TIMESTAMP)
        model_missing = self._rows_missing_required(df, REQUIRED_FOR_MODEL)
        if ts_missing > 0:
            warnings.append(f"{ts_missing} rows cannot form a valid block timestamp.")
        if model_missing > 0:
            warnings.append(f"{model_missing} rows missing MCP or core identifiers.")

        incomplete_days, complete_days = self._daily_block_coverage(df)
        if incomplete_days:
            warnings.append(
                f"{len(incomplete_days)} trading days do not contain exactly {BLOCKS_PER_DAY} blocks."
            )

        mcp_negative, mcp_zero, mcp_stats = self._mcp_checks(df)
        if mcp_negative > 0:
            warnings.append(f"{mcp_negative} rows have negative MCP values.")

        date_start, date_end = self._date_range(df)
        source_files = sorted(df["source_file"].dropna().unique().tolist()) if "source_file" in df.columns else []

        status = "pass"
        if ts_missing > 0 or model_missing > 0 or dup_remaining > 0:
            status = "warn"
        if df.empty or (model_missing == len(df)):
            status = "fail"

        return DataQualityReport(
            report_version=REPORT_VERSION,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            row_count=len(df),
            column_count=len(df.columns),
            source_files=source_files,
            date_range_start=date_start,
            date_range_end=date_end,
            duplicate_rows_removed=self.duplicate_rows_removed,
            duplicate_rows_remaining=dup_remaining,
            missing_by_column=missing_stats,
            rows_missing_required_timestamp=ts_missing,
            rows_missing_required_model=model_missing,
            incomplete_days=incomplete_days,
            incomplete_day_count=len(incomplete_days),
            complete_days=complete_days,
            mcp_negative_count=mcp_negative,
            mcp_zero_count=mcp_zero,
            mcp_stats=mcp_stats,
            files_ingested=self.files_ingested,
            warnings=warnings,
            status=status,
        )

    def _missing_by_column(self, df: pd.DataFrame) -> list[ColumnMissingStats]:
        stats: list[ColumnMissingStats] = []
        total = len(df)
        for column in df.columns:
            missing = int(df[column].isna().sum())
            if missing == 0 and df[column].dtype == object:
                missing = int((df[column].astype(str).str.strip() == "").sum())
            stats.append(
                ColumnMissingStats(
                    column=column,
                    missing_count=missing,
                    missing_pct=round((missing / total) * 100, 4) if total else 0.0,
                    dtype=str(df[column].dtype),
                )
            )
        return sorted(stats, key=lambda s: s.missing_count, reverse=True)

    def _rows_missing_required(self, df: pd.DataFrame, required: list[str]) -> int:
        mask = pd.Series(False, index=df.index)
        for column in required:
            if column not in df.columns:
                return len(df)
            col_missing = df[column].isna()
            if df[column].dtype == object:
                col_missing = col_missing | (df[column].astype(str).str.strip() == "")
            mask = mask | col_missing
        return int(mask.sum())

    def _count_duplicate_keys(self, df: pd.DataFrame) -> int:
        keys = [k for k in DEDUP_KEYS if k in df.columns]
        if not keys:
            return 0
        return int(df.duplicated(subset=keys, keep=False).sum())

    def _daily_block_coverage(self, df: pd.DataFrame) -> tuple[dict[str, int], int]:
        if "trade_date" not in df.columns or "block_timestamp" not in df.columns:
            return {}, 0
        counts = df.groupby("trade_date", dropna=True).size()
        incomplete = {
            str(day): int(count)
            for day, count in counts.items()
            if 0 < count != BLOCKS_PER_DAY
        }
        complete = int((counts == BLOCKS_PER_DAY).sum())
        return incomplete, complete

    def _mcp_checks(self, df: pd.DataFrame) -> tuple[int, int, dict[str, float]]:
        if "mcp_rs_mwh" not in df.columns:
            return 0, 0, {}
        series = pd.to_numeric(df["mcp_rs_mwh"], errors="coerce").dropna()
        if series.empty:
            return 0, 0, {}
        return (
            int((series < 0).sum()),
            int((series == 0).sum()),
            {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "p01": float(np.percentile(series, 1)),
                "p99": float(np.percentile(series, 99)),
            },
        )

    @staticmethod
    def _date_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
        if "trade_date" not in df.columns:
            return None, None
        dates = pd.to_datetime(df["trade_date"], errors="coerce").dropna()
        if dates.empty:
            return None, None
        return dates.min().date().isoformat(), dates.max().date().isoformat()
