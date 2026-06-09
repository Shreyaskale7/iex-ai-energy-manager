"""Production-grade RTM master dataset ingestion pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.schema import (
    BLOCK_MINUTES,
    BLOCKS_PER_DAY,
    COLUMN_ALIASES,
    COLUMN_RENAME_MAP,
    DEDUP_KEYS,
    NUMERIC_COLUMNS,
    RAW_COLUMNS,
    REQUIRED_FOR_MODEL,
    RTM_MARKET_GLOB_PATTERNS,
    STANDARD_COLUMNS,
    TIMEZONE,
)
from data.validation import DataQualityReport, DataQualityValidator

logger = logging.getLogger(__name__)

MONTH_FILE_PATTERN = re.compile(
    r"(?P<month>january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    r"[\s_\-]*(?P<year>20\d{2})",
    re.IGNORECASE,
)

MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


class IngestionError(Exception):
    """Raised when ingestion cannot complete."""


@dataclass
class IngestionResult:
    dataframe: pd.DataFrame
    quality_report: DataQualityReport
    parquet_path: Path
    csv_path: Path
    report_json_path: Path
    report_txt_path: Path


class RTMMasterIngestionPipeline:
    """
    Ingest IEX RTM Market Snapshot Excel files into a single chronologically sorted master table.

    Supports files named e.g. ``RTM_Market Snapshot jan 2024.xlsx`` in the project root.
    """

    def __init__(
        self,
        raw_dir: Path | str = ".",
        processed_dir: Path | str = "data/processed",
        rtm_market_only: bool = True,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.rtm_market_only = rtm_market_only
        self.parquet_path = self.processed_dir / "rtm_master.parquet"
        self.csv_path = self.processed_dir / "rtm_master.csv"

    def run(self) -> IngestionResult:
        logger.info("Starting RTM master ingestion from %s", self.raw_dir.resolve())
        file_stats: list[dict[str, Any]] = []
        frames: list[pd.DataFrame] = []

        skipped: list[str] = []
        for path in self._discover_files():
            try:
                df, stats = self._read_and_standardize(path)
            except PermissionError as exc:
                skipped.append(path.name)
                logger.error(
                    "Skipping %s (file locked or permission denied). Close Excel and re-run ingestion: %s",
                    path.name,
                    exc,
                )
                continue
            frames.append(df)
            file_stats.append(stats)

        if skipped:
            logger.warning(
                "Skipped %d file(s) due to read errors: %s",
                len(skipped),
                ", ".join(skipped),
            )

        if not frames:
            raise IngestionError(f"No Excel files could be read in {self.raw_dir}")

        merged = pd.concat(frames, ignore_index=True)
        logger.info("Merged %d files into %d rows", len(frames), len(merged))

        merged = self._normalize_time_blocks(merged)
        merged = self._coerce_types(merged)
        merged = self._build_timestamps(merged)
        merged = self._sort_chronologically(merged)
        merged, duplicates_removed = self._remove_duplicates(merged)

        validator = DataQualityValidator(
            duplicate_rows_removed=duplicates_removed,
            files_ingested=file_stats,
        )
        report = validator.validate(merged)

        export_df = self._finalize_for_export(merged)
        self._save_outputs(export_df)
        json_path, txt_path = report.save(self.processed_dir)

        logger.info(
            "Ingestion complete: %d rows saved | quality status=%s",
            len(export_df),
            report.status,
        )

        return IngestionResult(
            dataframe=export_df,
            quality_report=report,
            parquet_path=self.parquet_path,
            csv_path=self.csv_path,
            report_json_path=json_path,
            report_txt_path=txt_path,
        )

    def _discover_files(self) -> list[Path]:
        if not self.raw_dir.exists():
            raise IngestionError(f"Raw data directory does not exist: {self.raw_dir}")

        patterns = ("RTM_Market*.xlsx", "RTM_Market*.xls") if self.rtm_market_only else RTM_MARKET_GLOB_PATTERNS
        files: list[Path] = []
        for pattern in patterns:
            files.extend(self.raw_dir.glob(pattern))

        files = sorted(set(files), key=self._file_sort_key)
        logger.info("Discovered %d Excel file(s)", len(files))
        return files

    def _read_and_standardize(self, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        logger.info("Reading %s", path.name)
        raw = self._read_excel_payload(path)
        raw = self._apply_column_aliases(raw)
        self._assert_required_columns(raw, path)

        df = raw.rename(columns=COLUMN_RENAME_MAP)
        df["source_file"] = path.name

        extra_cols = [c for c in df.columns if c not in STANDARD_COLUMNS and c != "source_file"]
        if extra_cols:
            logger.warning("Dropping unexpected columns from %s: %s", path.name, extra_cols)
            df = df.drop(columns=extra_cols, errors="ignore")

        stats = {
            "file": path.name,
            "rows_read": len(df),
            "columns": list(df.columns),
            "date_min": self._safe_series_min(df.get("trade_date")),
            "date_max": self._safe_series_max(df.get("trade_date")),
        }
        return df, stats

    @staticmethod
    def _read_excel_payload(path: Path) -> pd.DataFrame:
        """Read standard or RTM Market Snapshot layout (header row after metadata)."""
        preview = pd.read_excel(path, header=None, nrows=15, engine="openpyxl")
        header_row = 0
        for idx in range(len(preview)):
            row_vals = {str(v).strip() for v in preview.iloc[idx].tolist() if pd.notna(v)}
            if {"Date", "Hour", "Session ID"}.issubset(row_vals) or "Date" in row_vals and "MCP" in "".join(row_vals):
                header_row = idx
                break

        df = pd.read_excel(path, header=header_row, engine="openpyxl")
        df = df.dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for col in df.columns:
            if col in COLUMN_ALIASES:
                rename[col] = COLUMN_ALIASES[col]
            for alias, canonical in COLUMN_ALIASES.items():
                if alias.lower() in str(col).lower():
                    rename[col] = canonical
                    break
        if rename:
            df = df.rename(columns=rename)
        for canonical in COLUMN_ALIASES.values():
            matches = [c for c in df.columns if canonical.lower() in str(c).lower()]
            if canonical not in df.columns and matches:
                df = df.rename(columns={matches[0]: canonical})
        return df

    def _assert_required_columns(self, df: pd.DataFrame, path: Path) -> None:
        missing = [col for col in RAW_COLUMNS if col not in df.columns]
        if missing:
            raise IngestionError(
                f"{path.name} is missing required columns: {missing}. Found: {list(df.columns)}"
            )

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"], dayfirst=True, errors="coerce").dt.date

        for column in NUMERIC_COLUMNS:
            if column in out.columns:
                out[column] = pd.to_numeric(
                    out[column].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )

        return out

    @staticmethod
    def _normalize_time_blocks(df: pd.DataFrame) -> pd.DataFrame:
        """Convert interval labels like 00:00-00:15 into numeric 1-4 block index."""
        out = df.copy()

        def to_block_index(value: Any) -> float:
            if pd.isna(value):
                return np.nan
            if isinstance(value, (int, float)) and 1 <= value <= 4:
                return float(int(value))
            text = str(value).strip()
            if "-" in text and ":" in text:
                start = text.split("-")[0].strip()
                hour_str, minute_str = start.split(":")
                minutes = int(hour_str) * 60 + int(minute_str)
                return float((minutes % 60) // BLOCK_MINUTES + 1)
            if text.isdigit():
                return float(int(text))
            return np.nan

        out["time_block"] = out["time_block"].apply(to_block_index)
        return out

    def _build_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["daily_block_index"] = self._resolve_daily_block_index(out)

        base = pd.to_datetime(out["trade_date"].astype(str), errors="coerce")
        minutes = (out["daily_block_index"] - 1) * BLOCK_MINUTES
        out["block_timestamp"] = base + pd.to_timedelta(minutes, unit="m")
        out["block_timestamp"] = out["block_timestamp"].dt.tz_localize(TIMEZONE)
        return out

    @staticmethod
    def _resolve_daily_block_index(df: pd.DataFrame) -> pd.Series:
        time_block = df["time_block"]
        hour = df.get("hour")

        if hour is not None and hour.notna().sum() > len(df) * 0.5:
            hour_valid = hour.between(1, 24)
            block_valid = time_block.between(1, 4)
            index = (hour - 1) * 4 + time_block
            index = index.where(hour_valid & block_valid)
        else:
            index = time_block.where(time_block.between(1, BLOCKS_PER_DAY))

        fallback = time_block.where(time_block.between(1, BLOCKS_PER_DAY))
        return index.fillna(fallback)

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(
            ["block_timestamp", "session_id"],
            ascending=True,
            na_position="last",
        ).reset_index(drop=True)

    def _remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        before = len(df)
        keys = [k for k in DEDUP_KEYS if k in df.columns]
        deduped = df.drop_duplicates(subset=keys, keep="last")
        removed = before - len(deduped)
        if removed:
            logger.info("Removed %d duplicate rows (kept latest per key)", removed)
        return deduped.reset_index(drop=True), removed

    def _finalize_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        for column in REQUIRED_FOR_MODEL:
            col_mask = df[column].notna()
            if df[column].dtype == object:
                col_mask = col_mask & (df[column].astype(str).str.strip() != "")
            mask &= col_mask

        mask &= df["block_timestamp"].notna()
        mask &= df["time_block"].between(1, 4)
        dropped = (~mask).sum()
        if dropped:
            logger.warning("Excluding %d rows with missing/invalid fields from master export", dropped)

        out = df.loc[mask].copy()
        column_order = [
            "block_timestamp",
            "trade_date",
            "daily_block_index",
            "hour",
            "session_id",
            "time_block",
            "purchase_bid_mw",
            "sell_bid_mw",
            "mcv_mw",
            "scheduled_volume_mw",
            "mcp_rs_mwh",
            "source_file",
        ]
        existing = [c for c in column_order if c in out.columns]
        return out[existing]

    def _save_outputs(self, df: pd.DataFrame) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.parquet_path, index=False)
        df.to_csv(self.csv_path, index=False)
        logger.info("Saved parquet: %s", self.parquet_path)
        logger.info("Saved csv: %s", self.csv_path)

    @staticmethod
    def _file_sort_key(path: Path) -> tuple[int, int, str]:
        match = MONTH_FILE_PATTERN.search(path.stem.lower())
        if not match:
            return (9999, 12, path.stem)
        month_key = match.group("month").lower()[:4]
        for key, num in MONTH_MAP.items():
            if month_key.startswith(key[:3]):
                return (int(match.group("year")), num, path.stem)
        return (int(match.group("year")), 12, path.stem)

    @staticmethod
    def _safe_series_min(series: pd.Series | None) -> str | None:
        if series is None:
            return None
        converted = pd.to_datetime(series, errors="coerce").dropna()
        return None if converted.empty else converted.min().date().isoformat()

    @staticmethod
    def _safe_series_max(series: pd.Series | None) -> str | None:
        if series is None:
            return None
        converted = pd.to_datetime(series, errors="coerce").dropna()
        return None if converted.empty else converted.max().date().isoformat()


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> int:
    configure_logging()
    result = RTMMasterIngestionPipeline(raw_dir=".", rtm_market_only=True).run()
    print(result.quality_report.to_text())
    return 0 if result.quality_report.status != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
