"""Ingest monthly IEX RTM Excel files."""

import re
from pathlib import Path

import pandas as pd

from iex_forecast.core.exceptions import DataValidationError
from iex_forecast.core.logging import get_logger
from iex_forecast.domain.constants import COLUMN_MAP, EXPECTED_COLUMNS

logger = get_logger(__name__)

MONTH_FILE_PATTERN = re.compile(
    r"^(?P<month>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[_\-\s]?"
    r"(?P<year>20\d{2})",
    re.IGNORECASE,
)


class RTMExcelIngestor:
    """Reads RTM Excel workbooks and normalizes column names."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir

    def discover_files(self) -> list[Path]:
        files = sorted(
            list(self.raw_dir.glob("*.xlsx")) + list(self.raw_dir.glob("*.xls")),
            key=self._sort_key,
        )
        logger.info("discovered_excel_files", count=len(files), directory=str(self.raw_dir))
        return files

    def read_file(self, path: Path) -> pd.DataFrame:
        logger.info("reading_excel", path=str(path))
        df = pd.read_excel(path, engine="openpyxl")
        self._validate_columns(df, path)
        df = df.rename(columns=COLUMN_MAP)
        df["source_file"] = path.name
        return df

    def read_all(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in self.discover_files():
            try:
                frames.append(self.read_file(path))
            except DataValidationError:
                logger.exception("file_validation_failed", path=str(path))
                raise
        if not frames:
            raise DataValidationError(f"No Excel files found in {self.raw_dir}")
        combined = pd.concat(frames, ignore_index=True)
        logger.info("ingestion_complete", rows=len(combined), files=len(frames))
        return combined

    def _validate_columns(self, df: pd.DataFrame, path: Path) -> None:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"File {path.name} missing columns: {missing}. Found: {list(df.columns)}"
            )

    @staticmethod
    def _sort_key(path: Path) -> tuple[int, int, str]:
        stem = path.stem.lower()
        match = MONTH_FILE_PATTERN.search(stem)
        if not match:
            return (9999, 12, stem)
        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        return (int(match.group("year")), month_map[match.group("month")[:3].lower()], stem)
