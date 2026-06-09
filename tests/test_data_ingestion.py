"""Tests for src/data ingestion and validation."""

from pathlib import Path

import pandas as pd
import pytest

from data.ingestion import IngestionError, RTMMasterIngestionPipeline
from data.validation import DataQualityValidator


@pytest.fixture
def raw_excel_dir(tmp_path: Path) -> Path:
    rows = []
    for day in (1, 2):
        for hour in range(1, 25):
            for time_block in range(1, 5):
                rows.append(
                    {
                        "Date": f"2024-01-{day:02d}",
                        "Hour": hour,
                        "Session ID": 1,
                        "Time Block": time_block,
                        "Purchase Bid (MW)": 10000.0,
                        "Sell Bid (MW)": 9500.0,
                        "MCV (MW)": 5000.0,
                        "Scheduled Volume (MW)": 4800.0,
                        "MCP (Rs/MWh)": 3200.0 + hour,
                    }
                )
    df = pd.DataFrame(rows)
    path = tmp_path / "jan_2024.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return tmp_path


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    return tmp_path / "processed"


def test_pipeline_builds_master_files(raw_excel_dir: Path, processed_dir: Path):
    pipeline = RTMMasterIngestionPipeline(
        raw_dir=raw_excel_dir,
        processed_dir=processed_dir,
    )
    result = pipeline.run()

    assert result.parquet_path.exists()
    assert result.csv_path.exists()
    assert result.report_json_path.exists()
    assert result.report_txt_path.exists()
    assert len(result.dataframe) == 192
    assert result.dataframe["block_timestamp"].is_monotonic_increasing
    assert result.quality_report.status in {"pass", "warn"}


def test_duplicate_removal(raw_excel_dir: Path, processed_dir: Path):
    df = pd.read_excel(raw_excel_dir / "jan_2024.xlsx", engine="openpyxl")
    dup_path = raw_excel_dir / "jan_2024_copy.xlsx"
    df.to_excel(dup_path, index=False, engine="openpyxl")

    pipeline = RTMMasterIngestionPipeline(
        raw_dir=raw_excel_dir,
        processed_dir=processed_dir,
    )
    result = pipeline.run()
    assert result.quality_report.duplicate_rows_removed > 0
    assert result.quality_report.duplicate_rows_remaining == 0


def test_validator_detects_missing(processed_dir: Path):
    df = pd.DataFrame(
        {
            "trade_date": ["2024-01-01", "2024-01-01"],
            "hour": [1, 2],
            "session_id": [1, 1],
            "time_block": [1, 2],
            "mcp_rs_mwh": [3000.0, None],
            "block_timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:15"], utc=True
            ),
            "source_file": ["test.xlsx", "test.xlsx"],
        }
    )
    report = DataQualityValidator().validate(df)
    mcp_missing = next(s for s in report.missing_by_column if s.column == "mcp_rs_mwh")
    assert mcp_missing.missing_count == 1
    assert report.rows_missing_required_model == 1


def test_empty_raw_dir_raises(tmp_path: Path):
    empty = tmp_path / "raw"
    empty.mkdir()
    pipeline = RTMMasterIngestionPipeline(raw_dir=empty, processed_dir=tmp_path / "out")
    with pytest.raises(IngestionError):
        pipeline.run()
