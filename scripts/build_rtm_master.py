#!/usr/bin/env python3
"""Build RTM master dataset and data quality report from raw Excel files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.ingestion import RTMMasterIngestionPipeline, configure_logging


def main() -> int:
    configure_logging()
    result = RTMMasterIngestionPipeline(raw_dir=ROOT / "data" / "raw", rtm_market_only=True).run()
    report_text = result.quality_report.to_text()
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nParquet: {result.parquet_path}")
    print(f"CSV:     {result.csv_path}")
    print(f"Report:  {result.report_json_path}")
    return 0 if result.quality_report.status != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
