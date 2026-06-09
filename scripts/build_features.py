#!/usr/bin/env python3
"""Build RTM MCP forecasting feature matrix from rtm_master.parquet."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.build_features import RTMFeaturePipeline, configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTM MCP feature engineering pipeline")
    parser.add_argument(
        "--master",
        type=Path,
        default=ROOT / "data" / "processed" / "rtm_master.parquet",
        help="Input master parquet path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "features" / "features.parquet",
        help="Output features parquet path",
    )
    parser.add_argument(
        "--keep-warmup",
        action="store_true",
        help="Retain rows with NaN lags/rolling (not recommended for training)",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    result = RTMFeaturePipeline(
        master_path=args.master,
        output_path=args.output,
        drop_warmup_rows=not args.keep_warmup,
    ).run()
    print(f"Input:  {result.input_path} ({result.rows_in:,} rows)")
    print(f"Output: {result.output_path} ({result.rows_out:,} rows)")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
