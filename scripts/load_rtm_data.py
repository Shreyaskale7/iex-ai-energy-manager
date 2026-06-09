#!/usr/bin/env python3
"""Load processed RTM market data into PostgreSQL for local API validation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from iex_forecast.infrastructure.repositories import PostgresRTMRepository


def main() -> int:
    path = ROOT / "data" / "processed" / "rtm_master.parquet"
    df = pd.read_parquet(path)
    cols = [
        "trade_date",
        "hour",
        "session_id",
        "time_block",
        "purchase_bid_mw",
        "sell_bid_mw",
        "mcv_mw",
        "scheduled_volume_mw",
        "mcp_rs_mwh",
        "block_timestamp",
        "source_file",
    ]
    df2 = df[cols].copy()
    print(f"Loading {len(df2)} RTM blocks into PostgreSQL...")
    count = PostgresRTMRepository().upsert_blocks(df2)
    print(f"Upserted {count} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
