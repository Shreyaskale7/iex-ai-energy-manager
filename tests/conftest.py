"""Pytest fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_rtm_df() -> pd.DataFrame:
    rows = []
    mcp = 3500.0
    for day in range(1, 8):
        base_date = f"2024-01-{day:02d}"
        for hour in range(1, 25):
            for time_block in range(1, 5):
                rows.append(
                    {
                        "trade_date": base_date,
                        "hour": hour,
                        "session_id": 1,
                        "time_block": time_block,
                        "purchase_bid_mw": 12000.0,
                        "sell_bid_mw": 11500.0,
                        "mcv_mw": 8000.0,
                        "scheduled_volume_mw": 7500.0,
                        "mcp_rs_mwh": mcp,
                        "source_file": "jan_2024.xlsx",
                    }
                )
                mcp += 5.0
    df = pd.DataFrame(rows)
    from iex_forecast.data.cleaner import RTMDataCleaner

    return RTMDataCleaner().clean(df)
