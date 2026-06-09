"""Data cleaning tests."""

import pandas as pd
import pytest

from iex_forecast.core.exceptions import DataValidationError
from iex_forecast.data.cleaner import RTMDataCleaner
from iex_forecast.domain.constants import BLOCKS_PER_DAY


def test_clean_produces_block_timestamp(sample_rtm_df):
    assert "block_timestamp" in sample_rtm_df.columns
    assert sample_rtm_df["block_timestamp"].dt.tz is not None


def test_clean_full_day_block_count():
    rows = []
    for hour in range(1, 25):
        for time_block in range(1, 5):
            rows.append(
                {
                    "trade_date": "2024-06-15",
                    "hour": hour,
                    "session_id": 1,
                    "time_block": time_block,
                    "purchase_bid_mw": 1000.0,
                    "sell_bid_mw": 900.0,
                    "mcv_mw": 500.0,
                    "scheduled_volume_mw": 450.0,
                    "mcp_rs_mwh": 3000.0,
                }
            )
    df = RTMDataCleaner().clean(pd.DataFrame(rows))
    assert len(df) == BLOCKS_PER_DAY


def test_clean_rejects_empty():
    with pytest.raises(DataValidationError):
        RTMDataCleaner().clean(pd.DataFrame())
