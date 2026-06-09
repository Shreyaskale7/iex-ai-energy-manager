"""Tests for RTM feature engineering pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.build_features import (
    CYCLICAL_FEATURES,
    ENGINEERED_FEATURES,
    LAG_FEATURES,
    RTMFeaturePipeline,
    TARGET_COLUMN,
)


def _synthetic_master(n_days: int = 10) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for day in range(1, n_days + 1):
        for hour in range(1, 25):
            for time_block in range(1, 5):
                ts = pd.Timestamp(f"2024-01-{day:02d}", tz="Asia/Kolkata") + pd.Timedelta(
                    minutes=(hour - 1) * 60 + (time_block - 1) * 15
                )
                rows.append(
                    {
                        "block_timestamp": ts,
                        "trade_date": ts.date(),
                        "daily_block_index": (hour - 1) * 4 + time_block,
                        "hour": hour,
                        "session_id": 1,
                        "time_block": time_block,
                        "purchase_bid_mw": float(rng.uniform(8000, 12000)),
                        "sell_bid_mw": float(rng.uniform(8000, 12000)),
                        "mcv_mw": float(rng.uniform(3000, 7000)),
                        "scheduled_volume_mw": float(rng.uniform(3000, 6500)),
                        "mcp_rs_mwh": float(rng.uniform(2800, 4200)),
                        "source_file": "test.xlsx",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def master_path(tmp_path: Path) -> Path:
    path = tmp_path / "rtm_master.parquet"
    _synthetic_master(10).to_parquet(path, index=False)
    return path


def test_build_all_feature_columns(master_path: Path, tmp_path: Path):
    output = tmp_path / "features.parquet"
    pipeline = RTMFeaturePipeline(
        master_path=master_path,
        output_path=output,
        drop_warmup_rows=True,
    )
    result = pipeline.run()

    assert output.exists()
    df = result.dataframe
    for col in ENGINEERED_FEATURES:
        assert col in df.columns, f"Missing feature: {col}"
    assert TARGET_COLUMN in df.columns
    assert df["mcp_lag_1"].notna().all()
    assert df["mcp_lag_96"].notna().all()
    assert df["rolling_mean_96"].notna().all()
    assert set(CYCLICAL_FEATURES).issubset(df.columns)


def test_lag_values_correct(master_path: Path, tmp_path: Path):
    raw = pd.read_parquet(master_path)
    featured = RTMFeaturePipeline(
        master_path=master_path,
        output_path=tmp_path / "f.parquet",
        drop_warmup_rows=False,
    ).build_features(raw)

    idx = 700
    assert featured.loc[idx, "mcp_lag_1"] == pytest.approx(featured.loc[idx - 1, TARGET_COLUMN])
    assert featured.loc[idx, "mcp_lag_4"] == pytest.approx(featured.loc[idx - 4, TARGET_COLUMN])


def test_warmup_drop_reduces_rows(master_path: Path, tmp_path: Path):
    with_warmup = RTMFeaturePipeline(
        master_path=master_path,
        output_path=tmp_path / "all.parquet",
        drop_warmup_rows=False,
    ).run()
    without_warmup = RTMFeaturePipeline(
        master_path=master_path,
        output_path=tmp_path / "trim.parquet",
        drop_warmup_rows=True,
    ).run()
    assert without_warmup.rows_out < with_warmup.rows_out
    assert without_warmup.rows_out == with_warmup.rows_out - without_warmup.rows_dropped
