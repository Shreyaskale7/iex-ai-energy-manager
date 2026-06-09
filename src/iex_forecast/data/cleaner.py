"""Clean and validate RTM block records."""

from pathlib import Path

import pandas as pd

from iex_forecast.core.exceptions import DataValidationError
from iex_forecast.core.logging import get_logger
from iex_forecast.domain.constants import BLOCKS_PER_DAY

logger = get_logger(__name__)


class RTMDataCleaner:
    """Transforms raw ingested frames into analysis-ready block series."""

    NUMERIC_COLS = [
        "hour",
        "session_id",
        "time_block",
        "purchase_bid_mw",
        "sell_bid_mw",
        "mcv_mw",
        "scheduled_volume_mw",
        "mcp_rs_mwh",
    ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise DataValidationError("Cannot clean empty dataframe")

        out = df.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
        for col in self.NUMERIC_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=["trade_date", "hour", "time_block", "mcp_rs_mwh"])
        out = self._enforce_ranges(out)
        out = self._build_block_timestamps(out)
        out = out.sort_values("block_timestamp").drop_duplicates(
            subset=["trade_date", "hour", "session_id", "time_block"],
            keep="last",
        )
        out = out.reset_index(drop=True)
        self._validate_block_sequence(out)
        logger.info("cleaning_complete", rows=len(out))
        return out

    def _enforce_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (df["hour"].between(1, 24))
            & (df["time_block"].between(1, 4))
            & (df["mcp_rs_mwh"] >= 0)
            & (df["purchase_bid_mw"] >= 0)
            & (df["sell_bid_mw"] >= 0)
        )
        dropped = (~mask).sum()
        if dropped:
            logger.warning("rows_dropped_range_validation", count=int(dropped))
        return df.loc[mask].copy()

    def _build_block_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(df["trade_date"].astype(str))
        minutes_offset = (df["hour"] - 1) * 60 + (df["time_block"] - 1) * 15
        df["block_timestamp"] = ts + pd.to_timedelta(minutes_offset, unit="m")
        df["block_timestamp"] = df["block_timestamp"].dt.tz_localize("Asia/Kolkata")
        return df

    def _validate_block_sequence(self, df: pd.DataFrame) -> None:
        daily_counts = df.groupby("trade_date").size()
        irregular = daily_counts[(daily_counts > 0) & (daily_counts != BLOCKS_PER_DAY)]
        if len(irregular) > 0:
            sample = irregular.head(5).to_dict()
            logger.warning(
                "incomplete_days_detected",
                sample_days=sample,
                note="Partial days are retained; training uses contiguous windows.",
            )

    def save_processed(self, df: pd.DataFrame, path: Path | str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
        logger.info("processed_parquet_saved", path=str(output), rows=len(df))
