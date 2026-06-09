"""Advanced feature engineering pipeline for RTM MCP forecasting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import BLOCKS_PER_DAY, BLOCK_MINUTES, TIMEZONE

logger = logging.getLogger(__name__)

TARGET_COLUMN = "mcp_rs_mwh"

MARKET_INPUT_COLUMNS = [
    "purchase_bid_mw",
    "sell_bid_mw",
    "mcv_mw",
    "scheduled_volume_mw",
]

TIME_FEATURES = [
    "hour",
    "minute",
    "block_number",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "weekend_flag",
]

MCP_LAG_STEPS = [1, 2, 4, 8, 16, 96, 192, 672]

LAG_FEATURES = [f"mcp_lag_{k}" for k in MCP_LAG_STEPS]

MICROSTRUCTURE_FEATURES = [
    "demand_supply_ratio",
    "market_imbalance",
    "bid_spread",
    "relative_volume",
    "mcp_momentum",
    "mcp_velocity",
    "mcp_acceleration",
    "rolling_bid_ratio_4",
    "rolling_bid_ratio_8",
    "rolling_bid_ratio_96",
    "bid_ratio",
    "bid_ratio_ma4",
    "bid_ratio_ma16",
    "volume_fill_ratio",
    "sell_bid_ma4",
    "sell_bid_drop",
    "purchase_bid_ma4",
    "purchase_surge",
]

SEASONALITY_FEATURES = [
    "is_summer",
    "is_monsoon",
    "is_winter",
    "month_start",
    "month_end",
    "is_weekday",
    "is_weekend",
    "is_monday",
    "is_friday",
    "morning_peak",
    "afternoon_peak",
    "evening_peak",
    "night_period",
    "market_imbalance_x_evening_peak",
    "bid_spread_x_evening_peak",
    "demand_supply_ratio_x_summer",
    "market_imbalance_x_monsoon",
]

ROLLING_WINDOWS = [4, 8, 96]
ROLLING_BID_RATIO_WINDOWS = [4, 8, 96]
VOLUME_ROLLING_WINDOW = 96

_EPS = 1e-6

ROLLING_MEAN_FEATURES = [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
ROLLING_STD_FEATURES = [f"rolling_std_{w}" for w in ROLLING_WINDOWS]

CYCLICAL_FEATURES = [
    "sin_hour",
    "cos_hour",
    "sin_block",
    "cos_block",
]

WEATHER_FEATURES = [
    "temperature_2m",
    "cloudcover",
    "windspeed_10m",
    "precipitation",
    "temp_above_32",
    "temp_above_35",
    "solar_suppressor",
    "is_hot_evening",
]

HOLIDAY_FEATURES = [
    "is_holiday",
    "is_holiday_eve",
    "is_holiday_post",
    "is_rajyotsava",
]

ENGINEERED_FEATURES = (
    TIME_FEATURES
    + LAG_FEATURES
    + MICROSTRUCTURE_FEATURES
    + ROLLING_MEAN_FEATURES
    + ROLLING_STD_FEATURES
    + CYCLICAL_FEATURES
    + SEASONALITY_FEATURES
    + WEATHER_FEATURES
    + HOLIDAY_FEATURES
)

IDENTIFIER_COLUMNS = [
    "block_timestamp",
    "trade_date",
    "session_id",
    "source_file",
]


@dataclass
class FeaturePipelineResult:
    dataframe: pd.DataFrame
    input_path: Path
    output_path: Path
    manifest_path: Path
    rows_in: int
    rows_out: int
    rows_dropped: int


class RTMFeaturePipeline:
    """
    Builds model-ready features from rtm_master.parquet.

    Target: mcp_rs_mwh (MCP in Rs/MWh).
    Input market columns: purchase_bid_mw, sell_bid_mw, mcv_mw, scheduled_volume_mw.
    """

    def __init__(
        self,
        master_path: Path | str = "data/processed/rtm_master.parquet",
        output_path: Path | str = "data/features/features.parquet",
        drop_warmup_rows: bool = True,
    ) -> None:
        self.master_path = Path(master_path)
        self.output_path = Path(output_path)
        self.manifest_path = self.output_path.parent / "features_manifest.json"
        self.drop_warmup_rows = drop_warmup_rows

    def run(self) -> FeaturePipelineResult:
        if not self.master_path.exists():
            raise FileNotFoundError(
                f"Master dataset not found: {self.master_path}. "
                "Run scripts/build_rtm_master.py first."
            )

        raw = self._load_master()
        rows_in = len(raw)
        logger.info("Loaded %d rows from %s", rows_in, self.master_path)

        featured = self.build_features(raw)
        rows_before_drop = len(featured)

        if self.drop_warmup_rows:
            featured = self._drop_incomplete_rows(featured)

        rows_out = len(featured)
        self._save(featured)
        self._write_manifest(rows_in, rows_before_drop, rows_out)

        logger.info(
            "Feature pipeline complete: %d rows saved (%d dropped)",
            rows_out,
            rows_in - rows_out,
        )
        return FeaturePipelineResult(
            dataframe=featured,
            input_path=self.master_path,
            output_path=self.output_path,
            manifest_path=self.manifest_path,
            rows_in=rows_in,
            rows_out=rows_out,
            rows_dropped=rows_in - rows_out,
        )

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full feature transformations to a chronologically sorted frame."""
        out = df.copy()
        out = self._prepare_timestamps(out)
        out = self._add_block_number(out)
        out = self._add_time_features(out)
        out = self._add_microstructure_features(out)
        out = self._add_mcp_lags(out)
        out = self._add_mcp_dynamics(out)
        out = self._add_rolling_features(out)
        out = self._add_relative_volume(out)
        out = self._add_rolling_bid_ratios(out)
        out = self._add_cyclical_features(out)
        out = self._add_seasonality_features(out)
        out = self._add_weather_features(out)
        out = self._add_holiday_features(out)
        return self._select_output_columns(out)

    def _load_master(self) -> pd.DataFrame:
        df = pd.read_parquet(self.master_path)
        required = [TARGET_COLUMN, "block_timestamp", *MARKET_INPUT_COLUMNS]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Master dataset missing columns: {missing}")

        df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], utc=True)
        if df["block_timestamp"].dt.tz is not None:
            df["block_timestamp"] = df["block_timestamp"].dt.tz_convert(TIMEZONE)

        return df.sort_values("block_timestamp").reset_index(drop=True)

    @staticmethod
    def _prepare_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        if "trade_date" not in df.columns:
            df["trade_date"] = df["block_timestamp"].dt.date
        else:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
        return df

    @staticmethod
    def _add_block_number(df: pd.DataFrame) -> pd.DataFrame:
        if "daily_block_index" in df.columns and df["daily_block_index"].notna().any():
            df["block_number"] = df["daily_block_index"].astype(int)
        elif "hour" in df.columns and "time_block" in df.columns:
            df["block_number"] = (df["hour"] - 1) * 4 + df["time_block"]
        else:
            minutes = df["block_timestamp"].dt.hour * 60 + df["block_timestamp"].dt.minute
            df["block_number"] = (minutes // BLOCK_MINUTES) + 1

        df["block_number"] = df["block_number"].clip(1, BLOCKS_PER_DAY).astype(int)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["block_timestamp"]
        df["minute"] = ts.dt.minute.astype(int)

        if "hour" in df.columns and df["hour"].between(1, 24).all():
            df["hour"] = df["hour"].astype(int)
        else:
            df["hour"] = ts.dt.hour + 1

        df["day_of_week"] = ts.dt.dayofweek.astype(int)
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["month"] = ts.dt.month.astype(int)
        df["quarter"] = ts.dt.quarter.astype(int)
        df["weekend_flag"] = (df["day_of_week"] >= 5).astype(int)
        return df

    @staticmethod
    def _add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Point-in-time bid/volume microstructure (lags added separately)."""
        purchase = df["purchase_bid_mw"]
        sell = df["sell_bid_mw"]
        mcv = df["mcv_mw"]

        df["demand_supply_ratio"] = purchase / (sell + _EPS)
        df["market_imbalance"] = (purchase - sell) / (purchase + sell + _EPS)
        
        # Core market pressure features
        df["bid_ratio"]          = purchase / sell.clip(lower=1.0)
        df["bid_spread"]         = purchase - sell
        df["bid_ratio_ma4"]      = df["bid_ratio"].rolling(4, min_periods=1).mean()
        df["bid_ratio_ma16"]     = df["bid_ratio"].rolling(16, min_periods=1).mean()
        df["volume_fill_ratio"]  = mcv / purchase.clip(lower=1.0)
        df["sell_bid_ma4"]       = sell.rolling(4, min_periods=1).mean()
        df["sell_bid_drop"]      = sell / df["sell_bid_ma4"].clip(lower=1.0)
        df["purchase_bid_ma4"]   = purchase.rolling(4, min_periods=1).mean()
        df["purchase_surge"]     = purchase / df["purchase_bid_ma4"].clip(lower=1.0)
        
        return df

    @staticmethod
    def _add_mcp_lags(df: pd.DataFrame) -> pd.DataFrame:
        mcp = df[TARGET_COLUMN]
        for lag in MCP_LAG_STEPS:
            df[f"mcp_lag_{lag}"] = mcp.shift(lag)
        return df

    @staticmethod
    def _add_mcp_dynamics(df: pd.DataFrame) -> pd.DataFrame:
        """MCP momentum, velocity, and acceleration from short lags."""
        df["mcp_momentum"] = df["mcp_lag_1"] - df["mcp_lag_4"]
        df["mcp_velocity"] = df["mcp_lag_1"] - df["mcp_lag_2"]
        df["mcp_acceleration"] = (df["mcp_lag_1"] - df["mcp_lag_2"]) - (df["mcp_lag_2"] - df["mcp_lag_4"])
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        mcp = df[TARGET_COLUMN]
        for window in ROLLING_WINDOWS:
            df[f"rolling_mean_{window}"] = mcp.rolling(window=window, min_periods=window).mean()
            df[f"rolling_std_{window}"] = mcp.rolling(window=window, min_periods=window).std()
        return df

    @staticmethod
    def _add_relative_volume(df: pd.DataFrame) -> pd.DataFrame:
        volume = df["scheduled_volume_mw"]
        roll_mean = volume.rolling(window=VOLUME_ROLLING_WINDOW, min_periods=VOLUME_ROLLING_WINDOW).mean()
        df["relative_volume"] = volume / (roll_mean + _EPS)
        return df

    @staticmethod
    def _add_rolling_bid_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Rolling demand/supply ratio over 4, 8, and 96 blocks (sum purchase / sum sell)."""
        purchase = df["purchase_bid_mw"]
        sell = df["sell_bid_mw"]
        for window in ROLLING_BID_RATIO_WINDOWS:
            roll_purchase = purchase.rolling(window=window, min_periods=window).sum()
            roll_sell = sell.rolling(window=window, min_periods=window).sum()
            df[f"rolling_bid_ratio_{window}"] = roll_purchase / (roll_sell + _EPS)
        return df

    @staticmethod
    def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
        hour_angle = 2 * np.pi * (df["hour"] - 1) / 24
        block_angle = 2 * np.pi * (df["block_number"] - 1) / BLOCKS_PER_DAY

        df["sin_hour"] = np.sin(hour_angle)
        df["cos_hour"] = np.cos(hour_angle)
        df["sin_block"] = np.sin(block_angle)
        df["cos_block"] = np.cos(block_angle)
        return df

    @staticmethod
    def _add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
        ts = df["block_timestamp"]
        hour = ts.dt.hour
        day = ts.dt.day
        month = df["month"]
        day_of_week = df["day_of_week"]

        # Season Features
        df["is_summer"] = month.isin([3, 4, 5, 6]).astype(int)
        df["is_monsoon"] = month.isin([7, 8, 9, 10]).astype(int)
        df["is_winter"] = month.isin([11, 12, 1, 2]).astype(int)

        # Month Features
        df["month_start"] = (day <= 7).astype(int)
        df["month_end"] = (day >= 25).astype(int)

        # Week Features
        df["is_weekday"] = (day_of_week < 5).astype(int)
        df["is_weekend"] = (day_of_week >= 5).astype(int)
        df["is_monday"] = (day_of_week == 0).astype(int)
        df["is_friday"] = (day_of_week == 4).astype(int)

        # Peak Features
        df["morning_peak"] = ((hour >= 6) & (hour < 10)).astype(int)
        df["afternoon_peak"] = ((hour >= 11) & (hour < 17)).astype(int)
        df["evening_peak"] = ((hour >= 18) & (hour < 23)).astype(int)
        df["night_period"] = ((hour >= 0) & (hour < 5)).astype(int)

        # Interaction Features
        df["market_imbalance_x_evening_peak"] = df["market_imbalance"] * df["evening_peak"]
        df["bid_spread_x_evening_peak"] = df["bid_spread"] * df["evening_peak"]
        df["demand_supply_ratio_x_summer"] = df["demand_supply_ratio"] * df["is_summer"]
        df["market_imbalance_x_monsoon"] = df["market_imbalance"] * df["is_monsoon"]

        return df

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from pathlib import Path
        weather_path = Path("data/external/weather_bengaluru.parquet")
        if not weather_path.exists():
            df["temperature_2m"]  = 28.0
            df["cloudcover"]      = 40.0
            df["windspeed_10m"]   = 8.0
            df["precipitation"]   = 0.0
            df["temp_above_32"]   = 0.0
            df["temp_above_35"]   = 0.0
            df["solar_suppressor"]= 0.0
            df["is_hot_evening"]  = 0
            return df

        wx = pd.read_parquet(weather_path).set_index("time")
        wx.index = pd.to_datetime(wx.index)
        wx_15min = wx.resample("15min").ffill()

        df = df.copy()
        ts_floored = df["block_timestamp"].dt.tz_localize(None).dt.floor("h")
        for col in ["temperature_2m", "cloudcover", "windspeed_10m", "precipitation"]:
            df[col] = ts_floored.map(wx_15min[col])

        df["temp_above_32"]    = (df["temperature_2m"] - 32).clip(lower=0)
        df["temp_above_35"]    = (df["temperature_2m"] - 35).clip(lower=0)
        df["solar_suppressor"] = (df["cloudcover"] / 100.0) * \
                                  df["block_number"].between(33, 60).astype(float)
        df["is_hot_evening"]   = (
            (df["temperature_2m"] > 30) &
            (df["block_number"].between(68, 88))
        ).astype(int)
        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd
        holidays = pd.to_datetime([
            "2024-01-26","2024-03-25","2024-04-14","2024-04-17",
            "2024-05-23","2024-08-15","2024-10-02","2024-10-12",
            "2024-10-13","2024-11-01","2024-11-15","2024-12-25",
            "2025-01-26","2025-03-14","2025-04-06","2025-04-18",
            "2025-08-15","2025-10-02","2025-10-20","2025-10-21",
            "2025-11-01","2025-12-25",
            "2026-01-26","2026-03-03","2026-03-26","2026-04-03",
            "2026-08-15","2026-10-02","2026-11-01",
        ]).date
        holiday_set = set(holidays)
        dates = df["block_timestamp"].dt.date
        df["is_holiday"]      = dates.isin(holiday_set).astype(int)
        df["is_holiday_eve"]  = dates.map(lambda d: (d + pd.Timedelta(days=1)) in holiday_set).astype(int)
        df["is_holiday_post"] = dates.map(lambda d: (d - pd.Timedelta(days=1)) in holiday_set).astype(int)
        df["is_rajyotsava"]   = dates.map(lambda d: d.month == 11 and d.day == 1).astype(int)
        return df

    @staticmethod
    def _drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove warmup rows where long lags and full-window rolls are undefined."""
        required = (
            LAG_FEATURES
            + MICROSTRUCTURE_FEATURES
            + ROLLING_MEAN_FEATURES
            + ROLLING_STD_FEATURES
            + [TARGET_COLUMN]
        )
        return df.dropna(subset=required).reset_index(drop=True)

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for col in IDENTIFIER_COLUMNS:
            if col in df.columns:
                keep.append(col)
        keep.append(TARGET_COLUMN)
        keep.extend(MARKET_INPUT_COLUMNS)
        keep.extend(ENGINEERED_FEATURES)
        existing = [c for c in keep if c in df.columns]
        return df[existing]

    def _save(self, df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False)
        logger.info("Saved features to %s (%d rows, %d columns)", self.output_path, len(df), len(df.columns))

    def _write_manifest(self, rows_in: int, rows_before_drop: int, rows_out: int) -> None:
        manifest = {
            "target": TARGET_COLUMN,
            "input_path": str(self.master_path),
            "output_path": str(self.output_path),
            "rows_in": rows_in,
            "rows_before_drop": rows_before_drop,
            "rows_out": rows_out,
            "rows_dropped": rows_in - rows_out,
            "market_inputs": MARKET_INPUT_COLUMNS,
            "time_features": TIME_FEATURES,
            "lag_features": LAG_FEATURES,
            "microstructure_features": MICROSTRUCTURE_FEATURES,
            "rolling_mean_features": ROLLING_MEAN_FEATURES,
            "rolling_std_features": ROLLING_STD_FEATURES,
            "cyclical_features": CYCLICAL_FEATURES,
            "seasonality_features": SEASONALITY_FEATURES,
            "all_engineered_features": ENGINEERED_FEATURES,
            "max_lag_blocks": max(MCP_LAG_STEPS),
            "max_lag_days": round(max(MCP_LAG_STEPS) / BLOCKS_PER_DAY, 2),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> int:
    configure_logging()
    result = RTMFeaturePipeline().run()
    print(f"Features saved: {result.output_path}")
    print(f"Rows: {result.rows_out:,} (dropped {result.rows_dropped:,} warmup rows)")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
