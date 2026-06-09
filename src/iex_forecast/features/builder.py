"""Feature engineering for RTM 15-minute blocks."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from iex_forecast.domain.constants import BLOCKS_PER_DAY, FEATURE_COLUMNS


class FeatureBuilder:
    """Builds model features from ordered block-level history."""

    def transform(self, df: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        out = df.sort_values("block_timestamp").copy()
        out = self._add_calendar_features(out)
        out = self._add_bid_features(out)
        out = self._add_weather_features(out)
        out = self._add_market_pressure_features(out)
        out = self._add_mcp_lags(out)
        out = self._add_market_regime(out)
        out = self._drop_collinear_features(out)
        
        if horizon is not None:
            out["horizon"] = horizon
        elif "horizon" not in out.columns:
            out["horizon"] = 1

        available = [c for c in FEATURE_COLUMNS if c in out.columns]
        return out[available + ["block_timestamp", "mcp_rs_mwh"]].copy()

    def _drop_collinear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop highly collinear features to prevent overfitting."""
        # We manually drop some known highly collinear lag features 
        # or features that cause multi-collinearity issues.
        cols_to_drop = [
            "mcp_lag_2", 
            "mcp_roll_mean_4",
            "rolling_bid_ratio_4"
        ]
        to_drop = [c for c in cols_to_drop if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
        return df

    def transform_latest_row(
        self,
        history: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """Single inference row for a given forecast horizon."""
        enriched = self.transform(history)
        if enriched.empty:
            return pd.DataFrame()
        row = enriched.iloc[[-1]].copy()
        row["horizon"] = horizon
        feature_cols = [c for c in FEATURE_COLUMNS if c in row.columns]
        return row[feature_cols]

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(df["block_timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("Asia/Kolkata")
        df["day_of_week"] = ts.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        block_col = "time_block" if "time_block" in df.columns else "block_number"
        block_idx = (df["hour"] - 1) * 4 + (df[block_col] - 1)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["block_sin"] = np.sin(2 * np.pi * block_idx / BLOCKS_PER_DAY)
        df["block_cos"] = np.cos(2 * np.pi * block_idx / BLOCKS_PER_DAY)
        return df

    def _add_bid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bid_imbalance_mw"] = df["purchase_bid_mw"] - df["sell_bid_mw"]
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
            df["temp_ensemble_spread"] = 0.0
            return df

        wx = pd.read_parquet(weather_path).set_index("time")
        wx.index = pd.to_datetime(wx.index)
        wx_15min = wx.resample("15min").ffill()

        df = df.copy()
        ts_floored = df["block_timestamp"].dt.tz_localize(None).dt.floor("h")
        
        cols_to_merge = ["temperature_2m", "cloudcover", "windspeed_10m", "precipitation"]
        if "temp_ensemble_spread" in wx_15min.columns:
            cols_to_merge.append("temp_ensemble_spread")
            
        for col in cols_to_merge:
            df[col] = ts_floored.map(wx_15min[col]).fillna(0)
            
        if "temp_ensemble_spread" not in df.columns:
            df["temp_ensemble_spread"] = 0.0
            
        df["temp_above_32"] = (df["temperature_2m"] > 32).astype(int)
        df["temp_above_35"] = (df["temperature_2m"] > 35).astype(int)
        df["solar_suppressor"] = ((df["hour"] >= 9) & (df["hour"] <= 16) & (df["cloudcover"] < 30)).astype(int)
        df["is_hot_evening"] = ((df["hour"] >= 18) & (df["hour"] <= 23) & (df["temperature_2m"] > 30)).astype(int)
        return df

    def _add_market_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supply-demand imbalance signals from bid data."""
        df["bid_ratio"]          = df["purchase_bid_mw"] / df["sell_bid_mw"].clip(lower=1.0)
        df["bid_spread"]         = df["purchase_bid_mw"] - df["sell_bid_mw"]
        df["bid_ratio_ma4"]      = df["bid_ratio"].rolling(4, min_periods=1).mean()
        df["bid_ratio_ma16"]     = df["bid_ratio"].rolling(16, min_periods=1).mean()
        df["volume_fill_ratio"]  = df["mcv_mw"] / df["purchase_bid_mw"].clip(lower=1.0)
        df["sell_bid_ma4"]       = df["sell_bid_mw"].rolling(4, min_periods=1).mean()
        df["sell_bid_drop"]      = df["sell_bid_mw"] / df["sell_bid_ma4"].clip(lower=1.0)
        df["purchase_bid_ma4"]   = df["purchase_bid_mw"].rolling(4, min_periods=1).mean()
        df["purchase_surge"]     = df["purchase_bid_mw"] / df["purchase_bid_ma4"].clip(lower=1.0)
        return df

    def _add_mcp_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        mcp = df["mcp_rs_mwh"]
        df["mcp_lag_1"] = mcp.shift(1)
        df["mcp_lag_4"] = mcp.shift(4)
        df["mcp_lag_96"] = mcp.shift(BLOCKS_PER_DAY)
        df["mcp_roll_mean_4"] = mcp.rolling(4, min_periods=1).mean()
        df["mcp_roll_mean_96"] = mcp.rolling(BLOCKS_PER_DAY, min_periods=1).mean()
        df["mcp_roll_std_4"] = mcp.rolling(4, min_periods=1).std().fillna(0)
        return df

    def _add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """K-Means clustering to define market regimes."""
        if "bid_imbalance_mw" not in df.columns or "bid_spread" not in df.columns or len(df) < 3:
            df["market_regime_id"] = 0
            return df

        features = df[["bid_imbalance_mw", "bid_spread"]].fillna(0)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        df["market_regime_id"] = kmeans.fit_predict(features)
        return df
