"""Recursive multi-step ensemble forecasting for 96 RTM blocks (24 hours)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data.schema import BLOCK_MINUTES, BLOCKS_PER_DAY, TIMEZONE
from features.build_features import (
    MARKET_INPUT_COLUMNS,
    MCP_LAG_STEPS,
    ROLLING_WINDOWS,
    RTMFeaturePipeline,
    TARGET_COLUMN,
)
from models.ensemble import WeightedEnsemble

logger = logging.getLogger(__name__)

FORECAST_HORIZON = 96
MIN_HISTORY_BLOCKS = max(max(MCP_LAG_STEPS), max(ROLLING_WINDOWS)) + 1

@dataclass
class ForecastBlock:
    horizon: int
    timestamp: datetime
    predicted_mcp: float
    confidence_interval: float
    confidence_lower: float
    confidence_upper: float


@dataclass
class RecursiveForecastResult:
    origin_timestamp: datetime
    blocks: list[ForecastBlock]
    output_path: Path
    dataframe: pd.DataFrame


class RecursiveEnsembleForecaster:
    """
    Recursively forecasts the next 96 MCP values using the weighted ensemble.

    Each step:
        1. Engineer features from the rolling market-state history.
        2. Predict next-block MCP via ensemble (1-step model).
        3. Append synthetic block to history (MCP forecast + carried-forward bids).
        4. Advance timestamp by 15 minutes.
    """

    def __init__(
        self,
        ensemble_path: Path | str = "models/ensemble.pkl",
        master_path: Path | str = "data/processed/rtm_master.parquet",
        output_path: Path | str = "forecasts/forecast_96.csv",
        horizon: int = FORECAST_HORIZON,
        z_score: float = 1.96,
        min_history_blocks: int = MIN_HISTORY_BLOCKS,
    ) -> None:
        self.ensemble_path = Path(ensemble_path)
        self.master_path = Path(master_path)
        self.output_path = Path(output_path)
        self.horizon = horizon
        self.z_score = z_score
        self.min_history_blocks = min_history_blocks
        self._feature_pipeline = RTMFeaturePipeline()
        self._ensemble: WeightedEnsemble | None = None

    def run(
        self,
        market_state: pd.DataFrame | None = None,
        save: bool = True,
    ) -> RecursiveForecastResult:
        self._load_ensemble()
        history = self._resolve_market_state(market_state)
        origin_ts = pd.to_datetime(history["block_timestamp"].iloc[-1])
        if origin_ts.tzinfo is None:
            origin_ts = origin_ts.tz_localize(TIMEZONE)

        working = history.copy()
        blocks: list[ForecastBlock] = []

        logger.info(
            "Starting recursive forecast: origin=%s horizon=%d history_rows=%d",
            origin_ts,
            self.horizon,
            len(working),
        )

        for step in range(1, self.horizon + 1):
            forecast_ts = origin_ts + timedelta(minutes=BLOCK_MINUTES * step)
            pred, lower, upper = self._predict_next_block(working)

            half_width = float((upper[0] - lower[0]) / 2.0)
            blocks.append(
                ForecastBlock(
                    horizon=step,
                    timestamp=forecast_ts.to_pydatetime(),
                    predicted_mcp=round(float(pred[0]), 2),
                    confidence_interval=round(half_width, 2),
                    confidence_lower=round(float(lower[0]), 2),
                    confidence_upper=round(float(upper[0]), 2),
                )
            )

            working = self._append_synthetic_block(working, forecast_ts, float(pred[0]))

        result_df = self._to_dataframe(blocks)
        if save:
            self._save_csv(result_df)

        logger.info("Forecast complete: %d blocks saved to %s", len(blocks), self.output_path)
        return RecursiveForecastResult(
            origin_timestamp=origin_ts.to_pydatetime(),
            blocks=blocks,
            output_path=self.output_path,
            dataframe=result_df,
        )

    def _load_ensemble(self) -> None:
        if not self.ensemble_path.exists():
            raise FileNotFoundError(
                f"Ensemble model not found: {self.ensemble_path}. Run scripts/train_ensemble.py."
            )
        bundle = joblib.load(self.ensemble_path)
        self._ensemble = bundle.get("ensemble") or bundle
        if not isinstance(self._ensemble, WeightedEnsemble):
            self._ensemble = WeightedEnsemble(
                weights=bundle["weights"],
                model_paths=bundle["model_paths"],
                feature_names=bundle["feature_names"],
            )

    def _resolve_market_state(self, market_state: pd.DataFrame | None) -> pd.DataFrame:
        if market_state is not None:
            return self._normalize_market_state(market_state)

        if not self.master_path.exists():
            raise FileNotFoundError(
                f"No market state provided and master file missing: {self.master_path}"
            )

        df = pd.read_parquet(self.master_path)
        return self._normalize_market_state(df)

    def _normalize_market_state(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [TARGET_COLUMN, "block_timestamp", *MARKET_INPUT_COLUMNS]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Market state missing columns: {missing}")

        out = df.copy()
        out["block_timestamp"] = pd.to_datetime(out["block_timestamp"], utc=True)
        if out["block_timestamp"].dt.tz is not None:
            out["block_timestamp"] = out["block_timestamp"].dt.tz_convert(TIMEZONE)

        out = out.sort_values("block_timestamp").reset_index(drop=True)

        if len(out) < self.min_history_blocks:
            raise ValueError(
                f"Need at least {self.min_history_blocks} historical blocks for lag/rolling features; "
                f"got {len(out)}."
            )

        if "session_id" not in out.columns:
            out["session_id"] = 1
        if "source_file" not in out.columns:
            out["source_file"] = "inference"

        out = self._ensure_time_columns(out)
        return out

    @staticmethod
    def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
        ts = df["block_timestamp"]
        minutes = ts.dt.hour * 60 + ts.dt.minute
        if "hour" not in df.columns or df["hour"].isna().any():
            df["hour"] = (minutes // 60 + 1).astype(int)
        if "time_block" not in df.columns or df["time_block"].isna().any():
            df["time_block"] = ((minutes % 60) // BLOCK_MINUTES + 1).astype(int)
        if "daily_block_index" not in df.columns:
            df["daily_block_index"] = (df["hour"] - 1) * 4 + df["time_block"]
        if "trade_date" not in df.columns:
            df["trade_date"] = ts.dt.date
        return df

    def _predict_next_block(self, history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        featured = self._feature_pipeline.build_features(history)
        feature_cols = list(self._ensemble.feature_names)
        row = featured.iloc[[-1]][feature_cols]

        if row.isna().any().any():
            bad = row.columns[row.isna().any()].tolist()
            raise ValueError(
                f"Cannot forecast: feature row contains NaN in {bad}. "
                f"Provide longer history (>= {self.min_history_blocks} blocks)."
            )

        return self._ensemble.predict_with_interval(row, z_score=self.z_score)

    @staticmethod
    def _append_synthetic_block(
        history: pd.DataFrame,
        forecast_ts: pd.Timestamp,
        mcp: float,
    ) -> pd.DataFrame:
        last = history.iloc[-1]
        minutes = forecast_ts.hour * 60 + forecast_ts.minute
        hour = int(minutes // 60 + 1)
        time_block = int((minutes % 60) // BLOCK_MINUTES + 1)
        block_number = (hour - 1) * 4 + time_block

        row = {
            "block_timestamp": forecast_ts,
            "trade_date": forecast_ts.date(),
            "hour": hour,
            "time_block": time_block,
            "daily_block_index": block_number,
            "session_id": last.get("session_id", 1),
            "purchase_bid_mw": float(last["purchase_bid_mw"]),
            "sell_bid_mw": float(last["sell_bid_mw"]),
            "mcv_mw": float(last["mcv_mw"]),
            "scheduled_volume_mw": float(last["scheduled_volume_mw"]),
            "mcp_rs_mwh": mcp,
            "source_file": last.get("source_file", "forecast"),
        }
        return pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    @staticmethod
    def _to_dataframe(blocks: list[ForecastBlock]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "horizon": b.horizon,
                    "timestamp": b.timestamp.isoformat(),
                    "predicted_mcp": b.predicted_mcp,
                    "confidence_interval": b.confidence_interval,
                    "confidence_lower": b.confidence_lower,
                    "confidence_upper": b.confidence_upper,
                }
                for b in blocks
            ]
        )

    def _save_csv(self, df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        export = df[["timestamp", "predicted_mcp", "confidence_interval", "confidence_lower", "confidence_upper"]]
        export.to_csv(self.output_path, index=False)
        logger.info("Saved forecast CSV: %s", self.output_path)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> int:
    configure_logging()
    result = RecursiveEnsembleForecaster().run()
    print(f"Origin:   {result.origin_timestamp}")
    print(f"Forecast: {result.output_path}")
    print(result.dataframe.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
