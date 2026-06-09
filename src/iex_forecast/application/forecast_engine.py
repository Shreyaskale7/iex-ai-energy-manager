"""Production Forecast Generation Engine.

Orchestrates 24-Hour, 7-Day, and 30-Day MCP forecast generation using
the trained ensemble and direct-horizon models, then auto-persists results
to PostgreSQL and CSV.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data.schema import BLOCK_MINUTES, TIMEZONE
from features.build_features import RTMFeaturePipeline, TARGET_COLUMN, MARKET_INPUT_COLUMNS
from iex_forecast.application.forecast_service import ForecastService
from iex_forecast.domain.constants import classify_zone
from iex_forecast.domain.entities import ForecastPoint
from iex_forecast.infrastructure.database import get_session_factory
from iex_forecast.infrastructure.repositories import PostgresForecastRepository
from models.ensemble import WeightedEnsemble
from models.spike_classifier import SpikeClassifierBundle, load_spike_classifier

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
BLOCKS_24H = 96
BLOCKS_7D = 672
BLOCKS_30D = 2880

# Anchor horizons where we have trained direct models
DIRECT_ANCHORS = [1, 4, 8, 12, 24, 48, 60, 68, 72, 76, 78, 80, 82, 84, 86, 88, 92, 96, 192, 672, 2880]

# Spike classifier reconstruction configuration
SPIKE_PROB_THRESHOLD = 0.70
SPIKE_BOOST_FACTOR = 0.15

# Residual pattern reconstruction configuration
RESIDUAL_PROFILE_FILENAME = "residual_profile.parquet"
RESIDUAL_PROFILE_SCALE = {
    "7d": 3.0,
    "30d": 1.5,
}

# Map forecast type to block count
FORECAST_TYPES = {
    "24h": BLOCKS_24H,
    "7d": BLOCKS_7D,
    "30d": BLOCKS_30D,
}


@dataclass
class ForecastRunResult:
    """Summary of a single forecast generation run."""
    forecast_type: str
    run_id: str
    total_blocks: int
    generated_at: str
    model_version: str
    csv_latest_path: str
    csv_archive_path: str
    zone_summary: dict[str, int]


class ForecastGenerationService:
    """
    Production forecast engine that generates 24h / 7d / 30d MCP forecasts.

    Architecture:
        - 24-Hour: Recursive ensemble (step-by-step, 96 blocks)
        - 7-Day:   Direct multi-horizon interpolation (672 blocks)
        - 30-Day:  Direct long-horizon interpolation (2880 blocks)
    """

    def __init__(
        self,
        ensemble_path: Path | str = "models/ensemble.pkl",
        direct_model_dir: Path | str = "models/direct",
        spike_classifier_path: Path | str = "models/spike_classifier.pkl",
        master_path: Path | str = "data/processed/rtm_master.parquet",
        features_path: Path | str = "data/features/features.parquet",
        output_dir: Path | str = "forecasts",
        model_version: str = "ensemble-v2",
        feature_version: str = "v2-seasonality",
    ) -> None:
        self.ensemble_path = Path(ensemble_path)
        self.direct_model_dir = Path(direct_model_dir)
        self.spike_classifier_path = Path(spike_classifier_path)
        self.master_path = Path(master_path)
        self.features_path = Path(features_path)
        self.output_dir = Path(output_dir)
        self.model_version = model_version
        self.feature_version = feature_version

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "archive").mkdir(parents=True, exist_ok=True)

        # Lazy-loaded assets
        self._ensemble: WeightedEnsemble | None = None
        self._direct_models: dict[int, Any] = {}
        self._spike_classifier: SpikeClassifierBundle | None = None
        self._feature_pipeline = RTMFeaturePipeline()
        self._forecast_service: ForecastService | None = None

    # ── Public API ────────────────────────────────────────────────────

    def generate_24h(self, market_state: pd.DataFrame | None = None) -> ForecastRunResult:
        """Generate 24-hour forecast (96 blocks) using recursive ensemble."""
        logger.info("=== Generating 24-Hour Forecast (96 blocks) ===")
        self._ensure_loaded()

        history = self._get_market_state(market_state)
        origin_ts = pd.to_datetime(history["block_timestamp"].iloc[-1])
        if origin_ts.tzinfo is None:
            origin_ts = origin_ts.tz_localize(TIMEZONE)

        blocks = self._recursive_forecast(history, origin_ts, BLOCKS_24H)
        points = self._blocks_to_forecast_points(blocks)
        return self._persist_run(origin_ts, points, "24h")

    def generate_7d(self, market_state: pd.DataFrame | None = None) -> ForecastRunResult:
        """Generate 7-day forecast (672 blocks) using direct multi-horizon models."""
        logger.info("=== Generating 7-Day Forecast (672 blocks) ===")
        self._ensure_loaded()

        features_df = self._get_feature_state()
        origin_ts = pd.to_datetime(features_df["block_timestamp"].iloc[-1])
        if origin_ts.tzinfo is None:
            origin_ts = origin_ts.tz_localize(TIMEZONE)

        anchors = [h for h in DIRECT_ANCHORS if h <= BLOCKS_7D]
        profile = self._load_residual_profile()
        blocks = self._direct_interpolated_forecast(
            features_df,
            origin_ts,
            anchors,
            BLOCKS_7D,
            residual_profile=profile,
            residual_scale=RESIDUAL_PROFILE_SCALE.get("7d", 0.0),
        )
        points = self._blocks_to_forecast_points(blocks)
        return self._persist_run(origin_ts, points, "7d")

    def generate_30d(self, market_state: pd.DataFrame | None = None) -> ForecastRunResult:
        """Generate 30-day forecast (2880 blocks) using direct long-horizon models."""
        logger.info("=== Generating 30-Day Forecast (2880 blocks) ===")
        self._ensure_loaded()

        features_df = self._get_feature_state()
        origin_ts = pd.to_datetime(features_df["block_timestamp"].iloc[-1])
        if origin_ts.tzinfo is None:
            origin_ts = origin_ts.tz_localize(TIMEZONE)

        profile = self._load_residual_profile()
        blocks = self._direct_interpolated_forecast(
            features_df,
            origin_ts,
            DIRECT_ANCHORS,
            BLOCKS_30D,
            residual_profile=profile,
            residual_scale=RESIDUAL_PROFILE_SCALE.get("30d", 0.0),
        )
        points = self._blocks_to_forecast_points(blocks)
        return self._persist_run(origin_ts, points, "30d")

    def generate_all(self, market_state: pd.DataFrame | None = None) -> list[ForecastRunResult]:
        """Generate all three forecast types."""
        results = []
        results.append(self.generate_24h(market_state))
        results.append(self.generate_7d(market_state))
        results.append(self.generate_30d(market_state))
        return results

    def _compute_regime_confidence(self, block_number: int, base_confidence: float) -> float:
        if 33 <= block_number <= 60:
            regime_factor = 1.15
        elif 68 <= block_number <= 88:
            regime_factor = 0.75
        elif 61 <= block_number <= 67:
            regime_factor = 0.85
        else:
            regime_factor = 1.0
        return min(1.0, base_confidence * regime_factor)

    # ── Recursive Engine (24h) ────────────────────────────────────────

    def _recursive_forecast(
        self,
        history: pd.DataFrame,
        origin_ts: pd.Timestamp,
        n_blocks: int,
    ) -> list[dict[str, Any]]:
        """Step-by-step recursive prediction using the weighted ensemble."""
        working = history.copy()
        blocks: list[dict[str, Any]] = []

        for step in range(1, n_blocks + 1):
            forecast_ts = origin_ts + timedelta(minutes=BLOCK_MINUTES * step)

            # Build features from rolling history
            featured = self._feature_pipeline.build_features(working)
            feature_cols = list(self._ensemble.feature_names)
            row = featured.iloc[[-1]][feature_cols]

            if row.isna().any().any():
                bad = row.columns[row.isna().any()].tolist()
                logger.warning("NaN in features at step %d: %s — using forward fill", step, bad[:5])
                row = row.ffill(axis=1).bfill(axis=1)

            # Ensemble prediction with confidence interval
            mean_pred, lower, upper = self._ensemble.predict_with_interval(row, z_score=1.96)
            mcp = float(mean_pred[0])
            lb = float(lower[0])
            ub = float(upper[0])

            # Spike probability from classifier
            spike_prob = 0.0
            if self._spike_classifier is not None:
                try:
                    spike_prob = float(self._spike_classifier.spike_probability(row)[0])
                except Exception:
                    spike_prob = 0.0

            # Confidence from interval width
            ci_width = ub - lb
            base_confidence = max(0.0, min(1.0, 1.0 - ci_width / max(abs(mcp), 1.0)))
            block_number = int((forecast_ts.hour * 60 + forecast_ts.minute) // BLOCK_MINUTES) + 1
            confidence = self._compute_regime_confidence(block_number, base_confidence)

            # Spike-aware boost from the classifier
            if spike_prob > SPIKE_PROB_THRESHOLD:
                spike_scale = 1.0 + SPIKE_BOOST_FACTOR * spike_prob
                mcp = float(max(0.0, mcp * spike_scale))
                lb = float(max(0.0, lb * spike_scale))
                ub = float(max(0.0, ub * spike_scale))
                logger.debug(
                    "Applied spike boost: prob=%.3f scale=%.4f step=%d",
                    spike_prob,
                    spike_scale,
                    step,
                )

            blocks.append({
                "horizon": step,
                "forecast_timestamp": forecast_ts.to_pydatetime() if hasattr(forecast_ts, "to_pydatetime") else forecast_ts,
                "predicted_mcp": round(mcp, 2),
                "lower_bound": round(lb, 2),
                "upper_bound": round(ub, 2),
                "confidence": round(confidence, 4),
                "spike_probability": round(spike_prob, 4),
                "zone": classify_zone(mcp),
            })

            # Append synthetic block to history for next step
            working = self._append_synthetic_block(working, forecast_ts, mcp)

            if step % 24 == 0:
                logger.info("  Step %d/%d  MCP=%.1f  zone=%s", step, n_blocks, mcp, blocks[-1]["zone"])

        logger.info("Recursive forecast complete: %d blocks", len(blocks))
        return blocks

    # ── Direct Interpolation Engine (7d / 30d) ────────────────────────

    def _direct_interpolated_forecast(
        self,
        features_df: pd.DataFrame,
        origin_ts: pd.Timestamp,
        anchor_horizons: list[int],
        total_blocks: int,
        residual_profile: pd.DataFrame | None = None,
        residual_scale: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Predict at anchor horizons using direct models, then linearly
        interpolate between anchors for all intermediate blocks.
        """
        # Get feature row (latest available)
        feature_cols = list(self._ensemble.feature_names)
        row = features_df.iloc[[-1]][feature_cols]

        if row.isna().any().any():
            row = row.ffill(axis=1).bfill(axis=1)

        # Predict at each anchor
        anchor_preds: dict[int, float] = {}
        anchor_metrics: dict[int, dict] = {}

        spike_prob: float = 0.0
        if self._spike_classifier is not None:
            try:
                spike_prob = float(self._spike_classifier.spike_probability(row)[0])
            except Exception:
                spike_prob = 0.0

        for h in anchor_horizons:
            if h in self._direct_models:
                bundle = self._direct_models[h]
                model = bundle["model"]
                model_features = bundle["feature_names"]

                # Use intersection of available features
                available = [f for f in model_features if f in features_df.columns]
                pred_row = features_df.iloc[[-1]][available]
                if pred_row.isna().any().any():
                    pred_row = pred_row.ffill(axis=1).bfill(axis=1)

                pred = float(model.predict(pred_row)[0])
                anchor_preds[h] = max(pred, 0.0)  # MCP cannot be negative
                anchor_metrics[h] = bundle.get("metrics", {}).get("test", {})
                logger.info("  Direct h=%d  MCP=%.1f", h, anchor_preds[h])
            else:
                logger.warning("  No direct model for h=%d, will interpolate", h)

        if not anchor_preds:
            raise RuntimeError("No direct models loaded. Cannot generate forecast.")

        # Build full interpolated forecast
        sorted_anchors = sorted(anchor_preds.keys())
        blocks: list[dict[str, Any]] = []

        for step in range(1, total_blocks + 1):
            forecast_ts = origin_ts + timedelta(minutes=BLOCK_MINUTES * step)

            # Find surrounding anchors for interpolation
            mcp = self._interpolate_at_horizon(step, sorted_anchors, anchor_preds)

            # Confidence: decreases with horizon distance from nearest anchor
            nearest_anchor = min(sorted_anchors, key=lambda a: abs(a - step))
            anchor_rmse = anchor_metrics.get(nearest_anchor, {}).get("rmse", 500.0)
            base_confidence = max(0.0, min(1.0, 1.0 - anchor_rmse / max(abs(mcp), 1.0)))
            block_number = int((forecast_ts.hour * 60 + forecast_ts.minute) // BLOCK_MINUTES) + 1
            confidence = self._compute_regime_confidence(block_number, base_confidence)

            if residual_profile is not None and residual_scale > 0:
                profile_offset = self._lookup_residual_profile(residual_profile, origin_ts + timedelta(minutes=BLOCK_MINUTES * step))
                residual_adjustment = float(profile_offset * residual_scale * confidence)
                # Correct sign: add the deviation back onto the smooth interpolation
                mcp = float(max(0.0, mcp + residual_adjustment))

            # Use classifier probability when available, otherwise fall back to a generic decay
            if spike_prob <= 0.0:
                base_spike_rate = 0.10  # ~10% from training data
                horizon_decay = np.exp(-step / (total_blocks * 0.5))
                spike_prob = float(base_spike_rate * horizon_decay)

            if spike_prob > SPIKE_PROB_THRESHOLD:
                spike_scale = 1.0 + SPIKE_BOOST_FACTOR * spike_prob
                mcp = float(max(0.0, mcp * spike_scale))
                logger.debug(
                    "Applied direct spike boost: prob=%.3f scale=%.4f step=%d",
                    spike_prob,
                    spike_scale,
                    step,
                )

            blocks.append({
                "horizon": step,
                "forecast_timestamp": forecast_ts.to_pydatetime() if hasattr(forecast_ts, "to_pydatetime") else forecast_ts,
                "predicted_mcp": round(mcp, 2),
                "lower_bound": round(mcp * 0.85, 2),  # Approximate CI
                "upper_bound": round(mcp * 1.15, 2),
                "confidence": round(confidence, 4),
                "spike_probability": round(spike_prob, 4),
                "zone": classify_zone(mcp),
            })

        logger.info("Direct interpolated forecast complete: %d blocks", len(blocks))
        return blocks

    def _load_residual_profile(self) -> pd.DataFrame | None:
        profile_path = self.features_path.parent.parent / "processed" / RESIDUAL_PROFILE_FILENAME
        if not profile_path.exists():
            logger.info("Residual profile missing at %s; skipping residual reconstruction.", profile_path)
            return None

        profile_df = pd.read_parquet(profile_path)
        if not {"block_number", "is_weekend", "block_residual"}.issubset(profile_df.columns):
            logger.warning("Residual profile file %s does not contain expected columns", profile_path)
            return None

        return profile_df.set_index(["block_number", "is_weekend"])

    @staticmethod
    def _lookup_residual_profile(profile_df: pd.DataFrame, forecast_ts: pd.Timestamp) -> float:
        block_number = int((forecast_ts.hour * 60 + forecast_ts.minute) // BLOCK_MINUTES) + 1
        is_weekend = int(forecast_ts.dayofweek >= 5)
        key = (block_number, is_weekend)
        try:
            return float(profile_df.loc[key, "block_residual"])
        except (KeyError, TypeError):
            return 0.0

    @staticmethod
    def _interpolate_at_horizon(
        step: int,
        sorted_anchors: list[int],
        anchor_preds: dict[int, float],
    ) -> float:
        """Linearly interpolate between the two nearest anchor horizons."""
        if step in anchor_preds:
            return anchor_preds[step]

        # Clamp to nearest anchor if outside range
        if step <= sorted_anchors[0]:
            return anchor_preds[sorted_anchors[0]]
        if step >= sorted_anchors[-1]:
            return anchor_preds[sorted_anchors[-1]]

        # Find bracketing anchors
        lower_h = sorted_anchors[0]
        upper_h = sorted_anchors[-1]
        for i in range(len(sorted_anchors) - 1):
            if sorted_anchors[i] <= step <= sorted_anchors[i + 1]:
                lower_h = sorted_anchors[i]
                upper_h = sorted_anchors[i + 1]
                break

        # Linear interpolation
        span = upper_h - lower_h
        if span == 0:
            return anchor_preds[lower_h]

        frac = (step - lower_h) / span
        return anchor_preds[lower_h] + frac * (anchor_preds[upper_h] - anchor_preds[lower_h])

    # ── Persistence ───────────────────────────────────────────────────

    def _persist_run(
        self,
        origin_ts: pd.Timestamp | datetime,
        points: list[ForecastPoint],
        forecast_type_key: str,
    ) -> ForecastRunResult:
        """Persist forecast to PostgreSQL + CSV (latest + archive)."""
        svc = self._get_forecast_service()
        origin_dt = origin_ts.to_pydatetime() if hasattr(origin_ts, "to_pydatetime") else origin_ts

        # Persist to DB + timestamped CSV
        result = svc.save_forecast_run(
            origin_timestamp=origin_dt,
            points=points,
            model_version=self.model_version,
            feature_version=self.feature_version,
        )

        # Write forecast_latest.csv (overwrite)
        latest_path = self.output_dir / "forecast_latest.csv"
        self._write_points_csv(points, latest_path)

        # Write forecast_archive/YYYY_MM_DD.csv
        date_str = origin_dt.strftime("%Y_%m_%d")
        type_label = {"24h": "24h", "7d": "7d", "30d": "30d"}[forecast_type_key]
        archive_path = self.output_dir / "archive" / f"{date_str}_{type_label}.csv"
        self._write_points_csv(points, archive_path)

        logger.info(
            "Persisted %s forecast: run_id=%s  blocks=%d  latest=%s  archive=%s",
            type_label, result["run_id"], result["total_blocks"], latest_path, archive_path,
        )

        return ForecastRunResult(
            forecast_type=result["forecast_type"],
            run_id=result["run_id"],
            total_blocks=result["total_blocks"],
            generated_at=result["generated_at"],
            model_version=result["model_version"],
            csv_latest_path=str(latest_path),
            csv_archive_path=str(archive_path),
            zone_summary=result["zone_summary"],
        )

    @staticmethod
    def _write_points_csv(points: list[ForecastPoint], path: Path) -> None:
        """Write forecast points to a CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for p in points:
            block_num = ((p.forecast_timestamp.hour * 60 + p.forecast_timestamp.minute) // 15) + 1
            rows.append({
                "horizon": p.horizon,
                "forecast_timestamp": p.forecast_timestamp.isoformat(),
                "block_number": block_num,
                "predicted_mcp": p.mcp_forecast_rs_mwh,
                "zone": p.zone,
                "confidence": p.confidence,
                "spike_probability": p.spike_probability,
                "lower_bound": p.lower_bound,
                "upper_bound": p.upper_bound,
            })
        try:
            pd.DataFrame(rows).to_csv(path, index=False)
        except PermissionError as exc:
            logger.warning("Unable to write forecast CSV to %s: %s", path, exc)

    # ── Conversion ────────────────────────────────────────────────────

    @staticmethod
    def _blocks_to_forecast_points(blocks: list[dict[str, Any]]) -> list[ForecastPoint]:
        """Convert raw block dicts to ForecastPoint domain entities."""
        return [
            ForecastPoint(
                horizon=b["horizon"],
                forecast_timestamp=b["forecast_timestamp"],
                mcp_forecast_rs_mwh=b["predicted_mcp"],
                zone=b["zone"],
                confidence=b["confidence"],
                spike_probability=b["spike_probability"],
                lower_bound=b.get("lower_bound"),
                upper_bound=b.get("upper_bound"),
            )
            for b in blocks
        ]

    # ── Data Loading ──────────────────────────────────────────────────

    def _get_market_state(self, market_state: pd.DataFrame | None = None) -> pd.DataFrame:
        """Load market state for recursive forecasting."""
        if market_state is not None:
            df = market_state.copy()
        else:
            df = pd.read_parquet(self.master_path)

        df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], utc=True)
        if df["block_timestamp"].dt.tz is not None:
            df["block_timestamp"] = df["block_timestamp"].dt.tz_convert(TIMEZONE)
        df = df.sort_values("block_timestamp").reset_index(drop=True)

        # Ensure required columns
        if "session_id" not in df.columns:
            df["session_id"] = 1
        if "source_file" not in df.columns:
            df["source_file"] = "inference"

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

    def _get_feature_state(self) -> pd.DataFrame:
        """Load prebuilt features for direct-horizon prediction."""
        return pd.read_parquet(self.features_path)

    @staticmethod
    def _append_synthetic_block(
        history: pd.DataFrame,
        forecast_ts: pd.Timestamp,
        mcp: float,
    ) -> pd.DataFrame:
        """Append a synthetic block to history for recursive forecasting."""
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

    # ── Model Loading ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load all model artifacts."""
        if self._ensemble is None:
            self._load_ensemble()
        if not self._direct_models:
            self._load_direct_models()
        if self._spike_classifier is None:
            self._load_spike_classifier()

    def _load_ensemble(self) -> None:
        """Load the weighted ensemble model."""
        if not self.ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble not found: {self.ensemble_path}")

        import __main__
        __main__.WeightedEnsemble = WeightedEnsemble

        bundle = joblib.load(self.ensemble_path)
        self._ensemble = bundle.get("ensemble") or bundle
        if not isinstance(self._ensemble, WeightedEnsemble):
            self._ensemble = WeightedEnsemble(
                weights=bundle["weights"],
                model_paths=bundle["model_paths"],
                feature_names=bundle["feature_names"],
            )
        logger.info("Ensemble loaded: %s (weights=%s)", self.ensemble_path, self._ensemble.weights)

    def _load_direct_models(self) -> None:
        """Load all direct-horizon models from disk."""
        for h in DIRECT_ANCHORS:
            model_path = self.direct_model_dir / f"direct_h{h}.pkl"
            if model_path.exists():
                self._direct_models[h] = joblib.load(model_path)
                logger.info("Direct model loaded: h=%d from %s", h, model_path)
            else:
                logger.warning("Direct model missing: h=%d (%s)", h, model_path)

        logger.info("Loaded %d/%d direct models", len(self._direct_models), len(DIRECT_ANCHORS))

    def _load_spike_classifier(self) -> None:
        """Load the spike probability classifier."""
        if self.spike_classifier_path.exists():
            try:
                self._spike_classifier = load_spike_classifier(self.spike_classifier_path)
                logger.info("Spike classifier loaded: %s", self.spike_classifier_path)
            except Exception as e:
                logger.warning("Could not load spike classifier: %s", e)
                self._spike_classifier = None
        else:
            logger.warning("Spike classifier not found: %s", self.spike_classifier_path)

    def _get_forecast_service(self) -> ForecastService:
        """Get or create the ForecastService with DB repository."""
        if self._forecast_service is None:
            repo = PostgresForecastRepository()
            self._forecast_service = ForecastService(
                repository=repo,
                csv_backup_dir=self.output_dir,
            )
        return self._forecast_service
