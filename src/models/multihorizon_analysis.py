"""Multi-horizon analysis: recursive backtesting + direct model comparison."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data.schema import BLOCK_MINUTES, BLOCKS_PER_DAY
from features.build_features import (
    MARKET_INPUT_COLUMNS,
    MCP_LAG_STEPS,
    ROLLING_WINDOWS,
    RTMFeaturePipeline,
    TARGET_COLUMN,
)
from models.dataset import chronological_split, load_forecast_dataset, load_multihorizon_dataset
from models.ensemble import WeightedEnsemble
from models.metrics import regression_metrics, smape
from models.train_direct_horizon import train_direct_horizon

logger = logging.getLogger(__name__)

HORIZONS = [1, 4, 12, 24, 48, 60, 68, 72, 76, 80, 88, 96, 192, 672, 2880]
HORIZON_LABELS = {
    1: "15 min",
    4: "1 hour",
    12: "3 hours",
    24: "6 hours",
    48: "12 hours",
    60: "15 hours",
    68: "17 hours",
    72: "18 hours",
    76: "19 hours",
    80: "20 hours",
    88: "22 hours",
    96: "1 day",
    192: "2 days",
    672: "7 days",
    2880: "30 days",
}

# Cap recursive backtesting at 96 steps (1 day) for speed
RECURSIVE_MAX_HORIZON = 96
MIN_HISTORY = max(max(MCP_LAG_STEPS), max(ROLLING_WINDOWS)) + 1


def classify_zone(mcp: float) -> str:
    if mcp < 3000:
        return "GREEN"
    if mcp < 6000:
        return "YELLOW"
    return "RED"


# ═══════════════════════════════════════════════════════════════════════
#  RECURSIVE BACKTESTING
# ═══════════════════════════════════════════════════════════════════════

def recursive_backtest(
    master_path: Path,
    ensemble_path: Path,
    n_origins: int = 5,
    max_horizon: int = RECURSIVE_MAX_HORIZON,
) -> dict[str, Any]:
    """
    Run recursive forecasting from multiple origin points and measure
    error accumulation at each horizon checkpoint.
    """
    logger.info("Loading ensemble and market data for recursive backtest...")
    bundle = joblib.load(ensemble_path)
    ensemble = bundle.get("ensemble") or bundle
    if not isinstance(ensemble, WeightedEnsemble):
        ensemble = WeightedEnsemble(
            weights=bundle["weights"],
            model_paths=bundle["model_paths"],
            feature_names=bundle["feature_names"],
        )

    df = pd.read_parquet(master_path)
    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], utc=True)
    df = df.sort_values("block_timestamp").reset_index(drop=True)

    # Use the last 20% of data as test territory
    test_start_idx = int(len(df) * 0.80)
    test_df = df.iloc[test_start_idx:].reset_index(drop=True)

    # Pick origin points spaced evenly across test data (need room for max_horizon after)
    usable_len = len(test_df) - max_horizon
    if usable_len < n_origins:
        n_origins = max(1, usable_len)
    spacing = usable_len // n_origins
    origin_indices = [i * spacing for i in range(n_origins)]

    pipeline = RTMFeaturePipeline(drop_warmup_rows=False)
    checkpoints = [h for h in HORIZONS if h <= max_horizon]

    all_results: list[dict] = []

    for oi, origin_idx in enumerate(origin_indices):
        # Need MIN_HISTORY blocks before the origin
        global_origin_idx = test_start_idx + origin_idx
        history_start = max(0, global_origin_idx - MIN_HISTORY)
        history = df.iloc[history_start:global_origin_idx + 1].copy()

        origin_ts = pd.to_datetime(history["block_timestamp"].iloc[-1])
        logger.info("Recursive backtest origin %d/%d: %s", oi + 1, n_origins, origin_ts)

        # Ground truth
        actuals = df.iloc[global_origin_idx + 1: global_origin_idx + 1 + max_horizon]

        working = history.copy()
        predictions = []

        for step in range(1, max_horizon + 1):
            try:
                featured = pipeline.build_features(working)
                feature_cols = list(ensemble.feature_names)
                row = featured.iloc[[-1]][feature_cols]

                if row.isna().any().any():
                    predictions.append(np.nan)
                else:
                    pred = ensemble.predict(row)[0]
                    predictions.append(float(pred))

                    # Append synthetic block
                    forecast_ts = origin_ts + timedelta(minutes=BLOCK_MINUTES * step)
                    last = working.iloc[-1]
                    new_row = {
                        "block_timestamp": forecast_ts,
                        "trade_date": forecast_ts.date() if hasattr(forecast_ts, 'date') else forecast_ts,
                        TARGET_COLUMN: float(pred),
                        "session_id": last.get("session_id", 1),
                        "source_file": "backtest",
                    }
                    for col in MARKET_INPUT_COLUMNS:
                        new_row[col] = float(last[col])
                    # Add hour/time_block for feature pipeline
                    minutes_val = forecast_ts.hour * 60 + forecast_ts.minute
                    new_row["hour"] = int(minutes_val // 60 + 1)
                    new_row["time_block"] = int((minutes_val % 60) // BLOCK_MINUTES + 1)
                    new_row["daily_block_index"] = (new_row["hour"] - 1) * 4 + new_row["time_block"]

                    working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)
            except Exception as e:
                logger.warning("Recursive step %d failed: %s", step, e)
                predictions.append(np.nan)

        # Evaluate at checkpoints
        for h in checkpoints:
            if h > len(predictions) or h > len(actuals):
                continue
            pred_val = predictions[h - 1]
            if np.isnan(pred_val):
                continue
            actual_val = float(actuals.iloc[h - 1][TARGET_COLUMN])
            all_results.append({
                "origin": oi,
                "horizon": h,
                "predicted": pred_val,
                "actual": actual_val,
                "abs_error": abs(pred_val - actual_val),
            })

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        return {"metrics_by_horizon": {}, "raw": results_df}

    # Aggregate by horizon
    metrics_by_horizon: dict[int, dict[str, float]] = {}
    for h in checkpoints:
        subset = results_df[results_df["horizon"] == h]
        if subset.empty:
            continue
        y_true = subset["actual"].values
        y_pred = subset["predicted"].values
        m = regression_metrics(y_true, y_pred)
        m["smape"] = smape(y_true, y_pred)
        m["zone_accuracy"] = float(np.mean([
            classify_zone(p) == classify_zone(a)
            for p, a in zip(y_pred, y_true)
        ])) * 100.0
        m["n_samples"] = len(subset)
        metrics_by_horizon[h] = m

    return {"metrics_by_horizon": metrics_by_horizon, "raw": results_df}


# ═══════════════════════════════════════════════════════════════════════
#  DIRECT MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def train_all_direct_models(
    features_path: Path,
    output_dir: Path = Path("models"),
    horizons: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Train direct LightGBM models for each horizon and return metrics."""
    horizons = horizons or HORIZONS
    results: dict[int, dict[str, Any]] = {}

    for h in horizons:
        logger.info("Training direct model for horizon h=%d (%s)...", h, HORIZON_LABELS.get(h, ""))
        try:
            result = train_direct_horizon(
                features_path=features_path,
                horizon=h,
                output_dir=output_dir,
            )
            results[h] = result
        except Exception as e:
            logger.error("Failed to train h=%d: %s", h, e)
            results[h] = {"horizon": h, "error": str(e), "test_metrics": None}

    return results


# ═══════════════════════════════════════════════════════════════════════
#  COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def compare_recursive_vs_direct(
    recursive_metrics: dict[int, dict[str, float]],
    direct_results: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    """Build a comparison table of recursive vs direct metrics at each horizon."""
    rows = []
    for h in HORIZONS:
        row = {
            "horizon": h,
            "label": HORIZON_LABELS.get(h, f"t+{h}"),
        }

        # Recursive
        if h in recursive_metrics:
            rm = recursive_metrics[h]
            row["recursive_mae"] = rm["mae"]
            row["recursive_rmse"] = rm["rmse"]
            row["recursive_r2"] = rm["r2"]
            row["recursive_smape"] = rm.get("smape", np.nan)
            row["recursive_zone_acc"] = rm.get("zone_accuracy", np.nan)
        else:
            row["recursive_mae"] = np.nan
            row["recursive_rmse"] = np.nan
            row["recursive_r2"] = np.nan
            row["recursive_smape"] = np.nan
            row["recursive_zone_acc"] = np.nan

        # Direct
        if h in direct_results and direct_results[h].get("test_metrics"):
            dm = direct_results[h]["test_metrics"]
            row["direct_mae"] = dm["mae"]
            row["direct_rmse"] = dm["rmse"]
            row["direct_r2"] = dm["r2"]
            row["direct_smape"] = dm.get("smape", np.nan)
        else:
            row["direct_mae"] = np.nan
            row["direct_rmse"] = np.nan
            row["direct_r2"] = np.nan
            row["direct_smape"] = np.nan

        # Winner
        if not np.isnan(row.get("recursive_rmse", np.nan)) and not np.isnan(row.get("direct_rmse", np.nan)):
            row["winner"] = "recursive" if row["recursive_rmse"] < row["direct_rmse"] else "direct"
        elif not np.isnan(row.get("direct_rmse", np.nan)):
            row["winner"] = "direct"
        elif not np.isnan(row.get("recursive_rmse", np.nan)):
            row["winner"] = "recursive"
        else:
            row["winner"] = "—"

        rows.append(row)

    return pd.DataFrame(rows)


def generate_architecture_recommendation(comparison_df: pd.DataFrame) -> dict[str, Any]:
    """Analyze comparison data and produce an architecture recommendation."""
    rec = {
        "short_term": {"strategy": "recursive", "horizons": "t+1 to t+96"},
        "medium_term": {"strategy": "direct", "horizons": "t+96 to t+672"},
        "long_term": {"strategy": "direct", "horizons": "t+672 to t+2880"},
    }

    # Find crossover point
    crossover = None
    for _, row in comparison_df.iterrows():
        if row["winner"] == "direct" and crossover is None:
            crossover = int(row["horizon"])

    rec["crossover_horizon"] = crossover

    # R² decay analysis
    direct_rows = comparison_df.dropna(subset=["direct_r2"])
    r2_below_90 = direct_rows[direct_rows["direct_r2"] < 0.90]
    r2_below_80 = direct_rows[direct_rows["direct_r2"] < 0.80]

    rec["r2_drops_below_90_at"] = int(r2_below_90.iloc[0]["horizon"]) if not r2_below_90.empty else None
    rec["r2_drops_below_80_at"] = int(r2_below_80.iloc[0]["horizon"]) if not r2_below_80.empty else None

    # Overall recommendation
    if crossover and crossover <= 96:
        rec["summary"] = (
            "Hybrid architecture recommended: Use recursive ensemble for short-term "
            f"(t+1 to t+{crossover}), direct models for medium and long-term."
        )
    else:
        rec["summary"] = (
            "Recursive ensemble dominates at all tested short-term horizons. "
            "Use direct models for long-term (t+192+) where recursive error accumulation "
            "makes it unreliable."
        )

    return rec
