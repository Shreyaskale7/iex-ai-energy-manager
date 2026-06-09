"""Shared dataset loading and chronological splits for MCP models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.build_features import ENGINEERED_FEATURES, MARKET_INPUT_COLUMNS, TARGET_COLUMN

NEXT_TARGET_COLUMN = "next_mcp_rs_mwh"
FEATURE_COLUMNS = MARKET_INPUT_COLUMNS + ENGINEERED_FEATURES


def load_forecast_dataset(features_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load features and build next-block MCP target."""
    df = pd.read_parquet(features_path)
    df = df.sort_values("block_timestamp").reset_index(drop=True)

    df[NEXT_TARGET_COLUMN] = df[TARGET_COLUMN].shift(-1)
    df = df.iloc[:-1].copy()

    feature_names = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(feature_names)
    if missing:
        raise ValueError(f"Feature parquet missing columns: {sorted(missing)}")

    X = df[feature_names].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1) & df[NEXT_TARGET_COLUMN].notna()
    X = X.loc[valid]
    y = df.loc[valid, NEXT_TARGET_COLUMN]
    timestamps = df.loc[valid, "block_timestamp"]
    return X, y, timestamps, feature_names


def load_multihorizon_dataset(
    features_path: Path,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Load features and build a target shifted by ``horizon`` blocks.

    Parameters
    ----------
    features_path : Path
        Path to features.parquet.
    horizon : int
        Number of blocks ahead to predict (1 = next block, 96 = 1 day, etc.).

    Returns
    -------
    (X, y, timestamps, feature_names)
    """
    df = pd.read_parquet(features_path)
    df = df.sort_values("block_timestamp").reset_index(drop=True)

    target_col = f"mcp_target_h{horizon}"
    df[target_col] = df[TARGET_COLUMN].shift(-horizon)
    df = df.iloc[:-horizon].copy()

    feature_names = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(feature_names)
    if missing:
        raise ValueError(f"Feature parquet missing columns: {sorted(missing)}")

    X = df[feature_names].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1) & df[target_col].notna()
    X = X.loc[valid]
    y = df.loc[valid, target_col]
    timestamps = df.loc[valid, "block_timestamp"]
    return X, y, timestamps, feature_names


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> dict[str, Any]:
    n = len(X)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_end = n - test_size - val_size
    val_end = n - test_size

    return {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "ts_train": timestamps.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "ts_val": timestamps.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
        "ts_test": timestamps.iloc[val_end:],
    }
