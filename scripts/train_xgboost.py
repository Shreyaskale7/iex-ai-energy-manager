#!/usr/bin/env python3
"""Train production XGBoost model for next 15-minute MCP forecast."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.train_xgboost import XGBoostMCPTrainer, configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost MCP forecaster")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "features" / "features.parquet",
        help="Input features parquet",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "xgboost.pkl",
        help="Output model path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "xgb_report.html",
        help="Output HTML report path",
    )
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials")
    parser.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit folds")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    result = XGBoostMCPTrainer(
        features_path=args.features,
        model_path=args.model,
        report_path=args.report,
        optuna_trials=args.trials,
        n_splits=args.splits,
    ).run()

    print(f"Model saved:  {result.model_path}")
    print(f"HTML report: {result.report_path}")
    print(f"Metrics JSON: {result.metrics_path}")
    print("Test metrics:", result.metrics["test"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
