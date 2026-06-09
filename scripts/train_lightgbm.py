#!/usr/bin/env python3
"""Train LightGBM MCP forecaster and compare against XGBoost."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.train_lightgbm import LightGBMMCPTrainer, configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM MCP forecaster")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "features" / "features.parquet",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "lightgbm.pkl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "lgbm_report.html",
    )
    parser.add_argument(
        "--comparison-report",
        type=Path,
        default=ROOT / "reports" / "model_comparison_report.html",
    )
    parser.add_argument(
        "--xgb-model",
        type=Path,
        default=ROOT / "models" / "xgboost.pkl",
        help="XGBoost model for comparison (train XGB first)",
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--splits", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    result = LightGBMMCPTrainer(
        features_path=args.features,
        model_path=args.model,
        report_path=args.report,
        comparison_report_path=args.comparison_report,
        xgb_model_path=args.xgb_model,
        optuna_trials=args.trials,
        n_splits=args.splits,
    ).run()

    print(f"Model saved:       {result.model_path}")
    print(f"LightGBM report:   {result.report_path}")
    print(f"Comparison report: {result.comparison_report_path}")
    print(f"Test metrics:      {result.metrics['test']}")
    if result.xgb_comparison:
        winner = result.xgb_comparison["comparison"]["overall_winner"]
        print(f"Overall winner:    {winner}")
    else:
        print("Comparison skipped — train XGBoost first: python scripts/train_xgboost.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
