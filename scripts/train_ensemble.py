#!/usr/bin/env python3
"""Build weighted ensemble from XGBoost, LightGBM, and CatBoost."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.ensemble import EnsembleTrainer, configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train weighted MCP ensemble")
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "features" / "features.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ensemble.pkl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "ensemble_comparison_report.html")
    parser.add_argument("--metric", choices=["rmse", "mae", "mape"], default="rmse")
    args = parser.parse_args()

    result = EnsembleTrainer(
        features_path=args.features,
        model_path=args.output,
        report_path=args.report,
        optimization_metric=args.metric,
    ).run()

    print(f"Ensemble saved: {result.model_path}")
    print(f"Weights:        {result.weights}")
    print(f"Report:         {result.report_path}")
    print(f"Test metrics:   {result.metrics['test']['ensemble']}")
    print(f"Overall winner: {result.overall_winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
