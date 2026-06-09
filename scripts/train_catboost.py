#!/usr/bin/env python3
"""Train CatBoost MCP forecaster."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.train_catboost import CatBoostMCPTrainer, configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "features" / "features.parquet")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    result = CatBoostMCPTrainer(features_path=args.features, optuna_trials=args.trials).run()
    print(f"Model: {result.model_path}")
    print(f"Test:  {result.metrics['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
