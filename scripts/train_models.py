#!/usr/bin/env python3
"""CLI: train ensemble models and direct-horizon models."""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iex_forecast.application.services import ForecastApplicationService
from iex_forecast.core.logging import configure_logging, get_logger
from models.multihorizon_analysis import train_all_direct_models
from iex_forecast.application.forecast_engine import DIRECT_ANCHORS

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train IEX Forecast Models")
    parser.add_argument("--direct-only", action="store_true", help="Train only direct horizon models")
    args = parser.parse_args()

    configure_logging()
    
    if args.direct_only:
        logger.info("Training ONLY direct horizon models for DIRECT_ANCHORS...")
        features_path = ROOT / "data" / "features" / "features.parquet"
        output_dir = ROOT / "models" / "direct"
        results = train_all_direct_models(
            features_path=features_path,
            output_dir=output_dir,
            horizons=DIRECT_ANCHORS
        )
        logger.info("Finished training %d direct horizon models", len(results))
        return 0

    service = ForecastApplicationService()
    metadata = service.train()
    logger.info(
        "cli_train_success",
        run_id=metadata.run_id,
        mean_mae=metadata.mean_mae,
        mean_rmse=metadata.mean_rmse,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
