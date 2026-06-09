#!/usr/bin/env python3
"""CLI: ingest monthly RTM Excel files into PostgreSQL."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iex_forecast.application.services import ForecastApplicationService
from iex_forecast.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    configure_logging()
    service = ForecastApplicationService()
    rows = service.ingest()
    logger.info("cli_ingest_success", rows=rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
