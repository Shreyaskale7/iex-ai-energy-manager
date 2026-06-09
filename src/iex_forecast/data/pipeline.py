"""End-to-end data pipeline: ingest → clean → persist."""

from pathlib import Path

import pandas as pd

from iex_forecast.config.settings import Settings
from iex_forecast.core.logging import get_logger
from iex_forecast.data.cleaner import RTMDataCleaner
from iex_forecast.data.ingestion import RTMExcelIngestor
from iex_forecast.infrastructure.repositories import PostgresRTMRepository

logger = get_logger(__name__)


class DataPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ingestor = RTMExcelIngestor(settings.data_raw_dir)
        self.cleaner = RTMDataCleaner()
        self.repository = PostgresRTMRepository()

    def run(self, persist_db: bool = True, save_parquet: bool = True) -> pd.DataFrame:
        raw = self.ingestor.read_all()
        clean = self.cleaner.clean(raw)

        if save_parquet:
            parquet_path = self.settings.data_processed_dir / "rtm_blocks.parquet"
            self.cleaner.save_processed(clean, parquet_path)

        if persist_db:
            rows = self.repository.upsert_blocks(
                clean[
                    [
                        "trade_date",
                        "hour",
                        "session_id",
                        "time_block",
                        "purchase_bid_mw",
                        "sell_bid_mw",
                        "mcv_mw",
                        "scheduled_volume_mw",
                        "mcp_rs_mwh",
                        "block_timestamp",
                        "source_file",
                    ]
                ]
            )
            logger.info("database_upsert_complete", rows=rows)

        return clean

    def load_from_db(self) -> pd.DataFrame:
        return self.repository.load_blocks()

    def load_processed_parquet(self) -> pd.DataFrame:
        path = self.settings.data_processed_dir / "rtm_blocks.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
