"""Application services orchestrating domain use cases."""

from iex_forecast.config.settings import Settings
from iex_forecast.core.logging import get_logger
from iex_forecast.data.pipeline import DataPipeline
from iex_forecast.domain.entities import TrainingRunMetadata
from iex_forecast.inference.predictor import RTMForecastPredictor
from iex_forecast.models.trainer import ModelTrainer

logger = get_logger(__name__)


class ForecastApplicationService:
    """Facade for ingest, train, and forecast workflows."""

    def __init__(self, settings: Settings | None = None) -> None:
        from iex_forecast.config.settings import get_settings

        self.settings = settings or get_settings()
        self.data_pipeline = DataPipeline(self.settings)
        self.trainer = ModelTrainer(self.settings)
        self.predictor = RTMForecastPredictor(self.settings)

    def ingest(self) -> int:
        df = self.data_pipeline.run(persist_db=True, save_parquet=True)
        logger.info("ingest_workflow_complete", rows=len(df))
        return len(df)

    def train(self) -> TrainingRunMetadata:
        df = self.data_pipeline.load_from_db()
        if df.empty:
            df = self.data_pipeline.load_processed_parquet()
        if df.empty:
            raise ValueError("No data available. Run ingest first.")
        return self.trainer.train(df)

    def forecast(self, persist: bool = True):
        return self.predictor.predict(persist=persist)
