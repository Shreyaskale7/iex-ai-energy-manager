"""Train per-horizon gradient boosting ensembles."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from iex_forecast.config.settings import Settings
from iex_forecast.core.exceptions import TrainingError
from iex_forecast.core.logging import get_logger
from iex_forecast.domain.constants import FORECAST_HORIZON
from iex_forecast.domain.entities import TrainingRunMetadata
from iex_forecast.features.training_matrix import TrainingMatrixBuilder
from iex_forecast.infrastructure.repositories import PostgresTrainingRunRepository
from iex_forecast.models.ensemble import HorizonEnsemble
from iex_forecast.models.registry import ModelRegistry

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.matrix_builder = TrainingMatrixBuilder()
        self.registry = ModelRegistry(settings.model_registry_dir, version="v1")
        self.training_repo = PostgresTrainingRunRepository()

    def train(self, df: pd.DataFrame) -> TrainingRunMetadata:
        if len(df) < self.settings.min_training_rows:
            raise TrainingError(
                f"Insufficient rows ({len(df)}) for training; "
                f"minimum is {self.settings.min_training_rows}"
            )

        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        train_df, test_df = self.matrix_builder.temporal_split(
            df, self.settings.train_test_split_date
        )

        weights = self.settings.parsed_ensemble_weights
        horizon_metrics: dict[int, dict[str, float]] = {}
        train_rows = 0
        test_rows = 0

        # Load tuned parameters if available
        model_params = {}
        params_path = Path("config/model_params.json")
        if params_path.exists():
            try:
                with open(params_path, "r") as f:
                    model_params = json.load(f)
                logger.info("Loaded tuned model parameters from config/model_params.json")
            except Exception as e:
                logger.warning(f"Failed to load model params: {e}")

        logger.info(
            "training_started",
            run_id=run_id,
            train_blocks=len(train_df),
            test_blocks=len(test_df),
        )

        for horizon in range(1, FORECAST_HORIZON + 1):
            X_train, y_train = self.matrix_builder.build_for_horizon(train_df, horizon)
            X_test, y_test = self.matrix_builder.build_for_horizon(test_df, horizon)

            if X_train.empty or X_test.empty:
                raise TrainingError(f"Empty training matrix at horizon {horizon}")

            train_rows = max(train_rows, len(X_train))
            test_rows = max(test_rows, len(X_test))

            ensemble = HorizonEnsemble(weights, model_params=model_params).fit(X_train, y_train)
            self.registry.save(horizon, ensemble)

            preds = ensemble.predict(X_test)
            metrics = self.matrix_builder.compute_metrics(
                y_test.to_numpy(), preds
            )
            horizon_metrics[horizon] = metrics

            if horizon % 12 == 0:
                logger.info("training_progress", horizon=horizon, mae=metrics["mae"])

        mean_mae = sum(m["mae"] for m in horizon_metrics.values()) / len(horizon_metrics)
        mean_rmse = sum(m["rmse"] for m in horizon_metrics.values()) / len(horizon_metrics)
        finished_at = datetime.now(timezone.utc)

        metadata = TrainingRunMetadata(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            horizons_trained=FORECAST_HORIZON,
            train_rows=train_rows,
            test_rows=test_rows,
            mean_mae=mean_mae,
            mean_rmse=mean_rmse,
            model_version=self.registry.version,
            status="completed",
        )

        self.registry.save_metadata(
            {
                "run_id": run_id,
                "mean_mae": mean_mae,
                "mean_rmse": mean_rmse,
                "horizon_metrics": horizon_metrics,
            }
        )

        self.training_repo.save(
            {
                "id": run_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "horizons_trained": FORECAST_HORIZON,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "mean_mae": mean_mae,
                "mean_rmse": mean_rmse,
                "model_version": self.registry.version,
                "status": "completed",
                "metrics_json": json.dumps(horizon_metrics),
            }
        )

        logger.info(
            "training_complete",
            run_id=run_id,
            mean_mae=mean_mae,
            mean_rmse=mean_rmse,
        )
        return metadata
