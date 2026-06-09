"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "iex-rtm-forecast"
    app_env: str = "development"
    log_level: str = "INFO"
    log_json: bool = False

    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    model_registry_dir: Path = Path("data/models")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "iex_rtm"
    postgres_user: str = "iex_user"
    postgres_password: str = "change_me"
    database_url: str | None = None

    forecast_horizon_blocks: int = 1440
    block_minutes: int = 15
    forecast_csv_backup_dir: Path = Path("forecasts")
    train_test_split_date: str = "2025-06-01"
    min_training_rows: int = 5000
    ensemble_weights: str = "xgboost:0.35,lightgbm:0.35,catboost:0.30"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2
    api_key: str = "dev-api-key"

    streamlit_server_port: int = 8501
    forecast_api_base_url: str = "http://localhost:8000"

    @field_validator("data_raw_dir", "data_processed_dir", "model_registry_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def parsed_ensemble_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for part in self.ensemble_weights.split(","):
            name, value = part.strip().split(":")
            weights[name.strip()] = float(value.strip())
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def ensure_directories(self) -> None:
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
