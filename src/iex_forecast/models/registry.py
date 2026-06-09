"""Persist and load per-horizon ensemble models."""

import json
from pathlib import Path

import joblib

from iex_forecast.core.exceptions import ModelNotFoundError
from iex_forecast.core.logging import get_logger
from iex_forecast.domain.constants import FORECAST_HORIZON
from iex_forecast.models.ensemble import HorizonEnsemble

logger = get_logger(__name__)


class ModelRegistry:
    def __init__(self, base_dir: Path, version: str = "v1") -> None:
        self.base_dir = base_dir
        self.version = version
        self.version_dir = base_dir / version
        self.version_dir.mkdir(parents=True, exist_ok=True)

    def _horizon_path(self, horizon: int) -> Path:
        return self.version_dir / f"horizon_{horizon:03d}.joblib"

    def save(self, horizon: int, ensemble: HorizonEnsemble) -> None:
        path = self._horizon_path(horizon)
        joblib.dump(ensemble, path)
        logger.debug("model_saved", horizon=horizon, path=str(path))

    def load(self, horizon: int) -> HorizonEnsemble:
        path = self._horizon_path(horizon)
        if not path.exists():
            raise ModelNotFoundError(f"No model for horizon {horizon} at {path}")
        return joblib.load(path)

    def load_all(self) -> dict[int, HorizonEnsemble]:
        models: dict[int, HorizonEnsemble] = {}
        for horizon in range(1, FORECAST_HORIZON + 1):
            path = self._horizon_path(horizon)
            if path.exists():
                models[horizon] = joblib.load(path)
        if len(models) < FORECAST_HORIZON:
            missing = FORECAST_HORIZON - len(models)
            raise ModelNotFoundError(
                f"Registry incomplete: {missing} horizons missing in {self.version_dir}"
            )
        return models

    def save_metadata(self, metadata: dict) -> None:
        path = self.version_dir / "metadata.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        logger.info("registry_metadata_saved", path=str(path))

    def load_metadata(self) -> dict:
        path = self.version_dir / "metadata.json"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def is_complete(self) -> bool:
        return all(self._horizon_path(h).exists() for h in range(1, FORECAST_HORIZON + 1))
