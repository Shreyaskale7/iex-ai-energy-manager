"""Domain and application exceptions."""


class IEXForecastError(Exception):
    """Base exception for the forecasting system."""


class DataValidationError(IEXForecastError):
    """Raised when ingested or transformed data fails validation."""


class TrainingError(IEXForecastError):
    """Raised when model training fails."""


class ModelNotFoundError(IEXForecastError):
    """Raised when required model artifacts are missing."""


class ForecastError(IEXForecastError):
    """Raised when inference cannot produce a valid forecast."""
