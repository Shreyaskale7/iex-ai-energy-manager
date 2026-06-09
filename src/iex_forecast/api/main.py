"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from iex_forecast.api.routes import router
from iex_forecast.config.settings import get_settings
from iex_forecast.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    settings.ensure_directories()
    logger.info("api_startup", app=settings.app_name, env=settings.app_env)
    yield
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="IEX RTM MCP Forecast API",
        description=(
            "30-day (2880-block) RTM electricity price forecasting with "
            "zone classification, confidence intervals, and accuracy tracking."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
