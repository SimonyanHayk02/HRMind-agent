from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.error_handler import register_exception_handlers
from app.api.middleware.request_id import RequestIdMiddleware
from app.api.routes import chat, health, ingest
from app.composition import build_container
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    container = await build_container(settings)
    catalog = container.extras.get("catalog_service")
    if catalog is not None:
        try:
            await catalog.warm()
            logger.info("schema_catalog_warmed")
        except Exception as exc:
            logger.warning("schema_catalog_warm_failed", error=str(exc))
    app.state.container = container
    logger.info("app_started", env=settings.app_env)
    try:
        yield
    finally:
        await container.aclose()
        logger.info("app_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="HRMind HR AI Agent", version="0.1.0", lifespan=lifespan)
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(chat.router, prefix="/v1")
    app.include_router(ingest.router, prefix="/v1")
    return app


app = create_app()
