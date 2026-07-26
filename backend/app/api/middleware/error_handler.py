from __future__ import annotations

import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config.logging import get_logger
from app.domain.errors import DomainError

logger = get_logger(__name__)


def _request_id(request: Request) -> str:
    return (
        getattr(request.state, "request_id", None)
        or request.headers.get("X-Request-Id")
        or request.headers.get("x-request-id")
        or "unknown"
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        status = 400
        if exc.code == "not_found":
            status = 404
        elif exc.code == "forbidden":
            status = 403
        request_id = _request_id(request)
        logger.warning(
            "domain_error",
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
            error_code=exc.code,
            message=exc.message,
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=status,
            content={
                "error": exc.code,
                "message": exc.message,
                "request_id": request_id,
            },
            headers={"X-Request-Id": request_id},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = _request_id(request)
        logger.info(
            "http_error",
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
            status_code=exc.status_code,
            detail=str(exc.detail),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail,
                "request_id": request_id,
            },
            headers={"X-Request-Id": request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = _request_id(request)
        logger.warning(
            "request_validation_error",
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
                "request_id": request_id,
            },
            headers={"X-Request-Id": request_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        # Full traceback for Railway / container logs — search by request_id
        logger.error(
            "unhandled_error",
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
            error_type=type(exc).__name__,
            error_message=str(exc),
            client=request.client.host if request.client else None,
            exc_info=exc,
            traceback=traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "error_type": type(exc).__name__,
            },
            headers={"X-Request-Id": request_id},
        )
