from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.deps import get_auth_context
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.config.logging import get_logger
from app.config.settings import Settings, get_settings
from app.domain.auth import AuthContext
from app.domain.errors import DomainError

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)


def _truthy_header(request: Request, name: str) -> bool:
    raw = (request.headers.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _request_settings(request: Request) -> Settings:
    container = getattr(request.app.state, "container", None)
    if container is not None and getattr(container, "settings", None) is not None:
        return container.settings
    return get_settings()


@router.post("/chat", response_model=None)
async def chat(
    body: ChatRequest,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
) -> ChatResponse | JSONResponse:
    container = request.app.state.container
    chat_service = container.extras.get("chat_service")
    if chat_service is None:
        return ChatResponse(
            session_id=body.session_id or "pending",
            answer="Chat service is not fully wired yet.",
            confidence=0.0,
            sources=[],
            degraded=True,
        )
    request_id = getattr(request.state, "request_id", None) or request.headers.get(
        "X-Request-Id", "unknown"
    )
    settings = _request_settings(request)
    debug_meta = bool(settings.chat_debug_meta) or _truthy_header(
        request, "X-HRMind-Debug"
    )
    force_repair = _truthy_header(request, "X-HRMind-Force-Repair") and (
        settings.app_env.lower() in {"development", "dev", "test", "local"}
    )
    try:
        result = await chat_service.handle(
            body,
            auth=auth,
            debug_meta=debug_meta,
            force_repair=force_repair,
        )
        logger.info(
            "chat_ok",
            request_id=request_id,
            session_id=result.session_id,
            trace_id=result.trace_id,
            degraded=result.degraded,
            question=body.question[:200],
        )
        return result
    except DomainError:
        raise
    except Exception as exc:
        logger.error(
            "chat_handler_failed",
            request_id=request_id,
            session_id=body.session_id,
            user_id=auth.user_id,
            role=getattr(auth.role, "value", str(auth.role)),
            question=body.question[:300],
            error_type=type(exc).__name__,
            error_message=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "error_type": type(exc).__name__,
            },
            headers={"X-Request-Id": str(request_id)},
        )
