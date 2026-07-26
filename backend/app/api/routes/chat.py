from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.deps import get_auth_context
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.config.logging import get_logger
from app.domain.auth import AuthContext
from app.domain.errors import DomainError

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)


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
    try:
        result = await chat_service.handle(body, auth=auth)
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
