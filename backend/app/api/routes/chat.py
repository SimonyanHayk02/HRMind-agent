from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_auth_context
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.domain.auth import AuthContext

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
) -> ChatResponse:
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
    return await chat_service.handle(body, auth=auth)
