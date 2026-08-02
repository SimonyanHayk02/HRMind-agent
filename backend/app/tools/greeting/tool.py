from __future__ import annotations

from typing import Any

from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.application.planning.tool_schemas import (
    GREETING_DESCRIPTION,
    GREETING_INPUT_SCHEMA,
)
from app.domain.tools.base import ToolMeta, ToolResult
from app.tools.greeting.replies import render_social_reply


class GreetingTool:
    def __init__(self) -> None:
        self._meta = ToolMeta(
            name="greeting",
            description=GREETING_DESCRIPTION,
            input_schema=GREETING_INPUT_SCHEMA,
            estimated_latency_ms=5,
            permissions=list(Role),
            cache_policy="none",
        )

    @property
    def meta(self) -> ToolMeta:
        return self._meta

    async def run(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        text = (params.get("message") or params.get("question") or "").strip()
        answer, intent = render_social_reply(text, role=auth.role)
        return ToolResult(
            data={"answer": answer, "intent": intent.value},
            confidence=1.0,
        )
