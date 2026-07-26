from __future__ import annotations

from typing import Any

from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.tools.base import ToolMeta, ToolResult


class GreetingTool:
    def __init__(self) -> None:
        self._meta = ToolMeta(
            name="greeting",
            description="Respond to hello/bye/thanks without LLM",
            estimated_latency_ms=5,
            permissions=list(Role),
            cache_policy="none",
        )

    @property
    def meta(self) -> ToolMeta:
        return self._meta

    async def run(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        text = (params.get("message") or params.get("question") or "").lower().strip()
        if any(x in text for x in ("bye", "goodbye", "see you")):
            answer = "Goodbye! Feel free to ask if you need anything else."
        elif any(x in text for x in ("thank", "thanks")):
            answer = "You're welcome!"
        else:
            answer = "Hello! I can help with employee data, resumes, and HR analytics."
        return ToolResult(data={"answer": answer}, confidence=1.0)
