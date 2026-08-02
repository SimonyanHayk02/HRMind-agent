from __future__ import annotations

from typing import Any

from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.application.planning.tool_schemas import (
    CLARIFY_DESCRIPTION,
    CLARIFY_INPUT_SCHEMA,
)
from app.domain.tools.base import ToolMeta, ToolResult


class ClarifyTool:
    def __init__(self) -> None:
        self._meta = ToolMeta(
            name="clarify",
            description=CLARIFY_DESCRIPTION,
            input_schema=CLARIFY_INPUT_SCHEMA,
            permissions=list(Role),
            estimated_latency_ms=5,
        )

    @property
    def meta(self) -> ToolMeta:
        return self._meta

    async def run(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        question = params.get("question") or "Could you clarify who you mean?"
        candidates = params.get("candidates") or []
        return ToolResult(
            data={"clarify": question, "candidates": candidates},
            confidence=0.4,
        )
