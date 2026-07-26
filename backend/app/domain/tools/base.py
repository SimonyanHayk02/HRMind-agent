from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.domain.auth import AuthContext
from app.domain.enums import Role


class SourceRef(BaseModel):
    kind: str
    ref: str
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    data: Any = None
    confidence: float = 1.0
    sources: list[SourceRef] = Field(default_factory=list)
    cache_hit: bool = False
    error: str | None = None
    degraded: bool = False


class ToolMeta(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    estimated_latency_ms: int = 100
    permissions: list[Role] = Field(default_factory=lambda: list(Role))
    cache_policy: str = "none"


class Tool(Protocol):
    @property
    def meta(self) -> ToolMeta: ...

    async def run(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult: ...
