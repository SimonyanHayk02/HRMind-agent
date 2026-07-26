from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.auth import AuthContext
from app.domain.tools.base import ToolResult


class GraphState(BaseModel):
    question: str
    auth: AuthContext
    node_results: dict[str, ToolResult | Any] = Field(default_factory=dict)
    degraded: bool = False
    errors: list[str] = Field(default_factory=list)
