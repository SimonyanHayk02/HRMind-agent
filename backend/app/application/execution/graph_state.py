from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.auth import AuthContext
from app.domain.session import EntityRef
from app.domain.tools.base import ToolResult


class GraphState(BaseModel):
    question: str
    auth: AuthContext
    node_results: dict[str, ToolResult | Any] = Field(default_factory=dict)
    degraded: bool = False
    errors: list[str] = Field(default_factory=list)
    # Session context injected for server-side entity / cohort recovery
    session_entities: list[EntityRef] = Field(default_factory=list)
    last_employee_ids: list[str] = Field(default_factory=list)
    person_bindings: dict[str, str] = Field(default_factory=dict)
