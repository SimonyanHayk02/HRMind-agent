from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.enums import Role


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime | None = None


class EntityRef(BaseModel):
    employee_id: UUID
    display_name: str
    confidence: float = 1.0
    aliases: list[str] = Field(default_factory=list)


class ConstraintRef(BaseModel):
    field: str
    op: str
    value: Any


class LastFocus(BaseModel):
    """What the last useful answer was about — drives short follow-ups like 'names please'."""

    kind: Literal["facet", "cohort"] = "cohort"
    # Facet dimension (country/city/department) or "employees" for a people cohort.
    dimension: str | None = None
    values: list[str] = Field(default_factory=list)


class SessionMemory(BaseModel):
    session_id: str
    tenant_id: str
    user_id: str
    role: Role
    messages: list[ChatMessage] = Field(default_factory=list)
    summary: str = ""
    entity_memory: list[EntityRef] = Field(default_factory=list)
    constraint_memory: list[ConstraintRef] = Field(default_factory=list)
    # Last employee ID set from resume/SQL tools — used for follow-ups like "say their names".
    last_employee_ids: list[str] = Field(default_factory=list)
    last_focus: LastFocus | None = None
    updated_at: datetime | None = None
