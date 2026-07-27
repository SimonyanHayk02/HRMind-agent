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
    dimension: str | None = None
    values: list[str] = Field(default_factory=list)


class ActiveReferent(BaseModel):
    """Explicit active 'them' cohort with provenance for follow-ups."""

    type: Literal["employee_cohort"] = "employee_cohort"
    ids: list[str] = Field(default_factory=list)
    label: str | None = None
    source_turn: int | None = None
    confidence: float = 1.0


class ToolFact(BaseModel):
    """Compact cached tool fact — never stores raw CV text."""

    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int = 180


class SessionMemory(BaseModel):
    session_id: str
    tenant_id: str
    user_id: str
    role: Role
    messages: list[ChatMessage] = Field(default_factory=list)
    summary: str = ""
    entity_memory: list[EntityRef] = Field(default_factory=list)
    constraint_memory: list[ConstraintRef] = Field(default_factory=list)
    last_employee_ids: list[str] = Field(default_factory=list)
    last_focus: LastFocus | None = None
    active_referent: ActiveReferent | None = None
    named_sets: dict[str, list[str]] = Field(default_factory=dict)
    person_bindings: dict[str, str] = Field(default_factory=dict)
    tool_fact_cache: dict[str, ToolFact] = Field(default_factory=dict)
    updated_at: datetime | None = None
