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


class SessionMemory(BaseModel):
    session_id: str
    tenant_id: str
    user_id: str
    role: Role
    messages: list[ChatMessage] = Field(default_factory=list)
    summary: str = ""
    entity_memory: list[EntityRef] = Field(default_factory=list)
    constraint_memory: list[ConstraintRef] = Field(default_factory=list)
    updated_at: datetime | None = None
