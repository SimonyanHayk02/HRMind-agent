from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.enums import ResumeStatus


class Resume(BaseModel):
    id: UUID
    employee_id: UUID
    storage_path: str
    content_type: str
    checksum: str
    parse_version: str = "1"
    status: ResumeStatus = ResumeStatus.PENDING
    error: str | None = None
    updated_at: datetime | None = None


class ResumeChunk(BaseModel):
    id: UUID | None = None
    employee_id: UUID
    resume_id: UUID
    section: str
    chunk_index: int
    content: str
    metadata: dict = Field(default_factory=dict)
