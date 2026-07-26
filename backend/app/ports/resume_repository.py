from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.domain.resume import Resume


class ResumeRepository(Protocol):
    async def get_by_employee_id(self, employee_id: UUID) -> Resume | None: ...

    async def list_pending(self, *, limit: int = 200) -> list[Resume]: ...

    async def save(self, resume: Resume) -> Resume: ...

    async def update_status(self, resume_id: UUID, status: str, *, error: str | None = None) -> None: ...
