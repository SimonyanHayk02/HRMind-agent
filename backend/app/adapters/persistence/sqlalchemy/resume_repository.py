from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.persistence.sqlalchemy.mappers import resume_to_domain
from app.adapters.persistence.sqlalchemy.models import ResumeModel
from app.domain.resume import Resume


class SqlAlchemyResumeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_employee_id(self, employee_id: uuid.UUID) -> Resume | None:
        result = await self._session.execute(
            select(ResumeModel).where(ResumeModel.employee_id == employee_id)
        )
        row = result.scalar_one_or_none()
        return resume_to_domain(row) if row else None

    async def list_pending(self, *, limit: int = 200) -> list[Resume]:
        result = await self._session.execute(
            select(ResumeModel).where(ResumeModel.status == "pending").limit(limit)
        )
        return [resume_to_domain(r) for r in result.scalars().all()]

    async def save(self, resume: Resume) -> Resume:
        row = await self._session.get(ResumeModel, resume.id)
        if row is None:
            row = ResumeModel(
                id=resume.id,
                employee_id=resume.employee_id,
                storage_path=resume.storage_path,
                content_type=resume.content_type,
                checksum=resume.checksum,
                parse_version=resume.parse_version,
                status=resume.status.value,
                error=resume.error,
            )
            self._session.add(row)
        else:
            row.storage_path = resume.storage_path
            row.content_type = resume.content_type
            row.checksum = resume.checksum
            row.parse_version = resume.parse_version
            row.status = resume.status.value
            row.error = resume.error
        await self._session.flush()
        return resume_to_domain(row)

    async def update_status(self, resume_id: uuid.UUID, status: str, *, error: str | None = None) -> None:
        row = await self._session.get(ResumeModel, resume_id)
        if row is None:
            return
        row.status = status
        row.error = error
        await self._session.flush()
