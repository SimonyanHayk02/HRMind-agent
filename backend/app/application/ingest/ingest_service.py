from __future__ import annotations

from uuid import UUID, uuid4

from app.ingest.pipeline import IngestPipeline


class IngestService:
    def __init__(self, pipeline: IngestPipeline, redis=None) -> None:
        self._pipeline = pipeline
        self._redis = redis

    async def enqueue(self, employee_id: UUID) -> str:
        job_id = str(uuid4())
        # Sync path for v1 when worker unavailable; ARQ can call process later
        return job_id

    async def process(
        self,
        *,
        path: str,
        employee_id: str,
        employee_name: str,
        resume_id: str,
        position: str | None = None,
        department: str | None = None,
    ) -> int:
        from pathlib import Path

        return await self._pipeline.ingest_file(
            Path(path),
            employee_id=employee_id,
            employee_name=employee_name,
            resume_id=resume_id,
            position=position,
            department=department,
        )
