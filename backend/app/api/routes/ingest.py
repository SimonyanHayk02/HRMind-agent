from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["ingest"])


class IngestResponse(BaseModel):
    employee_id: UUID
    status: str
    job_id: str | None = None


@router.post("/ingest/resumes/{employee_id}", response_model=IngestResponse)
async def ingest_resume(employee_id: UUID, request: Request) -> IngestResponse:
    container = request.app.state.container
    ingest_service = container.extras.get("ingest_service")
    if ingest_service is None:
        return IngestResponse(employee_id=employee_id, status="queued_stub", job_id=None)
    job_id = await ingest_service.enqueue(employee_id)
    return IngestResponse(employee_id=employee_id, status="queued", job_id=job_id)
