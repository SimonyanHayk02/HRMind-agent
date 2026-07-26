from __future__ import annotations

async def ingest_resume_task(ctx, employee_id: str) -> dict:
    # Placeholder worker entry; full wiring uses composition container in production
    return {"employee_id": employee_id, "status": "accepted"}
