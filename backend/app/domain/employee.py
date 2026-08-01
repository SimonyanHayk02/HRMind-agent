from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from app.domain.enums import EmploymentStatus


class Employee(BaseModel):
    id: UUID
    tenant_id: UUID
    first_name: str
    last_name: str
    email: str
    department: str
    position: str
    salary: Decimal | None = None
    hire_date: date
    manager_id: UUID | None = None
    education: str | None = None
    employment_status: EmploymentStatus = EmploymentStatus.ACTIVE
    # Agent boolean flag (not employment_status; not resumes ingest status).
    status: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def public_dict(self, *, include_salary: bool = False) -> dict:
        data = self.model_dump(mode="json")
        if not include_salary:
            data.pop("salary", None)
        return data
