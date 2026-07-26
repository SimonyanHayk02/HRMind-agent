from __future__ import annotations

from typing import Protocol

from app.ports.employee_repository import EmployeeRepository
from app.ports.resume_repository import ResumeRepository


class UnitOfWork(Protocol):
    employees: EmployeeRepository
    resumes: ResumeRepository

    async def __aenter__(self) -> UnitOfWork: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...
