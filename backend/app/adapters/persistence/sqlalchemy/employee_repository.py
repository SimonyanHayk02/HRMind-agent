from __future__ import annotations

import uuid

from sqlalchemy import Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.persistence.sqlalchemy.mappers import employee_to_domain
from app.adapters.persistence.sqlalchemy.models import EmployeeModel
from app.domain.employee import Employee


class SqlAlchemyEmployeeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, employee_id: uuid.UUID) -> Employee | None:
        row = await self._session.get(EmployeeModel, employee_id)
        return employee_to_domain(row) if row else None

    async def get_by_email(self, email: str) -> Employee | None:
        result = await self._session.execute(select(EmployeeModel).where(EmployeeModel.email == email))
        row = result.scalar_one_or_none()
        return employee_to_domain(row) if row else None

    async def search_by_name(self, query: str, *, limit: int = 10) -> list[Employee]:
        q = f"%{query.lower()}%"
        stmt: Select[tuple[EmployeeModel]] = (
            select(EmployeeModel)
            .where(
                or_(
                    func.lower(EmployeeModel.first_name).like(q),
                    func.lower(EmployeeModel.last_name).like(q),
                    func.lower(EmployeeModel.first_name + " " + EmployeeModel.last_name).like(q),
                )
            )
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [employee_to_domain(r) for r in result.scalars().all()]

    async def list_by_department(self, department: str) -> list[Employee]:
        result = await self._session.execute(
            select(EmployeeModel).where(EmployeeModel.department == department)
        )
        return [employee_to_domain(r) for r in result.scalars().all()]

    async def get_manager_chain(self, employee_id: uuid.UUID, *, max_depth: int = 5) -> list[Employee]:
        chain: list[Employee] = []
        current = await self.get_by_id(employee_id)
        depth = 0
        while current and current.manager_id and depth < max_depth:
            manager = await self.get_by_id(current.manager_id)
            if manager is None:
                break
            chain.append(manager)
            current = manager
            depth += 1
        return chain

    async def list_all(self, *, limit: int = 500) -> list[Employee]:
        result = await self._session.execute(select(EmployeeModel).limit(limit))
        return [employee_to_domain(r) for r in result.scalars().all()]
