from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from uuid import UUID

from app.domain.employee import Employee
from app.domain.session import EntityRef, SessionMemory
from app.ports.employee_repository import EmployeeRepository


@dataclass
class ResolveResult:
    candidates: list[Employee]
    confidence: float
    selected: Employee | None = None


class EntityResolver:
    def __init__(self, employees: EmployeeRepository) -> None:
        self._employees = employees

    async def resolve(self, name: str, memory: SessionMemory | None = None) -> ResolveResult:
        # Prefer entity memory aliases
        if memory:
            for ent in memory.entity_memory:
                aliases = [ent.display_name.lower(), *[a.lower() for a in ent.aliases]]
                if name.lower() in aliases or name.lower() == ent.display_name.lower():
                    emp = await self._employees.get_by_id(ent.employee_id)
                    if emp:
                        return ResolveResult(candidates=[emp], confidence=ent.confidence, selected=emp)

        matches = await self._employees.search_by_name(name, limit=5)
        if not matches:
            return ResolveResult(candidates=[], confidence=0.0)
        if len(matches) == 1:
            return ResolveResult(candidates=matches, confidence=0.95, selected=matches[0])
        return ResolveResult(candidates=matches, confidence=0.4, selected=None)
