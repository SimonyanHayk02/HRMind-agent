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
    def __init__(self, employees: EmployeeRepository | None = None) -> None:
        self._employees = employees

    def resolve_from_refs(
        self, name: str, entities: Sequence[EntityRef]
    ) -> tuple[UUID | None, float]:
        """Resolve a name against session entity memory without a DB round-trip.

        Returns a hit only when exactly one remembered person matches — avoids
        picking the wrong Katya when several people share a first name.
        """
        needle = (name or "").strip().lower()
        if not needle or not entities:
            return None, 0.0
        hits: list[tuple[UUID, float]] = []
        seen: set[UUID] = set()
        for ent in entities:
            aliases = [ent.display_name.lower(), *[a.lower() for a in ent.aliases]]
            matched = False
            conf = ent.confidence
            if needle in aliases or any(
                a == needle or needle in a.split() or a in needle.split()
                for a in aliases
            ):
                matched = True
            else:
                for a in aliases:
                    tokens = a.split()
                    if needle in tokens or any(
                        t.startswith(needle) for t in tokens if len(needle) > 2
                    ):
                        matched = True
                        conf = max(0.7, ent.confidence * 0.9)
                        break
            if matched and ent.employee_id not in seen:
                seen.add(ent.employee_id)
                hits.append((ent.employee_id, conf))
        if len(hits) == 1:
            return hits[0]
        return None, 0.0

    async def resolve(
        self, name: str, memory: SessionMemory | None = None
    ) -> ResolveResult:
        if memory:
            eid, conf = self.resolve_from_refs(name, memory.entity_memory)
            if eid is not None and self._employees is not None:
                emp = await self._employees.get_by_id(eid)
                if emp:
                    return ResolveResult(
                        candidates=[emp], confidence=conf, selected=emp
                    )
            if eid is not None and self._employees is None:
                return ResolveResult(candidates=[], confidence=conf, selected=None)

        if self._employees is None:
            return ResolveResult(candidates=[], confidence=0.0)

        matches = await self._employees.search_by_name(name, limit=5)
        if not matches:
            return ResolveResult(candidates=[], confidence=0.0)
        if len(matches) == 1:
            return ResolveResult(
                candidates=matches, confidence=0.95, selected=matches[0]
            )
        return ResolveResult(candidates=matches, confidence=0.4, selected=None)
