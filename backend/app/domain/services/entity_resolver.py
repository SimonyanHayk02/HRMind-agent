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

    @staticmethod
    def _alias_matches_needle(alias: str, needle: str) -> bool:
        """Match remembered aliases without first-name stealing a full name.

        ``Alice`` must not bind ``Alice Nguyen`` when the remembered person is
        Alice Bauer — multi-token queries require full-name / last-name agreement.
        """
        a = (alias or "").strip().lower()
        n = (needle or "").strip().lower()
        if not a or not n:
            return False
        if a == n or n == a:
            return True
        a_tokens = [t for t in a.split() if t]
        n_tokens = [t for t in n.split() if t]
        if not a_tokens or not n_tokens:
            return False
        # Exact full-name containment either way.
        if a in n.split() and len(a_tokens) >= 2:
            return True
        if n in a.split() and len(n_tokens) >= 2:
            return True
        if a == n or f"{' '.join(a_tokens)}" == f"{' '.join(n_tokens)}":
            return True
        # Multi-token needle: require full equality or last-name match with
        # matching first token — never first-name-only alias hits.
        if len(n_tokens) >= 2:
            if a == n:
                return True
            if len(a_tokens) >= 2 and a_tokens[0] == n_tokens[0] and a_tokens[-1] == n_tokens[-1]:
                return True
            if len(a_tokens) == 1:
                # Alias is a bare first name — only match single-token needles.
                return False
            # Needle last name equals alias last name and first names agree.
            if a_tokens[0] == n_tokens[0] and a_tokens[-1] == n_tokens[-1]:
                return True
            return False
        # Single-token needle: allow first/last token equality or unique alias.
        if n in a_tokens or a == n:
            return True
        if any(t.startswith(n) for t in a_tokens if len(n) > 2):
            return True
        return False

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
            for a in aliases:
                if self._alias_matches_needle(a, needle):
                    matched = True
                    # Full-name equality keeps full confidence; partial token lower.
                    if a == needle:
                        conf = ent.confidence
                    elif len(needle.split()) >= 2 and a != needle:
                        conf = max(0.75, ent.confidence * 0.95)
                    else:
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
