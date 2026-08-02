"""Projects live in the resume Projects section."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

_BULLET_RE = re.compile(r"^\s*[-•\*]\s*(.+)$")


def _body_lines(text: str) -> list[str]:
    """Drop ingest enrichment header + section title; keep Projects bullets."""
    lines = text.splitlines()
    if not lines:
        return []
    if "—" in lines[0] or " - " in lines[0][:80]:
        lines = lines[1:]
    if lines and re.match(r"^\s*projects?\s*:?\s*$", lines[0], re.I):
        lines = lines[1:]
    if lines and re.match(r"^\s*Employee\s*:", lines[0], re.I):
        lines = lines[1:]
    return lines


def parse_projects(text: str) -> list[str] | None:
    """Return bullet lines from a Projects section."""
    if not text:
        return None
    lines = _body_lines(text)
    if lines and re.match(
        r"^\s*(summary|skills?|experience|education|languages?|personal|location|"
        r"certificates?)\s*:?\s*$",
        lines[0],
        re.I,
    ):
        return None
    items: list[str] = []
    for ln in lines:
        m = _BULLET_RE.match(ln)
        if m:
            item = m.group(1).strip()
            if item and "—" not in item[:40]:
                items.append(item)
            continue
    if items:
        return items
    for ln in lines:
        stripped = ln.strip()
        if not stripped or re.match(r"^projects?\s*:?\s*$", stripped, re.I):
            continue
        if "—" in stripped[:60] or re.match(
            r"^(summary|skills|experience|education|languages)\b", stripped, re.I
        ):
            continue
        items.append(stripped)
    return items or None


@dataclass
class ProjectFact:
    employee_id: str
    name: str
    items: list[str]
    chunk_id: str = ""

    def to_dict(self, _today: date | None = None) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "projects": list(self.items),
            "extracted_from_chunk_id": self.chunk_id,
        }


def facts_from_hits(hits: list[dict[str, Any]]) -> list[ProjectFact]:
    out: dict[str, ProjectFact] = {}
    for hit in hits:
        eid = str(hit.get("employee_id") or "")
        if not eid or eid in out:
            continue
        items = parse_projects(str(hit.get("content") or ""))
        if not items:
            continue
        meta = hit.get("metadata") or {}
        name = (
            hit.get("employee_name")
            or (meta.get("employee_name") if isinstance(meta, dict) else None)
            or "This employee"
        )
        out[eid] = ProjectFact(
            employee_id=eid,
            name=str(name),
            items=items,
            chunk_id=str(hit.get("id") or ""),
        )
    return list(out.values())


def build_person_answer(
    facts: list[ProjectFact],
    *,
    name_asked: str,
    note: str | None = None,
    **_ignored: Any,
) -> str:
    prefix = f"{note.strip()} " if note and note.strip() else ""
    asked = (name_asked or "").strip() or "that employee"
    if not facts:
        return (
            prefix
            + f"I couldn't find projects listed in {asked}'s resume. "
            "Only details written in the resumes are available to me."
        )
    if len(facts) > 1:
        listed = "; ".join(f"{f.name}" for f in facts[:5])
        return (
            prefix
            + f"{len(facts)} employees match {asked}: {listed}. "
            "Tell me which one you mean."
        )
    fact = facts[0]
    bullets = "; ".join(fact.items[:8])
    return prefix + f"{fact.name}'s resume lists these projects: {bullets}."
