"""Work experience lives in the resume Experience section."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

_BULLET_RE = re.compile(r"^\s*[-•\*]\s*(.+)$")


def _body_lines(text: str) -> list[str]:
    """Drop ingest enrichment header + section title; keep Experience bullets."""
    lines = text.splitlines()
    if not lines:
        return []
    if "—" in lines[0] or " - " in lines[0][:80]:
        lines = lines[1:]
    if lines and re.match(r"^\s*experience\s*:?\s*$", lines[0], re.I):
        lines = lines[1:]
    if lines and re.match(r"^\s*Employee\s*:", lines[0], re.I):
        lines = lines[1:]
    return lines


def parse_experience(text: str) -> list[str] | None:
    """Return bullet lines from an Experience section."""
    if not text:
        return None
    lines = _body_lines(text)
    # Wrong-section guard: Summary/Skills/etc. bleed must not become experience.
    if lines and re.match(
        r"^\s*(summary|skills?|projects?|education|languages?|personal|location|"
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
    # Prefer bullets only — prose leftovers are usually enrichment/summary bleed.
    if items:
        return items
    for ln in lines:
        stripped = ln.strip()
        if not stripped or re.match(r"^experience\s*:?\s*$", stripped, re.I):
            continue
        if "—" in stripped[:60] or re.match(
            r"^(summary|skills|projects|education|languages)\b", stripped, re.I
        ):
            continue
        items.append(stripped)
    return items or None


@dataclass
class ExperienceFact:
    employee_id: str
    name: str
    items: list[str]
    chunk_id: str = ""

    def to_dict(self, _today: date | None = None) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "experience": list(self.items),
            "extracted_from_chunk_id": self.chunk_id,
        }


def facts_from_hits(hits: list[dict[str, Any]]) -> list[ExperienceFact]:
    out: dict[str, ExperienceFact] = {}
    for hit in hits:
        eid = str(hit.get("employee_id") or "")
        if not eid or eid in out:
            continue
        items = parse_experience(str(hit.get("content") or ""))
        if not items:
            continue
        meta = hit.get("metadata") or {}
        name = (
            hit.get("employee_name")
            or (meta.get("employee_name") if isinstance(meta, dict) else None)
            or "This employee"
        )
        out[eid] = ExperienceFact(
            employee_id=eid,
            name=str(name),
            items=items,
            chunk_id=str(hit.get("id") or ""),
        )
    return list(out.values())


def build_person_answer(
    facts: list[ExperienceFact],
    *,
    name_asked: str,
    note: str | None = None,
    completeness_ask: bool = False,
    **_ignored: Any,
) -> str:
    prefix = f"{note.strip()} " if note and note.strip() else ""
    asked = (name_asked or "").strip() or "that employee"
    if not facts:
        return (
            prefix
            + f"I couldn't find work experience listed in {asked}'s resume. "
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
    items = list(fact.items)
    listed = "; ".join(items[:8])
    if completeness_ask:
        if len(items) <= 1:
            only = items[0] if items else "nothing"
            return (
                prefix
                + f"Yes — {fact.name}'s resume lists only one experience entry: {only}. "
                "There are no additional experience items beyond that."
            )
        return (
            prefix
            + f"No — {fact.name}'s resume lists {len(items)} experience entries: {listed}. "
            "That is everything written in the Experience section; "
            "there are no further experience items beyond these."
        )
    if len(items) == 1:
        return prefix + f"{fact.name}'s resume lists this experience: {items[0]}."
    return (
        prefix
        + f"{fact.name}'s resume lists these experience entries: {listed}."
    )
