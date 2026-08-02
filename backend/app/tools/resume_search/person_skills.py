"""Skills listed on a person's resume (Skills section) — person-scoped answers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.tools.resume_search.skills import canonicalize_skill, content_mentions_skill

_LABEL_RE = re.compile(r"^\s*skills?\s*[:\-]", re.IGNORECASE)
# Soft skills often listed next to technical ones in seed resumes.
_SOFT = ("Communication", "Collaboration", "Leadership", "Teamwork")


def _body_lines(text: str) -> list[str]:
    """Drop ingest enrichment header + section title; keep the Skills body."""
    lines = text.splitlines()
    if not lines:
        return []
    # ``Maya Khan — Product Designer, Product (Dubai, UAE)``
    if "—" in lines[0] or " - " in lines[0][:80]:
        lines = lines[1:]
    if lines and re.match(r"^\s*skills?\s*:?\s*$", lines[0], re.I):
        lines = lines[1:]
    if lines and re.match(r"^\s*Employee\s*:", lines[0], re.I):
        lines = lines[1:]
    return lines


def parse_skills_list(text: str) -> list[str] | None:
    """Extract skill tokens from a Skills section (comma/newline separated)."""
    if not text:
        return None
    body = "\n".join(_body_lines(text))
    body = _LABEL_RE.sub("", body)
    found: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;\n•]+", body):
        raw = part.strip(" -•\t")
        if not raw or len(raw) > 60:
            continue
        # Skip leftover section headers / enrichment crumbs.
        if re.match(r"^(skills?|experience|summary|projects?|education)\s*:?\s*$", raw, re.I):
            continue
        if "—" in raw or re.search(r"\([^)]*(UAE|USA|UK|France|Germany)\)", raw, re.I):
            continue
        canon = canonicalize_skill(raw)
        label = canon or (raw.title() if raw.lower() in {s.lower() for s in _SOFT} else raw)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(label)
    return found or None


@dataclass
class SkillFact:
    employee_id: str
    name: str
    skills: list[str]
    chunk_id: str = ""

    def has_skill(self, skill: str | None) -> bool:
        if not skill:
            return bool(self.skills)
        want = canonicalize_skill(skill) or skill.strip()
        if any(s.lower() == want.lower() for s in self.skills):
            return True
        # Fall back to lexical mention across listed skills joined.
        return content_mentions_skill(", ".join(self.skills), want)

    def to_dict(self, _today: date | None = None) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "skills": list(self.skills),
            "extracted_from_chunk_id": self.chunk_id,
        }


def facts_from_hits(hits: list[dict[str, Any]]) -> list[SkillFact]:
    out: dict[str, SkillFact] = {}
    for hit in hits:
        eid = str(hit.get("employee_id") or "")
        if not eid or eid in out:
            continue
        skills = parse_skills_list(str(hit.get("content") or ""))
        if not skills:
            continue
        meta = hit.get("metadata") or {}
        name = (
            hit.get("employee_name")
            or (meta.get("employee_name") if isinstance(meta, dict) else None)
            or "This employee"
        )
        out[eid] = SkillFact(
            employee_id=eid,
            name=str(name),
            skills=skills,
            chunk_id=str(hit.get("id") or ""),
        )
    return list(out.values())


def select_cohort(
    facts: list[SkillFact],
    *,
    params: dict[str, Any],
    today: date | None = None,
) -> list[SkillFact]:
    skill = params.get("skill")
    if not skill:
        return list(facts)
    return [f for f in facts if f.has_skill(str(skill))]


def build_person_answer(
    facts: list[SkillFact],
    *,
    name_asked: str,
    skill: str | None = None,
    note: str | None = None,
    **_ignored: Any,
) -> str:
    prefix = f"{note.strip()} " if note and note.strip() else ""
    asked = (name_asked or "").strip() or "that employee"
    if not facts:
        return (
            prefix
            + f"I couldn't find skills listed in {asked}'s resume. "
            "Only details written in the resumes are available to me."
        )
    if len(facts) > 1:
        listed = "; ".join(f"{f.name} — {', '.join(f.skills)}" for f in facts[:5])
        return (
            prefix
            + f"{len(facts)} employees match {asked}: {listed}. "
            "Tell me which one you mean."
        )
    fact = facts[0]
    if skill:
        want = canonicalize_skill(skill) or skill.strip()
        if fact.has_skill(want):
            return (
                prefix
                + f"Yes — {fact.name}'s resume lists {want} "
                f"(skills: {', '.join(fact.skills)})."
            )
        return (
            prefix
            + f"No — I don't see {want} listed in {fact.name}'s resume skills "
            f"({', '.join(fact.skills)})."
        )
    return prefix + f"{fact.name}'s resume lists these skills: {', '.join(fact.skills)}."
