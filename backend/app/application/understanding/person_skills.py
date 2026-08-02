"""NLU for person-scoped skill questions answered from the resume Skills section."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.tools.resume_search.skills import extract_skill

_PRONOUN_RE = re.compile(r"\b(she|he|her|him|his|hers|they|them|their)\b", re.I)
_COHORT_HINT_RE = re.compile(
    r"\b(?:who|employees?|people|staff|anyone|someone)\b",
    re.I,
)
_SKILL_TOPIC_RE = re.compile(
    r"\b(skills?|proficient|expertise|knows?|knowing|familiar)\b",
    re.I,
)
_PERSON_SKILLS_LIST_RE = re.compile(
    r"\b(?:"
    r"(?:what|which)\s+skills?\s+(?:does\s+)?"
    r"(?P<name>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)(?:'s)?\s+have\b|"
    r"(?P<name2>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)'s\s+skills?\b|"
    r"skills?\s+(?:of|for)\s+(?:the\s+)?"
    r"(?P<name3>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\b|"
    r"(?:her|his|their)\s+skills?\b|"
    r"what\s+skills?\s+does\s+(?:she|he|they)\s+have\b"
    r")",
    re.I,
)
_PERSON_KNOWS_RE = re.compile(
    r"\b(?:does|do)\s+"
    r"(?P<name>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?|she|he|they)\s+"
    r"(?:know|have|use)\b",
    re.I,
)
_BAD_NAMES = frozenset(
    {
        "who",
        "anyone",
        "someone",
        "employees",
        "people",
        "she",
        "he",
        "her",
        "him",
        "his",
        "hers",
        "they",
        "them",
        "their",
    }
)


@dataclass
class PersonSkillsRequest:
    matched: bool = False
    person_name: str | None = None
    skill: str | None = None  # set when asking "does X know Python?"
    refers_to_prior: bool = False


def _clean_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = re.sub(r"'s$", "", raw.strip(), flags=re.I).strip(" '")
    if not name or name.lower() in _BAD_NAMES:
        return None
    return name


def extract_person_skills(question: str) -> PersonSkillsRequest:
    q = (question or "").strip()
    if not q:
        return PersonSkillsRequest()

    # Cohort "who knows Python?" stays on the existing skill RAG path.
    if _COHORT_HINT_RE.search(q) and not _PRONOUN_RE.search(q):
        return PersonSkillsRequest()

    m = _PERSON_SKILLS_LIST_RE.search(q)
    if m:
        name = _clean_name(m.group("name") or m.group("name2") or m.group("name3"))
        return PersonSkillsRequest(matched=True, person_name=name)

    m = _PERSON_KNOWS_RE.search(q)
    if m and _SKILL_TOPIC_RE.search(q):
        raw_name = (m.group("name") or "").strip()
        name = None if raw_name.lower() in _BAD_NAMES else _clean_name(raw_name)
        skill = extract_skill(q)
        if skill or name or _PRONOUN_RE.search(q):
            return PersonSkillsRequest(
                matched=True,
                person_name=name,
                skill=skill,
            )

    # "her skills" / "skills she has"
    if _SKILL_TOPIC_RE.search(q) and _PRONOUN_RE.search(q) and not _COHORT_HINT_RE.search(q):
        return PersonSkillsRequest(matched=True, skill=extract_skill(q))

    return PersonSkillsRequest()
