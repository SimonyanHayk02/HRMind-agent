"""NLU for project questions answered from the resume Projects section."""
from __future__ import annotations

import re
from dataclasses import dataclass

_PRONOUN_RE = re.compile(r"\b(she|he|her|him|his|hers|they|them|their)\b", re.I)
_COHORT_HINT_RE = re.compile(
    r"\b(?:who|employees?|people|staff|anyone|someone)\b",
    re.I,
)
_PROJECT_TOPIC_RE = re.compile(
    r"\b(projects?|initiatives?|portfolio)\b",
    re.I,
)
_PERSON_PROJECT_RE = re.compile(
    r"\b(?:"
    r"(?:what|which|tell\s+me\s+about|give\s+(?:me\s+)?(?:info(?:rmation)?\s+about)?)\s+"
    r"(?:her|his|their)\s+projects?\b|"
    r"(?:her|his|their)\s+projects?\b|"
    r"what\s+(?:are|is)\s+(?:her|his|their)\s+projects?\b|"
    r"(?:what|which)\s+projects?\s+(?:does\s+)?"
    r"(?P<name>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?|she|he|they)\s+have\b|"
    r"(?P<name2>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)'s\s+projects?\b|"
    r"projects?\s+(?:of|for)\s+(?:the\s+)?"
    r"(?P<name3>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\b|"
    r"(?:about|information\s+about)\s+(?:her|his|their)\s+projects?\b"
    r")",
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
class ProjectsRequest:
    matched: bool = False
    person_name: str | None = None


def _clean_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = re.sub(r"'s$", "", raw.strip(), flags=re.I).strip(" '")
    if not name or name.lower() in _BAD_NAMES:
        return None
    return name


def extract_projects(question: str) -> ProjectsRequest:
    q = (question or "").strip()
    if not q:
        return ProjectsRequest()

    m = _PERSON_PROJECT_RE.search(q)
    if m:
        name = _clean_name(
            (m.groupdict().get("name") or m.groupdict().get("name2") or m.groupdict().get("name3"))
        )
        return ProjectsRequest(matched=True, person_name=name)

    if _PROJECT_TOPIC_RE.search(q) and (
        _PRONOUN_RE.search(q) or not _COHORT_HINT_RE.search(q)
    ):
        named = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", q)
        return ProjectsRequest(
            matched=True,
            person_name=_clean_name(named.group(1)) if named else None,
        )

    return ProjectsRequest()
