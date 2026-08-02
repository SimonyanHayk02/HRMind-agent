"""NLU for work-experience questions answered from the resume Experience section."""
from __future__ import annotations

import re
from dataclasses import dataclass

_PRONOUN_RE = re.compile(r"\b(she|he|her|him|his|hers|they|them|their)\b", re.I)
_COHORT_HINT_RE = re.compile(
    r"\b(?:who|employees?|people|staff|anyone|someone)\b",
    re.I,
)
_EXPERIENCE_TOPIC_RE = re.compile(
    r"\b("
    r"experience|experiences|worked|work\s+history|employment\s+history|"
    r"employer|employers|job\s+history|previous\s+(?:roles?|jobs?|companies)|"
    r"where\s+(?:has|have|did)\b.+\bwork(?:ed)?|"
    r"companies?\s+(?:has|have|did)\b.+\bwork"
    r")\b",
    re.I,
)
_PERSON_EXP_RE = re.compile(
    r"\b(?:"
    r"where\s+(?:has|have|did)\s+"
    r"(?P<name>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?|she|he|they)\s+"
    r"(?:worked|work)\b|"
    r"(?P<name2>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)'s\s+"
    r"(?:work\s+)?experience\b|"
    r"(?:work\s+)?experience\s+(?:of|for)\s+(?:the\s+)?"
    r"(?P<name3>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\b|"
    r"(?:her|his|their)\s+(?:work\s+)?experience\b|"
    r"where\s+(?:has|have)\s+(?:she|he|they)\s+worked\b"
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
class ExperienceRequest:
    matched: bool = False
    person_name: str | None = None


def _clean_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = re.sub(r"'s$", "", raw.strip(), flags=re.I).strip(" '")
    if not name or name.lower() in _BAD_NAMES:
        return None
    return name


def extract_experience(question: str) -> ExperienceRequest:
    q = (question or "").strip()
    if not q:
        return ExperienceRequest()

    # Living/based location asks are not experience.
    if re.search(r"\bwhere\s+(?:does|do|is)\b.+\b(?:live|lives|living|based|located)\b", q, re.I):
        return ExperienceRequest()
    if re.search(r"\b(?:live|lives|living|based in|located)\b", q, re.I) and not re.search(
        r"\b(?:worked|work\s+history|experience)\b", q, re.I
    ):
        return ExperienceRequest()

    m = _PERSON_EXP_RE.search(q)
    if m:
        name = _clean_name(m.group("name") or m.group("name2") or m.group("name3"))
        return ExperienceRequest(matched=True, person_name=name)

    if _EXPERIENCE_TOPIC_RE.search(q) and (
        _PRONOUN_RE.search(q) or not _COHORT_HINT_RE.search(q)
    ):
        # Named person elsewhere in the question
        named = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
            q,
        )
        return ExperienceRequest(
            matched=True,
            person_name=_clean_name(named.group(1)) if named else None,
        )

    return ExperienceRequest()
