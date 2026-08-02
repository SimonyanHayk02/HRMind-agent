"""NLU for language questions answered from resume Languages sections."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.tools.resume_search.languages import LANG_ALT, canon_language

_SPEAKS_RE = re.compile(
    rf"\b(?:who\s+)?(?:speaks?|speaking|fluent\s+in|knows?\s+(?:how\s+to\s+)?speak)\s+"
    rf"(?:the\s+)?(?P<lang>{LANG_ALT})\b",
    re.I,
)
_LANG_OF_RE = re.compile(
    rf"\b(?:languages?|spoken\s+languages?)\s+(?:of|for)\s+"
    rf"(?:the\s+)?(?P<name>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\b",
    re.I,
)
_PERSON_LANG_RE = re.compile(
    rf"\b(?:what\s+languages?\s+does\s+)?"
    rf"(?P<name>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)"
    rf"(?:'s)?\s+(?:languages?|speak|speaks)\b|"
    rf"\bdoes\s+(?P<name2>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\s+"
    rf"speak\s+(?:the\s+)?(?P<lang2>{LANG_ALT})\b",
    re.I,
)
_COHORT_HINT_RE = re.compile(
    r"\b(?:who|employees?|people|staff|anyone|someone)\b",
    re.I,
)
_PRIOR_SET_PHRASE_RE = re.compile(
    r"\b("
    r"from there|of them|among them|from those|from them|"
    r"from (?:the )?(?:previous|prior|last) (?:list|set|group|search|results?)"
    r")\b",
    re.I,
)
_PRONOUN_RE = re.compile(r"\b(she|he|her|him|his|hers|they|them|their)\b", re.I)
_LANG_TOPIC_RE = re.compile(
    r"\b("
    r"languages?|spoken\s+languages?|known\s+languages?|"
    r"speak|speaks|speaking|spoken|fluent"
    r")\b",
    re.I,
)
_PRONOUN_LANG_RE = re.compile(
    r"\b(?:"
    r"(?:her|his|their)\s+(?:known\s+)?languages?|"
    r"(?:known\s+)?languages?\s+(?:does\s+)?(?:she|he|they)\s+speak|"
    r"what\s+languages?\s+does\s+(?:she|he|they)\s+speak|"
    r"(?:does|do)\s+(?:she|he|they)\s+speak|"
    r"languages?\s+she\s+speaks?|languages?\s+he\s+speaks?"
    r")\b",
    re.I,
)
_BAD_PERSON_NAMES = frozenset(
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
        "known",
        "her known",
        "his known",
        "their known",
    }
)


@dataclass
class LanguageRequest:
    matched: bool = False
    scope: str = "cohort"  # person | cohort
    language: str | None = None
    person_name: str | None = None
    refers_to_prior: bool = False


def _clean_person_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = re.sub(r"'s$", "", raw.strip(), flags=re.I).strip(" '")
    if not name or name.lower() in _BAD_PERSON_NAMES:
        return None
    # Reject "her known" style fragments.
    if any(tok in _BAD_PERSON_NAMES for tok in name.lower().split()):
        return None
    return name


def extract_language(question: str) -> LanguageRequest:
    q = (question or "").strip()
    if not q:
        return LanguageRequest()

    refers = bool(_PRIOR_SET_PHRASE_RE.search(q))

    m = _SPEAKS_RE.search(q)
    if m and _COHORT_HINT_RE.search(q):
        return LanguageRequest(
            matched=True,
            scope="cohort",
            language=canon_language(m.group("lang")),
            refers_to_prior=refers,
        )

    # Pronoun / deictic person languages — bind from session, never treat "she" as a name.
    if _PRONOUN_LANG_RE.search(q) or (
        _LANG_TOPIC_RE.search(q)
        and _PRONOUN_RE.search(q)
        and not _COHORT_HINT_RE.search(q)
    ):
        return LanguageRequest(
            matched=True,
            scope="person",
            refers_to_prior=refers,
        )

    m = _LANG_OF_RE.search(q)
    if m:
        name = _clean_person_name(m.group("name"))
        if name:
            return LanguageRequest(
                matched=True,
                scope="person",
                person_name=name,
                refers_to_prior=refers,
            )

    m = _PERSON_LANG_RE.search(q)
    if m:
        name = _clean_person_name(m.group("name") or m.group("name2"))
        lang = canon_language(m.group("lang2")) if m.groupdict().get("lang2") else None
        if name:
            return LanguageRequest(
                matched=True,
                scope="person",
                person_name=name,
                language=lang,
                refers_to_prior=refers,
            )
        # "what languages does she speak" already handled above; keep a soft match.
        if _LANG_TOPIC_RE.search(q):
            return LanguageRequest(matched=True, scope="person", refers_to_prior=refers)

    # Bare "who speaks German among them"
    m = re.search(rf"\b({LANG_ALT})\b", q, re.I)
    if m and re.search(r"\b(?:speak|speaks|speaking|fluent|language)\b", q, re.I):
        return LanguageRequest(
            matched=True,
            scope="cohort" if _COHORT_HINT_RE.search(q) or refers else "person",
            language=canon_language(m.group(1)),
            refers_to_prior=refers,
        )

    return LanguageRequest()
