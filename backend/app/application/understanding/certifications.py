"""NLU for certification/licence questions (resume Certificates section).

Planner rule: cert/licence wording → attribute; 'experience/knows' → skill RAG.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.tools.resume_search.certifications import CERT_ALT, canon_certification

_CERT_HINT_RE = re.compile(
    r"\b("
    r"certifi(?:ed|cation|cations|cate|cates)|"
    r"licen[cs]e[ds]?|credential|credentials|"
    r"holds?\s+(?:a\s+)?(?:cert|licence|license)|"
    r"has\s+(?:a\s+)?(?:cert|licence|license)"
    r")\b",
    re.I,
)
# Must not steal skill-RAG asks like "AWS experience" / "knows AWS".
_SKILL_EXPERIENCE_RE = re.compile(
    r"\b(?:experience|knows?|skilled|familiar)\b",
    re.I,
)

_WHO_HAS_CERT_RE = re.compile(
    rf"\b(?:who\s+)?(?:has|have|holds?)\s+(?:an?\s+)?"
    rf"(?:(?:aws|cka|ckad|pmp|cissp|cpa|cfa|csm)\b|"
    rf"(?:the\s+)?(?P<cert>{CERT_ALT}))",
    re.I,
)
_CERTIFIED_RE = re.compile(
    rf"\b(?:who\s+is\s+)?(?:aws\s+)?certified\b|"
    rf"\b(?:who\s+has\s+)?(?P<cert>{CERT_ALT})\b",
    re.I,
)
_PERSON_CERT_RE = re.compile(
    r"\b(?:what\s+)?(?:certifications?|certificates?|licen[cs]es?)\s+(?:does\s+)?"
    r"(?P<name>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\s+(?:have|hold)\b|"
    r"\b(?P<name2>[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)'s\s+"
    r"(?:certifications?|certificates?|licen[cs]es?)\b",
    re.I,
)
_PRIOR_SET_PHRASE_RE = re.compile(
    r"\b("
    r"from there|of them|among them|from those|from them|"
    r"from (?:the )?(?:previous|prior|last) (?:list|set|group|search|results?)"
    r")\b",
    re.I,
)
_COHORT_HINT_RE = re.compile(
    r"\b(?:who|employees?|people|staff|anyone|someone)\b",
    re.I,
)
_PRONOUN_RE = re.compile(r"\b(she|he|her|him|his|hers|they|them|their)\b", re.I)
_PRONOUN_CERT_RE = re.compile(
    r"\b(?:"
    r"(?:her|his|their)\s+(?:certifications?|certificates?|licen[cs]es?|credentials?)|"
    r"what\s+(?:certifications?|certificates?)\s+does\s+(?:she|he|they)\s+have|"
    r"(?:certifications?|certificates?)\s+(?:does\s+)?(?:she|he|they)\s+have"
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
    }
)


@dataclass
class CertificationRequest:
    matched: bool = False
    scope: str = "cohort"
    certification: str | None = None
    person_name: str | None = None
    refers_to_prior: bool = False


def _clean_person_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = re.sub(r"'s$", "", raw.strip(), flags=re.I).strip(" '")
    if not name or name.lower() in _BAD_PERSON_NAMES:
        return None
    if any(tok in _BAD_PERSON_NAMES for tok in name.lower().split()):
        return None
    return name


def extract_certification(question: str) -> CertificationRequest:
    q = (question or "").strip()
    if not q:
        return CertificationRequest()

    # Experience / knows → skill RAG, not certifications.
    if _SKILL_EXPERIENCE_RE.search(q) and not _CERT_HINT_RE.search(q):
        return CertificationRequest()

    refers = bool(_PRIOR_SET_PHRASE_RE.search(q))

    # Pronoun cert asks bind from session — never treat "she" as a person name.
    if _PRONOUN_CERT_RE.search(q) or (
        _CERT_HINT_RE.search(q) and _PRONOUN_RE.search(q) and not _COHORT_HINT_RE.search(q)
    ):
        return CertificationRequest(
            matched=True,
            scope="person",
            refers_to_prior=refers,
        )

    m = _PERSON_CERT_RE.search(q)
    if m:
        name = _clean_person_name(m.group("name") or m.group("name2"))
        if name:
            return CertificationRequest(
                matched=True,
                scope="person",
                person_name=name,
                refers_to_prior=refers,
            )
        if _CERT_HINT_RE.search(q):
            return CertificationRequest(
                matched=True,
                scope="person",
                refers_to_prior=refers,
            )

    if not _CERT_HINT_RE.search(q) and not re.search(
        rf"\b({CERT_ALT})\b", q, re.I
    ):
        return CertificationRequest()

    # Explicit cert names with cert wording, or "who has AWS certified"
    for pat in (_WHO_HAS_CERT_RE, _CERTIFIED_RE):
        m = pat.search(q)
        if not m:
            continue
        raw = m.groupdict().get("cert")
        cert = canon_certification(raw) if raw else None
        if cert is None:
            # Try any closed-vocab hit in the question.
            hit = re.search(rf"\b({CERT_ALT})\b", q, re.I)
            cert = canon_certification(hit.group(1)) if hit else None
        if cert is None and re.search(r"\baws\s+certified\b", q, re.I):
            cert = "AWS Certified"
        if cert or _CERT_HINT_RE.search(q):
            return CertificationRequest(
                matched=True,
                scope="cohort" if _COHORT_HINT_RE.search(q) or refers else "person",
                certification=cert,
                refers_to_prior=refers,
            )

    if _CERT_HINT_RE.search(q) and (_COHORT_HINT_RE.search(q) or refers):
        hit = re.search(rf"\b({CERT_ALT})\b", q, re.I)
        return CertificationRequest(
            matched=True,
            scope="cohort",
            certification=canon_certification(hit.group(1)) if hit else None,
            refers_to_prior=refers,
        )

    return CertificationRequest()
