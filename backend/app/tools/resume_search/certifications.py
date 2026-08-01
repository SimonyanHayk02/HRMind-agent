"""Certifications / licences live only in resume Certificates sections."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

# Closed vocabulary of common certifications (canonical display → aliases).
_CERTS: dict[str, str] = {
    "aws certified": "AWS Certified",
    "aws certification": "AWS Certified",
    "aws certified solutions architect": "AWS Certified Solutions Architect",
    "solutions architect": "AWS Certified Solutions Architect",
    "aws certified developer": "AWS Certified Developer",
    "cka": "CKA",
    "certified kubernetes administrator": "CKA",
    "ckad": "CKAD",
    "certified kubernetes application developer": "CKAD",
    "pmp": "PMP",
    "project management professional": "PMP",
    "cissp": "CISSP",
    "cpa": "CPA",
    "cfa": "CFA",
    "shrm": "SHRM-CP",
    "shrm-cp": "SHRM-CP",
    "google cloud professional": "Google Cloud Professional",
    "gcp professional": "Google Cloud Professional",
    "azure administrator": "Azure Administrator",
    "microsoft certified": "Microsoft Certified",
    "scrum master": "Certified Scrum Master",
    "csm": "Certified Scrum Master",
    "professional certificate in python": "Professional certificate in Python",
    "professional certificate in java": "Professional certificate in Java",
    "professional certificate in go": "Professional certificate in Go",
    "professional certificate in sql": "Professional certificate in SQL",
    "professional certificate in kubernetes": "Professional certificate in Kubernetes",
    "professional certificate in react": "Professional certificate in React",
    "professional certificate in machine learning": "Professional certificate in Machine Learning",
    "professional certificate in nlp": "Professional certificate in NLP",
    "professional certificate in aws": "Professional certificate in AWS",
    "professional certificate in docker": "Professional certificate in Docker",
    "professional certificate in recruiting": "Professional certificate in Recruiting",
    "professional certificate in salesforce": "Professional certificate in Salesforce",
    "professional certificate in accounting": "Professional certificate in Accounting",
    "professional certificate in product strategy": "Professional certificate in Product Strategy",
}

CERT_ALT = "|".join(
    sorted((re.escape(k) for k in _CERTS), key=len, reverse=True)
)
_CERT_RE = re.compile(rf"\b({CERT_ALT})\b", re.IGNORECASE)
_LABEL_RE = re.compile(r"certificates?|certifications?|licen[cs]es?", re.IGNORECASE)


def canon_certification(raw: str | None) -> str | None:
    if not raw:
        return None
    key = re.sub(r"\s+", " ", raw.strip().lower())
    return _CERTS.get(key)


def _body_lines(text: str) -> list[str]:
    """Skip the ingest enrichment header when present; keep bare section text intact."""
    lines = text.splitlines()
    if len(lines) > 1 and re.match(r"^\s*Employee\s*:", lines[0], re.I):
        return lines[1:]
    return lines


def parse_certifications(text: str) -> list[str] | None:
    if not text:
        return None
    found: list[str] = []
    seen: set[str] = set()
    body = _body_lines(text)
    labelled = [ln for ln in body if _LABEL_RE.search(ln) or _CERT_RE.search(ln)]
    for chunk in labelled or body or [text]:
        for m in _CERT_RE.finditer(chunk):
            canon = canon_certification(m.group(1))
            if canon and canon not in seen:
                seen.add(canon)
                found.append(canon)
        # Also accept "Professional certificate in {Skill}" lines even if skill
        # was not in the static map (seeded resumes).
        for m in re.finditer(
            r"professional\s+certificate\s+in\s+([A-Za-z][A-Za-z0-9+.#/\s-]{1,40})",
            chunk,
            re.I,
        ):
            label = f"Professional certificate in {m.group(1).strip()}"
            if label not in seen:
                seen.add(label)
                found.append(label)
    return found or None


@dataclass
class CertificationFact:
    employee_id: str
    name: str
    certifications: list[str]
    chunk_id: str = ""

    def matches(self, *, certification: str | None = None) -> bool:
        if not certification:
            return bool(self.certifications)
        want = canon_certification(certification) or certification.strip()
        want_l = want.lower()
        return any(
            c.lower() == want_l or want_l in c.lower() or c.lower() in want_l
            for c in self.certifications
        )

    def to_dict(self, _today: date | None = None) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "certifications": list(self.certifications),
            "extracted_from_chunk_id": self.chunk_id,
        }


def facts_from_hits(hits: list[dict[str, Any]]) -> list[CertificationFact]:
    out: dict[str, CertificationFact] = {}
    for hit in hits:
        eid = str(hit.get("employee_id") or "")
        if not eid or eid in out:
            continue
        certs = parse_certifications(str(hit.get("content") or ""))
        if not certs:
            continue
        meta = hit.get("metadata") or {}
        name = (
            hit.get("employee_name")
            or (meta.get("employee_name") if isinstance(meta, dict) else None)
            or "This employee"
        )
        out[eid] = CertificationFact(
            employee_id=eid,
            name=str(name),
            certifications=certs,
            chunk_id=str(hit.get("id") or ""),
        )
    return list(out.values())


def select_cohort(
    facts: list[CertificationFact],
    *,
    params: dict[str, Any],
    today: date | None = None,
) -> list[CertificationFact]:
    cert = params.get("certification") or params.get("certifications")
    if isinstance(cert, list):
        cert = cert[0] if cert else None
    if not cert:
        return list(facts)
    return [f for f in facts if f.matches(certification=str(cert))]


def build_person_answer(
    facts: list[CertificationFact],
    *,
    name_asked: str,
    today: date | None = None,
    note: str | None = None,
    **_ignored: Any,
) -> str:
    prefix = f"{note.strip()} " if note and note.strip() else ""
    asked = (name_asked or "").strip() or "that employee"
    if not facts:
        return (
            prefix
            + f"I couldn't find certifications listed in {asked}'s resume. "
            "Only details written in the resumes are available to me."
        )
    if len(facts) > 1:
        listed = "; ".join(
            f"{f.name} — {', '.join(f.certifications)}" for f in facts[:5]
        )
        return (
            prefix
            + f"{len(facts)} employees match {asked}: {listed}. "
            "Tell me which one you mean."
        )
    fact = facts[0]
    certs = ", ".join(fact.certifications)
    return prefix + f"{fact.name} holds: {certs}, according to their resume."


def build_cohort_answer(
    facts: list[CertificationFact],
    *,
    certification: str | None = None,
    coverage: str | None = None,
    among_prior: bool = False,
    **_ignored: Any,
) -> str:
    label = canon_certification(certification) or (certification or "that certification")
    scope = "among the previous list, " if among_prior else ""
    if not facts:
        base = f"I found no one {scope}who lists {label} on their resume."
        if coverage:
            return f"{base} {coverage}"
        return base
    names = ", ".join(f.name for f in facts[:25])
    more = f" (+{len(facts) - 25} more)" if len(facts) > 25 else ""
    base = (
        f"{len(facts)} employee(s) {scope}list {label} on their resume: "
        f"{names}{more}."
    )
    if coverage:
        return f"{base} {coverage}"
    return base
