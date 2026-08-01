"""Languages spoken live only in resume text (labelled Languages section)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

# Closed vocabulary — precision over recall; unlisted tokens are ignored.
_LANGUAGES: dict[str, str] = {
    "english": "English",
    "german": "German",
    "deutsch": "German",
    "french": "French",
    "français": "French",
    "francais": "French",
    "spanish": "Spanish",
    "español": "Spanish",
    "espanol": "Spanish",
    "arabic": "Arabic",
    "hindi": "Hindi",
    "mandarin": "Mandarin",
    "chinese": "Chinese",
    "japanese": "Japanese",
    "korean": "Korean",
    "portuguese": "Portuguese",
    "italian": "Italian",
    "dutch": "Dutch",
    "russian": "Russian",
    "turkish": "Turkish",
    "polish": "Polish",
    "swedish": "Swedish",
    "norwegian": "Norwegian",
    "danish": "Danish",
    "finnish": "Finnish",
    "greek": "Greek",
    "hebrew": "Hebrew",
    "urdu": "Urdu",
    "bengali": "Bengali",
    "tamil": "Tamil",
    "vietnamese": "Vietnamese",
    "thai": "Thai",
    "indonesian": "Indonesian",
    "malay": "Malay",
    "tagalog": "Tagalog",
    "filipino": "Filipino",
    "czech": "Czech",
    "hungarian": "Hungarian",
    "romanian": "Romanian",
    "ukrainian": "Ukrainian",
    "persian": "Persian",
    "farsi": "Persian",
}

LANG_ALT = "|".join(sorted((_LANGUAGES.keys()), key=len, reverse=True))
_LANG_TOKEN_RE = re.compile(rf"\b({LANG_ALT})\b", re.IGNORECASE)
_LABEL_RE = re.compile(r"^\s*languages?\s*[:\-]", re.IGNORECASE)


def canon_language(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    return _LANGUAGES.get(key)


def _body_lines(text: str) -> list[str]:
    """Skip the ingest enrichment header when present; keep bare section text intact."""
    lines = text.splitlines()
    if len(lines) > 1 and re.match(r"^\s*Employee\s*:", lines[0], re.I):
        return lines[1:]
    return lines


def parse_languages(text: str) -> list[str] | None:
    """Extract closed-vocab languages from resume text. Prefer labelled lines."""
    if not text:
        return None
    found: list[str] = []
    seen: set[str] = set()
    body = _body_lines(text)
    labelled = [ln for ln in body if _LABEL_RE.search(ln) or _LANG_TOKEN_RE.search(ln)]
    candidates = labelled or body or [text]
    for chunk in candidates:
        for m in _LANG_TOKEN_RE.finditer(chunk):
            canon = canon_language(m.group(1))
            if canon and canon not in seen:
                seen.add(canon)
                found.append(canon)
    return found or None


@dataclass
class LanguageFact:
    employee_id: str
    name: str
    languages: list[str]
    chunk_id: str = ""

    def matches(self, *, language: str | None = None) -> bool:
        if not language:
            return bool(self.languages)
        want = canon_language(language) or language.strip().title()
        return any(lang.lower() == want.lower() for lang in self.languages)

    def to_dict(self, _today: date | None = None) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "languages": list(self.languages),
            "extracted_from_chunk_id": self.chunk_id,
        }


def facts_from_hits(hits: list[dict[str, Any]]) -> list[LanguageFact]:
    out: dict[str, LanguageFact] = {}
    for hit in hits:
        eid = str(hit.get("employee_id") or "")
        if not eid or eid in out:
            continue
        langs = parse_languages(str(hit.get("content") or ""))
        if not langs:
            continue
        meta = hit.get("metadata") or {}
        name = (
            hit.get("employee_name")
            or (meta.get("employee_name") if isinstance(meta, dict) else None)
            or "This employee"
        )
        out[eid] = LanguageFact(
            employee_id=eid,
            name=str(name),
            languages=langs,
            chunk_id=str(hit.get("id") or ""),
        )
    return list(out.values())


def select_cohort(
    facts: list[LanguageFact],
    *,
    params: dict[str, Any],
    today: date | None = None,
) -> list[LanguageFact]:
    language = params.get("language") or params.get("languages")
    if isinstance(language, list):
        language = language[0] if language else None
    if not language:
        return list(facts)
    return [f for f in facts if f.matches(language=str(language))]


def build_person_answer(
    facts: list[LanguageFact],
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
            + f"I couldn't find languages listed in {asked}'s resume. "
            "Only details written in the resumes are available to me."
        )
    if len(facts) > 1:
        listed = "; ".join(
            f"{f.name} — {', '.join(f.languages)}" for f in facts[:5]
        )
        return (
            prefix
            + f"{len(facts)} employees match {asked}: {listed}. "
            "Tell me which one you mean."
        )
    fact = facts[0]
    langs = ", ".join(fact.languages)
    return prefix + f"{fact.name} speaks {langs}, according to their resume."


def build_cohort_answer(
    facts: list[LanguageFact],
    *,
    language: str | None = None,
    coverage: str | None = None,
    among_prior: bool = False,
    **_ignored: Any,
) -> str:
    label = canon_language(language) or (language or "that language")
    scope = "among the previous list, " if among_prior else ""
    if not facts:
        base = f"I found no one {scope}who lists {label} on their resume."
        if coverage:
            return f"{base} {coverage}"
        return base
    names = ", ".join(f.name for f in facts[:25])
    more = f" (+{len(facts) - 25} more)" if len(facts) > 25 else ""
    base = f"{len(facts)} employee(s) {scope}list {label} on their resume: {names}{more}."
    if coverage:
        return f"{base} {coverage}"
    return base
