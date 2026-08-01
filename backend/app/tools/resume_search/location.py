"""Work location lives only in resume text, so it is parsed back out of chunks.

The resume carries the place twice by design — a ``Location`` section and the
Summary line — because a document written for humans repeats itself. Both are
accepted, the labelled section first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.domain.places import CITY_ALT, COUNTRY_ALT, canon_city, canon_country

# "Berlin, Germany" — the canonical Location-section spelling.
_PAIR_RE = re.compile(rf"\b({CITY_ALT})\s*,\s*({COUNTRY_ALT})\b", re.IGNORECASE)
_CITY_RE = re.compile(rf"\b({CITY_ALT})\b", re.IGNORECASE)
_COUNTRY_RE = re.compile(rf"\b({COUNTRY_ALT})\b", re.IGNORECASE)
_LABEL_RE = re.compile(r"location|based\s+in|lives?\s+in|located", re.IGNORECASE)


@dataclass(frozen=True)
class Place:
    city: str | None = None
    country: str | None = None

    def __bool__(self) -> bool:
        return bool(self.city or self.country)

    @property
    def text(self) -> str:
        return ", ".join(p for p in (self.city, self.country) if p)


def parse_location(text: str) -> Place | None:
    """Pull a work location out of free resume text.

    The enriched chunk header also names the place, so the labelled body lines
    are read first and the header is only a fallback: they are written from the
    same source and cannot disagree, but preferring the body keeps the document
    authoritative.
    """
    if not text:
        return None
    for candidate in _location_candidates(text):
        place = _parse_place(candidate)
        if place:
            return place
    return None


def _location_candidates(text: str) -> list[str]:
    lines = text.splitlines()
    # Skip the enriched context header, which is always the first line.
    body = lines[1:] if len(lines) > 1 else lines
    labelled = [line for line in body if _LABEL_RE.search(line)]
    bare = [line for line in body if _PAIR_RE.search(line)]
    return [*labelled, *bare, text]


def _parse_place(text: str) -> Place | None:
    pair = _PAIR_RE.search(text)
    if pair:
        return Place(canon_city(pair.group(1)), canon_country(pair.group(2)))
    city = _CITY_RE.search(text)
    country = _COUNTRY_RE.search(text)
    if city or country:
        place = Place(
            canon_city(city.group(1)) if city else None,
            canon_country(country.group(1)) if country else None,
        )
        if place:
            return place
    return None


@dataclass
class LocationFact:
    """One person's work location as recovered from their resume text."""

    employee_id: str
    name: str
    city: str | None = None
    country: str | None = None
    chunk_id: str = ""

    @property
    def place(self) -> str:
        return ", ".join(p for p in (self.city, self.country) if p)

    def matches(self, *, city: str | None = None, country: str | None = None) -> bool:
        """Whether this person sits in the asked-for place.

        A country-only question includes every city in it, which is the whole
        point of asking by country.
        """
        if city and (self.city or "").lower() != city.lower():
            return False
        if country and (self.country or "").lower() != country.lower():
            return False
        return bool(city or country)

    def to_dict(self, _today: date | None = None) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "city": self.city,
            "country": self.country,
            "location": self.place,
            # Provenance: the resume chunk this place was read out of.
            "extracted_from_chunk_id": self.chunk_id,
        }


def facts_from_hits(hits: list[dict[str, Any]]) -> list[LocationFact]:
    """Parse one location per employee from retrieved resume chunks."""
    out: dict[str, LocationFact] = {}
    for hit in hits:
        eid = str(hit.get("employee_id") or "")
        if not eid or eid in out:
            continue
        place = parse_location(str(hit.get("content") or ""))
        if place is None:
            continue
        meta = hit.get("metadata") or {}
        name = (
            hit.get("employee_name")
            or (meta.get("employee_name") if isinstance(meta, dict) else None)
            or "This employee"
        )
        out[eid] = LocationFact(
            employee_id=eid,
            name=str(name),
            city=place.city,
            country=place.country,
            chunk_id=str(hit.get("id") or ""),
        )
    return list(out.values())


def select_cohort(
    facts: list[LocationFact],
    *,
    params: dict[str, Any],
    today: date | None = None,
) -> list[LocationFact]:
    """The people in the asked-for place.

    With no place asked the selection is everyone retrieved, which is what an
    id-scoped lookup (one known employee) wants.
    """
    city = params.get("city") or None
    country = params.get("country") or None
    if not city and not country:
        return list(facts)
    return [f for f in facts if f.matches(city=str(city) if city else None,
                                          country=str(country) if country else None)]


def build_person_answer(
    facts: list[LocationFact],
    *,
    name_asked: str,
    today: date | None = None,
    note: str | None = None,
    **_ignored: Any,
) -> str:
    """Answer "where does X work"."""
    prefix = f"{note.strip()} " if note and note.strip() else ""
    return prefix + _person_answer(facts, name_asked=name_asked)


def _person_answer(facts: list[LocationFact], *, name_asked: str) -> str:
    asked = (name_asked or "").strip() or "that employee"
    if not facts:
        return (
            f"I couldn't find a work location in {asked}'s resume. "
            "Only details written in the resumes are available to me."
        )
    if len(facts) > 1:
        listed = "; ".join(f"{f.name} — {f.place}" for f in facts[:5])
        return f"{len(facts)} employees match {asked}: {listed}. Tell me which one you mean."
    fact = facts[0]
    if not fact.place:
        return f"I couldn't find a work location in {fact.name}'s resume."
    return f"{fact.name} is based in {fact.place}, according to their resume."
