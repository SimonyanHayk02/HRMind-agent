"""One place vocabulary for the whole app.

City and country live only in resume text, so the names used to *ask* about them
cannot be maintained in four separate regexes that quietly disagree — a city
present in the corpus but missing from one copy is simply unaskable through that
path. Everything that recognises or canonicalises a place reads this module, and
the schema catalog seeds its values from here.
"""
from __future__ import annotations

import re

# Canonical (country, city) pairs — the only place→city mapping the seed generator
# and the catalog share. Lists below are derived so they cannot drift apart.
PLACE_PAIRS: tuple[tuple[str, str], ...] = (
    ("Germany", "Berlin"),
    ("Germany", "Munich"),
    ("Germany", "Hamburg"),
    ("USA", "New York"),
    ("USA", "San Francisco"),
    ("USA", "Austin"),
    ("USA", "Seattle"),
    ("UK", "London"),
    ("UK", "Manchester"),
    ("UAE", "Dubai"),
    ("UAE", "Abu Dhabi"),
    ("France", "Paris"),
    ("Netherlands", "Amsterdam"),
    ("Canada", "Toronto"),
    ("India", "Bangalore"),
    ("Singapore", "Singapore"),
)
COUNTRIES: tuple[str, ...] = tuple(dict.fromkeys(c for c, _ in PLACE_PAIRS))
CITIES: tuple[str, ...] = tuple(dict.fromkeys(city for _, city in PLACE_PAIRS))

CITY_ALIASES: dict[str, str] = {
    "ny": "New York",
    "nyc": "New York",
    "new york city": "New York",
    "sf": "San Francisco",
    "san fran": "San Francisco",
    "bengaluru": "Bangalore",
    "abu dhabbi": "Abu Dhabi",
}
COUNTRY_ALIASES: dict[str, str] = {
    "us": "USA",
    "u.s.": "USA",
    "u.s.a.": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "america": "USA",
    "united kingdom": "UK",
    "britain": "UK",
    "england": "UK",
    "united arab emirates": "UAE",
    "deutschland": "Germany",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
}


def _lookup(values: tuple[str, ...], aliases: dict[str, str]) -> dict[str, str]:
    table = {v.lower(): v for v in values}
    table.update(aliases)
    return table


_CITY_LOOKUP = _lookup(CITIES, CITY_ALIASES)
_COUNTRY_LOOKUP = _lookup(COUNTRIES, COUNTRY_ALIASES)


def _alternation(lookup: dict[str, str]) -> str:
    """Regex alternation, longest first so "New York" wins over "New"."""
    keys = sorted(lookup, key=len, reverse=True)
    return "|".join(re.escape(k) for k in keys)


CITY_ALT = _alternation(_CITY_LOOKUP)
COUNTRY_ALT = _alternation(_COUNTRY_LOOKUP)
PLACE_ALT = _alternation({**_CITY_LOOKUP, **_COUNTRY_LOOKUP})


def _normalise(raw: str | None) -> str:
    return re.sub(r"\s+", " ", (raw or "").strip().strip(".,;:!?")).lower()


def canon_city(raw: str | None) -> str | None:
    return _CITY_LOOKUP.get(_normalise(raw))


def canon_country(raw: str | None) -> str | None:
    return _COUNTRY_LOOKUP.get(_normalise(raw))


def canon_place(raw: str | None) -> tuple[str | None, str | None]:
    """Resolve a bare place phrase to (city, country); at most one is set."""
    key = _normalise(raw)
    if key in _CITY_LOOKUP:
        return _CITY_LOOKUP[key], None
    if key in _COUNTRY_LOOKUP:
        return None, _COUNTRY_LOOKUP[key]
    return None, None


def is_place(raw: str | None) -> bool:
    return any(canon_place(raw))
