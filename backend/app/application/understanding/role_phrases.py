from __future__ import annotations

import re

# Natural job-role phrases → one or more concrete position titles in the DB.
ROLE_PHRASES: list[tuple[str, list[str]]] = [
    ("software developers", ["Software Engineer"]),
    ("software developer", ["Software Engineer"]),
    ("software engineers", ["Software Engineer"]),
    ("software engineer", ["Software Engineer"]),
    ("senior engineers", ["Senior Engineer"]),
    ("senior engineer", ["Senior Engineer"]),
    ("staff engineers", ["Staff Engineer"]),
    ("staff engineer", ["Staff Engineer"]),
    ("engineering managers", ["Engineering Manager"]),
    ("engineering manager", ["Engineering Manager"]),
    ("product managers", ["Product Manager"]),
    ("product manager", ["Product Manager"]),
    ("product designers", ["Product Designer"]),
    ("product designer", ["Product Designer"]),
    ("developers", ["Software Engineer", "Senior Engineer", "Staff Engineer"]),
    ("developer", ["Software Engineer"]),
    ("engineers", ["Software Engineer", "Senior Engineer", "Staff Engineer"]),
    ("engineer", ["Software Engineer", "Senior Engineer", "Staff Engineer"]),
]


def match_role_positions(question: str) -> str | list[str] | None:
    """Map natural role phrases to concrete position title(s), or None."""
    lower = (question or "").lower()
    for phrase, positions in sorted(ROLE_PHRASES, key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", lower):
            return positions[0] if len(positions) == 1 else list(positions)
    return None
