"""Chunk text enrichment.

A raw ``Personal`` chunk reads ``Date of Birth: 12 March 1991`` — it carries no
name, so every such chunk in the corpus is near-identical in embedding space and
dense retrieval cannot tell one person from another. Prepending the entity
context makes each chunk self-contained, which is what lets both embedding and
full-text search resolve people, and what makes a cited snippet readable on its
own.
"""
from __future__ import annotations


def build_context_header(
    *,
    employee_name: str,
    position: str | None = None,
    department: str | None = None,
    city: str | None = None,
    country: str | None = None,
) -> str:
    """One line identifying whose resume this is, e.g.
    ``Eva Kim — Senior Engineer, Engineering (Dubai, UAE)``."""
    header = (employee_name or "").strip() or "Unknown employee"
    role = ", ".join(p for p in (position, department) if p)
    if role:
        header = f"{header} — {role}"
    location = ", ".join(p for p in (city, country) if p)
    if location:
        header = f"{header} ({location})"
    return header


def build_chunk_text(
    content: str,
    *,
    section: str,
    employee_name: str,
    position: str | None = None,
    department: str | None = None,
    city: str | None = None,
    country: str | None = None,
) -> str:
    """Chunk body prefixed with entity context and its section name."""
    header = build_context_header(
        employee_name=employee_name,
        position=position,
        department=department,
        city=city,
        country=country,
    )
    lines = [header]
    if section:
        lines.append(section)
    lines.append(content.strip())
    return "\n".join(lines)
