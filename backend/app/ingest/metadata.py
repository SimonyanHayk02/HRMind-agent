from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def build_chunk_metadata(
    *,
    employee_id: str,
    employee_name: str,
    document: str,
    section: str,
    chunk_index: int,
    position: str | None = None,
    department: str | None = None,
) -> dict[str, Any]:
    """Chunk metadata.

    Deliberately no ``city``/``country``: location is a resume-sourced fact, and
    copying it into metadata would recreate the duplicate source of truth this
    design removes. The place still appears in the enriched chunk *text*, which is
    what gets embedded and cited.
    """
    meta: dict[str, Any] = {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "document": document,
        "section": section,
        "chunk_index": chunk_index,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if position:
        meta["position"] = position
    if department:
        meta["department"] = department
    return meta
