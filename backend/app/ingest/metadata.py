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
) -> dict[str, Any]:
    return {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "document": document,
        "section": section,
        "chunk_index": chunk_index,
        "created_at": datetime.now(UTC).isoformat(),
    }
