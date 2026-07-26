from __future__ import annotations

from typing import Any


def build_structured_context(hits: list[dict[str, Any]]) -> dict[str, Any]:
    people: dict[str, dict[str, Any]] = {}
    for hit in hits:
        eid = str(hit.get("employee_id"))
        entry = people.setdefault(
            eid,
            {
                "employee_id": eid,
                "employee_name": hit.get("employee_name") or hit.get("metadata", {}).get("employee_name"),
                "snippets": [],
            },
        )
        entry["snippets"].append(
            {
                "section": hit.get("section") or hit.get("metadata", {}).get("section"),
                "text": hit.get("content", "")[:500],
                "score": hit.get("score"),
            }
        )
    return {"hits": list(people.values()), "employee_ids": list(people.keys())}
