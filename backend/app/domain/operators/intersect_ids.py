from __future__ import annotations

from typing import Any


def _as_id_list(raw: Any) -> list[str]:
    """Normalise a bound id payload into a flat list of UUID strings."""
    if raw is None:
        return []
    if isinstance(raw, (str, int)):
        text = str(raw).strip()
        return [text] if text else []
    if isinstance(raw, dict):
        if isinstance(raw.get("employee_ids"), list):
            return _as_id_list(raw["employee_ids"])
        if raw.get("employee_id") is not None and str(raw.get("employee_id")).strip():
            return [str(raw["employee_id"])]
        if raw.get("id") is not None and str(raw.get("id")).strip():
            return [str(raw["id"])]
        return []
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            out.extend(_as_id_list(item))
        return out
    return []


class IntersectIds:
    name = "intersect_ids"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> list[str]:
        """Intersect ID lists, preserving left (RAG) order.

        Missing ``other`` leaves the left list unchanged. An explicit empty
        ``other`` (e.g. empty location cohort) yields an empty intersection.
        """
        params = params or {}
        left = _as_id_list(data)
        if "other" not in params:
            return left
        other = params.get("other")
        if other is None:
            return []
        right = set(_as_id_list(other))
        return [x for x in left if x in right]
