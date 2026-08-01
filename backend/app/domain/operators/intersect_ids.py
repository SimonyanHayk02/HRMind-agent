from __future__ import annotations

from typing import Any


class IntersectIds:
    name = "intersect_ids"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> list[str]:
        """Intersect ID lists, preserving left (RAG) order.

        Missing ``other`` leaves the left list unchanged. An explicit empty
        ``other`` (e.g. empty location cohort) yields an empty intersection.
        """
        params = params or {}
        left = [str(x) for x in (data or [])]
        if "other" not in params:
            return left
        other = params.get("other")
        if other is None:
            return []
        if isinstance(other, (str, int)):
            right = {str(other)}
        else:
            right = {str(x) for x in other}
        return [x for x in left if x in right]
