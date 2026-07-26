from __future__ import annotations

from typing import Any


class IntersectIds:
    name = "intersect_ids"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> list[str]:
        """Intersect ID lists, preserving left (RAG) order."""
        params = params or {}
        left = [str(x) for x in (data or [])]
        right = {str(x) for x in params.get("other", [])}
        if not right:
            return left
        return [x for x in left if x in right]
