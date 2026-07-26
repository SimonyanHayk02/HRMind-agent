from __future__ import annotations

from typing import Any


class IntersectIds:
    name = "intersect_ids"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> list[str]:
        params = params or {}
        left = [str(x) for x in (data or [])]
        right = [str(x) for x in params.get("other", [])]
        return sorted(set(left) & set(right))
