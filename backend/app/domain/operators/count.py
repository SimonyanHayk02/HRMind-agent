from __future__ import annotations

from typing import Any


class Count:
    name = "count"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> dict[str, int]:
        if data is None:
            return {"count": 0}
        if isinstance(data, dict) and "count" in data:
            return {"count": int(data["count"])}
        if isinstance(data, dict) and "rows" in data:
            return {"count": len(data["rows"])}
        if isinstance(data, (list, tuple, set)):
            return {"count": len(data)}
        return {"count": 1}
