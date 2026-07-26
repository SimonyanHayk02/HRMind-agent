from __future__ import annotations

from typing import Any


class Dedupe:
    name = "dedupe"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> list[Any]:
        items = data if isinstance(data, list) else [data]
        key = (params or {}).get("key")
        seen: set[Any] = set()
        out: list[Any] = []
        for item in items:
            k = item.get(key) if key and isinstance(item, dict) else item
            marker = str(k)
            if marker in seen:
                continue
            seen.add(marker)
            out.append(item)
        return out
