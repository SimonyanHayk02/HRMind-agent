from __future__ import annotations

from typing import Any


class MergeResults:
    name = "merge_results"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        parts = params.get("parts") or (data if isinstance(data, list) else [data])
        merged: dict[str, Any] = {}
        for part in parts:
            if isinstance(part, dict):
                merged.update(part)
        return merged
