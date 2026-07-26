from __future__ import annotations

from app.domain.errors import NotFoundError
from app.domain.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.meta.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise NotFoundError(f"Unknown tool: {name}")
        return tool

    def discover(self) -> list[dict]:
        return [t.meta.model_dump() for t in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools.keys())
