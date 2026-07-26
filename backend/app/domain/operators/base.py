from __future__ import annotations

from typing import Any, Protocol


class Operator(Protocol):
    name: str

    def run(self, data: Any, params: dict[str, Any] | None = None) -> Any: ...


class OperatorRegistry:
    def __init__(self) -> None:
        self._ops: dict[str, Operator] = {}

    def register(self, op: Operator) -> None:
        self._ops[op.name] = op

    def get(self, name: str) -> Operator:
        if name not in self._ops:
            raise KeyError(name)
        return self._ops[name]

    def names(self) -> list[str]:
        return sorted(self._ops.keys())
