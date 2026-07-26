from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        response_json: bool = False,
    ) -> str: ...

    async def complete_with_usage(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        response_json: bool = False,
    ) -> tuple[str, dict[str, int]]: ...
