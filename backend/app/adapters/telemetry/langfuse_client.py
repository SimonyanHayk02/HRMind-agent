from __future__ import annotations

from typing import Any

from app.config.settings import Settings


class LangfuseClient:
    def __init__(self, settings: Settings) -> None:
        self._enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        self._events: list[dict[str, Any]] = []

    def trace(self, name: str, payload: dict[str, Any]) -> None:
        if not self._enabled:
            self._events.append({"name": name, "payload": payload})
            return
        # Integration hook — store locally if SDK not installed
        self._events.append({"name": name, "payload": payload})
