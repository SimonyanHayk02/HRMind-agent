from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from uuid import uuid4

from app.adapters.telemetry.redaction import redact_dict


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        event = {"id": str(uuid4()), "name": name, "attrs": redact_dict(attrs)}
        self.events.append(event)
        try:
            yield event
        except Exception as exc:
            event["error"] = str(exc)
            raise


def get_tracer() -> TraceRecorder:
    return TraceRecorder()
