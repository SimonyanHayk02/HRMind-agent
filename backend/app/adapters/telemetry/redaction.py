from __future__ import annotations

import re
from typing import Any

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_SALARY_KEYS = {"salary", "compensation", "pay"}


def redact_text(text: str) -> str:
    return _EMAIL_RE.sub("[REDACTED_EMAIL]", text)


def redact_dict(data: Any) -> Any:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            if key.lower() in _SALARY_KEYS:
                out[key] = "[REDACTED]"
            elif key.lower() in {"content", "resume_text", "chunk_text"} and isinstance(value, str):
                out[key] = f"[REDACTED_RESUME len={len(value)}]"
            else:
                out[key] = redact_dict(value)
        return out
    if isinstance(data, list):
        return [redact_dict(v) for v in data]
    if isinstance(data, str):
        return redact_text(data)
    return data
