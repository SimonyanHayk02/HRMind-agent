from __future__ import annotations

import hashlib
import json
from typing import Any

from app.domain.auth import AuthContext
from app.domain.errors import CacheKeyError


def build(namespace: str, auth: AuthContext | None, **parts: Any) -> str:
    if namespace != "embedding" and auth is None:
        raise CacheKeyError("AuthContext is required for cache key")
    material: dict[str, Any] = {"ns": namespace}
    if auth is not None:
        material["tenant_id"] = auth.tenant_id
        material["role"] = auth.role.value
        material["permission_hash"] = auth.permission_hash
    for key, value in sorted(parts.items()):
        material[key] = value
    raw = json.dumps(material, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"hrmind:{namespace}:{digest}"


def embedding_key(*, model: str, text: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"hrmind:embedding:{model}:{text_hash}"
