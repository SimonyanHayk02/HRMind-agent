from __future__ import annotations

from typing import Any

from app.config.logging import get_logger
from app.domain.auth import AuthContext

logger = get_logger("audit")


def audit_sensitive_access(
    auth: AuthContext, *, resource: str, fields: list[str], meta: dict[str, Any] | None = None
) -> None:
    logger.info(
        "sensitive_access",
        user_id=auth.user_id,
        role=auth.role.value,
        tenant_id=auth.tenant_id,
        resource=resource,
        fields=fields,
        meta=meta or {},
    )
