from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.deps import parse_role
from app.domain.enums import Role


def test_parse_role_aliases() -> None:
    assert parse_role("admin") == Role.RECRUITER
    assert parse_role("hr_manager") == Role.RECRUITER
    assert parse_role("Recruiter") == Role.RECRUITER
    assert parse_role("MANAGER") == Role.MANAGER
    assert parse_role("") == Role.RECRUITER
    assert parse_role(None) == Role.RECRUITER


def test_parse_role_rejects_unknown() -> None:
    with pytest.raises(HTTPException) as exc:
        parse_role("superuser")
    assert exc.value.status_code == 400
