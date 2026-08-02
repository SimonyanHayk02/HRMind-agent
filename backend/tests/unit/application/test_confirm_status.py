from __future__ import annotations

from uuid import uuid4

from app.application.planning.heuristic_planner import _match_entity_name
from app.application.understanding.status_change import (
    is_cancel_status_update,
    is_confirm_status_update,
)
from app.domain.enums import Role
from app.domain.services.entity_resolver import EntityResolver
from app.domain.session import EntityRef, SessionMemory


def test_is_confirm_status_update() -> None:
    assert is_confirm_status_update("confirm status update")
    assert is_confirm_status_update("Confirm status update.")
    assert is_confirm_status_update("yes, confirm")
    assert is_confirm_status_update("yes")
    assert not is_confirm_status_update("set Alice status to true")
    assert not is_confirm_status_update("who knows Python?")
    assert not is_confirm_status_update("no")
    assert not is_confirm_status_update("cancel")


def test_is_cancel_status_update() -> None:
    assert is_cancel_status_update("no")
    assert is_cancel_status_update("cancel")
    assert is_cancel_status_update("no wait cancel")
    assert is_cancel_status_update("never mind")
    assert not is_cancel_status_update("yes")
    assert not is_cancel_status_update("activate Carol Garcia")


def test_entity_resolver_full_name_not_stolen_by_first_alias() -> None:
    bauer = uuid4()
    refs = [
        EntityRef(
            employee_id=bauer,
            display_name="Alice Bauer",
            aliases=["Alice"],
        )
    ]
    resolver = EntityResolver()
    found, _conf = resolver.resolve_from_refs("Alice Nguyen", refs)
    assert found is None
    found_ada, conf = resolver.resolve_from_refs("Alice", refs)
    assert found_ada == bauer
    assert conf > 0


def test_match_entity_name_prefers_explicit_full_name() -> None:
    bauer = uuid4()
    mem = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        entity_memory=[
            EntityRef(
                employee_id=bauer,
                display_name="Alice Bauer",
                aliases=["Alice"],
            )
        ],
    )
    assert _match_entity_name("Tell me about Alice Nguyen", mem) is None
    assert _match_entity_name("Tell me about Alice Bauer", mem) == "Alice Bauer"
