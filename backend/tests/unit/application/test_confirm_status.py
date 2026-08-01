from __future__ import annotations

from app.application.understanding.status_change import is_confirm_status_update


def test_is_confirm_status_update() -> None:
    assert is_confirm_status_update("confirm status update")
    assert is_confirm_status_update("Confirm status update.")
    assert is_confirm_status_update("yes, confirm")
    assert not is_confirm_status_update("set Alice status to true")
    assert not is_confirm_status_update("who knows Python?")
