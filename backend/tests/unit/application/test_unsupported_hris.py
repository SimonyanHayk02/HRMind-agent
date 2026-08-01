from __future__ import annotations

from app.application.planning.unsupported import (
    is_unsupported_topic,
    unsupported_answer_for,
)


def test_unsupported_hris_keeps_refusal_with_alternative() -> None:
    assert is_unsupported_topic("how much PTO does she have?")
    ans = unsupported_answer_for("how much PTO does she have?")
    assert "not connected yet" in ans.lower() or "hris" in ans.lower()
    assert "leave" in ans.lower() or "hire" in ans.lower()


def test_unsupported_does_not_soft_answer_benefits() -> None:
    assert is_unsupported_topic("what are her dental benefits?")
    ans = unsupported_answer_for("what are her dental benefits?")
    assert "don't have" in ans.lower() or "not connected" in ans.lower()
