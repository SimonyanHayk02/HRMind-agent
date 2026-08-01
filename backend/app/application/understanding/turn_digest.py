"""Compact session digest for residual slot extraction — not a full planner packet."""
from __future__ import annotations

from typing import Any

from app.application.memory.context_view import (
    ContextNeed,
    classify_context_need,
    redact_turn_digest,
)
from app.domain.session import SessionMemory

_MAX_TURNS = 3
_MAX_MSG_CHARS = 240
_MAX_LISTED = 10


def build_turn_digest(
    question: str,
    memory: SessionMemory | None,
    *,
    max_turns: int = _MAX_TURNS,
    context_need: ContextNeed | str | None = None,
) -> dict[str, Any]:
    """Small context for the slot LLM: recent Q/A + actionable working memory."""
    need = (
        ContextNeed(context_need)
        if isinstance(context_need, str)
        else context_need
        if context_need is not None
        else classify_context_need(question, memory)
    )

    recent: list[dict[str, str]] = []
    last_listed: list[dict[str, Any]] = []
    bindings: dict[str, str] = {}
    last_focus: dict[str, Any] | None = None
    cohort_size = 0

    if memory:
        for msg in memory.messages[-max_turns * 2 :]:
            recent.append(
                {
                    "role": msg.role,
                    "content": (msg.content or "")[:_MAX_MSG_CHARS],
                }
            )
        last_listed = [
            {
                "index": i + 1,
                "employee_id": str(e.employee_id),
                "display_name": e.display_name,
            }
            for i, e in enumerate(memory.last_listed[:_MAX_LISTED])
        ]
        bindings = dict(list(memory.person_bindings.items())[:12])
        if memory.last_focus:
            last_focus = memory.last_focus.model_dump(mode="json")
        cohort_size = len(memory.last_employee_ids)

    digest = {
        "question": (question or "").strip(),
        "recent_messages": recent,
        "last_listed": last_listed,
        "person_bindings": bindings,
        "last_focus": last_focus,
        "cohort_size": cohort_size,
        "has_prior_cohort": cohort_size > 0,
    }
    return redact_turn_digest(digest, need)
