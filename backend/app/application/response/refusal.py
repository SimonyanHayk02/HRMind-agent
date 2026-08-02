"""Typed refusal outcomes for turns the agent cannot (or must not) answer.

Runtime resolver is the source of truth. Planners may *hint* via
``ExecutionPlan.refusal_code``; empty/missing/tool failures are classified
after tools run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan
from app.application.planning.unsupported import (
    UNSUPPORTED_ANSWER,
    is_count_question,
    is_unsupported_topic,
)
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.tools.base import ToolResult

OUT_OF_SCOPE_ANSWER = UNSUPPORTED_ANSWER

UNAUTHORIZED_ANSWER = (
    "You don't have access to that information with your current role. "
    "I can still help with profiles, departments, locations, hire dates, "
    "managers, and skills from resumes — within what you're allowed to see."
)

TOOL_ERROR_ANSWER = (
    "I couldn't complete that request because a backend tool failed. "
    "Please try again in a moment."
)

EMPTY_REFINE_ANSWER = "None of the previous set match that criteria."
EMPTY_SEARCH_ANSWER = "I couldn't find matching employees for that."
EMPTY_ZERO_ANSWER = "The answer is 0."


def empty_refine_answer(question: str | None = None) -> str:
    """Honest empty refine; skill asks mention the skill so RAG checkers stay green."""
    from app.tools.resume_search.skills import extract_skill

    skill = extract_skill(question or "")
    if skill:
        return f"None of the previous set know {skill}."
    return EMPTY_REFINE_ANSWER

_MISSING_DATA_RE = re.compile(
    r"couldn't find a (date of birth|work location)|"
    r"no date of birth recorded|"
    r"don't have a location on file|"
    r"could not find a date of birth",
    re.I,
)
_AMBIGUOUS_RE = re.compile(
    r"\b(which|whose|who did you mean|tell me which|need a bit more|"
    r"names first|ask for their names|which person|which employee)\b",
    re.I,
)
_UNAUTHORIZED_ERR_RE = re.compile(
    r"Tool not permitted:|Column not allowed|cannot use tool|Forbidden",
    re.I,
)
_PRIOR_RE = re.compile(
    r"\b(them|those|that set|of them|which of them|previous set)\b", re.I
)


class RefusalCode(StrEnum):
    OK = "ok"
    OUT_OF_SCOPE = "out_of_scope"
    MISSING_DATA = "missing_data"
    EMPTY_COHORT = "empty_cohort"
    AMBIGUOUS = "ambiguous"
    UNAUTHORIZED = "unauthorized"
    TOOL_ERROR = "tool_error"


# Cohort / list / bindings must survive these outcomes.
PRESERVE_COHORT_CODES = frozenset(
    {
        RefusalCode.OUT_OF_SCOPE,
        RefusalCode.UNAUTHORIZED,
        RefusalCode.AMBIGUOUS,
        RefusalCode.TOOL_ERROR,
        RefusalCode.MISSING_DATA,
    }
)


@dataclass(frozen=True)
class RefusalOutcome:
    code: RefusalCode
    message: str
    confidence: float = 0.4
    ui_clarify: str | None = None
    preserve_cohort: bool = True


def out_of_scope_outcome(message: str | None = None) -> RefusalOutcome:
    return RefusalOutcome(
        code=RefusalCode.OUT_OF_SCOPE,
        message=message or OUT_OF_SCOPE_ANSWER,
        confidence=0.4,
        ui_clarify=None,
        preserve_cohort=True,
    )


def unauthorized_outcome(message: str | None = None) -> RefusalOutcome:
    return RefusalOutcome(
        code=RefusalCode.UNAUTHORIZED,
        message=message or UNAUTHORIZED_ANSWER,
        confidence=0.4,
        ui_clarify=None,
        preserve_cohort=True,
    )


def ambiguous_outcome(message: str) -> RefusalOutcome:
    return RefusalOutcome(
        code=RefusalCode.AMBIGUOUS,
        message=message,
        confidence=0.4,
        ui_clarify=message,
        preserve_cohort=True,
    )


def unauthorized_plan(message: str | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        nodes=[],
        response_strategy="template",
        clarify_question=message or UNAUTHORIZED_ANSWER,
        refusal_code=RefusalCode.UNAUTHORIZED.value,
    )


def out_of_scope_plan(message: str | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        nodes=[],
        response_strategy="template",
        clarify_question=message or OUT_OF_SCOPE_ANSWER,
        refusal_code=RefusalCode.OUT_OF_SCOPE.value,
    )


def is_soft_unauthorized_error(message: str) -> bool:
    """Whitelist: only ACL denials become chat refuses (not cycles / bad plans)."""
    msg = message or ""
    return bool(
        msg.startswith("Tool not permitted:")
        or "Column not allowed for role:" in msg
        or re.search(r"Role .+ cannot use tool", msg)
    )


def resolve_refusal(
    plan: ExecutionPlan,
    state: GraphState,
    question: str,
    *,
    auth: AuthContext | None = None,
) -> RefusalOutcome | None:
    """Classify unanswerable turns. Return None when normal formatting should run.

    ``ok`` with a typed zero is left to the formatter (returns None here) except
    refine-empty paths which become ``empty_cohort``.
    """
    hint = (plan.refusal_code or "").strip().lower()
    if hint == RefusalCode.UNAUTHORIZED:
        return unauthorized_outcome(plan.clarify_question)
    if hint == RefusalCode.OUT_OF_SCOPE:
        return out_of_scope_outcome(plan.clarify_question)
    if hint == RefusalCode.AMBIGUOUS and plan.clarify_question:
        return ambiguous_outcome(plan.clarify_question)

    if not plan.nodes and plan.clarify_question:
        text = plan.clarify_question
        if is_unsupported_topic(question) or text.strip() == OUT_OF_SCOPE_ANSWER:
            return out_of_scope_outcome(text)
        if hint == RefusalCode.UNAUTHORIZED or text.strip() == UNAUTHORIZED_ANSWER:
            return unauthorized_outcome(text)
        if "don't have access" in text.lower():
            return unauthorized_outcome(text)
        if _AMBIGUOUS_RE.search(text) or hint == RefusalCode.AMBIGUOUS:
            return ambiguous_outcome(text)
        # Informational empty-node (e.g. legacy unsupported) — treat as OOS if
        # it matches the standard refuse blurb, else ambiguous clarify.
        if "don't have that information" in text.lower():
            return out_of_scope_outcome(text)
        return ambiguous_outcome(text)

    if is_unsupported_topic(question):
        return out_of_scope_outcome()

    payloads, tool_errors, has_usable = _collect_payloads(plan, state)

    # Tool-side clarify (namesakes / which person) — prefer UI clarify.
    for node in plan.nodes:
        result = state.node_results.get(node.id)
        if not isinstance(result, ToolResult) or not isinstance(result.data, dict):
            continue
        clarify = result.data.get("clarify")
        if clarify:
            return ambiguous_outcome(str(clarify))

    # Unauthorized / forbidden from tool errors.
    for err in tool_errors:
        if is_soft_unauthorized_error(err) or _UNAUTHORIZED_ERR_RE.search(err):
            return unauthorized_outcome()

    # Salary asked but ACL stripped it from the payload (cross-dept manager).
    salary_deny = _salary_stripped_unauthorized(question, auth, payloads)
    if salary_deny is not None:
        return salary_deny

    # Hard tool failure with nothing to show.
    if state.degraded and not has_usable:
        detail = "; ".join(tool_errors or state.errors)
        msg = (
            f"I couldn't complete that request ({detail}). "
            "Check that Postgres is running, then try again."
            if detail
            else TOOL_ERROR_ANSWER
        )
        return RefusalOutcome(
            code=RefusalCode.TOOL_ERROR,
            message=msg,
            confidence=0.0,
            preserve_cohort=True,
        )

    # Missing resume fact already worded by attribute tools.
    for p in payloads:
        if isinstance(p, dict):
            answer = str(p.get("answer") or "")
            if answer and _MISSING_DATA_RE.search(answer):
                return RefusalOutcome(
                    code=RefusalCode.MISSING_DATA,
                    message=answer,
                    confidence=0.5,
                    preserve_cohort=True,
                )

    # Empty refine cohort (intersect / prior-set filter) — typed empty, not OOS.
    empty = _empty_cohort_message(question, plan, payloads)
    if empty is not None:
        return RefusalOutcome(
            code=RefusalCode.EMPTY_COHORT,
            message=empty,
            confidence=0.85,
            preserve_cohort=False,
        )

    return None


def has_format_evidence(plan: ExecutionPlan, state: GraphState) -> bool:
    """True when tool results contain grounded facts for LLM / template formatting."""
    _, _, has_usable = _collect_payloads(plan, state)
    return has_usable


def _salary_stripped_unauthorized(
    question: str,
    auth: AuthContext | None,
    payloads: list[Any],
) -> RefusalOutcome | None:
    """Manager saw a person record but salary was redacted by column policy."""
    if auth is None or auth.role not in {Role.MANAGER, Role.EMPLOYEE}:
        return None
    # Avoid circular import: detect locally (salary_acl also uses unsupported).
    if not re.search(r"\b(salary|salaries|base\s*pay|compensation)\b", question or "", re.I):
        return None
    if is_unsupported_topic(question):
        return None
    saw_person = False
    saw_salary = False
    for p in payloads:
        if not isinstance(p, dict):
            continue
        if p.get("salary") is not None:
            saw_salary = True
        if p.get("id") or p.get("employee_id") or p.get("full_name"):
            saw_person = True
        for row in p.get("rows") or []:
            if isinstance(row, dict):
                if row.get("salary") is not None:
                    saw_salary = True
                if row.get("id") or row.get("full_name"):
                    saw_person = True
    if saw_person and not saw_salary:
        return unauthorized_outcome(
            "You don't have access to salary information for that employee."
        )
    return None


def _collect_payloads(
    plan: ExecutionPlan, state: GraphState
) -> tuple[list[Any], list[str], bool]:
    payloads: list[Any] = []
    tool_errors: list[str] = []
    has_usable = False
    for node in plan.nodes:
        result = state.node_results.get(node.id)
        if isinstance(result, ToolResult):
            if result.error:
                tool_errors.append(f"{node.name}: {result.error}")
            if result.data is not None:
                payloads.append(result.data)
                if _payload_has_evidence(result.data):
                    has_usable = True
        elif result is not None:
            payloads.append(result)
            if _payload_has_evidence(result):
                has_usable = True
    return payloads, tool_errors, has_usable


def _payload_has_evidence(data: Any) -> bool:
    if data is None:
        return False
    if isinstance(data, list):
        return bool(data)
    if not isinstance(data, dict):
        return True
    if data.get("clarify"):
        return True
    if data.get("answer"):
        return True
    if data.get("count") is not None:
        return True
    if isinstance(data.get("rows"), list) and data["rows"]:
        return True
    if isinstance(data.get("employees"), list) and data["employees"]:
        return True
    if isinstance(data.get("employee_ids"), list) and data["employee_ids"]:
        return True
    # Score-gated empty RAG must not unlock llm_format.
    if data.get("score_gated") and not data.get("employee_ids"):
        return False
    if isinstance(data.get("hits"), list) and data["hits"]:
        return True
    if isinstance(data.get("facts"), list) and data["facts"]:
        return True
    return False


def _empty_cohort_message(
    question: str, plan: ExecutionPlan, payloads: list[Any]
) -> str | None:
    """Return empty-cohort prose when refine/intersect matched nobody.

    Org-wide / facet zeros return None so the formatter emits typed ``ok`` zeros.
    """
    q = (question or "").lower()
    prior = bool(_PRIOR_RE.search(q)) or any(
        n.name == "intersect_ids"
        or (n.params or {}).get("employee_ids")
        or (n.params or {}).get("other")
        for n in plan.nodes
    )
    if not prior:
        return None

    count_asked = is_count_question(question) or any(
        n.name == "sql" and (n.params or {}).get("count_only") for n in plan.nodes
    )
    # Facet aggregations are not "them" refinements.
    if any(
        (n.params or {}).get("facet") or (n.params or {}).get("count_distinct")
        for n in plan.nodes
    ):
        return None

    refine = empty_refine_answer(question)
    for p in reversed(payloads):
        if isinstance(p, dict) and p.get("count") is not None:
            try:
                if int(p["count"]) == 0:
                    return EMPTY_ZERO_ANSWER if count_asked else refine
            except (TypeError, ValueError):
                continue
        if isinstance(p, dict) and isinstance(p.get("rows"), list) and not p["rows"]:
            return EMPTY_ZERO_ANSWER if count_asked else refine
        if isinstance(p, list) and not p:
            return EMPTY_ZERO_ANSWER if count_asked else refine
        if (
            isinstance(p, dict)
            and isinstance(p.get("employee_ids"), list)
            and not p["employee_ids"]
            and not p.get("answer")
            and not p.get("facts")
        ):
            return EMPTY_ZERO_ANSWER if count_asked else refine
    return None


def empty_unscoped_fallback() -> str:
    """When tools return nothing useful but the topic is in-scope."""
    return EMPTY_SEARCH_ANSWER
