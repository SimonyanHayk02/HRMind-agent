from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan
from app.application.planning.unsupported import (
    UNSUPPORTED_ANSWER,
    is_count_question,
    is_unsupported_topic,
)
from app.domain.tools.base import SourceRef, ToolResult
from app.ports.llm import LLMClient


class ResponseFormatter:
    def __init__(self, llm: LLMClient, prompts_dir: Path | None = None) -> None:
        self._llm = llm
        self._prompts_dir = prompts_dir or Path(__file__).resolve().parents[2] / "prompts"

    async def format(
        self, question: str, plan: ExecutionPlan, state: GraphState
    ) -> tuple[str, float, list[SourceRef], str | None]:
        if plan.clarify_question:
            # Empty-node plans are informational (e.g. unsupported topic), not UI clarifies.
            clarify = None if not plan.nodes else plan.clarify_question
            return plan.clarify_question, 0.4, [], clarify

        if is_unsupported_topic(question):
            return UNSUPPORTED_ANSWER, 0.4, [], None

        payloads: list[Any] = []
        sources: list[SourceRef] = []
        confidence = 1.0
        tool_errors: list[str] = []

        for node in plan.nodes:
            result = state.node_results.get(node.id)
            if isinstance(result, ToolResult):
                if result.error:
                    tool_errors.append(f"{node.name}: {result.error}")
                if result.data is not None:
                    payloads.append(result.data)
                sources.extend(result.sources)
                confidence = min(confidence, result.confidence)
                if result.data and isinstance(result.data, dict) and result.data.get("clarify"):
                    return (
                        str(result.data["clarify"]),
                        result.confidence,
                        sources,
                        str(result.data["clarify"]),
                    )
                if result.data and isinstance(result.data, dict) and result.data.get("answer"):
                    if plan.response_strategy == "template" or node.name == "greeting":
                        return str(result.data["answer"]), result.confidence, sources, None
            elif result is not None:
                payloads.append(result)

        if state.degraded and not payloads:
            detail = "; ".join(tool_errors or state.errors) or "a backend tool failed"
            return (
                f"I couldn't complete that request ({detail}). "
                "Check that Postgres is running, then try again.",
                0.0,
                sources,
                None,
            )

        count_asked = is_count_question(question) or _plan_is_count_only(plan)

        if plan.response_strategy == "template":
            for p in reversed(payloads):
                if (
                    count_asked
                    and isinstance(p, dict)
                    and "count" in p
                    and p["count"] is not None
                ):
                    return f"The answer is {p['count']}.", confidence, sources, None
                if (
                    count_asked
                    and isinstance(p, dict)
                    and "row_count" in p
                    and "rows" not in p
                ):
                    return f"The answer is {p['row_count']}.", confidence, sources, None
                if isinstance(p, dict) and isinstance(p.get("rows"), list):
                    named = _format_employee_rows(p["rows"])
                    if named:
                        return named, confidence, sources, None
            if tool_errors:
                detail = "; ".join(tool_errors)
                return (
                    f"I couldn't finish computing that answer ({detail}).",
                    0.0,
                    sources,
                    None,
                )
            if not payloads:
                return UNSUPPORTED_ANSWER, 0.3, sources, None
            last = payloads[-1]
            if isinstance(last, list) and last and all(isinstance(x, str) for x in last):
                return f"Found {len(last)} matching employees.", confidence, sources, None
            # Do not turn an unscoped employee dump into a fake numeric answer
            if _looks_unscoped_dump(last) and not count_asked:
                return UNSUPPORTED_ANSWER, 0.3, sources, None
            return json.dumps(last, default=str), confidence, sources, None

        path = self._prompts_dir / "response.md"
        system = (
            path.read_text()
            if path.exists()
            else (
                "Format a concise HR answer using ONLY the provided tool results. "
                "Do not invent facts. If results cannot answer the question, say you "
                "don't have that information."
            )
        )
        user = json.dumps({"question": question, "results": payloads}, default=str)
        answer = await self._llm.complete(system=system, user=user, temperature=0.0)
        if not answer or answer.strip().lower() in {"null", "none"}:
            if not payloads or _looks_unscoped_dump(payloads[-1]):
                return UNSUPPORTED_ANSWER, 0.3, sources, None
            return UNSUPPORTED_ANSWER, 0.3, sources, None
        # Guard: LLM sometimes answers headcount when asked about missing domains
        if _is_spurious_headcount_answer(answer, question, payloads):
            return UNSUPPORTED_ANSWER, 0.3, sources, None
        return answer, confidence, sources, None


def _plan_is_count_only(plan: ExecutionPlan) -> bool:
    return any(
        n.name == "sql" and bool((n.params or {}).get("count_only")) for n in plan.nodes
    )


def _looks_unscoped_dump(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return False
    sql = str(payload.get("sql") or "")
    if "WHERE 1=1" in sql and "employee_ids" not in sql.lower() and len(rows) >= 20:
        return True
    return False


def _is_spurious_headcount_answer(answer: str, question: str, payloads: list[Any]) -> bool:
    if is_unsupported_topic(question):
        return True
    if is_count_question(question) and not is_unsupported_topic(question):
        # Legitimate count questions may answer with a number
        return False
    text = answer.strip().lower()
    if text.startswith("the answer is ") and text.rstrip(".").split()[-1].isdigit():
        if any(_looks_unscoped_dump(p) for p in payloads):
            return True
    return False


def _format_employee_rows(rows: list[Any]) -> str | None:
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        first = str(row.get("first_name") or "").strip()
        last = str(row.get("last_name") or "").strip()
        full = f"{first} {last}".strip()
        if not full:
            continue
        dept = str(row.get("department") or "").strip()
        pos = str(row.get("position") or "").strip()
        extra = ", ".join(x for x in (pos, dept) if x)
        names.append(f"{full} ({extra})" if extra else full)
    if not names:
        return None
    if len(names) == 1:
        return f"The matching employee is {names[0]}."
    bullet = "\n".join(f"- {n}" for n in names)
    return f"Here are the {len(names)} matching employees:\n{bullet}"
