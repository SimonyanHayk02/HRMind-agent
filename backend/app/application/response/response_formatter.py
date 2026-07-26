from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan
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
            return plan.clarify_question, 0.4, [], plan.clarify_question

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

        if plan.response_strategy == "template":
            for p in reversed(payloads):
                if isinstance(p, dict) and "count" in p and p["count"] is not None:
                    return f"The answer is {p['count']}.", confidence, sources, None
                if isinstance(p, dict) and "row_count" in p and "rows" not in p:
                    return f"The answer is {p['row_count']}.", confidence, sources, None
            if tool_errors:
                detail = "; ".join(tool_errors)
                return (
                    f"I couldn't finish computing that answer ({detail}).",
                    0.0,
                    sources,
                    None,
                )
            if not payloads:
                return "No results.", confidence, sources, None
            # Avoid dumping raw ID lists as the user-facing answer
            last = payloads[-1]
            if isinstance(last, list) and last and all(isinstance(x, str) for x in last):
                return f"Found {len(last)} matching employees.", confidence, sources, None
            return json.dumps(last, default=str), confidence, sources, None

        path = self._prompts_dir / "response.md"
        system = (
            path.read_text()
            if path.exists()
            else "Format a concise HR answer using ONLY the provided tool results. Do not invent facts."
        )
        user = json.dumps({"question": question, "results": payloads}, default=str)
        answer = await self._llm.complete(system=system, user=user, temperature=0.0)
        if not answer or answer.strip().lower() in {"null", "none"}:
            if payloads:
                return json.dumps(payloads[-1], default=str), confidence, sources, None
            return "I could not find matching data.", confidence, sources, None
        return answer, confidence, sources, None
