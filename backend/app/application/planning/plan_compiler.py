from __future__ import annotations

import json
from pathlib import Path

from app.application.planning.heuristic_planner import try_heuristic_plan
from app.application.planning.plan_schema import ExecutionPlan
from app.domain.auth import AuthContext
from app.domain.session import SessionMemory
from app.domain.tools.registry import ToolRegistry
from app.ports.llm import LLMClient


class PlanCompiler:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        prompts_dir: Path | None = None,
        *,
        max_context: int = 8,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._prompts_dir = prompts_dir or Path(__file__).resolve().parents[2] / "prompts"
        self._max_context = max_context

    def _load_prompt(self) -> str:
        path = self._prompts_dir / "planner.md"
        if path.exists():
            return path.read_text()
        return (
            "You are an HR agent planner. Return JSON ExecutionPlan with nodes "
            "(tool|operator), depends_on, response_strategy."
        )

    def _context_packet(self, question: str, auth: AuthContext, memory: SessionMemory | None) -> dict:
        recent: list[dict[str, str]] = []
        entities: list[dict] = []
        constraints: list[dict] = []
        last_ids: list[str] = []
        summary = ""
        last_focus: dict | None = None
        if memory:
            for msg in memory.messages[-self._max_context :]:
                recent.append({"role": msg.role, "content": msg.content[:800]})
            entities = [e.model_dump(mode="json") for e in memory.entity_memory[-20:]]
            constraints = [c.model_dump(mode="json") for c in memory.constraint_memory]
            last_ids = list(memory.last_employee_ids)
            summary = memory.summary or ""
            if memory.last_focus:
                last_focus = memory.last_focus.model_dump(mode="json")
        return {
            "question": question,
            "role": auth.role.value,
            "tools": self._tools.discover(),
            "recent_messages": recent,
            "last_employee_ids": last_ids,
            "last_focus": last_focus,
            "constraints": constraints,
            "entities": entities,
            "summary": summary,
        }

    async def compile(
        self,
        question: str,
        *,
        auth: AuthContext,
        memory: SessionMemory | None = None,
    ) -> ExecutionPlan:
        heuristic = try_heuristic_plan(question, memory=memory)
        if heuristic is not None:
            return heuristic

        user = json.dumps(self._context_packet(question, auth, memory), default=str)
        raw = await self._llm.complete(
            system=self._load_prompt(), user=user, temperature=0.0, response_json=True
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        return ExecutionPlan.model_validate(data)
