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
    def __init__(self, llm: LLMClient, tools: ToolRegistry, prompts_dir: Path | None = None) -> None:
        self._llm = llm
        self._tools = tools
        self._prompts_dir = prompts_dir or Path(__file__).resolve().parents[2] / "prompts"

    def _load_prompt(self) -> str:
        path = self._prompts_dir / "planner.md"
        if path.exists():
            return path.read_text()
        return (
            "You are an HR agent planner. Return JSON ExecutionPlan with nodes "
            "(tool|operator), depends_on, response_strategy."
        )

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

        tool_catalog = json.dumps(self._tools.discover(), default=str)
        entities = []
        if memory:
            entities = [e.model_dump(mode="json") for e in memory.entity_memory]
        user = json.dumps(
            {
                "question": question,
                "role": auth.role.value,
                "tools": tool_catalog,
                "entities": entities,
                "summary": memory.summary if memory else "",
            },
            default=str,
        )
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
