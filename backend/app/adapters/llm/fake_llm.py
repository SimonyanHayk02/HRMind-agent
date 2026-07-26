from __future__ import annotations

import json
from typing import Any


class FakeLLM:
    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        response_json: bool = False,
    ) -> str:
        text, _ = await self.complete_with_usage(
            system=system, user=user, temperature=temperature, response_json=response_json
        )
        return text

    async def complete_with_usage(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        response_json: bool = False,
    ) -> tuple[str, dict[str, int]]:
        self.calls.append({"system": system, "user": user, "response_json": response_json})
        for key, value in self._responses.items():
            if key in user or key in system:
                return value, {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}

        system_l = system.lower()
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        # NL2SQL stub — prefer simple SELECT COUNT when asked how many
        if "postgresql select" in system_l or "sql" in system_l and "select" in system_l:
            if "how many" in user.lower() or "count" in user.lower():
                return "SELECT COUNT(*) AS count FROM employees LIMIT 200", usage
            return "SELECT id, first_name, last_name, department FROM employees LIMIT 50", usage

        if response_json:
            if "plan" in system_l or "execution" in system_l:
                plan = {
                    "version": "1",
                    "nodes": [
                        {
                            "id": "sql1",
                            "kind": "tool",
                            "name": "sql",
                            "input_bindings": {},
                            "params": {"mode": "nl2sql", "question": user[:500]},
                            "depends_on": [],
                        }
                    ],
                    "response_strategy": "llm_format",
                    "clarify_question": None,
                }
                return json.dumps(plan), usage
            if "format" in system_l or "concise" in system_l:
                try:
                    payload = json.loads(user)
                    results = payload.get("results") or []
                    for item in reversed(results):
                        if isinstance(item, dict) and "count" in item:
                            return f"There are {item['count']} matching employees.", usage
                        if isinstance(item, dict) and item.get("rows"):
                            return f"Found {len(item['rows'])} employees.", usage
                except Exception:
                    pass
                return "Here is what I found from the HR tools.", usage
            return json.dumps({"ok": True, "echo": user[:200]}), usage

        return f"Echo: {user[:500]}", usage
