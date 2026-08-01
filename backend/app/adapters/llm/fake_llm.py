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
            # Residual slot extractor (nlu_slots.md) — before free-form planner.
            if "dialogue slots" in system_l or "person_ref" in system_l or "nlu" in system_l:
                return json.dumps(_fake_slot_bundle(user)), usage
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


def _fake_slot_bundle(user: str) -> dict[str, Any]:
    """Deterministic slot JSON for hermetic tests / offline FakeLLM."""
    q = ""
    try:
        payload = json.loads(user)
        q = str(payload.get("question") or "").lower()
        listed = payload.get("last_listed") or []
    except Exception:
        q = user.lower()
        listed = []

    def base(**overrides: Any) -> dict[str, Any]:
        data: dict[str, Any] = {
            "intent": "unknown",
            "attribute": "none",
            "person_ref": {"kind": "none", "value": None, "index": None},
            "refers_to_prior": False,
            "skill": None,
            "facet_dimension": None,
            "city": None,
            "country": None,
            "department": None,
            "position": None,
            "status_value": None,
            "want_count": False,
            "wants_age": False,
            "wants_wish": False,
            "confidence": 0.85,
            "notes": ["fake_llm"],
        }
        data.update(overrides)
        return data

    if any(w in q for w in ("pto", "vacation", "benefits", "payroll")):
        return base(intent="unsupported", confidence=0.9)

    if "top one" in q or "earlier person" in q or "#1" in q:
        return base(
            intent="birthday",
            attribute="dob",
            person_ref={"kind": "ordinal", "value": "first", "index": 1},
            confidence=0.9 if listed else 0.7,
        )

    if "how many cities" in q or "different cities" in q:
        return base(
            intent="facet_count",
            attribute="facet",
            facet_dimension="city",
            want_count=True,
            confidence=0.9,
        )

    if "based in berlin" in q or "working out of berlin" in q or "in berlin" in q:
        return base(
            intent="location_cohort",
            attribute="location",
            city="Berlin",
            want_count="how many" in q,
            confidence=0.88,
        )

    if "bday" in q or "date of birth" in q or "dob" in q:
        return base(
            intent="birthday",
            attribute="dob",
            person_ref={"kind": "ordinal", "value": "first", "index": 1},
            confidence=0.8,
        )

    # Low-confidence unknown — forces fallthrough when tests need the planner.
    return base(intent="unknown", confidence=0.2)
