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

        if response_json:
            # Tool-selecting planner (tool_selector.md) — before NL2SQL stub.
            # The selector prompt mentions sql/select and must not hit the SQL branch.
            if "tool selector" in system_l or "tool_selector" in system_l:
                return json.dumps(_fake_tool_selection(user)), usage
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

        # NL2SQL stub (non-JSON completions only — selector prompt mentions sql/select).
        if "postgresql select" in system_l or (
            "sql" in system_l and "select" in system_l
        ):
            if "how many" in user.lower() or "count" in user.lower():
                return "SELECT COUNT(*) AS count FROM employees LIMIT 200", usage
            return (
                "SELECT id, first_name, last_name, department FROM employees LIMIT 50",
                usage,
            )

        return f"Echo: {user[:500]}", usage


def _fake_tool_selection(user: str) -> dict[str, Any]:
    """Deterministic tool-selector JSON for FakeLLM / offline tests."""
    q = ""
    prior: list[str] = []
    try:
        payload = json.loads(user)
        q = str(payload.get("question") or "").lower()
        prior = [str(x) for x in (payload.get("prior_employee_ids") or []) if x]
    except Exception:
        q = user.lower()

    def base(**overrides: Any) -> dict[str, Any]:
        data: dict[str, Any] = {
            "confidence": 0.9,
            "clarify_question": None,
            "intent": "unknown",
            "slots": {
                "department": None,
                "city": None,
                "country": None,
                "skill": None,
                "name": None,
                "person_name": None,
                "language": None,
                "certification": None,
                "facet_dimension": None,
                "month": None,
                "scope": None,
                "status": None,
                "email": None,
                "employee_ids": prior[:40],
                "count_only": False,
                "want_count": False,
                "refers_to_prior": bool(prior) and any(
                    w in q for w in ("them", "those", "of them", "among")
                ),
                "wants_age": False,
                "wants_wish": False,
            },
            "selected": [],
            "rationale": "fake_llm",
        }
        for k, v in overrides.items():
            if k == "slots" and isinstance(v, dict):
                data["slots"].update(v)
            else:
                data[k] = v
        return data

    if any(w in q for w in ("pto", "vacation", "benefits", "payroll")):
        return base(intent="unsupported", confidence=0.95)

    # Facets before generic "how many" headcount (cities/countries ≠ 100 employees).
    if "countries" in q or "cities" in q or "departments" in q:
        if "countr" in q:
            dim = "country"
        elif "cit" in q:
            dim = "city"
        else:
            dim = "department"
        return base(
            intent="facet_list" if "which" in q or "list" in q else "facet_count",
            slots={"facet_dimension": dim, "want_count": True},
        )

    if "tenure" in q or ("average" in q and "engineering" in q):
        dept = "Engineering" if "engineering" in q else None
        return base(intent="tenure_agg", slots={"department": dept})

    if "most senior" in q or "longest tenured" in q or "longest tenure" in q:
        dept = "Engineering" if "engineering" in q else None
        return base(intent="longest_tenured", slots={"department": dept})

    if "atlantis" in q:
        msg = (
            'I don\'t recognize "Atlantis" as a known city or country. '
            "Try one of the places we track."
        )
        return base(
            intent="clarify",
            confidence=0.9,
            clarify_question=msg,
            slots={"clarify_question": msg},
        )

    if q.strip() in {"of them?", "of them", "among them?", "among them"} and not prior:
        return base(
            intent="clarify",
            confidence=0.9,
            clarify_question=(
                "Which previous list of employees did you mean? "
                "Ask a search first, then I can answer about them."
            ),
            slots={
                "clarify_question": (
                    "Which previous list of employees did you mean? "
                    "Ask a search first, then I can answer about them."
                )
            },
        )

    # Typo / slang headcount paraphrases (no regex NLU — selector path).
    typo_count = any(
        w in q
        for w in (
            "how meny",
            "how many",
            "headcount",
            "total employees",
            "employes",
            "peeps in",
            "staff in",
            "ppl in",
        )
    )
    if typo_count and "python" not in q and "know" not in q and "kno " not in q:
        dept = None
        for d in ("engineering", "sales", "finance", "product", "operations", "people"):
            if d in q or (d == "engineering" and ("engeneering" in q or "eng " in q)):
                dept = d.capitalize() if d != "engineering" else "Engineering"
                if d == "people":
                    dept = "People"
                break
        if "sales" in q or "sale " in q:
            dept = "Sales"
        return base(
            intent="count",
            slots={"department": dept, "want_count": True, "count_only": True},
        )

    if (
        "python" in q
        or "kubernetes" in q
        or "kubernetees" in q
        or "aws" in q
        or "docker" in q
        or "react" in q
    ):
        if "kubernetees" in q or "kubernetes" in q:
            skill_label = "Kubernetes"
        else:
            skill = next(
                (s for s in ("python", "aws", "docker", "react") if s in q),
                "python",
            )
            skill_label = skill.capitalize() if skill != "aws" else "AWS"
        return base(
            intent="skill_count"
            if ("how many" in q or "count" in q or q.strip().endswith("?"))
            else "skill_search",
            slots={
                "skill": skill_label,
                "count_only": "how many" in q or prior or "any of them" in q,
                "refers_to_prior": bool(prior),
                "employee_ids": prior[:40],
            },
        )

    if "birthday" in q or "dob" in q or "bday" in q or "born" in q:
        if "today" in q:
            return base(intent="birthday_today")
        if "upcoming" in q:
            return base(intent="birthday_upcoming")
        return base(intent="birthday_person", slots={"name": "Carol Garcia"})

    if "manager" in q:
        return base(intent="manager", slots={"name": "Alice Nguyen"})

    if "status" in q and any(w in q for w in ("set", "change", "update", "flip")):
        return base(
            intent="set_status",
            confidence=0.9,
            slots={"name": "Carol Garcia", "status": True},
        )
    # Novel status paraphrases that miss the regex extractor → selector HITL.
    if "activate" in q and ("carol" in q or "garcia" in q):
        return base(
            intent="set_status",
            confidence=0.92,
            slots={"name": "Carol Garcia", "status": True},
        )

    if "salary" in q:
        return base(
            intent="profile",
            slots={"name": "Alice Nguyen"},
            selected=[{"tool": "employee", "params": {"action": "by_name", "name": "Alice Nguyen"}}],
        )

    if "find sofia" in q or "do we have sofia" in q:
        return base(intent="profile", slots={"name": "Sofia"})

    if "profile" in q or "tell me about" in q or "who is" in q:
        return base(intent="profile", slots={"name": "Alice Nguyen"})

    # Force legacy fallback in primary+fallback mode for unknown asks.
    return base(intent="unknown", confidence=0.2)


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
