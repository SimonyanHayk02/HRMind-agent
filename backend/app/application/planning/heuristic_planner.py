from __future__ import annotations

import re

from app.application.planning.plan_schema import ExecutionPlan, PlanNode

_DEPARTMENTS = [
    "Engineering",
    "People",
    "Sales",
    "Finance",
    "Product",
    "Operations",
]


def try_heuristic_plan(question: str) -> ExecutionPlan | None:
    """Deterministic plans for common HR questions (works without OpenAI)."""
    q = question.strip()
    lower = q.lower()

    # Count by department: "How many employees work in Engineering?"
    for dept in _DEPARTMENTS:
        if dept.lower() in lower and any(
            w in lower for w in ("how many", "count", "number of")
        ):
            return ExecutionPlan(
                nodes=[
                    PlanNode(
                        id="sql1",
                        kind="tool",
                        name="sql",
                        params={
                            "mode": "constrained",
                            "count_only": True,
                            "filters": {"department": dept},
                        },
                    )
                ],
                response_strategy="template",
            )

    # List/filter by city/country
    city_match = re.search(
        r"\b(?:in|from)\s+(Berlin|Dubai|London|Paris|New York)\b", q, re.I
    )
    if city_match and any(w in lower for w in ("employee", "people", "staff", "list", "who")):
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="sql1",
                    kind="tool",
                    name="sql",
                    params={
                        "mode": "constrained",
                        "count_only": False,
                        "filters": {"city": city_match.group(1)},
                    },
                )
            ],
            response_strategy="llm_format",
        )

    # Resume / skills search
    skill_match = re.search(
        r"\b(Python|Java|Go|React|Kubernetes|NLP|Machine Learning|SQL)\b", q, re.I
    )
    if skill_match and any(
        w in lower for w in ("resume", "skill", "developer", "find", "search", "who has")
    ):
        # Hybrid count after year
        year_match = re.search(r"(?:after|since)\s+(20\d{2})", lower)
        if year_match and any(w in lower for w in ("how many", "count")):
            return ExecutionPlan(
                nodes=[
                    PlanNode(
                        id="r1",
                        kind="tool",
                        name="resume_search",
                        params={"question": q},
                    ),
                    PlanNode(
                        id="ids",
                        kind="operator",
                        name="extract_employee_ids",
                        depends_on=["r1"],
                        input_bindings={"data": "nodes.r1"},
                    ),
                    PlanNode(
                        id="sql1",
                        kind="tool",
                        name="sql",
                        depends_on=["ids"],
                        params={
                            "mode": "constrained",
                            "count_only": True,
                            "filters": {"hire_date_gt": f"{year_match.group(1)}-01-01"},
                        },
                        input_bindings={"employee_ids": "nodes.ids"},
                    ),
                ],
                response_strategy="template",
            )
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="r1",
                    kind="tool",
                    name="resume_search",
                    params={"question": q},
                )
            ],
            response_strategy="llm_format",
        )

    # Manager lookup
    if "manager" in lower:
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="e1",
                    kind="tool",
                    name="employee",
                    params={"action": "manager", "question": q},
                )
            ],
            response_strategy="llm_format",
        )

    # "Tell me about Bob" / "Who is Alice?"
    about = re.search(
        r"(?:tell me about|who is|what about|profile of)\s+([A-Za-z][A-Za-z\-']+)",
        q,
        re.I,
    )
    if about:
        name = about.group(1)
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="e1",
                    kind="tool",
                    name="employee",
                    params={"action": "by_name", "name": name},
                )
            ],
            response_strategy="llm_format",
        )

    # Generic structured employee analytics
    if any(
        w in lower
        for w in (
            "employee",
            "department",
            "hired",
            "salary",
            "headcount",
            "how many",
            "count",
        )
    ):
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="sql1",
                    kind="tool",
                    name="sql",
                    params={"mode": "nl2sql", "question": q},
                )
            ],
            response_strategy="llm_format",
        )

    return None
