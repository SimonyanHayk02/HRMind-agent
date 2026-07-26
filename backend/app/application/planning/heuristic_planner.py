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

# Keep in sync with scripts/generate_resumes.py SKILLS (+ common cloud terms).
_SKILLS = (
    "Python|Java|Go|React|Kubernetes|NLP|Machine Learning|SQL|"
    "AWS|Docker|Salesforce|Recruiting|Accounting|Product Strategy"
)

_SKILL_RE = re.compile(rf"\b({_SKILLS})\b", re.I)
_SKILL_INTENT_RE = re.compile(
    r"\b(knows?|knowing|experience|experienced|skill|skills|proficient|"
    r"familiar|resume|developer|find|search|with)\b",
    re.I,
)
_COUNT_RE = re.compile(r"\b(how many|how much|count|number of)\b", re.I)


def _resume_search_plan(question: str, *, count_only: bool = False, hire_date_gt: str | None = None) -> ExecutionPlan:
    nodes = [
        PlanNode(
            id="r1",
            kind="tool",
            name="resume_search",
            params={"question": question},
        )
    ]
    if count_only or hire_date_gt:
        nodes.append(
            PlanNode(
                id="ids",
                kind="operator",
                name="extract_employee_ids",
                depends_on=["r1"],
                input_bindings={"data": "nodes.r1"},
            )
        )
        filters: dict = {}
        if hire_date_gt:
            filters["hire_date_gt"] = hire_date_gt
        nodes.append(
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                depends_on=["ids"],
                params={
                    "mode": "constrained",
                    "count_only": True,
                    "filters": filters,
                },
                input_bindings={"employee_ids": "nodes.ids"},
            )
        )
        return ExecutionPlan(nodes=nodes, response_strategy="template")
    return ExecutionPlan(nodes=nodes, response_strategy="llm_format")


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

    # Resume / skills search — must run before generic SQL ("employees" + "how many")
    skill_match = _SKILL_RE.search(q)
    if skill_match and _SKILL_INTENT_RE.search(q):
        year_match = re.search(r"(?:after|since)\s+(20\d{2})", lower)
        count_only = bool(_COUNT_RE.search(q))
        hire_date_gt = f"{year_match.group(1)}-01-01" if year_match and count_only else None
        return _resume_search_plan(
            q,
            count_only=count_only and not hire_date_gt,
            hire_date_gt=hire_date_gt,
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
