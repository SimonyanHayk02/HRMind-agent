from __future__ import annotations

import re

from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.domain.session import SessionMemory

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
# Broad skill context — includes "developing", not only "developer"
_SKILL_INTENT_RE = re.compile(
    r"\b("
    r"knows?|knowing|experience|experienced|skill|skills|proficient|familiar|"
    r"resume|develop(?:er|ers|ing|ed)?|coding|code|programm(?:er|ers|ing|ed)?|"
    r"using|uses|used|stack|find|search|with|who|"
    r"how many|how much|employee|employees|people|staff|them|those"
    r")\b",
    re.I,
)
_COUNT_RE = re.compile(r"\b(how many|how much|count|number of)\b", re.I)
_ANAPHORA_RE = re.compile(
    r"\b("
    r"of them|of those|from them|from those|among them|among those|"
    r"that group|those employees|these employees|"
    r"the (previous|last|same) (set|group|list|results?)"
    r")\b",
    re.I,
)
_FOLLOWUP_NAMES_RE = re.compile(
    r"\b("
    r"their names?|the names?|there names?|"
    r"who are (they|those|them)|"
    r"list (them|those|their names?)|"
    r"say (their|there|the) names?|"
    r"name them|show (me )?them|what are (their|there) names?"
    r")\b",
    re.I,
)
_CITY_RE = re.compile(r"\b(?:in|from)\s+(Berlin|Dubai|London|Paris|New York)\b", re.I)

_NAME_COLUMNS = ["id", "first_name", "last_name", "department", "position"]


def refers_to_prior_set(question: str) -> bool:
    return bool(_ANAPHORA_RE.search(question) or _FOLLOWUP_NAMES_RE.search(question))


def _sql_over_ids(
    employee_ids: list[str],
    *,
    count_only: bool,
    extra_filters: dict | None = None,
    columns: list[str] | None = None,
) -> ExecutionPlan:
    filters: dict = {"employee_ids": list(employee_ids)}
    if extra_filters:
        filters.update(extra_filters)
    params: dict = {
        "mode": "constrained",
        "count_only": count_only,
        "filters": filters,
    }
    if not count_only:
        params["columns"] = columns or _NAME_COLUMNS
    return ExecutionPlan(
        nodes=[
            PlanNode(id="sql1", kind="tool", name="sql", params=params),
        ],
        response_strategy="template",
    )


def _resume_search_plan(
    question: str,
    *,
    count_only: bool = False,
    hire_date_gt: str | None = None,
    intersect_with: list[str] | None = None,
) -> ExecutionPlan:
    nodes: list[PlanNode] = [
        PlanNode(
            id="r1",
            kind="tool",
            name="resume_search",
            params={"question": question},
        ),
        PlanNode(
            id="ids",
            kind="operator",
            name="extract_employee_ids",
            depends_on=["r1"],
            input_bindings={"data": "nodes.r1"},
        ),
    ]
    id_source = "ids"
    if intersect_with:
        nodes.append(
            PlanNode(
                id="ix",
                kind="operator",
                name="intersect_ids",
                depends_on=["ids"],
                input_bindings={"data": "nodes.ids"},
                params={"other": list(intersect_with)},
            )
        )
        id_source = "ix"

    if count_only or hire_date_gt or intersect_with is not None:
        filters: dict = {}
        if hire_date_gt:
            filters["hire_date_gt"] = hire_date_gt
        nodes.append(
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                depends_on=[id_source],
                params={
                    "mode": "constrained",
                    "count_only": True if (count_only or hire_date_gt) else False,
                    "filters": filters,
                    **(
                        {}
                        if (count_only or hire_date_gt)
                        else {"columns": _NAME_COLUMNS}
                    ),
                },
                input_bindings={"employee_ids": f"nodes.{id_source}"},
            )
        )
        return ExecutionPlan(
            nodes=nodes,
            response_strategy="template" if (count_only or hire_date_gt) else "llm_format",
        )

    # Plain resume list (no count / intersect)
    return ExecutionPlan(
        nodes=[nodes[0]],
        response_strategy="llm_format",
    )


def try_heuristic_plan(
    question: str,
    *,
    memory: SessionMemory | None = None,
) -> ExecutionPlan | None:
    """Deterministic plans for common HR questions (works without OpenAI)."""
    q = question.strip()
    lower = q.lower()
    prior_ids = list(memory.last_employee_ids) if memory and memory.last_employee_ids else []
    # Full-company dumps are not a meaningful "them" cohort
    if len(prior_ids) > 50:
        prior_ids = []
    anaphora = bool(prior_ids) and refers_to_prior_set(q)

    # Follow-up: list names of the active result set
    if prior_ids and _FOLLOWUP_NAMES_RE.search(q):
        return _sql_over_ids(prior_ids, count_only=False)

    # Follow-up: refine prior set by city ("of them in Berlin")
    city_match = _CITY_RE.search(q)
    if anaphora and city_match:
        return _sql_over_ids(
            prior_ids,
            count_only=bool(_COUNT_RE.search(q)),
            extra_filters={"city": city_match.group(1)},
        )

    # Follow-up: refine prior set by department
    if anaphora:
        for dept in _DEPARTMENTS:
            if dept.lower() in lower:
                return _sql_over_ids(
                    prior_ids,
                    count_only=bool(_COUNT_RE.search(q)),
                    extra_filters={"department": dept},
                )

    # Follow-up: count the prior set ("how many of them?")
    if anaphora and _COUNT_RE.search(q) and not _SKILL_RE.search(q):
        return _sql_over_ids(prior_ids, count_only=True)

    # Count by department (global, not anaphora)
    if not anaphora:
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

    # List/filter by city (global)
    if city_match and not anaphora and any(
        w in lower for w in ("employee", "people", "staff", "list", "who")
    ):
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
                        "columns": _NAME_COLUMNS,
                    },
                )
            ],
            response_strategy="template",
        )

    # Resume / skills search — intersect prior set when anaphoric.
    # Skill token + any skill/HR context word is enough (covers "developing in python").
    skill_match = _SKILL_RE.search(q)
    if skill_match and _SKILL_INTENT_RE.search(q):
        year_match = re.search(r"(?:after|since)\s+(20\d{2})", lower)
        count_only = bool(_COUNT_RE.search(q))
        hire_date_gt = f"{year_match.group(1)}-01-01" if year_match and count_only else None
        intersect = prior_ids if anaphora else None
        return _resume_search_plan(
            q,
            count_only=count_only and not hire_date_gt,
            hire_date_gt=hire_date_gt,
            intersect_with=intersect,
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
