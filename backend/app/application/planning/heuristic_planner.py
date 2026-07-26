from __future__ import annotations

import re

from app.application.memory.context_updates import filters_from_constraint_memory
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.unsupported import UNSUPPORTED_ANSWER, is_unsupported_topic
from app.domain.session import SessionMemory
from app.tools.employee.tool import extract_manager_subject

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
# Short utterances that almost always continue the prior answer ("names please")
_SHORT_LIST_RE = re.compile(
    r"^\s*("
    r"names?(?:\s+please)?|"
    r"(?:the\s+)?names?(?:\s+please)?|"
    r"list(?:\s+them|\s+those)?(?:\s+please)?|"
    r"which\s+ones?(?:\s+please)?|"
    r"which\s+(?:countries|cities|departments)(?:\s+please)?|"
    r"what\s+are\s+they|"
    r"give\s+(?:me\s+)?(?:the\s+)?names?"
    r")[\s?.!]*$",
    re.I,
)
# "how many different countries" / typo-tolerant "in how different countries"
_FACET_DIM_RE = re.compile(
    r"\b(?P<dim>countries|country|cities|city|departments|department)\b",
    re.I,
)
_FACET_COUNT_RE = re.compile(
    r"\b("
    r"(?:how many|how much|number of|count)\s+(?:different\s+|unique\s+)?"
    r"(?:countries|country|cities|city|departments|department)"
    r"|"
    r"in how(?:\s+many)?(?:\s+different)?\s+(?:countries|country|cities|city|departments|department)"
    r"|"
    r"(?:different|unique)\s+(?:countries|country|cities|city|departments|department)"
    r")\b",
    re.I,
)
_FACET_LIST_RE = re.compile(
    r"\b("
    r"(?:which|what|list(?:\s+the)?|name(?:\s+the)?)\s+"
    r"(?:different\s+|unique\s+)?(?:countries|country|cities|city|departments|department)"
    r")\b",
    re.I,
)
_CITY_RE = re.compile(r"\b(?:in|from)\s+(Berlin|Dubai|London|Paris|New York)\b", re.I)
_PERSON_LOCATION_RE = re.compile(
    r"\bwhere\s+(?:(?:does|do|is)\s+)?(?:the\s+)?"
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)"
    r"(?:\s+in\s+[A-Za-z]+)?"
    r"\s+(?:live|lives|living|located|from|based)\b",
    re.I,
)
_ABOUT_PERSON_RE = re.compile(
    r"(?:tell me about|who is|what about|profile of)\s+"
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)",
    re.I,
)

_DEPT_IN_TEXT_RE = re.compile(
    r"\b(?:in|from)\s+(Engineering|People|Sales|Finance|Product|Operations)\b",
    re.I,
)
_LIST_DEPT_RE = re.compile(
    r"\b(list|show|who|employees?|people|staff)\b",
    re.I,
)

_NAME_COLUMNS = ["id", "first_name", "last_name", "department", "position"]


def _dept_hint(question: str) -> str | None:
    m = _DEPT_IN_TEXT_RE.search(question)
    return m.group(1) if m else None


_FACET_WORD_TO_COL = {
    "countries": "country",
    "country": "country",
    "cities": "city",
    "city": "city",
    "departments": "department",
    "department": "department",
}


def refers_to_prior_set(question: str) -> bool:
    q = question.strip()
    return bool(
        _ANAPHORA_RE.search(q)
        or _FOLLOWUP_NAMES_RE.search(q)
        or _SHORT_LIST_RE.search(q)
    )


def is_list_followup(question: str) -> bool:
    q = question.strip()
    return bool(_FOLLOWUP_NAMES_RE.search(q) or _SHORT_LIST_RE.search(q))


def _normalize_facet_dim(raw: str) -> str | None:
    return _FACET_WORD_TO_COL.get(raw.lower())


def _detect_facet_dimension(question: str) -> str | None:
    m = _FACET_DIM_RE.search(question)
    if not m:
        return None
    return _normalize_facet_dim(m.group("dim"))


def _facet_count_plan(dimension: str) -> ExecutionPlan:
    """Count distinct facet values and materialize the value list for follow-ups."""
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="facet",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "distinct": True,
                    "columns": [dimension],
                },
            ),
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_distinct": dimension,
                },
            ),
        ],
        response_strategy="template",
    )


def _facet_list_plan(dimension: str) -> ExecutionPlan:
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="facet",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "distinct": True,
                    "columns": [dimension],
                },
            )
        ],
        response_strategy="template",
    )


def _employee_by_name_plan(name: str, *, department: str | None = None) -> ExecutionPlan:
    params: dict = {"action": "by_name", "name": name.strip()}
    if department:
        params["department"] = department
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="e1",
                kind="tool",
                name="employee",
                params=params,
            )
        ],
        response_strategy="template",
        active_cohort_node="e1",
    )


def _manager_plan(
    question: str, *, name: str | None = None, department: str | None = None
) -> ExecutionPlan:
    params: dict = {"action": "manager", "question": question}
    if name:
        params["name"] = name
    if department:
        params["department"] = department
    return ExecutionPlan(
        nodes=[
            PlanNode(id="e1", kind="tool", name="employee", params=params),
        ],
        response_strategy="template",
        active_cohort_node="e1",
    )


def _match_entity_name(question: str, memory: SessionMemory | None) -> str | None:
    if not memory or not memory.entity_memory:
        return None
    lower = question.lower()
    entities = sorted(memory.entity_memory, key=lambda e: len(e.display_name), reverse=True)
    # Exact full-name / alias containment first
    for ent in entities:
        name = ent.display_name.strip()
        if name and name.lower() in lower:
            return name
        for alias in ent.aliases:
            if alias and alias.lower() in lower:
                return name or alias
    # First/last token match when unique among remembered people
    token_hits: dict[str, list[str]] = {}
    for ent in entities:
        name = ent.display_name.strip()
        for token in name.lower().split():
            if len(token) < 2:
                continue
            if re.search(rf"\b{re.escape(token)}\b", lower):
                token_hits.setdefault(token, []).append(name)
    for names in token_hits.values():
        uniq = list(dict.fromkeys(names))
        if len(uniq) == 1:
            return uniq[0]
    return None


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
    # Count refinements still materialize the filtered ID set so "names please"
    # follows the refined cohort, not the previous broader one.
    if count_only:
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="cohort",
                    kind="tool",
                    name="sql",
                    params={
                        "mode": "constrained",
                        "count_only": False,
                        "filters": filters,
                        "columns": ["id"],
                    },
                ),
                PlanNode(
                    id="sql1",
                    kind="tool",
                    name="sql",
                    params={
                        "mode": "constrained",
                        "count_only": True,
                        "filters": filters,
                    },
                ),
            ],
            response_strategy="template",
            active_cohort_node="cohort",
        )
    params: dict = {
        "mode": "constrained",
        "count_only": False,
        "filters": filters,
        "columns": columns or _NAME_COLUMNS,
    }
    return ExecutionPlan(
        nodes=[
            PlanNode(id="sql1", kind="tool", name="sql", params=params),
        ],
        response_strategy="template",
        active_cohort_node="sql1",
    )


def _department_count_plan(dept: str) -> ExecutionPlan:
    """Count + materialize department cohort IDs for follow-up 'of them…' turns."""
    filters = {"department": dept}
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="cohort",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": False,
                    "filters": filters,
                    "columns": ["id"],
                },
            ),
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": True,
                    "filters": filters,
                },
            ),
        ],
        response_strategy="template",
        active_cohort_node="cohort",
    )


def _department_list_plan(dept: str) -> ExecutionPlan:
    filters = {"department": dept}
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": False,
                    "filters": filters,
                    "columns": _NAME_COLUMNS + ["city", "country"],
                },
            )
        ],
        response_strategy="template",
        active_cohort_node="sql1",
    )


def _filtered_count_plan(filters: dict) -> ExecutionPlan:
    """Attribute-filtered count that also materializes the matching cohort."""
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="cohort",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": False,
                    "filters": dict(filters),
                    "columns": ["id"],
                },
            ),
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": True,
                    "filters": dict(filters),
                },
            ),
        ],
        response_strategy="template",
        active_cohort_node="cohort",
    )


def _resume_search_plan(
    question: str,
    *,
    count_only: bool = False,
    hire_date_gt: str | None = None,
    intersect_with: list[str] | None = None,
    scope_filters: dict | None = None,
) -> ExecutionPlan:
    """RAG plan with optional prior-cohort / SQL-filter scoping.

    - intersect_with: prior employee IDs (SQL→RAG). Passed into resume_search and
      intersected after extract as a safety net.
    - scope_filters: remembered department/city/… when IDs were not materialized.
    """
    resume_params: dict = {"question": question}
    if intersect_with:
        resume_params["employee_ids"] = list(intersect_with)

    nodes: list[PlanNode] = [
        PlanNode(
            id="r1",
            kind="tool",
            name="resume_search",
            params=resume_params,
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

    need_sql = bool(count_only or hire_date_gt or intersect_with is not None or scope_filters)
    if need_sql:
        filters: dict = dict(scope_filters or {})
        if hire_date_gt:
            filters["hire_date_gt"] = hire_date_gt
        list_names = not count_only and not hire_date_gt
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
                    **({"columns": _NAME_COLUMNS} if list_names else {}),
                },
                input_bindings={"employee_ids": f"nodes.{id_source}"},
            )
        )
        return ExecutionPlan(
            nodes=nodes,
            response_strategy="template" if (count_only or hire_date_gt) else "llm_format",
            active_cohort_node="sql1" if list_names else id_source,
        )

    # Plain resume list (no count / intersect) — still extract IDs for cohort memory
    return ExecutionPlan(
        nodes=nodes,
        response_strategy="llm_format",
        active_cohort_node="ids",
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
    scope_filters = filters_from_constraint_memory(
        memory.constraint_memory if memory else None
    )
    refers = refers_to_prior_set(q)
    # Anaphora with IDs, or "of them" with remembered SQL constraints (dept/city).
    anaphora = refers and bool(prior_ids)
    scoped_followup = refers and (bool(prior_ids) or bool(scope_filters))

    # Out-of-schema topics (vacation, benefits, …) — do not invent via SQL
    if is_unsupported_topic(q):
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=UNSUPPORTED_ANSWER,
        )

    # Named person from prior result set ("where ivy chen lives?")
    entity_name = _match_entity_name(q, memory)
    dept_hint = _dept_hint(q)
    if entity_name and (
        _PERSON_LOCATION_RE.search(q)
        or any(w in lower for w in ("live", "lives", "living", "located", "city", "country", "based"))
        or _ABOUT_PERSON_RE.search(q)
    ):
        return _employee_by_name_plan(entity_name, department=dept_hint)

    # "where does Ivy Chen live?" / "where the ivy chen lives"
    loc = _PERSON_LOCATION_RE.search(q)
    if loc:
        return _employee_by_name_plan(loc.group(1), department=dept_hint)

    # Manager lookup (before generic analytics)
    mgr_name = extract_manager_subject(q)
    if mgr_name or re.search(r"\b(manager of|'s manager|who manages)\b", lower):
        return _manager_plan(q, name=mgr_name or entity_name, department=dept_hint)

    focus = memory.last_focus if memory else None

    # Short list follow-up: "names please" → facet values OR employee cohort
    if is_list_followup(q):
        if focus and focus.kind == "facet" and focus.dimension:
            return _facet_list_plan(focus.dimension)
        if prior_ids:
            return _sql_over_ids(prior_ids, count_only=False)
        # Explicit "which countries" without prior focus still works via facet list regex below
        if not _FACET_LIST_RE.search(q):
            return ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question=(
                    "Which names should I list — countries, cities, departments, "
                    "or employees from a previous search?"
                ),
            )

    # Distinct facet count/list ("how many different countries", "which countries")
    if _FACET_COUNT_RE.search(q):
        dim = _detect_facet_dimension(q)
        if dim:
            return _facet_count_plan(dim)
    if _FACET_LIST_RE.search(q):
        dim = _detect_facet_dimension(q)
        if dim:
            return _facet_list_plan(dim)

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

    # Constraint-only follow-up with optional city refine
    if scoped_followup and not prior_ids and not _SKILL_RE.search(q):
        filters = dict(scope_filters)
        if city_match:
            filters["city"] = city_match.group(1)
        if filters and (_COUNT_RE.search(q) or city_match or is_list_followup(q)):
            if _COUNT_RE.search(q) or (city_match and not is_list_followup(q)):
                return _filtered_count_plan(filters)
            return ExecutionPlan(
                nodes=[
                    PlanNode(
                        id="sql1",
                        kind="tool",
                        name="sql",
                        params={
                            "mode": "constrained",
                            "count_only": False,
                            "filters": filters,
                            "columns": _NAME_COLUMNS,
                        },
                    )
                ],
                response_strategy="template",
                active_cohort_node="sql1",
            )

    # Count by department (global) — also materialize cohort IDs
    if not scoped_followup:
        for dept in _DEPARTMENTS:
            if dept.lower() in lower and any(
                w in lower for w in ("how many", "count", "number of")
            ):
                return _department_count_plan(dept)

    # List employees by department (global) — materialize names + IDs
    if not scoped_followup and _LIST_DEPT_RE.search(q):
        for dept in _DEPARTMENTS:
            if dept.lower() in lower and not any(
                w in lower for w in ("how many", "count", "number of", "different")
            ):
                return _department_list_plan(dept)

    # List/filter by city (global)
    if city_match and not scoped_followup and any(
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
            active_cohort_node="sql1",
        )

    # Resume / skills search — intersect prior set or apply constraint scope.
    skill_match = _SKILL_RE.search(q)
    if skill_match and _SKILL_INTENT_RE.search(q):
        year_match = re.search(r"(?:after|since)\s+(20\d{2})", lower)
        count_only = bool(_COUNT_RE.search(q))
        hire_date_gt = f"{year_match.group(1)}-01-01" if year_match and count_only else None
        intersect = prior_ids if anaphora else None
        # When user says "of them" but we only have dept/city memory (no IDs yet).
        filters = scope_filters if (scoped_followup and not prior_ids) else None
        return _resume_search_plan(
            q,
            count_only=count_only and not hire_date_gt,
            hire_date_gt=hire_date_gt,
            intersect_with=intersect,
            scope_filters=filters,
        )

    # "Tell me about Bob" / "Who is Alice Nguyen?"
    about = _ABOUT_PERSON_RE.search(q)
    if about:
        return _employee_by_name_plan(about.group(1), department=dept_hint)

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
            "where",
            "live",
            "lives",
            "living",
            "located",
            "city",
            "country",
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
            active_cohort_node="sql1",
        )

    return None
