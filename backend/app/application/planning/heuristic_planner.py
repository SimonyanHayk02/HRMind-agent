from __future__ import annotations

import re
from datetime import date

from app.application.memory.context_updates import (
    filters_from_constraint_memory,
    place_from_constraint_memory,
)
from app.application.planning.location_plans import (
    LOCATION_FIELDS,
    location_cohort_nodes,
    location_cohort_plan,
    location_facet_plan,
    location_person_plan,
    profile_location_node,
    split_location_filters,
)
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.unsupported import (
    is_unsupported_topic,
    unsupported_answer_for,
)
from app.application.response.refusal import RefusalCode
from app.application.understanding.person_existence import extract_person_existence_name
from app.application.understanding.role_phrases import match_role_positions
from app.application.understanding.status_change import extract_status_change
from app.domain.places import CITY_ALT, COUNTRY_ALT, canon_city, canon_country
from app.domain.session import SessionMemory
from app.tools.employee.tool import extract_manager_subject, extract_reports_subject

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
_COUNT_RE = re.compile(
    r"\b("
    r"how many|how much|count|number of|headcount|"
    r"total\s+employees?|total\s+headcount|our\s+headcount"
    r")\b|"
    r"^\s*headcount\s*\??\s*$",
    re.I,
)
_ORG_HEADCOUNT_RE = re.compile(
    r"^\s*(?:"
    r"(?:what(?:'s|\s+is)\s+)?(?:our\s+|the\s+)?headcount\s*\??|"
    r"total\s+employees?\s*\??|"
    r"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:total\s+)?(?:number\s+of\s+)?employees?\s*\??|"
    r"how many people work here\s*\??|"
    # Paraphrases that the LLM misroutes to a department-sized 17 — not the
    # canonical "How many employees do we have?" (kept for tool-select / repair).
    r"(?:quick\s*[—\-]?\s*)?how many people(?:\s+do\s+we\s+have)?(?:\s+in\s+total)?\s*\??|"
    r"how many people(?:\s+(?:overall|altogether))?\s*\??|"
    r"overall\s+headcount\s*\??"
    r")\s*$",
    re.I,
)
_BARE_DEPT_RE = re.compile(
    r"^\s*(engineering|engineers|engeneering|sales|finance|product|operations|people)\s*\??\s*$",
    re.I,
)
_BARE_SKILL_RE = re.compile(
    rf"^\s*({_SKILLS})\s*\??\s*$",
    re.I,
)
_PERSON_NAME = r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)"
_SALARY_NAME_RE = re.compile(
    rf"\b(?:what(?:'s|\s+is)\s+)?{_PERSON_NAME}\s*'s\s+(?:salary|pay|compensation)\b|"
    rf"\b(?:salary|pay|compensation)\s+(?:of|for)\s+(?:the\s+)?{_PERSON_NAME}\b|"
    rf"\bhow\s+much\s+does\s+(?:the\s+)?{_PERSON_NAME}\s+(?:make|earn|get\s+paid)\b",
    re.I,
)
_DEPT_ALIASES: dict[str, str] = {
    "engineering": "Engineering",
    "engineers": "Engineering",
    "engineer": "Engineering",
    "engeneering": "Engineering",
    "eng": "Engineering",
    "sales": "Sales",
    "finance": "Finance",
    "product": "Product",
    "operations": "Operations",
    "ops": "Operations",
    "people": "People",
    "hr": "People",
}
_META_COUNT_RE = re.compile(
    r"\b("
    r"how many was that(?: again)?|"
    r"what was (?:the|that) (?:count|number|total)|"
    r"(?:repeat|remind me of|recall) (?:the|that) (?:count|number)|"
    r"(?:same|that) (?:count|number) again|"
    r"how many (?:was|were) (?:there|that)"
    r")\b",
    re.I,
)
_PRONOUN_ONLY_RE = re.compile(r"\b(she|he|her|him|his|hers)\b", re.I)
_PERSON_ATTR_RE = re.compile(
    r"\b("
    r"live|lives|living|located|based|about|profile|manager|"
    r"education|degree|school|university|college|"
    r"title|position|role|job|"
    r"email|e-?mail|mail|"
    r"department|team|"
    r"status|hired|hire|phone|mobile|"
    r"where|"
    r"birth\s*day|birthdays?|date\s+of\s+birth|birth\s*date|\bdob\b|"
    r"born|how\s+old|age"
    r")\b",
    re.I,
)
_WHICH_OF_THEM_SKILL_RE = re.compile(
    r"\bwhich of them\s+(?:know|knows|have|has)\b",
    re.I,
)
_ANAPHORA_RE = re.compile(
    r"\b("
    r"of them|of those|from them|from those|from there|among them|among those|"
    r"that group|those employees|these employees|that (?:\d+\s+)?employees?|"
    r"from that (?:\d+\s+)?employees?|of that (?:\d+\s+)?(?:group|set|employees?)|"
    r"from (?:the )?(?:previous|prior|last) (?:list|set|group|search|results?)|"
    r"(?:the )?(?:previous|prior|last) (?:list|set|group)|"
    r"which of them|which ones?(?:\s+from|\s+know|\s+have|\s+joined)?|"
    r"do (?:any|they|those) (?:of them )?"
    r")\b",
    re.I,
)
_PREVIOUS_SEARCH_EMPLOYEES_RE = re.compile(
    r"\b(?:"
    r"employees?\s+from\s+(?:a\s+)?previous\s+search|"
    r"previous\s+search|"
    r"(?:the\s+)?(?:prior|last)\s+(?:set|cohort|search|results?)|"
    r"list\s+(?:those|them)|"
    r"employees?\s+please"
    r")\b",
    re.I,
)

_FOLLOWUP_NAMES_RE = re.compile(
    r"\b("
    r"their names?|the names?|there names?|"
    r"who are (they|those|them)|"
    r"list (them|those|their names?)|"
    r"say (their|there|the) names?|"
    r"name them|show (me )?them|what are (their|there) names?|"
    r"gimme\s+(?:the\s+)?names?|lemme\s+(?:see\s+)?(?:the\s+)?names?"
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
    r"give\s+(?:me\s+)?(?:the\s+)?names?|"
    r"gimme\s+(?:the\s+)?names?|lemme\s+(?:see\s+)?(?:the\s+)?names?"
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
    r"(?:\s+are\s+we\s+in)?"
    r")\b",
    re.I,
)
# Place names come from the shared vocabulary, so a city the corpus knows about
# is askable through every path rather than only the ones that remembered it.
_CITY_RE = re.compile(rf"\b(?:in|from)\s+({CITY_ALT})\b", re.IGNORECASE)
_COUNTRY_RE = re.compile(
    rf"\b(?:in|from|based in)\s+(?:the\s+)?({COUNTRY_ALT})\b", re.IGNORECASE
)
# Wording that is about location even when no place is named.
_LOCATION_TOPIC_RE = re.compile(
    r"\b(where|live|lives|living|located|location|based|city|cities|country|countries)\b",
    re.IGNORECASE,
)

_PERSON_LOCATION_RE = re.compile(
    r"\bwhere\s+(?:(?:does|do|is)\s+)?(?:the\s+)?"
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)"
    r"(?:\s+in\s+[A-Za-z]+)?"
    r"\s+(?:live|lives|living|located|from|based)\b",
    re.I,
)
_ABOUT_PERSON_RE = re.compile(
    r"(?:tell me about|who is|what about|profile of)\s+"
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)|"
    r"([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+)?)\s*(?:'s)?\s+"
    r"(?:profile|details)\b",
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


def _department_mentioned(lower: str) -> str | None:
    for alias, canon in _DEPT_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return canon
    return next((d for d in _DEPARTMENTS if d.lower() in lower), None)


def _detect_country(question: str) -> str | None:
    m = _COUNTRY_RE.search(question)
    return canon_country(m.group(1)) if m else None


def _detect_city(question: str) -> str | None:
    m = _CITY_RE.search(question)
    return canon_city(m.group(1)) if m else None


_PLACE_PHRASE_RE = re.compile(
    r"\b(?:in|from|based\s+in|living\s+in|located\s+in|near)\s+"
    r"([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)?)\b"
)


def _unknown_place_phrase(question: str) -> str | None:
    """Return a capitalised place token that is not in the closed vocab / depts."""
    for m in _PLACE_PHRASE_RE.finditer(question or ""):
        raw = m.group(1).strip()
        if not raw:
            continue
        if canon_city(raw) or canon_country(raw):
            continue
        if any(d.lower() == raw.lower() for d in _DEPARTMENTS):
            continue
        # Skip common non-place capitals ("Engineering" already handled; roles).
        if raw.lower() in {
            "python",
            "react",
            "java",
            "kubernetes",
            "docker",
            "aws",
            "salesforce",
        }:
            continue
        return raw
    return None


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
        or _META_COUNT_RE.search(q)
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
    if dimension in LOCATION_FIELDS:
        return location_facet_plan(dimension, count=True)
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
    if dimension in LOCATION_FIELDS:
        return location_facet_plan(dimension)
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


def _employee_by_name_plan(
    name: str, *, department: str | None = None, with_location: bool = False
) -> ExecutionPlan:
    params: dict = {"action": "by_name", "name": name.strip()}
    if department:
        params["department"] = department
    nodes = [PlanNode(id="e1", kind="tool", name="employee", params=params)]
    if with_location:
        nodes.append(profile_location_node("e1"))
    return ExecutionPlan(
        nodes=nodes,
        response_strategy="template",
        active_cohort_node="e1",
    )


def _employee_by_id_plan(employee_id: str, *, with_location: bool = False) -> ExecutionPlan:
    nodes = [
        PlanNode(
            id="e1",
            kind="tool",
            name="employee",
            params={"action": "by_id", "employee_id": str(employee_id)},
        )
    ]
    if with_location:
        nodes.append(profile_location_node("e1"))
    return ExecutionPlan(
        nodes=nodes,
        response_strategy="template",
        active_cohort_node="e1",
    )


def _meta_count_plan(memory: SessionMemory | None) -> ExecutionPlan | None:
    """Answer 'how many was that again?' from tool_fact_cache or prior cohort — never nl2sql."""
    if memory and memory.tool_fact_cache.get("last_count") is not None:
        value = memory.tool_fact_cache["last_count"].value
        try:
            n = int(value)
        except (TypeError, ValueError):
            n = value
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=f"The answer is {n}.",
        )
    prior = list(memory.last_employee_ids) if memory and memory.last_employee_ids else []
    if prior and len(prior) <= 50:
        return _sql_over_ids(prior, count_only=True)
    return ExecutionPlan(
        nodes=[],
        response_strategy="template",
        clarify_question=(
            "I don't have a previous count in this conversation. "
            "Ask a headcount question first, then I can repeat it."
        ),
    )


def _entity_display_name(memory: SessionMemory | None, employee_id: str) -> str:
    if memory:
        for ent in list(memory.last_listed or []) + list(memory.entity_memory or []):
            if str(ent.employee_id) == str(employee_id):
                return ent.display_name or "that employee"
    return "that employee"


def _entity_id_for_display_name(
    memory: SessionMemory | None, display_name: str
) -> str | None:
    if not memory or not display_name:
        return None
    needle = display_name.strip().lower()
    for ent in list(memory.last_listed or []) + list(memory.entity_memory or []):
        if (ent.display_name or "").strip().lower() == needle:
            return str(ent.employee_id)
        for alias in ent.aliases:
            if (alias or "").strip().lower() == needle:
                return str(ent.employee_id)
    return None


def _resolve_pronoun_employee_id(
    question: str, memory: SessionMemory | None
) -> str | None:
    if not memory or not memory.person_bindings:
        return None
    lower = question.lower()
    for key in ("she", "he", "her", "him", "his", "hers"):
        if re.search(rf"\b{key}\b", lower) and key in memory.person_bindings:
            return str(memory.person_bindings[key])
    return None


_COHORT_STATUS_READ_RE = re.compile(
    r"\b("
    r"status(?:es)?\s+of\s+(?:them|those|these|the\s+(?:group|list|set|people|employees))|"
    r"(?:their|there|these|those)\s+status(?:es)?|"
    r"(?:what|whats|what's|give|show|list|tell).{0,48}\b"
    r"(?:statuses|status\s+of\s+(?:them|those)|their\s+status|there\s+status)\b|"
    r"\bstatuses?\s+please\b|"
    r"give\s+(?:me\s+)?(?:their|there)\s+status(?:es)?"
    r")\b",
    re.I,
)


def try_cohort_status_read_plan(
    question: str, *, memory: SessionMemory | None = None
) -> ExecutionPlan | None:
    """List agent + employment status for the prior cohort (read, not write)."""
    q = (question or "").strip()
    if not q or not _COHORT_STATUS_READ_RE.search(q):
        return None
    # Singular pronoun status reads stay on the person path ("her status").
    if re.search(r"\b(her|his|she|he)\b", q, re.I) and not re.search(
        r"\b(them|those|their|there|statuses)\b", q, re.I
    ):
        return None
    # Writes are handled elsewhere.
    if extract_status_change(q).matched:
        return None
    prior = list(memory.last_employee_ids) if memory and memory.last_employee_ids else []
    listed = list(memory.last_listed) if memory and memory.last_listed else []
    if not prior and listed:
        prior = [str(e.employee_id) for e in listed if e.employee_id]
    if not prior:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                "Which employees' statuses should I show? "
                "List a cohort first (for example: names in Dubai), then ask again."
            ),
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )
    columns = [
        "id",
        "first_name",
        "last_name",
        "department",
        "position",
        "status",
        "employment_status",
    ]
    return _sql_over_ids(prior, count_only=False, columns=columns)


def try_status_write_plan(
    question: str,
    *,
    memory: SessionMemory | None = None,
    scope_filters: dict | None = None,
) -> ExecutionPlan | None:
    """Compile a set_status DAG when the cheap status extractor matches.

    Used as a PlanCompiler guard so seeded English writes stay deterministic;
    novel paraphrases fall through to the tool-selecting planner (HITL).
    """
    q = (question or "").strip()
    if not q:
        return None
    status_req = extract_status_change(q)
    if not status_req.matched:
        return None
    from app.application.understanding.plan_from_query_state import _set_status_plan
    from app.application.understanding.status_change import (
        bind_status_subject_from_memory,
        is_deictic_status_subject,
    )
    from app.domain.query_state import FilterSlot, QueryState as _QS

    person_name = status_req.person_name
    if is_deictic_status_subject(q) or (
        person_name
        and person_name.lower().split()[0] in {"current", "this", "that", "same"}
    ):
        person_name = None
    filters: dict = dict(scope_filters or {})
    slots: list[FilterSlot] = []
    if status_req.city:
        filters["city"] = status_req.city
        slots.append(
            FilterSlot(field="city", op="eq", value=status_req.city, confidence=0.95)
        )
    if status_req.country:
        filters["country"] = status_req.country
        slots.append(
            FilterSlot(
                field="country", op="eq", value=status_req.country, confidence=0.95
            )
        )
    bound_id, bound_name = bind_status_subject_from_memory(
        person_name=person_name,
        email=status_req.email,
        employee_id=status_req.employee_id,
        city=status_req.city,
        country=status_req.country,
        memory=memory,
    )
    if (
        not bound_id
        and not person_name
        and not status_req.email
        and not status_req.city
        and not status_req.country
    ):
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question="Which employee's status should I update?",
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )
    return _set_status_plan(
        _QS(
            intent="set_status",
            person_name=person_name or bound_name,
            status_value=status_req.status_value,
            status_email=status_req.email,
            status_employee_id=status_req.employee_id or bound_id,
            person_employee_ids=[bound_id] if bound_id else [],
            filters=slots,
            confidence=0.95,
        ),
        filters=filters,
    )


def wants_person_lookup(question: str) -> bool:
    """True when the question is asking about a specific person's profile attributes."""
    q = (question or "").strip()
    lower = q.lower()
    if _PERSON_LOCATION_RE.search(q) or _ABOUT_PERSON_RE.search(q):
        return True
    if _PERSON_ATTR_RE.search(q):
        return True
    # Possessive attribute: "her job", "his email", "what's her …"
    if re.search(r"\b(her|his)\s+\w+", lower):
        return True
    if re.search(r"\b(?:what|where|who)\b.*\b(she|he|her|him)\b", lower):
        return True
    return False


_MAX_ELLIPTICAL_TOKENS = 8

# Org/cohort asks must never bind to the focused person ("how many employees…",
# "names who live in Dubai") even when session focus is a single employee.
_COHORT_ASK_RE = re.compile(
    r"\b("
    r"how\s+many|how\s+much|"
    r"(?:all|which|list|show|give|gimme|find)\s+"
    r"(?:(?:the|all|our)\s+)?(?:employees?|people|folks|staff|names?|ppl)|"
    r"employees?\s+(?:who|that|with|have|knows?|lives?|living|in|from)|"
    r"names?\s+(?:who|that|of)|"
    r"who\s+(?:knows?|lives?|works?|are|is\s+in)|"
    r"headcount|total\s+employees?"
    r")\b",
    re.I,
)


def is_cohort_question(question: str) -> bool:
    """True when the utterance is about a set of people, not the bound person."""
    return bool(_COHORT_ASK_RE.search(question or ""))


def _focal_bound_employee_id(memory: SessionMemory | None) -> str | None:
    """Singular focus for elliptical follow-ups (bindings / active referent).

    Prefer the active cohort / last listed person over older pronoun bindings so
    "the current employee" tracks the latest focus, not a stale he/she id.
    """
    if memory is None:
        return None
    if memory.active_referent and len(memory.active_referent.ids) == 1:
        return str(memory.active_referent.ids[0])
    if memory.last_listed and len(memory.last_listed) == 1:
        return str(memory.last_listed[0].employee_id)
    for key in ("she", "he", "her", "him", "his", "hers"):
        eid = memory.person_bindings.get(key)
        if eid:
            return str(eid)
    return None


def elliptical_bound_person_plan(
    question: str, *, memory: SessionMemory | None
) -> ExecutionPlan | None:
    """Bare attribute follow-up about the bound person ("what is the education?").

    Education / title / email / department live on ``employees`` — never invent a
    resume_search purpose like ``education_person``.
    """
    q = (question or "").strip()
    if not q or not wants_person_lookup(q):
        return None
    if len(q.split()) > _MAX_ELLIPTICAL_TOKENS:
        return None
    # Org-wide / cohort asks beat singular person focus.
    if is_cohort_question(q):
        return None
    # Named or pronoun subjects are handled on dedicated paths.
    if _PRONOUN_ONLY_RE.search(q):
        return None
    if _ABOUT_PERSON_RE.search(q):
        return None
    from app.application.memory.context_view import extract_lookup_name_candidate

    if extract_lookup_name_candidate(q):
        return None
    if _match_entity_name(q, memory):
        return None
    if extract_manager_subject(q) or extract_reports_subject(q):
        return None

    # Status writes must not fall through to a by_id status *read*.
    status_req = extract_status_change(q)
    if status_req.matched:
        from app.application.understanding.plan_from_query_state import _set_status_plan
        from app.application.understanding.status_change import (
            bind_status_subject_from_memory,
            is_deictic_status_subject,
        )
        from app.domain.query_state import QueryState as _QS

        # Fake names like "current employee" are deictic — bind from session focus.
        person_name = status_req.person_name
        if is_deictic_status_subject(q) or (
            person_name and person_name.lower().split()[0]
            in {"current", "this", "that", "same"}
        ):
            person_name = None
        bound_id, bound_name = bind_status_subject_from_memory(
            person_name=person_name,
            email=status_req.email,
            employee_id=status_req.employee_id,
            city=status_req.city,
            country=status_req.country,
            memory=memory,
        )
        if not bound_id and not person_name and not status_req.email:
            return ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question="Which employee's status should I update?",
                refusal_code=RefusalCode.AMBIGUOUS.value,
            )
        return _set_status_plan(
            _QS(
                intent="set_status",
                person_name=person_name or bound_name,
                status_value=status_req.status_value,
                status_email=status_req.email,
                status_employee_id=status_req.employee_id or bound_id,
                person_employee_ids=[bound_id] if bound_id else [],
                confidence=0.95,
            ),
            filters={},
        )

    bound_id = _focal_bound_employee_id(memory)
    if not bound_id:
        return None
    name = _entity_display_name(memory, bound_id)
    asks_where = bool(_PERSON_LOCATION_RE.search(q)) or bool(
        _LOCATION_TOPIC_RE.search(q)
    )
    if asks_where:
        return location_person_plan(name, employee_ids=[bound_id])
    from app.application.understanding.birthday import extract_birthday

    birthday = extract_birthday(q)
    if birthday.matched:
        # Cohort birthday asks must not dump the bound person's profile.
        if birthday.scope != "person":
            return None
        from app.application.understanding.plan_from_query_state import _birthday_plan
        from app.domain.query_state import QueryState as _QS

        return _birthday_plan(
            _QS(
                intent="birthday",
                birthday_scope="person",
                person_name=name,
                person_employee_ids=[bound_id],
                wants_age=birthday.wants_age,
                wants_wish=birthday.wants_wish,
                confidence=0.95,
            )
        )
    if re.search(r"\bmanagers?\b", q, re.I):
        return _manager_plan(q, name=name, employee_id=bound_id)
    # SQL-backed profile fields (education, email, department, position, …).
    return _employee_by_id_plan(bound_id)


def _pronoun_clarify_plan() -> ExecutionPlan:
    return ExecutionPlan(
        nodes=[],
        response_strategy="template",
        clarify_question=(
            "Which employee do you mean? Tell me their name "
            "(or pick one from the previous options)."
        ),
        refusal_code=RefusalCode.AMBIGUOUS.value,
    )


def _manager_plan(
    question: str,
    *,
    name: str | None = None,
    department: str | None = None,
    employee_id: str | None = None,
) -> ExecutionPlan:
    params: dict = {"action": "manager", "question": question}
    if employee_id:
        params["employee_id"] = str(employee_id)
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


def _reports_plan(
    *,
    name: str | None = None,
    employee_id: str | None = None,
    department: str | None = None,
) -> ExecutionPlan:
    if not name and not employee_id:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question="Whose direct reports should I list?",
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )
    params: dict = {"action": "reports"}
    if employee_id:
        params["employee_id"] = str(employee_id)
    if name:
        params["name"] = name
        params["question"] = name
    if department:
        params["department"] = department
    return ExecutionPlan(
        nodes=[PlanNode(id="e1", kind="tool", name="employee", params=params)],
        response_strategy="template",
        active_cohort_node="e1",
    )


def _reports_skill_place_plan(
    *,
    manager_name: str,
    skill: str,
    city: str | None = None,
    country: str | None = None,
) -> ExecutionPlan:
    """Compose manager → reports ∩ skill RAG ∩ optional place."""
    if not manager_name or not skill:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                "I need both a manager and a skill to answer that "
                "(for example: who on Alice's team knows Kubernetes?)."
            ),
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )
    nodes: list[PlanNode] = [
        PlanNode(
            id="mgr",
            kind="tool",
            name="employee",
            params={"action": "reports", "name": manager_name, "question": manager_name},
        ),
        PlanNode(
            id="r1",
            kind="tool",
            name="resume_search",
            depends_on=["mgr"],
            params={"question": f"employees with {skill} experience"},
            input_bindings={"employee_ids": "nodes.mgr"},
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
    place: dict = {}
    if city:
        place["city"] = city
    if country:
        place["country"] = country
    if place:
        loc_nodes, loc_source = location_cohort_nodes(
            city=place.get("city"), country=place.get("country")
        )
        nodes = loc_nodes + nodes
        nodes.append(
            PlanNode(
                id="skloc",
                kind="operator",
                name="intersect_ids",
                depends_on=[id_source, loc_source],
                input_bindings={
                    "data": f"nodes.{id_source}",
                    "other": f"nodes.{loc_source}",
                },
            )
        )
        id_source = "skloc"
    nodes.append(
        PlanNode(
            id="sql1",
            kind="tool",
            name="sql",
            depends_on=[id_source],
            params={
                "mode": "constrained",
                "count_only": False,
                "filters": {},
                "columns": _NAME_COLUMNS,
            },
            input_bindings={"employee_ids": f"nodes.{id_source}"},
        )
    )
    return ExecutionPlan(
        nodes=nodes,
        response_strategy="template",
        active_cohort_node=id_source,
    )


def _match_entity_name(question: str, memory: SessionMemory | None) -> str | None:
    if not memory or not memory.entity_memory:
        return None
    lower = question.lower()
    entities = sorted(memory.entity_memory, key=lambda e: len(e.display_name), reverse=True)
    # Explicit multi-token person in the question beats first-name aliases.
    # "Tell me about Alice Nguyen" must not bind remembered "Alice Bauer".
    # Prefer Title-Case name pairs so "me about" is not captured.
    explicit = re.search(
        r"\b([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+)\b",
        question or "",
    )
    if explicit:
        asked = explicit.group(1).strip()
        asked_l = asked.lower()
        for ent in entities:
            name = ent.display_name.strip()
            if name.lower() == asked_l:
                return name
            aliases = [name.lower(), *[a.lower() for a in ent.aliases if a]]
            if asked_l in aliases:
                return name
        # Conflicting last names among remembered people → do not bind.
        asked_tokens = asked_l.split()
        asked_last = asked_tokens[-1]
        asked_first = asked_tokens[0]
        for ent in entities:
            name = ent.display_name.strip()
            tokens = name.lower().split()
            if (
                len(tokens) >= 2
                and tokens[0] == asked_first
                and tokens[-1] != asked_last
            ):
                return None
        return None
    # Exact full-name / alias containment first
    for ent in entities:
        name = ent.display_name.strip()
        if name and name.lower() in lower:
            return name
        for alias in ent.aliases:
            if alias and alias.lower() in lower and " " in alias.strip():
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
                    "columns": _NAME_COLUMNS,
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


_JOINED_RE = re.compile(
    r"\b(?:joined|hired|started(?:\s+work(?:ing)?)?|new\s+hires?)\b",
    re.I,
)


_MONTH_NAME_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _quarter_bounds(year: int, quarter: int) -> tuple[str, str]:
    start_month = 1 + (quarter - 1) * 3
    end_month = start_month + 3
    end_year = year
    if end_month > 12:
        end_month = 1
        end_year = year + 1
    return f"{year}-{start_month:02d}-01", f"{end_year}-{end_month:02d}-01"


def _hire_date_filters(question: str, *, today: date | None = None) -> dict[str, str] | None:
    """Map join/hire phrasing onto constrained hire_date filters."""
    q = question or ""
    if not _JOINED_RE.search(q):
        return None
    today = today or date.today()
    lower = q.lower()
    if re.search(r"\bthis\s+year\b", lower):
        y = today.year
        return {
            "hire_date_gte": f"{y}-01-01",
            "hire_date_lt": f"{y + 1}-01-01",
        }
    if re.search(r"\blast\s+year\b", lower):
        y = today.year - 1
        return {
            "hire_date_gte": f"{y}-01-01",
            "hire_date_lt": f"{y + 1}-01-01",
        }
    if re.search(r"\bthis\s+quarter\b", lower):
        qtr = (today.month - 1) // 3 + 1
        gte, lt = _quarter_bounds(today.year, qtr)
        return {"hire_date_gte": gte, "hire_date_lt": lt}
    if re.search(r"\blast\s+quarter\b", lower):
        qtr = (today.month - 1) // 3 + 1
        year = today.year
        qtr -= 1
        if qtr < 1:
            qtr = 4
            year -= 1
        gte, lt = _quarter_bounds(year, qtr)
        return {"hire_date_gte": gte, "hire_date_lt": lt}
    m = re.search(r"\blast\s+(\d{1,3})\s+days?\b", lower)
    if m:
        n = int(m.group(1))
        start = today.fromordinal(today.toordinal() - n)
        return {
            "hire_date_gte": start.isoformat(),
            "hire_date_lt": today.fromordinal(today.toordinal() + 1).isoformat(),
        }
    m = re.search(r"\blast\s+(\d{1,2})\s+months?\b", lower)
    if m:
        n = int(m.group(1))
        month = today.month - n
        year = today.year
        while month <= 0:
            month += 12
            year -= 1
        return {
            "hire_date_gte": f"{year}-{month:02d}-01",
            "hire_date_lt": today.fromordinal(today.toordinal() + 1).isoformat(),
        }
    m = re.search(
        r"\bsince\s+(january|february|march|april|may|june|july|august|"
        r"september|october|november|december|jan|feb|mar|apr|jun|jul|"
        r"aug|sep|sept|oct|nov|dec)\s+(20\d{2})\b",
        lower,
    )
    if m:
        month = _MONTH_NAME_TO_NUM[m.group(1)]
        year = int(m.group(2))
        return {"hire_date_gte": f"{year}-{month:02d}-01"}
    m = re.search(r"\b(?:in|during)\s+(20\d{2})\b", lower)
    if m:
        y = int(m.group(1))
        return {
            "hire_date_gte": f"{y}-01-01",
            "hire_date_lt": f"{y + 1}-01-01",
        }
    m = re.search(r"(?:after|since)\s+(20\d{2})", lower)
    if m:
        return {"hire_date_gt": f"{m.group(1)}-01-01"}
    return None


_TENURE_RE = re.compile(
    r"\b(?:average|avg|mean)\s+tenure\b|\btenure\s+(?:in|for|by)\b|"
    r"\bhow\s+long\s+(?:have|has)\s+(?:people|employees|staff)\b",
    re.I,
)
_LONGEST_TENURED_RE = re.compile(
    r"\b(?:longest[\s-]?tenured|most\s+senior|earliest\s+hire|"
    r"who\s+(?:has|have)\s+(?:been\s+)?(?:here|with\s+us)\s+longest)\b",
    re.I,
)


def _tenure_agg_plan(*, department: str | None = None) -> ExecutionPlan:
    filters: dict = {}
    if department:
        filters["department"] = department
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "template": "agg_tenure" if department else "agg_tenure_by_dept",
                    "filters": filters,
                    "answer_hint": (
                        "Tenure is computed from hire_date "
                        "(CURRENT_DATE - hire_date), not job title level."
                    ),
                },
            )
        ],
        response_strategy="template",
    )


def _longest_tenured_plan(
    *, department: str | None = None, limit: int = 5
) -> ExecutionPlan:
    filters: dict = {}
    if department:
        filters["department"] = department
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "template": "longest_tenured",
                    "filters": filters,
                    "limit": limit,
                    "answer_hint": (
                        "Ranked by earliest hire_date (longest tenure). "
                        "This is not the same as title seniority."
                    ),
                },
            )
        ],
        response_strategy="template",
        active_cohort_node="sql1",
    )


def _hire_date_list_plan(filters: dict) -> ExecutionPlan:
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": False,
                    "filters": dict(filters),
                    "columns": _NAME_COLUMNS,
                },
            )
        ],
        response_strategy="template",
        active_cohort_node="sql1",
    )


def _resume_search_plan(
    question: str,
    *,
    count_only: bool = False,
    hire_date_gt: str | None = None,
    intersect_with: list[str] | None = None,
    scope_filters: dict | None = None,
    city: str | None = None,
    country: str | None = None,
    skill: str | None = None,
) -> ExecutionPlan:
    """RAG plan with optional prior-cohort / place / SQL-filter scoping.

    - intersect_with: prior employee IDs (SQL→RAG). Passed into resume_search and
      intersected after extract as a safety net.
    - city/country: second retrieval pass; ids intersected so SQL never sees place.
    - scope_filters: remembered department/… when IDs were not materialized.
    """
    place, sql_scope = split_location_filters(dict(scope_filters or {}))
    if city:
        place["city"] = city
    if country:
        place["country"] = country

    resume_params: dict = {"question": question}
    if skill:
        resume_params["purpose"] = "skill"
        resume_params["skill"] = skill
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
    if place:
        loc_nodes, loc_source = location_cohort_nodes(
            city=place.get("city"),
            country=place.get("country"),
        )
        nodes = loc_nodes + nodes
        nodes.append(
            PlanNode(
                id="skloc",
                kind="operator",
                name="intersect_ids",
                depends_on=[id_source, loc_source],
                input_bindings={
                    "data": f"nodes.{id_source}",
                    "other": f"nodes.{loc_source}",
                },
            )
        )
        id_source = "skloc"

    # Always materialize through SQL for list answers so display order and
    # SessionMemory.last_listed stay aligned for ordinal follow-ups ("first one").
    # llm_format-only skill lists used to show names without updating last_listed.
    filters: dict = dict(sql_scope)
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
        response_strategy="template",
        active_cohort_node="sql1" if list_names else id_source,
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
    # A remembered place is a retrieval scope, not a SQL filter.
    remembered_city, remembered_country = place_from_constraint_memory(
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
            clarify_question=unsupported_answer_for(q),
            refusal_code=RefusalCode.OUT_OF_SCOPE.value,
        )

    # Casual org headcount ("headcount?", "total employees?", "what's our headcount?")
    if _ORG_HEADCOUNT_RE.match(q) or re.fullmatch(
        r"\s*(?:what(?:'s|\s+is)\s+)?(?:our\s+|the\s+)?headcount\s*\??\s*",
        q,
        flags=re.I,
    ):
        return _filtered_count_plan({})

    # Bare department follow-up after org headcount ("engineering?")
    bare_dept = _BARE_DEPT_RE.match(q)
    if bare_dept:
        dept = _department_mentioned(bare_dept.group(1).lower())
        if dept:
            return _department_count_plan(dept)

    # Named salary (recruiter ACL already passed in the compiler).
    sal = _SALARY_NAME_RE.search(q)
    if sal:
        name = next((g for g in sal.groups() if g), None)
        if name:
            return _employee_by_name_plan(name.strip())

    # Bare skill token ("python?") — expand to a skill ask; count in prior cohort.
    bare_skill = _BARE_SKILL_RE.match(q)
    if bare_skill:
        skill = bare_skill.group(1)
        return _resume_search_plan(
            f"who knows {skill}",
            count_only=bool(prior_ids or scope_filters),
            intersect_with=prior_ids or None,
            scope_filters=scope_filters if (scope_filters and not prior_ids) else None,
            skill=skill,
        )

    # Birthdays live only in resume text — always retrieval, never SQL
    from app.application.understanding.birthday import extract_birthday

    birthday_req = extract_birthday(q)
    if birthday_req.matched:
        from app.application.understanding.plan_from_query_state import _birthday_plan
        from app.domain.query_state import QueryState as _QS

        # "closest birthday from there" must stay inside the prior cohort.
        scoped_ids = (
            prior_ids
            if (
                birthday_req.scope in {"closest", "upcoming", "today", "month"}
                and refers
                and prior_ids
            )
            else []
        )
        return _birthday_plan(
            _QS(
                intent="birthday",
                birthday_scope=birthday_req.scope,
                birthday_month=birthday_req.month,
                person_name=birthday_req.person_name,
                person_employee_ids=list(scoped_ids),
                refers_to_prior=bool(scoped_ids) or refers,
                wants_age=birthday_req.wants_age,
                wants_wish=birthday_req.wants_wish,
                confidence=0.95,
            )
        )

    from app.application.understanding.languages import extract_language

    language_req = extract_language(q)
    if language_req.matched:
        from app.application.understanding.plan_from_query_state import _languages_plan
        from app.domain.query_state import QueryState as _QS

        scoped_ids = (
            prior_ids
            if language_req.refers_to_prior or (refers and not language_req.person_name)
            else []
        )
        if scoped_ids and not prior_ids:
            scoped_ids = []
        return _languages_plan(
            _QS(
                intent="languages",
                language=language_req.language,
                person_name=language_req.person_name,
                person_employee_ids=list(scoped_ids) if not language_req.person_name else [],
                refers_to_prior=bool(scoped_ids) or language_req.refers_to_prior,
                confidence=0.95,
            )
        )

    from app.application.understanding.certifications import extract_certification

    cert_req = extract_certification(q)
    if cert_req.matched:
        from app.application.understanding.plan_from_query_state import (
            _certifications_plan,
        )
        from app.domain.query_state import QueryState as _QS

        scoped_ids = (
            prior_ids
            if cert_req.refers_to_prior or (refers and not cert_req.person_name)
            else []
        )
        return _certifications_plan(
            _QS(
                intent="certifications",
                certification=cert_req.certification,
                person_name=cert_req.person_name,
                person_employee_ids=list(scoped_ids) if not cert_req.person_name else [],
                refers_to_prior=bool(scoped_ids) or cert_req.refers_to_prior,
                confidence=0.95,
            )
        )

    # Agent status-flag write (any role; open access) — resolve via RAG for names
    status_plan = try_status_write_plan(
        q, memory=memory, scope_filters=scope_filters
    )
    if status_plan is not None:
        return status_plan

    # Meta-count: "how many was that again?" — use cached count / prior cohort, never nl2sql
    if _META_COUNT_RE.search(q):
        return _meta_count_plan(memory)

    pronoun_id = _resolve_pronoun_employee_id(q, memory)
    wants_person = wants_person_lookup(q)

    # "what is the education?" after a bound person — employees column, not RAG.
    elliptical = elliptical_bound_person_plan(q, memory=memory)
    if elliptical is not None:
        return elliptical

    # Named person from prior result set ("where ivy chen lives?")
    entity_name = _match_entity_name(q, memory)
    dept_hint = _dept_hint(q)
    # A person's location is written in their resume, so it is retrieved by name
    # rather than read off a profile row.
    asks_where = bool(_PERSON_LOCATION_RE.search(q)) or (
        wants_person and _LOCATION_TOPIC_RE.search(q) is not None
    )
    if entity_name and wants_person:
        if asks_where:
            return location_person_plan(entity_name)
        # Prefer the remembered id so short names ("Katya") stay on the listed person.
        bound = _entity_id_for_display_name(memory, entity_name)
        if bound:
            return _employee_by_id_plan(bound)
        return _employee_by_name_plan(entity_name, department=dept_hint)

    # Pronoun person questions ("where does she live?", "her job title") — never by_name("she")
    if wants_person and _PRONOUN_ONLY_RE.search(q):
        # Prefer explicit name tokens over pure pronouns when both appear
        loc = _PERSON_LOCATION_RE.search(q)
        captured = (loc.group(1) if loc else "").strip()
        if captured and captured.lower() not in {
            "she",
            "he",
            "her",
            "him",
            "his",
            "hers",
            "they",
            "them",
        }:
            return (
                location_person_plan(captured)
                if asks_where
                else _employee_by_name_plan(captured, department=dept_hint)
            )
        bound_id = pronoun_id
        if not bound_id and memory and len(memory.entity_memory) == 1:
            bound_id = str(memory.entity_memory[0].employee_id)
        if not bound_id:
            return _pronoun_clarify_plan()
        name = _entity_display_name(memory, bound_id)
        if asks_where:
            # The pronoun already fixes who, so read the place for that id — a
            # name lookup could land on a namesake.
            return location_person_plan(name, employee_ids=[bound_id])
        if re.search(r"\bmanagers?\b", q, re.I):
            return _manager_plan(q, name=name, employee_id=bound_id)
        return _employee_by_id_plan(bound_id)

    # "where does Ivy Chen live?" / "where the ivy chen lives"
    loc = _PERSON_LOCATION_RE.search(q)
    if loc:
        return location_person_plan(loc.group(1))

    # Direct reports / team (+ optional skill×place composition)
    reports_name = extract_reports_subject(q)
    if reports_name:
        skill_m = _SKILL_RE.search(q)
        if skill_m and _SKILL_INTENT_RE.search(q):
            return _reports_skill_place_plan(
                manager_name=reports_name,
                skill=skill_m.group(1),
                city=_detect_city(q) or remembered_city,
                country=_detect_country(q) or remembered_country,
            )
        return _reports_plan(name=reports_name, department=dept_hint)

    # Manager lookup (before generic analytics)
    mgr_name = extract_manager_subject(q)
    if mgr_name or re.search(r"\b(manager of|'s manager|who manages)\b", lower):
        return _manager_plan(q, name=mgr_name or entity_name, department=dept_hint)

    focus = memory.last_focus if memory else None

    # Clarify option / explicit recovery: "employees from a previous search"
    if _PREVIOUS_SEARCH_EMPLOYEES_RE.search(q):
        if prior_ids:
            return _sql_over_ids(prior_ids, count_only=False)
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                "I don't have a previous employee list in this conversation. "
                "Ask a search first (for example: who knows Python?), then I can list the names."
            ),
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )

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
                refusal_code=RefusalCode.AMBIGUOUS.value,
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

    # Follow-up: refine prior set by place ("of them in Berlin"). Retrieval decides
    # who is in Berlin; SQL then reads those ids from employees.
    city = _detect_city(q)
    country = _detect_country(q)
    if anaphora and (city or country):
        any_of_them = bool(re.search(r"\bany(?:one)?\s+of\s+them\b", q, re.I))
        return location_cohort_plan(
            city=city,
            country=country,
            count_only=bool(_COUNT_RE.search(q)) or any_of_them,
            intersect_with=prior_ids,
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

    # Follow-up without saved IDs ("of them from USA" after an org-wide headcount,
    # or a remembered constraint). A place applies globally through retrieval; the
    # remembered employees-table filters stay in SQL.
    if refers and not prior_ids and not _SKILL_RE.search(q):
        place_city = city or remembered_city
        place_country = country or remembered_country
        wants_count = bool(_COUNT_RE.search(q))
        listing = is_list_followup(q)
        if place_city or place_country:
            return location_cohort_plan(
                city=place_city,
                country=place_country,
                count_only=wants_count or not listing,
                extra_filters=scope_filters,
            )
        if scope_filters and (wants_count or listing):
            if wants_count:
                return _filtered_count_plan(scope_filters)
            return ExecutionPlan(
                nodes=[
                    PlanNode(
                        id="sql1",
                        kind="tool",
                        name="sql",
                        params={
                            "mode": "constrained",
                            "count_only": False,
                            "filters": dict(scope_filters),
                            "columns": _NAME_COLUMNS,
                        },
                    )
                ],
                response_strategy="template",
                active_cohort_node="sql1",
            )

    dept_mentioned = _department_mentioned(lower)

    # Tenure / longest-tenured (hire_date proxy — not title seniority).
    # Must beat department list ("most senior in Engineering").
    if _TENURE_RE.search(q):
        return _tenure_agg_plan(department=dept_mentioned)
    if _LONGEST_TENURED_RE.search(q):
        return _longest_tenured_plan(department=dept_mentioned, limit=5)

    # Count by department (global) — also materialize cohort IDs
    if dept_mentioned and not scoped_followup and _COUNT_RE.search(q):
        # "How many Engineering employees in Berlin": the department is an
        # employees column and the place is not, so each half goes to its owner.
        if city or country:
            return location_cohort_plan(
                city=city,
                country=country,
                count_only=True,
                extra_filters={"department": dept_mentioned},
            )
        return _department_count_plan(dept_mentioned)

    # List employees by department (global) — materialize names + IDs
    if dept_mentioned and not scoped_followup and _LIST_DEPT_RE.search(q) and not any(
        w in lower for w in ("how many", "count", "number of", "different")
    ):
        if city or country:
            return location_cohort_plan(
                city=city,
                country=country,
                extra_filters={"department": dept_mentioned},
            )
        return _department_list_plan(dept_mentioned)

    # Count/list by place (global): retrieval resolves the cohort, SQL names it.
    # Skill+place questions fall through so _resume_search_plan can intersect.
    if (
        (country or city)
        and not scoped_followup
        and not (_SKILL_RE.search(q) and _SKILL_INTENT_RE.search(q))
        and (
            _COUNT_RE.search(q)
            or any(w in lower for w in ("employee", "people", "staff", "list", "who"))
        )
    ):
        return location_cohort_plan(
            city=city,
            country=country,
            count_only=bool(_COUNT_RE.search(q)),
        )

    # List employees by role phrase ("software developers", "product managers")
    role = match_role_positions(q)
    if role is not None and not scoped_followup and not _COUNT_RE.search(q):
        if any(
            w in lower
            for w in ("list", "show", "who", "names", "employees", "people", "staff", "can you")
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
                            "filters": {"position": role},
                            "columns": _NAME_COLUMNS,
                        },
                    )
                ],
                response_strategy="template",
                active_cohort_node="sql1",
            )

    # Hire / join window ("who joined this year?", "hired in 2024")
    hire_filters = _hire_date_filters(q)
    if hire_filters:
        if _COUNT_RE.search(q):
            return _filtered_count_plan(hire_filters)
        return _hire_date_list_plan(hire_filters)

    # Resume / skills search — intersect prior set, place cohort, or constraint scope.
    skill_match = _SKILL_RE.search(q)
    if skill_match and _SKILL_INTENT_RE.search(q):
        year_match = re.search(r"(?:after|since)\s+(20\d{2})", lower)
        # "which of them know X?" is a cardinality question over the prior set
        count_only = bool(_COUNT_RE.search(q) or _WHICH_OF_THEM_SKILL_RE.search(q))
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
            city=city or remembered_city,
            country=country or remembered_country,
            skill=skill_match.group(1),
        )

    # "Tell me about Bob" / "Who is Alice Nguyen?" — the profile keeps its Location
    # line, filled from the resume of whoever the employee tool resolved.
    about = _ABOUT_PERSON_RE.search(q)
    if about:
        name = next((g for g in about.groups() if g), None)
        if name:
            return _employee_by_name_plan(
                name, department=dept_hint, with_location=True
            )

    # "do we have Sofia?" / "find Sofia" — directory lookup.
    existence = extract_person_existence_name(q)
    if existence:
        return _employee_by_name_plan(existence, department=dept_hint)
    from app.application.memory.context_view import extract_lookup_name_candidate

    lookup = extract_lookup_name_candidate(q)
    if lookup:
        return _employee_by_name_plan(lookup, department=dept_hint)

    # Generic structured employee analytics
    # Never send meta-count / pure pronoun questions to nl2sql (handled above).
    if _META_COUNT_RE.search(q) or (
        wants_person and _PRONOUN_ONLY_RE.search(q) and not entity_name
    ):
        return _meta_count_plan(memory) if _META_COUNT_RE.search(q) else _pronoun_clarify_plan()

    # Anything about location goes to retrieval, never to generated SQL: there is
    # no column to generate against, so nl2sql would fail validation.
    if _LOCATION_TOPIC_RE.search(q):
        if city or country:
            return location_cohort_plan(
                city=city, country=country, count_only=bool(_COUNT_RE.search(q))
            )
        dim = _detect_facet_dimension(q)
        if dim in LOCATION_FIELDS:
            return _facet_count_plan(dim) if _COUNT_RE.search(q) else _facet_list_plan(dim)
        if entity_name:
            return location_person_plan(entity_name)
        # "Where are our employees based?" — the honest answer is the place list.
        return location_facet_plan("city")

    # "employees in Atlantis" — place-like phrasing outside the closed vocab.
    unknown_place = _unknown_place_phrase(q)
    if unknown_place and any(
        w in lower for w in ("employee", "people", "staff", "list", "who", "based", "live")
    ):
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                f'I don\'t recognize "{unknown_place}" as a known city or country. '
                "Try one of the places we track (for example Berlin, Dubai, or USA)."
            ),
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )

    # Last-chance headcount / dept count before giving up.
    if _COUNT_RE.search(q) or "headcount" in lower:
        dept = _department_mentioned(lower)
        if dept:
            return _department_count_plan(dept)
        if not (city or country or _SKILL_RE.search(q)):
            return _filtered_count_plan({})

    # Gate residual analytics: prefer constrained templates / clarify over nl2sql.
    if any(
        w in lower
        for w in (
            "employee",
            "department",
            "hired",
            "how many",
            "count",
        )
    ):
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                "I couldn't build a reliable constrained plan for that. "
                "Try asking with a department, country, city, or job title "
                "(for example: how many Software Engineers in Engineering)."
            ),
        )

    return None
