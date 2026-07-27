from __future__ import annotations

import re

from app.application.planning.heuristic_planner import (
    is_list_followup,
    refers_to_prior_set,
)
from app.application.planning.unsupported import is_unsupported_topic
from app.application.understanding.role_phrases import match_role_positions
from app.domain.query_state import FilterSlot, QueryState
from app.domain.schema_catalog import SchemaCatalog
from app.domain.session import SessionMemory
from app.tools.employee.tool import extract_manager_subject

_COUNT_RE = re.compile(r"\b(how many|how much|count|number of)\b", re.I)
_WHICH_OF_THEM_SKILL_RE = re.compile(
    r"\bwhich of them\s+(?:know|knows|have|has)\b",
    re.I,
)
_FACET_RE = re.compile(
    r"\b(different|unique|distinct)\s+(countries|country|cities|city|departments|department|"
    r"education|statuses|status|positions|position)\b|"
    r"\b(?:which|what|list(?:\s+the)?)\s+(?:different\s+|unique\s+)?"
    r"(countries|country|cities|city|departments|department)\b|"
    r"\bin how(?:\s+many)?(?:\s+different)?\s+(countries|country|cities|city)\b",
    re.I,
)
_SKILL_RE = re.compile(
    r"\b(Python|Java|Go|React|Kubernetes|NLP|Machine Learning|SQL|AWS|Docker|"
    r"Salesforce|Recruiting|Accounting|Product Strategy)\b",
    re.I,
)
_SKILL_INTENT_RE = re.compile(
    r"\b(knows?|knowing|experience|experienced|skill|skills|proficient|familiar|"
    r"resume|develop(?:er|ers|ing|ed)?|coding|programm(?:er|ers|ing)?)\b",
    re.I,
)
_ABOUT_RE = re.compile(
    r"(?:tell me about|who is|what about|profile of)\s+"
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)",
    re.I,
)
_YEAR_RE = re.compile(r"(?:after|since)\s+(20\d{2})", re.I)

_FACET_WORD = {
    "countries": "country",
    "country": "country",
    "cities": "city",
    "city": "city",
    "departments": "department",
    "department": "department",
    "education": "education",
    "statuses": "employment_status",
    "status": "employment_status",
    "positions": "position",
    "position": "position",
}


def extract_query_state(
    question: str,
    *,
    catalog: SchemaCatalog,
    memory: SessionMemory | None = None,
) -> QueryState:
    """Catalog-grounded NLU: map the utterance onto intents + filter slots."""
    q = (question or "").strip()
    lower = q.lower()
    state = QueryState()

    if is_unsupported_topic(q):
        state.intent = "unsupported"
        state.confidence = 1.0
        return state

    state.refers_to_prior = refers_to_prior_set(q)
    state.filters = _extract_filters(q, catalog)
    state.want_count = bool(_COUNT_RE.search(q) or _WHICH_OF_THEM_SKILL_RE.search(q))

    year = _YEAR_RE.search(q)
    if year:
        state.hire_year_gt = int(year.group(1))
        state.filters.append(
            FilterSlot(
                field="hire_date",
                op="gt",
                value=f"{year.group(1)}-01-01",
                confidence=0.9,
            )
        )

    skill = _SKILL_RE.search(q)
    if skill and _SKILL_INTENT_RE.search(q):
        state.skill = skill.group(1)
        state.intent = "skill_search"
        state.confidence = 0.85
        return _merge_prior_filters(state, memory)

    mgr = extract_manager_subject(q)
    if mgr or re.search(r"\b(manager of|'s manager|who manages)\b", lower):
        state.intent = "manager"
        state.person_name = mgr
        state.confidence = 0.9 if mgr else 0.6
        return state

    about = _ABOUT_RE.search(q)
    if about and "manager" not in lower:
        state.intent = "profile"
        state.person_name = about.group(1).strip()
        state.confidence = 0.85
        return state

    facet = _FACET_RE.search(q)
    if facet:
        raw = next((g for g in facet.groups() if g), None)
        dim = _FACET_WORD.get((raw or "").lower())
        if dim:
            state.facet_dimension = dim
            state.intent = "facet_count" if state.want_count or "how" in lower else "facet_list"
            state.confidence = 0.9
            return state

    if is_list_followup(q) or (
        any(w in lower for w in ("list", "show", "who are", "names", "can you list"))
        and not state.want_count
    ):
        state.intent = "list"
        state.confidence = 0.85 if state.filters or state.refers_to_prior else 0.55
        return _merge_prior_filters(state, memory)

    if state.want_count or (
        state.refers_to_prior
        and state.filters
        and any(w in lower for w in ("are", "from", "in", "with"))
    ):
        state.intent = "count"
        # Org-wide headcount or refined of-them both OK
        if state.filters or state.refers_to_prior or re.search(r"\bemployees?\b", lower):
            state.confidence = 0.88
        else:
            state.confidence = 0.55
        return _merge_prior_filters(state, memory)

    if state.filters:
        state.intent = "list"
        state.confidence = 0.7
        return _merge_prior_filters(state, memory)

    state.intent = "unknown"
    state.confidence = 0.2
    return state


def _merge_prior_filters(state: QueryState, memory: SessionMemory | None) -> QueryState:
    """When user says of-them, keep prior structured filters unless overridden."""
    if not memory or not state.refers_to_prior:
        return state
    have = {f.field for f in state.filters}
    for c in memory.constraint_memory:
        if c.field in have:
            continue
        if c.field in {"department", "country", "city", "position", "education", "employment_status"}:
            if c.op == "eq":
                state.filters.append(
                    FilterSlot(field=c.field, op="eq", value=c.value, confidence=0.7)
                )
    return state


def _extract_filters(question: str, catalog: SchemaCatalog) -> list[FilterSlot]:
    lower = question.lower()
    found: list[FilterSlot] = []
    used_fields: set[str] = set()

    # Role phrases first (developers / software engineers → position IN (...))
    role = match_role_positions(question)
    if role is not None:
        if isinstance(role, list):
            found.append(
                FilterSlot(field="position", op="in", value=role, confidence=0.9)
            )
        else:
            found.append(
                FilterSlot(field="position", op="eq", value=role, confidence=0.92)
            )
        used_fields.add("position")

    # Longer values first so "New York" wins over "York" if present
    candidates: list[tuple[str, str, str, float]] = []  # field, canonical, matched, conf
    for col in catalog.filterable_columns():
        if col.kind != "enum":
            continue
        if col.name in used_fields:
            continue
        variants: list[tuple[str, str]] = []
        for v in col.values:
            variants.append((v, v))
        for alias, canon in col.aliases.items():
            variants.append((alias, canon))
        variants.sort(key=lambda x: len(x[0]), reverse=True)
        for raw, canon in variants:
            if not raw:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(raw.lower())}(?![a-z0-9])"
            if re.search(pattern, lower):
                candidates.append((col.name, canon, raw, 0.95))
                break  # one value per column per utterance for P0

    for field, canon, _raw, conf in candidates:
        if field in used_fields:
            continue
        used_fields.add(field)
        found.append(FilterSlot(field=field, op="eq", value=canon, confidence=conf))
    return found
