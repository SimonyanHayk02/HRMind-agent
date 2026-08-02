from __future__ import annotations

import re

from app.application.planning.heuristic_planner import (
    _LONGEST_TENURED_RE,
    _TENURE_RE,
    _hire_date_filters,
    is_list_followup,
    refers_to_prior_set,
)
from app.application.planning.unsupported import is_unsupported_topic
from app.application.understanding.birthday import extract_birthday
from app.application.understanding.certifications import extract_certification
from app.application.understanding.languages import extract_language
from app.application.understanding.person_existence import extract_person_existence_name
from app.application.understanding.role_phrases import match_role_positions
from app.application.understanding.status_change import extract_status_change
from app.domain.query_state import FilterSlot, QueryState
from app.domain.schema_catalog import SchemaCatalog
from app.domain.session import SessionMemory
from app.tools.employee.tool import extract_manager_subject, extract_reports_subject

_COUNT_RE = re.compile(
    r"\b(how many|how much|count|number of|headcount|total\s+employees?|ppl)\b|"
    r"^\s*headcount\s*\??\s*$",
    re.I,
)
_WHICH_OF_THEM_SKILL_RE = re.compile(
    r"\bwhich of them\s+(?:know|knows|have|has)\b",
    re.I,
)
_FACET_RE = re.compile(
    r"\b(different|unique|distinct)\s+(countries|country|cities|city|departments|department|"
    r"education|statuses|status|positions|position)\b|"
    r"\b(?:which|what|list(?:\s+the)?)\s+(?:different\s+|unique\s+)?"
    r"(countries|country|cities|city|departments|department)"
    r"(?:\s+are\s+we\s+in)?\b|"
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
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)|"
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\s*(?:'s)?\s+"
    r"(?:profile|details)\b",
    re.I,
)
_YEAR_RE = re.compile(r"(?:after|since)\s+(20\d{2})", re.I)
_JOINED_RE = re.compile(
    r"\b(?:joined|hired|started(?:\s+work(?:ing)?)?|new\s+hires?)\b",
    re.I,
)

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

    # Birthdays come only from resume text, so claim them before person-lookup rules.
    birthday_req = extract_birthday(q)
    if birthday_req.matched:
        state.intent = "birthday"
        state.birthday_scope = birthday_req.scope
        state.birthday_month = birthday_req.month
        state.person_name = birthday_req.person_name
        state.wants_age = birthday_req.wants_age
        state.wants_wish = birthday_req.wants_wish
        state.refers_to_prior = refers_to_prior_set(q)
        has_subject = bool(birthday_req.person_name) or birthday_req.scope != "person"
        state.confidence = 0.95 if has_subject else 0.85
        if not has_subject:
            state.notes.append("missing_person")
        return state

    language_req = extract_language(q)
    if language_req.matched:
        state.intent = "languages"
        state.language = language_req.language
        state.person_name = language_req.person_name
        state.refers_to_prior = language_req.refers_to_prior or refers_to_prior_set(q)
        if language_req.scope == "person":
            state.confidence = 0.95 if language_req.person_name else 0.85
        else:
            state.confidence = 0.95 if language_req.language else 0.85
        return state

    cert_req = extract_certification(q)
    if cert_req.matched:
        state.intent = "certifications"
        state.certification = cert_req.certification
        state.person_name = cert_req.person_name
        state.refers_to_prior = cert_req.refers_to_prior or refers_to_prior_set(q)
        if cert_req.scope == "person":
            state.confidence = 0.95 if cert_req.person_name else 0.85
        else:
            state.confidence = 0.95 if cert_req.certification else 0.85
        return state

    # Status-flag writes take priority over profile/manager/facet "status" wording.
    status_req = extract_status_change(q)
    if status_req.matched:
        state.intent = "set_status"
        state.person_name = status_req.person_name
        state.status_value = status_req.status_value
        state.status_email = status_req.email
        state.status_employee_id = status_req.employee_id
        if status_req.city:
            state.filters.append(
                FilterSlot(field="city", op="eq", value=status_req.city, confidence=0.95)
            )
        if status_req.country:
            state.filters.append(
                FilterSlot(
                    field="country", op="eq", value=status_req.country, confidence=0.95
                )
            )
        has_subject = bool(
            status_req.person_name
            or status_req.email
            or status_req.employee_id
            or status_req.city
            or status_req.country
        )
        if has_subject and status_req.status_value is not None:
            state.confidence = 0.95
        else:
            state.confidence = 0.85
            if not has_subject:
                state.notes.append("missing_person")
            if status_req.status_value is None:
                state.notes.append("missing_status_value")
        return state

    state.refers_to_prior = refers_to_prior_set(q)
    state.filters = _extract_filters(q, catalog)
    state.want_count = bool(_COUNT_RE.search(q) or _WHICH_OF_THEM_SKILL_RE.search(q))

    # Bare department token ("engineering?") → count, not a full roster dump.
    if re.fullmatch(
        r"\s*(engineering|engineers|engeneering|sales|finance|product|operations|people)\s*\??\s*",
        q,
        flags=re.I,
    ):
        state.intent = "count"
        state.want_count = True
        state.confidence = 0.9
        return _merge_prior_filters(state, memory)

    # Tenure analytics must beat department-list / "who is …" profile salvage.
    if _TENURE_RE.search(q):
        state.intent = "tenure_agg"
        state.confidence = 0.95
        return _merge_prior_filters(state, memory)
    if _LONGEST_TENURED_RE.search(q):
        state.intent = "longest_tenured"
        state.confidence = 0.95
        return _merge_prior_filters(state, memory)

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
    elif _JOINED_RE.search(q):
        hire = _hire_date_filters(q)
        if hire:
            if "hire_date_gt" in hire:
                state.filters.append(
                    FilterSlot(
                        field="hire_date",
                        op="gt",
                        value=hire["hire_date_gt"],
                        confidence=0.9,
                    )
                )
            if "hire_date_gte" in hire:
                state.filters.append(
                    FilterSlot(
                        field="hire_date",
                        op="gte",
                        value=hire["hire_date_gte"],
                        confidence=0.9,
                    )
                )
            if "hire_date_lt" in hire:
                state.filters.append(
                    FilterSlot(
                        field="hire_date",
                        op="lt",
                        value=hire["hire_date_lt"],
                        confidence=0.9,
                    )
                )
            state.intent = "list" if not state.want_count else "count"
            state.confidence = 0.9
            return _merge_prior_filters(state, memory)

    skill = _SKILL_RE.search(q)
    if skill and _SKILL_INTENT_RE.search(q):
        state.skill = skill.group(1)
        state.intent = "skill_search"
        state.confidence = 0.85
        return _merge_prior_filters(state, memory)

    reports = extract_reports_subject(q)
    if reports:
        state.intent = "reports"
        state.person_name = reports
        skill = _SKILL_RE.search(q)
        if skill and _SKILL_INTENT_RE.search(q):
            state.skill = skill.group(1)
            state.confidence = 0.92
        else:
            state.confidence = 0.9
        return _merge_prior_filters(state, memory)

    mgr = extract_manager_subject(q)
    if mgr or re.search(r"\b(manager of|'s manager|who manages)\b", lower):
        state.intent = "manager"
        state.person_name = mgr
        state.confidence = 0.9 if mgr else 0.6
        return state

    about = _ABOUT_RE.search(q)
    if about and "manager" not in lower:
        name = next((g for g in about.groups() if g), None)
        if name:
            state.intent = "profile"
            state.person_name = name.strip()
            state.confidence = 0.85
            return state

    # "do we have Sofia?" / "find Sofia" — directory lookup (not prior-cohort refine).
    existence = extract_person_existence_name(q)
    if existence:
        state.intent = "profile"
        state.person_name = existence
        state.confidence = 0.88
        return state
    from app.application.memory.context_view import extract_lookup_name_candidate

    lookup = extract_lookup_name_candidate(q)
    if lookup:
        state.intent = "profile"
        state.person_name = lookup
        state.confidence = 0.88
        return state

    facet = _FACET_RE.search(q)
    if facet:
        # Prefer a dimension word ("cities") over a qualifier ("different").
        raw = next(
            (g for g in facet.groups() if g and g.lower() in _FACET_WORD),
            None,
        )
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

    # Longer values first so "MSc Data Science" wins over alias "msc".
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

    for field, canon, raw, conf in candidates:
        if field in used_fields:
            continue
        used_fields.add(field)
        if field == "education":
            expanded = _expand_education_filter(raw, canon, catalog)
            if len(expanded) > 1:
                found.append(
                    FilterSlot(field=field, op="in", value=expanded, confidence=conf)
                )
                continue
            if len(expanded) == 1:
                found.append(
                    FilterSlot(field=field, op="eq", value=expanded[0], confidence=conf)
                )
                continue
        found.append(FilterSlot(field=field, op="eq", value=canon, confidence=conf))
    return found


# Degree-level aliases → match every catalog education that is that level.
# Exact values like "MSc Data Science" stay exact (matched as a catalog value first).
_EDU_LEVEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "masters": re.compile(r"\b(msc|mba|masters?)\b", re.I),
    "master": re.compile(r"\b(msc|mba|masters?)\b", re.I),
    "master's": re.compile(r"\b(msc|mba|masters?)\b", re.I),
    "msc": re.compile(r"\bmsc\b", re.I),
    "mba": re.compile(r"\bmba\b", re.I),
    "bachelor": re.compile(r"\b(bsc|ba|bachelors?)\b", re.I),
    "bachelors": re.compile(r"\b(bsc|ba|bachelors?)\b", re.I),
    "bachelor's": re.compile(r"\b(bsc|ba|bachelors?)\b", re.I),
    "bs": re.compile(r"\b(bsc|bs)\b", re.I),
    "bsc": re.compile(r"\bbsc\b", re.I),
    "ba": re.compile(r"\bba\b", re.I),
}


def _expand_education_filter(
    matched_raw: str, canon: str, catalog: SchemaCatalog
) -> list[str]:
    """Map degree-level wording onto every matching education value in the catalog."""
    col = catalog.by_name("education")
    values = list(col.values) if col is not None else []
    key = (matched_raw or "").strip().lower()
    # Exact catalog value (including warmed free-text like "MSc Data Science").
    for v in values:
        if v.lower() == key:
            return [v]
    pat = _EDU_LEVEL_PATTERNS.get(key)
    if pat is not None and values:
        matched = [v for v in values if pat.search(v)]
        if matched:
            return matched
    return [canon] if canon else []
