"""Validate residual SlotBundle and map it onto deterministic plans.

Ownership rules (non-negotiable):
- city/country never become SQL filters
- birthday / location person facts go through resume_search
- ordinals index SessionMemory.last_listed only
"""
from __future__ import annotations

from dataclasses import dataclass

from app.application.planning.bound_person import bound_person_attribute_plan
from app.application.planning.location_plans import (
    location_cohort_plan,
    location_facet_plan,
    location_person_plan,
)
from app.application.planning.plan_schema import ExecutionPlan
from app.application.planning.unsupported import UNSUPPORTED_ANSWER
from app.application.response.refusal import RefusalCode
from app.application.understanding.plan_from_query_state import plan_from_query_state
from app.application.understanding.slot_bundle import PersonRef, SlotBundle
from app.domain.places import canon_city, canon_country
from app.domain.query_state import FilterSlot, QueryState
from app.domain.session import SessionMemory

_MIN_CONFIDENCE = 0.75
_ORDINAL_WORDS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
    "tenth": 10,
    "10th": 10,
    "last": -1,
    "top": 1,
}


@dataclass
class SlotPlanResult:
    plan: ExecutionPlan | None = None
    clarify: str | None = None
    # True when slots were understood enough to act or clarify (do not LLM-plan).
    handled: bool = False


def plan_from_slots(
    bundle: SlotBundle,
    *,
    memory: SessionMemory | None,
    question: str = "",
) -> SlotPlanResult:
    """Return a deterministic plan, a clarify, or handled=False to fall through."""
    if bundle.confidence < _MIN_CONFIDENCE:
        return SlotPlanResult(handled=False)

    if bundle.intent == "unsupported":
        return SlotPlanResult(
            plan=ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question=UNSUPPORTED_ANSWER,
                refusal_code=RefusalCode.OUT_OF_SCOPE.value,
            ),
            handled=True,
        )

    if bundle.intent == "clarify" or bundle.intent == "unknown":
        if bundle.intent == "clarify":
            return SlotPlanResult(
                plan=ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=(
                        "I need a bit more detail — which employees or attribute "
                        "should I look up?"
                    ),
                    refusal_code=RefusalCode.AMBIGUOUS.value,
                ),
                handled=True,
            )
        return SlotPlanResult(handled=False)

    city = canon_city(bundle.city) if bundle.city else None
    country = canon_country(bundle.country) if bundle.country else None
    # Do not invent places outside the closed vocabulary — clarify instead.
    if bundle.city and not city:
        return SlotPlanResult(
            plan=ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question=(
                    f'I don\'t recognize "{bundle.city}" as a known city. '
                    "Try Berlin, Dubai, or another location in our directory."
                ),
                refusal_code=RefusalCode.AMBIGUOUS.value,
            ),
            handled=True,
        )
    if bundle.country and not country:
        return SlotPlanResult(
            plan=ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question=(
                    f'I don\'t recognize "{bundle.country}" as a known country. '
                    "Try a country from our directory (for example Germany or UAE)."
                ),
                refusal_code=RefusalCode.AMBIGUOUS.value,
            ),
            handled=True,
        )

    # Person-bound attributes (dob / location / profile / manager / status).
    person_plan = _person_attribute_plan(
        bundle, memory=memory, question=question, city=city, country=country
    )
    if person_plan.handled:
        return person_plan

    # Place cohort / facet (no person subject).
    if bundle.intent in {"location_cohort", "list", "count"} and (city or country):
        if bundle.attribute == "facet" or bundle.intent.startswith("facet"):
            dim = bundle.facet_dimension if bundle.facet_dimension in {"city", "country"} else "city"
            return SlotPlanResult(
                plan=location_facet_plan(dim, count=bundle.want_count or bundle.intent == "facet_count"),
                handled=True,
            )
        return SlotPlanResult(
            plan=location_cohort_plan(
                city=city, country=country, count_only=bundle.want_count or bundle.intent == "count"
            ),
            handled=True,
        )

    if bundle.intent in {"facet_count", "facet_list"} and bundle.facet_dimension:
        dim = bundle.facet_dimension
        if dim in {"city", "country"}:
            return SlotPlanResult(
                plan=location_facet_plan(dim, count=bundle.intent == "facet_count"),
                handled=True,
            )
        state = QueryState(
            intent=bundle.intent,
            facet_dimension=dim,
            want_count=bundle.intent == "facet_count",
            confidence=max(bundle.confidence, 0.8),
        )
        plan = plan_from_query_state(state, memory=memory, min_confidence=0.5)
        if plan is not None:
            return SlotPlanResult(plan=plan, handled=True)

    if bundle.intent == "skill_search" and bundle.skill:
        # Place filters stay on QueryState so _skill_plan can intersect cohorts;
        # they are never passed to SQL.
        skill_filters = _sql_filters(bundle)
        if city:
            skill_filters.append(FilterSlot(field="city", op="eq", value=city))
        if country:
            skill_filters.append(FilterSlot(field="country", op="eq", value=country))
        state = QueryState(
            intent="skill_search",
            skill=bundle.skill,
            refers_to_prior=bundle.refers_to_prior,
            want_count=bundle.want_count,
            confidence=max(bundle.confidence, 0.8),
            filters=skill_filters,
        )
        plan = plan_from_query_state(state, memory=memory, min_confidence=0.5)
        if plan is not None:
            return SlotPlanResult(plan=plan, handled=True)

    if bundle.intent in {"count", "list"} and not (city or country):
        state = QueryState(
            intent=bundle.intent,
            refers_to_prior=bundle.refers_to_prior,
            want_count=bundle.want_count or bundle.intent == "count",
            confidence=max(bundle.confidence, 0.8),
            filters=_sql_filters(bundle),
        )
        plan = plan_from_query_state(state, memory=memory, min_confidence=0.5)
        if plan is not None:
            return SlotPlanResult(plan=plan, handled=True)

    return SlotPlanResult(handled=False)


def _sql_filters(bundle: SlotBundle) -> list[FilterSlot]:
    """SQL-safe filters only — never city/country."""
    out: list[FilterSlot] = []
    if bundle.department:
        out.append(FilterSlot(field="department", op="eq", value=bundle.department))
    if bundle.position:
        out.append(FilterSlot(field="position", op="eq", value=bundle.position))
    return out


def _resolve_listed_person(
    ref: PersonRef, memory: SessionMemory | None
) -> tuple[str | None, str | None, str | None]:
    """Return (employee_id, display_name, clarify) for ordinal/pronoun/id refs."""
    listed = list(memory.last_listed) if memory and memory.last_listed else []

    if ref.kind == "ordinal":
        idx = ref.index
        if idx is None and ref.value:
            key = ref.value.strip().lower()
            idx = _ORDINAL_WORDS.get(key)
        if idx is None:
            return None, None, "Which person in the list did you mean?"
        if not listed:
            return None, None, (
                "Which person did you mean? Ask for their names first, then I can "
                "answer about the first, second, and so on."
            )
        if idx == -1:
            ent = listed[-1]
            return str(ent.employee_id), ent.display_name, None
        if idx < 1 or idx > len(listed):
            names = ", ".join(e.display_name for e in listed[:5])
            return None, None, (
                f"I only have {len(listed)} people in that list. I have: {names}."
            )
        ent = listed[idx - 1]
        return str(ent.employee_id), ent.display_name, None

    if ref.kind == "pronoun":
        key = (ref.value or "").strip().lower()
        # they/them after a multi-person list is cohort, not a person.
        if key in {"they", "them", "their", "theirs"}:
            if memory and len(memory.last_listed) == 1:
                ent = memory.last_listed[0]
                return str(ent.employee_id), ent.display_name, None
            if memory and len(memory.last_listed) > 1:
                return None, None, (
                    "Which person did you mean? Say the first, second, or their name."
                )
        if memory and memory.person_bindings and key in memory.person_bindings:
            eid = memory.person_bindings[key]
            name = next(
                (
                    e.display_name
                    for e in (memory.entity_memory + memory.last_listed)
                    if str(e.employee_id) == eid
                ),
                "that employee",
            )
            return eid, name, None
        if memory and len(memory.entity_memory) == 1:
            ent = memory.entity_memory[0]
            return str(ent.employee_id), ent.display_name, None
        return None, None, (
            "Which employee do you mean? Tell me their name "
            "(or pick one from the previous list)."
        )

    if ref.kind == "id" and ref.value:
        eid = str(ref.value).strip()
        known: set[str] = set()
        if memory:
            known |= {str(x) for x in (memory.last_employee_ids or [])}
            known |= {str(e.employee_id) for e in (memory.last_listed or [])}
            known |= {str(e.employee_id) for e in (memory.entity_memory or [])}
            known |= {str(v) for v in (memory.person_bindings or {}).values()}
            if memory.active_referent:
                known |= {str(x) for x in memory.active_referent.ids}
        if eid not in known:
            return None, None, (
                "I can't use that employee id from this session. "
                "Name the person or pick someone from the previous list."
            )
        label = "that employee"
        if memory:
            for ent in list(memory.last_listed or []) + list(memory.entity_memory or []):
                if str(ent.employee_id) == eid and ent.display_name:
                    label = ent.display_name
                    break
        return eid, label, None

    return None, None, None


def _person_attribute_plan(
    bundle: SlotBundle,
    *,
    memory: SessionMemory | None,
    question: str,
    city: str | None,
    country: str | None,
) -> SlotPlanResult:
    ref = bundle.person_ref
    attr = bundle.attribute
    intent = bundle.intent

    wants_person_attr = attr in {
        "dob",
        "location",
        "profile",
        "manager",
        "status",
        "education",
        "email",
        "department",
        "position",
        "title",
    } or intent in {
        "birthday",
        "location_person",
        "profile",
        "manager",
        "set_status",
    }
    if not wants_person_attr and ref.kind == "none":
        return SlotPlanResult(handled=False)

    # Named person without ordinal/pronoun.
    if ref.kind == "name" and ref.value:
        name = ref.value.strip()
        if attr == "dob" or intent == "birthday":
            state = QueryState(
                intent="birthday",
                birthday_scope="person",
                person_name=name,
                wants_age=bundle.wants_age,
                wants_wish=bundle.wants_wish,
                confidence=max(bundle.confidence, 0.8),
            )
            return SlotPlanResult(plan=plan_from_query_state(state, min_confidence=0.5), handled=True)
        if attr == "location" or intent == "location_person":
            return SlotPlanResult(plan=location_person_plan(name), handled=True)
        if attr == "manager" or intent == "manager":
            state = QueryState(
                intent="manager",
                person_name=name,
                confidence=max(bundle.confidence, 0.8),
                filters=_sql_filters(bundle),
            )
            return SlotPlanResult(
                plan=plan_from_query_state(state, memory=memory, min_confidence=0.5),
                handled=True,
            )
        if attr == "status" or intent == "set_status":
            if bundle.status_value is None:
                return SlotPlanResult(
                    clarify="Should I set the employee's status to true or false?",
                    plan=ExecutionPlan(
                        nodes=[],
                        response_strategy="template",
                        clarify_question="Should I set the employee's status to true or false?",
                    ),
                    handled=True,
                )
            state = QueryState(
                intent="set_status",
                person_name=name,
                status_value=bundle.status_value,
                confidence=max(bundle.confidence, 0.8),
            )
            return SlotPlanResult(
                plan=plan_from_query_state(state, memory=memory, min_confidence=0.5),
                handled=True,
            )
        # Default profile
        state = QueryState(
            intent="profile",
            person_name=name,
            confidence=max(bundle.confidence, 0.8),
            filters=_sql_filters(bundle),
        )
        return SlotPlanResult(
            plan=plan_from_query_state(state, memory=memory, min_confidence=0.5),
            handled=True,
        )

    if ref.kind in {"ordinal", "pronoun", "id"}:
        eid, display, clarify = _resolve_listed_person(ref, memory)
        if clarify:
            return SlotPlanResult(
                plan=ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=clarify,
                ),
                clarify=clarify,
                handled=True,
            )
        if not eid:
            return SlotPlanResult(handled=False)
        # Synthesize a question the bound-person helper understands.
        q = question or ""
        if attr == "dob" or intent == "birthday":
            q = q if "birth" in q.lower() or "old" in q.lower() or "dob" in q.lower() else (
                f"what is {display}'s date of birth"
            )
        elif attr == "location" or intent == "location_person":
            q = q if "live" in q.lower() or "where" in q.lower() else f"where does {display} live"
        elif attr == "status" or intent == "set_status":
            if bundle.status_value is None:
                return SlotPlanResult(
                    plan=ExecutionPlan(
                        nodes=[],
                        response_strategy="template",
                        clarify_question="Should I set the employee's status to true or false?",
                    ),
                    handled=True,
                )
            q = f"set {display}'s status to {'true' if bundle.status_value else 'false'}"
        elif attr == "manager" or intent == "manager":
            q = f"who is the manager of {display}"
        else:
            q = q or f"tell me about {display}"
        return SlotPlanResult(
            plan=bound_person_attribute_plan(
                q, employee_id=eid, display_name=display or "that employee"
            ),
            handled=True,
        )

    # Attribute without a resolvable person.
    if wants_person_attr and ref.kind == "none":
        if attr == "dob" or intent == "birthday":
            # Cohort birthday scopes need month/today — without that, clarify.
            return SlotPlanResult(
                plan=ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question="Whose birthday would you like to know?",
                    refusal_code=RefusalCode.AMBIGUOUS.value,
                ),
                handled=True,
            )
        if (attr == "location" or intent == "location_person") and not (city or country):
            return SlotPlanResult(
                plan=ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question="Whose location should I look up?",
                    refusal_code=RefusalCode.AMBIGUOUS.value,
                ),
                handled=True,
            )

    return SlotPlanResult(handled=False)
