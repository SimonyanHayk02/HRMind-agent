"""Build person-attribute plans when the subject is already bound to an id."""
from __future__ import annotations

from app.application.planning.heuristic_planner import (
    _LOCATION_TOPIC_RE,
    _PERSON_LOCATION_RE,
    _employee_by_id_plan,
    wants_person_lookup,
)
from app.application.planning.location_plans import location_person_plan
from app.application.planning.plan_schema import ExecutionPlan
from app.application.planning.unsupported import is_unsupported_topic
from app.application.response.refusal import RefusalCode, out_of_scope_plan
from app.application.understanding.birthday import extract_birthday
from app.application.understanding.certifications import extract_certification
from app.application.understanding.experience import extract_experience
from app.application.understanding.languages import extract_language
from app.application.understanding.person_skills import extract_person_skills
from app.application.understanding.plan_from_query_state import (
    _birthday_plan,
    _certifications_plan,
    _experience_person_plan,
    _languages_plan,
    _set_status_plan,
    _skills_person_plan,
)
from app.application.understanding.status_change import extract_status_change
from app.domain.query_state import QueryState


def bound_person_attribute_plan(
    question: str,
    *,
    employee_id: str,
    display_name: str,
) -> ExecutionPlan:
    """Plan for a birthday / location / status / profile ask about a known id."""
    q = (question or "").strip()
    name = display_name or "that employee"
    ids = [str(employee_id)]

    # Never answer lifestyle / non-schema asks with a profile dump.
    if is_unsupported_topic(q):
        return out_of_scope_plan()

    birthday = extract_birthday(q)
    if birthday.matched and birthday.scope == "person":
        return _birthday_plan(
            QueryState(
                intent="birthday",
                birthday_scope="person",
                person_name=name,
                person_employee_ids=ids,
                wants_age=birthday.wants_age,
                wants_wish=birthday.wants_wish,
                confidence=0.95,
            )
        )

    language = extract_language(q)
    if language.matched and language.scope == "person":
        return _languages_plan(
            QueryState(
                intent="languages",
                language=language.language,
                person_name=name,
                person_employee_ids=ids,
                confidence=0.95,
            )
        )

    certification = extract_certification(q)
    if certification.matched and certification.scope == "person":
        return _certifications_plan(
            QueryState(
                intent="certifications",
                certification=certification.certification,
                person_name=name,
                person_employee_ids=ids,
                confidence=0.95,
            )
        )

    skills = extract_person_skills(q)
    if skills.matched:
        return _skills_person_plan(
            person_name=name,
            employee_ids=ids,
            skill=skills.skill,
        )

    experience_req = extract_experience(q)
    if experience_req.matched:
        return _experience_person_plan(person_name=name, employee_ids=ids)

    status = extract_status_change(q)
    if status.matched:
        return _set_status_plan(
            QueryState(
                intent="set_status",
                person_name=name,
                person_employee_ids=ids,
                status_employee_id=ids[0],
                status_value=status.status_value,
                confidence=0.95,
            ),
            filters={},
        )

    asks_where = bool(_PERSON_LOCATION_RE.search(q)) or bool(
        _LOCATION_TOPIC_RE.search(q)
    )
    if asks_where and not extract_experience(q).matched:
        return location_person_plan(name, employee_ids=ids)

    if wants_person_lookup(q):
        return _employee_by_id_plan(ids[0])

    return ExecutionPlan(
        nodes=[],
        response_strategy="template",
        clarify_question=(
            f"What would you like to know about {name}? "
            "For example: profile, manager, location, birthday, languages, "
            "skills, experience, or status."
        ),
        refusal_code=RefusalCode.AMBIGUOUS.value,
    )
