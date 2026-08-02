from __future__ import annotations

from uuid import UUID

from app.application.planning.bound_person import bound_person_attribute_plan
from app.application.planning.plan_compiler import PlanCompiler
from app.application.understanding.birthday import extract_birthday
from app.application.understanding.list_referents import resolve_list_referent
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.session import EntityRef, SessionMemory
from app.domain.tools.registry import ToolRegistry

E1 = UUID("00000000-0000-0000-0000-000000000001")
E2 = UUID("00000000-0000-0000-0000-000000000002")


def _memory(*names: tuple[UUID, str]) -> SessionMemory:
    return SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_listed=[
            EntityRef(employee_id=eid, display_name=name) for eid, name in names
        ],
        last_employee_ids=[str(eid) for eid, _ in names],
    )


def test_first_person_binds_index_zero() -> None:
    mem = _memory((E1, "Kara Petrov"), (E2, "Maya Khan"))
    ref = resolve_list_referent("give me the first persons date of birth", mem)
    assert ref.kind == "resolved"
    assert ref.employee_id == str(E1)
    assert ref.display_name == "Kara Petrov"


def test_second_and_last_and_1st() -> None:
    mem = _memory((E1, "Kara Petrov"), (E2, "Maya Khan"))
    assert resolve_list_referent("the second one", mem).employee_id == str(E2)
    assert resolve_list_referent("the last person", mem).employee_id == str(E2)
    assert resolve_list_referent("the 1st person", mem).employee_id == str(E1)


def test_former_one_and_latter_one() -> None:
    mem = _memory((E1, "Kara Petrov"), (E2, "Maya Khan"))
    assert resolve_list_referent("the former one", mem).employee_id == str(E1)
    assert resolve_list_referent("the latter person", mem).employee_id == str(E2)


def test_bare_former_is_not_a_list_referent() -> None:
    mem = _memory((E1, "Kara Petrov"), (E2, "Maya Khan"))
    ref = resolve_list_referent("show former employees", mem)
    assert ref.kind == "none"


def test_oob_and_empty_list_clarify() -> None:
    mem = _memory((E1, "Kara Petrov"), (E2, "Maya Khan"))
    oob = resolve_list_referent("the third person", mem)
    assert oob.kind == "clarify"
    assert "Kara Petrov" in (oob.clarify_question or "")

    empty = resolve_list_referent(
        "the first person's date of birth",
        SessionMemory(
            session_id="s", tenant_id="t", user_id="u", role=Role.RECRUITER
        ),
    )
    assert empty.kind == "clarify"
    assert "names first" in (empty.clarify_question or "").lower()


def test_that_one_needs_singleton_list() -> None:
    two = _memory((E1, "Kara Petrov"), (E2, "Maya Khan"))
    assert resolve_list_referent("that one", two).kind == "clarify"
    one = _memory((E1, "Kara Petrov"))
    assert resolve_list_referent("that person", one).employee_id == str(E1)


def test_plural_ones_clarify() -> None:
    mem = _memory((E1, "Kara Petrov"), (E2, "Maya Khan"))
    ref = resolve_list_referent("the first ones date of birth", mem)
    assert ref.kind == "clarify"


def test_birthday_extract_never_keeps_ordinal_or_pronoun_as_name() -> None:
    for q in (
        "give me the first persons date of birth",
        "what is her date of birth",
        "when was she born?",
    ):
        req = extract_birthday(q)
        assert req.matched
        assert req.person_name is None, q


def test_plan_compiler_binds_ordinal_birthday() -> None:
    mem = _memory((E1, "Kara Petrov"), (E2, "Maya Khan"))
    compiler = PlanCompiler(llm=None, tools=ToolRegistry())
    import asyncio

    async def _run() -> None:
        plan, mode, _meta = await compiler.compile(
            "give me the first persons date of birth",
            auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
            memory=mem,
        )
        assert mode == "heuristic_list_referent"
        assert plan.nodes[0].name == "resume_search"
        assert plan.nodes[0].params["purpose"] == "birthday_person"
        assert plan.nodes[0].params["employee_ids"] == [str(E1)]
        assert plan.nodes[0].params["name"] == "Kara Petrov"

    asyncio.run(_run())


def test_plan_compiler_clarifies_ordinal_without_list() -> None:
    compiler = PlanCompiler(llm=None, tools=ToolRegistry())
    import asyncio

    async def _run() -> None:
        plan, mode, _meta = await compiler.compile(
            "the first person's date of birth",
            auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
            memory=SessionMemory(
                session_id="s", tenant_id="t", user_id="u", role=Role.RECRUITER
            ),
        )
        assert mode == "heuristic_list_referent"
        assert plan.nodes == []
        assert plan.clarify_question

    asyncio.run(_run())


def test_bound_person_birthday_plan_includes_ids() -> None:
    plan = bound_person_attribute_plan(
        "date of birth",
        employee_id=str(E1),
        display_name="Kara Petrov",
    )
    assert plan.nodes[0].params["employee_ids"] == [str(E1)]
