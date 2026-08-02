from app.application.planning.heuristic_planner import try_heuristic_plan
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.person_existence import extract_person_existence_name
from app.application.understanding.plan_from_query_state import plan_from_query_state
from app.domain.enums import Role
from app.domain.schema_catalog import default_employee_catalog
from app.domain.session import ActiveReferent, EntityRef, SessionMemory


def test_extract_of_them_from_usa() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("how much of them are from USA?", catalog=catalog)
    assert state.intent == "count"
    assert state.refers_to_prior is True
    # NLU still recognises the country; the plan routes it to retrieval because
    # the value lives in resume text.
    assert any(f.field == "country" and f.value == "USA" for f in state.filters)
    plan = plan_from_query_state(state)
    assert plan is not None
    assert plan.nodes[0].params["purpose"] == "location_cohort"
    assert plan.nodes[0].params["country"] == "USA"
    assert plan.nodes[-1].name == "sql"
    assert "country" not in plan.nodes[-1].params["filters"]


def test_extract_education_bootcamp() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state(
        "how many employees have Bootcamp education?", catalog=catalog
    )
    assert state.intent == "count"
    assert any(f.field == "education" and f.value == "Bootcamp" for f in state.filters)
    plan = plan_from_query_state(state)
    assert plan is not None
    assert plan.nodes[-1].params["filters"]["education"] == "Bootcamp"


def test_extract_masters_includes_msc_data_science() -> None:
    """'masters' must not collapse to only the old MBA / MSc enum label."""
    catalog = default_employee_catalog()
    state = extract_query_state(
        "give all employees who has masters degree", catalog=catalog
    )
    assert state.intent == "list"
    edu = next(f for f in state.filters if f.field == "education")
    assert edu.op == "in"
    assert "MBA / MSc" in edu.value
    assert "MSc Data Science" in edu.value
    plan = plan_from_query_state(state)
    assert plan is not None
    assert set(plan.nodes[-1].params["filters"]["education"]) >= {
        "MBA / MSc",
        "MSc Data Science",
    }


def test_extract_exact_msc_data_science() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state(
        "give all employees education is MSc Data Science", catalog=catalog
    )
    edu = next(f for f in state.filters if f.field == "education")
    assert edu.op == "eq"
    assert edu.value == "MSc Data Science"
    plan = plan_from_query_state(state)
    assert plan is not None
    assert plan.nodes[-1].params["filters"]["education"] == "MSc Data Science"


def test_extract_united_states_alias() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state(
        "how many of them are from the United States?", catalog=catalog
    )
    assert any(f.field == "country" and f.value == "USA" for f in state.filters)


def test_list_software_developers() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state(
        "can you list me the software developers?", catalog=catalog
    )
    assert state.intent == "list"
    assert any(f.field == "position" for f in state.filters)
    plan = plan_from_query_state(state)
    assert plan is not None
    filters = plan.nodes[0].params["filters"]
    assert filters.get("position") == "Software Engineer" or (
        isinstance(filters.get("position"), list)
        and "Software Engineer" in filters["position"]
    )


def test_query_builder_invalid_uuids_force_empty() -> None:
    """Invalid employee_ids must not fall through to an unscoped org dump."""
    from app.adapters.sql.query_builder import QueryBuilder

    sql, params = QueryBuilder().build(
        columns=["id", "first_name"],
        filters={
            "employee_ids": ["not-a-uuid", "also-bad"],
            "position": "Software Engineer",
        },
    )
    assert "eid_" not in sql
    assert "1=0" in sql
    assert "Software Engineer" in params.values() or any(
        v == "Software Engineer" for v in params.values()
    )

    catalog = default_employee_catalog()
    memory = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        constraint_memory=[],
    )
    head = extract_query_state("How many employees do we have?", catalog=catalog)
    assert head.intent == "count"
    assert head.confidence >= 0.7
    plan = plan_from_query_state(head, memory=memory)
    assert plan is not None
    assert plan.nodes[-1].params.get("count_only") is True


def test_do_we_have_sofia_with_prior_person_memory() -> None:
    """Existence asks must not fall through to LLM when Alice is still bound."""
    assert extract_person_existence_name("do we have Sofia ?") == "Sofia"
    assert extract_person_existence_name("do we have sofia?") == "sofia"
    assert extract_person_existence_name("do we have Python?") is None

    catalog = default_employee_catalog()
    memory = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_employee_ids=["91cb25fe-2e51-5e7e-9466-155d30ffaf63"],
        entity_memory=[
            EntityRef(
                employee_id="91cb25fe-2e51-5e7e-9466-155d30ffaf63",
                display_name="Alice Nguyen",
                aliases=["Alice"],
            )
        ],
        person_bindings={
            "she": "91cb25fe-2e51-5e7e-9466-155d30ffaf63",
            "her": "91cb25fe-2e51-5e7e-9466-155d30ffaf63",
        },
        active_referent=ActiveReferent(
            ids=["91cb25fe-2e51-5e7e-9466-155d30ffaf63"],
            label="Alice Nguyen",
        ),
    )
    state = extract_query_state("do we have Sofia ?", catalog=catalog, memory=memory)
    assert state.intent == "profile"
    assert state.person_name == "Sofia"
    plan = plan_from_query_state(state, memory=memory)
    assert plan is not None
    assert plan.nodes[0].name == "employee"
    assert plan.nodes[0].params.get("name") == "Sofia"

    heuristic = try_heuristic_plan("do we have Sofia ?", memory=memory)
    assert heuristic is not None
    assert heuristic.nodes[0].params.get("name") == "Sofia"


def test_plan_node_promotes_nested_input_bindings_params() -> None:
    """LLM sometimes puts tool args under input_bindings.params (a dict)."""
    node = PlanNode.model_validate(
        {
            "id": "e1",
            "kind": "tool",
            "name": "employee",
            "input_bindings": {"params": {"name": "Sofia", "action": "by_name"}},
            "params": {},
        }
    )
    assert node.params.get("name") == "Sofia"
    assert node.input_bindings == {}
    plan = ExecutionPlan(nodes=[node], response_strategy="template")
    assert plan.nodes[0].params["name"] == "Sofia"
