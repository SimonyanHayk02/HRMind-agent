"""Plans for questions about work location.

Location is written in resume documents, so every one of these plans starts with
retrieval. When the answer also needs employees-table data — names, counts — the
resolved ids are handed to `sql`, which never sees the place itself. One question,
two owners, no overlap.
"""
from __future__ import annotations

from typing import Any

from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.domain.places import canon_city, canon_country
from app.domain.schema_catalog import resume_sourced_fields

NAME_COLUMNS = ["id", "first_name", "last_name", "department", "position"]

LOCATION_FIELDS = frozenset({"city", "country"})


def split_location_filters(filters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate the resume-sourced filters from the ones SQL may still apply."""
    resume_sourced = resume_sourced_fields()
    place = {k: v for k, v in filters.items() if k in resume_sourced and v is not None}
    sql_side = {k: v for k, v in filters.items() if k not in resume_sourced}
    return place, sql_side


def has_location_filter(filters: dict[str, Any] | None) -> bool:
    return bool(filters) and any(
        filters.get(field) is not None for field in LOCATION_FIELDS
    )


def _place_label(city: str | None, country: str | None) -> str:
    return str(city or country or "location")


def location_person_plan(
    name: str,
    *,
    employee_ids: list[str] | None = None,
    reference: str = "name",
) -> ExecutionPlan:
    """"Where does X work?" — identity from the corpus, place from their resume."""
    params: dict[str, Any] = {
        "purpose": "location_person",
        "question": name,
        "name": name,
        "reference": reference,
    }
    if employee_ids:
        params["employee_ids"] = list(employee_ids)
    return ExecutionPlan(
        nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params=params)],
        response_strategy="template",
    )


def location_cohort_nodes(
    *,
    city: str | None = None,
    country: str | None = None,
    intersect_with: list[str] | None = None,
) -> tuple[list[PlanNode], str]:
    """Retrieval nodes resolving who is in a place, and the id-producing node id."""
    city = canon_city(city) if city else None
    country = canon_country(country) if country else None
    nodes = [
        PlanNode(
            id="loc",
            kind="tool",
            name="resume_search",
            params={
                "purpose": "location_cohort",
                "question": _place_label(city, country),
                "city": city,
                "country": country,
            },
        ),
        PlanNode(
            id="locids",
            kind="operator",
            name="extract_employee_ids",
            depends_on=["loc"],
            input_bindings={"data": "nodes.loc"},
        ),
    ]
    id_source = "locids"
    if intersect_with:
        nodes.append(
            PlanNode(
                id="locix",
                kind="operator",
                name="intersect_ids",
                depends_on=["locids"],
                input_bindings={"data": "nodes.locids"},
                params={"other": list(intersect_with)},
            )
        )
        id_source = "locix"
    return nodes, id_source


def location_cohort_plan(
    *,
    city: str | None = None,
    country: str | None = None,
    count_only: bool = False,
    extra_filters: dict[str, Any] | None = None,
    intersect_with: list[str] | None = None,
    columns: list[str] | None = None,
) -> ExecutionPlan:
    """"Employees in Berlin", optionally counted, narrowed or intersected.

    ``extra_filters`` are employees-table filters such as department: the composite
    question keeps each half with the tool that owns it.
    """
    nodes, id_source = location_cohort_nodes(
        city=city, country=country, intersect_with=intersect_with
    )
    filters = {
        k: v
        for k, v in (extra_filters or {}).items()
        if k not in LOCATION_FIELDS and v is not None
    }
    nodes.append(
        PlanNode(
            id="sql1",
            kind="tool",
            name="sql",
            depends_on=[id_source],
            params={
                "mode": "constrained",
                "count_only": count_only,
                "filters": filters,
                **({} if count_only else {"columns": columns or NAME_COLUMNS}),
            },
            input_bindings={"employee_ids": f"nodes.{id_source}"},
        )
    )
    return ExecutionPlan(
        nodes=nodes,
        response_strategy="template",
        # A counted cohort publishes the retrieved ids, since the sql node returns
        # only a number: "them" must still mean the people in that place.
        active_cohort_node=id_source if count_only else "sql1",
    )


def location_facet_plan(dimension: str, *, count: bool = False) -> ExecutionPlan:
    """"How many different cities" / "which cities" — aggregate, no cohort."""
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="facet",
                kind="tool",
                name="resume_search",
                params={
                    "purpose": "location_facet",
                    "facet": dimension,
                    "question": f"distinct {dimension}",
                    # The node returns both the values and their count; this says
                    # which one the question asked for, as count_distinct does
                    # for the SQL facets.
                    "facet_count": count,
                },
            )
        ],
        response_strategy="template",
    )


def profile_location_node(employee_node_id: str) -> PlanNode:
    """Location for whoever the employee tool just resolved.

    Bound to that node's id rather than resolving the name a second time: the
    corpus contains namesakes, and two independent resolutions could disagree and
    print one person's city under another's name.
    """
    return PlanNode(
        id="loc",
        kind="tool",
        name="resume_search",
        params={"purpose": "location_person", "question": "location", "reference": "id"},
        depends_on=[employee_node_id],
        input_bindings={"employee_ids": f"nodes.{employee_node_id}"},
    )
