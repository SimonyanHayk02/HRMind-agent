from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.places import CITIES, CITY_ALIASES, COUNTRIES, COUNTRY_ALIASES


class ColumnSpec(BaseModel):
    name: str
    kind: Literal["enum", "text", "date", "number", "id"] = "enum"
    filterable: bool = True
    facetable: bool = False
    description: str = ""
    # Canonical values known for this tenant/dataset (from DISTINCT or seed defaults).
    values: list[str] = Field(default_factory=list)
    # Map lowercased alias -> canonical value
    aliases: dict[str, str] = Field(default_factory=dict)
    # Where the value lives. "resume" fields exist only in resume documents, so
    # they are retrieved by `resume_search` and must never appear in SQL. This is
    # the single declaration of that boundary; PlanValidator enforces it.
    source: Literal["sql", "resume"] = "sql"


class SchemaCatalog(BaseModel):
    """Grounding surface for NLU + constrained SQL — not a raw DDL dump."""

    table: str = "employees"
    columns: list[ColumnSpec] = Field(default_factory=list)

    def by_name(self, name: str) -> ColumnSpec | None:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def filterable_columns(self) -> list[ColumnSpec]:
        return [c for c in self.columns if c.filterable]

    def facetable_columns(self) -> list[ColumnSpec]:
        return [c for c in self.columns if c.facetable]

    def resume_sourced(self) -> frozenset[str]:
        """Fields that exist only in resume text, so SQL cannot answer them."""
        return frozenset(c.name for c in self.columns if c.source == "resume")

    def resolve_value(self, field: str, raw: str) -> str | None:
        col = self.by_name(field)
        if col is None:
            return None
        key = raw.strip().lower()
        if key in col.aliases:
            return col.aliases[key]
        for v in col.values:
            if v.lower() == key:
                return v
        return None

    def planner_view(self) -> dict[str, Any]:
        """Compact catalog for LLM planner prompts."""
        return {
            "table": self.table,
            "filterable": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "description": c.description,
                    "values": c.values[:40],
                    "aliases": list(c.aliases.keys())[:20],
                }
                for c in self.filterable_columns()
            ],
            "facetable": [c.name for c in self.facetable_columns()],
        }


def resume_sourced_fields() -> frozenset[str]:
    """The resume-only fields, for callers without a live catalog.

    Declared once here so the plan validator, the prompts and the retrieval
    attributes cannot drift apart about who owns a field.
    """
    return default_employee_catalog().resume_sourced()


def default_employee_catalog() -> SchemaCatalog:
    """Seed catalog aligned with HRMind employees table + known seed values."""
    return SchemaCatalog(
        table="employees",
        columns=[
            ColumnSpec(
                name="department",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Org department",
                values=["Engineering", "People", "Sales", "Finance", "Product", "Operations"],
            ),
            # Location is written in the resume and nowhere else. It stays in the
            # catalog so NLU still recognises "Berlin", but `source="resume"`
            # routes it to retrieval instead of SQL.
            ColumnSpec(
                name="country",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Work country (resume text only; retrieved, never queried)",
                values=list(COUNTRIES),
                aliases=dict(COUNTRY_ALIASES),
                source="resume",
            ),
            ColumnSpec(
                name="city",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Work city (resume text only; retrieved, never queried)",
                values=list(CITIES),
                aliases=dict(CITY_ALIASES),
                source="resume",
            ),
            ColumnSpec(
                name="position",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Job title",
                values=[
                    "Software Engineer",
                    "Senior Engineer",
                    "Staff Engineer",
                    "Engineering Manager",
                    "HR Specialist",
                    "Recruiter",
                    "HR Manager",
                    "People Ops",
                    "Account Executive",
                    "Sales Manager",
                    "SDR",
                    "Accountant",
                    "Financial Analyst",
                    "Finance Manager",
                    "Product Manager",
                    "Product Designer",
                    "Head of Product",
                    "Ops Specialist",
                    "Ops Manager",
                    "Coordinator",
                ],
                aliases={
                    "swe": "Software Engineer",
                    "software engineer": "Software Engineer",
                    "software engineers": "Software Engineer",
                    "eng manager": "Engineering Manager",
                    "engineering manager": "Engineering Manager",
                },
            ),
            ColumnSpec(
                name="education",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Highest education / background",
                values=[
                    "Bootcamp",
                    "MBA / MSc",
                    "MSc Data Science",
                    "BSc",
                    "BSc Computer Science",
                    "BA Business",
                    "PhD",
                    "Self-taught",
                ],
                aliases={
                    "mba": "MBA / MSc",
                    "msc": "MBA / MSc",
                    "masters": "MBA / MSc",
                    "master": "MBA / MSc",
                    "master's": "MBA / MSc",
                    "bachelor": "BSc",
                    "bachelors": "BSc",
                    "bachelor's": "BSc",
                    "bs": "BSc",
                    "phd": "PhD",
                    "doctorate": "PhD",
                    "self taught": "Self-taught",
                    "selftaught": "Self-taught",
                },
            ),
            ColumnSpec(
                name="employment_status",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Employment status",
                values=["active", "leave", "terminated"],
                aliases={
                    "employed": "active",
                    "working": "active",
                    "on leave": "leave",
                    "fired": "terminated",
                    "left": "terminated",
                },
            ),
            ColumnSpec(
                name="status",
                kind="enum",
                filterable=True,
                facetable=False,
                description=(
                    "Agent boolean flag on employees (true/false). "
                    "Not employment_status and not resume ingest status."
                ),
                values=["true", "false"],
                aliases={"yes": "true", "no": "false", "1": "true", "0": "false"},
            ),
            ColumnSpec(
                name="hire_date",
                kind="date",
                filterable=True,
                facetable=False,
                description="Hire date (supports after/since year filters)",
            ),
        ],
    )
