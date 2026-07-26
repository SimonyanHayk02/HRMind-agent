from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
            ColumnSpec(
                name="country",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Work country",
                values=["Germany", "USA", "UK", "UAE", "France"],
                aliases={
                    "us": "USA",
                    "u.s.": "USA",
                    "u.s.a.": "USA",
                    "united states": "USA",
                    "united states of america": "USA",
                    "america": "USA",
                    "united kingdom": "UK",
                    "britain": "UK",
                    "england": "UK",
                    "united arab emirates": "UAE",
                    "dubai country": "UAE",
                },
            ),
            ColumnSpec(
                name="city",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Work city",
                values=["Berlin", "New York", "London", "Dubai", "Paris"],
                aliases={"ny": "New York", "new york city": "New York", "nyc": "New York"},
            ),
            ColumnSpec(
                name="position",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Job title",
                values=[],
            ),
            ColumnSpec(
                name="education",
                kind="enum",
                filterable=True,
                facetable=True,
                description="Highest education / background",
                values=["Bootcamp", "MBA / MSc", "BSc", "PhD", "Self-taught"],
                aliases={
                    "mba": "MBA / MSc",
                    "msc": "MBA / MSc",
                    "masters": "MBA / MSc",
                    "master": "MBA / MSc",
                    "bachelor": "BSc",
                    "bachelors": "BSc",
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
                name="hire_date",
                kind="date",
                filterable=True,
                facetable=False,
                description="Hire date (supports after/since year filters)",
            ),
        ],
    )
