"""Rich ToolMeta schemas / anti-pattern docs for LLM tool selection."""
from __future__ import annotations

from typing import Any

SQL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["constrained", "nl2sql"],
            "description": "Prefer constrained. Use nl2sql only when filters cannot express the ask.",
        },
        "count_only": {"type": "boolean"},
        "count_distinct": {
            "type": "string",
            "description": "Facet dimension column, e.g. department|position|education|employment_status",
        },
        "filters": {
            "type": "object",
            "description": (
                "SQL filters on employees columns only: department, position, education, "
                "employment_status, hire_date_*, employee_ids, id. "
                "NEVER city/country/salary-as-filter for location facts."
            ),
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to return for list/name answers",
        },
        "question": {"type": "string", "description": "Only for nl2sql mode"},
    },
}

RESUME_SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "purpose": {
            "type": "string",
            "enum": [
                "birthday_person",
                "birthday_cohort",
                "location_person",
                "location_cohort",
                "location_facet",
                "languages_person",
                "languages_cohort",
                "certifications_person",
                "certifications_cohort",
                "status_resolve",
                "skill",
            ],
            "description": "Required for attribute asks. Omit/skill for general skill search.",
        },
        "name": {"type": "string"},
        "employee_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Restrict to a prior cohort when refining 'of them'",
        },
        "scope": {
            "type": "string",
            "enum": ["today", "upcoming", "month", "closest", "person"],
        },
        "month": {"type": "integer", "minimum": 1, "maximum": 12},
        "city": {"type": "string"},
        "country": {"type": "string"},
        "language": {"type": "string"},
        "certification": {"type": "string"},
        "skill": {
            "type": "string",
            "description": (
                "Canonical skill for purpose=skill (e.g. Kubernetes). "
                "Used to lexically confirm RAG hits after embedding recall."
            ),
        },
        "facet": {"type": "string", "enum": ["city", "country"]},
        "wants_age": {"type": "boolean"},
        "wants_wish": {"type": "boolean"},
    },
    "required": ["question"],
}

EMPLOYEE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "profile",
                "manager",
                "reports",
                "department",
                "by_email",
                "by_id",
                "by_name",
                "set_status",
            ],
        },
        "employee_id": {"type": "string"},
        "email": {"type": "string"},
        "department": {"type": "string"},
        "name": {"type": "string"},
        "status": {"type": "boolean"},
        "employee_ids": {"type": "array", "items": {"type": "string"}},
        "question": {"type": "string"},
    },
    "required": ["action"],
}

CLARIFY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "Clarifying question for the user"},
        "candidates": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question"],
}

GREETING_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
    },
}

SQL_DESCRIPTION = (
    "Query structured employee table fields: department, position, education, "
    "employment_status, hire_date, names, email, salary (role-ACL). "
    "Use for headcount, department/position lists, tenure, salary (when allowed). "
    "DO NOT use for: date of birth, age, city/country/location, languages spoken, "
    "certifications, or free-text skills — those live in resumes (resume_search)."
)

RESUME_SEARCH_DESCRIPTION = (
    "Semantic search over resume text. Use for skills (Python/AWS/…), location "
    "(city/country), birthday/DOB/age, spoken languages, certifications, and "
    "resolving people before status writes when needed. "
    "Set purpose appropriately. For skill∩prior-cohort counts, pass employee_ids "
    "from the previous set. DO NOT use for pure SQL headcount by department alone."
)

EMPLOYEE_DESCRIPTION = (
    "Directory/profile tool: by_name/by_id profile, manager, direct reports, "
    "department roster, set_status (agent flag). "
    "DO NOT use for birthday/location/skills (resume_search) or org-wide counts (sql). "
    "set_status is a write — require high confidence; prefer confirm for bulk/location writes."
)

CLARIFY_DESCRIPTION = (
    "Ask the user a clarifying question when the ask is ambiguous "
    "(which person, which list, missing status true/false). "
    "Prefer clarify over guessing when confidence is low."
)

GREETING_DESCRIPTION = (
    "Social hello/bye/thanks/chitchat only. Never use for HR data questions."
)
