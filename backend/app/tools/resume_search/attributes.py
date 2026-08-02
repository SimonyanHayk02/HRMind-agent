"""Registry of facts that live only inside resume documents.

The retrieval pipeline is attribute-agnostic: it resolves a person, fetches the
sections declared here, and hands the chunk text to this attribute's extractor.
Adding another resume-only fact (languages, certifications) means one entry here
plus golden rows — no change to the tool, the store or the planner.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from app.tools.resume_search import (
    birthday,
    certifications,
    experience,
    languages,
    location,
    person_skills,
)

AttributeMode = Literal["person", "cohort", "facet"]


@dataclass(frozen=True)
class ResumeAttribute:
    """How to retrieve, extract and phrase one resume-only fact."""

    name: str
    # Resume sections that carry the fact; drives the stage 2 retrieval filter.
    sections: tuple[str, ...]
    # Content fallbacks for resumes whose section header was not recognised.
    content_hints: tuple[str, ...]
    # Extractor used both at query time and for ingest-time coverage checks.
    extract: Callable[[str], Any | None]
    # Chunks -> one fact per employee.
    facts_from_hits: Callable[[list[dict[str, Any]]], list[Any]]
    answer_person: Callable[..., str]
    # A cohort attribute either answers in prose (birthdays) or resolves a set of
    # employees for a later node (location). Absent means the latter.
    answer_cohort: Callable[..., str] | None = None
    # Facts matching the asked-for cohort, used for citations and — when the
    # attribute publishes ids — for the employee set handed downstream.
    select_cohort: Callable[..., list[Any]] | None = None
    # True when the resolved cohort *is* the result, so nothing unrelated may be
    # attached: an id that did not match would become a wrong SQL row downstream.
    publishes_cohort_ids: bool = False
    # Fields this attribute can be aggregated by ("how many different cities").
    facets: tuple[str, ...] = field(default_factory=tuple)


BIRTH_DATE = ResumeAttribute(
    name="birth_date",
    sections=("Personal",),
    content_hints=("date of birth",),
    extract=birthday.parse_birth_date,
    facts_from_hits=birthday.facts_from_hits,
    answer_person=birthday.build_person_answer,
    answer_cohort=birthday.build_cohort_answer,
    select_cohort=birthday.select_cohort,
)

LOCATION = ResumeAttribute(
    name="location",
    sections=("Location",),
    content_hints=("based in", "location"),
    extract=location.parse_location,
    facts_from_hits=location.facts_from_hits,
    answer_person=location.build_person_answer,
    # No cohort prose: a place question resolves employees, then the employees
    # table supplies their names. The answer belongs to whoever owns that data.
    answer_cohort=None,
    select_cohort=location.select_cohort,
    publishes_cohort_ids=True,
    facets=("city", "country"),
)

LANGUAGES = ResumeAttribute(
    name="languages",
    sections=("Languages",),
    content_hints=("languages", "speaks"),
    extract=languages.parse_languages,
    facts_from_hits=languages.facts_from_hits,
    answer_person=languages.build_person_answer,
    answer_cohort=languages.build_cohort_answer,
    select_cohort=languages.select_cohort,
    publishes_cohort_ids=True,
)

CERTIFICATIONS = ResumeAttribute(
    name="certifications",
    sections=("Certificates",),
    content_hints=("certificate", "certification", "certified"),
    extract=certifications.parse_certifications,
    facts_from_hits=certifications.facts_from_hits,
    answer_person=certifications.build_person_answer,
    answer_cohort=certifications.build_cohort_answer,
    select_cohort=certifications.select_cohort,
    publishes_cohort_ids=True,
)

SKILLS = ResumeAttribute(
    name="skills",
    sections=("Skills",),
    # No content_hints: hints OR with section and pull Experience chunks that
    # mention "Python". Section filter alone is enough for a bound person.
    content_hints=(),
    extract=person_skills.parse_skills_list,
    facts_from_hits=person_skills.facts_from_hits,
    answer_person=person_skills.build_person_answer,
    answer_cohort=None,
    select_cohort=person_skills.select_cohort,
)

EXPERIENCE = ResumeAttribute(
    name="experience",
    sections=("Experience",),
    content_hints=(),
    extract=experience.parse_experience,
    facts_from_hits=experience.facts_from_hits,
    answer_person=experience.build_person_answer,
)

ATTRIBUTES: dict[str, ResumeAttribute] = {
    BIRTH_DATE.name: BIRTH_DATE,
    LOCATION.name: LOCATION,
    LANGUAGES.name: LANGUAGES,
    CERTIFICATIONS.name: CERTIFICATIONS,
    SKILLS.name: SKILLS,
    EXPERIENCE.name: EXPERIENCE,
}

# Plan node purposes map onto (attribute, mode) so the tool needs no per-fact branch.
_PURPOSES: dict[str, tuple[str, AttributeMode]] = {
    "birthday_person": (BIRTH_DATE.name, "person"),
    "birthday_cohort": (BIRTH_DATE.name, "cohort"),
    "location_person": (LOCATION.name, "person"),
    "location_cohort": (LOCATION.name, "cohort"),
    "location_facet": (LOCATION.name, "facet"),
    "languages_person": (LANGUAGES.name, "person"),
    "languages_cohort": (LANGUAGES.name, "cohort"),
    "certifications_person": (CERTIFICATIONS.name, "person"),
    "certifications_cohort": (CERTIFICATIONS.name, "cohort"),
    "skills_person": (SKILLS.name, "person"),
    "experience_person": (EXPERIENCE.name, "person"),
}


def get_attribute(name: str) -> ResumeAttribute | None:
    return ATTRIBUTES.get(name)


def resolve_purpose(purpose: str) -> tuple[ResumeAttribute, AttributeMode] | None:
    """Map a plan node purpose to the attribute and retrieval mode it needs."""
    mapped = _PURPOSES.get(purpose or "")
    if mapped is None:
        return None
    attribute, mode = mapped
    found = ATTRIBUTES.get(attribute)
    return (found, mode) if found else None


def facet_dimensions() -> frozenset[str]:
    """Every field a resume attribute can be aggregated by.

    These are resume-sourced, so no SQL node may filter, select or count them —
    `PlanValidator` reads this to enforce it.
    """
    return frozenset(f for attr in ATTRIBUTES.values() for f in attr.facets)
