"""Location is a resume-only fact, retrieved the same way birth dates are.

These tests run the real pipeline over the shared in-memory corpus: ingest parses
the place out of the document, and every question about it is answered from the
chunks rather than from a column.
"""
from __future__ import annotations

import pytest

from app.adapters.vectorstore.memory_vector_store import MemoryVectorStore
from app.tools.resume_search.attributes import LOCATION, facet_dimensions, resolve_purpose
from app.tools.resume_search.location import (
    LocationFact,
    facts_from_hits,
    parse_location,
)
from app.tools.resume_search.tool import ResumeSearchTool
from tests.unit.tools.conftest import AUTH, PEOPLE, ask as _ask

# --- parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "city", "country"),
    [
        ("Berlin, Germany", "Berlin", "Germany"),
        ("Location\nNew York, USA", "New York", "USA"),
        ("Experienced Recruiter in People based in London, UK.", "London", "UK"),
        ("Lives in Dubai", "Dubai", None),
        ("Based in the UAE", None, "UAE"),
        # Aliases canonicalise, so the corpus and the question can spell it
        # differently without the answer changing.
        ("Location\nNYC, United States", "New York", "USA"),
    ],
)
def test_parse_location_reads_the_written_place(
    text: str, city: str | None, country: str | None
) -> None:
    place = parse_location(text)
    assert place is not None
    assert (place.city, place.country) == (city, country)


@pytest.mark.parametrize("text", ["", "Skills\nPython, SQL", "Location\nRemote"])
def test_parse_location_refuses_to_guess(text: str) -> None:
    assert parse_location(text) is None


def test_body_wins_over_the_enriched_header() -> None:
    """The document is authoritative; the header is only context for retrieval."""
    chunk = "Eva Kim — Senior Engineer, Engineering (Dubai, UAE)\nLocation\nBerlin, Germany"
    place = parse_location(chunk)
    assert place is not None
    assert place.text == "Berlin, Germany"


# --- ingest -----------------------------------------------------------------


async def test_ingest_puts_the_place_in_the_text_and_not_in_metadata(
    corpus: MemoryVectorStore,
) -> None:
    """The place is embedded text, never a metadata copy: one source of truth."""
    chunks = await corpus.fetch_section_chunks(("Location",))
    assert {c["employee_id"] for c in chunks} == {f"e{i}" for i in range(len(PEOPLE))}
    for chunk in chunks:
        assert "city" not in chunk["metadata"]
        assert "country" not in chunk["metadata"]
    eva = next(c for c in chunks if c["employee_id"] == "e0")
    assert "Dubai, UAE" in eva["content"]


async def test_facts_are_parsed_once_per_employee(corpus: MemoryVectorStore) -> None:
    chunks = await corpus.fetch_section_chunks(
        LOCATION.sections, content_hints=LOCATION.content_hints
    )
    facts = facts_from_hits(chunks)
    # Liam Novak's resume says "Remote, Elsewhere", which is not a place we know.
    assert {f.employee_id for f in facts} == {"e0", "e1", "e2", "e3", "e4"}
    assert {f.name for f in facts if f.city == "London"} == {"Ivy Chen"}


# --- person -----------------------------------------------------------------


async def test_person_location_is_answered_from_the_resume(
    tool: ResumeSearchTool, corpus: MemoryVectorStore
) -> None:
    data = await _ask(tool, purpose="location_person", name="Eva Kim", question="Eva Kim")
    assert "Dubai, UAE" in data["answer"]
    assert data["resolved_via"] == "name"
    chunk_ids = {c["id"] for c in await corpus.fetch_section_chunks(("Location",))}
    assert data["facts"][0]["extracted_from_chunk_id"] in chunk_ids


async def test_person_location_lists_every_namesake(tool: ResumeSearchTool) -> None:
    data = await _ask(tool, purpose="location_person", name="Ivy Chen", question="Ivy Chen")
    assert "2 employees match" in data["answer"]
    assert "Berlin, Germany" in data["answer"]
    assert "London, UK" in data["answer"]


async def test_person_location_missing_from_resume_is_refused(
    tool: ResumeSearchTool,
) -> None:
    data = await _ask(tool, purpose="location_person", name="Liam Novak", question="Liam")
    assert "couldn't find a work location" in data["answer"]
    assert data["facts"] == []


async def test_descriptive_reference_can_ask_where_someone_works(
    tool: ResumeSearchTool,
) -> None:
    data = await _ask(
        tool,
        purpose="location_person",
        name="Kubernetes engineer",
        reference="descriptive",
        question="where does the Kubernetes engineer work",
    )
    assert "Dubai, UAE" in data["answer"]


# --- cohort -----------------------------------------------------------------


async def test_city_cohort_publishes_only_matching_employees(
    tool: ResumeSearchTool,
) -> None:
    data = await _ask(tool, purpose="location_cohort", city="Berlin", question="Berlin")
    assert data["employee_ids"] == ["e1"]
    # No prose: the employees table owns the names, so this node only resolves who.
    assert "answer" not in data


async def test_country_cohort_covers_every_city_in_it(tool: ResumeSearchTool) -> None:
    data = await _ask(tool, purpose="location_cohort", country="USA", question="USA")
    assert data["employee_ids"] == ["e4"]


async def test_empty_cohort_publishes_nothing_rather_than_a_sample(
    tool: ResumeSearchTool,
) -> None:
    """A near-miss here would put unrelated people into the next SQL node."""
    data = await _ask(tool, purpose="location_cohort", city="Paris", country="UK")
    assert data["employee_ids"] == []
    assert data["facts"] == []


async def test_id_scoped_cohort_reads_one_known_person(tool: ResumeSearchTool) -> None:
    """The profile path: identity is already resolved, so no place is asked for."""
    data = await _ask(tool, purpose="location_cohort", employee_ids=["e2"])
    assert data["employee_ids"] == ["e2"]
    assert data["facts"][0]["location"] == "London, UK"
    # Corpus-wide coverage would be a lie about a single-person lookup.
    assert "coverage" not in data


async def test_places_do_not_share_a_cache_entry(tool: ResumeSearchTool) -> None:
    berlin = await _ask(tool, purpose="location_cohort", city="Berlin", question="q")
    dubai = await _ask(tool, purpose="location_cohort", city="Dubai", question="q")
    assert berlin["employee_ids"] == ["e1"]
    assert dubai["employee_ids"] == ["e0"]


# --- facet ------------------------------------------------------------------


async def test_facet_counts_distinct_places_without_publishing_a_cohort(
    tool: ResumeSearchTool,
) -> None:
    data = await _ask(tool, purpose="location_facet", facet="city", question="how many cities")
    assert data["rows"] == [
        {"city": "Berlin"},
        {"city": "Dubai"},
        {"city": "London"},
        {"city": "New York"},
        {"city": "Paris"},
    ]
    assert data["count"] == 5
    # An aggregate is not a cohort: "them" must not become all 100 employees.
    assert data["employee_ids"] == []
    assert data["coverage"] == {"covered": len(PEOPLE) - 1, "total": len(PEOPLE)}


async def test_country_facet_deduplicates(tool: ResumeSearchTool) -> None:
    data = await _ask(tool, purpose="location_facet", facet="country", question="countries")
    assert [row["country"] for row in data["rows"]] == [
        "France",
        "Germany",
        "UAE",
        "UK",
        "USA",
    ]


async def test_unknown_facet_is_rejected(tool: ResumeSearchTool) -> None:
    result = await tool.run(
        {"purpose": "location_facet", "facet": "salary"}, auth=AUTH
    )
    assert result.degraded
    assert "cannot be aggregated" in (result.error or "")


# --- registry ---------------------------------------------------------------


def test_purposes_resolve_to_the_location_attribute() -> None:
    assert resolve_purpose("location_person") == (LOCATION, "person")
    assert resolve_purpose("location_cohort") == (LOCATION, "cohort")
    assert resolve_purpose("location_facet") == (LOCATION, "facet")
    assert resolve_purpose("status_location") is None


def test_facet_dimensions_declare_the_resume_sourced_fields() -> None:
    """The plan validator reads this to keep these fields out of SQL."""
    assert facet_dimensions() == {"city", "country"}


def test_a_country_only_question_matches_every_city_in_it() -> None:
    fact = LocationFact(employee_id="e", name="X", city="Berlin", country="Germany")
    assert fact.matches(country="germany")
    assert fact.matches(city="berlin", country="Germany")
    assert not fact.matches(city="Paris")
    assert not fact.matches()


def test_as_employee_ids_accepts_payload_and_scalar() -> None:
    """Profile plans bind the whole employee payload; coerce it to ids."""
    from app.tools.resume_search.tool import _as_employee_ids

    assert _as_employee_ids("abc") == ["abc"]
    assert _as_employee_ids({"id": "abc", "full_name": "Ada"}) == ["abc"]
    assert _as_employee_ids({"employee": {"id": "abc"}}) == ["abc"]
    assert _as_employee_ids([{"id": "a"}, "b"]) == ["a", "b"]
    assert _as_employee_ids(None) == []
