"""Hermetic tier of the retrieval evaluation.

These tests run the real pipeline — section splitting, chunk enrichment, the
store's filters, fusion, resolution and extraction — over an in-memory corpus.
They deliberately do not claim to measure embedding quality: FakeEmbeddings
produces hash-derived vectors, so dense recall is only meaningful against real
embeddings. That is what ``scripts/eval_retrieval.py`` is for.
"""
from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from app.adapters.embeddings.fake_embeddings import FakeEmbeddings
from app.adapters.sql.validator import validate_sql
from app.adapters.vectorstore.memory_vector_store import MemoryVectorStore
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.errors import ValidationFailedError
from app.tools.resume_search.attributes import BIRTH_DATE, resolve_purpose
from app.tools.resume_search.entity_resolution import resolve_employees
from app.tools.resume_search.fusion import reciprocal_rank_fusion
from app.tools.resume_search.tool import ResumeSearchTool
from tests.unit.tools.conftest import AUTH, PEOPLE, ask as _ask


# --- chunk enrichment -------------------------------------------------------


async def test_chunks_are_self_contained(corpus: MemoryVectorStore) -> None:
    """A Personal chunk must identify its owner, or dense search cannot use it."""
    chunks = await corpus.fetch_section_chunks(("Personal",))
    assert chunks
    for chunk in chunks:
        content = chunk["content"]
        assert chunk["employee_name"] in content
        assert "Date of Birth" in content
    eva = next(c for c in chunks if c["employee_id"] == "e0")
    assert "Senior Engineer" in eva["content"]
    assert "Dubai" in eva["content"]


# --- store filters ----------------------------------------------------------


async def test_similarity_search_honours_filters(corpus: MemoryVectorStore) -> None:
    embedding = await FakeEmbeddings().embed("anything")
    only = await corpus.similarity_search(
        embedding=embedding, top_k=50, filters={"sections": ["Personal"]}
    )
    assert only and {h["section"] for h in only} == {"Personal"}

    without = await corpus.similarity_search(
        embedding=embedding, top_k=50, filters={"exclude_sections": ["Personal"]}
    )
    assert without and "Personal" not in {h["section"] for h in without}

    scoped = await corpus.similarity_search(
        embedding=embedding, top_k=50, filters={"employee_ids": ["e1"]}
    )
    assert scoped and {h["employee_id"] for h in scoped} == {"e1"}


async def test_fetch_section_chunks_is_complete(corpus: MemoryVectorStore) -> None:
    """Cohort answers need every owner of the section, not a ranked sample."""
    chunks = await corpus.fetch_section_chunks(
        BIRTH_DATE.sections, content_hints=BIRTH_DATE.content_hints
    )
    # Bob Smith's resume has no Personal section, so everyone else appears.
    assert {c["employee_id"] for c in chunks} == {"e0", "e1", "e2", "e4", "e5"}
    assert await corpus.count_indexed_employees() == len(PEOPLE)


# --- fusion -----------------------------------------------------------------


def test_rrf_prefers_agreement_and_respects_weights() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert [item for item, _ in fused] == ["a", "b"]

    # A heavier arm outranks a lighter arm's top hit.
    weighted = reciprocal_rank_fusion([["x"], ["y"]], weights=(2.0, 1.0))
    assert weighted[0][0] == "x"

    assert reciprocal_rank_fusion([[], []]) == []


# --- stage 1: identity resolution ------------------------------------------


async def test_exact_name_returns_every_namesake(corpus: MemoryVectorStore) -> None:
    resolution = await resolve_employees(
        query="Ivy Chen", store=corpus, embeddings=FakeEmbeddings()
    )
    assert resolution.via == "name"
    assert set(resolution.employee_ids) == {"e1", "e2"}


async def test_misspelled_name_resolves_fuzzily(corpus: MemoryVectorStore) -> None:
    resolution = await resolve_employees(
        query="Eva Kimm", store=corpus, embeddings=FakeEmbeddings()
    )
    assert resolution.via == "fuzzy"
    assert resolution.employee_ids[0] == "e0"
    assert resolution.note and "Eva Kim" in resolution.note


async def test_descriptive_reference_resolves_without_a_name(
    corpus: MemoryVectorStore,
) -> None:
    """No name in the question, so identity comes from the descriptive sections."""
    resolution = await resolve_employees(
        query="Kubernetes engineer in Dubai",
        store=corpus,
        embeddings=FakeEmbeddings(),
        exclude_sections=BIRTH_DATE.sections,
    )
    assert resolution.via == "fused"
    assert resolution.employee_ids[0] == "e0"


async def test_resolution_survives_embedding_failure(corpus: MemoryVectorStore) -> None:
    class BrokenEmbeddings(FakeEmbeddings):
        async def embed(self, text: str) -> list[float]:
            raise RuntimeError("embeddings offline")

    resolution = await resolve_employees(
        query="Kubernetes engineer in Dubai",
        store=corpus,
        embeddings=BrokenEmbeddings(),
        exclude_sections=BIRTH_DATE.sections,
    )
    assert resolution.employee_ids[0] == "e0"
    assert resolution.note and "text only" in resolution.note


# --- stage 2 and 3: attribute retrieval and extraction ---------------------


async def test_person_birthday_is_extracted_with_provenance(
    tool: ResumeSearchTool, corpus: MemoryVectorStore
) -> None:
    data = await _ask(tool, purpose="birthday_person", name="Eva Kim", question="Eva Kim")
    assert "12 March 1991" in data["answer"]
    assert data["resolved_via"] == "name"

    fact = data["facts"][0]
    assert fact["birth_date"] == "1991-03-12"
    chunk_ids = {c["id"] for c in await corpus.fetch_section_chunks(("Personal",))}
    assert fact["extracted_from_chunk_id"] in chunk_ids


async def test_person_birthday_lists_every_namesake(tool: ResumeSearchTool) -> None:
    data = await _ask(
        tool, purpose="birthday_person", name="Ivy Chen", question="Ivy Chen"
    )
    # Two Ivy Chens exist, so the honest answer disambiguates instead of guessing.
    assert "2 employees match" in data["answer"]
    assert "31 July 1985" in data["answer"]
    assert "3 April 1980" in data["answer"]


async def test_person_birthday_missing_from_resume_is_refused(
    tool: ResumeSearchTool,
) -> None:
    data = await _ask(
        tool, purpose="birthday_person", name="Bob Smith", question="Bob Smith"
    )
    assert "couldn't find a date of birth" in data["answer"]
    assert data["facts"] == []


async def test_person_birthday_tolerates_a_typo(tool: ResumeSearchTool) -> None:
    data = await _ask(
        tool, purpose="birthday_person", name="Eva Kimm", question="Eva Kimm"
    )
    assert "No exact match" in data["answer"]
    assert "12 March 1991" in data["answer"]


async def test_unknown_person_is_not_invented(tool: ResumeSearchTool) -> None:
    """An unmatched name must not fall through to whoever ranked highest."""
    data = await _ask(
        tool, purpose="birthday_person", name="Zzz Nobody", question="Zzz Nobody"
    )
    assert "couldn't find a date of birth" in data["answer"]
    assert data["facts"] == []
    assert data["resolved_via"] == "none"


async def test_cohort_today_names_the_person_and_admits_coverage(
    tool: ResumeSearchTool,
) -> None:
    data = await _ask(tool, purpose="birthday_cohort", scope="today", question="dob")
    assert "Ivy Chen" in data["answer"]
    assert "Happy birthday" in data["answer"]
    # One resume has no date of birth; the answer says so rather than implying
    # it read everyone.
    assert data["coverage"] == {"covered": len(PEOPLE) - 1, "total": len(PEOPLE)}
    assert f"{len(PEOPLE) - 1} of {len(PEOPLE)} resumes" in data["answer"]


async def test_cohort_month_lists_everyone_in_that_month(tool: ResumeSearchTool) -> None:
    data = await _ask(
        tool, purpose="birthday_cohort", scope="month", month=3, question="dob"
    )
    assert "Eva Kim" in data["answer"]
    assert "Ivy Chen" not in data["answer"]


async def test_cohort_handles_leap_day(tool: ResumeSearchTool) -> None:
    data = await _ask(
        tool, purpose="birthday_cohort", scope="month", month=2, question="dob"
    )
    assert "Grace Okafor" in data["answer"]
    assert "29 February" in data["answer"]


async def test_cohort_answers_are_sourced_from_resume_chunks(
    tool: ResumeSearchTool,
) -> None:
    result = await tool.run(
        {"purpose": "birthday_cohort", "scope": "today", "question": "dob"}, auth=AUTH
    )
    assert result.sources
    assert {s.kind for s in result.sources} == {"resume_chunk"}


# --- guardrails -------------------------------------------------------------


def test_purposes_resolve_to_the_registered_attribute() -> None:
    assert resolve_purpose("birthday_person") == (BIRTH_DATE, "person")
    assert resolve_purpose("birthday_cohort") == (BIRTH_DATE, "cohort")
    assert resolve_purpose("status_resolve") is None
    assert resolve_purpose("") is None


def test_sql_tool_cannot_reach_the_chunk_table() -> None:
    """Resume-sourced facts must come from retrieval, never generated SQL."""
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    with pytest.raises(ValidationFailedError):
        validate_sql("SELECT content FROM resume_chunks", auth)


# --- observability ----------------------------------------------------------


async def test_retrieval_emits_structured_events(tool: ResumeSearchTool) -> None:
    """Every retrieval reports which arm resolved it and how much it read, so a
    silent recall drop shows up in logs rather than only in a wrong answer."""
    with capture_logs() as events:
        await _ask(tool, purpose="birthday_person", name="Eva Kim", question="dob")
        await _ask(tool, purpose="birthday_cohort", scope="today", question="dob")

    person = next(e for e in events if e["event"] == "resume_attribute_person")
    assert person["attribute"] == "birth_date"
    assert person["resolved_via"] == "name"
    assert person["candidates"] == 1
    assert person["facts"] == 1

    cohort = next(e for e in events if e["event"] == "resume_attribute_cohort")
    assert cohort["scope"] == "today"
    assert cohort["indexed_employees"] == len(PEOPLE)
    assert cohort["facts"] == len([p for p in PEOPLE if p[-1]])
