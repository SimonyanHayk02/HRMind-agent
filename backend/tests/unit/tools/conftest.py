"""One in-memory resume corpus shared by the attribute retrieval tests.

Birth date and location are both resume-only facts read from the same documents,
so they are tested against the same ingested corpus rather than two fixtures that
can drift apart.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.adapters.cache.memory_cache import MemoryCache
from app.adapters.embeddings.fake_embeddings import FakeEmbeddings
from app.adapters.vectorstore.memory_vector_store import MemoryVectorStore
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.ingest.pipeline import IngestPipeline
from app.tools.resume_search.tool import ResumeSearchTool

TODAY = date(2026, 7, 31)

AUTH = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)

# name, position, department, city, country, skill, date-of-birth line
PEOPLE = [
    ("Eva Kim", "Senior Engineer", "Engineering", "Dubai", "UAE", "Kubernetes", "12 March 1991"),
    ("Ivy Chen", "Recruiter", "People", "Berlin", "Germany", "Recruiting", "31 July 1985"),
    ("Ivy Chen", "Accountant", "Finance", "London", "UK", "Accounting", "3 April 1980"),
    ("Bob Smith", "Product Manager", "Product", "Paris", "France", "Roadmaps", None),
    ("Grace Okafor", "Data Scientist", "Engineering", "New York", "USA", "NLP", "29 February 1988"),
    # Deliberate location gap: the place written here is not a known city, so
    # coverage and the "not in the resume" refusal are exercised for real.
    ("Liam Novak", "Ops Specialist", "Operations", "Remote", "Elsewhere", "Docker", "5 May 1990"),
]


def resume_text(person: tuple) -> str:
    _name, position, department, city, country, skill, born = person
    personal = f"Personal\nDate of Birth: {born}\n\n" if born else ""
    return (
        f"Summary\nExperienced {position} in {department} based in {city}, {country}.\n\n"
        f"Location\n{city}, {country}\n\n"
        f"{personal}"
        f"Experience\n- {position} at Acme Corp\n- Delivered projects using {skill}\n\n"
        f"Skills\n{skill}, Communication\n\n"
        f"Education\nBSc\n"
    )


@pytest.fixture
async def corpus(tmp_path: Path) -> MemoryVectorStore:
    store = MemoryVectorStore()
    pipeline = IngestPipeline(FakeEmbeddings(), store)
    for index, person in enumerate(PEOPLE):
        name, position, department, *_ = person
        path = tmp_path / f"resume_{index}.txt"
        path.write_text(resume_text(person))
        await pipeline.ingest_file(
            path,
            employee_id=f"e{index}",
            employee_name=name,
            resume_id=f"r{index}",
            position=position,
            department=department,
        )
    return store


@pytest.fixture
def tool(corpus: MemoryVectorStore) -> ResumeSearchTool:
    return ResumeSearchTool(
        embeddings=FakeEmbeddings(),
        vector_store=corpus,
        cache=MemoryCache(),
    )


@pytest.fixture(autouse=True)
def _frozen_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.resume_search.tool.today_utc", lambda: TODAY)


async def ask(tool: ResumeSearchTool, **params) -> dict:
    result = await tool.run(params, auth=AUTH)
    assert not result.degraded, result.error
    return result.data
