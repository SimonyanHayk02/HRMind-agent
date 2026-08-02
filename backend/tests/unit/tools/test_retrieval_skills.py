"""Skill RAG must lexically confirm hits so embedding neighbors cannot join."""
from __future__ import annotations

import pytest

from app.adapters.cache.memory_cache import MemoryCache
from app.adapters.embeddings.fake_embeddings import FakeEmbeddings
from app.adapters.vectorstore.memory_vector_store import MemoryVectorStore
from app.tools.resume_search.skills import (
    canonicalize_skill,
    content_mentions_skill,
    extract_skill,
    filter_hits_for_skill,
)
from app.tools.resume_search.tool import ResumeSearchTool
from tests.unit.tools.conftest import AUTH, ask as _ask


@pytest.mark.parametrize(
    ("raw", "canon"),
    [
        ("kubernetes", "Kubernetes"),
        ("K8s", "Kubernetes"),
        ("CKA", "Kubernetes"),
        ("docker", "Docker"),
        ("machine learning", "Machine Learning"),
        ("ML", "Machine Learning"),
        ("golang", "Go"),
    ],
)
def test_canonicalize_skill(raw: str, canon: str) -> None:
    assert canonicalize_skill(raw) == canon


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("who knows Kubernetes?", "Kubernetes"),
        ("who knows k8s", "Kubernetes"),
        ("how many of them know Docker?", "Docker"),
        ("who knows Machine Learning", "Machine Learning"),
        ("looking for python experience", "Python"),
    ],
)
def test_extract_skill_from_question(question: str, expected: str) -> None:
    assert extract_skill(question) == expected


def test_extract_skill_prefers_explicit_param() -> None:
    assert extract_skill("who knows Docker?", explicit="Kubernetes") == "Kubernetes"


def test_docker_chunk_does_not_mention_kubernetes() -> None:
    docker = (
        "Jonas Martin — Ops Specialist, Operations (Dubai, UAE)\n"
        "Skills\nDocker, Communication, Collaboration\n"
        "Certificates\n- Professional certificate in Docker"
    )
    k8s = (
        "Hugo Berg — Senior Engineer, Engineering (Dubai, UAE)\n"
        "Skills\nKubernetes, Communication, Collaboration"
    )
    assert not content_mentions_skill(docker, "Kubernetes")
    assert content_mentions_skill(k8s, "Kubernetes")
    assert content_mentions_skill(docker, "Docker")
    assert content_mentions_skill(
        "Certificates\n- CKA\n- Professional certificate in Go", "Kubernetes"
    )


def test_filter_drops_semantic_neighbors() -> None:
    hits = [
        {
            "employee_id": "jonas",
            "content": "Skills\nDocker, Communication\nProfessional certificate in Docker",
            "score": 0.34,
        },
        {
            "employee_id": "hugo",
            "content": "Skills\nKubernetes, Communication",
            "score": 0.45,
        },
        {
            "employee_id": "diego",
            "content": "Delivered projects using Kubernetes",
            "score": 0.43,
        },
    ]
    kept = filter_hits_for_skill(hits, "Kubernetes")
    assert {h["employee_id"] for h in kept} == {"hugo", "diego"}


def test_filter_noop_without_skill() -> None:
    hits = [{"employee_id": "x", "content": "Docker", "score": 0.9}]
    assert filter_hits_for_skill(hits, None) == hits


async def test_skill_purpose_drops_docker_neighbor_for_kubernetes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if embeddings return a Docker chunk for a K8s query, it must not join."""
    store = MemoryVectorStore()
    tool = ResumeSearchTool(
        embeddings=FakeEmbeddings(),
        vector_store=store,
        cache=MemoryCache(),
    )
    fake_hits = [
        {
            "id": "c-docker",
            "employee_id": "jonas",
            "employee_name": "Jonas Martin",
            "section": "Skills",
            "content": (
                "Jonas Martin — Ops Specialist\nSkills\nDocker, Communication\n"
                "Professional certificate in Docker"
            ),
            "score": 0.41,
            "metadata": {"employee_name": "Jonas Martin"},
        },
        {
            "id": "c-k8s",
            "employee_id": "hugo",
            "employee_name": "Hugo Berg",
            "section": "Skills",
            "content": "Hugo Berg — Senior Engineer\nSkills\nKubernetes, Communication",
            "score": 0.40,
            "metadata": {"employee_name": "Hugo Berg"},
        },
    ]

    async def _fake_sim(**_kwargs):
        return list(fake_hits)

    monkeypatch.setattr(store, "similarity_search", _fake_sim)

    data = await _ask(
        tool,
        question="who knows Kubernetes",
        purpose="skill",
        skill="Kubernetes",
    )
    assert "jonas" not in data["employee_ids"]
    assert "hugo" in data["employee_ids"]
