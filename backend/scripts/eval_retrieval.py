#!/usr/bin/env python3
"""Measure resume retrieval against the real stack.

The hermetic tests in ``tests/unit/tools/test_retrieval_birthday.py`` pin the
pipeline's logic, but they run on hash-derived fake vectors and so cannot say
anything about embedding recall. This script runs the same questions against
Postgres with real embeddings and reports, per retrieval arm:

    recall@k, MRR, hit-rate  — did we find the right person?
    answer exactness         — did the answer state the date written in the resume?

Ground truth comes from the resume corpus itself (the ``Personal`` chunks), while
the questions in ``data/golden/retrieval_birthday.jsonl`` are hand-authored and
adversarial: typos, partial names, namesakes, descriptive references, a resume
with no date at all, and a leap-day birthday.

Usage: python scripts/eval_retrieval.py [--verbose]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.cache.memory_cache import MemoryCache
from app.adapters.embeddings.cached_embeddings import CachedEmbeddings
from app.adapters.embeddings.fake_embeddings import FakeEmbeddings
from app.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
from app.adapters.vectorstore.pgvector_store import PgVectorStore
from app.application.understanding.birthday import extract_birthday
from app.config.settings import get_settings
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.tools.resume_search.attributes import BIRTH_DATE
from app.tools.resume_search.birthday import (
    format_birth_date,
    format_day_month,
    month_name,
    next_occurrence,
    today_utc,
)
from app.tools.resume_search.entity_resolution import ARM_WEIGHTS
from app.tools.resume_search.fusion import reciprocal_rank_fusion
from app.tools.resume_search.tool import ResumeSearchTool

GOLDEN_BIRTHDAY = Path("data/golden/retrieval_birthday.jsonl")
GOLDEN_LOCATION = Path("data/golden/retrieval_location.jsonl")
KS = (1, 5, 20)
ARMS = ("name", "lexical", "dense", "fused")


@dataclass
class Person:
    employee_id: str
    name: str
    born: date | None = None
    chunk_id: str = ""
    position: str = ""
    city: str = ""
    country: str = ""


@dataclass
class Corpus:
    people: dict[str, Person] = field(default_factory=dict)
    name_counts: Counter[str] = field(default_factory=Counter)

    @property
    def dated(self) -> list[Person]:
        return [p for p in self.people.values() if p.born is not None]

    def sorted_people(self) -> list[Person]:
        return sorted(self.people.values(), key=lambda p: p.employee_id)


async def load_corpus(store: PgVectorStore, session_factory: Any) -> Corpus:
    """Ground truth read straight from the indexed resumes."""
    from app.tools.resume_search.attributes import LOCATION

    corpus = Corpus()
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT DISTINCT employee_id,
                           COALESCE(metadata->>'employee_name', '') AS name,
                           COALESCE(metadata->>'position', '') AS position
                    FROM resume_chunks
                    """
                )
            )
        ).mappings().all()
    for row in rows:
        eid = str(row["employee_id"])
        corpus.people[eid] = Person(
            employee_id=eid,
            name=row["name"],
            position=row["position"],
        )
        corpus.name_counts[row["name"]] += 1

    chunks = await store.fetch_section_chunks(
        BIRTH_DATE.sections, content_hints=BIRTH_DATE.content_hints
    )
    for chunk in chunks:
        person = corpus.people.get(str(chunk["employee_id"]))
        if person is None or person.born is not None:
            continue
        born = BIRTH_DATE.extract(str(chunk.get("content") or ""))
        if born is not None:
            person.born = born
            person.chunk_id = str(chunk.get("id") or "")

    # City/country live only in resume text — parse them the same way the tool does.
    location_chunks = await store.fetch_section_chunks(
        LOCATION.sections, content_hints=LOCATION.content_hints
    )
    for chunk in location_chunks:
        person = corpus.people.get(str(chunk["employee_id"]))
        if person is None or person.city:
            continue
        place = LOCATION.extract(str(chunk.get("content") or ""))
        if place is not None:
            if place.city:
                person.city = place.city
            if place.country:
                person.country = place.country
    return corpus


def _typo(name: str) -> str:
    """Deterministic misspelling: double the final letter of the surname."""
    parts = name.split()
    if not parts:
        return name
    parts[-1] = parts[-1] + parts[-1][-1]
    return " ".join(parts)


def pick_subjects(corpus: Corpus) -> dict[str, Person | None]:
    """Deterministic fixtures drawn from the live corpus."""
    unique = next(
        (
            p
            for p in corpus.sorted_people()
            if p.born is not None and corpus.name_counts[p.name] == 1
        ),
        None,
    )
    duplicate = next(
        (
            p
            for p in corpus.sorted_people()
            if p.born is not None and corpus.name_counts[p.name] > 1
        ),
        None,
    )
    no_dob = next((p for p in corpus.sorted_people() if p.born is None), None)
    leap_day = next(
        (
            p
            for p in corpus.sorted_people()
            if p.born is not None and (p.born.month, p.born.day) == (2, 29)
        ),
        None,
    )
    located = next(
        (
            p
            for p in corpus.sorted_people()
            if p.city and corpus.name_counts[p.name] == 1
        ),
        None,
    )
    duplicate_located = next(
        (
            p
            for p in corpus.sorted_people()
            if p.city and corpus.name_counts[p.name] > 1
        ),
        None,
    )
    no_location = next((p for p in corpus.sorted_people() if not p.city), None)
    return {
        "unique": unique,
        "duplicate": duplicate,
        "no_dob": no_dob,
        "leap_day": leap_day,
        "located": located,
        "duplicate_located": duplicate_located,
        "no_location": no_location,
        "none": None,
        "cohort": None,
        "facet": None,
    }


def render(template: str, person: Person | None, corpus: Corpus) -> str | None:
    if person is None:
        # Cohort/facet templates need no person; month is for birthday cohorts.
        try:
            return template.format(month=month_name(today_utc().month))
        except KeyError:
            return template
    first, _, last = person.name.partition(" ")
    values = {
        "name": person.name,
        "name_lower": person.name.lower(),
        "name_typo": _typo(person.name),
        "first_name": first,
        "last_name": last or first,
        "position": person.position or "engineer",
        "city": person.city or "Berlin",
        "month": month_name(person.born.month) if person.born else "July",
    }
    try:
        return template.format(**values)
    except KeyError:
        return None


async def arm_rankings(
    *,
    store: PgVectorStore,
    embeddings: Any,
    query: str,
    min_similarity: float,
) -> dict[str, list[str]]:
    """Ranked employee ids from each retrieval arm, plus their fusion."""
    filters = {"exclude_sections": list(BIRTH_DATE.sections)}
    name_rows = await store.resolve_names(query, limit=40, min_similarity=min_similarity)
    name_ranked = [str(r["employee_id"]) for r in name_rows]

    def order(hits: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for hit in hits:
            eid = str(hit.get("employee_id") or "")
            if eid and eid not in out:
                out.append(eid)
        return out

    lexical_ranked = order(await store.lexical_search(query, top_k=60, filters=filters))
    embedding = await embeddings.embed(query)
    dense_ranked = order(
        await store.similarity_search(embedding=embedding, top_k=60, filters=filters)
    )
    fused = reciprocal_rank_fusion(
        [name_ranked, lexical_ranked, dense_ranked], weights=ARM_WEIGHTS
    )
    return {
        "name": name_ranked,
        "lexical": lexical_ranked,
        "dense": dense_ranked,
        "fused": [eid for eid, _ in fused],
    }


def score_ranking(ranked: list[str], targets: set[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for k in KS:
        top = ranked[:k]
        found = len(targets & set(top))
        scores[f"recall@{k}"] = found / len(targets)
        scores[f"hit@{k}"] = 1.0 if found else 0.0
    rank = next((i for i, eid in enumerate(ranked, 1) if eid in targets), 0)
    scores["mrr"] = 1.0 / rank if rank else 0.0
    return scores


def check_answer(row: dict[str, Any], person: Person | None, answer: str, corpus: Corpus) -> tuple[bool, str]:
    expect = row["expect"]
    lowered = answer.lower()
    if expect == "refusal":
        ok = "couldn't find a date of birth" in lowered
        return ok, "refused" if ok else "did not refuse"
    if expect == "disambiguation":
        ok = "employees match" in lowered
        return ok, "listed namesakes" if ok else "did not disambiguate"
    if expect in {"date", "age", "day_month", "date_or_disambiguation"}:
        if person is None or person.born is None:
            return False, "no ground truth"
        has_date = format_birth_date(person.born) in answer
        if expect == "day_month":
            # A wish names the day, not the birth year.
            ok = has_date or format_day_month(person.born) in answer
            return ok, "day and month stated" if ok else "day and month missing"
        if expect == "age":
            ok = has_date or str(_age(person.born)) in answer
            return ok, "age stated" if ok else "age missing"
        if expect == "date_or_disambiguation":
            ok = has_date or "employees match" in lowered
            return ok, "date or namesakes" if ok else "neither date nor namesakes"
        return has_date, "date stated" if has_date else "date missing or wrong"
    return False, f"unknown expectation {expect}"


def _age(born: date) -> int:
    today = today_utc()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def check_cohort(row: dict[str, Any], answer: str, corpus: Corpus) -> tuple[bool, str]:
    """Cohort answers must name every match in the corpus, not a sample."""
    today = today_utc()
    scope = row.get("scope") or "today"
    if scope == "today":
        expected = [p for p in corpus.dated if (p.born.month, p.born.day) == (today.month, today.day)]
    elif scope == "month":
        expected = [p for p in corpus.dated if p.born.month == today.month]
    else:
        # Upcoming lists at most ten names, so the claimed total is what matters.
        upcoming = [p for p in corpus.dated if _days_until(p.born, today) <= 30]
        claimed = f"{len(upcoming)} employee" if len(upcoming) != 1 else "1 employee"
        ok = claimed in answer
        return ok, (
            f"counted all {len(upcoming)} within 30 days"
            if ok
            else f"expected {len(upcoming)} within 30 days"
        )
    missing = [p.name for p in expected if p.name not in answer]
    if missing:
        return False, f"missing {len(missing)}: {', '.join(missing[:3])}"
    if not expected:
        return True, "no matches expected; answer is honest"
    return True, f"named all {len(expected)}"


def _days_until(born: date, today: date) -> int:
    return (next_occurrence(born, today) - today).days


async def main(verbose: bool) -> int:
    if not verbose:
        # Retrieval events are useful in the app log, noise in a report.
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
        )
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    store = PgVectorStore(session_factory)

    if settings.openai_api_key:
        embeddings = CachedEmbeddings(
            OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
                dims=settings.embedding_dims,
            ),
            MemoryCache(),
        )
    else:
        embeddings = FakeEmbeddings(
            model_name=settings.embedding_model, dims=settings.embedding_dims
        )
        print("WARNING: no OPENAI_API_KEY — dense numbers below are meaningless\n")

    tool = ResumeSearchTool(
        embeddings=embeddings,
        vector_store=store,
        cache=MemoryCache(),
        rrf_k=settings.retrieval_rrf_k,
        name_similarity=settings.retrieval_name_similarity,
    )
    auth = AuthContext(
        user_id="eval", tenant_id=settings.default_tenant_id, role=Role.RECRUITER
    )

    corpus = await load_corpus(store, session_factory)
    subjects = pick_subjects(corpus)
    rows = [
        json.loads(line)
        for line in GOLDEN_BIRTHDAY.read_text().splitlines()
        if line.strip()
    ]

    located_n = sum(1 for p in corpus.people.values() if p.city)
    print(
        f"corpus: {len(corpus.people)} indexed employees, "
        f"{len(corpus.dated)} with a date of birth, "
        f"{located_n} with a parseable location, "
        f"{len(corpus.people) - len(corpus.dated)} dob gap(s), "
        f"{len(corpus.people) - located_n} location gap(s)\n"
    )

    per_arm: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    answer_pass = answer_fail = skipped = 0

    print("--- birthday ---")
    for row in rows:
        person = subjects.get(row["subject"], None)
        if row["subject"] not in {"none", "cohort"} and person is None:
            print(f"SKIP {row['id']:<28} no {row['subject']} subject in this corpus")
            skipped += 1
            continue
        question = render(row["template"], person, corpus)
        if question is None:
            print(f"SKIP {row['id']:<28} template needs data this corpus lacks")
            skipped += 1
            continue

        request = extract_birthday(question)
        if row["subject"] == "cohort":
            data = (
                await tool.run(
                    {
                        "purpose": "birthday_cohort",
                        "question": question,
                        "scope": row.get("scope") or "today",
                        "month": today_utc().month if row.get("scope") == "month" else None,
                    },
                    auth=auth,
                )
            ).data
            ok, note = check_cohort(row, str(data.get("answer") or ""), corpus)
            coverage = data.get("coverage") or {}
            note = f"{note}; coverage {coverage.get('covered')}/{coverage.get('total')}"
        else:
            reference = row.get("reference") or "name"
            subject_query = (
                question if reference == "descriptive" else (request.person_name or question)
            )
            targets: set[str] = set()
            if person is not None:
                targets = (
                    {person.employee_id}
                    if row["subject"] != "duplicate"
                    else {
                        p.employee_id
                        for p in corpus.people.values()
                        if p.name == person.name
                    }
                )
            if targets:
                rankings = await arm_rankings(
                    store=store,
                    embeddings=embeddings,
                    query=subject_query,
                    min_similarity=settings.retrieval_name_similarity,
                )
                for arm, ranked in rankings.items():
                    per_arm[arm].append(score_ranking(ranked, targets))
                if verbose:
                    for arm, ranked in rankings.items():
                        hit = next(
                            (i for i, e in enumerate(ranked, 1) if e in targets), None
                        )
                        print(f"     {arm:<8} rank={hit or '-'} depth={len(ranked)}")

            data = (
                await tool.run(
                    {
                        "purpose": "birthday_person",
                        "question": question,
                        "name": subject_query,
                        "reference": reference,
                        "wants_age": request.wants_age,
                        "wants_wish": request.wants_wish,
                    },
                    auth=auth,
                )
            ).data
            ok, note = check_answer(row, person, str(data.get("answer") or ""), corpus)

        answer_pass += int(ok)
        answer_fail += int(not ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {row['id']:<28} {note}")
        if verbose or not ok:
            print(f"     q: {question}")
            print(f"     a: {str(data.get('answer') or '')[:300]}")

    print("\nretrieval (person questions, mean over queries)")
    header = "arm".ljust(10) + "".join(f"recall@{k}".ljust(11) for k in KS) + "MRR"
    print(header)
    for arm in ARMS:
        samples = per_arm[arm]
        if not samples:
            continue
        cells = "".join(
            f"{sum(s[f'recall@{k}'] for s in samples) / len(samples):.2f}".ljust(11)
            for k in KS
        )
        mrr = sum(s["mrr"] for s in samples) / len(samples)
        print(f"{arm.ljust(10)}{cells}{mrr:.2f}")

    # --- location attribute ---
    print("\n--- location ---")
    loc_rows = [
        json.loads(line)
        for line in GOLDEN_LOCATION.read_text().splitlines()
        if line.strip()
    ]
    for row in loc_rows:
        person = subjects.get(row["subject"], None)
        if row["subject"] not in {"none", "facet"} and person is None:
            print(f"SKIP {row['id']:<28} no {row['subject']} subject in this corpus")
            skipped += 1
            continue
        question = render(row["template"], person, corpus)
        if question is None:
            print(f"SKIP {row['id']:<28} template needs data this corpus lacks")
            skipped += 1
            continue

        if row["subject"] == "facet":
            data = (
                await tool.run(
                    {
                        "purpose": "location_facet",
                        "facet": row.get("facet") or "city",
                        "question": question,
                    },
                    auth=auth,
                )
            ).data
            ok, note = check_location_facet(row, data, corpus)
        else:
            reference = row.get("reference") or "name"
            name = person.name if person and reference != "descriptive" else question
            if reference == "descriptive":
                name = question
            data = (
                await tool.run(
                    {
                        "purpose": "location_person",
                        "question": question,
                        "name": name,
                        "reference": reference,
                    },
                    auth=auth,
                )
            ).data
            ok, note = check_location_answer(row, person, str(data.get("answer") or ""))

        answer_pass += int(ok)
        answer_fail += int(not ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {row['id']:<28} {note}")
        if verbose or not ok:
            print(f"     q: {question}")
            print(f"     a: {str(data.get('answer') or data)[:300]}")

    print(f"\nanswers: {answer_pass} pass, {answer_fail} fail, {skipped} skipped")
    await engine.dispose()
    return 1 if answer_fail else 0


def check_location_answer(
    row: dict[str, Any], person: Person | None, answer: str
) -> tuple[bool, str]:
    expect = row["expect"]
    lowered = answer.lower()
    if expect == "refusal":
        ok = "couldn't find a work location" in lowered
        return ok, "refused" if ok else "did not refuse"
    if expect == "disambiguation":
        ok = "employees match" in lowered
        return ok, "listed namesakes" if ok else "did not disambiguate"
    if expect in {"place", "place_or_disambiguation"}:
        if person is None or not person.city:
            return False, "no ground truth"
        has_place = person.city.lower() in lowered
        if expect == "place_or_disambiguation":
            ok = has_place or "employees match" in lowered
            return ok, "place or namesakes" if ok else "neither"
        return has_place, "place stated" if has_place else "place missing or wrong"
    return False, f"unknown expectation {expect}"


def check_location_facet(
    row: dict[str, Any], data: dict[str, Any], corpus: Corpus
) -> tuple[bool, str]:
    facet = row.get("facet") or "city"
    if data.get("employee_ids"):
        return False, "facet published a cohort"
    rows = data.get("rows") or []
    values = {str(r.get(facet) or "") for r in rows if isinstance(r, dict) and r.get(facet)}
    if facet == "country":
        expected = {p.country for p in corpus.people.values() if p.country}
    else:
        expected = {p.city for p in corpus.people.values() if p.city}
    missing = expected - values
    if missing:
        return False, f"missing {len(missing)} {facet}s"
    return True, f"{len(values)} {facet}s"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print per-arm ranks and answers")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.verbose)))
