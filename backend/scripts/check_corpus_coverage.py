#!/usr/bin/env python3
"""Report how much of the corpus a cohort answer can actually see.

Retrieving every ``Personal`` chunk is only complete if those chunks exist. A
resume that failed to ingest, or that carries no parseable date, silently drops
its owner out of "whose birthday is today" while every retrieval metric still
reads 1.0. This script makes that shortfall explicit.

Exit codes: 1 when a resume exists but was never indexed (an ingest failure);
attribute gaps are reported and only fail the run under ``--strict``.
"""
from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.vectorstore.pgvector_store import PgVectorStore
from app.config.settings import get_settings
from app.tools.resume_search.attributes import ATTRIBUTES


async def main(strict: bool) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    store = PgVectorStore(session_factory)

    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT e.id AS employee_id,
                           e.first_name || ' ' || e.last_name AS name,
                           r.id IS NOT NULL AS has_resume,
                           COALESCE(r.status, '-') AS resume_status,
                           (
                             SELECT count(*) FROM resume_chunks c
                             WHERE c.employee_id = e.id
                           ) AS chunks
                    FROM employees e
                    LEFT JOIN resumes r ON r.employee_id = e.id
                    ORDER BY e.last_name, e.first_name
                    """
                )
            )
        ).mappings().all()

    employees = {str(r["employee_id"]): dict(r) for r in rows}
    with_resume = [r for r in employees.values() if r["has_resume"]]
    not_indexed = [r for r in with_resume if int(r["chunks"] or 0) == 0]

    print(f"employees                : {len(employees)}")
    print(f"with a resume row        : {len(with_resume)}")
    print(f"indexed (>=1 chunk)      : {len(with_resume) - len(not_indexed)}")

    failures = 0
    for row in not_indexed:
        print(f"  NOT INDEXED  {row['name']} (resume status={row['resume_status']})")
        failures += 1

    attribute_gaps = 0
    for attribute in ATTRIBUTES.values():
        chunks = await store.fetch_section_chunks(
            attribute.sections, content_hints=attribute.content_hints
        )
        extracted = {
            str(c["employee_id"])
            for c in chunks
            if attribute.extract(str(c.get("content") or "")) is not None
        }
        indexed = {
            eid for eid, r in employees.items() if int(r["chunks"] or 0) > 0
        }
        missing = sorted(indexed - extracted, key=lambda e: employees[e]["name"])
        print(
            f"{attribute.name:<25}: {len(extracted)}/{len(indexed)} indexed employees"
        )
        for eid in missing:
            print(f"  NO {attribute.name.upper()}  {employees[eid]['name']}")
        attribute_gaps += len(missing)

    if attribute_gaps and strict:
        failures += attribute_gaps

    await engine.dispose()
    if failures:
        print(f"FAIL: {failures} coverage problem(s)")
        return 1
    print("OK: corpus coverage is complete for every registered attribute"
          if not attribute_gaps
          else f"OK: indexing complete; {attribute_gaps} documented attribute gap(s)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when an indexed resume has no value for a registered attribute",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.strict)))
