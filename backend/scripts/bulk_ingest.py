#!/usr/bin/env python3
"""Bulk-ingest pending resumes into pgvector using configured embeddings."""
from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.embeddings.cached_embeddings import CachedEmbeddings
from app.adapters.embeddings.fake_embeddings import FakeEmbeddings
from app.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
from app.adapters.cache.memory_cache import MemoryCache
from app.adapters.persistence.sqlalchemy.models import EmployeeModel, ResumeModel
from app.adapters.vectorstore.pgvector_store import PgVectorStore
from app.config.settings import get_settings
from app.ingest.pipeline import IngestPipeline


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    if settings.openai_api_key:
        embeddings = CachedEmbeddings(
            OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
                dims=settings.embedding_dims,
            ),
            MemoryCache(),
        )
        print(f"Using OpenAI embeddings ({settings.embedding_model})")
    else:
        # Dev-only: will NOT match 1536-d pgvector column well — prefer OpenAI key
        embeddings = FakeEmbeddings(model_name=settings.embedding_model, dims=settings.embedding_dims)
        print("WARNING: no OPENAI_API_KEY; fake embeddings may fail against vector(1536)")

    store = PgVectorStore(session_factory)
    pipeline = IngestPipeline(embeddings, store, parse_version=settings.parse_version)

    async with session_factory() as session:
        resumes = (await session.execute(select(ResumeModel))).scalars().all()
        total = 0
        for i, resume in enumerate(resumes, 1):
            emp = await session.get(EmployeeModel, resume.employee_id)
            if emp is None:
                continue
            path = Path(resume.storage_path)
            if not path.exists():
                resume.status = "failed"
                resume.error = "file missing"
                continue
            try:
                n = await pipeline.ingest_file(
                    path,
                    employee_id=str(resume.employee_id),
                    employee_name=f"{emp.first_name} {emp.last_name}",
                    resume_id=str(resume.id),
                    position=emp.position,
                    department=emp.department,
                )
                resume.status = "ready"
                resume.error = None
                total += n
                print(f"[{i}/{len(resumes)}] {emp.first_name} {emp.last_name}: {n} chunks")
            except Exception as exc:
                resume.status = "failed"
                resume.error = str(exc)[:500]
                print(f"[{i}/{len(resumes)}] FAILED {emp.email}: {exc}")
        await session.commit()

    await engine.dispose()
    print(f"Done. Ingested chunks={total}")


if __name__ == "__main__":
    asyncio.run(main())
