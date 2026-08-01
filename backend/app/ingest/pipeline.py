from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from app.adapters.documents.docx_parser import parse_docx
from app.adapters.documents.pdf_parser import parse_pdf
from app.adapters.documents.section_splitter import split_sections
from app.adapters.documents.semantic_chunker import chunk_sections
from app.ingest.enrichment import build_chunk_text
from app.ingest.metadata import build_chunk_metadata
from app.ports.embeddings import EmbeddingClient
from app.ports.vector_store import VectorStore
from app.tools.resume_search.attributes import ATTRIBUTES
from app.tools.resume_search.location import parse_location

log = structlog.get_logger(__name__)


class IngestPipeline:
    def __init__(self, embeddings: EmbeddingClient, vector_store: VectorStore, *, parse_version: str = "1") -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._parse_version = parse_version

    async def ingest_file(
        self,
        path: Path,
        *,
        employee_id: str,
        employee_name: str,
        resume_id: str,
        position: str | None = None,
        department: str | None = None,
    ) -> int:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = parse_pdf(path)
            doc_name = path.name
        elif suffix in {".docx", ".doc"}:
            text = parse_docx(path)
            doc_name = path.name
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            doc_name = path.name

        sections = split_sections(text)
        pieces = chunk_sections(sections)
        # The document is the only source of location, so the header is built from
        # what the resume itself says rather than from a database row.
        place = parse_location(sections.get("Location") or text)
        # Each chunk carries its own entity context so it is retrievable and
        # citable without the surrounding document.
        contents = [
            build_chunk_text(
                content,
                section=section,
                employee_name=employee_name,
                position=position,
                department=department,
                city=place.city if place else None,
                country=place.country if place else None,
            )
            for section, content, _ in pieces
        ]
        vectors = await self._embeddings.embed_many(contents) if contents else []

        chunks = []
        for (section, _, idx), content, emb in zip(pieces, contents, vectors):
            chunks.append(
                {
                    "id": str(uuid4()),
                    "employee_id": employee_id,
                    "resume_id": resume_id,
                    "section": section,
                    "chunk_index": idx,
                    "content": content,
                    "embedding": emb,
                    "metadata": build_chunk_metadata(
                        employee_id=employee_id,
                        employee_name=employee_name,
                        document=doc_name,
                        section=section,
                        chunk_index=idx,
                        position=position,
                        department=department,
                    ),
                }
            )
        self._log_attribute_gaps(chunks, employee_id=employee_id, document=doc_name)
        await self._vector_store.delete_by_employee(employee_id)
        if chunks:
            await self._vector_store.upsert(chunks=chunks)
        return len(chunks)

    @staticmethod
    def _log_attribute_gaps(
        chunks: list[dict[str, Any]],
        *,
        employee_id: str,
        document: str,
    ) -> None:
        """Surface resumes that carry no extractable value for a registered attribute.

        Cohort answers claim to cover the whole corpus, so a resume that silently
        lacks a parseable fact must be visible rather than quietly missing.
        """
        for attribute in ATTRIBUTES.values():
            found = any(
                attribute.extract(str(chunk.get("content") or "")) is not None
                for chunk in chunks
                if str(chunk.get("section") or "") in attribute.sections
                or any(
                    hint in str(chunk.get("content") or "").lower()
                    for hint in attribute.content_hints
                )
            )
            if not found:
                log.warning(
                    "attribute_extraction_gap",
                    attribute=attribute.name,
                    employee_id=employee_id,
                    document=document,
                )
