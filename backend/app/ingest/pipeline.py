from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.adapters.documents.docx_parser import parse_docx
from app.adapters.documents.pdf_parser import parse_pdf
from app.adapters.documents.section_splitter import split_sections
from app.adapters.documents.semantic_chunker import chunk_sections
from app.ingest.metadata import build_chunk_metadata
from app.ports.embeddings import EmbeddingClient
from app.ports.vector_store import VectorStore


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
        contents = [c for _, c, _ in pieces]
        vectors = await self._embeddings.embed_many(contents) if contents else []

        # Pad/truncate vectors to store dims if using fake 64-d vs 1536 schema — memory store OK
        chunks = []
        for (section, content, idx), emb in zip(pieces, vectors):
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
                    ),
                }
            )
        await self._vector_store.delete_by_employee(employee_id)
        if chunks:
            await self._vector_store.upsert(chunks=chunks)
        return len(chunks)
