"""Retrieval indexes: HNSW vectors, full-text content, trigram names.

Notes for anyone porting this to a large table:

* ``CREATE INDEX CONCURRENTLY`` cannot run inside Alembic's transaction. At this
  corpus size (hundreds of chunks) a plain build takes milliseconds, so the lock
  is irrelevant; on a large table, run the concurrent variant outside migrations.
* Filtered vector search does not simply narrow the candidate set. Postgres
  applies the filter while traversing the HNSW graph, so a selective ``WHERE``
  can return fewer rows than requested. ``hnsw.iterative_scan`` makes the scan
  continue until the limit is satisfied; it is set as a database default so every
  connection inherits it without per-query GUC juggling (custom GUCs are not
  registered until the extension's library loads in a session).
* The tuning statements are optional: a role without ALTER DATABASE rights, or a
  pgvector older than 0.8, leaves the defaults in place instead of failing.
"""

from __future__ import annotations

from alembic import op

revision = "003_retrieval_indexes"
down_revision = "002_location_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # Dense arm: cosine HNSW, matching the `embedding <=> query` operator used by
    # PgVectorStore.similarity_search.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_resume_chunks_embedding_hnsw
        ON resume_chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    # Sparse arm: full-text over chunk content. The 'simple' configuration is
    # immutable, which is what allows an expression index here.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_resume_chunks_content_fts
        ON resume_chunks USING gin (to_tsvector('simple', content))
        """
    )

    # Name arm: trigram similarity for typo-tolerant person lookup.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_resume_chunks_employee_name_trgm
        ON resume_chunks USING gin ((metadata->>'employee_name') gin_trgm_ops)
        """
    )

    # Attribute retrieval reads whole sections ("Personal") rather than a top-k.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_resume_chunks_section
        ON resume_chunks (section)
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
          EXECUTE format(
            'ALTER DATABASE %I SET hnsw.iterative_scan = %L',
            current_database(), 'relaxed_order'
          );
          EXECUTE format(
            'ALTER DATABASE %I SET hnsw.ef_search = %s',
            current_database(), 100
          );
        EXCEPTION WHEN OTHERS THEN
          RAISE NOTICE 'skipped hnsw tuning (%): defaults remain in effect', SQLERRM;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
          EXECUTE format('ALTER DATABASE %I RESET hnsw.iterative_scan', current_database());
          EXECUTE format('ALTER DATABASE %I RESET hnsw.ef_search', current_database());
        EXCEPTION WHEN OTHERS THEN
          RAISE NOTICE 'skipped hnsw reset (%)', SQLERRM;
        END $$;
        """
    )
    op.execute("DROP INDEX IF EXISTS ix_resume_chunks_section")
    op.execute("DROP INDEX IF EXISTS ix_resume_chunks_employee_name_trgm")
    op.execute("DROP INDEX IF EXISTS ix_resume_chunks_content_fts")
    op.execute("DROP INDEX IF EXISTS ix_resume_chunks_embedding_hnsw")
