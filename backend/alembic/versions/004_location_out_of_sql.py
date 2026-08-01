"""Drop resumes.city and resumes.country — location is resume-text only.

Revision ID: 004_location_out_of_sql
Revises: 003_retrieval_indexes
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "004_location_out_of_sql"
down_revision = "003_retrieval_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("resumes", "city")
    op.drop_column("resumes", "country")


def downgrade() -> None:
    # Values cannot be restored: the resume document is the only source.
    op.add_column("resumes", sa.Column("city", sa.String(100), nullable=True))
    op.add_column("resumes", sa.Column("country", sa.String(100), nullable=True))
