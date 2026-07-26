"""Initial employees, resumes, resume_chunks + RLS."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "employees",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("first_name", sa.String(100), nullable=False),
        sa.Column("last_name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("department", sa.String(120), nullable=False),
        sa.Column("position", sa.String(120), nullable=False),
        sa.Column("salary", sa.Numeric(12, 2), nullable=False),
        sa.Column("hire_date", sa.Date(), nullable=False),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("city", sa.String(100), nullable=False),
        sa.Column("manager_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("employees.id"), nullable=True),
        sa.Column("education", sa.Text(), nullable=True),
        sa.Column("employment_status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_employees_tenant_id", "employees", ["tenant_id"])
    op.create_index("ix_employees_department", "employees", ["department"])

    op.create_table(
        "resumes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "employee_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("employees.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("storage_path", sa.String(512), nullable=False),
        sa.Column("content_type", sa.String(32), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("parse_version", sa.String(32), nullable=False, server_default="1"),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("employee_id", name="uq_resumes_employee_id"),
    )

    op.create_table(
        "resume_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "employee_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("employees.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "resume_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("resumes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("section", sa.String(64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_resume_chunks_employee_id", "resume_chunks", ["employee_id"])

    op.execute("ALTER TABLE employees ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE employees FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY employees_tenant_isolation ON employees
        USING (
          tenant_id::text = current_setting('app.tenant_id', true)
          AND (
            current_setting('app.role', true) = 'recruiter'
            OR (
              current_setting('app.role', true) = 'manager'
              AND department = current_setting('app.department_id', true)
            )
            OR (
              current_setting('app.role', true) = 'employee'
              AND id::text = current_setting('app.employee_id', true)
            )
            OR COALESCE(current_setting('app.role', true), '') = ''
          )
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS employees_tenant_isolation ON employees")
    op.drop_table("resume_chunks")
    op.drop_table("resumes")
    op.drop_table("employees")
