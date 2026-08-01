"""Move city/country to resumes; add employees.status boolean."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002_location_status"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Expand resumes with nullable location columns
    op.add_column("resumes", sa.Column("city", sa.String(100), nullable=True))
    op.add_column("resumes", sa.Column("country", sa.String(100), nullable=True))

    # 2. Fail loudly if any employee lacks a resume (1:1 required for location ownership)
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM employees e
            LEFT JOIN resumes r ON r.employee_id = e.id
            WHERE r.id IS NULL
          ) THEN
            RAISE EXCEPTION
              'Migration 002_location_status aborted: employees without resumes exist. '
              'Generate resumes before upgrading.';
          END IF;
        END $$;
        """
    )

    # 3. Backfill location from employees onto resumes
    op.execute(
        """
        UPDATE resumes AS r
        SET
          city = e.city,
          country = e.country
        FROM employees AS e
        WHERE r.employee_id = e.id
        """
    )

    # 4. Tighten resumes location to NOT NULL
    op.alter_column("resumes", "city", existing_type=sa.String(100), nullable=False)
    op.alter_column("resumes", "country", existing_type=sa.String(100), nullable=False)

    # 5. Employee boolean status flag (default false)
    op.add_column(
        "employees",
        sa.Column(
            "status",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    # 6. Drop location from employees
    op.drop_column("employees", "city")
    op.drop_column("employees", "country")


def downgrade() -> None:
    op.add_column("employees", sa.Column("country", sa.String(100), nullable=True))
    op.add_column("employees", sa.Column("city", sa.String(100), nullable=True))

    op.execute(
        """
        UPDATE employees AS e
        SET
          city = r.city,
          country = r.country
        FROM resumes AS r
        WHERE r.employee_id = e.id
        """
    )

    op.alter_column("employees", "city", existing_type=sa.String(100), nullable=False)
    op.alter_column("employees", "country", existing_type=sa.String(100), nullable=False)

    op.drop_column("employees", "status")
    op.drop_column("resumes", "city")
    op.drop_column("resumes", "country")
