"""Salary / compensation asks — column ACL before nl2sql or LLM planning.

Salary stays out of ``UNSUPPORTED_TOPIC_RE`` (recruiters may answer). Bonus,
payroll, and benefits remain out-of-scope via that regex.
"""
from __future__ import annotations

import re

from app.application.planning.plan_schema import ExecutionPlan
from app.application.planning.unsupported import is_unsupported_topic
from app.application.response.refusal import unauthorized_plan
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.session import SessionMemory

# Base pay / salary — not benefits/payroll (those stay OOS).
_SALARY_RE = re.compile(
    r"\b("
    r"salary|salaries|base\s*pay|"
    r"how\s+much\s+(?:do|does|is|are)\s+.+\s+(?:make|earn|get\s+paid|paid)|"
    r"(?:make|earn|paid)\s+(?:a\s+)?(?:year|annually)|"
    r"compensation(?!\s+package)|"
    r"what(?:'s|\s+is)\s+.+\s+(?:salary|pay)\b"
    r")",
    re.I,
)


def is_salary_question(question: str) -> bool:
    q = question or ""
    if not q.strip():
        return False
    # Payroll / bonus / benefits are out-of-scope, not salary ACL.
    if is_unsupported_topic(q):
        return False
    return bool(_SALARY_RE.search(q))


def salary_acl_plan(
    question: str,
    *,
    auth: AuthContext,
    memory: SessionMemory | None = None,
) -> ExecutionPlan | None:
    """Return an unauthorized empty plan when the role may not see salary.

    Recruiters always proceed. Employees are soft-refused here (they cannot use
    ``sql``). Managers proceed and rely on column policy at SQL time; known
    cross-department subjects are refused early when we can detect them.
    """
    del memory  # reserved when EntityRef carries department
    if not is_salary_question(question):
        return None

    if auth.role == Role.RECRUITER:
        return None

    if auth.role == Role.EMPLOYEE:
        return unauthorized_plan(
            "You don't have access to salary information. "
            "I can help with profiles, departments, locations, and skills instead."
        )

    # Managers: allow planning; SQL column ACL enforces same-department.
    return None
