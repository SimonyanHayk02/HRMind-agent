from enum import StrEnum


class Role(StrEnum):
    RECRUITER = "recruiter"
    MANAGER = "manager"
    EMPLOYEE = "employee"


class EmploymentStatus(StrEnum):
    ACTIVE = "active"
    LEAVE = "leave"
    TERMINATED = "terminated"


class ResumeStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class PlanNodeKind(StrEnum):
    TOOL = "tool"
    OPERATOR = "operator"


class ResponseStrategy(StrEnum):
    TEMPLATE = "template"
    LLM_FORMAT = "llm_format"


class RouterLabel(StrEnum):
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    NEEDS_TOOLS = "needs_tools"


class SqlMode(StrEnum):
    NL2SQL = "nl2sql"
    CONSTRAINED = "constrained"
