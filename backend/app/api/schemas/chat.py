from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.tools.base import SourceRef


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    confidence: float = 1.0
    sources: list[SourceRef] = Field(default_factory=list)
    clarify: str | None = None
    # Which tool answered: "sql", "resume_search", "employee", "greeting", "clarify".
    # Hybrid plans join their tools with "+", e.g. "sql+resume_search". None when no
    # tool produced anything, such as a clarify-only turn or a total failure.
    tool: str | None = None
    trace_id: str | None = None
    degraded: bool = False
