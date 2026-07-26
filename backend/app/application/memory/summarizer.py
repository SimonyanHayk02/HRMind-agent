from __future__ import annotations

from pathlib import Path

from app.domain.session import ChatMessage
from app.ports.llm import LLMClient


class Summarizer:
    def __init__(self, llm: LLMClient, prompts_dir: Path | None = None) -> None:
        self._llm = llm
        self._prompts_dir = prompts_dir or Path(__file__).resolve().parents[2] / "prompts"

    async def summarize(self, previous: str, messages: list[ChatMessage]) -> str:
        path = self._prompts_dir / "summary.md"
        system = path.read_text() if path.exists() else (
            "Summarize the conversation. Never include raw resume chunk text or salaries."
        )
        body = previous + "\n" + "\n".join(f"{m.role}: {m.content}" for m in messages)
        return await self._llm.complete(system=system, user=body, temperature=0.0)
