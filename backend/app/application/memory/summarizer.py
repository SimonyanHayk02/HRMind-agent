from __future__ import annotations

import re
from pathlib import Path

from app.domain.session import ChatMessage, SessionMemory
from app.ports.llm import LLMClient

_NAME_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)


class Summarizer:
    """Prefer extractive session digests; LLM only as a last resort."""

    def __init__(self, llm: LLMClient, prompts_dir: Path | None = None) -> None:
        self._llm = llm
        self._prompts_dir = prompts_dir or Path(__file__).resolve().parents[2] / "prompts"

    async def summarize(self, previous: str, messages: list[ChatMessage]) -> str:
        extractive = self._extractive(previous, messages)
        if extractive:
            return extractive
        path = self._prompts_dir / "summary.md"
        system = path.read_text() if path.exists() else (
            "Summarize the conversation as bullet facts only. "
            "Never invent employees, counts, salaries, or resume details. "
            "If unsure, omit."
        )
        body = previous + "\n" + "\n".join(f"{m.role}: {m.content}" for m in messages)
        return await self._llm.complete(system=system, user=body, temperature=0.0)

    def summarize_from_memory(self, memory: SessionMemory) -> str:
        """Deterministic digest from structured memory (preferred)."""
        bits: list[str] = []
        if memory.active_referent and memory.active_referent.ids:
            label = memory.active_referent.label or "prior set"
            bits.append(
                f"Active set: {label} ({len(memory.active_referent.ids)} people)."
            )
        elif memory.last_employee_ids:
            bits.append(f"Last cohort size: {len(memory.last_employee_ids)}.")
        if memory.last_listed:
            names = ", ".join(
                e.display_name for e in memory.last_listed[:8] if e.display_name
            )
            if names:
                bits.append(f"Last listed: {names}.")
        if memory.entity_memory:
            focus = memory.entity_memory[-1].display_name
            if focus:
                bits.append(f"Recent person focus: {focus}.")
        if memory.constraint_memory:
            cons = ", ".join(
                f"{c.field}={c.value}"
                for c in memory.constraint_memory[:6]
                if c.field != "_universe"
            )
            if cons:
                bits.append(f"Constraints: {cons}.")
        return " ".join(bits)

    def _extractive(self, previous: str, messages: list[ChatMessage]) -> str:
        names: list[str] = []
        seen: set[str] = set()
        for msg in messages:
            if msg.role != "assistant":
                continue
            for m in _NAME_RE.finditer(msg.content or ""):
                name = m.group(1)
                key = name.lower()
                if key in seen or len(name) < 3:
                    continue
                # Skip common sentence starters
                if name.lower() in {
                    "the",
                    "this",
                    "that",
                    "here",
                    "none",
                    "please",
                    "sorry",
                }:
                    continue
                seen.add(key)
                names.append(name)
                if len(names) >= 8:
                    break
        parts: list[str] = []
        if previous.strip():
            parts.append(previous.strip()[:400])
        if names:
            parts.append("Mentioned: " + ", ".join(names) + ".")
        user_asks = [
            m.content.strip()[:120]
            for m in messages
            if m.role == "user" and (m.content or "").strip()
        ][-3:]
        if user_asks:
            parts.append("Recent asks: " + " | ".join(user_asks))
        return " ".join(parts).strip()
