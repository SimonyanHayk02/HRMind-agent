"""Residual LLM slot extraction — runs only after regex/heuristics miss."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.application.memory.context_view import ContextNeed
from app.application.understanding.slot_bundle import SlotBundle
from app.application.understanding.turn_digest import build_turn_digest
from app.config.logging import get_logger
from app.domain.session import SessionMemory
from app.ports.llm import LLMClient

logger = get_logger(__name__)

_DEFAULT_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


class LlmSlotExtractor:
    def __init__(
        self,
        llm: LLMClient,
        *,
        prompts_dir: Path | None = None,
    ) -> None:
        self._llm = llm
        self._prompts_dir = prompts_dir or _DEFAULT_PROMPTS

    def _system_prompt(self) -> str:
        path = self._prompts_dir / "nlu_slots.md"
        if path.exists():
            return path.read_text()
        return (
            "Extract HR dialogue slots as JSON with keys intent, attribute, "
            "person_ref, confidence. Never invent SQL."
        )

    async def extract(
        self,
        question: str,
        memory: SessionMemory | None,
        *,
        context_need: ContextNeed | str | None = None,
    ) -> SlotBundle | None:
        digest = build_turn_digest(question, memory, context_need=context_need)
        user = json.dumps(digest, default=str)
        try:
            raw = await self._llm.complete(
                system=self._system_prompt(),
                user=user,
                temperature=0.0,
                response_json=True,
            )
            data = _parse_json_object(raw)
            if data is None:
                logger.warning("llm_slots_invalid_json", question=question[:160])
                return None
            bundle = SlotBundle.model_validate(data)
            logger.info(
                "llm_slots_extracted",
                intent=bundle.intent,
                attribute=bundle.attribute,
                confidence=bundle.confidence,
                person_kind=bundle.person_ref.kind,
                context_need=digest.get("context_need"),
            )
            return bundle
        except (ValidationError, TypeError, ValueError) as exc:
            logger.warning(
                "llm_slots_validation_failed",
                question=question[:160],
                error_type=type(exc).__name__,
                error=str(exc)[:300],
            )
            return None


def _parse_json_object(raw: str | None) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
