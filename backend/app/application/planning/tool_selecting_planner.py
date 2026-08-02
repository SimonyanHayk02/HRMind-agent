"""Confidence-scored LLM tool selector → ExecutionPlan."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.application.planning.intent_templates import expand_intent, expand_selected_tools
from app.application.planning.plan_schema import ExecutionPlan
from app.application.response.refusal import RefusalCode
from app.config.logging import get_logger
from app.domain.auth import AuthContext
from app.domain.session import SessionMemory
from app.domain.tools.registry import ToolRegistry
from app.ports.llm import LLMClient

logger = get_logger(__name__)

CONFIDENCE_CLARIFY = 0.55
CONFIDENCE_WRITE = 0.75
MAX_REPAIRS = 2


class ToolSelection(BaseModel):
    confidence: float = 0.0
    clarify_question: str | None = None
    intent: str = "unknown"
    slots: dict[str, Any] = Field(default_factory=dict)
    selected: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str | None = None


@dataclass
class ToolSelectResult:
    plan: ExecutionPlan
    mode: str
    selection: ToolSelection | None = None
    repairs: int = 0
    needs_hitl: bool = False


class ToolSelectingPlanner:
    """LLM chooses intent/tools with confidence; expands into a validated DAG."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        prompts_dir: Path | None = None,
        confidence_clarify: float = CONFIDENCE_CLARIFY,
        confidence_write: float = CONFIDENCE_WRITE,
        max_repairs: int = MAX_REPAIRS,
        hitl_status_writes: bool = True,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._prompts_dir = (
            prompts_dir or Path(__file__).resolve().parents[2] / "prompts"
        )
        self._confidence_clarify = confidence_clarify
        self._confidence_write = confidence_write
        self._max_repairs = max_repairs
        self._hitl_status_writes = hitl_status_writes

    def _load_prompt(self) -> str:
        path = self._prompts_dir / "tool_selector.md"
        if path.exists():
            return path.read_text()
        return (
            "Select HR tools as JSON with confidence, intent, slots, selected."
        )

    def _packet(
        self,
        question: str,
        *,
        auth: AuthContext,
        memory: SessionMemory | None,
        repair_feedback: str | None = None,
    ) -> dict[str, Any]:
        prior_ids: list[str] = []
        listed: list[dict[str, str]] = []
        if memory:
            prior_ids = [str(x) for x in (memory.last_employee_ids or [])[:40]]
            listed = [
                {
                    "employee_id": str(e.employee_id),
                    "display_name": e.display_name or "",
                }
                for e in (memory.last_listed or [])[:20]
            ]
        packet: dict[str, Any] = {
            "question": question,
            "role": auth.role.value,
            "tools": self._tools.discover(),
            "prior_employee_ids": prior_ids,
            "last_listed": listed,
            "person_bindings": dict(memory.person_bindings) if memory else {},
        }
        if repair_feedback:
            packet["repair_feedback"] = repair_feedback
        return packet

    async def _select_once(
        self,
        question: str,
        *,
        auth: AuthContext,
        memory: SessionMemory | None,
        repair_feedback: str | None = None,
    ) -> ToolSelection:
        packet = self._packet(
            question, auth=auth, memory=memory, repair_feedback=repair_feedback
        )
        raw = await self._llm.complete(
            system=self._load_prompt(),
            user=json.dumps(packet, default=str),
            temperature=0.0,
            response_json=True,
        )
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        return ToolSelection.model_validate(data)

    def _clarify(self, message: str) -> ExecutionPlan:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=message,
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )

    def _compile_selection(
        self,
        selection: ToolSelection,
        *,
        question: str,
        memory: SessionMemory | None,
    ) -> tuple[ExecutionPlan | None, str | None]:
        """Return (plan, error). error set when compilation refused."""
        conf = float(selection.confidence or 0.0)
        intent = (selection.intent or "unknown").lower()
        slots = dict(selection.slots or {})
        if selection.clarify_question and not slots.get("clarify_question"):
            slots["clarify_question"] = selection.clarify_question

        # Normalize status polarity for writes (models often emit "active"/strings).
        if intent == "set_status" or any(
            (s.get("params") or {}).get("action") == "set_status"
            for s in selection.selected
        ):
            coerced = self._coerce_status(slots.get("status"))
            if coerced is None:
                coerced = self._coerce_status(slots.get("status_value"))
            q_l = (question or "").lower()
            if coerced is None and "activate" in q_l:
                coerced = True
            elif coerced is None and any(
                w in q_l for w in ("deactivate", "disable", "inactive")
            ):
                coerced = False
            if coerced is not None:
                slots["status"] = coerced

        if conf < self._confidence_clarify or intent == "clarify":
            msg = (
                selection.clarify_question
                or slots.get("clarify_question")
                or "Could you clarify who or what you mean?"
            )
            return self._clarify(str(msg)), None

        if intent == "set_status" and conf < self._confidence_write:
            return (
                self._clarify(
                    "I want to be sure before changing status. "
                    "Please confirm the employee and whether status should be true or false."
                ),
                None,
            )

        # Block writes in the medium-confidence band even if mislabeled.
        write_tools = {
            str(s.get("tool") or s.get("name") or "")
            for s in selection.selected
        }
        if (
            conf < self._confidence_write
            and (intent == "set_status" or "employee" in write_tools)
            and any(
                (s.get("params") or {}).get("action") == "set_status"
                for s in selection.selected
            )
        ):
            return (
                self._clarify(
                    "Please confirm the status update (employee + true/false) before I apply it."
                ),
                None,
            )

        plan = expand_intent(
            intent, question=question, slots=slots, memory=memory
        )
        if plan is None and selection.selected:
            plan = expand_selected_tools(selection.selected, question=question)
        if plan is None:
            return None, "could_not_expand_selection"
        return plan, None

    def _needs_hitl(self, selection: ToolSelection, plan: ExecutionPlan) -> bool:
        if not self._hitl_status_writes:
            return False
        if (selection.intent or "").lower() == "set_status":
            # Location-cohort / multi-id writes already use pending confirm elsewhere;
            # single-person writes also pause when HITL is on.
            return True
        return any(
            n.name == "employee"
            and (n.params or {}).get("action") == "set_status"
            and not (n.params or {}).get("confirmed")
            for n in plan.nodes
        )

    @staticmethod
    def _coerce_status(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"true", "yes", "on", "1", "active", "enable", "enabled"}:
                return True
            if key in {"false", "no", "off", "0", "inactive", "disable", "disabled"}:
                return False
        return None

    def _hitl_plan(self, selection: ToolSelection, plan: ExecutionPlan) -> ExecutionPlan:
        """Convert a set_status plan into a confirm clarify + pending mutation fact."""
        slots = selection.slots or {}
        name = slots.get("name") or slots.get("person_name") or "that employee"
        status = self._coerce_status(slots.get("status"))
        if status is None:
            status = self._coerce_status(slots.get("status_value"))
        # Infer polarity from common verbs when the model omitted status.
        if status is None:
            rationale = f"{selection.rationale or ''} {selection.intent or ''}".lower()
            q = json.dumps(slots).lower()
            if "activate" in rationale or "activate" in q or "enable" in q:
                status = True
            elif "deactivate" in rationale or "disable" in q:
                status = False
        ids = [str(x) for x in (slots.get("employee_ids") or []) if x]
        # Pull ids from the compiled write plan when slots omitted them.
        if not ids:
            for node in plan.nodes:
                if node.name != "employee":
                    continue
                params = node.params or {}
                if params.get("employee_id"):
                    ids = [str(params["employee_id"])]
                elif params.get("employee_ids"):
                    ids = [str(x) for x in params["employee_ids"] if x]
                if status is None and isinstance(params.get("status"), bool):
                    status = params["status"]
        # Named single-person HITL: never stash prior-focus ids. Confirm must
        # re-resolve the name the user saw in the clarify copy (Carol ≠ Alice).
        explicit_name = bool(name and str(name).strip().lower() not in {"", "that employee"})
        if explicit_name:
            ids = []
        label = (
            "true/active"
            if status is True
            else "false/inactive"
            if status is False
            else "the requested value"
        )
        pending: dict[str, Any] | None = None
        if isinstance(status, bool) and (ids or explicit_name):
            pending = {
                "key": "pending_status_mutation",
                "value": {
                    "kind": "set_status",
                    "status": status,
                    "employee_ids": ids,
                    "name": name if explicit_name else None,
                    "resolve_via": "tool_select_hitl",
                },
            }
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                f"Confirm status update: set {name}'s status to {label}? "
                "Reply yes to confirm or no to cancel."
            ),
            refusal_code=RefusalCode.AMBIGUOUS.value,
            pending_tool_fact=pending,
        )

    async def plan(
        self,
        question: str,
        *,
        auth: AuthContext,
        memory: SessionMemory | None = None,
        validate: Any | None = None,
        force_repair: bool = False,
    ) -> ToolSelectResult:
        """Select tools, apply confidence policy, optionally repair up to max_repairs."""
        repair_feedback: str | None = None
        last_selection: ToolSelection | None = None
        for attempt in range(self._max_repairs + 1):
            try:
                selection = await self._select_once(
                    question,
                    auth=auth,
                    memory=memory,
                    repair_feedback=repair_feedback,
                )
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                logger.warning(
                    "tool_selector_invalid_json",
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error=str(exc)[:300],
                )
                repair_feedback = f"Invalid JSON ({type(exc).__name__}): {exc}"
                last_selection = None
                continue

            last_selection = selection
            plan, err = self._compile_selection(
                selection, question=question, memory=memory
            )
            if err or plan is None:
                repair_feedback = (
                    f"Could not expand intent={selection.intent!r} / selected="
                    f"{selection.selected!r}. Pick a listed intent or valid tools."
                )
                continue

            # Keep selection.slots aligned with coerced values used for expansion/HITL.
            if isinstance(selection.slots.get("status"), bool) is False:
                coerced = self._coerce_status(selection.slots.get("status"))
                if coerced is None:
                    coerced = self._coerce_status(selection.slots.get("status_value"))
                q_l = (question or "").lower()
                if coerced is None and "activate" in q_l:
                    coerced = True
                elif coerced is None and any(
                    w in q_l for w in ("deactivate", "disable", "inactive")
                ):
                    coerced = False
                if coerced is not None:
                    selection.slots["status"] = coerced

            if validate is not None:
                try:
                    validate(plan, auth)
                except Exception as exc:  # noqa: BLE001 — feed into repair
                    repair_feedback = f"Plan validation failed: {exc}"
                    logger.info(
                        "tool_selector_validate_repair",
                        attempt=attempt,
                        error=str(exc)[:300],
                    )
                    continue

            # Dev/QA probe: discard the first successful expansion so attempt 1 repairs.
            if force_repair and attempt == 0:
                repair_feedback = (
                    "Forced QA repair probe: re-select with the same intent once."
                )
                logger.info("tool_selector_force_repair", attempt=attempt)
                continue

            needs_hitl = self._needs_hitl(selection, plan)
            if needs_hitl:
                # Return clarify plan but signal HITL so caller can stash pending mutation.
                return ToolSelectResult(
                    plan=self._hitl_plan(selection, plan),
                    mode="tool_select_hitl",
                    selection=selection,
                    repairs=attempt,
                    needs_hitl=True,
                )

            return ToolSelectResult(
                plan=plan,
                mode="tool_select" if attempt == 0 else "tool_select_repair",
                selection=selection,
                repairs=attempt,
                needs_hitl=False,
            )

        msg = (
            (last_selection.clarify_question if last_selection else None)
            or "I couldn't build a reliable plan for that. "
            "Try asking with a department, country, city, person name, or skill."
        )
        return ToolSelectResult(
            plan=self._clarify(str(msg)),
            mode="tool_select_failed",
            selection=last_selection,
            repairs=self._max_repairs,
            needs_hitl=False,
        )
