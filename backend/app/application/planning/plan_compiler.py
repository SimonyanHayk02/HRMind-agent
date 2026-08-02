from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from app.application.memory.context_view import (
    ContextNeed,
    classify_context_need,
    lint_fresh_scope,
    minimal_planner_packet,
)
from app.application.planning.bound_person import bound_person_attribute_plan
from app.application.planning.heuristic_planner import (
    _ABOUT_PERSON_RE,
    _FACET_COUNT_RE,
    _FACET_LIST_RE,
    _LOCATION_TOPIC_RE,
    _META_COUNT_RE,
    _ORG_HEADCOUNT_RE,
    _PERSON_LOCATION_RE,
    _PRONOUN_ONLY_RE,
    _SKILL_RE,
    _detect_city,
    _detect_country,
    _detect_facet_dimension,
    _department_mentioned,
    _employee_by_id_plan,
    _employee_by_name_plan,
    _entity_display_name,
    _facet_count_plan,
    _facet_list_plan,
    _JOINED_RE,
    _filtered_count_plan,
    _hire_date_filters,
    _hire_date_list_plan,
    _longest_tenured_plan,
    _meta_count_plan,
    _pronoun_clarify_plan,
    _resolve_pronoun_employee_id,
    _sql_over_ids,
    _tenure_agg_plan,
    _unknown_place_phrase,
    is_list_followup,
    refers_to_prior_set,
    try_cohort_status_read_plan,
    try_heuristic_plan,
    try_status_write_plan,
    wants_person_lookup,
)
from app.application.planning.location_plans import (
    location_cohort_plan,
    location_person_plan,
)
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.salary_acl import salary_acl_plan
from app.application.planning.unsupported import is_unsupported_topic
from app.application.response.refusal import RefusalCode, out_of_scope_plan
from app.application.schema.catalog_service import CatalogService
from app.application.understanding.birthday import extract_birthday
from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.list_referents import resolve_list_referent
from app.application.understanding.llm_slot_extractor import LlmSlotExtractor
from app.application.understanding.plan_from_query_state import (
    _set_status_plan,
    plan_from_query_state,
)
from app.application.understanding.plan_from_slots import plan_from_slots
from app.application.understanding.status_change import (
    PENDING_SET_STATUS_KEY,
    PENDING_STATUS_MUTATION_KEY,
    extract_status_change,
    is_cancel_status_update,
    is_confirm_status_update,
    looks_like_bare_person_name,
)
from app.config.logging import get_logger
from app.domain.auth import AuthContext
from app.domain.query_state import QueryState
from app.domain.schema_catalog import default_employee_catalog
from app.domain.session import SessionMemory
from app.domain.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.application.memory.context_manager import ContextManager

logger = get_logger(__name__)

_CLARIFY_FALLBACK = (
    "I couldn't build a reliable plan for that. "
    "Try asking with a department, country, city, or job title "
    "(for example: list Software Engineers in Engineering)."
)


class PlanCompiler:
    def __init__(
        self,
        llm,
        tools: ToolRegistry,
        prompts_dir: Path | None = None,
        *,
        max_context: int = 8,
        catalog_service: CatalogService | None = None,
        tool_selecting_mode: str = "primary",
        tool_selecting_confidence_clarify: float = 0.55,
        tool_selecting_confidence_write: float = 0.75,
        tool_selecting_max_repairs: int = 2,
        tool_selecting_hitl_status: bool = True,
        tool_selecting_fallback_legacy: bool = True,
        plan_validator: object | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._prompts_dir = prompts_dir or Path(__file__).resolve().parents[2] / "prompts"
        self._max_context = max_context
        self._catalog = catalog_service
        self._slot_extractor = LlmSlotExtractor(llm, prompts_dir=self._prompts_dir)
        self._tool_selecting_mode = (tool_selecting_mode or "off").strip().lower()
        self._tool_selecting_fallback_legacy = tool_selecting_fallback_legacy
        self._plan_validator = plan_validator
        from app.application.planning.tool_selecting_planner import ToolSelectingPlanner

        self._tool_selector = ToolSelectingPlanner(
            llm,
            tools,
            prompts_dir=self._prompts_dir,
            confidence_clarify=tool_selecting_confidence_clarify,
            confidence_write=tool_selecting_confidence_write,
            max_repairs=tool_selecting_max_repairs,
            hitl_status_writes=tool_selecting_hitl_status,
        )

    def _load_prompt(self) -> str:
        path = self._prompts_dir / "planner.md"
        if path.exists():
            return path.read_text()
        return (
            "You are an HR agent planner. Return JSON ExecutionPlan with nodes "
            "(tool|operator), depends_on, response_strategy."
        )

    def _catalog_view(self) -> dict:
        if self._catalog is not None:
            return self._catalog.planner_view()
        return default_employee_catalog().planner_view()

    def _context_packet(
        self,
        question: str,
        auth: AuthContext,
        memory: SessionMemory | None,
        query_state: QueryState | None = None,
        context_manager: ContextManager | None = None,
        *,
        context_need: str | None = None,
    ) -> dict:
        if context_manager is not None and memory is not None:
            return context_manager.build_for_planner(
                question=question,
                auth=auth,
                memory=memory,
                tools=self._tools.discover(),
                schema_catalog=self._catalog_view(),
                query_state=query_state,
                context_need=context_need,
            )
        # Fallback (tests without ContextManager)
        from app.application.memory.context_budget import build_planner_packet

        return build_planner_packet(
            question=question,
            auth=auth,
            memory=memory,
            tools=self._tools.discover(),
            schema_catalog=self._catalog_view(),
            query_state=query_state.model_dump(mode="json") if query_state else None,
            max_context=self._max_context,
            context_need=context_need,
        )

    async def _llm_plan_once(
        self,
        question: str,
        *,
        packet: dict,
        context_need: ContextNeed,
    ) -> ExecutionPlan:
        """Call planner LLM once; raise on invalid JSON/schema/fresh-scope lint."""
        user = json.dumps(packet, default=str)
        raw = await self._llm.complete(
            system=self._load_prompt(), user=user, temperature=0.0, response_json=True
        )
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        plan = ExecutionPlan.model_validate(data)
        lint_err = lint_fresh_scope(plan, context_need, question)
        if lint_err:
            raise ValueError(f"fresh_scope_lint: {lint_err}")
        return plan

    async def compile(
        self,
        question: str,
        *,
        auth: AuthContext,
        memory: SessionMemory | None = None,
        context_manager: ContextManager | None = None,
        force_repair: bool = False,
    ) -> tuple[ExecutionPlan, str, dict]:
        """Compile a plan. Returns (plan, planner_mode, meta)."""
        self._force_repair = bool(force_repair)
        self._select_meta: dict = {}
        plan, mode = await self._compile_pair(
            question,
            auth=auth,
            memory=memory,
            context_manager=context_manager,
        )
        meta = {
            "planner_mode": mode,
            "plan_nodes": [n.name for n in plan.nodes],
            "repairs": int(self._select_meta.get("repairs", 0) or 0),
            "needs_hitl": bool(self._select_meta.get("needs_hitl", False)),
            "select_intent": self._select_meta.get("select_intent"),
            "select_confidence": self._select_meta.get("select_confidence"),
        }
        return plan, mode, meta

    async def _compile_pair(
        self,
        question: str,
        *,
        auth: AuthContext,
        memory: SessionMemory | None = None,
        context_manager: ContextManager | None = None,
    ) -> tuple[ExecutionPlan, str]:
        """Compile a plan. Returns (plan, planner_mode)."""
        q = (question or "").strip()

        # Deterministic escapes that must beat QueryState / nl2sql.
        if _META_COUNT_RE.search(q):
            plan = _meta_count_plan(memory)
            if plan is not None:
                return plan, "heuristic_meta_count"

        # Org-wide headcount paraphrases ("how many people in total?") must not
        # collapse to a department-sized count via the LLM selector.
        if _ORG_HEADCOUNT_RE.match(q) or re.fullmatch(
            r"\s*(?:quick\s*[—\-]?\s*)?how many people(?:\s+do\s+we\s+have)?(?:\s+in\s+total)?\s*\??\s*",
            q,
            re.I,
        ):
            return _filtered_count_plan({}), "guard_org_headcount"

        # Explicit "Tell me about First Last" must use that full name — never the
        # prior first-name focus (Alice Bauer stealing Alice Nguyen).
        about = _ABOUT_PERSON_RE.search(q)
        if about:
            about_name = next((g for g in about.groups() if g), None)
            about_name = re.sub(r"'s$", "", str(about_name or "").strip(), flags=re.I)
            about_name = about_name.strip(" '")
            tokens = about_name.split()
            # Title-Case two-token names only — avoid "who is the most senior…".
            if (
                len(tokens) >= 2
                and all(t[:1].isupper() for t in tokens)
                and not re.search(
                    r"\b(most|senior|junior|average|tenure|oldest|longest)\b", q, re.I
                )
            ):
                return (
                    _employee_by_name_plan(about_name, with_location=True),
                    "guard_about_person",
                )

        # Continue a prior location-cohort status write after explicit confirm.
        mutation = (
            memory.tool_fact_cache.get(PENDING_STATUS_MUTATION_KEY) if memory else None
        )
        if mutation is not None and is_cancel_status_update(q):
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question="Cancelled — I won't update anyone's status.",
                    refusal_code=RefusalCode.AMBIGUOUS.value,
                    pending_tool_fact={
                        "key": PENDING_STATUS_MUTATION_KEY,
                        "clear": True,
                    },
                ),
                "heuristic_cancel_status",
            )
        if mutation is not None and is_confirm_status_update(q):
            payload = mutation.value if isinstance(mutation.value, dict) else None
            ids = list((payload or {}).get("employee_ids") or [])
            status = (payload or {}).get("status")
            name = (payload or {}).get("name")
            resolve_via = str((payload or {}).get("resolve_via") or "")
            # Named HITL (tool_select) must re-resolve by name so prior-focus
            # ids cannot overwrite the person shown in the confirm copy.
            if (
                name
                and isinstance(status, bool)
                and (resolve_via == "tool_select_hitl" or not ids)
            ):
                return (
                    _set_status_plan(
                        QueryState(
                            intent="set_status",
                            person_name=str(name),
                            status_value=bool(status),
                            confidence=0.95,
                        ),
                        filters={},
                    ),
                    "heuristic_confirm_status",
                )
            if ids and isinstance(status, bool):
                return (
                    ExecutionPlan(
                        nodes=[
                            PlanNode(
                                id="e1",
                                kind="tool",
                                name="employee",
                                params={
                                    "action": "set_status",
                                    "status": status,
                                    "employee_ids": ids,
                                    "resolve_via": resolve_via or "location",
                                    "confirmed": True,
                                },
                            )
                        ],
                        response_strategy="template",
                        active_cohort_node="e1",
                    ),
                    "heuristic_confirm_status",
                )
            if name and isinstance(status, bool):
                return (
                    _set_status_plan(
                        QueryState(
                            intent="set_status",
                            person_name=str(name),
                            status_value=bool(status),
                            confidence=0.95,
                        ),
                        filters={},
                    ),
                    "heuristic_confirm_status",
                )

        # Also cancel bare pending_set_status (which-employee clarify).
        pending_bool = (
            memory.tool_fact_cache.get(PENDING_SET_STATUS_KEY) if memory else None
        )
        if pending_bool is not None and is_cancel_status_update(q):
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question="Cancelled — I won't update anyone's status.",
                    refusal_code=RefusalCode.AMBIGUOUS.value,
                    pending_tool_fact={
                        "key": PENDING_SET_STATUS_KEY,
                        "clear": True,
                    },
                ),
                "heuristic_cancel_status",
            )

        # Continue a prior "which employee's status?" clarify with a bare name.
        pending = (
            memory.tool_fact_cache.get(PENDING_SET_STATUS_KEY) if memory else None
        )
        if (
            pending is not None
            and isinstance(pending.value, bool)
            and looks_like_bare_person_name(q)
        ):
            return (
                _set_status_plan(
                    QueryState(
                        intent="set_status",
                        person_name=q.strip().strip(" .,?!"),
                        status_value=bool(pending.value),
                        confidence=0.95,
                    ),
                    filters={},
                ),
                "heuristic_pending_set_status",
            )

        # Salary/comp ACL before any SQL / LLM path (employees soft-refuse).
        acl = salary_acl_plan(q, auth=auth, memory=memory)
        if acl is not None:
            return acl, "salary_acl"

        # OOS before pronoun→profile dump ("her travelling preferences", PTO, …).
        if is_unsupported_topic(q):
            from app.application.planning.unsupported import unsupported_answer_for

            return out_of_scope_plan(unsupported_answer_for(q)), "unsupported"

        # Ordinals / list deixis before birthday name extract ("top one's bday").
        list_ref = resolve_list_referent(q, memory)
        if list_ref.kind == "clarify":
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=list_ref.clarify_question,
                    refusal_code=RefusalCode.AMBIGUOUS.value,
                ),
                "heuristic_list_referent",
            )
        if list_ref.kind == "resolved" and list_ref.employee_id:
            return (
                bound_person_attribute_plan(
                    q,
                    employee_id=list_ref.employee_id,
                    display_name=list_ref.display_name or "that employee",
                ),
                "heuristic_list_referent",
            )

        # Seeded status writes (cheap extractor) before LLM tool select.
        # Novel paraphrases miss the extractor and use tool-select HITL instead.
        status_write = try_status_write_plan(q, memory=memory)
        if status_write is not None:
            return status_write, "guard_status"

        # Cohort status *reads* ("statuses of them") — never resume status_resolve.
        status_read = try_cohort_status_read_plan(q, memory=memory)
        if status_read is not None:
            return status_read, "guard_status_read"

        # Birthday cohort / named person before elliptical profile dumps and
        # before pronoun binding steals "Alice — when was she born?".
        birthday_pre = extract_birthday(q)
        if birthday_pre.matched and (
            birthday_pre.scope != "person" or birthday_pre.person_name
        ):
            from app.application.understanding.plan_from_query_state import (
                _birthday_plan,
            )
            from app.domain.query_state import QueryState as _QS

            prior_ids = (
                list(memory.last_employee_ids)
                if memory and memory.last_employee_ids
                else []
            )
            if len(prior_ids) > 50:
                prior_ids = []
            scoped_ids = (
                prior_ids
                if (
                    birthday_pre.scope in {"closest", "upcoming", "today", "month"}
                    and refers_to_prior_set(q)
                    and prior_ids
                )
                else []
            )
            return (
                _birthday_plan(
                    _QS(
                        intent="birthday",
                        birthday_scope=birthday_pre.scope,
                        birthday_month=birthday_pre.month,
                        person_name=birthday_pre.person_name,
                        person_employee_ids=list(scoped_ids),
                        refers_to_prior=bool(scoped_ids),
                        wants_age=birthday_pre.wants_age,
                        wants_wish=birthday_pre.wants_wish,
                        confidence=0.95,
                    )
                ),
                "heuristic_birthday",
            )

        pronoun_id = _resolve_pronoun_employee_id(q, memory)
        if _PRONOUN_ONLY_RE.search(q) and wants_person_lookup(q):
            # Bare pronoun person questions — never treat "she"/"he" as a name.
            has_proper_name = bool(
                re.search(
                    r"\bwhere\s+(?:does|is)\s+(?!she\b|he\b|her\b|him\b)([A-Z][a-z]+)",
                    q,
                )
            )
            if not has_proper_name:
                bound_id = pronoun_id
                if not bound_id and memory and len(memory.entity_memory) == 1:
                    bound_id = str(memory.entity_memory[0].employee_id)
                asks_where = bool(_PERSON_LOCATION_RE.search(q)) or bool(
                    _LOCATION_TOPIC_RE.search(q)
                )
                if asks_where:
                    # Place lives in the resume — never the employee profile row.
                    if not bound_id:
                        return _pronoun_clarify_plan(), "heuristic_pronoun"
                    name = _entity_display_name(memory, bound_id)
                    return (
                        location_person_plan(name, employee_ids=[bound_id]),
                        "heuristic_pronoun",
                    )
                # Birthday / status with a pronoun must bind before profile fallback.
                birthday = extract_birthday(q)
                if birthday.matched and birthday.scope == "person":
                    if not bound_id:
                        return _pronoun_clarify_plan(), "heuristic_pronoun"
                    name = _entity_display_name(memory, bound_id)
                    return (
                        bound_person_attribute_plan(
                            q, employee_id=bound_id, display_name=name
                        ),
                        "heuristic_pronoun",
                    )
                status = extract_status_change(q)
                if status.matched:
                    if not bound_id:
                        return _pronoun_clarify_plan(), "heuristic_pronoun"
                    name = _entity_display_name(memory, bound_id)
                    return (
                        bound_person_attribute_plan(
                            q, employee_id=bound_id, display_name=name
                        ),
                        "heuristic_pronoun",
                    )
                if bound_id:
                    name = _entity_display_name(memory, bound_id)
                    if re.search(r"\bmanagers?\b", q, re.I):
                        from app.application.planning.heuristic_planner import (
                            _manager_plan,
                        )

                        return (
                            _manager_plan(q, name=name, employee_id=bound_id),
                            "heuristic_pronoun",
                        )
                    return (
                        bound_person_attribute_plan(
                            q, employee_id=bound_id, display_name=name
                        ),
                        "heuristic_pronoun",
                    )
                return _pronoun_clarify_plan(), "heuristic_pronoun"

        # Bare elliptical SQL attrs ("what is the education?") about the bound person.
        from app.application.planning.heuristic_planner import (
            elliptical_bound_person_plan,
        )

        elliptical = elliptical_bound_person_plan(q, memory=memory)
        if elliptical is not None:
            return elliptical, "heuristic_elliptical_person"

        # Birthday + pronoun without other profile cues (e.g. "when was she born").
        birthday_early = extract_birthday(q)
        if (
            birthday_early.matched
            and birthday_early.scope == "person"
            and _PRONOUN_ONLY_RE.search(q)
            and not birthday_early.person_name
        ):
            bound_id = pronoun_id
            if not bound_id and memory and len(memory.entity_memory) == 1:
                bound_id = str(memory.entity_memory[0].employee_id)
            if bound_id:
                name = _entity_display_name(memory, bound_id)
                return (
                    bound_person_attribute_plan(
                        q, employee_id=bound_id, display_name=name
                    ),
                    "heuristic_pronoun",
                )
            return _pronoun_clarify_plan(), "heuristic_pronoun"

        catalog = self._catalog.get() if self._catalog else default_employee_catalog()
        query_state = extract_query_state(question, catalog=catalog, memory=memory)

        # --- Primary-mode guards (deterministic paths the LLM often misroutes) ---
        if _FACET_COUNT_RE.search(q):
            dim = _detect_facet_dimension(q)
            if dim:
                return _facet_count_plan(dim), "guard_facet"
        if _FACET_LIST_RE.search(q):
            dim = _detect_facet_dimension(q)
            if dim:
                return _facet_list_plan(dim), "guard_facet"
        focus = memory.last_focus if memory else None
        if (
            is_list_followup(q)
            and focus is not None
            and getattr(focus, "kind", None) == "facet"
            and getattr(focus, "dimension", None)
        ):
            return _facet_list_plan(str(focus.dimension)), "guard_facet_list"
        if is_list_followup(q):
            listed_ids = (
                list(memory.last_employee_ids)
                if memory and memory.last_employee_ids
                else []
            )
            if not listed_ids and memory and memory.last_listed:
                listed_ids = [str(e.employee_id) for e in memory.last_listed if e.employee_id]
            if listed_ids:
                return (
                    _sql_over_ids(listed_ids, count_only=False),
                    "guard_list_followup",
                )
            # Empty cohort after a zero refine — do not fall through to a
            # department/skill roster via query_state ("list their names").
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=(
                        "None of the previous set match that criteria, "
                        "so there are no names to list. "
                        "Ask a new search (for example: who knows Python?)."
                    ),
                    refusal_code=RefusalCode.EMPTY_COHORT.value,
                ),
                "guard_empty_cohort_list",
            )

        # Join/hire windows beat tenure phrasing the LLM sometimes confuses.
        if _JOINED_RE.search(q):
            hire_filters = _hire_date_filters(q)
            if hire_filters:
                if query_state.want_count or bool(
                    re.search(r"\b(how many|count|number of)\b", q, re.I)
                ):
                    return _filtered_count_plan(hire_filters), "guard_hire_window"
                return _hire_date_list_plan(hire_filters), "guard_hire_window"

        if query_state.intent == "tenure_agg":
            return (
                _tenure_agg_plan(department=query_state.filters_dict().get("department")),
                "guard_tenure",
            )
        if query_state.intent == "longest_tenured":
            return (
                _longest_tenured_plan(
                    department=query_state.filters_dict().get("department"),
                    limit=5,
                ),
                "guard_tenure",
            )

        unknown_place = _unknown_place_phrase(q)
        if unknown_place and any(
            w in q.lower()
            for w in ("employee", "people", "staff", "list", "who", "based", "live")
        ):
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=(
                        f'I don\'t recognize "{unknown_place}" as a known city or country. '
                        "Try one of the places we track (for example Berlin, Dubai, or USA)."
                    ),
                    refusal_code=RefusalCode.AMBIGUOUS.value,
                ),
                "guard_unknown_place",
            )

        prior_ids = (
            list(memory.last_employee_ids) if memory and memory.last_employee_ids else []
        )
        # Prior cohort ∩ place ("how many of them are in Berlin?") — LLM often
        # drops the intersect and returns empty; keep the deterministic DAG.
        # Require a multi-person cohort: a single profile id left from "tell me
        # about X" is not a refine cohort (org headcount / place ask stays global).
        place_city = _detect_city(q)
        place_country = _detect_country(q)
        if (
            len(prior_ids) >= 2
            and refers_to_prior_set(q)
            and (place_city or place_country)
            and not _SKILL_RE.search(q)
        ):
            any_of_them = bool(re.search(r"\bany(?:one)?\s+of\s+them\b", q, re.I))
            wants_count = bool(
                re.search(r"\b(how many|how much|count|number of)\b", q, re.I)
            ) or any_of_them
            return (
                location_cohort_plan(
                    city=place_city,
                    country=place_country,
                    count_only=wants_count,
                    intersect_with=prior_ids,
                ),
                "guard_place_refine",
            )
        # Bare "of them?" only — not "names please" (has its own list/facet path).
        if (
            refers_to_prior_set(q)
            and not prior_ids
            and not is_list_followup(q)
            and not place_city
            and not place_country
            and not _SKILL_RE.search(q)
            and not _department_mentioned(q.lower())
        ):
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=(
                        "Which previous list of employees did you mean? "
                        "Ask a search first (for example: who knows Python?), "
                        "then I can answer about them."
                    ),
                    refusal_code=RefusalCode.AMBIGUOUS.value,
                ),
                "guard_empty_anaphora",
            )

        use_primary_selector = self._tool_selecting_mode == "primary"
        use_residual_selector = self._tool_selecting_mode in {"primary", "residual"}

        # Legacy deterministic cascade — skipped in primary mode (guards already ran).
        if not use_primary_selector:
            structured = plan_from_query_state(query_state, memory=memory)
            if structured is None and query_state.filters:
                structured = plan_from_query_state(
                    query_state, memory=memory, min_confidence=0.5
                )
            if structured is not None:
                return structured, "query_state"

            heuristic = try_heuristic_plan(question, memory=memory)
            if heuristic is not None:
                return heuristic, "heuristic"

        async def _run_tool_selector() -> tuple[ExecutionPlan, str] | None:
            if not use_residual_selector:
                return None
            validate = (
                self._plan_validator.validate
                if self._plan_validator is not None
                else None
            )
            result = await self._tool_selector.plan(
                q,
                auth=auth,
                memory=memory,
                validate=validate,
                force_repair=bool(getattr(self, "_force_repair", False)),
            )
            self._select_meta = {
                "repairs": result.repairs,
                "needs_hitl": result.needs_hitl,
                "select_intent": (
                    result.selection.intent if result.selection else None
                ),
                "select_confidence": (
                    float(result.selection.confidence)
                    if result.selection is not None
                    else None
                ),
            }
            logger.info(
                "planner_mode_tool_select",
                mode=result.mode,
                repairs=result.repairs,
                intent=(result.selection.intent if result.selection else None),
                confidence=(
                    result.selection.confidence if result.selection else None
                ),
                needs_hitl=result.needs_hitl,
            )
            if self._tool_selecting_fallback_legacy:
                if result.mode == "tool_select_failed":
                    return None
                sel = result.selection
                if (
                    sel is not None
                    and (sel.intent or "").lower() in {"unknown", ""}
                    and float(sel.confidence or 0.0) < 0.55
                ):
                    # Defer weak/unknown selections to regex heuristics.
                    return None
            return result.plan, result.mode

        def _legacy_cascade() -> tuple[ExecutionPlan, str] | None:
            structured = plan_from_query_state(query_state, memory=memory)
            if structured is None and query_state.filters:
                structured = plan_from_query_state(
                    query_state, memory=memory, min_confidence=0.5
                )
            if structured is not None:
                return structured, "query_state_fallback"
            heuristic = try_heuristic_plan(question, memory=memory)
            if heuristic is not None:
                return heuristic, "heuristic_fallback"
            return None

        if use_primary_selector:
            selected = await _run_tool_selector()
            if selected is not None:
                plan, mode = selected
                # Soft clarifications from the LLM lose to deterministic cascades
                # (HITL status confirms keep empty nodes on purpose).
                if (
                    self._tool_selecting_fallback_legacy
                    and mode != "tool_select_hitl"
                    and not plan.nodes
                    and plan.clarify_question
                ):
                    legacy = _legacy_cascade()
                    if legacy is not None:
                        return legacy
                return selected
            if self._tool_selecting_fallback_legacy:
                legacy = _legacy_cascade()
                if legacy is not None:
                    return legacy

        # One shared context view for residual slots + LLM planner.
        context_need = classify_context_need(q, memory)
        logger.info(
            "planner_context_need",
            question=q[:200],
            context_need=context_need.value,
        )

        if use_residual_selector and not use_primary_selector:
            selected = await _run_tool_selector()
            if selected is not None:
                return selected

        # Residual hybrid DST: structured slots before free-form LLM plans.
        try:
            bundle = await self._slot_extractor.extract(
                question, memory, context_need=context_need
            )
        except Exception as exc:  # noqa: BLE001 — fall through to LLM planner
            logger.warning(
                "llm_slots_extract_error",
                question=question[:200],
                error_type=type(exc).__name__,
                error=str(exc)[:300],
            )
            bundle = None
        if bundle is not None:
            slot_result = plan_from_slots(bundle, memory=memory, question=q)
            if slot_result.handled and slot_result.plan is not None:
                lint_err = lint_fresh_scope(slot_result.plan, context_need, q)
                if lint_err:
                    logger.warning(
                        "planner_fresh_scope_lint_failed",
                        source="llm_slots",
                        error=lint_err,
                        context_need=context_need.value,
                    )
                    return (
                        ExecutionPlan(
                            nodes=[],
                            response_strategy="template",
                            clarify_question=(
                                "I need a clearer ask without relying on the previous "
                                "result set. Name the person, department, city, or skill."
                            ),
                            refusal_code=RefusalCode.AMBIGUOUS.value,
                        ),
                        "llm_slots_lint_clarify",
                    )
                logger.info(
                    "planner_mode_llm_slots",
                    intent=bundle.intent,
                    confidence=bundle.confidence,
                    context_need=context_need.value,
                )
                return slot_result.plan, "llm_slots"
            if slot_result.handled and slot_result.plan is None and slot_result.clarify:
                return (
                    ExecutionPlan(
                        nodes=[],
                        response_strategy="template",
                        clarify_question=slot_result.clarify,
                        refusal_code=RefusalCode.AMBIGUOUS.value,
                    ),
                    "llm_slots",
                )

        packet = self._context_packet(
            question,
            auth,
            memory,
            query_state,
            context_manager,
            context_need=context_need.value,
        )
        try:
            plan = await self._llm_plan_once(
                q, packet=packet, context_need=context_need
            )
            return plan, "llm"
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning(
                "planner_llm_invalid_plan",
                question=question[:200],
                error_type=type(exc).__name__,
                error=str(exc)[:400],
                approx_tokens=packet.get("approx_tokens"),
                context_need=context_need.value,
            )
            stripped = minimal_planner_packet(
                question=q,
                role=auth.role.value,
                tools=self._tools.discover(),
                schema_catalog=self._catalog_view(),
                query_state=(
                    query_state.model_dump(mode="json") if query_state else None
                ),
                context_need=ContextNeed.FRESH.value,
            )
            try:
                plan = await self._llm_plan_once(
                    q, packet=stripped, context_need=ContextNeed.FRESH
                )
                logger.info(
                    "planner_llm_retry_stripped",
                    question=q[:200],
                    prior_need=context_need.value,
                )
                return plan, "llm_retry_stripped"
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc2:
                logger.warning(
                    "planner_llm_retry_failed",
                    question=question[:200],
                    error_type=type(exc2).__name__,
                    error=str(exc2)[:400],
                )
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=_CLARIFY_FALLBACK,
                ),
                "llm",
            )
