"""Token / size budgets for planner and responder context packets."""

from __future__ import annotations

from typing import Any

from app.domain.auth import AuthContext
from app.domain.policies.column_policy import SENSITIVE_COLUMNS, allowed_columns
from app.domain.session import SessionMemory

DEFAULT_MAX_RECENT_MESSAGES = 8
DEFAULT_MAX_MESSAGE_CHARS = 800
DEFAULT_MAX_SUMMARY_CHARS = 2000
DEFAULT_MAX_ENTITY_PACKET = 20
DEFAULT_MAX_IDS_IN_PACKET = 20
DEFAULT_MAX_CONSTRAINTS = 20
DEFAULT_MAX_NAMED_SETS = 8
DEFAULT_MAX_RESPONSE_PAYLOAD_CHARS = 12000
DEFAULT_MAX_RESUME_SNIPPETS = 5
DEFAULT_MAX_SNIPPET_CHARS = 400
DEFAULT_MAX_ENTITY_MEMORY = 50


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def clip_text(value: str | None, max_chars: int) -> str:
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def redact_mapping(data: dict[str, Any], auth: AuthContext) -> dict[str, Any]:
    allowed = allowed_columns(
        auth.role,
        department=auth.department_id,
        target_department=data.get("department"),
    )
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in SENSITIVE_COLUMNS and key not in allowed:
            continue
        if isinstance(value, dict):
            out[key] = redact_mapping(value, auth)
        elif isinstance(value, list):
            out[key] = [
                redact_mapping(v, auth) if isinstance(v, dict) else v for v in value
            ]
        else:
            out[key] = value
    return out


def redact_payload(payload: Any, auth: AuthContext) -> Any:
    if isinstance(payload, dict):
        return redact_mapping(payload, auth)
    if isinstance(payload, list):
        return [redact_payload(p, auth) for p in payload]
    return payload


def compact_tool_payloads(
    payloads: list[Any],
    *,
    auth: AuthContext,
    max_total_chars: int = DEFAULT_MAX_RESPONSE_PAYLOAD_CHARS,
    max_snippets: int = DEFAULT_MAX_RESUME_SNIPPETS,
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS,
) -> list[Any]:
    compact: list[Any] = []
    used = 0
    for raw in payloads:
        item = redact_payload(raw, auth)
        if isinstance(item, dict) and isinstance(item.get("hits"), list):
            hits = []
            for hit in item["hits"][:max_snippets]:
                if not isinstance(hit, dict):
                    continue
                h = dict(hit)
                snippets = h.get("snippets") or []
                if isinstance(snippets, list):
                    h["snippets"] = [
                        clip_text(str(s), max_snippet_chars) for s in snippets[:3]
                    ]
                hits.append(h)
            item = {**item, "hits": hits}
            ids = item.get("employee_ids")
            if isinstance(ids, list) and len(ids) > DEFAULT_MAX_IDS_IN_PACKET:
                item["employee_ids"] = ids[:DEFAULT_MAX_IDS_IN_PACKET]
                item["employee_ids_truncated"] = True
        elif isinstance(item, dict) and isinstance(item.get("rows"), list):
            rows = item["rows"]
            if len(rows) > 30:
                item = {**item, "rows": rows[:30], "rows_truncated": True}
        encoded = str(item)
        if used + len(encoded) > max_total_chars and compact:
            compact.append({"truncated": True, "note": "remaining payloads omitted"})
            break
        used += len(encoded)
        compact.append(item)
    return compact


def build_planner_packet(
    *,
    question: str,
    auth: AuthContext,
    memory: SessionMemory | None,
    tools: list[dict],
    schema_catalog: dict,
    query_state: dict | None,
    max_context: int = DEFAULT_MAX_RECENT_MESSAGES,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
    max_ids: int = DEFAULT_MAX_IDS_IN_PACKET,
    max_entities: int = DEFAULT_MAX_ENTITY_PACKET,
) -> dict[str, Any]:
    recent: list[dict[str, str]] = []
    entities: list[dict] = []
    constraints: list[dict] = []
    last_ids: list[str] = []
    summary = ""
    last_focus: dict | None = None
    active_referent: dict | None = None
    named_sets: dict[str, list[str]] = {}
    person_bindings: dict[str, str] = {}
    tool_facts: list[dict] = []

    if memory:
        for msg in memory.messages[-max_context:]:
            recent.append(
                {"role": msg.role, "content": clip_text(msg.content, max_message_chars)}
            )
        entities = [
            e.model_dump(mode="json") for e in memory.entity_memory[-max_entities:]
        ]
        constraints = [
            c.model_dump(mode="json")
            for c in memory.constraint_memory[-DEFAULT_MAX_CONSTRAINTS:]
        ]
        raw_ids = list(
            (memory.active_referent.ids if memory.active_referent else None)
            or memory.last_employee_ids
        )
        last_ids = raw_ids[:max_ids]
        summary = clip_text(memory.summary or "", max_summary_chars)
        if memory.last_focus:
            last_focus = memory.last_focus.model_dump(mode="json")
        if memory.active_referent:
            active_referent = memory.active_referent.model_dump(mode="json")
            active_referent["ids"] = active_referent.get("ids", [])[:max_ids]
        for i, (key, ids) in enumerate(memory.named_sets.items()):
            if i >= DEFAULT_MAX_NAMED_SETS:
                break
            named_sets[key] = list(ids)[:max_ids]
        person_bindings = dict(list(memory.person_bindings.items())[:20])
        for fact in list(memory.tool_fact_cache.values())[:10]:
            val = fact.value
            if isinstance(val, (list, dict)):
                val = str(val)[:200]
            tool_facts.append({"key": fact.key, "value": val})

    packet = {
        "question": question,
        "role": auth.role.value,
        "tools": tools,
        "schema_catalog": schema_catalog,
        "query_state": query_state,
        "recent_messages": recent,
        "last_employee_ids": last_ids,
        "last_employee_ids_truncated": bool(
            memory and len(memory.last_employee_ids) > max_ids
        ),
        "active_referent": active_referent,
        "last_focus": last_focus,
        "constraints": constraints,
        "entities": entities,
        "named_sets": named_sets,
        "person_bindings": person_bindings,
        "tool_facts": tool_facts,
        "summary": summary,
    }
    packet["approx_tokens"] = estimate_tokens(str(packet))
    return packet
