from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan
from app.application.planning.unsupported import (
    is_count_question,
    is_unsupported_topic,
)
from app.application.response.refusal import (
    empty_unscoped_fallback,
    has_format_evidence,
    resolve_refusal,
)
from app.domain.tools.base import SourceRef, ToolResult
from app.ports.llm import LLMClient


class ResponseFormatter:
    def __init__(self, llm: LLMClient, prompts_dir: Path | None = None) -> None:
        self._llm = llm
        self._prompts_dir = prompts_dir or Path(__file__).resolve().parents[2] / "prompts"

    async def format(
        self,
        question: str,
        plan: ExecutionPlan,
        state: GraphState,
        *,
        context_manager=None,
        auth=None,
    ) -> tuple[str, float, list[SourceRef], str | None]:
        effective_auth = auth or state.auth
        refusal = resolve_refusal(plan, state, question, auth=effective_auth)
        if refusal is not None:
            return (
                refusal.message,
                refusal.confidence,
                [],
                refusal.ui_clarify,
            )

        payloads: list[Any] = []
        sources: list[SourceRef] = []
        confidence = 1.0
        tool_errors: list[str] = []

        for node in plan.nodes:
            result = state.node_results.get(node.id)
            if isinstance(result, ToolResult):
                if result.error:
                    tool_errors.append(f"{node.name}: {result.error}")
                if result.data is not None:
                    payloads.append(result.data)
                sources.extend(result.sources)
                confidence = min(confidence, result.confidence)
                if result.data and isinstance(result.data, dict) and result.data.get("answer"):
                    if plan.response_strategy == "template" or node.name == "greeting":
                        return str(result.data["answer"]), result.confidence, sources, None
                # Prefer deterministic employee/manager formatting over raw JSON / LLM
                if (
                    result.data
                    and isinstance(result.data, dict)
                    and node.name == "employee"
                ):
                    pretty = _format_employee_tool_payload(
                        _with_resume_location(result.data, state), question
                    )
                    if pretty:
                        return pretty, result.confidence, sources, None
            elif result is not None:
                payloads.append(result)

        count_asked = is_count_question(question) or _plan_is_count_only(plan)
        facet_dim = _plan_facet_dimension(plan)

        # Honest empty-result answers — never dump raw JSON or LLM invent
        empty_msg = _format_empty_tool_results(question, plan, payloads, count_asked)
        if empty_msg is not None:
            return empty_msg, confidence, sources, None

        if plan.response_strategy == "template":
            hint = _plan_answer_hint(plan)
            tenure = _format_tenure_payloads(payloads, hint=hint)
            if tenure:
                return tenure, confidence, sources, None
            # Prefer explicit count payloads when the question/plan is a count
            # (cohort-materialization nodes may also return id rows).
            if count_asked:
                for p in reversed(payloads):
                    if isinstance(p, dict) and p.get("count") is not None:
                        return (
                            _format_facet_count(int(p["count"]), facet_dim),
                            confidence,
                            sources,
                            None,
                        )
                    if (
                        isinstance(p, dict)
                        and "row_count" in p
                        and "rows" not in p
                    ):
                        return (
                            _format_facet_count(int(p["row_count"]), facet_dim),
                            confidence,
                            sources,
                            None,
                        )
            for p in reversed(payloads):
                if (
                    count_asked
                    and isinstance(p, dict)
                    and "count" in p
                    and p["count"] is not None
                ):
                    return (
                        _format_facet_count(int(p["count"]), facet_dim),
                        confidence,
                        sources,
                        None,
                    )
                if (
                    count_asked
                    and isinstance(p, dict)
                    and "row_count" in p
                    and "rows" not in p
                ):
                    return (
                        _format_facet_count(int(p["row_count"]), facet_dim),
                        confidence,
                        sources,
                        None,
                    )
                if isinstance(p, dict) and isinstance(p.get("rows"), list):
                    if not p["rows"] and count_asked:
                        return "The answer is 0.", confidence, sources, None
                    if not p["rows"]:
                        from app.application.response.refusal import empty_refine_answer

                        return (
                            empty_refine_answer(question),
                            confidence,
                            sources,
                            None,
                        )
                    # Count plans also materialize an id cohort — never list names
                    # when the user asked for a number.
                    if count_asked:
                        continue
                    facet = _format_facet_rows(p["rows"], facet_dim)
                    if facet:
                        return facet, confidence, sources, None
                    ordered = _order_rows_by_plan_ids(p["rows"], plan)
                    named = _format_employee_rows(ordered)
                    if named:
                        return named, confidence, sources, None
                if isinstance(p, dict):
                    pretty = _format_employee_tool_payload(p, question)
                    if pretty:
                        return pretty, confidence, sources, None
            if tool_errors:
                detail = "; ".join(tool_errors)
                return (
                    f"I couldn't finish computing that answer ({detail}).",
                    0.0,
                    sources,
                    None,
                )
            if not payloads:
                return empty_unscoped_fallback(), 0.3, sources, None
            last = payloads[-1]
            if isinstance(last, list) and last and all(isinstance(x, str) for x in last):
                return f"Found {len(last)} matching employees.", confidence, sources, None
            if isinstance(last, list) and not last:
                from app.application.response.refusal import empty_refine_answer

                return (
                    empty_refine_answer(question),
                    confidence,
                    sources,
                    None,
                )
            # Do not turn an unscoped employee dump into a fake numeric answer
            if _looks_unscoped_dump(last) and not count_asked:
                return empty_unscoped_fallback(), 0.3, sources, None
            # Never leak raw tool JSON to the user
            if isinstance(last, dict):
                pretty = _format_employee_tool_payload(last, question)
                if pretty:
                    return pretty, confidence, sources, None
                if last.get("employees") == []:
                    return (
                        "I couldn't find that employee. "
                        "Which person did you mean?",
                        0.4,
                        sources,
                        "Which employee do you mean?",
                    )
            return empty_unscoped_fallback(), 0.3, sources, None

        # Hard gate: never LLM-format without grounded evidence.
        if not has_format_evidence(plan, state):
            return empty_unscoped_fallback(), 0.3, sources, None

        path = self._prompts_dir / "response.md"
        system = (
            path.read_text()
            if path.exists()
            else (
                "Format a concise HR answer using ONLY the provided tool results. "
                "Do not invent facts. If results cannot answer the question, say you "
                "don't have that information."
            )
        )
        for p in payloads:
            if isinstance(p, dict):
                pretty = _format_employee_tool_payload(p, question)
                if pretty:
                    return pretty, confidence, sources, None
        packed = payloads
        if context_manager is not None:
            packed = context_manager.build_for_responder(payloads, auth=effective_auth)
        else:
            from app.application.memory.context_budget import compact_tool_payloads

            packed = compact_tool_payloads(payloads, auth=effective_auth)
        user = json.dumps({"question": question, "results": packed}, default=str)
        answer = await self._llm.complete(system=system, user=user, temperature=0.0)
        if not answer or answer.strip().lower() in {"null", "none"}:
            return empty_unscoped_fallback(), 0.3, sources, None
        # Guard: LLM sometimes answers headcount when asked about missing domains
        if _is_spurious_headcount_answer(answer, question, payloads):
            return empty_unscoped_fallback(), 0.3, sources, None
        if _has_ungrounded_claims(answer, payloads):
            return empty_unscoped_fallback(), 0.3, sources, None
        return answer, confidence, sources, None


def _format_empty_tool_results(
    question: str,
    plan: ExecutionPlan,
    payloads: list[Any],
    count_asked: bool,
) -> str | None:
    """Grounded empty answers for zero counts / empty cohorts / missing people."""
    q = (question or "").lower()
    prior_filter = bool(
        re.search(r"\b(them|those|that set|of them|which of them)\b", q)
        or any(
            (n.params or {}).get("employee_ids")
            or (n.params or {}).get("other")
            for n in plan.nodes
            if n.name in {"resume_search", "intersect_ids", "sql"}
        )
    )

    for p in reversed(payloads):
        if isinstance(p, dict) and p.get("count") is not None:
            try:
                n = int(p["count"])
            except (TypeError, ValueError):
                continue
            if n == 0:
                if prior_filter and not count_asked:
                    from app.application.response.refusal import empty_refine_answer

                    return empty_refine_answer(question)
                return "The answer is 0."
        if (
            isinstance(p, dict)
            and "row_count" in p
            and "rows" not in p
            and int(p.get("row_count") or 0) == 0
        ):
            return "The answer is 0."
        if isinstance(p, dict) and isinstance(p.get("rows"), list) and not p["rows"]:
            if count_asked:
                return "The answer is 0."
            if prior_filter:
                from app.application.response.refusal import empty_refine_answer

                return empty_refine_answer(question)
            return "I couldn't find matching employees for that."
        if isinstance(p, dict) and p.get("employees") == []:
            return (
                "I couldn't find that employee. Which person did you mean?"
            )
        if isinstance(p, list) and not p:
            if count_asked:
                return "The answer is 0."
            if prior_filter:
                from app.application.response.refusal import empty_refine_answer

                return empty_refine_answer(question)
    return None


def _plan_answer_hint(plan: ExecutionPlan) -> str | None:
    for node in plan.nodes:
        hint = (node.params or {}).get("answer_hint")
        if hint:
            return str(hint)
    return None


def _format_tenure_payloads(payloads: list[Any], *, hint: str | None) -> str | None:
    """Format allowlisted tenure / longest-tenured SQL template rows."""
    for p in reversed(payloads):
        if not isinstance(p, dict):
            continue
        rows = p.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "avg_tenure_days" in sample and "department" in sample:
            bits = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                days = row.get("avg_tenure_days")
                dept = row.get("department") or "Unknown"
                if days is None:
                    continue
                years = float(days) / 365.25
                bits.append(f"{dept}: {years:.1f} years avg ({int(float(days))} days)")
            if not bits:
                continue
            body = "Average tenure by department (from hire_date): " + "; ".join(bits) + "."
            return f"{body} {hint}" if hint else body
        if "avg_tenure_days" in sample:
            days = sample.get("avg_tenure_days")
            if days is None:
                continue
            years = float(days) / 365.25
            body = (
                f"Average tenure is {years:.1f} years "
                f"({int(float(days))} days), based on hire_date."
            )
            return f"{body} {hint}" if hint else body
        if "tenure_days" in sample and "hire_date" in sample:
            lines = []
            for row in rows[:10]:
                if not isinstance(row, dict):
                    continue
                name = " ".join(
                    x
                    for x in (row.get("first_name"), row.get("last_name"))
                    if x
                ).strip() or "Unknown"
                hd = row.get("hire_date")
                td = row.get("tenure_days")
                dept = row.get("department") or ""
                detail = f"hired {hd}" if hd else ""
                if td is not None:
                    detail = (
                        f"{detail}; {int(float(td))} days tenure"
                        if detail
                        else f"{int(float(td))} days tenure"
                    )
                if dept:
                    detail = f"{detail}; {dept}" if detail else dept
                lines.append(f"{name} ({detail})" if detail else name)
            if not lines:
                continue
            body = (
                "Longest tenured by hire_date (not title seniority): "
                + "; ".join(lines)
                + "."
            )
            return f"{body} {hint}" if hint else body
    return None


def _plan_is_count_only(plan: ExecutionPlan) -> bool:
    return any(
        (
            n.name == "sql"
            and (
                bool((n.params or {}).get("count_only"))
                or bool((n.params or {}).get("count_distinct"))
            )
        )
        or (n.name == "resume_search" and bool((n.params or {}).get("facet_count")))
        for n in plan.nodes
    )


def _plan_facet_dimension(plan: ExecutionPlan) -> str | None:
    for node in plan.nodes:
        params = node.params or {}
        # Resume-sourced facets (city, country) are aggregated by retrieval.
        if node.name == "resume_search" and params.get("facet"):
            return str(params["facet"])
        if node.name != "sql":
            continue
        dim = params.get("count_distinct")
        if dim:
            return str(dim)
        if params.get("distinct"):
            cols = params.get("columns") or []
            if cols:
                return str(cols[0])
    return None


def _format_facet_count(count: int, dimension: str | None) -> str:
    labels = {
        "country": "countries",
        "city": "cities",
        "department": "departments",
        "position": "positions",
    }
    if dimension in labels:
        return f"We have employees in {count} different {labels[dimension]}."
    return f"The answer is {count}."


def _format_facet_rows(rows: list[Any], dimension: str | None) -> str | None:
    if not dimension:
        # Infer from row keys when plan dim missing
        if rows and isinstance(rows[0], dict):
            for key in ("country", "city", "department", "position"):
                if key in rows[0] and "id" not in rows[0] and "first_name" not in rows[0]:
                    dimension = key
                    break
    if not dimension:
        return None
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or dimension not in row:
            continue
        val = str(row.get(dimension) or "").strip()
        if not val or val in seen:
            continue
        seen.add(val)
        values.append(val)
    if not values:
        return None
    # Employee name rows also contain department — don't treat as facet lists
    if any(isinstance(r, dict) and (r.get("id") or r.get("first_name")) for r in rows):
        return None
    labels = {
        "country": "countries",
        "city": "cities",
        "department": "departments",
        "position": "positions",
    }
    label = labels.get(dimension, dimension)
    if len(values) == 1:
        return f"The matching {label[:-1] if label.endswith('s') else label} is {values[0]}."
    bullet = "\n".join(f"- {v}" for v in values)
    return f"Here are the {len(values)} {label}:\n{bullet}"


def _looks_unscoped_dump(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return False
    sql = str(payload.get("sql") or "")
    if "WHERE 1=1" in sql and "employee_ids" not in sql.lower() and len(rows) >= 20:
        return True
    return False


def _is_spurious_headcount_answer(answer: str, question: str, payloads: list[Any]) -> bool:
    if is_unsupported_topic(question):
        return True
    if is_count_question(question) and not is_unsupported_topic(question):
        # Legitimate count questions may answer with a number
        return False
    text = answer.strip().lower()
    if text.startswith("the answer is ") and text.rstrip(".").split()[-1].isdigit():
        if any(_looks_unscoped_dump(p) for p in payloads):
            return True
    return False


_PROPER_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b")
_NUMBER_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+)\b")
_STOP_NAMES = frozenset(
    {
        "the answer",
        "no one",
        "not found",
        "i found",
        "please provide",
        "human resources",
    }
)


def _payload_text_blob(payloads: list[Any]) -> str:
    try:
        return json.dumps(payloads, default=str).lower()
    except (TypeError, ValueError):
        return str(payloads).lower()


def _has_ungrounded_claims(answer: str, payloads: list[Any]) -> bool:
    """Reject llm_format answers that invent proper names or large counts."""
    blob = _payload_text_blob(payloads)
    if not blob.strip() or blob in {"null", "[]", "{}"}:
        return True
    for m in _PROPER_NAME_RE.finditer(answer or ""):
        name = m.group(1).strip()
        if name.lower() in _STOP_NAMES:
            continue
        # Require every token of a multi-word name to appear in tool payloads.
        tokens = [t for t in re.split(r"\s+", name) if t]
        if tokens and not all(t.lower() in blob for t in tokens):
            return True
    for m in _NUMBER_RE.finditer(answer or ""):
        raw = m.group(1).replace(",", "")
        if len(raw) > 6:
            continue  # long ids
        try:
            n = int(raw)
        except ValueError:
            continue
        # Small ints are too ambiguous ("2 people"); gate material figures only.
        if n < 10:
            continue
        if str(n) not in blob and raw not in blob:
            return True
    return False


def _resume_location_facts(state: GraphState) -> dict[str, dict[str, Any]]:
    """Location facts by employee id, from any retrieval node in this turn."""
    out: dict[str, dict[str, Any]] = {}
    for result in state.node_results.values():
        data = result.data if isinstance(result, ToolResult) else result
        if not isinstance(data, dict):
            continue
        for fact in data.get("facts") or []:
            if not isinstance(fact, dict) or not fact.get("employee_id"):
                continue
            if fact.get("city") or fact.get("country"):
                out[str(fact["employee_id"])] = fact
    return out


def _with_resume_location(data: dict[str, Any], state: GraphState) -> dict[str, Any]:
    """Attach the retrieved place to an employee payload.

    A profile still shows a Location line, but the value now comes from the
    person's resume and is matched on the id the employee tool resolved, so a
    namesake's city can never be printed under the wrong name.
    """
    eid = str(data.get("id") or "")
    fact = _resume_location_facts(state).get(eid) if eid else None
    if not fact:
        return data
    return {**data, "city": fact.get("city"), "country": fact.get("country")}


def _format_employee_tool_payload(data: dict[str, Any], question: str) -> str | None:
    """Human answers for employee tool profile / manager / roster payloads."""
    if data.get("clarify"):
        return str(data["clarify"])
    if data.get("updated") and data.get("answer"):
        return str(data["answer"])
    if data.get("answer") and "status" in (question or "").lower():
        return str(data["answer"])

    # Manager chain
    if "employee" in data and "managers" in data:
        emp = data.get("employee") or {}
        managers = data.get("managers") or []
        emp_name = _person_label(emp)
        if not managers:
            return f"{emp_name} has no manager on file." if emp_name else "No manager on file."
        mgr = managers[0] if isinstance(managers, list) and managers else None
        if not isinstance(mgr, dict):
            return None
        mgr_name = _person_label(mgr)
        extra = ", ".join(
            x for x in (mgr.get("position"), mgr.get("department"), mgr.get("city")) if x
        )
        if extra:
            return f"{emp_name}'s manager is {mgr_name} ({extra})."
        return f"{emp_name}'s manager is {mgr_name}."

    # Department / direct-reports roster
    if isinstance(data.get("employees"), list):
        rows = data["employees"]
        mgr = data.get("manager") if isinstance(data.get("manager"), dict) else None
        if not rows:
            if mgr:
                label = _person_label(mgr) or "That manager"
                return f"{label} has no direct reports on file."
            return None
        if all(isinstance(r, dict) for r in rows):
            # Map to row formatter fields
            mapped = []
            for r in rows:
                mapped.append(
                    {
                        "first_name": r.get("first_name") or "",
                        "last_name": r.get("last_name") or "",
                        "department": r.get("department") or "",
                        "position": r.get("position") or "",
                        "id": r.get("id"),
                    }
                )
            named = _format_employee_rows(mapped)
            if named and mgr:
                label = _person_label(mgr)
                if label:
                    return f"Direct reports of {label}: {named}"
            return named

    # Single profile
    if data.get("full_name") or (data.get("first_name") and data.get("id")):
        name = _person_label(data)
        q = (question or "").lower()
        if any(w in q for w in ("live", "lives", "living", "located", "where", "city", "based")):
            city = data.get("city")
            country = data.get("country")
            loc = ", ".join(x for x in (city, country) if x)
            if loc:
                return f"{name} lives in {loc}."
            return f"I don't have a location on file for {name}."
        if any(w in q for w in ("education", "degree", "school", "university", "college")):
            edu = data.get("education")
            if edu:
                return f"{name}'s education is {edu}."
            return f"I don't have education on file for {name}."
        if any(w in q for w in ("title", "position", "role", "job")):
            pos = data.get("position")
            if pos:
                return f"{name}'s job title is {pos}."
            return f"I don't have a job title on file for {name}."
        if any(w in q for w in ("email", "e-mail", "mail")):
            email = data.get("email")
            if email:
                return f"{name}'s email is {email}."
            return f"I don't have an email on file for {name}."
        if any(w in q for w in ("salary", "pay", "compensation", "earn", "make")):
            sal = data.get("salary")
            if sal is not None and sal != "":
                return f"{name}'s salary is {sal}."
            return f"I don't have salary on file for {name}."
        if "department" in q or "team" in q:
            dept = data.get("department")
            if dept:
                return f"{name} works in {dept}."
            return f"I don't have a department on file for {name}."
        # Agent flag (employees.status) — distinct from employment_status.
        if re.search(r"\bstatus\b", q) and "employment" not in q:
            flag = data.get("status")
            if isinstance(flag, bool):
                label = "active" if flag else "inactive"
                return f"{name}'s status is {label}."
            if flag is not None:
                return f"{name}'s status is {flag}."
        bits = [name]
        for label, key in (
            ("Position", "position"),
            ("Department", "department"),
            ("Education", "education"),
            ("Location", None),
            ("Email", "email"),
            ("Employment status", "employment_status"),
            ("Status", "status"),
        ):
            if key is None:
                loc = ", ".join(x for x in (data.get("city"), data.get("country")) if x)
                if loc:
                    bits.append(f"Location: {loc}")
                continue
            val = data.get(key)
            if key == "status" and isinstance(val, bool):
                bits.append(f"{label}: {'active' if val else 'inactive'}")
            elif val is not None and val != "":
                bits.append(f"{label}: {val}")
        return "\n".join(bits) if len(bits) > 1 else name
    return None


def _person_label(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return "Unknown"
    if row.get("full_name"):
        return str(row["full_name"])
    first = str(row.get("first_name") or "").strip()
    last = str(row.get("last_name") or "").strip()
    return f"{first} {last}".strip() or "Unknown"


def _plan_employee_id_order(plan: ExecutionPlan) -> list[str]:
    """Requested employee_ids order from the plan, if any."""
    for node in plan.nodes:
        if node.name != "sql":
            continue
        filters = (node.params or {}).get("filters") or {}
        ids = filters.get("employee_ids")
        if isinstance(ids, list) and ids:
            return [str(x) for x in ids if x]
    return []


def _order_rows_by_plan_ids(rows: list[Any], plan: ExecutionPlan) -> list[Any]:
    """Second guarantee that bullets match the cohort order the user just narrowed."""
    order = _plan_employee_id_order(plan)
    if not order:
        return rows
    rank = {eid: i for i, eid in enumerate(order)}
    named = [r for r in rows if isinstance(r, dict) and r.get("id")]
    rest = [r for r in rows if not (isinstance(r, dict) and r.get("id"))]
    named.sort(key=lambda r: rank.get(str(r.get("id")), len(rank)))
    return named + rest


def _format_employee_rows(rows: list[Any]) -> str | None:
    names: list[str] = []
    show_status = any(
        isinstance(r, dict) and ("status" in r or "employment_status" in r) for r in rows
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        first = str(row.get("first_name") or "").strip()
        last = str(row.get("last_name") or "").strip()
        full = f"{first} {last}".strip()
        if not full:
            continue
        dept = str(row.get("department") or "").strip()
        pos = str(row.get("position") or "").strip()
        extra = ", ".join(x for x in (pos, dept) if x)
        line = f"{full} ({extra})" if extra else full
        if show_status:
            bits: list[str] = []
            if "status" in row:
                flag = row.get("status")
                if isinstance(flag, bool):
                    bits.append(f"status={'active' if flag else 'inactive'}")
                elif flag is not None:
                    bits.append(f"status={flag}")
            emp = row.get("employment_status")
            if emp is not None and str(emp).strip():
                bits.append(f"employment_status={emp}")
            if bits:
                line = f"{line} — {', '.join(bits)}"
        names.append(line)
    if not names:
        return None
    if len(names) == 1:
        return f"The matching employee is {names[0]}."
    bullet = "\n".join(f"- {n}" for n in names)
    if show_status:
        return f"Here are the statuses for {len(names)} employees:\n{bullet}"
    return f"Here are the {len(names)} matching employees:\n{bullet}"
