from __future__ import annotations

from app.domain.query_state import QueryState
from app.domain.session import ConstraintRef, SessionMemory


def constraints_from_query_state(state: QueryState) -> list[ConstraintRef]:
    out: list[ConstraintRef] = []
    for f in state.filters:
        if f.op == "eq" and f.field not in {"hire_date"}:
            out.append(ConstraintRef(field=f.field, op=f.op, value=f.value))
        elif f.field == "hire_date":
            out.append(ConstraintRef(field="hire_date", op=f.op, value=f.value))
    if state.skill:
        out.append(ConstraintRef(field="skill", op="contains", value=state.skill))
    return out


def should_set_universe_all(state: QueryState) -> bool:
    """Org-wide questions establish an 'all employees' universe for of-them follow-ups."""
    return (
        state.intent == "count"
        and not state.refers_to_prior
        and not state.filters
        and not state.skill
    )


def apply_universe_marker(session: SessionMemory, state: QueryState) -> list[ConstraintRef]:
    """Return constraints to merge, including a soft universe marker when needed.

    When refining (anaphora or new filters), drop a stale `_universe=all` marker
    from the session so it does not pollute later planner packets.

    Org-wide headcount resets the filter stack and the active "them" cohort so a
    later "of them in Dubai" is not still bound to a single profile subject or an
    old department constraint.
    """
    constraints = constraints_from_query_state(state)
    if should_set_universe_all(state):
        session.last_employee_ids = []
        session.last_listed = []
        session.active_referent = None
        session.constraint_memory = []
        constraints = [ConstraintRef(field="_universe", op="eq", value="all")]
    elif state.refers_to_prior or state.filters or state.skill:
        # Drop stale universe-only marker when refining
        session.constraint_memory = [
            c for c in session.constraint_memory if c.field != "_universe"
        ]
    return constraints
