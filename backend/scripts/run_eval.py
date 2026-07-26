#!/usr/bin/env python3
"""Run golden eval cases against rule/plan heuristics."""
from __future__ import annotations

import json
from pathlib import Path

from app.application.routing.rule_router import RuleRouter
from app.domain.enums import RouterLabel


def main() -> None:
    path = Path("data/golden/cases.jsonl")
    router = RuleRouter()
    total = 0
    hits = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        total += 1
        expected = case.get("expected_route")
        if expected == "greeting":
            got = router.route(case["question"])
            ok = got == RouterLabel.GREETING
        else:
            # non-greeting expected -> rule router should return None
            got = router.route(case["question"])
            ok = got is None
        hits += int(ok)
        print(("PASS" if ok else "FAIL"), case["id"], case["question"][:60], "->", got)
    print(f"router_accuracy={hits}/{total}")


if __name__ == "__main__":
    main()
