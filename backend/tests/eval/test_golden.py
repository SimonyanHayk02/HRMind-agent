import json
from pathlib import Path

from app.application.routing.rule_router import RuleRouter
from app.domain.enums import RouterLabel


def test_golden_greeting_routes() -> None:
    path = Path("data/golden/cases.jsonl")
    router = RuleRouter()
    checked = 0
    for line in path.read_text().splitlines():
        case = json.loads(line)
        if case.get("expected_route") != "greeting":
            continue
        assert router.route(case["question"]) == RouterLabel.GREETING
        checked += 1
    assert checked >= 3
