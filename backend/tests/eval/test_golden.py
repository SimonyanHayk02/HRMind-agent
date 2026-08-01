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


def test_golden_tool_cases_are_not_rule_greetings() -> None:
    """HR / tool questions must not be swallowed by the rule greeting router."""
    path = Path("data/golden/cases.jsonl")
    router = RuleRouter()
    checked = 0
    for line in path.read_text().splitlines():
        case = json.loads(line)
        if case.get("expected_route") == "greeting":
            continue
        assert router.route(case["question"]) is None, case["question"]
        checked += 1
    assert checked >= 5


def test_orchestration_golden_file_present() -> None:
    path = Path("data/golden/orchestration.jsonl")
    assert path.exists()
    cases = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(cases) >= 10
    assert {c["check"] for c in cases} >= {
        "skill_location_intersect",
        "pronoun_two_entities_clarify",
        "ordinal_birthday_resume",
        "greeting_route",
    }
