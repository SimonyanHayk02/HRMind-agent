import pytest

from app.domain.auth import AuthContext
from app.domain.enums import Role, RouterLabel
from app.application.routing.rule_router import RuleRouter
from app.tools.greeting.replies import SocialIntent, classify_social_intent, render_social_reply
from app.tools.greeting.tool import GreetingTool


@pytest.mark.parametrize(
    "text,intent",
    [
        ("hello", SocialIntent.HELLO),
        ("Good morning!", SocialIntent.HELLO),
        ("thanks", SocialIntent.THANKS),
        ("thank you", SocialIntent.THANKS),
        ("bye", SocialIntent.BYE),
        ("see you", SocialIntent.BYE),
        ("how are you", SocialIntent.HOW_ARE_YOU),
        ("what's up", SocialIntent.HOW_ARE_YOU),
        ("who are you", SocialIntent.IDENTITY),
        ("what can you do", SocialIntent.IDENTITY),
        ("tell me a joke", SocialIntent.CHITCHAT),
    ],
)
def test_classify_social_intent(text: str, intent: SocialIntent) -> None:
    assert classify_social_intent(text) == intent


def test_hello_is_role_aware_and_stable() -> None:
    a1, i1 = render_social_reply("hello", role=Role.RECRUITER)
    a2, i2 = render_social_reply("hello", role=Role.RECRUITER)
    assert i1 == SocialIntent.HELLO
    assert a1 == a2
    assert "help" in a1.lower()
    assert "resume" in a1.lower() or "hr" in a1.lower()

    mgr, _ = render_social_reply("hello", role=Role.MANAGER)
    assert "team" in mgr.lower()

    emp, _ = render_social_reply("hi", role=Role.EMPLOYEE)
    assert "directory" in emp.lower() or "profile" in emp.lower()


def test_identity_mentions_hrmind() -> None:
    answer, intent = render_social_reply("who are you", role=Role.RECRUITER)
    assert intent == SocialIntent.IDENTITY
    assert "hrmind" in answer.lower()


@pytest.mark.asyncio
async def test_greeting_tool_returns_intent() -> None:
    tool = GreetingTool()
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    result = await tool.run({"message": "thanks"}, auth=auth)
    assert result.data["intent"] == "thanks"
    assert result.data["answer"]


@pytest.mark.parametrize(
    "question",
    [
        "hello",
        "thanks",
        "bye",
        "good morning",
        "how are you",
        "what's up",
        "who are you",
        "what can you do",
    ],
)
def test_rule_router_pure_social_expanded(question: str) -> None:
    assert RuleRouter().route(question) == RouterLabel.GREETING


def test_rule_router_still_rejects_hr_mixed_with_social() -> None:
    router = RuleRouter()
    assert router.route("thanks, say their names") is None
    assert router.route("hi who knows python") is None
    assert router.route("how are you finding engineers") is None
