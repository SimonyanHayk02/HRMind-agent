from __future__ import annotations

import hashlib
import re
from enum import Enum

from app.domain.enums import Role


class SocialIntent(str, Enum):
    HELLO = "hello"
    THANKS = "thanks"
    BYE = "bye"
    HOW_ARE_YOU = "how_are_you"
    IDENTITY = "identity"
    CHITCHAT = "chitchat"


_BYE = re.compile(r"\b(bye|goodbye|see\s+you|cya)\b", re.I)
_THANKS = re.compile(r"\b(thanks|thank\s+you|thx)\b", re.I)
_HOW_ARE_YOU = re.compile(
    r"\b(how(?:'s|\s+are)\s+(?:you|it\s+going)|what(?:'s|\s+is)\s+up)\b",
    re.I,
)
_IDENTITY = re.compile(
    r"\b(who\s+are\s+you|what(?:\s+can|\s+do)\s+you\s+do|what\s+are\s+you)\b",
    re.I,
)
_HELLO = re.compile(
    r"\b(hi|hello|hey|good\s+(?:morning|afternoon|evening|day))\b",
    re.I,
)

_ROLE_TIPS: dict[Role, str] = {
    Role.RECRUITER: "I can help with employee data, resumes, and HR analytics.",
    Role.MANAGER: "I can help with your team's roster, skills, and headcount.",
    Role.EMPLOYEE: "I can help with directory lookups and profile questions you're allowed to see.",
}

_HELLO_VARIANTS = (
    "Hello! {tip}",
    "Hi there! {tip}",
    "Hey! {tip}",
    "Hello — glad you're here. {tip}",
)

_THANKS_VARIANTS = (
    "You're welcome!",
    "Happy to help!",
    "Anytime — ask whenever you need HR data or resume search.",
)

_BYE_VARIANTS = (
    "Goodbye! Feel free to ask if you need anything else.",
    "See you! I'm here when you need employee or resume help.",
    "Take care — come back anytime for HR analytics or lookups.",
)

_HOW_ARE_YOU_VARIANTS = (
    "I'm doing well, thanks! {tip}",
    "All good on my side. {tip}",
)

_IDENTITY_VARIANTS = (
    "I'm HRMind, your HR assistant. {tip}",
    "I'm HRMind — I assist with people data and resumes. {tip}",
)

_CHITCHAT_VARIANTS = (
    "I'm here for HR questions. {tip}",
    "I keep things focused on people data and resumes. {tip}",
)


def classify_social_intent(text: str) -> SocialIntent:
    t = text.strip()
    if not t:
        return SocialIntent.HELLO
    if _BYE.search(t):
        return SocialIntent.BYE
    if _THANKS.search(t):
        return SocialIntent.THANKS
    if _HOW_ARE_YOU.search(t):
        return SocialIntent.HOW_ARE_YOU
    if _IDENTITY.search(t):
        return SocialIntent.IDENTITY
    if _HELLO.search(t):
        return SocialIntent.HELLO
    return SocialIntent.CHITCHAT


def role_tip(role: Role) -> str:
    return _ROLE_TIPS.get(role, _ROLE_TIPS[Role.RECRUITER])


def _pick(variants: tuple[str, ...], seed: str) -> str:
    digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=4).digest()
    idx = int.from_bytes(digest, "big") % len(variants)
    return variants[idx]


def render_social_reply(text: str, *, role: Role) -> tuple[str, SocialIntent]:
    intent = classify_social_intent(text)
    tip = role_tip(role)
    seed = f"{intent.value}:{text.strip().lower()}"

    if intent == SocialIntent.THANKS:
        answer = _pick(_THANKS_VARIANTS, seed)
    elif intent == SocialIntent.BYE:
        answer = _pick(_BYE_VARIANTS, seed)
    elif intent == SocialIntent.HOW_ARE_YOU:
        answer = _pick(_HOW_ARE_YOU_VARIANTS, seed).format(tip=tip)
    elif intent == SocialIntent.IDENTITY:
        answer = _pick(_IDENTITY_VARIANTS, seed).format(tip=tip)
    elif intent == SocialIntent.CHITCHAT:
        answer = _pick(_CHITCHAT_VARIANTS, seed).format(tip=tip)
    else:
        answer = _pick(_HELLO_VARIANTS, seed).format(tip=tip)

    return answer, intent
