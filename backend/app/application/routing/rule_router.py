from __future__ import annotations

import re

from app.domain.enums import RouterLabel

_GREETING = re.compile(r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b", re.I)
_BYE = re.compile(r"^(bye|goodbye|see\s+you|cya)\b", re.I)
_THANKS = re.compile(r"^(thanks|thank\s+you|thx)\b", re.I)


class RuleRouter:
    def route(self, question: str) -> RouterLabel | None:
        text = question.strip()
        if _GREETING.match(text) or _BYE.match(text) or _THANKS.match(text):
            return RouterLabel.GREETING
        return None
