from __future__ import annotations

import re

from app.domain.enums import RouterLabel

# Whole-utterance social only — do not swallow HR follow-ups after thanks/hi.
_PURE_SOCIAL = re.compile(
    r"^(?:"
    r"(?:hi|hello|hey)(?:\s+(?:there|friend|everyone|team|all))?|"
    r"good\s+(?:morning|afternoon|evening|day)|"
    r"bye|goodbye|see\s+you|cya|"
    r"thanks|thank\s+you|thx|"
    r"how(?:'s|\s+are)\s+(?:you|it\s+going)(?:\s+today)?|"
    r"what(?:'s|\s+is)\s+up|"
    r"who\s+are\s+you|"
    r"what(?:\s+can|\s+do)\s+you\s+do|"
    r"what\s+are\s+you"
    r")[\s!.?]*$",
    re.I,
)


class RuleRouter:
    def route(self, question: str) -> RouterLabel | None:
        text = question.strip()
        if _PURE_SOCIAL.match(text):
            return RouterLabel.GREETING
        return None
