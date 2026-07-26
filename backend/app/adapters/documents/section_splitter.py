from __future__ import annotations

import re

SECTION_HEADERS = [
    "Summary",
    "Experience",
    "Skills",
    "Projects",
    "Languages",
    "Education",
    "Certificates",
]


def split_sections(text: str) -> dict[str, str]:
    pattern = re.compile(
        r"^(?P<header>" + "|".join(SECTION_HEADERS) + r")\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {"Summary": text.strip()}

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = match.group("header").title()
        # normalize Certificates/Certificate etc
        for h in SECTION_HEADERS:
            if h.lower() == header.lower():
                header = h
                break
        sections[header] = text[start:end].strip()
    return sections
