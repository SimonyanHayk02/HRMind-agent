from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def parse_pdf(path: str | Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)
