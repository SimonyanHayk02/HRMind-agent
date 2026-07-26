from __future__ import annotations

from pathlib import Path

from docx import Document


def parse_docx(path: str | Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)
