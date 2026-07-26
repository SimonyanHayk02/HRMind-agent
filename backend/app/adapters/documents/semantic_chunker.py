from __future__ import annotations


def chunk_sections(sections: dict[str, str], *, max_chars: int = 800) -> list[tuple[str, str, int]]:
    """Return list of (section, content, chunk_index). Section-first; split long sections."""
    out: list[tuple[str, str, int]] = []
    for section, body in sections.items():
        parts = _split_long(body, max_chars=max_chars)
        for idx, part in enumerate(parts):
            if part.strip():
                out.append((section, part.strip(), idx))
    return out


def _split_long(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    # split on bullets/paragraphs
    pieces = []
    buf = ""
    for line in text.splitlines():
        if len(buf) + len(line) + 1 > max_chars and buf:
            pieces.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        pieces.append(buf)
    return pieces or [text]
