from app.adapters.documents.section_splitter import split_sections
from app.adapters.documents.semantic_chunker import chunk_sections


def test_section_splitter() -> None:
    text = """Summary
Hello world

Skills
Python, Go

Education
BSc
"""
    sections = split_sections(text)
    assert "Summary" in sections
    assert "Skills" in sections
    assert "Python" in sections["Skills"]


def test_semantic_chunker_splits_long() -> None:
    sections = {"Experience": "x\n" * 500}
    chunks = chunk_sections(sections, max_chars=100)
    assert len(chunks) > 1
    assert all(s == "Experience" for s, _, _ in chunks)
