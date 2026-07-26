from app.domain.operators.count import Count
from app.domain.operators.extract_ids import ExtractEmployeeIds
from app.domain.operators.intersect_ids import IntersectIds


def test_extract_ids_from_hits() -> None:
    op = ExtractEmployeeIds()
    ids = op.run({"hits": [{"employee_id": "a"}, {"employee_id": "b"}, {"employee_id": "a"}]})
    assert ids == ["a", "b"]


def test_count_list() -> None:
    assert Count().run([1, 2, 3]) == {"count": 3}


def test_intersect() -> None:
    assert IntersectIds().run(["a", "b"], {"other": ["b", "c"]}) == ["b"]
