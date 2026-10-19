import json
from pathlib import Path

import pytest

from eval import metrics


def chunk(doc, page, text="x"):
    return {"chunk_id": f"{doc}-p{page}-c1", "doc_id": doc, "page": page, "text": text}


RETRIEVED = [chunk("p463", 34, "the rate"), chunk("p463", 21), chunk("p463", 44, "is 70 cents ($0.70) per mile")]


def test_rank_hit_and_reciprocal_rank():
    assert metrics.first_relevant_rank(RETRIEVED, ["p463-p44", "p463-p1"]) == 3
    assert metrics.page_hit(RETRIEVED, ["p463-p44"])
    assert metrics.reciprocal_rank(RETRIEVED, ["p463-p44"]) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(RETRIEVED, ["p463-p34", "p463-p44"]) == 1.0
    assert metrics.reciprocal_rank(RETRIEVED, ["p17-p3"]) == 0.0
    assert not metrics.page_hit(RETRIEVED, ["p17-p3"])
    assert not metrics.page_hit(RETRIEVED, [])


def test_page_needs_the_document_too():
    assert not metrics.page_hit(RETRIEVED, ["p17-p44"])      # same page number, other publication


def test_all_groups_hit():
    assert metrics.all_groups_hit(RETRIEVED, [["p463-p44"], ["p463-p21", "p334-p35"]])
    assert not metrics.all_groups_hit(RETRIEVED, [["p463-p44"], ["p334-p35"]])


def test_evidence_hit_needs_the_text_in_a_retrieved_chunk():
    assert metrics.evidence_hit(RETRIEVED, [["70 cents ($0.70) per mile"]])
    # Page 34 is an answer page, but this chunk of it doesn't contain the sentence.
    assert not metrics.evidence_hit([RETRIEVED[0]], [["70 cents ($0.70) per mile"]])
    assert metrics.evidence_hit(RETRIEVED, [["IS 70 CENTS"], ["the rate", "nope"]])
    assert not metrics.evidence_hit(RETRIEVED, [["70 cents"], ["nope"]])


def test_percentile_and_mean():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert metrics.percentile(values, 50) == 5
    assert metrics.percentile(values, 95) == 10
    assert metrics.percentile([3.0], 95) == 3.0
    assert metrics.percentile([], 50) is None
    assert metrics.mean([True, False, None, True]) == pytest.approx(2 / 3)
    assert metrics.mean([]) is None


def test_question_file_is_consistent():
    qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(qs) == 50 and len({q["id"] for q in qs}) == 50
    assert sum(q["answerable"] for q in qs) == 40
    assert sum(q["split"] == "test" for q in qs) == 10
    for q in qs:
        if q["answerable"]:
            assert q["pages"] and all(q["page_groups"]) and len(q["page_groups"]) == len(q["evidence"])
            assert set(q["pages"]) == {p for g in q["page_groups"] for p in g}
        else:
            assert q["pages"] == [] and q["evidence"] == []
