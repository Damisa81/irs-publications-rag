"""Pipeline and citation checks with a fake store, embedder and generator: no model, no database."""
import numpy as np

from rag.generate import Answer, Citation
from rag.pipeline import Pipeline, check_citations, normalize

CHUNKS = [
    {"chunk_id": "p463-p1-c2", "doc_id": "p463", "page": 1, "score": 0.9,
     "text": "For 2025, the standard mileage rate for the cost of operating your car for business\n"
             "use is 70 cents ($0.70) per mile. You can’t use it for a fleet."},
    {"chunk_id": "p587-p3-c1", "doc_id": "p587", "page": 3, "score": 0.8,
     "text": "You do not meet the requirements of the exclusive use test if you use the area "
             "in question both for business and for personal purposes."},
]


def answer(*citations, answerable=True):
    return Answer(answerable=answerable, answer="...",
                  citations=[Citation(chunk_id=i, quote=q) for i, q in citations])


class FakeStore:
    def search(self, query_vec, k=5):
        return CHUNKS[:k]


def run(ans):
    pipeline = Pipeline(
        FakeStore(),
        embed_fn=lambda texts: np.ones((len(texts), 384), dtype=np.float32),
        generate_fn=lambda q, chunks: (ans, {"input_tokens": 100, "output_tokens": 20}),
    )
    return pipeline.ask("What is the standard mileage rate?")


def test_valid_citation_passes():
    r = run(answer(("p463-p1-c2", "is 70 cents ($0.70) per mile")))
    assert r.citations[0].valid
    assert (r.citations[0].doc_id, r.citations[0].page) == ("p463", 1)
    assert r.problems == []


def test_citation_to_chunk_not_retrieved_is_flagged():
    r = run(answer(("p17-p9-c1", "is 70 cents ($0.70) per mile")))
    c = r.citations[0]
    assert not c.cited_chunk_was_retrieved and not c.valid
    assert c.doc_id is None
    assert r.problems == ["cites p17-p9-c1, which was not retrieved"]


def test_made_up_quote_is_flagged():
    r = run(answer(("p463-p1-c2", "is 65.5 cents per mile")))
    assert r.citations[0].cited_chunk_was_retrieved
    assert not r.citations[0].quote_found_in_chunk
    assert r.problems == ["quote not found word for word in p463-p1-c2"]


def test_quote_from_the_wrong_retrieved_chunk_is_flagged():
    r = run(answer(("p587-p3-c1", "is 70 cents ($0.70) per mile")))
    assert not r.citations[0].valid


def test_line_breaks_and_curly_quotes_are_normalized():
    # The chunk has a line break inside the quote and a curly apostrophe in "can’t".
    r = run(answer(("p463-p1-c2", "operating your car for business use is 70 cents"),
                   ("p463-p1-c2", "You can't use it for a fleet.")))
    assert all(c.valid for c in r.citations)


def test_repaired_pdf_word_splits_still_match():
    # Real case from the first run: the chunk says "operat- ing", the model quoted "operating".
    chunks = [{"chunk_id": "c", "doc_id": "p463", "page": 44,
               "text": "the cost of operat- ing your car. Y ou can deduct business -related park- ing fees."}]
    a = answer(("c", "the cost of operating your car."),
               ("c", "You can deduct business-related parking fees."))
    assert all(c.valid for c in check_citations(a, chunks))


def test_changed_number_or_word_is_still_caught():
    chunks = [{"chunk_id": "c", "doc_id": "p463", "page": 44, "text": "is 70 cents ($0.70) per mile."}]
    for wrong in ["is 65.5 cents per mile.", "is 70 dollars ($0.70) per mile.", "is 7 0 cent ($0.70) per mile."]:
        assert not check_citations(answer(("c", wrong)), chunks)[0].valid


def test_empty_quote_is_not_found():
    for empty in ["   ", " - ", ""]:
        r = run(answer(("p463-p1-c2", empty)))
        assert not r.citations[0].quote_found_in_chunk


def test_answerable_without_citations_is_flagged():
    r = run(answer())
    assert r.problems == ["answered without any citation"]


def test_unanswerable_without_citations_is_fine():
    r = run(answer(answerable=False))
    assert r.problems == [] and r.citations == []


def test_one_bad_citation_among_good_ones_is_reported():
    r = run(answer(("p463-p1-c2", "70 cents ($0.70) per mile"),
                   ("p587-p3-c1", "exclusive use test"),
                   ("p587-p3-c1", "regular use test")))
    assert [c.valid for c in r.citations] == [True, True, False]
    assert len(r.problems) == 1


def test_result_has_usage_timings_and_retrieved_chunks():
    r = run(answer(("p463-p1-c2", "70 cents")))
    assert r.usage == {"input_tokens": 100, "output_tokens": 20}
    assert set(r.timings) == {"embed_s", "search_s", "generate_s", "total_s"}
    assert [c["chunk_id"] for c in r.retrieved] == ["p463-p1-c2", "p587-p3-c1"]


def test_check_citations_directly():
    checks = check_citations(answer(("p587-p3-c1", "both for business and for personal purposes")), CHUNKS)
    assert checks[0].valid


def test_normalize():
    assert normalize("  a\n\tb c “d” – e’s ") == 'a b c "d" - e\'s'
