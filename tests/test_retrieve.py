"""Retrievers with a fake embedder and fake rerank scores: no models are loaded."""
import numpy as np
import pytest

from rag.pipeline import Pipeline
from rag.retrieve import (BGE_QUERY_PREFIX, BM25Retriever, DenseRetriever, HybridRetriever,
                          RerankRetriever, tokenize)

CHUNKS = [
    {"chunk_id": "gift", "doc_id": "p463", "page": 16, "text": "You can deduct no more than $25 for business gifts."},
    {"chunk_id": "car", "doc_id": "p463", "page": 44, "text": "The cost of operat- ing your car is 70 cents per mile."},
    {"chunk_id": "home", "doc_id": "p587", "page": 3, "text": "Exclusive use test for a home office."},
    {"chunk_id": "se", "doc_id": "p334", "page": 43, "text": "The self- employment tax rate is 15.3% for 2025."},
]


class ListRetriever:
    """Returns a fixed ranking, to test the wrappers in isolation."""
    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, question, k):
        by_id = {c["chunk_id"]: c for c in CHUNKS}
        return [{**by_id[i], "score": 1.0} for i in self.ids[:k]]


def test_tokenize_rejoins_pdf_hyphen_splits_and_keeps_numbers():
    assert tokenize("The cost of operat- ing your car") == ["the", "cost", "of", "operating", "your", "car"]
    assert tokenize("self- employment") == tokenize("self-employment") == ["selfemployment"]
    assert tokenize("$25 and 15.3% and 1,000") == ["25", "and", "15.3", "and", "1,000"]


def test_bm25_finds_exact_terms():
    bm25 = BM25Retriever(CHUNKS)
    assert bm25.retrieve("deduct business gifts $25", k=1)[0]["chunk_id"] == "gift"
    assert bm25.retrieve("operating a car", k=1)[0]["chunk_id"] == "car"
    assert bm25.retrieve("self-employment tax", k=1)[0]["chunk_id"] == "se"
    top = bm25.retrieve("gifts", k=4)
    assert len(top) == 4 and set(top[0]) == {"chunk_id", "doc_id", "page", "text", "score"}


def test_dense_adds_the_prefix_only_when_asked():
    seen = []

    class Store:
        def search(self, vec, k):
            return CHUNKS[:k]

    def fake_embed(texts):
        seen.extend(texts)
        return np.ones((len(texts), 4))

    DenseRetriever(Store(), embed_fn=fake_embed).retrieve("q", 2)
    DenseRetriever(Store(), embed_fn=fake_embed, query_prefix=BGE_QUERY_PREFIX).retrieve("q", 2)
    assert seen == ["q", BGE_QUERY_PREFIX + "q"]


def test_rrf_rewards_chunks_both_lists_agree_on():
    hybrid = HybridRetriever([ListRetriever(["gift", "car", "home"]), ListRetriever(["home", "car", "se"])],
                             candidates=3, rrf_k=60)
    top = hybrid.retrieve("q", k=4)
    # car is 2nd in both lists: 2/62 beats gift (1/61 only) and home (1/61 + 1/63 = 0.03227 > 2/62 = 0.03226)
    assert [c["chunk_id"] for c in top] == ["home", "car", "gift", "se"]
    assert top[0]["score"] == pytest.approx(1 / 61 + 1 / 63)
    assert top[1]["score"] == pytest.approx(2 / 62)


def test_rrf_respects_candidates_and_k():
    hybrid = HybridRetriever([ListRetriever(["gift", "car", "home", "se"])], candidates=2)
    assert [c["chunk_id"] for c in hybrid.retrieve("q", k=5)] == ["gift", "car"]


def test_rerank_reorders_candidates_by_score():
    scores = {"gift": 0.1, "car": 0.9, "home": 0.5, "se": 0.7}
    rerank = RerankRetriever(ListRetriever(["gift", "car", "home", "se"]), candidates=3,
                             score_fn=lambda pairs: [scores[next(c["chunk_id"] for c in CHUNKS if c["text"] == t)]
                                                     for _, t in pairs])
    top = rerank.retrieve("q", k=2)
    assert [c["chunk_id"] for c in top] == ["car", "home"]     # "se" wasn't among the 3 candidates
    assert top[0]["score"] == pytest.approx(0.9)


def test_pipeline_uses_the_retriever_and_times_it():
    def fake_generate(question, chunks):
        from rag.generate import Answer
        return Answer(answerable=False, answer="-", citations=[]), {}

    pipeline = Pipeline(store=None, generate_fn=fake_generate, k=2, retriever=ListRetriever(["se", "gift"]))
    result = pipeline.ask("q")
    assert [c["chunk_id"] for c in result.retrieved] == ["se", "gift"]
    assert set(result.timings) == {"retrieve_s", "generate_s", "total_s"}
