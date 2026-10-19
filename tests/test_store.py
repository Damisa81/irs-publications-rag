import numpy as np
import pytest

from rag.store import InMemoryStore, search_exact


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_chunks(n):
    return [{"chunk_id": f"d-p1-c{i}", "doc_id": "d", "page": 1, "text": f"text {i}"} for i in range(n)]


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(50, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return make_chunks(50), vecs


def test_matches_search_exact(data):
    chunks, vecs = data
    store = InMemoryStore(dimension=8)
    store.add(chunks, vecs)
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = unit(rng.normal(size=8))
        expected = [chunks[i]["chunk_id"] for i in search_exact(q, vecs, k=5)]
        assert [r["chunk_id"] for r in store.search(q, k=5)] == expected


def test_results_are_dicts_with_score_best_first(data):
    chunks, vecs = data
    store = InMemoryStore(dimension=8)
    store.add(chunks, vecs)
    results = store.search(vecs[3], k=3)
    assert set(results[0]) == {"chunk_id", "doc_id", "page", "text", "score"}
    assert results[0]["chunk_id"] == "d-p1-c3"
    assert results[0]["score"] == pytest.approx(1.0)
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)


def test_query_length_does_not_matter(data):
    chunks, vecs = data
    store = InMemoryStore(dimension=8)
    store.add(chunks, vecs)
    assert store.search(vecs[7] * 5, k=5) == store.search(vecs[7], k=5)


def test_duplicate_ids_keep_the_first_version(data):
    chunks, vecs = data
    store = InMemoryStore(dimension=8)
    store.add(chunks[:1], vecs[:1])
    store.add([dict(chunks[0], text="changed")], vecs[1:2])
    assert len(store.chunks) == 1
    assert store.search(vecs[0], k=1)[0]["text"] == "text 0"


def test_search_sees_chunks_added_after_a_search(data):
    chunks, vecs = data
    store = InMemoryStore(dimension=8)
    store.add(chunks[:10], vecs[:10])
    store.search(vecs[0], k=1)                    # builds the matrix
    store.add(chunks[10:], vecs[10:])             # must invalidate it
    assert store.search(vecs[42], k=1)[0]["chunk_id"] == "d-p1-c42"


def test_k_larger_than_store_and_empty_store():
    store = InMemoryStore(dimension=8)
    assert store.search(unit(np.ones(8)), k=5) == []
    store.add(make_chunks(2), np.eye(8, dtype=np.float32)[:2])
    assert len(store.search(unit(np.ones(8)), k=5)) == 2
    assert store.search(unit(np.ones(8)), k=0) == []


@pytest.mark.parametrize("bad", [np.zeros(8), np.ones(7), np.array([np.nan] + [1.0] * 7)])
def test_bad_vectors_rejected(bad):
    store = InMemoryStore(dimension=8)
    with pytest.raises(ValueError):
        store.add(make_chunks(1), [bad])
    store.add(make_chunks(1), [unit(np.ones(8))])
    with pytest.raises(ValueError):
        store.search(bad)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        InMemoryStore(dimension=8).add(make_chunks(2), [unit(np.ones(8))])
