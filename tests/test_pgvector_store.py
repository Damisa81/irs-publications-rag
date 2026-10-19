"""PgVectorStore against a real database. Skipped unless DATABASE_URL is set.

Each test uses its own small 8-dimension table with a random name and drops it afterwards,
so it never touches the real chunks tables.
"""
import os
import uuid

import numpy as np
import pytest

from rag.store import InMemoryStore, PgVectorStore

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")


def make_chunks(n):
    return [{"chunk_id": f"d-p{i}-c1", "doc_id": "d", "page": i, "text": f"text {i}"} for i in range(n)]


@pytest.fixture
def vecs():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def store():
    s = PgVectorStore(table=f"test_{uuid.uuid4().hex[:12]}", dimension=8)
    yield s
    s.drop()
    s.close()


def test_same_results_as_in_memory_store(store, vecs):
    chunks = make_chunks(40)
    store.add(chunks, vecs)
    mem = InMemoryStore(dimension=8)
    mem.add(chunks, vecs)
    for q in vecs[:10]:
        got, want = store.search(q, k=5), mem.search(q, k=5)
        # HNSW is approximate, but 40 rows is far too few for it to miss anything.
        assert [r["chunk_id"] for r in got] == [r["chunk_id"] for r in want]
        assert [r["score"] for r in got] == pytest.approx([r["score"] for r in want], abs=1e-5)
        assert set(got[0]) == {"chunk_id", "doc_id", "page", "text", "score"}


def test_re_adding_keeps_row_count_and_first_version(store, vecs):
    chunks = make_chunks(40)
    store.add(chunks, vecs)
    store.add([dict(c, text="changed") for c in chunks], vecs)
    assert store.count() == 40
    assert store.search(vecs[0], k=1)[0]["text"] == "text 0"


def test_reset_empties_the_table(store, vecs):
    store.add(make_chunks(40), vecs)
    store.reset()
    assert store.count() == 0
    store.add(make_chunks(5), vecs[:5])
    assert store.count() == 5


def test_bad_vectors_rejected(store):
    with pytest.raises(ValueError):
        store.add(make_chunks(1), [np.zeros(8)])
    with pytest.raises(ValueError):
        store.search(np.ones(7))
