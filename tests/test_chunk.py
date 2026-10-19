import json

import pytest

from rag.chunk import chunk_pages, load_chunks, save_chunks


def make_page(n_words, doc_id="p463", page=1):
    # Distinct words ("w0 w1 w2 ...") so overlaps and gaps are easy to check.
    return {"doc_id": doc_id, "page": page, "text": " ".join(f"w{i}" for i in range(n_words))}


def words(chunk):
    return chunk["text"].split()


def test_short_page_is_one_chunk():
    chunks = chunk_pages([make_page(100)])
    assert len(chunks) == 1
    assert words(chunks[0]) == words(make_page(100))


def test_page_just_under_size_has_no_duplicate_end_chunk():
    # Regression: a 240-word page used to produce a second chunk of words 210-240,
    # all of them already in the first chunk.
    assert len(chunk_pages([make_page(240)])) == 1


def test_page_of_exactly_size_is_one_chunk():
    assert len(chunk_pages([make_page(250)])) == 1


def test_long_page_windows():
    # Each chunk starts size - overlap = 210 words after the previous one.
    chunks = chunk_pages([make_page(600)])
    assert [words(c)[0] for c in chunks] == ["w0", "w210", "w420"]
    assert words(chunks[-1])[-1] == "w599"


def test_every_chunk_at_most_size_words():
    chunks = chunk_pages([make_page(n) for n in (1, 249, 250, 251, 600, 1234)])
    assert all(len(words(c)) <= 250 for c in chunks)


def test_consecutive_chunks_share_exactly_overlap_words():
    chunks = chunk_pages([make_page(1000)])
    for a, b in zip(chunks, chunks[1:]):
        assert words(a)[-40:] == words(b)[:40]
        assert len(set(words(a)) & set(words(b))) == 40


def test_no_words_lost():
    page = make_page(1000)
    chunks = chunk_pages([page])
    assert set().union(*(words(c) for c in chunks)) == set(words(page))


def test_custom_size_and_overlap():
    # Windows of 10 words, stepping 7: [0, 10), [7, 17), [14, 24) covers all 24 words.
    chunks = chunk_pages([make_page(24)], size=10, overlap=3)
    assert [words(c)[0] for c in chunks] == ["w0", "w7", "w14"]
    assert all(len(words(c)) <= 10 for c in chunks)


def test_ids_unique_and_chunks_stay_on_their_page():
    pages = [make_page(600, "p463", 1), make_page(600, "p463", 2), make_page(600, "p587", 1)]
    chunks = chunk_pages(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[:3] == ["p463-p1-c1", "p463-p1-c2", "p463-p1-c3"]
    for c in chunks:
        assert c["chunk_id"].startswith(f"{c['doc_id']}-p{c['page']}-")


def test_empty_page_gives_no_chunks():
    assert chunk_pages([make_page(0)]) == []


@pytest.mark.parametrize("size, overlap", [(40, 40), (40, 50), (250, -1)])
def test_invalid_overlap_raises(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([make_page(100)], size=size, overlap=overlap)


def test_save_chunks_round_trip(tmp_path):
    chunks = chunk_pages([make_page(600)])
    path = tmp_path / "out" / "chunks.jsonl"     # parent folder doesn't exist yet
    save_chunks(chunks, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == chunks
    assert load_chunks(path) == chunks
