"""API tests through TestClient with a fake pipeline: no model, no Ollama, no database."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import app, get_pipeline
from rag.generate import Answer, Citation
from rag.pipeline import Pipeline

CHUNKS = [
    {"chunk_id": "p463-p44-c5", "doc_id": "p463", "page": 44, "score": 0.82,
     "text": "For 2025, the standard mileage rate for the cost of operat- ing your car for business use "
             "is 70 cents ($0.70) per mile."},
    {"chunk_id": "p463-p34-c3", "doc_id": "p463", "page": 34, "score": 0.88, "text": "Depreciation was included."},
]
GOOD = Answer(answerable=True, answer="70 cents per mile.", citations=[
    Citation(chunk_id="p463-p44-c5", quote="the cost of operating your car for business use is 70 cents ($0.70) per mile")])


class FakeStore:
    def search(self, query_vec, k=5):
        return CHUNKS[:k]


def fake_pipeline(generate_fn):
    return Pipeline(FakeStore(), embed_fn=lambda texts: np.ones((len(texts), 384), dtype=np.float32),
                    generate_fn=generate_fn)


@pytest.fixture
def client_with():
    """client_with(generate_fn) -> a TestClient whose pipeline uses that fake generator."""
    def make(generate_fn):
        app.dependency_overrides[get_pipeline] = lambda: fake_pipeline(generate_fn)
        return TestClient(app)
    yield make
    app.dependency_overrides.clear()


def answering(answer):
    return lambda question, chunks: (answer, {"input_tokens": 1500, "output_tokens": 40})


def test_answer_with_checked_citation(client_with):
    r = client_with(answering(GOOD)).post("/ask", json={"question": "What is the mileage rate?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"] is True
    assert body["all_citations_valid"] is True and body["problems"] == []
    c = body["citations"][0]
    assert (c["doc_id"], c["page"]) == ("p463", 44)
    assert c["cited_chunk_was_retrieved"] and c["quote_found_in_chunk"]
    assert [s["chunk_id"] for s in body["sources"]] == ["p463-p44-c5", "p463-p34-c3"]
    assert body["usage"] == {"input_tokens": 1500, "output_tokens": 40}
    assert set(body["timings"]) == {"embed_s", "search_s", "generate_s", "total_s"}


def test_bad_citation_is_reported_not_hidden(client_with):
    bad = Answer(answerable=True, answer="65.5 cents.", citations=[
        Citation(chunk_id="p463-p44-c5", quote="is 65.5 cents per mile"),
        Citation(chunk_id="p17-p1-c1", quote="anything")])
    body = client_with(answering(bad)).post("/ask", json={"question": "Mileage rate?"}).json()
    assert body["all_citations_valid"] is False
    assert len(body["problems"]) == 2
    assert body["citations"][1]["doc_id"] is None and body["citations"][1]["page"] is None


def test_unanswerable(client_with):
    no = Answer(answerable=False, answer="The sources don't cover Canadian lottery tax.", citations=[])
    body = client_with(answering(no)).post("/ask", json={"question": "Lottery tax in Canada?"}).json()
    assert body["answerable"] is False and body["citations"] == [] and body["all_citations_valid"] is True


def test_question_is_stripped(client_with):
    body = client_with(answering(GOOD)).post("/ask", json={"question": "  Mileage rate?  "}).json()
    assert body["question"] == "Mileage rate?"


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "    "},
    {"question": "x" * 1001},
    {"question": 42},
    {},
])
def test_invalid_requests_get_422_and_never_reach_the_model(client_with, payload):
    calls = []
    client = client_with(lambda q, c: calls.append(q) or (GOOD, {}))
    r = client.post("/ask", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][:2] == ["body", "question"] or r.json()["detail"][0]["loc"] == ["body"]
    assert calls == []


def test_ollama_down_is_503(client_with):
    def down(question, chunks):
        raise ConnectionError("Failed to connect to Ollama")
    r = client_with(down).post("/ask", json={"question": "Mileage rate?"})
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]


def test_malformed_model_output_is_502(client_with):
    def malformed(question, chunks):
        return Answer.model_validate_json('{"answer": "no answerable field"}'), {}
    r = client_with(malformed).post("/ask", json={"question": "Mileage rate?"})
    assert r.status_code == 502


def test_health_needs_no_pipeline():
    app.dependency_overrides.clear()
    assert TestClient(app).get("/health").json() == {"status": "ok"}
    assert get_pipeline.cache_info().currsize == 0     # the real pipeline was never built
