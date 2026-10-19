"""SearchAgent with a scripted fake model: no Ollama calls."""
from types import SimpleNamespace as NS

from rag.agent import SearchAgent
from rag.generate import Answer, Citation

CHUNKS = {
    "rule": {"chunk_id": "rule", "doc_id": "p463", "page": 16, "text": "You can deduct no more than $25 per person."},
    "exc": {"chunk_id": "exc", "doc_id": "p463", "page": 17, "text": "An item that costs $4 or less isn't a gift."},
    "other": {"chunk_id": "other", "doc_id": "p17", "page": 3, "text": "Unrelated."},
}


class FakeRetriever:
    """'exception' queries find the exception; anything else finds the rule (and 'other')."""
    def __init__(self):
        self.queries = []

    def retrieve(self, query, k):
        self.queries.append(query)
        ids = ["exc", "other"] if "exception" in query else ["rule", "other"]
        return [{**CHUNKS[i], "score": 1.0} for i in ids[:k]]


def reply(*queries, text=""):
    """One model turn: tool calls for each query, or plain text when there are none."""
    calls = [NS(function=NS(name="search", arguments={"query": q})) for q in queries]
    return NS(message=NS(role="assistant", content=text, tool_calls=calls or None),
              prompt_eval_count=100, eval_count=10)


def scripted(*turns):
    turns = list(turns)
    seen = []

    def chat(**kwargs):
        seen.append(kwargs)
        return turns.pop(0)
    chat.seen = seen
    return chat


def fake_generate(question, chunks):
    ids = [c["chunk_id"] for c in chunks]
    cites = [Citation(chunk_id="rule", quote="no more than $25 per person")] if "rule" in ids else []
    return Answer(answerable=bool(cites), answer="$25.", citations=cites), {"input_tokens": 500, "output_tokens": 20}


def test_two_searches_then_done():
    retriever = FakeRetriever()
    chat = scripted(reply("business gift limit"), reply("gift limit exception"), reply(text="DONE"))
    result = SearchAgent(retriever, max_searches=3, k_per_search=2, chat_fn=chat, generate_fn=fake_generate).ask("q")
    assert [s["query"] for s in result.searches] == ["business gift limit", "gift limit exception"]
    assert [c["chunk_id"] for c in result.retrieved] == ["rule", "other", "exc"]     # deduplicated, in order
    assert result.stop_reason == "done"
    assert result.citations[0].valid
    assert result.usage == {"input_tokens": 3 * 100 + 500, "output_tokens": 3 * 10 + 20}
    assert set(result.timings) == {"research_s", "generate_s", "total_s"}
    # The model sees the tool results: the last call's messages end with a tool message.
    assert chat.seen[-1]["messages"][-1]["role"] == "tool"
    assert "chunk_id=\"exc\"" in chat.seen[-1]["messages"][-1]["content"]


def test_search_cap_is_enforced():
    retriever = FakeRetriever()
    chat = scripted(reply("a", "b"), reply("c", "d"), reply("e"))
    result = SearchAgent(retriever, max_searches=3, k_per_search=1, chat_fn=chat, generate_fn=fake_generate).ask("q")
    assert len(result.searches) == 3 and retriever.queries == ["a", "b", "c"]
    assert result.stop_reason == "max_searches"


def test_no_search_falls_back_to_the_question():
    retriever = FakeRetriever()
    chat = scripted(reply(text="The answer is $25."))
    result = SearchAgent(retriever, chat_fn=chat, generate_fn=fake_generate).ask("How much for gifts?")
    assert result.stop_reason == "no_search"
    assert result.searches == [{"query": "How much for gifts?", "chunk_ids": ["rule", "other"], "fallback": True}]
    assert result.citations[0].valid


def test_citation_to_a_chunk_never_found_is_flagged():
    def generate_bad(question, chunks):
        return Answer(answerable=True, answer="x", citations=[Citation(chunk_id="exc", quote="$4 or less")]), {}
    chat = scripted(reply("gift limit"), reply(text="DONE"))
    result = SearchAgent(FakeRetriever(), chat_fn=chat, generate_fn=generate_bad).ask("q")
    assert not result.citations[0].cited_chunk_was_retrieved
    assert result.problems == ["cites exc, which was not retrieved"]
