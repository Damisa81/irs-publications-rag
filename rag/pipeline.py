"""Step 5: ask(question) = retrieve + generate + citation checks.

The citation checks run in code, not by asking the model: the model can't be the only judge
of its own grounding. Each citation must (1) point at a chunk that was actually retrieved and
(2) quote that chunk's text character for character, ignoring only spaces, hyphens and the
style of quote marks (see quote_in_text).
"""
import re
import time
from dataclasses import dataclass, field

import numpy as np

from rag.chunk import load_chunks
from rag.embed import embed
from rag.generate import Answer, generate
from rag.store import InMemoryStore, PgVectorStore

# Typographic quotes and dashes -> plain ASCII, so a model writing ' for ’ isn't flagged as
# misquoting.
_PLAIN = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", " ": " "})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_PLAIN)).strip()


def quote_in_text(quote: str, text: str) -> bool:
    """True if quote appears in text, ignoring spaces, hyphens and quote-mark style.

    PDF extraction splits words ("operat- ing", "Y ou", "business -related") and models
    usually repair them when quoting; on the first real run, a correct citation was flagged
    because the model wrote "operating". Every other character must still match, so a
    changed word or number ("65.5 cents" for "70 cents") is still caught.
    """
    def squash(s: str) -> str:
        return re.sub(r"[\s-]+", "", normalize(s))

    q = squash(quote)
    # An empty quote "appears" in every text, so it never counts as found.
    return bool(q) and q in squash(text)


@dataclass
class CitationCheck:
    chunk_id: str
    quote: str
    doc_id: str | None               # from the retrieved chunk; None if it wasn't retrieved
    page: int | None
    cited_chunk_was_retrieved: bool
    quote_found_in_chunk: bool

    @property
    def valid(self) -> bool:
        return self.cited_chunk_was_retrieved and self.quote_found_in_chunk


def check_citations(answer: Answer, retrieved: list[dict]) -> list[CitationCheck]:
    by_id = {c["chunk_id"]: c for c in retrieved}
    checks = []
    for cit in answer.citations:
        chunk = by_id.get(cit.chunk_id)
        checks.append(CitationCheck(
            chunk_id=cit.chunk_id,
            quote=cit.quote,
            doc_id=chunk["doc_id"] if chunk else None,
            page=chunk["page"] if chunk else None,
            cited_chunk_was_retrieved=chunk is not None,
            quote_found_in_chunk=chunk is not None and quote_in_text(cit.quote, chunk["text"]),
        ))
    return checks


@dataclass
class Result:
    question: str
    answer: Answer
    citations: list[CitationCheck]
    retrieved: list[dict]                       # the chunks the model was given, best first
    usage: dict                                 # input_tokens, output_tokens
    timings: dict = field(default_factory=dict)  # seconds per stage

    @property
    def problems(self) -> list[str]:
        """Why this answer shouldn't be trusted as-is; empty when every check passes."""
        out = []
        if self.answer.answerable and not self.citations:
            out.append("answered without any citation")
        for c in self.citations:
            if not c.cited_chunk_was_retrieved:
                out.append(f"cites {c.chunk_id}, which was not retrieved")
            elif not c.quote_found_in_chunk:
                out.append(f"quote not found word for word in {c.chunk_id}")
        return out


class Pipeline:
    def __init__(self, store, embed_fn=embed, generate_fn=generate, k: int = 5, retriever=None):
        self.store = store
        self.embed_fn = embed_fn
        self.generate_fn = generate_fn
        self.k = k
        self.retriever = retriever

    def retrieve(self, question: str) -> tuple[list[dict], dict]:
        """The k best chunks for a question, and seconds spent getting them.

        Without a retriever: embed + store search (timed separately). With one (rag.retrieve,
        step 8: BM25, hybrid, reranking), it does the whole job and is timed as retrieve_s.
        """
        if self.retriever is not None:
            t0 = time.perf_counter()
            retrieved = self.retriever.retrieve(question, self.k)
            return retrieved, {"retrieve_s": time.perf_counter() - t0}
        t0 = time.perf_counter()
        query_vec = self.embed_fn([question])[0]
        t1 = time.perf_counter()
        retrieved = self.store.search(query_vec, k=self.k)
        t2 = time.perf_counter()
        return retrieved, {"embed_s": t1 - t0, "search_s": t2 - t1}

    def ask(self, question: str) -> Result:
        retrieved, timings = self.retrieve(question)
        t0 = time.perf_counter()
        answer, usage = self.generate_fn(question, retrieved)
        timings["generate_s"] = time.perf_counter() - t0
        timings["total_s"] = sum(timings.values())
        checks = check_citations(answer, retrieved)
        return Result(question=question, answer=answer, citations=checks, retrieved=retrieved,
                      usage=usage, timings=timings)


def chunk_files(size: int = 250, overlap: int = 40) -> tuple[str, str]:
    """Where scripts.build_index saves one chunking configuration's chunks and vectors.
    The default 250/40 keeps the original names; others get a suffix, so experiments don't
    overwrite the baseline (and its 80-second embedding)."""
    suffix = "" if (size, overlap) == (250, 40) else f"_{size}_{overlap}"
    return f"data/chunks{suffix}.jsonl", f"data/chunk_vecs{suffix}.npy"


def load_store(kind: str = "memory", table: str = "chunks_250_40",
               chunks_path: str = "data/chunks.jsonl", vecs_path: str = "data/chunk_vecs.npy"):
    """The store built by scripts.build_index: in memory from the saved files, or Postgres."""
    if kind == "pg":
        return PgVectorStore(table=table)
    store = InMemoryStore()
    store.add(load_chunks(chunks_path), np.load(vecs_path))
    return store


def default_pipeline(store_kind: str = "memory", retriever: str = "hybrid", k: int = 5) -> "Pipeline":
    """The system as configured after step 8: hybrid retrieval (dense + BM25), prompt v1.
    retriever="dense" gives the step 5-7 baseline."""
    from rag.retrieve import make_retriever       # imported here: rag.retrieve imports this module

    store = load_store(store_kind)
    if retriever == "dense":
        return Pipeline(store, k=k)
    return Pipeline(store, k=k, retriever=make_retriever(retriever, store, load_chunks("data/chunks.jsonl")))
