"""Steps 3 and 8: retrievers. Each one has retrieve(question, k) -> chunk dicts with a "score", best first.

    DenseRetriever    embeddings + a store (step 3/4), optionally with bge's query prefix
    BM25Retriever     keyword search: exact terms such as "$25", "Section 179", "Form 1099-MISC"
    HybridRetriever   merges several retrievers by reciprocal rank fusion (RRF)
    RerankRetriever   re-scores another retriever's candidates with a cross-encoder

They wrap each other, so a configuration is just a composition, e.g.
    RerankRetriever(HybridRetriever([DenseRetriever(store), BM25Retriever(chunks)]))
"""
import re
from functools import cache

import numpy as np
from rank_bm25 import BM25Okapi

from rag.embed import embed
from rag.pipeline import normalize

# bge-small-en-v1.5's recommended instruction for short queries searching longer passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    def __init__(self, store, embed_fn=embed, query_prefix: str = ""):
        self.store = store
        self.embed_fn = embed_fn
        self.query_prefix = query_prefix

    def retrieve(self, question: str, k: int) -> list[dict]:
        query_vec = self.embed_fn([self.query_prefix + question])[0]
        return self.store.search(query_vec, k=k)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with hyphens inside words removed so the PDF's line-break splits
    ("operat- ing", "self- employed") match the query's spelling ("operating", "self-employed")."""
    text = normalize(text).lower()
    text = re.sub(r"(?<=[a-z])\s*-\s*(?=[a-z])", "", text)
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text)


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = [{key: c[key] for key in ("chunk_id", "doc_id", "page", "text")} for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def retrieve(self, question: str, k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(question))
        top = np.argsort(-scores, kind="stable")[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


class HybridRetriever:
    """Reciprocal rank fusion: each chunk scores the sum of 1 / (rrf_k + rank) over the lists it
    appears in. Only ranks are used, so dense cosine scores and BM25 scores never need rescaling."""

    def __init__(self, retrievers: list, candidates: int = 20, rrf_k: int = 60):
        self.retrievers = retrievers
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int) -> list[dict]:
        fused, chunks = {}, {}
        for retriever in self.retrievers:
            for rank, c in enumerate(retriever.retrieve(question, self.candidates), start=1):
                fused[c["chunk_id"]] = fused.get(c["chunk_id"], 0.0) + 1 / (self.rrf_k + rank)
                chunks.setdefault(c["chunk_id"], c)
        best = sorted(fused, key=lambda cid: -fused[cid])[:k]
        return [{**chunks[cid], "score": fused[cid]} for cid in best]


@cache
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


class RerankRetriever:
    """Retrieve `candidates` chunks from `base`, then keep the k a cross-encoder scores highest.
    A cross-encoder reads the question and the chunk together, so it can tell "states the fact"
    from "is about the topic", which a single embedding per chunk can't."""

    def __init__(self, base, candidates: int = 20,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", score_fn=None):
        self.base = base
        self.candidates = candidates
        self.model_name = model_name
        self.score_fn = score_fn      # tests pass a fake; default is the cross-encoder

    def retrieve(self, question: str, k: int) -> list[dict]:
        cands = self.base.retrieve(question, self.candidates)
        if not cands:
            return []
        score = self.score_fn or (lambda pairs: _cross_encoder(self.model_name).predict(pairs))
        scores = score([(question, c["text"]) for c in cands])
        order = np.argsort(-np.asarray(scores), kind="stable")[:k]
        return [{**cands[i], "score": float(scores[i])} for i in order]


def make_retriever(kind: str, store, chunks: list[dict], prefix: bool = False, rerank: int = 0,
                   candidates: int = 20, embed_fn=embed):
    """Build a configuration by name. Step 8's winner is "hybrid" without prefix or reranking.

    kind: "dense" (store only), "bm25" (keywords only) or "hybrid" (both, fused by RRF).
    chunks: the same chunks the store holds, for BM25 (built in memory in about a second).
    """
    dense = DenseRetriever(store, embed_fn=embed_fn, query_prefix=BGE_QUERY_PREFIX if prefix else "")
    if kind == "dense":
        base = dense
    elif kind == "bm25":
        base = BM25Retriever(chunks)
    elif kind == "hybrid":
        base = HybridRetriever([dense, BM25Retriever(chunks)], candidates=candidates)
    else:
        raise ValueError(f"unknown retriever {kind!r}")
    return RerankRetriever(base, candidates=rerank) if rerank else base
