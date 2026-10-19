"""Step 6: POST /ask over HTTP. Run from the project folder:

    py -m uvicorn api.app:app --reload          # then open http://127.0.0.1:8000/docs

The logic lives in rag.pipeline; this file only translates between HTTP and Python. The
pipeline comes in through a dependency, so tests swap in a fake with app.dependency_overrides
and never load a model or call Ollama. Set RAG_STORE=pg to search Postgres instead of memory,
and RAG_RETRIEVER=dense for the pre-step-8 retrieval (the default is hybrid).
"""
import os
from functools import cache
from typing import Annotated

import pydantic
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from rag.pipeline import Pipeline, default_pipeline

app = FastAPI(title="IRS publications Q&A", description="Answers only from five 2025 IRS publications, with checked citations.")


# ---- Request and response: the API contract ---------------------------------------------------

class AskRequest(BaseModel):
    # Blank or whitespace-only questions get a 422 before our code runs; the length cap keeps
    # one request from sending an unbounded prompt to the model.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str | None          # None when the model cited a chunk it wasn't given
    page: int | None
    quote: str
    cited_chunk_was_retrieved: bool
    quote_found_in_chunk: bool


class SourceOut(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    score: float


class AskResponse(BaseModel):
    question: str
    answerable: bool
    answer: str
    citations: list[CitationOut]
    all_citations_valid: bool   # every citation passed both checks, and an answer has at least one
    problems: list[str]         # why not, in words; empty when all_citations_valid
    sources: list[SourceOut]    # the chunks the model was given, best first
    usage: dict                 # input_tokens, output_tokens
    timings: dict               # seconds per stage


# ---- Dependency: built once, on the first request ---------------------------------------------

@cache
def get_pipeline() -> Pipeline:
    # Step 8's best configuration by default; RAG_RETRIEVER=dense gives the step 5-7 baseline.
    return default_pipeline(store_kind=os.environ.get("RAG_STORE", "memory"),
                            retriever=os.environ.get("RAG_RETRIEVER", "hybrid"))


# ---- Endpoints ---------------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: Annotated[Pipeline, Depends(get_pipeline)]) -> AskResponse:
    try:
        result = pipeline.ask(req.question)
    except ConnectionError:
        # The ollama client raises ConnectionError when the Ollama server isn't running.
        raise HTTPException(status_code=503, detail="The language model is unavailable. Is Ollama running?") from None
    except pydantic.ValidationError:
        # The model's reply didn't fit the Answer schema: our upstream failed, not the caller.
        raise HTTPException(status_code=502, detail="The language model returned a malformed answer.") from None

    return AskResponse(
        question=req.question,
        answerable=result.answer.answerable,
        answer=result.answer.answer,
        citations=[CitationOut(chunk_id=c.chunk_id, doc_id=c.doc_id, page=c.page, quote=c.quote,
                               cited_chunk_was_retrieved=c.cited_chunk_was_retrieved,
                               quote_found_in_chunk=c.quote_found_in_chunk)
                   for c in result.citations],
        all_citations_valid=not result.problems,
        problems=result.problems,
        sources=[SourceOut(chunk_id=c["chunk_id"], doc_id=c["doc_id"], page=c["page"], score=c["score"])
                 for c in result.retrieved],
        usage=result.usage,
        timings={k: round(v, 3) for k, v in result.timings.items()},
    )


@app.get("/health")
def health() -> dict:
    """Liveness only: doesn't load the model or touch the store."""
    return {"status": "ok"}
