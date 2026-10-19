"""Step 7 scoring: pure functions, so they're unit-tested without a model or a database.

Retrieval is scored two ways:
  - by page (the guide's definition): a retrieved chunk counts if its page answers the question,
    even when the answer is in a different chunk of that page;
  - by evidence (stricter): a retrieved chunk must itself contain an evidence sentence, i.e. the
    model was actually shown the answer.
The gap between the two is how often "right page, wrong chunk" happens.
"""
import math
import re
import statistics

from rag.pipeline import normalize


def squash(text: str) -> str:
    """Same matching as eval.make_questions: ignore spaces, hyphens, quote style and case."""
    return re.sub(r"[\s-]+", "", normalize(text)).lower()


def page_of(chunk: dict) -> str:
    return f"{chunk['doc_id']}-p{chunk['page']}"


def first_relevant_rank(retrieved: list[dict], pages: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk on an expected page, or None."""
    wanted = set(pages)
    for rank, chunk in enumerate(retrieved, start=1):
        if page_of(chunk) in wanted:
            return rank
    return None


def reciprocal_rank(retrieved: list[dict], pages: list[str]) -> float:
    rank = first_relevant_rank(retrieved, pages)
    return 1 / rank if rank else 0.0


def page_hit(retrieved: list[dict], pages: list[str]) -> bool:
    return first_relevant_rank(retrieved, pages) is not None


def all_groups_hit(retrieved: list[dict], page_groups: list[list[str]]) -> bool:
    """For multi-part questions: every part has at least one of its pages retrieved."""
    got = {page_of(c) for c in retrieved}
    return all(got & set(group) for group in page_groups)


def evidence_hit(retrieved: list[dict], evidence: list[list[str]]) -> bool:
    """Every evidence group has a sentence inside some retrieved chunk's text."""
    texts = [squash(c["text"]) for c in retrieved]
    return all(any(squash(s) in t for s in group for t in texts) for group in evidence)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0-100); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))      # smallest value with >= p% at or below it
    return ordered[rank - 1]


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None
