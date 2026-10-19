"""Step 4 "Done when": does pgvector's HNSW search return what exact search returns?

For each question, compares the top-10 chunk ids from Postgres with the exact top-10 from
InMemoryStore ("recall vs. exact"; expect about 1.0 at this size), checks the row count,
and times the database search after one warm-up query (a hosted database may be asleep).

Run after `py -m scripts.build_index --pg`, from the project folder:
    py -m scripts.check_pgvector
    py -m scripts.check_pgvector --size 400 --overlap 60
"""
import argparse
import statistics
import time

import numpy as np

from rag.chunk import load_chunks
from rag.embed import embed
from rag.pipeline import chunk_files
from rag.store import InMemoryStore, PgVectorStore
from scripts.build_index import pg_table

QUESTIONS = [
    "What is the standard mileage rate for business use of a car?",
    "Can I deduct a home office if I also use the room for personal use?",
    "How long is the recovery period for residential rental property?",
    "Who must file a federal income tax return?",
    "What is the standard deduction for married filing jointly?",
    "Can I claim my elderly parent as a dependent?",
    "How do I report tip income?",
    "What is self-employment tax and who has to pay it?",
    "Which business expenses can a sole proprietor deduct on Schedule C?",
    "How much of a business meal can I deduct?",
    "What counts as a business gift and what is the deduction limit?",
    "When is travel away from home considered business travel?",
    "What is the simplified method for the home office deduction?",
    "Can a daycare provider deduct part of their home?",
    "What is the Section 179 deduction limit?",
    "What property qualifies for bonus depreciation?",
    "How do I depreciate a computer used for business?",
    "What records do I need to keep for car expenses?",
    "Are estimated tax payments required if I am self-employed?",
    "What is the difference between a capital expense and a deductible expense?",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=250)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    chunks_path, vecs_path = chunk_files(args.size, args.overlap)
    chunks = load_chunks(chunks_path)
    vecs = np.load(vecs_path)
    exact = InMemoryStore()
    exact.add(chunks, vecs)
    query_vecs = embed(QUESTIONS)

    with PgVectorStore(table=pg_table(args.size, args.overlap)) as pg:
        rows = pg.count()
        pg.search(query_vecs[0], k=args.k)             # warm-up: wake a suspended database

        recalls, times = [], []
        for q, qv in zip(QUESTIONS, query_vecs):
            want = {r["chunk_id"] for r in exact.search(qv, k=args.k)}
            t = time.perf_counter()
            got = pg.search(qv, k=args.k)
            times.append((time.perf_counter() - t) * 1000)
            recall = len(want & {r["chunk_id"] for r in got}) / args.k
            recalls.append(recall)
            if recall < 1:
                print(f"  recall {recall:.1f}: {q}")

    times.sort()
    print(f"rows in table:        {rows} ({chunks_path} has {len(chunks)})")
    print(f"recall@{args.k} vs exact: {statistics.mean(recalls):.3f} over {len(QUESTIONS)} questions "
          f"(min {min(recalls):.1f})")
    print(f"search latency:       p50 {statistics.median(times):.0f} ms, "
          f"p95 {times[int(0.95 * (len(times) - 1))]:.0f} ms (includes the network round trip)")


if __name__ == "__main__":
    main()
