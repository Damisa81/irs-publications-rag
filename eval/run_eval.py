"""Step 7: run the evaluation and write eval/results/<name>.json. Run from the project folder:

    py -m eval.run_eval --name baseline --retrieval-only     # free: recall@k and MRR, seconds
    py -m eval.run_eval --name baseline                      # + answers, citation checks, judge
    py -m eval.run_eval --name final --split test            # the 10 held-out questions, at the end

Change one thing per run and give it a name that says what changed (e.g. "k8", "hybrid").
The dev split (40 questions) is for tuning; only score the test split once, on the final system.
"""
import argparse
import functools
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from eval import metrics
from rag.embed import MODEL_NAME as EMBED_MODEL
from rag.generate import MODEL as LLM
from rag.generate import PROMPTS, generate
from rag.chunk import load_chunks
from rag.pipeline import Pipeline, check_citations, chunk_files, load_store

RESULTS = Path("eval/results")


def load_questions(path: str, split: str) -> list[dict]:
    qs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return qs if split == "all" else [q for q in qs if q["split"] == split]


def score_retrieval(q: dict, retrieved: list[dict]) -> dict:
    if not q["answerable"]:
        return {}
    rank = metrics.first_relevant_rank(retrieved, q["pages"])
    return {
        "first_relevant_rank": rank,
        "page_hit": rank is not None,
        "reciprocal_rank": 1 / rank if rank else 0.0,
        "all_groups_hit": metrics.all_groups_hit(retrieved, q["page_groups"]),
        "evidence_hit": metrics.evidence_hit(retrieved, q["evidence"]),
    }


def run(args) -> dict:
    if not (args.retrieval_only or args.no_judge):
        from eval.judge import judge      # only needed when answers are graded

    questions = load_questions(args.questions, args.split)
    if args.limit:
        questions = questions[:args.limit]
    chunks_path, vecs_path = chunk_files(args.size, args.overlap)
    store = load_store(args.store, table=f"chunks_{args.size}_{args.overlap}",
                       chunks_path=chunks_path, vecs_path=vecs_path)
    retriever = make_retriever(args, store, chunks_path)
    generate_fn = functools.partial(generate, system=PROMPTS[args.prompt])
    pipeline = Pipeline(store, k=args.k, retriever=retriever, generate_fn=generate_fn)
    pipeline.retrieve("warm-up")      # load the models (and wake a hosted database) before timing
    agent = None
    if args.agent:
        from rag.agent import SearchAgent
        from rag.retrieve import DenseRetriever
        if args.retrieval_only:
            raise SystemExit("--agent needs the model to choose its searches; drop --retrieval-only")
        agent = SearchAgent(retriever or DenseRetriever(store), max_searches=args.max_searches,
                            k_per_search=args.k_per_search, generate_fn=generate_fn)

    rows = []
    for i, q in enumerate(questions, 1):
        if agent:
            res = agent.ask(q["question"])
            retrieved, timings = res.retrieved, res.timings
            answer, usage, checks = res.answer, res.usage, res.citations
        else:
            retrieved, timings = pipeline.retrieve(q["question"])
        row = {"id": q["id"], "question": q["question"], "answerable_expected": q["answerable"],
               "expected": q["expected"], "pages": q["pages"],
               "retrieved": [{"chunk_id": c["chunk_id"], "page": metrics.page_of(c), "score": round(c["score"], 4)}
                             for c in retrieved],
               **score_retrieval(q, retrieved), "timings": timings}
        if agent:
            row.update({"searches": res.searches, "stop_reason": res.stop_reason})

        if not args.retrieval_only:
            if not agent:
                t0 = time.perf_counter()
                answer, usage = pipeline.generate_fn(q["question"], retrieved)
                timings["generate_s"] = time.perf_counter() - t0
                timings["total_s"] = sum(timings.values())
                checks = check_citations(answer, retrieved)
            row.update({
                "answerable_predicted": answer.answerable,
                "answer": answer.answer,
                "citations": [{"chunk_id": c.chunk_id, "quote": c.quote, "valid": c.valid,
                               "cited_chunk_was_retrieved": c.cited_chunk_was_retrieved,
                               "quote_found_in_chunk": c.quote_found_in_chunk} for c in checks],
                "usage": usage,
            })
            # Judge only real answers to answerable questions; a refusal there is simply incorrect.
            if q["answerable"] and answer.answerable and not args.no_judge:
                v = judge(q["question"], q["expected"], answer)
                row["judge"] = {"correct": v.correct, "faithful": v.faithful, "reason": v.reason}
            elif q["answerable"]:
                row["judge"] = {"correct": False, "faithful": None, "reason": "refused an answerable question"}

        rows.append(row)
        print(f"  {i:2}/{len(questions)} {q['id']} {status(row)}", flush=True)

    return {"config": config(args, questions), "summary": summarize(rows, args.retrieval_only), "questions": rows}


def make_retriever(args, store, chunks_path):
    """None means the plain embed + store path used in steps 5-7 (the baseline)."""
    if args.retriever == "dense" and not args.prefix and not args.rerank:
        return None
    from rag import retrieve
    return retrieve.make_retriever(args.retriever, store, load_chunks(chunks_path), prefix=args.prefix,
                                   rerank=args.rerank, candidates=args.candidates)


def describe(c: dict) -> str:
    parts = [c["retriever"] + (" + prefix" if c["prefix"] else "")]
    if c["rerank"]:
        parts.append(f"rerank top {c['rerank']}")
    if c.get("prompt", "v1") != "v1" and c.get("llm"):
        parts.append(f"prompt {c['prompt']}")
    if c.get("agent"):
        parts.append(f"agent: up to {c['agent']['max_searches']} searches x {c['agent']['k_per_search']} chunks")
    return ", ".join(parts)


def status(row: dict) -> str:
    parts = []
    if "page_hit" in row:
        parts.append(f"rank {row['first_relevant_rank'] or '-'}")
    if "answerable_predicted" in row:
        parts.append("answered" if row["answerable_predicted"] else "refused")
    if row.get("judge", {}).get("correct") is not None and "answerable_predicted" in row and row["answerable_predicted"]:
        parts.append("correct" if row["judge"]["correct"] else "WRONG")
    return " | ".join(parts)


def summarize(rows: list[dict], retrieval_only: bool) -> dict:
    ans = [r for r in rows if r["answerable_expected"]]
    unans = [r for r in rows if not r["answerable_expected"]]
    multi = [r for r in ans if len(r["pages"]) and r.get("all_groups_hit") is not None]
    s = {
        "questions": len(rows), "answerable": len(ans), "unanswerable": len(unans),
        "recall_at_k_page": metrics.mean(r["page_hit"] for r in ans),
        "mrr": metrics.mean(r["reciprocal_rank"] for r in ans),
        "recall_at_k_all_parts": metrics.mean(r["all_groups_hit"] for r in multi),
        "recall_at_k_evidence": metrics.mean(r["evidence_hit"] for r in ans),
        "retrieval_misses": [r["id"] for r in ans if not r["page_hit"]],
    }
    if retrieval_only:
        return s

    answered = [r for r in ans if r["answerable_predicted"]]
    all_cits = [c for r in rows for c in r["citations"]]
    judged = [r for r in answered if r.get("judge", {}).get("faithful") is not None]
    has_judge = len(judged) == len(answered)                                  # False with --no-judge
    totals = [r["timings"]["total_s"] for r in rows]
    s.update({
        # Refusing an answerable question counts as wrong; faithfulness is over answers given.
        "correct": metrics.mean(r["judge"]["correct"] for r in ans) if has_judge else None,
        "faithful": metrics.mean(r["judge"]["faithful"] for r in judged) if has_judge else None,
        "citation_validity": metrics.mean(c["valid"] for c in all_cits),
        "answers_with_all_citations_valid": metrics.mean(
            bool(r["citations"]) and all(c["valid"] for c in r["citations"]) for r in answered),
        "abstained_correctly": sum(not r["answerable_predicted"] for r in unans),
        "wrongly_refused": sum(not r["answerable_predicted"] for r in ans),
        "latency_p50_s": metrics.percentile(totals, 50),
        "latency_p95_s": metrics.percentile(totals, 95),
        "avg_input_tokens": metrics.mean(r["usage"].get("input_tokens") for r in rows),
        "avg_output_tokens": metrics.mean(r["usage"].get("output_tokens") for r in rows),
        "wrong": [r["id"] for r in ans if has_judge and not r["judge"]["correct"]],
        "unfaithful": [r["id"] for r in judged if not r["judge"]["faithful"]],
        "answered_unanswerable": [r["id"] for r in unans if r["answerable_predicted"]],
    })
    if rows and "searches" in rows[0]:
        stops = [r["stop_reason"] for r in rows]
        s.update({
            "avg_searches": metrics.mean(len(r["searches"]) for r in rows),
            "searches_histogram": {n: sum(len(r["searches"]) == n for r in rows)
                                   for n in sorted({len(r["searches"]) for r in rows})},
            "stop_reasons": {reason: stops.count(reason) for reason in sorted(set(stops))},
            "avg_chunks_given": metrics.mean(len(r["retrieved"]) for r in rows),
        })
    return s


def config(args, questions) -> dict:
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        git = ""
    return {"name": args.name, "time": datetime.now().isoformat(timespec="seconds"), "split": args.split,
            "questions": len(questions), "k": args.k, "store": args.store,
            "chunking": f"{args.size}/{args.overlap}", "embed_model": EMBED_MODEL,
            "retriever": args.retriever, "prefix": args.prefix, "rerank": args.rerank, "prompt": args.prompt,
            "agent": {"max_searches": args.max_searches, "k_per_search": args.k_per_search} if args.agent else None,
            "candidates": args.candidates if args.retriever == "hybrid" else None,
            "llm": None if args.retrieval_only else LLM, "judge": not (args.retrieval_only or args.no_judge),
            "git": git or None}


def fmt(v, pct=True):
    if v is None:
        return "-"
    return f"{v:.0%}" if pct else f"{v:.2f}"


def print_summary(result: dict):
    s, c = result["summary"], result["config"]
    print(f"\n=== {c['name']}  ({c['split']}, {s['questions']} questions, k={c['k']}, {c['store']}, "
          f"{c['chunking']}, {describe(c)})")
    print(f"recall@{c['k']} by page   {fmt(s['recall_at_k_page'])}   MRR {fmt(s['mrr'], False)}")
    print(f"recall@{c['k']} evidence  {fmt(s['recall_at_k_evidence'])}   (a retrieved chunk contains the answer)")
    print(f"all parts retrieved  {fmt(s['recall_at_k_all_parts'])}")
    print(f"retrieval misses     {', '.join(s['retrieval_misses']) or 'none'}")
    if "correct" in s:
        print(f"correct {fmt(s['correct'])} | faithful {fmt(s['faithful'])} | "
              f"citations valid {fmt(s['citation_validity'])} | "
              f"abstained {s['abstained_correctly']}/{s['unanswerable']} | wrongly refused {s['wrongly_refused']}")
        print(f"latency p50 {s['latency_p50_s']:.1f} s, p95 {s['latency_p95_s']:.1f} s | "
              f"tokens {s['avg_input_tokens']:.0f} in, {s['avg_output_tokens']:.0f} out (avg)")
        print(f"wrong: {', '.join(s['wrong']) or 'none'} | unfaithful: {', '.join(s['unfaithful']) or 'none'} | "
              f"answered unanswerable: {', '.join(s['answered_unanswerable']) or 'none'}")
        if "avg_searches" in s:
            print(f"agent: {s['avg_searches']:.1f} searches per question {s['searches_histogram']} | "
                  f"stopped: {s['stop_reasons']} | {s['avg_chunks_given']:.1f} chunks given to the answer "
                  f"(recall above is over all of them, not just 5)")
        print("\nresults-log row:")
        print(f"| {c['name']} | {fmt(s['recall_at_k_page'])} | {fmt(s['mrr'], False)} | {fmt(s['correct'])} | "
              f"{fmt(s['faithful'])} | {s['abstained_correctly']}/{s['unanswerable']} | "
              f"{s['latency_p95_s']:.1f} | {s['avg_input_tokens']:.0f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="what this run changed, e.g. baseline, k8, hybrid")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--store", choices=["memory", "pg"], default="memory")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--size", type=int, default=250, help="chunk size (selects the pg table)")
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="dense")
    parser.add_argument("--prefix", action="store_true", help="add bge's query instruction to questions")
    parser.add_argument("--rerank", type=int, default=0, metavar="N",
                        help="retrieve N candidates, rerank with a cross-encoder, keep k (0 = off)")
    parser.add_argument("--candidates", type=int, default=20, help="per-retriever list length for hybrid fusion")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v1", help="system prompt version")
    parser.add_argument("--agent", action="store_true", help="step 9: the model chooses its own searches")
    parser.add_argument("--max-searches", type=int, default=3, help="with --agent: the cap on searches")
    parser.add_argument("--k-per-search", type=int, default=4, help="with --agent: chunks per search")
    parser.add_argument("--retrieval-only", action="store_true", help="no LLM: recall and MRR only")
    parser.add_argument("--no-judge", action="store_true", help="answers and citation checks, no judge")
    parser.add_argument("--limit", type=int, help="only the first N questions (for a quick check)")
    args = parser.parse_args()

    if args.split == "test":
        print("Scoring the held-out test split: do this once, on the final system.")
    result = run(args)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.name}{'-retrieval' if args.retrieval_only else ''}-{args.split}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_summary(result)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
