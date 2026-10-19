"""Ask one question from the command line. Run from the project folder:

    py -m scripts.ask "What is the standard mileage rate for business use of a car?"
    py -m scripts.ask "..." --store pg       # search Postgres instead of memory
    py -m scripts.ask "..." -k 8 --show-sources
"""
import argparse
import textwrap

from rag.pipeline import default_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--store", choices=["memory", "pg"], default="memory")
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid",
                        help="hybrid (default) won step 8; dense is the step 5-7 baseline")
    parser.add_argument("-k", type=int, default=5, help="chunks sent to the model")
    parser.add_argument("--show-sources", action="store_true", help="print the retrieved chunks too")
    args = parser.parse_args()

    result = default_pipeline(args.store, retriever=args.retriever, k=args.k).ask(args.question)
    a = result.answer

    print(f"\nanswerable: {a.answerable}")
    print(textwrap.fill(a.answer, 100))
    for c in result.citations:
        mark = "ok " if c.valid else "BAD"
        where = f"{c.doc_id} p.{c.page}" if c.doc_id else "not retrieved"
        print(f"  [{mark}] {c.chunk_id} ({where}): \"{c.quote}\"")
    for p in result.problems:
        print(f"  problem: {p}")

    t = result.timings
    print(f"\nretrieved: {', '.join(c['chunk_id'] for c in result.retrieved)}")
    seconds = ", ".join(f"{name.removesuffix('_s')} {value:.2f}" for name, value in t.items())
    print(f"tokens: {result.usage['input_tokens']} in, {result.usage['output_tokens']} out | seconds: {seconds}")

    if args.show_sources:
        for c in result.retrieved:
            print(f"\n--- {c['chunk_id']} (score {c['score']:.3f}) ---\n{textwrap.fill(c['text'], 100)}")


if __name__ == "__main__":
    main()
