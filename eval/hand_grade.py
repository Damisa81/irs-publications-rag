"""Step 7: grade 15 answers yourself, then measure how often the LLM judge agrees with you.

    py -m eval.hand_grade make  eval/results/baseline-dev.json     # writes eval/hand_grades.csv
    (open the CSV, fill my_correct and my_faithful with y or n for each row, save)
    py -m eval.hand_grade score eval/results/baseline-dev.json     # agreement with the judge

The CSV leaves out the judge's verdicts so they can't sway you.
"""
import csv
import json
import random
import sys
from pathlib import Path

CSV = Path("eval/hand_grades.csv")
FIELDS = ["id", "question", "expected", "answer", "cited_quotes", "my_correct", "my_faithful"]


def judged_rows(results_path: str) -> list[dict]:
    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))["questions"]
    return [r for r in rows if r.get("judge", {}).get("faithful") is not None]


def make(results_path: str, n: int = 15):
    if CSV.exists():
        sys.exit(f"{CSV} already exists; delete it first if you want a new sample.")
    rows = judged_rows(results_path)
    sample = sorted(random.Random(0).sample(rows, min(n, len(rows))), key=lambda r: r["id"])
    with CSV.open("w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig so Excel reads it
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sample:
            w.writerow({"id": r["id"], "question": r["question"], "expected": r["expected"],
                        "answer": r["answer"],
                        "cited_quotes": " | ".join(f"[{c['chunk_id']}] {c['quote']}" for c in r["citations"]),
                        "my_correct": "", "my_faithful": ""})
    print(f"wrote {len(sample)} answers to {CSV}: fill my_correct and my_faithful with y or n")


def score(results_path: str):
    judge = {r["id"]: r["judge"] for r in judged_rows(results_path)}
    with CSV.open(encoding="utf-8-sig") as f:
        mine = [r for r in csv.DictReader(f) if r["my_correct"].strip() and r["my_faithful"].strip()]
    if not mine:
        sys.exit(f"No graded rows in {CSV} yet.")
    for field in ("correct", "faithful"):
        pairs = [(r[f"my_{field}"].strip().lower().startswith("y"), judge[r["id"]][field]) for r in mine]
        agree = sum(a == b for a, b in pairs)
        print(f"{field:9} judge agrees with you on {agree}/{len(pairs)} ({agree / len(pairs):.0%})")
        for r, (me, j) in zip(mine, pairs):
            if me != j:
                print(f"    {r['id']}: you {'y' if me else 'n'}, judge {'y' if j else 'n'} "
                      f"({judge[r['id']]['reason']})")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("make", "score"):
        sys.exit(__doc__)
    {"make": make, "score": score}[sys.argv[1]](sys.argv[2])
