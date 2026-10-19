# IRS Publications Q&A: a measured RAG system

A question-answering system for US federal tax that answers **only** from five 2025 IRS publications, cites the
publication, page and exact sentence for every claim, **checks those citations in code**, and says when the
publications don't cover a question. Everything runs locally: embeddings, keyword search and the LLM.

It's a personal learning project, built to understand retrieval-augmented generation end to end and, above all,
to **measure** it: a 50-question evaluation with 10 questions held out, and every design change compared against it.

```
POST /ask {"question": "What is the standard mileage rate for business use of a car in 2025?"}

"answer": "The standard mileage rate for the business use of a car in 2025 is 70 cents ($0.70) per mile."
"citations": [{"chunk_id": "p463-p44-c5", "doc_id": "p463", "page": 44,
               "quote": "For 2025, the standard mileage rate for the cost of operating your car for business use
                         is 70 cents ($0.70) per mile.",
               "cited_chunk_was_retrieved": true, "quote_found_in_chunk": true}, ...]
"all_citations_valid": true
```

## Results

40 development questions (32 answerable, 8 that the publications can't answer), plus 10 held out and scored once
at the end. Correctness and faithfulness are graded by an LLM judge against hand-written expected answers.

| Run | Recall@5 | Correct | Faithful | Citations valid | Abstained | p95 latency |
|---|---|---|---|---|---|---|
| Baseline: vector search only | 94% | 84% (27/32) | 97% | 85% | 8/8 | 9.1 s |
| Reranking (cross-encoder) | 100% | 81% (26/32) | 97% | 93% | 8/8 | 9.3 s |
| **Hybrid: vectors + BM25 (current)** | **100%** | **91% (29/32)** | 100% | 95% | 8/8 | 8.6 s |
| Hybrid + a prompt that answers more readily | 100% | 84% (27/32) | 94% | 85% | 7/8 | 10.4 s |
| Agent that chooses its own searches | 94% | 81% (26/32) | 100% | 98% | 8/8 | 12.8 s |
| **Held-out 10, final system, scored once** | 88% (7/8) | **75% (6/8)** | 100% | 100% | **1/2** | 6.8 s |

What the numbers say:

- **Keyword search rescued exact facts.** Vectors match topics; they missed the "$25" gift limit and the
  "15.3%" self-employment tax rate. Merging BM25 and vector results by reciprocal rank fusion fixed both, for
  about 5 ms per question.
- **Better retrieval metrics didn't mean better answers.** The cross-encoder reranker had the best retrieval
  numbers and the worst answers: it filled the top 5 with special cases (refund deadlines for bad debts) that
  pulled the model away from the general rule.
- **The agent didn't plan.** Given a search tool and up to three searches, the 14B model searched exactly once per
  question, and its rewritten queries dropped words that mattered ("how many years", "2025"). It cost about twice
  the tokens and latency.
- **The held-out drop is real.** Every choice was made on the dev questions, and the honest number is 6 of 8 on
  unseen ones. With samples this small, one question is 3-12 points, so counts are shown next to percentages.
- **The judge is lenient.** On a sample of 15 answers graded separately from the judge, it agreed on correctness
  15/15 but on faithfulness only 11/15, each time accepting claims its quotes didn't support. Treat faithfulness as
  an upper bound. (That sample was graded by Claude, not by me; see `eval/hand_grades.csv`.)

### What still fails

- **Contradicting its own quote:** asked whether a $3 promotional pen counts as a business gift, the model
  answered "yes, it may count" while quoting the rule that says it doesn't. The citation check passes, because the
  quote is real.
- **Answering from a near-miss source:** asked about collecting sales tax on Etsy sales to Texas (not covered), it
  answered "yes" from a sentence about reporting sales tax that was already collected.

The next fixes would be a check that an answer's conclusion agrees with what it quotes, and judging answers to
unanswerable questions as well as whether they were refused.

## How it works

```
Build once:   5 IRS PDFs (409 pages) -> clean -> 1,865 chunks of 250 words (40 overlap)
                                              -> bge-small embeddings (384 numbers each)
                                              -> in memory, or Postgres + pgvector (HNSW index)

Every question:  question -> hybrid retrieval: vector search + BM25 keyword search, merged by rank -> best 5 chunks
                          -> qwen3:14b (Ollama) answers from those 5 only, as JSON:
                             {answerable, answer, citations: [{chunk_id, quote}]}
                          -> code checks each citation: was the chunk retrieved, is the quote really in it?
                          -> answer + citations with publication and page + check results
```

The documents: Publications 17 (individual income tax), 334 (small business), 463 (travel, gift and car
expenses), 587 (business use of home) and 946 (depreciation), for tax year 2025.

## Design decisions

| Choice | Why | When I'd choose differently |
|---|---|---|
| Hybrid retrieval | Measured: fixed exact-fact misses at almost no cost | No exact terms in the domain |
| Citations checked in code | The model can't be the only judge of its own grounding | Never skip it |
| Structured JSON output | Abstaining is a field, not a phrase to detect; malformed output is an error (502), not a silent bad answer | Free-text chat where streaming matters more |
| pgvector, not a dedicated vector DB | One system for metadata, filters and vectors; recall vs. exact search measured at 1.000 | Hundreds of millions of vectors |
| Local LLM and embeddings | Free, private, no API key; the generator is one swappable function | When answer quality matters more than cost |
| No framework (LangChain etc.) | Each stage is small, visible and unit-tested | Complex multi-agent graphs |
| A fixed pipeline, not an agent | Measured: worse and twice as expensive here | A stronger planner, or tasks needing chained lookups |

## Running it

Requirements: Python 3.13, [Ollama](https://ollama.com), about 10 GB of disk for the model. Tested on Windows 11.

```powershell
py -m venv C:\dev\rag-venv            # outside OneDrive/Dropbox: it holds ~40,000 files
C:\dev\rag-venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen3:14b

# The documents (public IRS downloads)
mkdir data\raw
foreach ($p in "p17","p334","p463","p587","p946") {
    Invoke-WebRequest "https://www.irs.gov/pub/irs-pdf/$p.pdf" -OutFile "data\raw\$p.pdf" }

py -m scripts.build_index                                 # parse, chunk, embed (~2 minutes on CPU)
py -m scripts.ask "How much can I deduct for business gifts?"
py -m uvicorn api.app:app --reload                        # then http://127.0.0.1:8000/docs
py -m pytest tests                                        # 71 tests, ~5 s, no model calls
```

Run every command from the project folder with `py -m`, so `import rag` resolves.

Optional, Postgres + pgvector (any Postgres with the `vector` extension; I used a free [Neon](https://neon.tech)
database):

```powershell
setx DATABASE_URL "postgresql://user:password@host/db?sslmode=require"   # then open a new terminal
py -m scripts.build_index --pg
py -m scripts.check_pgvector          # recall of the HNSW index vs. exact search
$env:RAG_STORE = "pg"                 # make the API search Postgres
```

Evaluation:

```powershell
py -m eval.run_eval --name baseline --retrieval-only        # retrieval metrics only, seconds
py -m eval.run_eval --name hybrid --retriever hybrid        # answers + LLM judge, ~4 minutes on an RTX 4090
py -m eval.run_eval --name agent --retriever hybrid --agent # the search agent
```

Each run writes `eval/results/<name>-<split>.json` with every question's retrieved chunks, answer, citation
checks and verdict. The results behind the table above are in that folder.

## Layout

```
rag/        load.py, chunk.py, embed.py, store.py (in-memory + pgvector), retrieve.py (dense, BM25,
            hybrid, rerank), generate.py (Ollama + JSON schema), pipeline.py (retrieve -> answer -> check),
            agent.py (tool-calling search agent)
api/        app.py: FastAPI POST /ask, GET /health
scripts/    build_index.py, check_pgvector.py, ask.py
eval/       make_questions.py -> questions.jsonl (expected pages found from evidence sentences),
            metrics.py, judge.py, run_eval.py, hand_grade.py, results/
tests/      71 tests with fake stores, embedders and models: no network or GPU needed
```

## Limits

Not tax advice. It knows only these five publications for tax year 2025, nothing about the person asking, and
does no calculations. Tables come out of the PDFs as rows of numbers, so table lookups are unreliable. The
evaluation is small (50 questions), so differences of one or two questions are within noise.

## How this was built

I designed the system, ran every experiment and made the decisions. Much of the code was written with Claude Code
(Anthropic's AI coding assistant), working from my step-by-step plan.
