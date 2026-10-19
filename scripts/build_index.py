r"""Build the index: PDFs -> cleaned pages -> chunks -> vectors, each saved under data\.

The two heavy stages run once and are reused afterwards:
  - PDF parsing  -> data\pages.jsonl      (redone only with --rebuild)
  - embedding    -> data\chunk_vecs.npy   (redone when the chunks or the model change)
Chunking is fast, so it reruns every time; changing --size or --overlap never
re-reads the PDFs, and only re-embeds when the chunks actually differ.

Run from the project folder:
    py -m scripts.build_index                         # defaults: 250 words, 40 overlap
    py -m scripts.build_index --size 400 --overlap 60
    py -m scripts.build_index --rebuild               # re-read the PDFs too
    py -m scripts.build_index --pg                    # also load into Postgres (DATABASE_URL)

With --pg, each chunking configuration gets its own table (chunks_250_40, chunks_400_60, ...),
because chunk ids don't encode the chunk size and ON CONFLICT DO NOTHING would keep old chunks.
If you change the chunks some other way (e.g. cleaning) at the same size, add --drop-pg.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rag.chunk import chunk_pages, load_chunks, save_chunks
from rag.embed import MODEL_NAME, embed
from rag.load import clean, load_pages
from rag.pipeline import chunk_files
from rag.store import PgVectorStore

DATA = Path("data")
RAW = DATA / "raw"
PAGES = DATA / "pages.jsonl"
CHUNKS, VECS = map(Path, chunk_files())   # the default 250/40 configuration


def build_pages(rebuild: bool) -> list[dict]:
    if PAGES.exists() and not rebuild:
        pages = load_chunks(PAGES)       # same JSONL format as chunks
        print(f"pages:  {len(pages)} from {PAGES} (cached)")
        return pages
    t = time.perf_counter()
    pages = []
    for pdf in sorted(RAW.glob("*.pdf")):
        for p in load_pages(pdf):
            p["text"] = clean(p["text"])
            pages.append(p)
    save_chunks(pages, PAGES)
    print(f"pages:  {len(pages)} from {RAW} in {time.perf_counter() - t:.0f} s -> {PAGES}")
    return pages


def fingerprint(chunks: list[dict]) -> str:
    """Changes whenever any chunk's id or text changes."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_id"].encode() + b"\0" + c["text"].encode() + b"\0")
    return h.hexdigest()


def build_vectors(chunks: list[dict], rebuild: bool, vecs_path: Path = VECS) -> np.ndarray:
    meta_path = vecs_path.with_suffix(".json")       # which chunks and model the vectors belong to
    meta = {"model": MODEL_NAME, "chunks": fingerprint(chunks)}
    if vecs_path.exists() and meta_path.exists() and not rebuild:
        if json.loads(meta_path.read_text(encoding="utf-8")) == meta:
            vecs = np.load(vecs_path)
            print(f"vectors: {vecs.shape} from {vecs_path} (cached)")
            return vecs
    t = time.perf_counter()
    vecs = embed([c["text"] for c in chunks])
    np.save(vecs_path, vecs)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"vectors: {vecs.shape} in {time.perf_counter() - t:.0f} s -> {vecs_path}")
    return vecs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=250, help="max words per chunk")
    parser.add_argument("--overlap", type=int, default=40, help="words shared by consecutive chunks")
    parser.add_argument("--rebuild", action="store_true", help="ignore cached pages and vectors")
    parser.add_argument("--pg", action="store_true", help="also load into Postgres (needs DATABASE_URL)")
    parser.add_argument("--drop-pg", action="store_true", help="with --pg: empty this configuration's table first")
    args = parser.parse_args()

    pages = build_pages(args.rebuild)
    chunks = chunk_pages(pages, size=args.size, overlap=args.overlap)
    chunks_path, vecs_path = map(Path, chunk_files(args.size, args.overlap))
    save_chunks(chunks, chunks_path)
    print(f"chunks: {len(chunks)} at {args.size}/{args.overlap} -> {chunks_path}")
    vecs = build_vectors(chunks, args.rebuild, vecs_path)

    if args.pg:
        table = pg_table(args.size, args.overlap)
        t = time.perf_counter()
        with PgVectorStore(table=table) as store:
            if args.drop_pg:
                store.reset()
            before = store.count()
            store.add(chunks, vecs)
            after = store.count()
        print(f"postgres: table {table}: {before} -> {after} rows in {time.perf_counter() - t:.0f} s")


def pg_table(size: int, overlap: int) -> str:
    """One table per chunking configuration."""
    return f"chunks_{size}_{overlap}"


if __name__ == "__main__":
    main()
