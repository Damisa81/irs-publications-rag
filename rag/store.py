"""Steps 3-4: vector stores. InMemoryStore and PgVectorStore go here, with the same interface:

    add(chunks, vecs)       insert chunks with their embeddings; existing chunk_ids are skipped
    search(query_vec, k)    the k most similar chunks, best first, as dicts:
                            {"chunk_id", "doc_id", "page", "text", "score"}
"""
import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from rag.embed import DIM


def search_exact(query_vec: np.ndarray, chunk_vecs: np.ndarray, k: int = 5) -> list[int]:
    """Brute force: score every chunk, return the indexes of the k best. O(n * d) per query.

    Assumes unit-length vectors (dot product = cosine). Kept as the reference that
    InMemoryStore and pgvector's approximate HNSW search are checked against.
    """
    scores = chunk_vecs @ query_vec
    return np.argsort(-scores)[:k].tolist()


class InMemoryStore:
    def __init__(self, dimension=DIM):
        self.dimension = dimension
        self.chunks = {}       # chunk_id -> chunk data and embedding
        self._entries = None   # chunks in matrix row order, rebuilt after add()
        self._matrix = None    # unit-length embeddings, one row per entry

    def add(self, chunks, vecs):
        """Insert chunks, ignoring IDs that already exist.

        Matches PostgreSQL's ON CONFLICT DO NOTHING: re-adding a chunk_id keeps the old
        text, even if the new chunk differs. Chunk ids don't encode the chunk size, so
        use a fresh store for each chunking experiment.
        """
        if len(chunks) != len(vecs):
            raise ValueError("chunks and vecs must have the same length")

        for chunk, vec in zip(chunks, vecs):
            chunk_id = chunk["chunk_id"]

            if chunk_id in self.chunks:
                continue

            embedding = check_vector(vec, self.dimension, "Embedding")

            self.chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "doc_id": chunk["doc_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "embedding": embedding.copy(),
            }
            self._matrix = None   # out of date; rebuilt on the next search

    def search(self, query_vec, k=5):
        """Return the top-k chunks by cosine similarity, best first."""
        if k <= 0 or not self.chunks:
            return []

        query = check_vector(query_vec, self.dimension, "Query embedding")
        query = query / np.linalg.norm(query)

        if self._matrix is None:
            self._build_matrix()

        # Rows are unit length, so a dot product is the cosine similarity.
        scores = self._matrix @ query
        top_indices = np.argsort(-scores, kind="stable")[:k]

        return [{**self._entries[i], "score": float(scores[i])} for i in top_indices]

    def _build_matrix(self):
        self._entries = [
            {key: value for key, value in entry.items() if key != "embedding"}
            for entry in self.chunks.values()
        ]
        embeddings = np.stack([entry["embedding"] for entry in self.chunks.values()])
        self._matrix = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


class PgVectorStore:
    """Chunks and embeddings in a Postgres table with pgvector, searched through an HNSW index.

    The connection string comes from the DATABASE_URL environment variable (hosted Neon or
    local Docker; only the string differs). Use a direct, non-pooled connection: building
    the index and the bulk load are long, session-level operations.

    One table per chunking configuration (e.g. "chunks_400_60"): chunk ids don't encode the
    chunk size, so reusing a table would silently keep the old chunks (ON CONFLICT DO NOTHING).
    """

    def __init__(self, table="chunks", dimension=DIM, dsn=None):
        self.dimension = dimension
        self._table = sql.Identifier(table)
        self._index = sql.Identifier(f"{table}_hnsw")
        self.conn = psycopg.connect(dsn or os.environ["DATABASE_URL"], autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)   # after the extension exists: lets psycopg send numpy arrays
        self._create_table()

    def _create_table(self):
        self.conn.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                chunk_id  text PRIMARY KEY,           -- stable id, e.g. p463-p34-c2
                doc_id    text NOT NULL,
                page      int  NOT NULL,
                text      text NOT NULL,
                embedding vector({}) NOT NULL         -- must match the model's dimension
            )""").format(self._table, sql.Literal(self.dimension)))

    def add(self, chunks, vecs):
        """Insert chunks, ignoring IDs that already exist (ON CONFLICT DO NOTHING).

        Re-running a build is safe: the row count doesn't change. The HNSW index is created
        after the first bulk load, which is much faster than building it row by row; later
        inserts update it incrementally.
        """
        if len(chunks) != len(vecs):
            raise ValueError("chunks and vecs must have the same length")

        rows = [
            (c["chunk_id"], c["doc_id"], c["page"], c["text"], check_vector(v, self.dimension, "Embedding"))
            for c, v in zip(chunks, vecs)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                sql.SQL("INSERT INTO {} VALUES (%s, %s, %s, %s, %s) ON CONFLICT (chunk_id) DO NOTHING")
                .format(self._table),
                rows,
            )
        # Approximate nearest-neighbour index; <=> (cosine distance) needs vector_cosine_ops.
        self.conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops)")
            .format(self._index, self._table)
        )

    def search(self, query_vec, k=5):
        """Return the top-k chunks by cosine similarity, best first (approximate, via HNSW)."""
        if k <= 0:
            return []
        query = check_vector(query_vec, self.dimension, "Query embedding")
        rows = self.conn.execute(
            sql.SQL("SELECT chunk_id, doc_id, page, text, 1 - (embedding <=> %s) AS score "
                    "FROM {} ORDER BY embedding <=> %s LIMIT %s").format(self._table),
            (query, query, k),
        ).fetchall()
        return [
            {"chunk_id": r[0], "doc_id": r[1], "page": r[2], "text": r[3], "score": float(r[4])}
            for r in rows
        ]

    def count(self):
        return self.conn.execute(sql.SQL("SELECT count(*) FROM {}").format(self._table)).fetchone()[0]

    def reset(self):
        """Empty the table by recreating it (the index is rebuilt on the next add)."""
        self.drop()
        self._create_table()

    def drop(self):
        """Delete the table and its index (after an experiment). The store can't be used afterwards."""
        self.conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self._table))

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def check_vector(vec, dimension, name):
    """A float32 copy of vec, or ValueError if it has the wrong shape, NaN/inf, or zero length."""
    vector = np.asarray(vec, dtype=np.float32)

    if vector.shape != (dimension,):
        raise ValueError(f"Expected {name.lower()} shape ({dimension},), got {vector.shape}")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")

    if np.linalg.norm(vector) == 0:
        raise ValueError(f"{name} cannot be a zero vector")

    return vector
