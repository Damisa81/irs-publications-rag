"""Step 3: text -> vectors, with a small local embedding model."""
from functools import cache

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"      # 384 dimensions, runs locally
DIM = 384


@cache
def get_model():
    # Imported and loaded on first use, so importing this module stays fast (tests
    # and the API don't pay for loading PyTorch until they actually embed something).
    # The first run downloads the model (about 130 MB).
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    # normalize_embeddings=True makes every vector length 1, so a dot product IS cosine similarity
    return get_model().encode(texts, normalize_embeddings=True, batch_size=64)
