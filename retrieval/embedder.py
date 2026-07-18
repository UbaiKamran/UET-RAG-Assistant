"""
retrieval/embedder.py

Thin wrapper around sentence-transformers for encoding text into dense vectors.
The model is loaded once and cached for the lifetime of the process.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load (and cache) the sentence-transformer model."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded (dim=%d)", model.get_sentence_embedding_dimension())
    return model


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Encode a list of strings into a 2-D float32 numpy array.

    Parameters
    ----------
    texts         : list of strings to encode
    batch_size    : mini-batch size (tune down if RAM is tight)
    show_progress : display tqdm progress bar

    Returns
    -------
    np.ndarray of shape (len(texts), embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # normalise for cosine similarity
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> List[float]:
    """
    Encode a single query string and return a plain Python list of floats.
    (ChromaDB expects List[float] for query embeddings.)
    """
    vec = embed_texts([query])[0]
    return vec.tolist()
