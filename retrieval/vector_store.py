"""
retrieval/vector_store.py

ChromaDB interface — initialisation, upsertion, and raw query.

The collection is persisted to disk at CHROMA_DIR so it only needs to be
built once (via build_index.py).  The Chainlit app opens the same persisted
collection read-only at chat time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb

# Settings import path changed between chromadb versions — handle both
try:
    from chromadb.config import Settings
except ImportError:
    try:
        from chromadb import Settings
    except ImportError:
        Settings = None  # very new chromadb — Settings not needed

from config import CHROMA_DIR, COLLECTION_NAME
from ingestion.chunker import Chunk
from retrieval.embedder import embed_texts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client / collection helpers
# ---------------------------------------------------------------------------

def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client pointing to CHROMA_DIR."""
    kwargs = {"path": CHROMA_DIR}
    if Settings is not None:
        try:
            kwargs["settings"] = Settings(anonymized_telemetry=False)
        except Exception:
            pass  # newer chromadb ignores unknown settings fields
    return chromadb.PersistentClient(**kwargs)



def get_or_create_collection(client: Optional[chromadb.PersistentClient] = None):
    """Return (or create) the main prospectus collection."""
    if client is None:
        client = get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine distance
    )
    logger.info(
        "Collection '%s' ready — existing items: %d",
        COLLECTION_NAME,
        collection.count(),
    )
    return collection


# ---------------------------------------------------------------------------
# Upsertion
# ---------------------------------------------------------------------------

def upsert_chunks(chunks: List[Chunk], batch_size: int = 128) -> None:
    """
    Embed and upsert all Chunk objects into ChromaDB.

    Uses batched upsertion to stay within ChromaDB memory limits.
    Idempotent: re-running overwrites existing entries with the same chunk_id.
    """
    client     = get_client()
    collection = get_or_create_collection(client)

    logger.info("Upserting %d chunks (batch_size=%d)…", len(chunks), batch_size)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]

        # Compute embeddings for this batch
        vecs = embed_texts(texts, show_progress=False)

        ids        = [c.chunk_id for c in batch]
        embeddings = [v.tolist() for v in vecs]
        metadatas  = [
            {
                "section_title": c.section_title,
                "page_num":      c.page_num,
                "content_type":  c.content_type,
                "department":    c.department,
                "has_table":     str(c.has_table),   # Chroma metadata must be str/int/float
                "ocr":           str(c.ocr),
            }
            for c in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(
            "  Upserted batch %d-%d", start + 1, min(start + batch_size, len(chunks))
        )

    logger.info("Upsert complete. Total items in collection: %d", collection.count())


def delete_by_page_nums(page_nums: List[int]) -> int:
    """
    Delete all chunks whose page_num metadata is in `page_nums`.
    Returns the number of deleted ids.
    """
    if not page_nums:
        return 0
    client     = get_client()
    collection = get_or_create_collection(client)
    res = collection.get(
        where={"page_num": {"$in": list(page_nums)}},
        include=[],
    )
    ids = res.get("ids") or []
    if ids:
        collection.delete(ids=ids)
        logger.info("Deleted %d chunks for pages %s", len(ids), page_nums)
    return len(ids)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_collection(
    query_embedding: List[float],
    n_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a vector similarity query against the persisted collection.

    Parameters
    ----------
    query_embedding : pre-computed query vector (list of floats)
    n_results       : how many results to return
    where           : optional ChromaDB metadata filter dict

    Returns
    -------
    Raw ChromaDB query result dict with keys:
      ids, documents, metadatas, distances
    """
    client     = get_client()
    collection = get_or_create_collection(client)

    kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results":        min(n_results, collection.count() or 1),
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def collection_count() -> int:
    """Return the total number of chunks in the persisted collection."""
    client     = get_client()
    collection = get_or_create_collection(client)
    return collection.count()
