"""
config.py — Central configuration for the UET Prospectus RAG pipeline.
Change values here; no other file needs editing for tuning.
"""

import os

# ---------------------------------------------------------------------------
# PDF Source
# ---------------------------------------------------------------------------
PDF_PATH = "./prospectus.pdf"

# ---------------------------------------------------------------------------
# Embedding (local, no API cost)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Alternatives (change here to swap):
#   "BAAI/bge-base-en-v1.5"
#   "intfloat/multilingual-e5-base"

# ---------------------------------------------------------------------------
# ChromaDB (persisted on disk)
# ---------------------------------------------------------------------------
CHROMA_DIR = "./data/chroma_db"
COLLECTION_NAME = "prospectus_chunks"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
MAX_CHUNK_TOKENS = 400          # max tokens per text chunk before splitting
MIN_CHUNK_CHARS  = 80           # discard chunks shorter than this

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K               = 5         # number of chunks to pass to the LLM
CANDIDATE_FETCH     = TOP_K * 3 # over-fetch before re-ranking
SIMILARITY_THRESHOLD = 0.70     # cosine-distance threshold (lower = more similar)
BOOST_DELTA         = 0.10      # score improvement for metadata/keyword match

# ---------------------------------------------------------------------------
# Generation (Groq API)
# ---------------------------------------------------------------------------
GROQ_MODEL   = "llama-3.3-70b-versatile"   # swap to any Groq model slug here
# e.g.  "openai/gpt-oss-120b"
#       "llama-3.1-8b-instant"
#       "qwen-qwen3-32b"
#       "mixtral-8x7b-32768"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.1

# ---------------------------------------------------------------------------
# Feedback log
# ---------------------------------------------------------------------------
FEEDBACK_LOG = "./logs/feedback.jsonl"
