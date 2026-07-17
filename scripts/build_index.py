"""
scripts/build_index.py — One-time ingestion script

Run this ONCE (or after updating the PDF) to:
  1. Parse prospectus.pdf (text + tables)
  2. Clean repeating headers/footers
  3. Chunk by logical structure
  4. Embed chunks locally (sentence-transformers)
  5. Store in persisted ChromaDB

Usage (from project root):
    python scripts/build_index.py

The Chainlit app reads the same ChromaDB directory at chat time — you do NOT
need to re-run this script on every launch.
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
from pathlib import Path

# Project root = parent of /scripts (so relative paths in config.py resolve)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment & validate API key early
# ---------------------------------------------------------------------------
load_dotenv(_ROOT / ".env")

if not os.getenv("GROQ_API_KEY"):
    sys.exit(
        "ERROR: GROQ_API_KEY is not set in your .env file.\n"
        "Add it as:  GROQ_API_KEY=gsk_...\n"
        "The build script does not need it, but the chat app will.\n"
        "Continuing anyway — set the key before launching Chainlit."
    )

from config import PDF_PATH, CHROMA_DIR
from ingestion.pdf_parser import parse_pdf
from ingestion.cleaner import build_cleaner, clean_text
from ingestion.chunker import chunk_pages
from retrieval.vector_store import upsert_chunks, collection_count

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_sample_chunk(chunk, index: int) -> None:
    print(f"\n{'='*70}")
    print(f"  SAMPLE CHUNK #{index}")
    print(f"{'='*70}")
    print(f"  chunk_id     : {chunk.chunk_id}")
    print(f"  section_title: {chunk.section_title}")
    print(f"  page_num     : {chunk.page_num}")
    print(f"  content_type : {chunk.content_type}")
    print(f"  department   : {chunk.department or '(none)'}")
    print(f"  has_table    : {chunk.has_table}")
    print(f"  ocr          : {chunk.ocr}")
    print(f"  text ({len(chunk.text)} chars):")
    preview = textwrap.indent(
        textwrap.fill(chunk.text[:600], width=66) + ("…" if len(chunk.text) > 600 else ""),
        prefix="    ",
    )
    print(preview)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        sys.exit(f"ERROR: PDF not found at '{pdf_path.resolve()}'\n"
                 "Place prospectus.pdf in the project root and re-run.")

    logger.info("=" * 60)
    logger.info("UET Prospectus RAG — Index Builder")
    logger.info("=" * 60)
    logger.info("PDF       : %s", pdf_path.resolve())
    logger.info("ChromaDB  : %s", Path(CHROMA_DIR).resolve())

    # Ensure data directory exists
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Parse PDF --------------------------------------------------
    logger.info("\n[1/4] Parsing PDF…")
    pages = parse_pdf(pdf_path)
    logger.info("  → %d pages parsed", len(pages))

    # ---- Step 2: Clean headers/footers / page numbers -----------------------
    logger.info("\n[2/4] Cleaning headers, footers, page numbers…")
    raw_texts = [p.raw_text for p in pages]
    cleaner   = build_cleaner(raw_texts)
    logger.info("  → Detected %d noise lines to strip", len(cleaner.noise_lines))

    for page in pages:
        page.raw_text = clean_text(page.raw_text, cleaner)

    # ---- Step 3: Chunk ------------------------------------------------------
    logger.info("\n[3/4] Chunking by logical structure…")
    chunks = chunk_pages(pages)
    logger.info("  → %d total chunks created", len(chunks))

    # Breakdown by content type
    from collections import Counter
    ctype_counts = Counter(c.content_type for c in chunks)
    table_count  = sum(1 for c in chunks if c.has_table)
    ocr_count    = sum(1 for c in chunks if c.ocr)
    print("\n  Content-type breakdown:")
    for ctype, count in ctype_counts.most_common():
        print(f"    {ctype:<22} : {count:>4} chunks")
    print(f"    {'(table chunks)':<22} : {table_count:>4} chunks")
    print(f"    {'(ocr chunks)':<22} : {ocr_count:>4} chunks")

    # ---- Step 4: Embed + store in ChromaDB ----------------------------------
    logger.info("\n[4/4] Embedding and upserting to ChromaDB…")
    logger.info("  (This may take a few minutes on first run — model download + encoding)")
    upsert_chunks(chunks)
    final_count = collection_count()
    logger.info("  → ChromaDB now contains %d items", final_count)

    # ---- Print 3 sample chunks (including ≥1 table) -------------------------
    print("\n\n" + "="*70)
    print("  SAMPLE CHUNKS (for verification)")
    print("="*70)

    # Sample 1: first text chunk (policy_text or admission_criteria)
    text_chunks = [c for c in chunks if not c.has_table]
    if text_chunks:
        _print_sample_chunk(text_chunks[0], 1)

    # Sample 2: first table chunk
    table_chunks = [c for c in chunks if c.has_table]
    if table_chunks:
        _print_sample_chunk(table_chunks[0], 2)

    # Sample 3: a course_list or faculty_list chunk
    typed_chunks = [c for c in chunks if c.content_type in ("faculty_list", "course_list")]
    if typed_chunks:
        _print_sample_chunk(typed_chunks[0], 3)
    elif len(chunks) >= 3:
        _print_sample_chunk(chunks[2], 3)

    print("="*70)
    logger.info("\n✅ Index built successfully!")
    logger.info("   Run the chat app with:  chainlit run app/main.py --port 8001")
    logger.info("="*60)


if __name__ == "__main__":
    main()
