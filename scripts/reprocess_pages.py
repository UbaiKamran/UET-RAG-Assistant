"""
scripts/reprocess_pages.py — Re-process only specific prospectus pages into ChromaDB.

Used to fix parsing/chunking issues without rebuilding the full index.

Usage (from project root):
    python scripts/reprocess_pages.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pdfplumber

# Project root = parent of /scripts (so relative paths in config.py resolve)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from config import PDF_PATH
from ingestion.chunker import chunk_pages
from ingestion.pdf_parser import PageContent, TableData, _rows_to_markdown
from retrieval.vector_store import delete_by_page_nums, upsert_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reprocess_pages")

# Known repeating header/footer to strip on targeted reprocess
_NOISE = {
    "Undergraduate Prospectus Fall 2026 www.uet.edu.pk",
}


def _parse_pages(pdf_path: Path, page_nums: list[int]) -> list[PageContent]:
    """Parse only the given 1-based page numbers."""
    results: list[PageContent] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pn in page_nums:
            if pn < 1 or pn > len(pdf.pages):
                raise ValueError(f"Page {pn} out of range (1-{len(pdf.pages)})")
            page = pdf.pages[pn - 1]
            raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            # Strip known noise + standalone page-number line
            lines = []
            for line in raw_text.splitlines():
                s = line.strip()
                if s in _NOISE:
                    continue
                if s.isdigit() and s == str(pn):
                    continue
                lines.append(line)
            raw_text = "\n".join(lines)

            tables: list[TableData] = []
            try:
                for t_idx, raw_table in enumerate(page.extract_tables() or []):
                    md = _rows_to_markdown(raw_table)
                    if md:
                        tables.append(TableData(
                            markdown=md, page_num=pn, table_index=t_idx,
                        ))
            except Exception as exc:
                logger.warning("Page %d table extract failed: %s", pn, exc)

            results.append(PageContent(
                page_num=pn, raw_text=raw_text, tables=tables, used_ocr=False,
            ))
            logger.info("Parsed page %d (%d chars, %d tables)", pn, len(raw_text), len(tables))
    return results


def main() -> None:
    # Page 12: affiliated institutions (Sharif etc.)
    # Page 34: EE department — split faculty from BSAI program list
    page_nums = [12, 34]
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    logger.info("Reprocessing pages %s from %s", page_nums, pdf_path)
    pages = _parse_pages(pdf_path, page_nums)
    chunks = chunk_pages(pages)
    logger.info("Created %d new chunks", len(chunks))
    for c in chunks:
        preview = c.text[:120].replace("\n", " | ")
        logger.info(
            "  p%d | type=%-20s | dept=%-25s | %s",
            c.page_num, c.content_type, c.department or "(none)", preview,
        )

    deleted = delete_by_page_nums(page_nums)
    logger.info("Removed %d old chunks", deleted)
    upsert_chunks(chunks)
    logger.info("Done. Upserted %d chunks for pages %s", len(chunks), page_nums)


if __name__ == "__main__":
    main()
