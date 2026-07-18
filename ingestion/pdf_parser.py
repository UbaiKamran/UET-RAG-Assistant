"""
ingestion/pdf_parser.py

Extracts text and tables from each page of the prospectus PDF.

Primary engine : pdfplumber (text + structured tables)
Fallback       : OCR via ocr_fallback.py (for scanned/image-only pages)

Returns a list of PageContent dataclass instances, one per page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber

from ingestion.ocr_fallback import ocr_page

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TableData:
    """A single extracted table rendered as a Markdown string."""
    markdown: str               # Full markdown table (header + rows)
    page_num: int
    table_index: int            # Which table on this page (0-based)


@dataclass
class PageContent:
    """All content extracted from one PDF page."""
    page_num: int               # 1-based
    raw_text: str               # Plain text (possibly empty for scanned pages)
    tables: List[TableData]     # Structured tables found on this page
    used_ocr: bool = False      # True if OCR fallback was used


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows_to_markdown(rows: List[List[Optional[str]]]) -> str:
    """
    Convert pdfplumber table rows to a GitHub-flavoured Markdown table.

    The first non-empty row is treated as the header.
    Multi-line cell values have internal newlines collapsed to spaces.
    """
    if not rows:
        return ""

    # Clean cells: replace None with "", collapse internal newlines
    cleaned: List[List[str]] = []
    for row in rows:
        cleaned_row = [
            (cell or "").replace("\n", " ").strip()
            for cell in row
        ]
        cleaned.append(cleaned_row)

    # Remove completely empty rows
    cleaned = [r for r in cleaned if any(c for c in r)]
    if not cleaned:
        return ""

    # First row = header
    header = cleaned[0]
    sep = ["---"] * len(header)
    body = cleaned[1:]

    def _fmt_row(cells: List[str]) -> str:
        # Pad / align columns
        return "| " + " | ".join(cells) + " |"

    lines = [_fmt_row(header), _fmt_row(sep)]
    for row in body:
        # Ensure row has same column count as header (pad or truncate)
        padded = row + [""] * max(0, len(header) - len(row))
        padded = padded[: len(header)]
        lines.append(_fmt_row(padded))

    return "\n".join(lines)


def _is_page_empty(text: str, tables: List[TableData]) -> bool:
    """True if a page yielded neither text nor tables."""
    return not text.strip() and not tables


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path) -> List[PageContent]:
    """
    Parse the entire PDF and return one PageContent per page.

    Parameters
    ----------
    pdf_path : path to the PDF file

    Returns
    -------
    List[PageContent] — one item per page, in order.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results: List[PageContent] = []

    logger.info("Opening PDF: %s", pdf_path)

    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        logger.info("Total pages: %d", total)

        for i, page in enumerate(pdf.pages):
            page_num = i + 1  # 1-based

            # --- Extract plain text ------------------------------------------
            try:
                raw_text: str = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            except Exception as exc:
                logger.warning("Page %d: text extraction failed (%s)", page_num, exc)
                raw_text = ""

            # --- Extract tables -----------------------------------------------
            tables: List[TableData] = []
            try:
                raw_tables = page.extract_tables()
                for t_idx, raw_table in enumerate(raw_tables or []):
                    md = _rows_to_markdown(raw_table)
                    if md:
                        tables.append(TableData(
                            markdown=md,
                            page_num=page_num,
                            table_index=t_idx,
                        ))
            except Exception as exc:
                logger.warning("Page %d: table extraction failed (%s)", page_num, exc)

            # --- OCR fallback for image-only pages ---------------------------
            used_ocr = False
            if _is_page_empty(raw_text, tables):
                logger.info("Page %d: no content — attempting OCR fallback", page_num)
                ocr_text = ocr_page(str(pdf_path), page_num - 1)  # 0-based index
                if ocr_text.strip():
                    raw_text = ocr_text
                    used_ocr = True
                    logger.info("Page %d: OCR recovered %d chars", page_num, len(ocr_text))
                else:
                    logger.warning("Page %d: OCR also yielded nothing — skipping", page_num)

            results.append(PageContent(
                page_num=page_num,
                raw_text=raw_text,
                tables=tables,
                used_ocr=used_ocr,
            ))

            if page_num % 10 == 0 or page_num == total:
                logger.info("Parsed %d / %d pages", page_num, total)

    logger.info(
        "Parsing complete. Pages with tables: %d  |  Pages with OCR: %d",
        sum(1 for p in results if p.tables),
        sum(1 for p in results if p.used_ocr),
    )
    return results
