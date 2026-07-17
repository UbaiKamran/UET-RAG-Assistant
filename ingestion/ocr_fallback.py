"""
ingestion/ocr_fallback.py

Renders a single PDF page to an image and runs Tesseract OCR on it.
Used only when pdfplumber extracts zero text AND zero tables from a page
(typically scanned/image-only pages).

Requires:
  - pymupdf  (pip install pymupdf)
  - pytesseract + Tesseract binary (see README for install instructions)
"""

from __future__ import annotations

import logging
from io import BytesIO

logger = logging.getLogger(__name__)


def ocr_page(pdf_path: str, page_index: int, dpi: int = 200) -> str:
    """
    Render PDF page at `page_index` (0-based) to an image and OCR it.

    Parameters
    ----------
    pdf_path   : path to the PDF file
    page_index : 0-based page index
    dpi        : rendering resolution (200 dpi is a good balance of speed / accuracy)

    Returns
    -------
    Extracted text string (may be empty if Tesseract finds nothing).
    """
    # --- Try to import optional heavy dependencies lazily --------------------
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.warning("pymupdf not installed — OCR fallback unavailable")
        return ""

    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("pytesseract / Pillow not installed — OCR fallback unavailable")
        return ""

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_index)

        # Render at given DPI (default 72 dpi → scale factor = dpi/72)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert pixmap to PIL Image via bytes
        img_bytes = pix.tobytes("png")
        image = Image.open(BytesIO(img_bytes))

        # Run Tesseract
        text: str = pytesseract.image_to_string(image, lang="eng")
        doc.close()
        return text

    except pytesseract.TesseractNotFoundError:
        logger.warning(
            "Tesseract binary not found. Install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "and ensure it is on your PATH.  OCR fallback disabled."
        )
        return ""
    except Exception as exc:
        logger.warning("OCR failed on page %d: %s", page_index, exc)
        return ""
