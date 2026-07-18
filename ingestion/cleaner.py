"""
ingestion/cleaner.py

Post-processes raw extracted text from each page:
  1. Detects repeating headers/footers (strings appearing verbatim on ≥ N pages
     near the top / bottom of the page text) and strips them.
  2. Removes standalone page-number lines.
  3. Collapses excessive blank lines.

Usage
-----
    from ingestion.cleaner import build_cleaner, clean_text

    cleaner = build_cleaner(page_texts)          # pass ALL raw page texts first
    cleaned = [clean_text(p, cleaner) for p in page_texts]
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HEADER_FOOTER_LINES = 3     # How many lines from top/bottom to inspect
MIN_REPEAT_PAGES    = 3     # A line must appear on ≥ this many pages to be noise
MIN_LINE_LENGTH     = 4     # Ignore very short lines (single chars, etc.)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Cleaner:
    noise_lines: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cleaner(page_texts: List[str]) -> Cleaner:
    """
    Analyse all page texts to identify repeated header/footer lines.

    Parameters
    ----------
    page_texts : raw text strings, one per page (order = page order)

    Returns
    -------
    Cleaner instance that can be passed to clean_text()
    """
    # Count how many pages each candidate header/footer line appears on
    line_page_count: Counter = Counter()

    for text in page_texts:
        lines = text.splitlines()
        if not lines:
            continue

        # Candidate noise lines: first N and last N non-empty lines of the page
        top_lines    = _non_empty_head(lines, HEADER_FOOTER_LINES)
        bottom_lines = _non_empty_tail(lines, HEADER_FOOTER_LINES)
        candidates   = set(top_lines + bottom_lines)

        for line in candidates:
            normalised = line.strip()
            if len(normalised) >= MIN_LINE_LENGTH:
                line_page_count[normalised] += 1

    noise: Set[str] = {
        line
        for line, count in line_page_count.items()
        if count >= MIN_REPEAT_PAGES
    }
    return Cleaner(noise_lines=noise)


def clean_text(text: str, cleaner: Cleaner) -> str:
    """
    Remove headers, footers, and page numbers from a single page's text.

    Parameters
    ----------
    text    : raw page text
    cleaner : built by build_cleaner()

    Returns
    -------
    Cleaned text string.
    """
    lines = text.splitlines()
    cleaned: List[str] = []

    for line in lines:
        stripped = line.strip()

        # 1. Skip page-number-only lines  (e.g. "42", "- 42 -", "Page 42")
        if _is_page_number(stripped):
            continue

        # 2. Skip identified header/footer noise
        if stripped in cleaner.noise_lines:
            continue

        cleaned.append(line)

    # 3. Collapse 3+ consecutive blank lines → single blank line
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PAGE_NUM_RE = re.compile(
    r"^[-–|]*\s*(?:page|pg\.?)?\s*\d+\s*[-–|]*$",
    re.IGNORECASE,
)


def _is_page_number(line: str) -> bool:
    return bool(_PAGE_NUM_RE.match(line))


def _non_empty_head(lines: List[str], n: int) -> List[str]:
    result = []
    for line in lines:
        if line.strip():
            result.append(line.strip())
            if len(result) == n:
                break
    return result


def _non_empty_tail(lines: List[str], n: int) -> List[str]:
    return _non_empty_head(list(reversed(lines)), n)
