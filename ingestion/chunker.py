"""
ingestion/chunker.py

Structure-aware chunking strategy:

  1. Heading detection   — finds section / subsection boundaries in the text
  2. Table chunks        — each markdown table becomes its own chunk (never split)
  3. Text chunks         — prose under each heading, split at paragraph boundaries
                           if > MAX_CHUNK_TOKENS
  4. Metadata tagging    — every chunk carries: section_title, page_num,
                           content_type, department, has_table, ocr flag

The output is a list of Chunk dataclass instances ready for embedding.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import tiktoken

from config import MAX_CHUNK_TOKENS, MIN_CHUNK_CHARS
from ingestion.pdf_parser import PageContent, TableData

# ---------------------------------------------------------------------------
# Tokeniser (for size-capping text chunks)
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:      str
    text:          str          # The actual content (plain text or markdown table)
    section_title: str          # Nearest heading above this chunk
    page_num:      int
    content_type:  str          # See _classify_content_type()
    department:    str          # "" if not determinable
    has_table:     bool
    ocr:           bool


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

# Patterns that strongly suggest a section heading:
_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s&,\-]{4,}$"),                   # ALL-CAPS line
    re.compile(r"^(Department|Faculty|School|College)\s+of\s+", re.I),
    re.compile(r"^(Admission|Eligibility|Fee|Semester|Subject|Course|Program)\b", re.I),
    re.compile(r"^\d+\.\s+[A-Z]"),                             # "1. Introduction"
    re.compile(r"^[A-Z].{0,60}:$"),                            # "Something Title:"
    re.compile(r"^(Introduction|Overview|About|Vision|Mission|History)\b", re.I),
    # Program-list headings — must split from faculty blocks so offerings embed cleanly
    re.compile(r"^Courses of Study$", re.I),
    re.compile(r"^Undergraduate Programs?$", re.I),
    re.compile(r"^Postgraduate Programs?$", re.I),
    re.compile(r"^Programs? Offered$", re.I),
    re.compile(r"^The department offers:?$", re.I),
]

# Affiliated / external institution section markers
_AFFILIATED_SECTION_RE = re.compile(
    r"AFFILIATED\s+INSTITUTIONS|AFFILIATED\s+COLLEGES|AFFILIATED\s+CAMPUSES",
    re.I,
)
_AFFILIATED_INSTITUTION_RE = re.compile(
    r"(?:College|Institute)\s+of\s+.+"
    r"|Government\s+College\s+of\s+Technology"
    r"|NFC\s+Institute",
    re.I,
)


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    return any(p.match(stripped) for p in _HEADING_PATTERNS)


# ---------------------------------------------------------------------------
# Department extraction
# ---------------------------------------------------------------------------

_DEPT_RE = re.compile(
    r"(?:Department|Dept\.?)\s+of\s+([A-Za-z][A-Za-z\s&,\-]+?)(?:\s*\n|$|\.)",
    re.I,
)


def _extract_department(text: str) -> str:
    m = _DEPT_RE.search(text)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Content-type classifier (rule-based keyword matching)
# ---------------------------------------------------------------------------

_CONTENT_RULES: List[Tuple[str, List[str]]] = [
    # program_offering before faculty_list so dept program lists aren't buried
    ("program_offering",   ["undergraduate programs", "postgraduate programs",
                             "courses of study", "programs offered",
                             "the department offers", "degree is offered",
                             "b.sc. artificial intelligence", "b.sc. electrical"]),
    ("faculty_list",       ["phd", "msc", "m.sc", "lecturer", "professor",
                             "designation", "qualification", "assistant professor",
                             "associate professor", "dr.", "mr.", "ms."]),
    ("course_list",        ["credit hours", "credit hour", "course code",
                             "semester", "lab", "theory", "elective", "l-t-p"]),
    ("fee_structure",      ["fee", "tuition", "pkr", "rs.", "amount",
                             "per semester", "per annum", "charges", "dues"]),
    ("admission_criteria", ["eligibility", "merit", "aggregate", "criteria",
                             "criteria for admission", "fsc", "a-level",
                             "intermediate", "matric", "entry test"]),
]


def _classify_content_type(text: str, *, in_affiliated: bool = False) -> str:
    # Also catch affiliated content by heading cues if the section flag was missed
    lower = text.lower()
    if in_affiliated or "affiliated institutions" in lower or (
        "sharif college" in lower and "b.sc." in lower
    ):
        return "affiliated_college"
    for content_type, keywords in _CONTENT_RULES:
        if any(kw in lower for kw in keywords):
            return content_type
    return "policy_text"


# ---------------------------------------------------------------------------
# Text splitting (paragraph-aware, token-capped)
# ---------------------------------------------------------------------------

def _split_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """
    Split a block of text into chunks ≤ max_tokens, preserving paragraph breaks.
    Never splits mid-sentence if avoidable.
    """
    if _count_tokens(text) <= max_tokens:
        return [text]

    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_tokens = 0

        # A single paragraph bigger than max_tokens: split at sentence boundaries
        if para_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_tokens = _count_tokens(sent)
                if current_tokens + sent_tokens > max_tokens and current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                current_parts.append(sent)
                current_tokens += sent_tokens
        else:
            current_parts.append(para)
            current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_pages(pages: List[PageContent]) -> List[Chunk]:
    """
    Convert a list of PageContent objects into a flat list of Chunk objects.

    Strategy
    --------
    * Walk each page line-by-line, tracking the current section heading.
    * When a table is encountered, emit it as a separate chunk first, then
      continue with surrounding text.
    * Accumulated text under a heading is flushed when a new heading is
      detected OR when the page changes.
    """
    chunks: List[Chunk] = []

    # Track current section context
    current_heading: str = "Introduction"
    current_department: str = ""
    in_affiliated: bool = False
    text_buffer: List[str] = []
    buffer_page: int = 1
    buffer_ocr: bool = False

    def flush_text_buffer() -> None:
        """Emit chunks from the accumulated text buffer."""
        nonlocal text_buffer, buffer_page, buffer_ocr
        joined = "\n".join(text_buffer).strip()
        if len(joined) >= MIN_CHUNK_CHARS:
            for part in _split_text(joined):
                if len(part.strip()) >= MIN_CHUNK_CHARS:
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=part.strip(),
                        section_title=current_heading,
                        page_num=buffer_page,
                        content_type=_classify_content_type(
                            part, in_affiliated=in_affiliated
                        ),
                        # Affiliated pages are not UET departments — clear dept
                        department="" if in_affiliated else current_department,
                        has_table=False,
                        ocr=buffer_ocr,
                    ))
        text_buffer = []

    # Build a mapping: page_num → list of TableData (already sorted by table_index)
    table_map: dict[int, List[TableData]] = {}
    for page in pages:
        if page.tables:
            table_map[page.page_num] = sorted(page.tables, key=lambda t: t.table_index)

    # Track which table on each page we've emitted to avoid duplicates
    emitted_tables: dict[int, set] = {pn: set() for pn in table_map}

    def emit_pending_tables(page_num: int, before_line_idx: int = -1) -> None:
        """
        Emit any table chunks for `page_num` that haven't been emitted yet.
        We emit ALL pending tables when we flush the buffer (i.e. on heading change
        or page end), so tables stay close to their surrounding context.
        """
        if page_num not in table_map:
            return
        for td in table_map[page_num]:
            if td.table_index in emitted_tables.get(page_num, set()):
                continue
            emitted_tables[page_num].add(td.table_index)
            dept = (
                ""
                if in_affiliated
                else (current_department or _extract_department(td.markdown))
            )
            # Prefix the table with its section heading for retrieval context
            table_text = f"**{current_heading}**\n\n{td.markdown}"
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=table_text,
                section_title=current_heading,
                page_num=page_num,
                content_type=_classify_content_type(
                    td.markdown, in_affiliated=in_affiliated
                ),
                department=dept,
                has_table=True,
                ocr=False,
            ))

    # ---- Main pass ---------------------------------------------------------
    for page in pages:
        buffer_page = page.page_num
        buffer_ocr  = page.used_ocr
        lines        = page.raw_text.splitlines() if page.raw_text else []

        for line in lines:
            stripped = line.strip()

            # Enter / leave affiliated-institution sections
            if _AFFILIATED_SECTION_RE.search(stripped):
                flush_text_buffer()
                in_affiliated = True
                current_department = ""
                emit_pending_tables(page.page_num)
                current_heading = stripped
                text_buffer = [stripped]
                continue

            # A real UET department heading exits the affiliated context
            new_dept = _extract_department(stripped) if _is_heading(stripped) else ""

            if _is_heading(stripped):
                # Flush existing buffer before starting new section
                flush_text_buffer()
                emit_pending_tables(page.page_num)

                new_heading = stripped
                # If this heading contains a department name, update context
                if new_dept:
                    current_department = new_dept
                    in_affiliated = False
                elif in_affiliated and _AFFILIATED_INSTITUTION_RE.search(new_heading):
                    # Keep affiliated flag; do not invent a UET department
                    current_department = ""

                current_heading = new_heading
                # Add the heading itself into the new buffer so it has context
                text_buffer = [stripped]
            else:
                text_buffer.append(line)

        # End of page: flush text
        flush_text_buffer()
        # Emit any remaining tables on this page
        emit_pending_tables(page.page_num)

    # Final flush
    flush_text_buffer()

    return chunks
