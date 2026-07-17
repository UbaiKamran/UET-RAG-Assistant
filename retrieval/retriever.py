"""
retrieval/retriever.py

Hybrid retrieval: semantic similarity + keyword / metadata boosting.

Pipeline
--------
1. Embed the user query.
2. Fetch CANDIDATE_FETCH results from ChromaDB (over-fetch for re-ranking).
3. Analyse the query for entity signals:
   - Department name  → boost chunks whose `department` metadata matches
   - Content keywords → boost chunks whose `content_type` matches
4. Re-rank by boosted score, keep TOP_K.
5. Apply SIMILARITY_THRESHOLD — drop anything too dissimilar.
6. Return list of RetrievedChunk dataclass instances.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from config import (
    BOOST_DELTA,
    CANDIDATE_FETCH,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from retrieval.embedder import embed_query
from retrieval.vector_store import query_collection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text:          str
    section_title: str
    page_num:      int
    content_type:  str
    department:    str
    has_table:     bool
    score:         float    # boosted cosine-distance (lower = better match)


# ---------------------------------------------------------------------------
# Entity detection helpers
# ---------------------------------------------------------------------------

# Department name keywords — extend if more departments are added
_DEPT_KEYWORDS = [
    "computer science", "cs", "electrical", "mechanical", "civil",
    "chemical", "software", "mathematics", "physics", "chemistry",
    "management", "architecture", "metallurgy", "petroleum",
    "industrial", "environmental", "urban", "city", "planning",
    "humanities", "social", "economics",
]

# Content-type signal keywords
_CONTENT_SIGNALS = {
    "faculty_list":       ["faculty", "teacher", "lecturer", "professor", "who teaches",
                            "instructor", "dr.", "department staff"],
    "course_list":        ["subject", "course", "semester", "credit hour", "syllabus",
                            "curriculum", "scheme of studies", "elective"],
    "program_offering":   ["which department", "which dept", "who offers", "offers",
                            "offered by", "program offered", "degree offered",
                            "which program", "bsai", "bsc ", "b.sc"],
    "fee_structure":      ["fee", "fees", "tuition", "cost", "charges", "payment",
                            "pkr", "how much"],
    "admission_criteria": ["admission", "eligibility", "merit", "criteria",
                            "requirements", "entry test", "aggregate", "qualify"],
}

# Queries about UET departments should not surface affiliated colleges first
_DEPT_QUERY_RE = re.compile(
    r"\b(department|dept\.?|which\s+faculty|offered\s+by|who\s+offers)\b",
    re.I,
)

# Penalty added to cosine-distance for affiliated_college on department queries
_AFFILIATED_PENALTY = 0.20

# Extra boost (distance reduction) when a UET dept chunk names the queried program
_PROGRAM_MATCH_BOOST = 0.12

_PROGRAM_TERMS = [
    ("artificial intelligence", ["artificial intelligence", "bsai"]),
    ("computer science", ["computer science", "bscs"]),
    ("electrical engineering", ["electrical engineering", "bsee"]),
    ("software engineering", ["software engineering"]),
    ("data science", ["data science"]),
    ("cyber security", ["cyber security", "cybersecurity"]),
]


def _detect_department(query: str) -> Optional[str]:
    lower = query.lower()
    for kw in _DEPT_KEYWORDS:
        if kw in lower:
            return kw
    return None


def _detect_content_type(query: str) -> Optional[str]:
    lower = query.lower()
    for ctype, signals in _CONTENT_SIGNALS.items():
        if any(s in lower for s in signals):
            return ctype
    return None


def _detect_program_terms(query: str) -> List[str]:
    lower = query.lower()
    matched: List[str] = []
    for canonical, aliases in _PROGRAM_TERMS:
        if any(a in lower for a in aliases):
            matched.append(canonical)
    return matched


def _is_department_query(query: str) -> bool:
    return bool(_DEPT_QUERY_RE.search(query))


# ---------------------------------------------------------------------------
# Booster
# ---------------------------------------------------------------------------

def _boost_score(
    base_score: float,
    metadata: dict,
    document: str,
    target_dept: Optional[str],
    target_ctype: Optional[str],
    program_terms: List[str],
    demote_affiliated: bool,
) -> float:
    """
    Lower the distance score (improving rank) when metadata signals match.
    ChromaDB cosine-distance: 0 = perfect match, 2 = worst.
    Subtracting BOOST_DELTA moves a chunk up the ranking.
    """
    boosted = base_score
    dept = (metadata.get("department") or "").lower()
    ctype = metadata.get("content_type") or ""
    doc_lower = (document or "").lower()

    if target_dept and target_dept in dept:
        boosted -= BOOST_DELTA

    if target_ctype and ctype == target_ctype:
        boosted -= BOOST_DELTA

    # Prefer real UET department program listings over affiliated colleges
    if demote_affiliated and ctype == "affiliated_college":
        boosted += _AFFILIATED_PENALTY

    if program_terms and ctype != "affiliated_college":
        if any(term in doc_lower for term in program_terms):
            # Stronger boost when a named UET department owns the chunk
            if dept:
                boosted -= _PROGRAM_MATCH_BOOST
            elif ctype == "program_offering":
                boosted -= _PROGRAM_MATCH_BOOST * 0.75

    return boosted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for `query` using hybrid ranking.

    Parameters
    ----------
    query : natural-language user question
    top_k : maximum number of chunks to return (default from config)

    Returns
    -------
    List of RetrievedChunk, sorted by boosted score (best first).
    Empty list if nothing clears SIMILARITY_THRESHOLD.
    """
    # 1. Embed query
    q_vec = embed_query(query)

    # 2. Semantic search (over-fetch — larger window so diluted but correct
    #    department chunks can still reach the re-ranker)
    n_fetch = max(CANDIDATE_FETCH, top_k * 6)
    raw = query_collection(query_embedding=q_vec, n_results=n_fetch)

    if not raw["ids"] or not raw["ids"][0]:
        logger.warning("ChromaDB returned no results for query: %s", query)
        return []

    # 3. Unpack results (ChromaDB returns nested lists)
    ids        = raw["ids"][0]
    documents  = raw["documents"][0]
    metadatas  = raw["metadatas"][0]
    distances  = raw["distances"][0]

    # 4. Detect query signals
    target_dept  = _detect_department(query)
    target_ctype = _detect_content_type(query)
    program_terms = _detect_program_terms(query)
    demote_affiliated = _is_department_query(query)

    if target_dept:
        logger.debug("Detected department signal: %s", target_dept)
    if target_ctype:
        logger.debug("Detected content-type signal: %s", target_ctype)
    if program_terms:
        logger.debug("Detected program terms: %s", program_terms)

    # 5. Boost + collect candidates
    candidates = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        boosted = _boost_score(
            dist, meta, doc, target_dept, target_ctype,
            program_terms, demote_affiliated,
        )
        candidates.append((boosted, doc, meta))

    # 6. Sort by boosted score (ascending = better)
    candidates.sort(key=lambda x: x[0])

    # 7. Threshold filter + pick top_k
    results: List[RetrievedChunk] = []
    for boosted_score, doc, meta in candidates[:top_k * 2]:  # keep window for threshold
        if boosted_score > SIMILARITY_THRESHOLD:
            continue  # too dissimilar — skip
        results.append(RetrievedChunk(
            text=doc,
            section_title=meta.get("section_title", "Unknown"),
            page_num=int(meta.get("page_num", 0)),
            content_type=meta.get("content_type", "general"),
            department=meta.get("department", ""),
            has_table=meta.get("has_table", "False") == "True",
            score=boosted_score,
        ))
        if len(results) >= top_k:
            break

    logger.info(
        "Retrieved %d chunks for query (threshold=%.2f, top_k=%d)",
        len(results), SIMILARITY_THRESHOLD, top_k,
    )
    return results
