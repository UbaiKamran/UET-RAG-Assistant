"""
app/rag_chain.py

Orchestrates the full RAG pipeline for a single user turn:
  1. Retrieve relevant chunks (hybrid retrieval)
  2. Format context + build prompts
  3. Call Groq API with streaming
  4. Yield text tokens for Chainlit streaming display

Also exports a helper to build Chainlit source elements from retrieved chunks.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, List, Optional

from dotenv import load_dotenv
from groq import AsyncGroq

from prompts import SYSTEM_PROMPT, build_user_prompt, format_chunks_as_context
from config import GROQ_MODEL, MAX_TOKENS, TEMPERATURE
from retrieval.retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

# Load env once at import time
load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not _GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set.  "
        "Please add it to your .env file:  GROQ_API_KEY=gsk_..."
    )

# Lazily-created async client (one per process)
_groq_client: Optional[AsyncGroq] = None


def _get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=_GROQ_API_KEY)
    return _groq_client


# ---------------------------------------------------------------------------
# Conversation history helpers
# ---------------------------------------------------------------------------

def build_history_messages(history: List[dict]) -> List[dict]:
    """
    Convert session history (list of {role, content} dicts) to Groq message format.
    Keeps only the last 6 turns to avoid context blowout.
    """
    return history[-12:]   # 6 user + 6 assistant turns


# ---------------------------------------------------------------------------
# Main RAG function
# ---------------------------------------------------------------------------

async def rag_stream(
    question: str,
    history: Optional[List[dict]] = None,
) -> AsyncGenerator[tuple[str, List[RetrievedChunk]], None]:
    """
    Async generator that yields (token_str, chunks) tuples.

    The `chunks` list is yielded exactly ONCE on the first iteration
    so the caller can display source elements immediately.  Subsequent
    yields have chunks=[] to keep the protocol simple.

    Parameters
    ----------
    question : the user's raw question string
    history  : list of {role, content} prior turns (can be None)

    Yields
    ------
    (token: str, chunks: List[RetrievedChunk])
    """
    history = history or []

    # 1. Retrieve context
    chunks = retrieve(question)

    if not chunks:
        # Nothing clears threshold — decline gracefully
        decline_msg = (
            "This information isn't covered in the prospectus I have access to."
        )
        yield (decline_msg, [])
        return

    # 2. Build prompts
    context_str  = format_chunks_as_context(chunks)
    user_message = build_user_prompt(context=context_str, question=question)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(build_history_messages(history))
    messages.append({"role": "user", "content": user_message})

    # 3. Call Groq with streaming
    client = _get_groq_client()

    try:
        stream = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )

        first = True
        async for chunk_obj in stream:
            delta = chunk_obj.choices[0].delta
            token = delta.content or ""
            if token:
                if first:
                    yield (token, chunks)   # send chunks alongside first token
                    first = False
                else:
                    yield (token, [])

        # If nothing was yielded (empty LLM response)
        if first:
            yield ("I was unable to generate a response. Please try again.", chunks)

    except Exception as exc:
        logger.error("Groq API error: %s", exc, exc_info=True)
        error_msg = f"⚠️ API error: {exc}"
        yield (error_msg, chunks)
