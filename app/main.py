"""
app/main.py — Chainlit chat application

Run with:
    chainlit run app/main.py

Features
--------
* Streaming responses from Groq via the RAG chain
* Collapsible source citations (one per retrieved chunk)
* Session-scoped conversation history for multi-turn context
* Thumbs-up / thumbs-down feedback logged to logs/feedback.jsonl
* Clean welcome message explaining the app's scope
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv

from rag_chain import rag_stream
from config import FEEDBACK_LOG

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Ensure logs directory exists
Path(FEEDBACK_LOG).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------
WELCOME_MSG = """\
## 🎓 UET Guidance Chatbot
*(powered by a RAG pipeline)*

Ask me anything about **UET admissions, departments, faculty, courses, or fees** — \
every answer is grounded directly in the official prospectus, with page citations \
so you can verify it yourself.

- 🎓 **Faculty** — who teaches which subject / department
- 📚 **Courses & Semesters** — subject codes, credit hours, scheme of studies
- 📋 **Admission Criteria** — eligibility, merit, entry test requirements
- 💰 **Fee Structure** — tuition and other charges
- 🏛️ **Departments & Programs** — descriptions, duration, specialisations

> **Scope:** I only answer from the official prospectus. \
If something isn't covered there, I'll tell you honestly.

---
📖 *Knowledge source: Undergraduate Prospectus 2026*
 *--Created by UBAI KAMRAN--*
"""


# ---------------------------------------------------------------------------
# Chainlit lifecycle hooks
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Initialise session state and display the welcome banner."""
    cl.user_session.set("history", [])
    await cl.Message(content=WELCOME_MSG, author="UET Assistant").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle an incoming user message through the RAG pipeline."""
    history: list = cl.user_session.get("history") or []
    question = message.content.strip()

    if not question:
        return

    # ---- Stream the response -----------------------------------------------
    response_msg = cl.Message(content="", author="UET Assistant")
    await response_msg.send()

    full_response = []
    source_elements = []
    retrieved_chunks = None

    async for token, chunks in rag_stream(question, history=history):
        if chunks and retrieved_chunks is None:
            retrieved_chunks = chunks
            # Build source elements (shown collapsed under the answer)
            for i, chunk in enumerate(chunks, start=1):
                label = (
                    f"📄 Source {i}: {chunk.section_title[:60]} "
                    f"(Page {chunk.page_num})"
                )
                source_elements.append(
                    cl.Text(
                        name=label,
                        content=chunk.text,
                        display="side",
                    )
                )

        await response_msg.stream_token(token)
        full_response.append(token)

    # Attach sources and finalise the message
    if source_elements:
        response_msg.elements = source_elements

    await response_msg.update()

    # ---- Update conversation history ----------------------------------------
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": "".join(full_response)})
    cl.user_session.set("history", history)


# ---------------------------------------------------------------------------
# Feedback logging
# ---------------------------------------------------------------------------

@cl.on_feedback
async def on_feedback(feedback: cl.types.Feedback) -> None:
    """Log thumbs-up / thumbs-down feedback to a local JSONL file."""
    record = {
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "message_id": feedback.forId,
        "value":      feedback.value,           # 1 = thumbs up, 0 = thumbs down
        "comment":    getattr(feedback, "comment", None),
    }
    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Feedback logged: %s", record)
    except Exception as exc:
        logger.error("Failed to write feedback: %s", exc)
