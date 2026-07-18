import sys
import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
import uuid

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag_chain import rag_stream
import app.rag_chain as rag_chain
from config import FEEDBACK_LOG

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

Path(FEEDBACK_LOG).parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="UET Guidance", page_icon="🎓")

st.title("🎓 UET Guidance")
st.markdown("Ask me anything about UET admissions, departments, faculty, courses, or fees — every answer is grounded directly in the official prospectus, with page citations.")

# ---------------------------------------------------------------------------
# Session State & History
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = set()

# Display chat history
for i, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        
        # Display feedback buttons for assistant messages
        if msg["role"] == "assistant" and i not in st.session_state.feedbacks:
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("👍", key=f"up_{i}"):
                    # Log thumbs up
                    record = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "message_id": str(uuid.uuid4()),
                        "value": 1,
                        "comment": None
                    }
                    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    st.session_state.feedbacks.add(i)
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{i}"):
                    # Log thumbs down
                    record = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "message_id": str(uuid.uuid4()),
                        "value": 0,
                        "comment": None
                    }
                    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    st.session_state.feedbacks.add(i)
                    st.rerun()

# ---------------------------------------------------------------------------
# Chat Input & RAG Stream Execution
# ---------------------------------------------------------------------------
if question := st.chat_input("Ask a question about the UET Prospectus..."):
    # Add user message to state and display it
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # We need to collect the response.
    # Since Streamlit and asyncio don't always mix well across reruns (global client issue),
    # we collect the full response via asyncio.run and display it.
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        async def fetch_response():
            full_response = []
            retrieved_chunks = []
            try:
                async for token, chunks in rag_stream(question, history=st.session_state.history[:-1]):
                    if chunks and not retrieved_chunks:
                        retrieved_chunks.extend(chunks)
                    full_response.append(token)
            except Exception as e:
                logger.error(f"Error during RAG stream: {e}")
                full_response.append(f"\n\nError: {e}")
            return "".join(full_response), retrieved_chunks

        try:
            answer, sources = asyncio.run(fetch_response())
        finally:
            # Reset the global AsyncGroq client to avoid "attached to a different loop" errors
            # on subsequent Streamlit reruns.
            rag_chain._groq_client = None

        message_placeholder.markdown(answer)
        
        if sources:
            with st.expander("Sources"):
                for i, chunk in enumerate(sources, start=1):
                    st.markdown(f"**📄 Source {i}: {chunk.section_title[:60]} (Page {chunk.page_num})**")
                    st.text(chunk.text)
                    
        st.session_state.history.append({
            "role": "assistant", 
            "content": answer
        })
        st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("📖 Knowledge source: Undergraduate Prospectus 2026")
st.caption("✨ Created by Ubai Kamran")
