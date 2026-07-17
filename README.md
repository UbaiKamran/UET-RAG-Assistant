
# UET Prospectus RAG Chatbot

A production-quality **Retrieval-Augmented Generation (RAG)** chatbot that answers questions
grounded exclusively in the UET undergraduate prospectus — with source citations and graceful
decline when a question is out of scope.

---

## Folder Structure

```
.
├── .env                    ← GROQ_API_KEY (never commit)
├── .gitignore
├── prospectus.pdf          ← Source document (gitignored; place locally)
├── config.py               ← All tunable config values
├── requirements.txt
├── chainlit.md             ← Chainlit sidebar "Readme" panel content
├── .chainlit/
│   └── config.toml         ← Chainlit app name, description, custom CSS path
├── public/
│   ├── style.css           ← Footer credit line + visual tweaks
│   └── theme.json          ← Color palette overrides (if supported by installed version)
├── scripts/
│   ├── build_index.py      ← Run ONCE to build the vector index
│   └── reprocess_pages.py  ← Optional: re-index specific pages only
├── ingestion/
│   ├── pdf_parser.py       ← pdfplumber text + table extraction
│   ├── ocr_fallback.py     ← pytesseract OCR for scanned pages
│   ├── cleaner.py          ← Strip headers, footers, page numbers
│   └── chunker.py          ← Structure-aware chunking + metadata tagging
├── retrieval/
│   ├── embedder.py         ← sentence-transformers wrapper
│   ├── vector_store.py     ← ChromaDB (persisted to disk)
│   └── retriever.py        ← Hybrid retrieval (semantic + metadata boost)
├── app/
│   ├── prompts.py          ← System & user prompt templates
│   ├── rag_chain.py        ← Groq API streaming + RAG orchestration
│   └── main.py             ← Chainlit UI, including WELCOME_MSG
├── data/
│   └── chroma_db/          ← Persisted vector store (gitignored)
└── logs/
    └── feedback.jsonl      ← Thumbs-up/down feedback log
```

**Note:** `prospectus.pdf` and `data/chroma_db/` are excluded from git. After cloning,
place the PDF in the project root and rebuild the index with
`python scripts/build_index.py`.

---

## Prerequisites

### 1. Python 3.10+ (this project uses a Python 3.13 venv: `venv313`)

### 2. Chainlit 2.11.0+

Config structure (`.chainlit/config.toml`) differs meaningfully from Chainlit 1.x —
make sure `requirements.txt` resolves to `chainlit>=2.11.0`, not an older version.

### 3. Tesseract OCR *(optional — only needed for scanned pages)*

**Windows**: Download the installer from
[UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
and add the install directory (e.g. `C:\Program Files\Tesseract-OCR`) to your `PATH`.

If Tesseract is not installed, the pipeline still works for text-based PDFs and logs
a warning for any image-only pages.

---

## Setup & Running

### Step 1 — Install dependencies

```bash
# Activate the existing venv (Windows PowerShell)
.\venv313\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Step 2 — Configure the API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_...
```

### Step 3 — Build the index *(run once, or after updating the PDF)*

```bash
python scripts/build_index.py
```

This will:
- Parse `prospectus.pdf` (text + tables)
- Strip headers/footers/page numbers
- Chunk by logical structure (headings, departments, etc.)
- Embed locally with `sentence-transformers`
- Persist to ChromaDB in `./data/chroma_db/`

**Estimated time**: 2–5 minutes on first run (model download + encoding).
Subsequent runs are faster (model cached).

To re-index only specific pages without a full rebuild, use:

```bash
python scripts/reprocess_pages.py
```

### Step 4 — Launch the chat app

```bash
chainlit run app/main.py
```

Open `http://localhost:8000` in your browser.

---

## Configuration

All tunable values are in **`config.py`**:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `GROQ_MODEL` | *(check config.py)* | Groq LLM model slug — check [console.groq.com/docs/models](https://console.groq.com/docs/models) for current availability, as Groq periodically deprecates older models (e.g. `llama-3.3-70b-versatile` has reportedly been deprecated in favor of `openai/gpt-oss-120b`) |
| `TOP_K` | `5` | Chunks sent to LLM per query |
| `SIMILARITY_THRESHOLD` | `0.70` | Max cosine distance to accept |
| `MAX_CHUNK_TOKENS` | `400` | Max tokens per text chunk |
| `TEMPERATURE` | `0.1` | LLM temperature |

To swap the Groq model, change `GROQ_MODEL` in `config.py`. No other file needs editing.

---

## UI / Branding

The chat interface is customized via:
- **`chainlit.md`** — sidebar "Readme" panel content
- **`app/main.py`** → `WELCOME_MSG` — the in-chat welcome message shown on first load
- **`.chainlit/config.toml`** → `[UI]` section — app name, description, custom CSS path
- **`public/style.css`** — footer credit line and visual tweaks (linked via `custom_css` in config.toml)
- **`public/theme.json`** — color palette overrides (not confirmed to auto-load in all Chainlit versions — verify via browser dev tools if colors don't apply, and fall back to CSS custom properties in `style.css` if needed)

After changing any of these, fully restart the Chainlit server (`Ctrl+C` then re-run) and
hard-refresh the browser (`Ctrl+Shift+R`) or use an incognito window — Chainlit's frontend
caches aggressively.

---

## How It Works

```
User question
    ↓
[Hybrid Retriever]
  Semantic search (ChromaDB cosine similarity)
  + Metadata boost (department / content-type / program signals)
  + Affiliated-college demotion on department queries
  + Threshold filter
    ↓
[Groq LLM] ← system prompt enforcing citation-only answers
    ↓
[Chainlit UI] — streamed answer + collapsible source panels
```

---

## Feedback

Every thumbs-up / thumbs-down in the chat UI is appended to `logs/feedback.jsonl`:
```json
{"timestamp": "2026-07-18T10:23:00Z", "message_id": "...", "value": 1, "comment": null}
```
`value=1` = thumbs up, `value=0` = thumbs down.

---

## Troubleshooting

- **Windows async errors on launch**: install `nest-asyncio` (already in `requirements.txt`)
- **`ModuleNotFoundError` on launch**: confirm you're running inside `venv313` and ran
  `pip install -r requirements.txt` while it was active
- **Port already in use** (`WinError 10048`): another Chainlit process may still be running
  from a previous session — check with `netstat -ano | findstr :8001`, then
  `taskkill /PID <pid> /F`, or launch on a different port with `--port 8002`
- **UI/branding changes not showing**: fully stop and restart the server (not just a browser
  refresh), and open in an incognito window to rule out frontend caching
- **Answers seem to be missing content you know is in the prospectus**: check
  `SIMILARITY_THRESHOLD` in `config.py` first — too strict a threshold silently discards
  valid matches before the LLM ever sees them
