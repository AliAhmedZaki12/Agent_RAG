# Agentic RAG

A production-ready multi-agent Retrieval-Augmented Generation system. Three collaborative AI agents — Router, Retriever, and Critic — work together to answer questions from your documents, the web, or conversation memory, with automatic answer verification before every response.

---

## How it works

```
User Question
     │
     ▼
┌─────────────┐    {"route": "rag"|"web"|"memory"}
│   Router    │─────────────────────────────────────┐
│   Agent     │                                     │
└─────────────┘                                     ▼
                                          ┌──────────────────┐
                                          │  Retriever Agent │
                                          │  ┌────────────┐  │
                                          │  │ RAG Tool   │  │  Dense + BM25
                                          │  │ Web Tool   │  │  + CrossEncoder
                                          │  └────────────┘  │
                                          └──────────────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────┐
                                          │  Critic Agent    │  grounded: true/false
                                          │  (fact-checks    │─────────────────────┐
                                          │   every claim)   │                     │
                                          └──────────────────┘                     │
                                                                                   ▼
                                                                         Final Verified Answer
                                                                         + saved to Memory
```

If the Critic marks the answer as ungrounded, the pipeline retries automatically (up to 3 times) with an escalating search strategy.

---

## Project structure

```
agentic_rag/
├── README.md
├── .env.example               ← copy to .env and fill in your keys
├── requirements.txt           ← all dependencies with pinned versions
├── dashboard.html             ← standalone web UI (open in browser)
├── main.py                    ← FastAPI server  →  uvicorn main:app
├── celery_app.py              ← async task queue
└── backend/
    ├── config.py              ← validates all env vars at startup
    ├── agents.py              ← Router · Retriever · Critic agents
    ├── tasks.py               ← CrewAI task definitions
    ├── orchestrator.py        ← pipeline controller (MAX_ITERS=3)
    ├── ingestion.py           ← chunk → embed → upsert to Pinecone
    ├── utils.py               ← embed() with retry
    ├── tools/
    │   ├── rag_tool.py        ← Hybrid search + CrossEncoder rerank
    │   └── web_tool.py        ← Tavily web search
    └── memory/
        ├── short_term.py      ← in-process conversation buffer
        └── vector_memory.py   ← long-term semantic memory (Pinecone)
```

---

## Quick start

### 1. Clone and install

```bash
git clone <your-repo>
cd agentic_rag
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys (see "API keys" section below)
```

### 3. Create Pinecone indexes

Log in to [app.pinecone.io](https://app.pinecone.io) and create **two serverless indexes**:

| Index name      | Dimension | Metric    |
|-----------------|-----------|-----------|
| `agentic-rag`   | 1536      | cosine    |
| `memory-index`  | 1536      | cosine    |

> Dimension 1536 matches `text-embedding-3-small` from OpenRouter.

### 4. Start the API server

```bash
uvicorn main:app --reload --port 8000
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) to explore the interactive API documentation.

### 5. (Optional) Start Celery worker

Required only if you want background ingestion and async query processing:

```bash
# In a second terminal
celery -A celery_app worker --loglevel=info
```

Requires Redis running locally or set `REDIS_URL` in `.env`.

### 6. Open the dashboard

Open `dashboard.html` directly in your browser. The UI works in **Demo mode** out of the box (no API needed). To connect to your backend:

In the dashboard, it auto-reads the `<meta name="api-url">` tag or defaults to `http://localhost:8000`. To change the target server, either:
- Edit the meta tag in `dashboard.html`: `<meta name="api-url" content="https://your-server.com">`
- Or modify `const API` at the top of the script block

---

## API keys required

All three keys are required. The server will exit immediately at startup if any are missing.

### OPENROUTER_API_KEY

**Service**: [OpenRouter](https://openrouter.ai/keys)  
**Used for**: Running `deepseek/deepseek-r1` as the LLM for all three agents, and generating embeddings with `text-embedding-3-small`.  
**Pricing**: Pay-per-use. DeepSeek-R1 is very affordable.  
**To change the model**: Edit `LLM_MODEL` in `backend/config.py`.

```
OPENROUTER_API_KEY=sk-or-v1-...
```

### PINECONE_API_KEY

**Service**: [Pinecone](https://app.pinecone.io)  
**Used for**: Storing document embeddings (`agentic-rag` index) and long-term conversation memory (`memory-index`).  
**Pricing**: Free tier includes 1 project with serverless indexes.

```
PINECONE_API_KEY=pcsk_...
```

### TAVILY_API_KEY

**Service**: [Tavily](https://tavily.com)  
**Used for**: Web search when the Router decides `route: "web"`.  
**Pricing**: Free tier — 1,000 searches per month.

```
TAVILY_API_KEY=tvly-...
```

### Optional

```bash
PINECONE_INDEX=agentic-rag     # default: agentic-rag
MEMORY_INDEX=memory-index       # default: memory-index
REDIS_URL=redis://localhost:6379/0   # default: localhost, needed only for Celery
```

---

## API reference

### POST /query

Ask a question. The pipeline routes, retrieves, and validates automatically.

**Request**
```json
{
  "question": "What are the main risks mentioned in the report?",
  "namespace": ""
}
```

**Response**
```json
{
  "answer": "The report identifies three main risks: ...",
  "grounded": true,
  "iterations": 1,
  "trace": [
    {
      "route": "rag",
      "grounded": true,
      "final_answer": "...",
      "confidence": 0.94,
      "iteration": 1
    }
  ]
}
```

---

### POST /ingest

Ingest a list of text chunks into the vector database.

**Request**
```json
{
  "texts": ["chunk one...", "chunk two..."],
  "namespace": "project-alpha"
}
```

**Response**
```json
{ "status": "ingestion queued", "chunks": 2 }
```

> Ingestion runs in a background task — the response is immediate.

---

### POST /ingest/file

Upload a file directly. Accepts `.txt`, `.md`, `.pdf`.

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@my_document.txt" \
  -F "namespace=my-project"
```

**Response**
```json
{ "status": "ingestion queued", "filename": "my_document.txt", "chunks": 42 }
```

---

### GET /health

```json
{ "status": "ok", "memory_entries": 7 }
```

---

### DELETE /memory

Clears short-term (in-process) conversation memory.

```json
{ "status": "short-term memory cleared" }
```

---

## Ingest your documents (Python)

```python
import requests

# Option 1: upload a file
with open("my_report.txt", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/ingest/file",
        files={"file": f}
    )
print(resp.json())  # {"status": "ingestion queued", "chunks": 38}

# Option 2: ingest raw text chunks
chunks = ["paragraph one...", "paragraph two...", "..."]
resp = requests.post(
    "http://localhost:8000/ingest",
    json={"texts": chunks}
)
print(resp.json())
```

---

## Architecture decisions

### Why three agents?

A single LLM asked to route, retrieve, and validate simultaneously produces lower-quality answers. Separating concerns gives each agent a focused role and a structured output format it is responsible for.

| Agent     | Input                  | Output                              |
|-----------|------------------------|-------------------------------------|
| Router    | User question          | `{"route": "rag"\|"web"\|"memory"}` |
| Retriever | Route + question       | `{"context": "...", "answer": "..."}` |
| Critic    | Context + answer       | `{"grounded": bool, "final_answer": "...", "confidence": float}` |

### Why Hybrid retrieval (Dense + BM25)?

Dense-only search misses exact keyword matches (names, codes, numbers). BM25-only search misses semantic similarity. Hybrid retrieval combines both via Reciprocal Rank Fusion (RRF), then Cross-Encoder reranking selects the top 5 most relevant chunks.

```
Dense search (Pinecone)  ─┐
                           ├─ RRF merge ─ CrossEncoder rerank ─ top-5 chunks
BM25 search (in-memory) ─┘
```

### Why two memory layers?

Short-term memory (`ShortMemory`) is a simple in-process deque — zero latency, perfect for caching repeated questions within a session. Long-term vector memory (`Pinecone`) survives server restarts and supports semantic lookup across sessions.

---

## Changing the LLM

Edit `backend/config.py`:

```python
LLM_MODEL = "deepseek/deepseek-r1"        # current (reasoning model, affordable)
LLM_MODEL = "anthropic/claude-3-5-sonnet" # higher quality, higher cost
LLM_MODEL = "openai/gpt-4o"               # OpenAI via OpenRouter
LLM_MODEL = "meta-llama/llama-3.3-70b"    # open-source option
```

All models are accessed through OpenRouter — same API key, just change the model string.

---

## Known limitations

- **BM25 is in-memory only** — it is rebuilt at each ingestion call using all texts ingested in the current process lifetime. If the server restarts, BM25 is lost (dense search via Pinecone continues to work normally). For persistent BM25, persist `_all_ingested_texts` to disk or a database.

- **Celery + BM25** — if ingestion runs in a Celery worker (separate process), the BM25 index built there is not shared with the FastAPI process. The FastAPI process only rebuilds BM25 when `POST /ingest` is called directly (not via Celery). Use `POST /ingest` directly, or add a post-ingestion signal to trigger a BM25 rebuild in the main process.

- **Memory route** — the Retriever agent receives a `memory` route instruction but the retrieval itself is handled by the LLM using conversation context rather than calling a dedicated tool. For production use, wire `retrieve_memory()` from `vector_memory.py` as a `@tool` on the Retriever agent.

---

## Dependencies

```
fastapi          0.115.0    HTTP server
uvicorn          0.32.0     ASGI runner
crewai           0.80.0     multi-agent framework
crewai-tools     0.14.0     base tool utilities
langchain        0.3.0      LLM abstraction
langchain-openai 0.2.0      OpenAI-compatible LLMs
langchain-community 0.3.0   TavilySearchResults
pinecone         5.4.0      vector database
sentence-transformers 3.3.0 CrossEncoder reranker
rank-bm25        0.2.2      BM25 sparse retrieval
tavily-python    0.5.0      web search client
celery           5.4.0      async task queue
redis            5.1.0      Celery broker/backend
streamlit        1.40.0     alternative UI (optional)
requests         2.32.3     HTTP client
python-dotenv    1.0.1      .env file loading
pydantic         2.9.2      data validation
```

---

## File compatibility matrix

| File                        | Talks to                                     | Status  |
|-----------------------------|----------------------------------------------|---------|
| `dashboard.html`            | `POST /query`, `POST /ingest`, `POST /ingest/file` | ✓ Compatible |
| `main.py`                   | `orchestrator.run()`, `ingestion.ingest()`   | ✓ Compatible |
| `orchestrator.py`           | `agents.py`, `tasks.py`, `memory/`           | ✓ Compatible |
| `agents.py`                 | `tools/rag_tool.py`, `tools/web_tool.py`     | ✓ Compatible |
| `tasks.py`                  | Used by `orchestrator.py`                    | ✓ Compatible |
| `backend/ingestion.py`      | `utils.embed()`, `rag_tool.build_bm25()`     | ✓ Compatible |
| `backend/tools/rag_tool.py` | `utils.embed()`, Pinecone                    | ✓ Compatible |
| `backend/tools/web_tool.py` | Tavily via LangChain                         | ✓ Compatible |
| `backend/memory/short_term.py`   | `orchestrator.py`, `main.py`            | ✓ Compatible |
| `backend/memory/vector_memory.py`| `orchestrator.py`                       | ✓ Compatible |
| `celery_app.py`             | `orchestrator.run()`, `ingestion.ingest()`   | ✓ Compatible |
