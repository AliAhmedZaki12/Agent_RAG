import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        sys.exit(f"[config] ❌ Missing required env var: {key}\n"
                 f"  → Add it to your .env file or export it before running.")
    return value


# ── Required keys ──────────────────────────────────────────────
OPENROUTER_API_KEY = _require("OPENROUTER_API_KEY")
PINECONE_API_KEY   = _require("PINECONE_API_KEY")
TAVILY_API_KEY     = _require("TAVILY_API_KEY")

# ── Optional with defaults ──────────────────────────────────────
PINECONE_INDEX     = os.getenv("PINECONE_INDEX",  "agentic-rag")
MEMORY_INDEX       = os.getenv("MEMORY_INDEX",    "memory-index")
REDIS_URL          = os.getenv("REDIS_URL",        "redis://localhost:6379/0")

# ── Model config ────────────────────────────────────────────────
EMBEDDING_MODEL    = "text-embedding-3-small"
LLM_MODEL          = "deepseek/deepseek-r1"        # ← بدون "openrouter/" prefix
RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
