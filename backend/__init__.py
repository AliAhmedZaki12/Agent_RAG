"""
backend — Agentic RAG core package.

Sub-packages:
    tools/   — RAG retrieval tool + web search tool
    memory/  — Short-term buffer + long-term vector memory

Top-level modules:
    config       — Environment variable validation
    agents       — CrewAI agent factories (Router, Retriever, Critic)
    tasks        — CrewAI task definitions
    orchestrator — Pipeline controller (run() entry point)
    ingestion    — Chunk → embed → upsert to Pinecone
    utils        — embed() + cached_embed()
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("agentic-rag")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = [
    "config",
    "agents",
    "tasks",
    "orchestrator",
    "ingestion",
    "utils",
]
