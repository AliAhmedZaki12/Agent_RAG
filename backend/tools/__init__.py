"""
backend.tools — Retrieval and search tools for the Retriever agent.

Modules:
    rag_tool  — Hybrid Dense + BM25 retrieval with CrossEncoder reranking.
                Key exports:
                    hybrid_retrieve()     — main retrieval function
                    grounding_score()     — token-overlap heuristic
                    build_bm25()          — rebuild BM25 after ingest
                    load_bm25_from_disk() — restore BM25 corpus at startup
                    preload_reranker()    — warm up CrossEncoder at startup

    web_tool  — Tavily-powered web search.
                Key exports:
                    web_search()          — search and return formatted results
"""

from backend.tools.rag_tool import (
    hybrid_retrieve,
    grounding_score,
    build_bm25,
    load_bm25_from_disk,
    preload_reranker,
)
from backend.tools.web_tool import web_search

__all__ = [
    "hybrid_retrieve",
    "grounding_score",
    "build_bm25",
    "load_bm25_from_disk",
    "preload_reranker",
    "web_search",
]
