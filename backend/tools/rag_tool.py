"""
RAG Tool — Hybrid Dense + BM25 Retrieval + Cross-Encoder Reranking

Fixes applied:
  Fix 1 — BM25 corpus persisted to disk (bm25_corpus.json) — survives restarts
  Fix 2 — Embedding calls cached via cached_embed() from utils
  Fix 3 — RRF ID collision resolved: dense IDs "dense_*", sparse IDs "sparse_*"
  Fix 4 — grounding_score() heuristic added (token overlap)
  Fix 5 — preload_reranker() for startup warmup, zero first-request spike
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pinecone import Pinecone
from backend.config import PINECONE_API_KEY, PINECONE_INDEX, RERANKER_MODEL
from backend.utils import cached_embed

logger = logging.getLogger(__name__)

# ── Disk path for BM25 corpus persistence ────────────────────────
BM25_CORPUS_PATH = Path("bm25_corpus.json")

# ── Singletons ───────────────────────────────────────────────────
_pc:      Optional[Pinecone] = None
_index                        = None
_reranker                     = None

# ── BM25 state ──────────────────────────────────────────────────
_bm25_corpus: list[str] = []
_bm25                    = None


# ════════════════════════════════════════════════════════════════
# Pinecone
# ════════════════════════════════════════════════════════════════

def _get_index():
    global _pc, _index
    if _index is None:
        _pc    = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX)
        logger.info(f"Pinecone index '{PINECONE_INDEX}' connected.")
    return _index


# ════════════════════════════════════════════════════════════════
# Reranker — Fix 5: preloaded at startup
# ════════════════════════════════════════════════════════════════

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker ready.")
    return _reranker


def preload_reranker() -> None:
    """
    Call once at application startup (lifespan handler in main.py).
    Loads the CrossEncoder model into memory so the first real
    request has zero loading latency.
    """
    _get_reranker()


# ════════════════════════════════════════════════════════════════
# BM25 — Fix 1: persistent corpus
# ════════════════════════════════════════════════════════════════

def _save_corpus(texts: list[str]) -> None:
    """Persist BM25 corpus to disk after every ingestion."""
    try:
        BM25_CORPUS_PATH.write_text(
            json.dumps(texts, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug(f"BM25 corpus saved ({len(texts)} texts) → {BM25_CORPUS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save BM25 corpus: {e}")


def _load_corpus() -> list[str]:
    """Load BM25 corpus from disk if it exists."""
    if not BM25_CORPUS_PATH.exists():
        logger.info("No BM25 corpus file found — starting fresh.")
        return []
    try:
        texts = json.loads(BM25_CORPUS_PATH.read_text(encoding="utf-8"))
        logger.info(f"BM25 corpus loaded from disk ({len(texts)} texts).")
        return texts
    except Exception as e:
        logger.error(f"Failed to load BM25 corpus: {e}")
        return []


def _build_bm25_index(texts: list[str]) -> None:
    global _bm25_corpus, _bm25
    if not texts:
        return
    try:
        from rank_bm25 import BM25Okapi
        _bm25_corpus = texts
        _bm25        = BM25Okapi([t.split() for t in texts])
        logger.info(f"BM25 index built on {len(texts)} documents.")
    except ImportError:
        logger.warning("rank-bm25 not installed — BM25 disabled.")


def build_bm25(texts: list[str]) -> None:
    """Build BM25 index and persist corpus to disk. Called after ingest."""
    _build_bm25_index(texts)
    _save_corpus(texts)


def load_bm25_from_disk() -> None:
    """
    Restore BM25 from last persisted corpus.
    Call once at app startup so BM25 survives server restarts.
    """
    texts = _load_corpus()
    if texts:
        _build_bm25_index(texts)


# ════════════════════════════════════════════════════════════════
# Grounding heuristic — Fix 4
# ════════════════════════════════════════════════════════════════

def grounding_score(answer: str, context: str) -> float:
    """
    Token-overlap ratio: what fraction of answer tokens appear in context.
    Returns 0.0–1.0. Used alongside the LLM Critic's verdict to add
    a non-LLM signal for hallucination detection.
    """
    if not answer or not context:
        return 0.0
    a_tokens = set(answer.lower().split())
    c_tokens = set(context.lower().split())
    overlap  = len(a_tokens & c_tokens)
    return round(overlap / max(len(a_tokens), 1), 3)


# ════════════════════════════════════════════════════════════════
# Search — Fix 2 (cached embed) + Fix 3 (prefixed IDs)
# ════════════════════════════════════════════════════════════════

def dense_search(query: str, top_k: int = 10, namespace: str = "") -> list[dict]:
    """Dense vector search via Pinecone. Uses cached embeddings."""
    q_emb  = cached_embed(query)          # Fix 2: no re-embed same query
    kwargs: dict = dict(vector=q_emb, top_k=top_k, include_metadata=True)
    if namespace:
        kwargs["namespace"] = namespace
    res = _get_index().query(**kwargs)
    return [
        {
            "id":    f"dense_{m['id']}",   # Fix 3: prefix prevents RRF collision
            "text":  m["metadata"].get("text", ""),
            "score": m["score"],
        }
        for m in res["matches"]
    ]


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 sparse search over in-memory (disk-restored) corpus."""
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(query.split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "id":    f"sparse_{i}",         # Fix 3: prefix prevents RRF collision
            "text":  _bm25_corpus[i],
            "score": float(scores[i]),
        }
        for i in ranked
    ]


# ════════════════════════════════════════════════════════════════
# Hybrid retrieve — dynamic top_k + grounding score
# ════════════════════════════════════════════════════════════════

def hybrid_retrieve(
    query:     str,
    alpha:     float = 0.7,
    top_k:     int   = 0,          # 0 = auto (dynamic)
    namespace: str   = "",
) -> tuple[list[str], float]:
    """
    Reciprocal Rank Fusion of dense + BM25,
    then Cross-Encoder reranking.

    Args:
        query:     User question.
        alpha:     Dense weight in RRF (0–1). Default 0.7.
        top_k:     Candidates per source. 0 = dynamic by query length.
        namespace: Pinecone namespace filter.

    Returns:
        (top-5 text chunks, retrieval grounding score 0.0–1.0)
    """
    # Dynamic top_k: longer queries benefit from a wider retrieval net
    if top_k == 0:
        word_count = len(query.split())
        top_k      = 20 if word_count > 20 else 10

    dense  = dense_search(query, top_k=top_k, namespace=namespace)
    sparse = bm25_search(query,  top_k=top_k)

    # ── Reciprocal Rank Fusion ──────────────────────────────────
    rrf_scores: dict[str, float] = {}
    for i, d in enumerate(dense):
        rrf_scores[d["id"]] = rrf_scores.get(d["id"], 0) + alpha * (1 / (i + 1))
    for i, s in enumerate(sparse):
        rrf_scores[s["id"]] = rrf_scores.get(s["id"], 0) + (1 - alpha) * (1 / (i + 1))

    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]  # type: ignore
    id2text    = {x["id"]: x["text"] for x in dense + sparse}
    candidates = [id2text[rid] for rid in ranked_ids if rid in id2text]

    if not candidates:
        return [], 0.0

    # ── Cross-Encoder rerank ────────────────────────────────────
    reranker  = _get_reranker()
    pairs     = [(query, c) for c in candidates]
    rr_scores = reranker.predict(pairs)
    ranked    = [t for _, t in sorted(zip(rr_scores, candidates), reverse=True)]
    top_chunks = ranked[:5]

    # ── Grounding heuristic (Fix 4) ─────────────────────────────
    full_context = " ".join(top_chunks)
    g_score      = grounding_score(query, full_context)

    return top_chunks, g_score
