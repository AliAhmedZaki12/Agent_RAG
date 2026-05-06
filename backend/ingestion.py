"""
Ingestion Pipeline — Chunk → Embed → Upsert to Pinecone.
"""
from __future__ import annotations

import uuid
import logging
from pathlib import Path
from typing import Union

from pinecone import Pinecone

from backend.config import PINECONE_API_KEY, PINECONE_INDEX
from backend.utils  import embed
from backend.tools.rag_tool import build_bm25

logger      = logging.getLogger(__name__)
BATCH_SIZE  = 100
_all_ingested_texts: list[str] = []  # cumulative for BM25   # Pinecone upsert limit


_pc_inst    = None
_index_inst = None

def _get_index():
    global _pc_inst, _index_inst
    if _index_inst is None:
        _pc_inst    = Pinecone(api_key=PINECONE_API_KEY)
        _index_inst = _pc_inst.Index(PINECONE_INDEX)
    return _index_inst


# ── Text chunking ────────────────────────────────────────────────

def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Sliding-window character chunker with overlap."""
    words  = text.split()
    chunks = []
    step   = size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_file(path: Union[str, Path], size: int = 512, overlap: int = 64) -> list[str]:
    """Read a text file and chunk it."""
    text = Path(path).read_text(encoding="utf-8")
    return chunk_text(text, size=size, overlap=overlap)


# ── Upsert ───────────────────────────────────────────────────────

def ingest(chunks: list[str], namespace: str = "") -> int:
    """
    Embed and upsert chunks into Pinecone.

    Args:
        chunks:    List of text strings to ingest.
        namespace: Pinecone namespace (optional grouping).

    Returns:
        Number of vectors upserted.
    """
    if not chunks:
        logger.warning("ingest() called with empty chunks list.")
        return 0

    index    = _get_index()
    total    = 0
    all_texts = []

    for start in range(0, len(chunks), BATCH_SIZE):
        batch      = chunks[start : start + BATCH_SIZE]
        embeddings = embed(batch)

        vectors = [
            (str(uuid.uuid4()), emb, {"text": chunk})
            for emb, chunk in zip(embeddings, batch)
        ]

        kwargs = {"vectors": vectors}
        if namespace:
            kwargs["namespace"] = namespace

        index.upsert(**kwargs)
        total     += len(vectors)
        all_texts += batch
        logger.info(f"Upserted batch {start // BATCH_SIZE + 1}: {len(vectors)} vectors")

    # Rebuild BM25 with ALL texts ever ingested (not just this batch)
    _all_ingested_texts.extend(all_texts)
    build_bm25(_all_ingested_texts)
    logger.info(f"✅ Ingestion complete: {total} vectors.")
    return total


def ingest_file(path: Union[str, Path], **kwargs) -> int:
    chunks = chunk_file(path)
    logger.info(f"Ingesting '{path}' → {len(chunks)} chunks")
    return ingest(chunks, **kwargs)
