"""
Vector Memory — Long-term semantic memory via Pinecone.
"""
import uuid
import logging
from typing import Optional

from pinecone import Pinecone
from backend.config import PINECONE_API_KEY, MEMORY_INDEX
from backend.utils import embed

logger = logging.getLogger(__name__)

_pc    = None
_index = None


def _get_index():
    global _pc, _index
    if _index is None:
        _pc    = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(MEMORY_INDEX)
    return _index


def store_memory(text: str, meta: Optional[dict] = None) -> str:
    """Store a piece of text in long-term memory. Returns the generated ID."""
    mem_id = str(uuid.uuid4())
    emb    = embed([text])[0]
    meta   = meta or {}
    meta["text"] = text
    _get_index().upsert([(mem_id, emb, meta)])
    logger.debug(f"Memory stored: {mem_id}")
    return mem_id


def retrieve_memory(query: str, top_k: int = 3) -> list[str]:
    """Retrieve the top-k most semantically similar memories."""
    q_emb = embed([query])[0]
    res   = _get_index().query(
        vector=q_emb, top_k=top_k, include_metadata=True
    )
    return [m["metadata"]["text"] for m in res["matches"]]


def delete_memory(mem_id: str) -> None:
    _get_index().delete(ids=[mem_id])
