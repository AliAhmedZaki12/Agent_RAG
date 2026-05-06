import time
import logging
import requests
from functools import lru_cache
from backend.config import OPENROUTER_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def embed(texts: list[str], retries: int = 3) -> list[list[float]]:
    """
    Embed a list of texts via OpenRouter embeddings API.
    Retries with exponential back-off on transient failures.
    """
    if not texts:
        return []

    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://agentic-rag.app",
        "X-Title":       "Agentic RAG",
    }
    payload = {"model": EMBEDDING_MODEL, "input": texts}

    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "data" not in data:
                raise ValueError(f"Unexpected response shape: {data}")

            return [item["embedding"] for item in data["data"]]

        except requests.HTTPError as e:
            logger.error(f"Embedding HTTP error [{resp.status_code}]: {e}")
            if resp.status_code in (401, 403):
                raise RuntimeError("Invalid OPENROUTER_API_KEY") from e

        except Exception as e:
            logger.warning(f"Embedding attempt {attempt + 1}/{retries} failed: {e}")

        if attempt < retries - 1:
            wait = 2 ** attempt
            logger.info(f"Retrying in {wait}s…")
            time.sleep(wait)

    raise RuntimeError(f"Embedding failed after {retries} attempts")


# ── Fix 2: Embedding cache ───────────────────────────────────────

@lru_cache(maxsize=1000)
def cached_embed(text: str) -> list:
    """
    Single-text embed with LRU cache (up to 1000 entries).
    Prevents re-embedding identical queries across requests.
    Uses embed() internally so retry logic is preserved.
    """
    return embed([text])[0]
