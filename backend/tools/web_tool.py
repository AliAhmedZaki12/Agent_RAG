"""
Web Search Tool — Tavily-powered, with graceful fallback.
"""
import os
import logging

from backend.config import TAVILY_API_KEY

# LangChain reads TAVILY_API_KEY from env — set it explicitly
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)

_search = TavilySearchResults(k=5)


def web_search(query: str) -> str:
    """
    Search the web and return a formatted string of results.
    Returns an error message string on failure (never raises).
    """
    try:
        results = _search.run(query)

        if isinstance(results, list):
            formatted = []
            for i, r in enumerate(results, 1):
                url     = r.get("url", "")
                content = r.get("content", "").strip()
                formatted.append(f"[{i}] {url}\n{content}")
            return "\n\n".join(formatted)

        return str(results)

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"[web_search] Failed: {e}"
