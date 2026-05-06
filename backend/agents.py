"""
CrewAI Agents — Router · Retriever · Critic
"""
from crewai import Agent
from crewai.tools import tool
from langchain_openai import ChatOpenAI

from backend.config import OPENROUTER_API_KEY, LLM_MODEL
from backend.tools.rag_tool import hybrid_retrieve
from backend.tools.web_tool import web_search as _web_search

# ── LLM via OpenRouter ──────────────────────────────────────────
llm = ChatOpenAI(
    model       = LLM_MODEL,
    api_key     = OPENROUTER_API_KEY,
    base_url    = "https://openrouter.ai/api/v1",
    temperature = 0,
    default_headers = {
        "HTTP-Referer": "https://agentic-rag.app",
        "X-Title":      "Agentic RAG",
    },
)


# ── Tool wrappers (required by CrewAI) ─────────────────────────

@tool("hybrid_rag_retrieve")
def rag_tool(query: str) -> str:
    """
    Retrieve the most relevant document chunks from the vector database
    using hybrid dense + sparse retrieval and cross-encoder reranking.
    Use this when the question relates to ingested documents.
    """
    chunks = hybrid_retrieve(query)
    if not chunks:
        return "No relevant documents found."
    return "\n---\n".join(chunks)


@tool("web_search")
def web_tool(query: str) -> str:
    """
    Search the web for current or general information.
    Use this when the question requires up-to-date data or is outside
    the scope of the document knowledge base.
    """
    return _web_search(query)


# ── Agent factories ─────────────────────────────────────────────

def router_agent() -> Agent:
    return Agent(
        role  = "Router",
        goal  = (
            "Decide the best information source for the user's question. "
            "Return exactly one of: 'rag', 'web', or 'memory'."
        ),
        backstory = (
            "You are a routing expert. You analyse user questions and decide "
            "whether to answer from the local knowledge base (rag), "
            "the internet (web), or recent conversation history (memory)."
        ),
        llm     = llm,
        verbose = False,
    )


def retriever_agent() -> Agent:
    return Agent(
        role  = "Retriever",
        goal  = "Retrieve the most relevant information and formulate a clear answer.",
        backstory = (
            "You are a skilled researcher. Given a route decision, you use the "
            "appropriate tool to retrieve information and synthesise a precise, "
            "grounded answer in English or the user's language."
        ),
        tools   = [rag_tool, web_tool],
        llm     = llm,
        verbose = True,
    )


def critic_agent() -> Agent:
    return Agent(
        role  = "Critic",
        goal  = "Verify that the answer is grounded in retrieved evidence.",
        backstory = (
            "You are a rigorous fact-checker. You examine the retriever's answer "
            "and check that every claim is supported by the provided context. "
            "You flag hallucinations and mark answers as grounded or not."
        ),
        llm     = llm,
        verbose = False,
    )
