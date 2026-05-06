"""
Orchestrator — The brain of the Agentic RAG pipeline.
Builds the Crew once per question and retries with escalating search strategy.
"""
from __future__ import annotations

import json
import re
import logging
import time
from typing import Any

from crewai import Crew, Process

from backend.agents import router_agent, retriever_agent, critic_agent
from backend.tasks  import router_task, retriever_task, critic_task
from backend.memory.short_term    import ShortMemory
from backend.memory.vector_memory import store_memory
from backend.tools.rag_tool       import grounding_score, load_bm25_from_disk, preload_reranker

logger     = logging.getLogger(__name__)
MAX_ITERS  = 3
short_mem  = ShortMemory(maxlen=20)   # Shared across requests


# ── JSON parser ─────────────────────────────────────────────────

def _parse(raw: str) -> dict[str, Any]:
    """
    Extract the LAST JSON object from raw output.
    Uses greedy match to handle nested JSON correctly.
    Falls back to json_repair if available.
    """
    # Greedy match — captures full nested JSON
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        logger.warning(f"No JSON found in output:\n{raw[:300]}")
        return {}
    candidate = match.group()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Try json_repair if installed
    try:
        from json_repair import repair_json
        return json.loads(repair_json(candidate))
    except Exception:
        pass
    logger.error(f"JSON parse failed. Raw snippet: {raw[:300]}")
    return {}


def _extract_route(text: str) -> str:
    """Extract the route decision (rag/web/memory) from the crew's raw output."""
    m = re.search(r'"route"\s*:\s*"(rag|web|memory)"', text, re.IGNORECASE)
    return m.group(1) if m else "rag"


# ── Crew builder ─────────────────────────────────────────────────

def _build_crew():
    """Build agents + tasks + crew. Call once per request."""
    r_agent   = router_agent()
    ret_agent = retriever_agent()
    c_agent   = critic_agent()

    rt    = router_task(r_agent)
    ret_t = retriever_task(ret_agent, rt)     # rt passed as router_task_ref
    ct    = critic_task(c_agent, ret_t)       # ret_t passed as retriever_task_ref

    crew = Crew(
        agents  = [r_agent, ret_agent, c_agent],
        tasks   = [rt, ret_t, ct],
        process = Process.sequential,
        verbose = True,
    )
    return crew


# ── Main entry point ─────────────────────────────────────────────

def run(question: str) -> tuple[str, list[dict]]:
    """
    Run the Agentic RAG pipeline for a question.

    Returns:
        (final_answer, trace)
    """
    t0 = time.perf_counter()

    # Short-circuit: exact match in short-term memory
    cached = short_mem.find(question)
    if cached:
        logger.info("Cache hit — returning from short-term memory.")
        return cached, [{"grounded": True, "final_answer": cached, "source": "cache"}]

    crew  = _build_crew()
    trace = []
    q     = question

    for iteration in range(MAX_ITERS):
        logger.info(f"Iteration {iteration + 1}/{MAX_ITERS} — q: {q[:80]}")

        # Inject short-term memory into every crew run
        mem_context = short_mem.format_for_prompt(n=5)

        try:
            raw     = crew.kickoff(inputs={"question": q, "memory": mem_context})
            raw_str = str(raw)
            parsed  = _parse(raw_str)
            parsed["route"]  = _extract_route(raw_str)
            parsed["reason"] = parsed.get("reason", "")
        except Exception as e:
            logger.error(f"Crew kickoff error: {e}")
            parsed = {"route": "rag", "reason": ""}

        parsed["iteration"] = iteration + 1
        trace.append(parsed)

        if parsed.get("grounded"):
            answer   = parsed.get("final_answer", "")
            conf     = parsed.get("confidence", 1.0)
            context  = parsed.get("context", answer)  # from retriever

            # Fix 4: heuristic cross-check alongside LLM verdict
            g_score  = grounding_score(answer, context)
            parsed["grounding_heuristic"] = g_score
            logger.info(f"✅ Grounded (LLM conf={conf}, heuristic={g_score}) "
                        f"in {time.perf_counter()-t0:.2f}s")

            # Persist to memories
            short_mem.add(question, answer)
            store_memory(f"Q: {question}\nA: {answer}")

            return answer, trace

        # Smart escalation strategy
        conf    = parsed.get("confidence", 0)
        context = parsed.get("context", "")
        if iteration == 0:
            if len(context.split()) < 30:
                # Low context → broaden retrieval
                q = question + " — retrieve more documents, search broadly."
            else:
                # Context exists but unverified → add web cross-check
                q = question + " — cross-verify using web search."
        elif iteration == 1:
            q = question + " — use web search only, ignore the local knowledge base."

    elapsed = time.perf_counter() - t0
    logger.warning(f"❌ No grounded answer after {MAX_ITERS} iterations ({elapsed:.2f}s)")
    return (
        "I was unable to find a fully verified answer. "
        "Please rephrase your question or provide more context.",
        trace,
    )
