"""
CrewAI Tasks — Router · Retriever · Critic
"""
from crewai import Task


def router_task(agent) -> Task:
    return Task(
        description=(
            "You are a query classifier. Analyse the question below and classify it "
            "into exactly one of three routes. Think step by step before deciding.\n\n"
            "Question: {question}\n\n"
            "Conversation memory:\n{memory}\n\n"
            "Classification rules:\n"
            "- 'rag'    → question is about uploaded documents, internal knowledge base, "
            "domain-specific content, or files the user has provided.\n"
            "- 'web'    → question needs current events, live data, news, prices, "
            "or anything that changes over time.\n"
            "- 'memory' → question explicitly references the current conversation, "
            "asks about something mentioned earlier, or says 'you said', 'earlier', 'before'.\n\n"
            "Classify and provide a brief reason.\n"
            "Respond with ONLY valid JSON:\n"
            '{"route": "rag" | "web" | "memory", "reason": "<one sentence why>"}'
        ),
        expected_output='JSON with keys "route" and "reason".',
        agent=agent,
    )


def retriever_task(agent, router_task_ref) -> Task:
    return Task(
        description=(
            "You are a retriever. Based on the Router's decision, "
            "retrieve the most relevant information and formulate a precise answer.\n\n"
            "Question: {question}\n\n"
            "Conversation memory (use if route=memory):\n{memory}\n\n"
            "Instructions:\n"
            "- route=rag    → call hybrid_rag_retrieve tool with the question.\n"
            "- route=web    → call web_search tool with a focused search query.\n"
            "- route=memory → answer directly from the conversation memory above.\n"
            "- Always capture the raw retrieved text as context.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"context": "<raw retrieved text>", "answer": "<your answer>"}'
        ),
        expected_output='JSON with keys "context" and "answer".',
        context=[router_task_ref],
        agent=agent,
    )


def critic_task(agent, retriever_task_ref) -> Task:
    return Task(
        description=(
            "Review the retriever's answer and verify it is grounded in the context.\n\n"
            "Question: {question}\n\n"
            "Instructions:\n"
            "- Check every factual claim against the provided context.\n"
            "- If all claims are supported → set grounded: true.\n"
            "- If any claim is unsupported or hallucinated → set grounded: false.\n"
            "- Improve or correct the final_answer if needed.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"grounded": true|false, "final_answer": "...", "confidence": 0.0-1.0}'
        ),
        expected_output='JSON with keys "grounded", "final_answer", "confidence".',
        context=[retriever_task_ref],
        agent=agent,
    )
