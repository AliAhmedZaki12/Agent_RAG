"""
backend.memory — Two-layer memory system.

Modules:
    short_term    — In-process conversation buffer (ShortMemory).
                    Fast O(1) exact-match lookup + FIFO deque (maxlen=20).
                    Key exports:
                        ShortMemory          — class
                        ShortMemory.add()    — store Q/A pair
                        ShortMemory.find()   — exact-match cache hit
                        ShortMemory.format_for_prompt() — inject into crew

    vector_memory — Long-term semantic memory backed by Pinecone.
                    Survives restarts. Semantic similarity lookup.
                    Key exports:
                        store_memory()       — embed and upsert text
                        retrieve_memory()    — semantic top-k lookup
                        delete_memory()      — remove by ID

Memory flow:
    User question
        │
        ▼
    ShortMemory.find()   ← exact match? → return immediately (cache hit)
        │ miss
        ▼
    vector_memory.retrieve_memory()  ← semantic match from past sessions
        │
        ▼
    inject into crew as {memory} context
        │
        ▼
    After grounded answer:
        ShortMemory.add()    ← fast in-session cache
        store_memory()       ← persist to Pinecone for future sessions
"""

from backend.memory.short_term    import ShortMemory
from backend.memory.vector_memory import store_memory, retrieve_memory, delete_memory

__all__ = [
    "ShortMemory",
    "store_memory",
    "retrieve_memory",
    "delete_memory",
]
