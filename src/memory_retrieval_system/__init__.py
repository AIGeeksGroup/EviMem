"""
LaceMem - three-layer conversational memory.
============================================

Three layers:
1. RawMemory      - verbatim dialogue, traceable source of facts
2. MemoryIndex    - atomic (subject, predicate, object, event_time) tuples
3. MemoryIndexEdge - graph links connecting index tuples for multi-hop expansion

This package handles reading/retrieval only; offline ingestion lives elsewhere.

Workflow:
1. Run `precompute_embeddings.py` to populate embeddings on the Index layer.
2. Use `MemoryRetriever` to retrieve facts.
"""

from .models import RawMemory, MemoryIndex, MemoryIndexEdge, Base
from .retriever import MemoryRetriever, RetrievalResult
from .precompute_embeddings import precompute_embeddings, verify_embeddings

__all__ = [
    "RawMemory",
    "MemoryIndex",
    "MemoryIndexEdge",
    "Base",
    "MemoryRetriever",
    "RetrievalResult",
    "precompute_embeddings",
    "verify_embeddings",
]

__version__ = "1.0.0"
