"""
Three-tier memory retrieval manager.

Retrieval pipeline:
1. Initial retrieval on the MemoryIndex table (embedding/keyword)
2. Multi-hop expansion on the MemoryIndexEdge table
3. Raw-text lookup on the RawMemory table
"""

import json
import math
import re
import string
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any, Callable

from sqlalchemy import create_engine, select, or_, and_, func
from sqlalchemy.orm import sessionmaker, Session

from .models import RawMemory, MemoryIndex, MemoryIndexEdge


@dataclass
class RetrievalResult:
    """Retrieval result wrapper."""
    query: str

    # Retrieved indexes
    indexes: List[MemoryIndex] = field(default_factory=list)
    index_scores: Dict[str, float] = field(default_factory=dict)  # id -> score

    # Multi-hop expanded indexes
    expanded_indexes: List[MemoryIndex] = field(default_factory=list)

    # Raw-text lookups
    raw_memories: List[RawMemory] = field(default_factory=list)

    # Meta info
    search_method: str = ""
    hops_used: int = 0

    @property
    def total_count(self) -> int:
        return len(self.indexes) + len(self.expanded_indexes)

    @property
    def all_indexes(self) -> List[MemoryIndex]:
        """Return all indexes (initial + expanded)."""
        return self.indexes + self.expanded_indexes

    def get_facts(self) -> List[str]:
        """Return all facts as a list of strings."""
        facts = []
        for idx in self.all_indexes:
            facts.append(idx.to_fact_string())
        return facts

    def get_raw_texts(self) -> List[str]:
        """Return all raw texts."""
        return [r.raw_text for r in self.raw_memories]


class MemoryRetriever:
    """
    Three-tier memory retrieval manager.

    Supports:
    - Embedding similarity retrieval
    - Keyword retrieval
    - BM25 retrieval
    - Multi-hop graph expansion (edge)
    - Raw-text lookup
    """

    def __init__(self, db_path: str, echo: bool = False, embedding_provider: str = "openai"):
        """
        Args:
            db_path: database path, e.g. "sqlite:///memory.db"
            echo: whether to print SQL statements
            embedding_provider: embedding provider ('openai' or 'google_ai')
        """
        self.db_path = db_path
        self.engine = create_engine(db_path, echo=echo)
        self.Session = sessionmaker(bind=self.engine)
        self.embedding_provider = embedding_provider

        # Cache (optional optimization)
        self._all_indexes_cache: Optional[List[MemoryIndex]] = None
        self._embeddings_cache: Optional[Dict[str, List[float]]] = None
        self._openai_client = None
        self._gemini_client = None

    def get_query_embedding(
        self,
        query: str,
        model: str = "text-embedding-3-small"
    ) -> Optional[List[float]]:
        """
        Generate a query embedding via OpenAI or Gemini API.

        Args:
            query: query text
            model: embedding model

        Returns:
            embedding vector, or None on failure
        """
        import os
        try:
            if self.embedding_provider == "openai":
                if self._openai_client is None:
                    import openai
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        print("[Retriever] OPENAI_API_KEY not set")
                        return None
                    self._openai_client = openai.OpenAI(api_key=api_key)

                response = self._openai_client.embeddings.create(
                    input=query,
                    model=model
                )
                return response.data[0].embedding

            elif self.embedding_provider == "google_ai":
                if self._gemini_client is None:
                    import google.generativeai as genai
                    api_key = os.environ.get("GEMINI_API_KEY")
                    if not api_key:
                        print("[Retriever] GEMINI_API_KEY not set")
                        return None
                    genai.configure(api_key=api_key)
                    self._gemini_client = genai

                # Gemini embedding API (retrieval_query is used for query text)
                result = self._gemini_client.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query"
                )
                return result['embedding']

            else:
                print(f"[Retriever] Unsupported embedding provider: {self.embedding_provider}")
                return None

        except Exception as e:
            print(f"[Retriever] Failed to get embedding: {e}")
            return None

    def has_precomputed_embeddings(self) -> bool:
        """Check whether the database has precomputed embeddings."""
        from sqlalchemy import text as sql_text
        try:
            with self.Session() as session:
                result = session.execute(sql_text(
                    "SELECT COUNT(*) FROM memory_index WHERE embedding IS NOT NULL"
                )).scalar()
                return result and result > 0
        except Exception:
            return False

    # ========================================================================
    # Step 1: initial retrieval over the MemoryIndex table
    # ========================================================================

    def search_by_embedding(
        self,
        query_embedding: List[float],
        embedding_func: Optional[Callable[[str], List[float]]] = None,
        limit: int = 10,
        threshold: float = 0.5,
        use_precomputed: bool = True,
    ) -> List[Tuple[MemoryIndex, float]]:
        """
        Embedding similarity retrieval.

        Prefers precomputed embeddings (the database `embedding` column);
        falls back to dynamic generation (requires `embedding_func`) if none.

        Args:
            query_embedding: query vector
            embedding_func: function to dynamically generate index embeddings (optional fallback)
            limit: number of results to return
            threshold: similarity threshold
            use_precomputed: whether to prefer precomputed embeddings

        Returns:
            [(MemoryIndex, score), ...] sorted by similarity (desc)
        """
        with self.Session() as session:
            # Try to read precomputed embeddings from the database
            if use_precomputed:
                # Pick the embedding column based on provider
                embedding_col = "gemini_embedding" if self.embedding_provider == "google_ai" else "embedding"

                # Check whether the database has an embedding column with data
                from sqlalchemy import text as sql_text
                try:
                    check_result = session.execute(sql_text(
                        f"SELECT COUNT(*) FROM memory_index WHERE {embedding_col} IS NOT NULL"
                    )).scalar()
                    has_precomputed = check_result and check_result > 0
                except Exception:
                    has_precomputed = False

                if has_precomputed:
                    # Pick the embedding column to read based on provider
                    embedding_col = "gemini_embedding" if self.embedding_provider == "google_ai" else "embedding"

                    # Use precomputed embeddings
                    result = session.execute(sql_text(
                        f"SELECT id, subject, predicate, object, {embedding_col} FROM memory_index WHERE {embedding_col} IS NOT NULL"
                    ))
                    rows = result.fetchall()

                    scored = []
                    for row in rows:
                        idx_id, subject, predicate, obj, embedding_json = row
                        try:
                            idx_embedding = json.loads(embedding_json)
                            score = self._cosine_similarity(query_embedding, idx_embedding)
                            if score >= threshold:
                                # Get the full MemoryIndex object
                                idx = session.execute(
                                    select(MemoryIndex).where(MemoryIndex.id == idx_id)
                                ).scalar_one_or_none()
                                if idx:
                                    scored.append((idx, score))
                        except (json.JSONDecodeError, TypeError):
                            continue

                    scored.sort(key=lambda x: x[1], reverse=True)
                    return scored[:limit]
                else:
                    print(f"[Retriever] No precomputed {self.embedding_provider} embeddings found in database")

            # Fallback: dynamically generate embeddings
            if embedding_func is None:
                print("[Retriever] No embedding_func provided and no precomputed embeddings, cannot do vector search")
                return []

            print("[Retriever] Using dynamic embedding generation (slow)")
            stmt = select(MemoryIndex).limit(500)  # cap to avoid being too slow
            all_indexes = list(session.execute(stmt).scalars().all())

            if not all_indexes:
                return []

            scored = []
            for idx in all_indexes:
                try:
                    # Dynamically generate embedding
                    text = f"{idx.subject} {idx.predicate} {idx.object or ''}"
                    idx_embedding = embedding_func(text)
                    score = self._cosine_similarity(query_embedding, idx_embedding)
                    if score >= threshold:
                        scored.append((idx, score))
                except Exception as e:
                    continue

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]

    def search_by_keyword(
        self,
        query: str,
        search_fields: List[str] = None,
        limit: int = 10,
    ) -> List[MemoryIndex]:
        """
        Keyword retrieval.

        Args:
            query: search keywords
            search_fields: fields to search, default ["subject", "predicate", "object"]
            limit: number of results to return
        """
        if search_fields is None:
            search_fields = ["subject", "predicate", "object"]

        with self.Session() as session:
            conditions = []
            query_lower = query.lower()
            query_terms = self._tokenize(query_lower)

            for field in search_fields:
                col = getattr(MemoryIndex, field, None)
                if col is not None:
                    for term in query_terms:
                        if len(term) > 1:  # skip single characters
                            conditions.append(func.lower(col).contains(term))

            if not conditions:
                return []

            stmt = select(MemoryIndex).where(or_(*conditions)).limit(limit)
            return list(session.execute(stmt).scalars().all())

    def search_by_bm25(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Tuple[MemoryIndex, float]]:
        """
        BM25 retrieval (requires the rank_bm25 library).

        Returns:
            [(MemoryIndex, bm25_score), ...]
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[Retriever] rank_bm25 not installed, falling back to keyword search")
            return [(idx, 1.0) for idx in self.search_by_keyword(query, limit=limit)]

        with self.Session() as session:
            stmt = select(MemoryIndex)
            all_indexes = list(session.execute(stmt).scalars().all())

            if not all_indexes:
                return []

            # Build documents
            documents = []
            valid_indexes = []
            for idx in all_indexes:
                text = f"{idx.subject} {idx.predicate} {idx.object or ''}"
                tokens = self._tokenize(text.lower())
                if tokens:
                    documents.append(tokens)
                    valid_indexes.append(idx)

            if not documents:
                return []

            # BM25 retrieval
            bm25 = BM25Okapi(documents)
            query_tokens = self._tokenize(query.lower())

            if not query_tokens:
                return []

            scores = bm25.get_scores(query_tokens)
            scored = list(zip(valid_indexes, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            return [(idx, score) for idx, score in scored[:limit] if score > 0]

    def search_by_subject(
        self,
        subject: str,
        exact_match: bool = False,
        limit: int = 20,
    ) -> List[MemoryIndex]:
        """Retrieve by subject (entity aggregation)."""
        with self.Session() as session:
            if exact_match:
                stmt = select(MemoryIndex).where(
                    func.lower(MemoryIndex.subject) == subject.lower()
                )
            else:
                stmt = select(MemoryIndex).where(
                    func.lower(MemoryIndex.subject).contains(subject.lower())
                )
            return list(session.execute(stmt.limit(limit)).scalars().all())

    def search_by_time(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 20,
    ) -> List[MemoryIndex]:
        """Retrieve by event time range."""
        with self.Session() as session:
            conditions = [MemoryIndex.event_time.isnot(None)]

            if start_time:
                conditions.append(MemoryIndex.event_time >= start_time)
            if end_time:
                conditions.append(MemoryIndex.event_time <= end_time)

            stmt = (
                select(MemoryIndex)
                .where(and_(*conditions))
                .order_by(MemoryIndex.event_time.desc())
                .limit(limit)
            )
            return list(session.execute(stmt).scalars().all())

    # ========================================================================
    # Step 2: multi-hop expansion over the MemoryIndexEdge table
    # ========================================================================

    def expand_by_edges(
        self,
        index_ids: List[str],
        hops: int = 1,
        edge_types: Optional[List[str]] = None,
        min_weight: float = 0.5,
    ) -> List[MemoryIndex]:
        """
        Multi-hop expansion using edges.

        Args:
            index_ids: starting node IDs
            hops: number of expansion hops (1-2)
            edge_types: restrict to edge types ["same_raw", "semantic_similarity"]
            min_weight: minimum weight threshold

        Returns:
            Expanded MemoryIndex list (excluding the starting nodes)
        """
        if hops <= 0 or not index_ids:
            return []

        with self.Session() as session:
            visited: Set[str] = set(index_ids)
            current_layer = set(index_ids)
            all_expanded = []

            for _ in range(hops):
                if not current_layer:
                    break

                # Find connected edges
                conditions = [
                    or_(
                        MemoryIndexEdge.src_id.in_(current_layer),
                        MemoryIndexEdge.dst_id.in_(current_layer),
                    ),
                    MemoryIndexEdge.weight >= min_weight,
                ]

                if edge_types:
                    conditions.append(MemoryIndexEdge.edge_type.in_(edge_types))

                edges = session.execute(
                    select(MemoryIndexEdge).where(and_(*conditions))
                ).scalars().all()

                # Collect next-layer nodes
                next_layer = set()
                for edge in edges:
                    if edge.src_id not in visited:
                        next_layer.add(edge.src_id)
                    if edge.dst_id not in visited:
                        next_layer.add(edge.dst_id)

                # Fetch the next-layer MemoryIndex objects
                if next_layer:
                    expanded = session.execute(
                        select(MemoryIndex).where(MemoryIndex.id.in_(next_layer))
                    ).scalars().all()
                    all_expanded.extend(expanded)
                    visited.update(next_layer)

                current_layer = next_layer

            return all_expanded

    def get_neighbors(
        self,
        index_id: str,
        edge_type: Optional[str] = None,
    ) -> List[Tuple[MemoryIndex, float, str]]:
        """
        Get directly connected neighbor nodes.

        Returns:
            [(MemoryIndex, weight, edge_type), ...]
        """
        with self.Session() as session:
            conditions = [
                or_(
                    MemoryIndexEdge.src_id == index_id,
                    MemoryIndexEdge.dst_id == index_id,
                )
            ]
            if edge_type:
                conditions.append(MemoryIndexEdge.edge_type == edge_type)

            edges = session.execute(
                select(MemoryIndexEdge).where(and_(*conditions))
            ).scalars().all()

            neighbor_info = {}
            for edge in edges:
                neighbor_id = edge.dst_id if edge.src_id == index_id else edge.src_id
                if neighbor_id not in neighbor_info:
                    neighbor_info[neighbor_id] = (edge.weight, edge.edge_type)

            if not neighbor_info:
                return []

            neighbors = session.execute(
                select(MemoryIndex).where(MemoryIndex.id.in_(neighbor_info.keys()))
            ).scalars().all()

            return [
                (n, neighbor_info[n.id][0], neighbor_info[n.id][1])
                for n in neighbors
            ]

    # ========================================================================
    # Step 3: raw-text lookup over the RawMemory table
    # ========================================================================

    def get_raw_memories(self, raw_ids: List[str]) -> List[RawMemory]:
        """Fetch raw memories in batch."""
        if not raw_ids:
            return []

        with self.Session() as session:
            stmt = select(RawMemory).where(RawMemory.id.in_(raw_ids))
            return list(session.execute(stmt).scalars().all())

    def get_raw_for_index(self, index: MemoryIndex) -> Optional[RawMemory]:
        """Look up raw text for a single index."""
        with self.Session() as session:
            return session.execute(
                select(RawMemory).where(RawMemory.id == index.raw_id)
            ).scalar_one_or_none()

    # ========================================================================
    # Raw BM25 retrieval (used for ablation experiments)
    # ========================================================================

    def search_raw_bm25(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Tuple[RawMemory, float]]:
        """
        BM25 retrieval directly on the RawMemory table (ablation: skip Index and Edge).

        Returns:
            [(RawMemory, bm25_score), ...]
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[Retriever] rank_bm25 not installed")
            return []

        with self.Session() as session:
            all_raws = list(session.execute(select(RawMemory)).scalars().all())

        if not all_raws:
            return []

        documents = []
        valid_raws = []
        for raw in all_raws:
            tokens = self._tokenize(raw.raw_text.lower())
            if tokens:
                documents.append(tokens)
                valid_raws.append(raw)

        if not documents:
            return []

        bm25 = BM25Okapi(documents)
        query_tokens = self._tokenize(query.lower())
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)
        scored = sorted(zip(valid_raws, scores), key=lambda x: x[1], reverse=True)
        return [(raw, score) for raw, score in scored[:limit] if score > 0]

    # ========================================================================
    # Full retrieval pipeline
    # ========================================================================

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
        top_k: int = 10,
        hops: int = 1,
        include_raw: bool = True,
        search_method: str = "auto",
        embedding_threshold: float = 0.5,
        edge_min_weight: float = 0.5,
    ) -> RetrievalResult:
        """
        Full retrieval pipeline.

        Pipeline:
        1. Initial retrieval: embedding (if precomputed) / bm25 / keyword
        2. Multi-hop expansion: edge graph traversal
        3. Raw-text lookup: RawMemory

        Args:
            query: query text
            query_embedding: query vector (optional, auto-generated in auto mode)
            embedding_func: embedding generation function (used for dynamic index embedding fallback only)
            top_k: top-K count
            hops: multi-hop expansion depth
            include_raw: whether to look up raw text
            search_method: "embedding" / "keyword" / "bm25" / "auto"
                - "auto": prefer embedding (if precomputed), otherwise BM25
                - "embedding": force vector retrieval
                - "bm25": use BM25
                - "keyword": use keyword matching
            embedding_threshold: similarity threshold
            edge_min_weight: edge weight threshold

        Returns:
            RetrievalResult
        """
        result = RetrievalResult(query=query)

        # Step 1: initial retrieval
        indexes = []
        scores = {}
        used_method = search_method

        # Try embedding retrieval first
        use_embedding = search_method == "embedding" or (
            search_method == "auto" and self.has_precomputed_embeddings()
        )

        if use_embedding:
            # Auto-generate query_embedding if not provided
            if query_embedding is None:
                query_embedding = self.get_query_embedding(query)

            if query_embedding:
                scored_results = self.search_by_embedding(
                    query_embedding,
                    embedding_func=embedding_func,
                    limit=top_k,
                    threshold=embedding_threshold
                )
                indexes = [idx for idx, _ in scored_results]
                scores = {idx.id: score for idx, score in scored_results}
                if indexes:
                    used_method = "embedding"

        # Fall back to BM25
        if not indexes and search_method in ("bm25", "auto"):
            scored_results = self.search_by_bm25(query, limit=top_k)
            indexes = [idx for idx, _ in scored_results]
            scores = {idx.id: score for idx, score in scored_results}
            used_method = "bm25"

        if not indexes:  # fallback to keyword
            indexes = self.search_by_keyword(query, limit=top_k)
            scores = {idx.id: 1.0 for idx in indexes}
            used_method = "keyword"

        result.indexes = indexes
        result.index_scores = scores
        result.search_method = used_method

        # Step 2: multi-hop expansion
        if hops > 0 and indexes:
            initial_ids = [idx.id for idx in indexes]
            expanded = self.expand_by_edges(
                initial_ids, hops=hops, min_weight=edge_min_weight
            )
            result.expanded_indexes = expanded
            result.hops_used = hops

        # Step 3: raw-text lookup
        if include_raw:
            all_raw_ids = set()
            for idx in result.all_indexes:
                if idx.raw_id:
                    all_raw_ids.add(idx.raw_id)
            if all_raw_ids:
                result.raw_memories = self.get_raw_memories(list(all_raw_ids))

        return result

    def retrieve_by_entity(
        self,
        entity: str,
        top_k: int = 20,
        hops: int = 1,
        include_raw: bool = True,
    ) -> RetrievalResult:
        """
        Entity-based retrieval (aggregate all memories of a given entity).
        """
        result = RetrievalResult(query=f"entity:{entity}")

        with self.Session() as session:
            entity_lower = entity.lower()
            stmt = select(MemoryIndex).where(
                or_(
                    func.lower(MemoryIndex.subject).contains(entity_lower),
                    func.lower(MemoryIndex.object).contains(entity_lower),
                )
            ).limit(top_k)

            indexes = list(session.execute(stmt).scalars().all())

        result.indexes = indexes
        result.search_method = "entity"

        # Multi-hop expansion
        if hops > 0 and indexes:
            initial_ids = [idx.id for idx in indexes]
            result.expanded_indexes = self.expand_by_edges(initial_ids, hops=hops)
            result.hops_used = hops

        # Raw-text lookup
        if include_raw:
            all_raw_ids = set()
            for idx in result.all_indexes:
                if idx.raw_id:
                    all_raw_ids.add(idx.raw_id)
            if all_raw_ids:
                result.raw_memories = self.get_raw_memories(list(all_raw_ids))

        return result

    # ========================================================================
    # Formatting helpers
    # ========================================================================

    def format_for_prompt(
        self,
        result: RetrievalResult,
        include_expanded: bool = True,
        include_raw: bool = False,
        max_items: int = 20,
    ) -> str:
        """
        Format result as a string usable in a system prompt.
        """
        lines = [f'<retrieved_memory query="{result.query}">']

        # Primary retrieval results
        lines.append("\n<relevant_facts>")
        for idx in result.indexes[:max_items]:
            fact = f"  [{idx.subject}] {idx.predicate}"
            if idx.object:
                fact += f" [{idx.object}]"
            if idx.event_time:
                fact += f" (time: {idx.event_time})"
            score = result.index_scores.get(idx.id)
            if score:
                fact += f" (score: {score:.2f})"
            lines.append(fact)

        if not result.indexes:
            lines.append("  (No relevant facts found)")
        lines.append("</relevant_facts>")

        # Expanded results
        if include_expanded and result.expanded_indexes:
            remaining = max_items - len(result.indexes)
            if remaining > 0:
                lines.append("\n<related_facts>")
                for idx in result.expanded_indexes[:remaining]:
                    fact = f"  [{idx.subject}] {idx.predicate}"
                    if idx.object:
                        fact += f" [{idx.object}]"
                    lines.append(fact)
                lines.append("</related_facts>")

        # Raw-text lookup
        if include_raw and result.raw_memories:
            lines.append("\n<source_evidence>")
            for raw in result.raw_memories[:10]:
                text_preview = (
                    raw.raw_text[:200] + "..."
                    if len(raw.raw_text) > 200
                    else raw.raw_text
                )
                lines.append(f"  [{raw.speaker}]: {text_preview}")
            lines.append("</source_evidence>")

        lines.append("\n</retrieved_memory>")
        return "\n".join(lines)

    def format_as_context(self, result: RetrievalResult) -> str:
        """Concise format suitable for use as context."""
        facts = []
        for idx in result.all_indexes:
            fact = f"{idx.subject} {idx.predicate}"
            if idx.object:
                fact += f" {idx.object}"
            facts.append(fact)

        if not facts:
            return "No relevant memories found."

        return "Relevant facts:\n" + "\n".join(f"- {f}" for f in facts)

    # ========================================================================
    # Stats and utility methods
    # ========================================================================

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        with self.Session() as session:
            raw_count = session.execute(
                select(func.count(RawMemory.id))
            ).scalar_one()
            index_count = session.execute(
                select(func.count(MemoryIndex.id))
            ).scalar_one()
            edge_count = session.execute(
                select(func.count(MemoryIndexEdge.id))
            ).scalar_one()

            return {
                "raw_memory_count": raw_count,
                "memory_index_count": index_count,
                "memory_index_edge_count": edge_count,
            }

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity."""
        if not vec1 or not vec2:
            return 0.0

        min_len = min(len(vec1), len(vec2))
        vec1, vec2 = vec1[:min_len], vec2[:min_len]

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text."""
        # Strip punctuation
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        text = text.translate(translator)
        # Tokenize and filter
        tokens = [t.strip() for t in text.split() if t.strip() and len(t.strip()) > 1]
        return tokens
