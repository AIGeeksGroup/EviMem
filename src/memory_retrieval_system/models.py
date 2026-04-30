"""
ORM model definitions - three-tier memory architecture.

RawMemory: raw memory layer
MemoryIndex: index layer (SPO triples)
MemoryIndexEdge: graph link layer
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, String, Text, DateTime, Float, ForeignKey, Index as SQLIndex,
    create_engine
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class RawMemory(Base):
    """
    Raw memory layer - traceable source of facts.

    Design goals:
    - Acts as "source of truth"; all downstream Index/Edge entries must be traceable here
    - Stores raw text without rewriting, to avoid LLM hallucination during abstraction/summarization
    - For QA-style memories, context has already been integrated at storage time
    """
    __tablename__ = "raw_memory"

    id = Column(String, primary_key=True)
    dia_id = Column(String, nullable=True)  # LoCoMo dialogue marker, e.g. D1:3
    speaker = Column(String, nullable=False)  # user / assistant / third-party name
    raw_text = Column(Text, nullable=False)  # full raw text
    record_time = Column(DateTime, nullable=False)
    update_time = Column(DateTime, nullable=True)

    # Linked indexes
    indexes = relationship("MemoryIndex", back_populates="raw_memory")

    __table_args__ = (
        SQLIndex("idx_raw_memory_dia_id", "dia_id"),
        SQLIndex("idx_raw_memory_speaker", "speaker"),
        SQLIndex("idx_raw_memory_record_time", "record_time"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dia_id": self.dia_id,
            "speaker": self.speaker,
            "raw_text": self.raw_text,
            "record_time": self.record_time.isoformat() if self.record_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
        }

    def __repr__(self):
        return f"<RawMemory(id={self.id}, speaker={self.speaker})>"


class MemoryIndex(Base):
    """
    Memory index layer - SPO-triple structured index.

    Design goals:
    - Make memory retrievable, aggregatable, sortable
    - Address recall loss caused by synonym rewrites, varied temporal expressions, etc.
    - Provide structured semantics for the answering stage

    Generation rules:
    - One RawMemory can produce multiple MemoryIndex entries
    - Each Index expresses a single "minimum fact unit"
    """
    __tablename__ = "memory_index"

    id = Column(String, primary_key=True)
    raw_id = Column(String, ForeignKey("raw_memory.id"), nullable=False)
    dia_id = Column(String, nullable=True)
    speaker = Column(String, nullable=True)

    # Index type (optional classification)
    memory_type = Column(String, nullable=True)

    # SPO triple - minimum fact unit
    subject = Column(String, nullable=False)
    subject_type = Column(String, nullable=True)  # person/org/place/concept
    predicate = Column(String, nullable=False)  # relation/action
    object = Column(String, nullable=True)
    object_type = Column(String, nullable=True)

    # Temporal info - supports multiple precisions
    event_time = Column(String, nullable=True)  # YYYY / YYYY-MM / YYYY-MM-DD
    event_time_text = Column(String, nullable=True)  # raw expression: yesterday, soon

    # Deduplication and confidence
    fingerprint = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)  # confidence score

    record_time = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=True)  # note: real table uses updated_at

    # Relationship
    raw_memory = relationship("RawMemory", back_populates="indexes")

    __table_args__ = (
        SQLIndex("idx_memory_index_raw_id", "raw_id"),
        SQLIndex("idx_memory_index_subject", "subject"),
        SQLIndex("idx_memory_index_predicate", "predicate"),
        SQLIndex("idx_memory_index_object", "object"),
        SQLIndex("idx_memory_index_fingerprint", "fingerprint"),
        SQLIndex("idx_memory_index_dia_id", "dia_id"),
        SQLIndex("idx_memory_index_speaker", "speaker"),
        SQLIndex("idx_memory_index_event_time", "event_time"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "raw_id": self.raw_id,
            "dia_id": self.dia_id,
            "speaker": self.speaker,
            "memory_type": self.memory_type,
            "subject": self.subject,
            "subject_type": self.subject_type,
            "predicate": self.predicate,
            "object": self.object,
            "object_type": self.object_type,
            "event_time": self.event_time,
            "event_time_text": self.event_time_text,
            "fingerprint": self.fingerprint,
            "confidence": self.confidence,
            "record_time": self.record_time.isoformat() if self.record_time else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_spo_string(self) -> str:
        """Convert to SPO string representation."""
        obj_str = f" [{self.object}]" if self.object else ""
        time_str = f" (time: {self.event_time})" if self.event_time else ""
        return f"[{self.subject}] {self.predicate}{obj_str}{time_str}"

    def to_fact_string(self) -> str:
        """Convert to a natural-language fact (including time)."""
        obj_str = f" {self.object}" if self.object else ""
        # Prefer absolute time; fall back to the raw temporal expression
        time_str = ""
        if self.event_time:
            time_str = f" (time: {self.event_time})"
        elif self.event_time_text:
            time_str = f" (time_ref: {self.event_time_text})"
        return f"{self.subject} {self.predicate}{obj_str}{time_str}"

    def __repr__(self):
        return f"<MemoryIndex(id={self.id}, {self.subject} {self.predicate} {self.object})>"


class MemoryIndexEdge(Base):
    """
    Memory graph link layer - connects Index nodes into a graph.

    Design goals:
    - Cross-turn aggregation for the same entity
    - Event chains (temporally adjacent, causal/motivational)
    - Conflict detection and latest-version selection

    Edge types:
    - same_raw: multiple Index entries from the same RawMemory, weight=1.0
    - semantic_similarity: embedding similarity above threshold
    """
    __tablename__ = "memory_index_edge"

    id = Column(String, primary_key=True)
    src_id = Column(String, ForeignKey("memory_index.id"), nullable=False)
    dst_id = Column(String, ForeignKey("memory_index.id"), nullable=False)
    edge_type = Column(String, nullable=False)  # same_raw / semantic_similarity
    weight = Column(Float, default=1.0)  # link strength

    __table_args__ = (
        SQLIndex("idx_edge_src", "src_id"),
        SQLIndex("idx_edge_dst", "dst_id"),
        SQLIndex("idx_edge_type", "edge_type"),
        SQLIndex("idx_edge_weight", "weight"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
        }

    def __repr__(self):
        return f"<MemoryIndexEdge({self.src_id} --[{self.edge_type}]--> {self.dst_id})>"


def init_database(db_path: str, echo: bool = False):
    """
    Initialize the database (only creates table schema, used for testing).
    In production, tables are created by the writer project.
    """
    engine = create_engine(db_path, echo=echo)
    Base.metadata.create_all(engine)
    return engine
