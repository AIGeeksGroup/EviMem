"""
db_index.py

Index table for memory retrieval.

This is NOT the full memory store. It is a structured "map" extracted from raw
turns (raw_memory) to enable precise filtering and graph-style traversal using SPO triples plus time.

Design notes:
- Store SPO triples (subject, predicate, object) with optional types for structured indexing.
- Use record_time as the authoritative record time instead of separate record_time.
- Keep columns simple scalar TEXT/TIMESTAMPTZ where appropriate; event_time is TEXT (partial ISO-8601) to preserve precision without inventing timestamps.
- Allow multiple index rows to point to the same raw_memory row (split complex events),
"""

from __future__ import annotations

from typing import Optional


INDEX_TABLE_NAME = "memory_index"


def create_index_table(conn) -> None:
    """
    Create the memory_index table (and useful indexes) if it doesn't exist.

    Args:
        conn: psycopg connection (from db.db_conn.get_conn()).
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {INDEX_TABLE_NAME} (
          -- index row id
          id             TEXT PRIMARY KEY,

          -- link back to raw memory (ground truth)
          raw_id         TEXT NOT NULL REFERENCES raw_memory(id) ON DELETE CASCADE,

          -- type of memory that produced this index row (raw / episodic / semantic / core / procedural / resource / kv)
          memory_type    TEXT NOT NULL,

          -- dialogue / session identifier
          dia_id         TEXT NOT NULL,

          -- speaker who uttered the raw memory (first-level filter)
          speaker        TEXT NOT NULL,

          -- SPO triple
          subject        TEXT NOT NULL,
          subject_type   TEXT,
          predicate      TEXT NOT NULL,
          object         TEXT,
          object_type    TEXT,

          -- time dimensions
          event_time      TEXT,
          event_time_text TEXT,

          -- fast candidate grouping / optional confidence
          fingerprint    TEXT,
          confidence     DOUBLE PRECISION,

          -- record_time is authoritative record time
          record_time     TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )

    # Migration for older local schemas where event_time was TIMESTAMPTZ
    try:
        conn.execute(
            f"ALTER TABLE {INDEX_TABLE_NAME} ALTER COLUMN event_time TYPE TEXT USING event_time::text;"
        )
    except Exception:
        # Ignore if column is already TEXT or table doesn't yet exist in this connection context
        pass

    # Indexes for typical query patterns
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_raw_id ON {INDEX_TABLE_NAME} (raw_id);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_dia_id ON {INDEX_TABLE_NAME} (dia_id);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_speaker ON {INDEX_TABLE_NAME} (speaker);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_event_time ON {INDEX_TABLE_NAME} (event_time);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_fingerprint ON {INDEX_TABLE_NAME} (fingerprint);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_record_time ON {INDEX_TABLE_NAME} (record_time);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_subject ON {INDEX_TABLE_NAME} (subject);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_predicate ON {INDEX_TABLE_NAME} (predicate);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_object ON {INDEX_TABLE_NAME} (object);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_subject_type ON {INDEX_TABLE_NAME} (subject_type);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE_NAME}_object_type ON {INDEX_TABLE_NAME} (object_type);"
    )


def ensure_index_schema(conn) -> None:
    """
    Ensure memory_index schema exists.
    """
    create_index_table(conn)


def normalize_fingerprint(
    *,
    subject: Optional[str],
    predicate: Optional[str],
    object: Optional[str],
    event_time: Optional[str],
) -> Optional[str]:
    """
    Build a coarse deterministic fingerprint for candidate grouping.

    Format:
        subject|predicate|object|YYYY-MM-DD (if event_time provides at least day precision)

    Any missing component is skipped. If all are missing, returns None.
    """
    parts = []
    if subject:
        parts.append(subject.strip().lower())
    if predicate:
        parts.append(predicate.strip().lower())
    if object:
        parts.append(object.strip().lower())
    if event_time:
        # Only include day-level precision in the fingerprint to avoid inventing time.
        # Accept partial ISO-8601 strings; include only when at least YYYY-MM-DD is present.
        et = event_time.strip()
        if len(et) >= 10 and et[4] == "-" and et[7] == "-":
            parts.append(et[:10])

    if not parts:
        return None
    return "|".join(parts)