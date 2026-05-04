"""db_edge.py

Edge storage for the index graph.

We store undirected edges between index items (rows in `memory_index`). Because Postgres
is not a native graph DB, we simulate a graph with an edge table.

Design choices:
- Undirected edge: we canonicalize (src_id, dst_id) so src_id < dst_id (lexicographic).
- De-duplication: rely on UNIQUE (src_id, dst_id, edge_type) at the DB level.
- Weight: float confidence/strength of the edge. Same-batch edges can use weight=1.0.

This module only defines table creation and basic CRUD helpers. It does NOT run LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EDGE_TABLE_NAME = "memory_index_edge"


@dataclass(frozen=True)
class EdgeRow:
    id: str
    src_id: str
    dst_id: str
    edge_type: str
    weight: float


def _canon_pair(a: str, b: str) -> tuple[str, str]:
    """Canonicalize an undirected edge pair.

    We enforce a stable ordering to avoid storing both (A,B) and (B,A).
    """
    if a == b:
        raise ValueError("self-loop edges are not allowed: src_id == dst_id")
    return (a, b) if a < b else (b, a)


def create_edge_table(conn) -> None:
    """Create the edge table and indexes if they do not exist.

    Note: If you already manage schema via `db/schema.sql`, you can keep this as a
    safety net for local testing.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_index_edge (
              id         TEXT PRIMARY KEY,
              src_id     TEXT NOT NULL,
              dst_id     TEXT NOT NULL,
              edge_type  TEXT NOT NULL,
              weight     REAL NOT NULL,

              CHECK (src_id <> dst_id),
              FOREIGN KEY (src_id) REFERENCES memory_index(id) ON DELETE CASCADE,
              FOREIGN KEY (dst_id) REFERENCES memory_index(id) ON DELETE CASCADE,
              UNIQUE (src_id, dst_id, edge_type)
            );
            """
        )

        # Migration for older local schemas that included evidence/created_at
        cur.execute(
            f"ALTER TABLE {EDGE_TABLE_NAME} DROP COLUMN IF EXISTS evidence;"
        )
        cur.execute(
            f"ALTER TABLE {EDGE_TABLE_NAME} DROP COLUMN IF EXISTS created_at;"
        )

        # Helpful indexes for traversal
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{EDGE_TABLE_NAME}_src ON {EDGE_TABLE_NAME} (src_id);"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{EDGE_TABLE_NAME}_dst ON {EDGE_TABLE_NAME} (dst_id);"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{EDGE_TABLE_NAME}_type ON {EDGE_TABLE_NAME} (edge_type);"
        )
    conn.commit()


def insert_edge(
    *,
    conn,
    edge_id: str,
    src_id: str,
    dst_id: str,
    edge_type: str,
    weight: float = 1.0,
) -> EdgeRow:
    """Insert an edge. If the edge already exists, update its weight.

    We keep it simple: ON CONFLICT updates weight to the max of old/new.
    """
    src_id, dst_id = _canon_pair(src_id, dst_id)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO memory_index_edge (
              id, src_id, dst_id, edge_type, weight
            ) VALUES (
              %s, %s, %s, %s, %s
            )
            ON CONFLICT (src_id, dst_id, edge_type)
            DO UPDATE SET
              weight = GREATEST(memory_index_edge.weight, EXCLUDED.weight)
            RETURNING id, src_id, dst_id, edge_type, weight;
            """,
            (edge_id, src_id, dst_id, edge_type, float(weight)),
        )
        row = cur.fetchone()
    conn.commit()

    return EdgeRow(
        id=row[0],
        src_id=row[1],
        dst_id=row[2],
        edge_type=row[3],
        weight=float(row[4]),
    )


def neighbors(
    *,
    conn,
    node_id: str,
    edge_type: None | str = None,
    min_weight: float = 0.0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return neighbor node ids for a given node.

    Because edges are undirected, a node can appear in src_id or dst_id.
    Returns a list of dicts: {neighbor_id, edge_type, weight}
    """
    where = ["weight >= %s", "(src_id = %s OR dst_id = %s)"]
    params: list[Any] = [float(min_weight), node_id, node_id]

    if edge_type is not None:
        where.append("edge_type = %s")
        params.append(edge_type)

    where_sql = " AND ".join(where)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
              CASE WHEN src_id = %s THEN dst_id ELSE src_id END AS neighbor_id,
              edge_type,
              weight
            FROM memory_index_edge
            WHERE {where_sql}
            ORDER BY weight DESC
            LIMIT %s;
            """,
            [node_id, *params, int(limit)],
        )
        rows = cur.fetchall()

    return [
        {
            "neighbor_id": r[0],
            "edge_type": r[1],
            "weight": float(r[2]),
        }
        for r in rows
    ]


def delete_edge(
    *,
    conn,
    src_id: str,
    dst_id: str,
    edge_type: str,
) -> int:
    """Delete an edge by (src_id, dst_id, edge_type). Returns number of rows deleted."""
    src_id, dst_id = _canon_pair(src_id, dst_id)
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {EDGE_TABLE_NAME} WHERE src_id = %s AND dst_id = %s AND edge_type = %s;",
            (src_id, dst_id, edge_type),
        )
        n = cur.rowcount
    conn.commit()
    return int(n)
