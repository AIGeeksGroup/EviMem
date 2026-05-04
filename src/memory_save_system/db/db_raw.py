from typing import Any, Protocol


class DBAPICursor(Protocol):
    description: Any
    def execute(self, operation: str, parameters: Any = None) -> Any: ...
    def fetchall(self) -> Any: ...


class DBAPIConnection(Protocol):
    def cursor(self) -> DBAPICursor: ...
    def commit(self) -> None: ...


RAW_TURN_DDL = """
CREATE TABLE IF NOT EXISTS raw_memory (
    id TEXT PRIMARY KEY,
    dia_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    record_time TIMESTAMPTZ NOT NULL,
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def ensure_raw_memory_table(conn: DBAPIConnection) -> None:
    """
    Ensure the raw_memory table exists in PostgreSQL.
    """
    cur = conn.cursor()
    # Some drivers don't allow multi-statement execute; run statements separately.
    for stmt in [s.strip() for s in RAW_TURN_DDL.split(";") if s.strip()]:
        cur.execute(stmt)
    conn.commit()


def insert_turn(conn: DBAPIConnection, turn: Any) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO raw_memory (id, dia_id, speaker, raw_text, record_time, update_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            turn.id,
            turn.dia_id,
            turn.speaker,
            turn.text,
            turn.record_time,
            turn.record_time,
        ),
    )
    conn.commit()


def update_raw_memory(conn: DBAPIConnection, raw_id: str, new_text: str, update_time) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE raw_memory
        SET raw_text = %s,
            update_time = %s
        WHERE id = %s
        """,
        (new_text, update_time, raw_id),
    )
    conn.commit()
