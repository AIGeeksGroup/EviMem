import os
import psycopg
from contextlib import contextmanager


# Expected env vars:
#   MEM_EVAL_DB_URL=postgresql://user:password@localhost:5432/LaceMem
#
# Fallback is intentionally explicit to avoid silent misconfig.
DEFAULT_DB_URL = "postgresql://localhost:5432/LaceMem"


def get_db_url() -> str:
    return os.getenv("MEM_EVAL_DB_URL", DEFAULT_DB_URL)


@contextmanager
def get_conn():
    """
    Context-managed PostgreSQL connection.
    Usage:
        with get_conn() as conn:
            conn.execute(...)
    """
    conn = psycopg.connect(get_db_url())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_db():
    """
    Simple connectivity check.
    Does NOT create tables.
    """
    with get_conn() as conn:
        conn.execute("SELECT 1")
