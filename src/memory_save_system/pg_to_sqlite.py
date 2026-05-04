import os
import psycopg2
import sqlite3

PG_CONN = dict(
    dbname=os.getenv("PGDATABASE", "LaceMem"),
    user=os.getenv("PGUSER", os.getenv("USER", "")),
    password=os.getenv("PGPASSWORD"),
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
)

print("[PG_CONN]", {k: ("***" if k == "password" and PG_CONN[k] else PG_CONN[k]) for k in PG_CONN})

SQLITE_DB = "LaceMem.sqlite.db"
TABLES = ["raw_memory", "memory_index", "memory_index_edge"]

pg = psycopg2.connect(**PG_CONN)
pg_cur = pg.cursor()

sqlite = sqlite3.connect(SQLITE_DB)
sq_cur = sqlite.cursor()

def copy_table(table):
    pg_cur.execute(f"SELECT * FROM {table}")
    rows = pg_cur.fetchall()
    cols = [d[0] for d in pg_cur.description]

    sq_cur.execute(f"DROP TABLE IF EXISTS {table}")
    col_defs = ", ".join([f'"{c}" TEXT' for c in cols])
    sq_cur.execute(f'CREATE TABLE "{table}" ({col_defs})')

    placeholders = ", ".join(["?"] * len(cols))
    sq_cur.executemany(
        f'INSERT INTO "{table}" VALUES ({placeholders})',
        rows
    )

for t in TABLES:
    copy_table(t)

sqlite.commit()
sqlite.close()
pg.close()
print("[OK] wrote", SQLITE_DB)