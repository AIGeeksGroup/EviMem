# LaceMem-1.0

## Overview

**LaceMem** (Layered Architecture for Conversational Evidence Memory) is a coarse-to-fine memory hierarchy of three layers — **Index** (atomic semantic tuples for search), **Edge** (graph links for multi-hop expansion), and **Raw** (verbatim dialogue for grounding).

This package (**LaceMem-1.0**) lives inside the EviMem repo at **`src/memory_save_system/`**. It materialises those layers in PostgreSQL: it ingests [LoCoMo](https://github.com/snap-research/locomo)-style multi-session dialogue, builds Raw → Index → Edge via LLM-backed managers, and can export the same tables to SQLite for tools that expect a single-file database.

## Method summary

**LaceMem** organises dialogue into a coarse-to-fine three-layer hierarchy: an *Index layer* of atomic semantic tuples for fine-grained search, an *Edge layer* of graph links for multi-hop expansion, and a *Raw layer* of verbatim dialogue for grounded generation.

## What is in this repository


| Path                                              | Role                                                                                     |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `db/schema.sql`                                   | DDL for `raw_memory`, `memory_index`, `memory_index_edge`                                |
| `db/db_conn.py`                                   | PostgreSQL connection (`MEM_EVAL_DB_URL`, default `postgresql://localhost:5432/LaceMem`) |
| `db/db_raw.py`, `db/db_index.py`, `db/db_edge.py` | Typed accessors for each layer                                                           |
| `prompts/`                                        | LLM prompts for raw ingest, indexing, and linking                                        |
| `db_managers/db_managers.py`                      | `RawMemoryManager`, `IndexManager`, `LinkManager` orchestration                          |
| `men_llm/llm_client.py`                           | OpenAI client wrapper (`OPENAI_API_KEY` or `./api.key`)                                  |
| `run_eval.py`                                     | End-to-end ingest of one LoCoMo sample (default reads `data/locomo10.json`) into Postgres  |
| `pg_to_sqlite.py`                                 | Copy the three LaceMem tables from Postgres into `LaceMem.sqlite.db`                     |
| `requirements.txt`                                | Python dependencies (`pip install -r requirements.txt`)                                  |
| `data/`                                           | Drop `locomo10.json` from the [official LoCoMo repo](https://github.com/snap-research/locomo) here (not tracked in git) |


## Setup

### 1. Python environment

Use Python 3.10+ (3.11 recommended). From the **EviMem repository root**, create a virtual environment, activate it, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r src/memory_save_system/requirements.txt
```

`psycopg[binary]` powers `run_eval` / `db/db_conn.py`; `psycopg2-binary` is only needed for `pg_to_sqlite.py`.

### 2. PostgreSQL

Install and start PostgreSQL locally (or point `MEM_EVAL_DB_URL` at a remote instance). Ensure `psql` and `createdb` / `dropdb` are on your `PATH`.

### 3. OpenAI API key

Either export:

```bash
export OPENAI_API_KEY=sk-...
```

or create a file `api.key` in the project root containing the key on a single line (as expected by `LLMClient` / `RawMemoryManager`).

Optional: `MEM_EVAL_MODEL` selects the chat model (default `gpt-4o-mini`).

### 4. LoCoMo data

Download **`locomo10.json`** from the [LoCoMo repository](https://github.com/snap-research/locomo) and place it at `**data/locomo10.json**` relative to `src/memory_save_system/` (or change `DATA_PATH` in `run_eval.py`).

## Usage

### Create the database and tables

Run these from **`src/memory_save_system/`** (so relative paths resolve):

```bash
cd src/memory_save_system
dropdb "LaceMem"   # optional; remove old DB if it exists and nothing else needs it
createdb "LaceMem"
psql -d "LaceMem" -f db/schema.sql -q
```

If your Postgres role or host differ from the default, set `MEM_EVAL_DB_URL`, e.g.:

```bash
export MEM_EVAL_DB_URL=postgresql://user:password@localhost:5432/LaceMem
```

### Ingest one conversation (LaceMem build)

1. Set `target_sample_id` in `run_eval.py` to the LoCoMo sample you want (e.g. `conv-26`, `conv-30`).
2. From `src/memory_save_system/`, run:

```bash
cd src/memory_save_system
python run_eval.py
```

This fills **Raw** turns, then **Index**, then **Edge** links for the selected sample.

### Export Postgres → SQLite (optional)

For downstream stacks that expect SQLite:

```bash
cd src/memory_save_system
python pg_to_sqlite.py
```

Writes `**LaceMem.sqlite.db**` with tables `raw_memory`, `memory_index`, `memory_index_edge`. Connection parameters follow `PG*` env vars and defaults in `pg_to_sqlite.py`.

## Notes

- `**run_eval.py**` currently targets a single `sample_id`; adjust `target_sample_id` per run.
- Ensure no other process holds open connections to `"LaceMem"` before `dropdb` (terminate sessions or use `DROP DATABASE ... WITH (FORCE)` in PostgreSQL 13+).
- `**.ruff_cache/**` is created by the [Ruff](https://docs.astral.sh/ruff/) linter when you run `ruff check`; it is safe to delete and will be recreated.

## Acknowledgements

- [LoCoMo](https://github.com/snap-research/locomo) — long-term conversational memory benchmark and data format.

