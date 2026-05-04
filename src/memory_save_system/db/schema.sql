-- ===============================
-- Raw Memory
-- ===============================
-- Stores verbatim user / assistant messages selected for memory.
-- No interpretation here.

CREATE TABLE IF NOT EXISTS raw_memory (
  id           TEXT PRIMARY KEY,
  dia_id       TEXT NOT NULL,
  speaker      TEXT NOT NULL,
  raw_text     TEXT NOT NULL,
  record_time  TIMESTAMPTZ NOT NULL DEFAULT now(),
  update_time  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_raw_memory_dia_id
  ON raw_memory (dia_id);

CREATE INDEX IF NOT EXISTS idx_raw_memory_record_time
  ON raw_memory (record_time);

CREATE INDEX IF NOT EXISTS idx_raw_memory_speaker
  ON raw_memory (speaker);


-- ===============================
-- Memory Index
-- ===============================
-- Structured, query-friendly abstraction of memories.
-- Built from raw_memory by IndexAgent.

CREATE TABLE IF NOT EXISTS memory_index (
  id            TEXT PRIMARY KEY,
  raw_id        TEXT NOT NULL,          -- FK to raw_memory.id
  dia_id        TEXT NOT NULL,
  speaker       TEXT NOT NULL,          -- who uttered the raw message (first-level filter)

  memory_type   TEXT NOT NULL,          -- episodic / semantic / procedural / etc.

  subject       TEXT,                   -- e.g. "jack"
  subject_type  TEXT,                   -- person / org / object

  predicate     TEXT NOT NULL,           -- event / relation / fact
  object        TEXT,                   -- e.g. "dinner", "Unity"
  object_type   TEXT,                   -- event / concept / place

  event_time    TEXT,                    -- partial ISO-8601 (YYYY, YYYY-MM, YYYY-MM-DD, YYYY-MM-DDThh:mm) if resolvable
  event_time_text TEXT,                  -- original or fuzzy expression
  fingerprint   TEXT,                    -- coarse key for near-duplicate / same-event candidate grouping
  confidence    REAL,                    -- model confidence for this index item (0..1 or similar)

  record_time    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ,

  FOREIGN KEY (raw_id) REFERENCES raw_memory(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_index_raw_id
  ON memory_index (raw_id);

CREATE INDEX IF NOT EXISTS idx_memory_index_dia_id
  ON memory_index (dia_id);

CREATE INDEX IF NOT EXISTS idx_memory_index_speaker
  ON memory_index (speaker);

CREATE INDEX IF NOT EXISTS idx_memory_index_subject
  ON memory_index (subject);

CREATE INDEX IF NOT EXISTS idx_memory_index_predicate
  ON memory_index (predicate);

CREATE INDEX IF NOT EXISTS idx_memory_index_object
  ON memory_index (object);

CREATE INDEX IF NOT EXISTS idx_memory_index_event_time
  ON memory_index (event_time);

CREATE INDEX IF NOT EXISTS idx_memory_index_fingerprint
  ON memory_index (fingerprint);

-- Composite indexes for common access patterns
CREATE INDEX IF NOT EXISTS idx_memory_index_speaker_event_time
  ON memory_index (speaker, event_time);

CREATE INDEX IF NOT EXISTS idx_memory_index_speaker_fingerprint
  ON memory_index (speaker, fingerprint);


-- ===============================
-- Memory Index Edge
-- ===============================
-- Graph edges between indexed memories.

CREATE TABLE IF NOT EXISTS memory_index_edge (
  id        TEXT PRIMARY KEY,
  src_id    TEXT NOT NULL,   -- memory_index.id
  dst_id    TEXT NOT NULL,   -- memory_index.id
  edge_type TEXT NOT NULL,   -- same_event / temporal / semantic / etc.
  weight    REAL NOT NULL,

  CHECK (src_id <> dst_id),
  FOREIGN KEY (src_id) REFERENCES memory_index(id) ON DELETE CASCADE,
  FOREIGN KEY (dst_id) REFERENCES memory_index(id) ON DELETE CASCADE,
  UNIQUE (src_id, dst_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edge_src
  ON memory_index_edge (src_id);

CREATE INDEX IF NOT EXISTS idx_edge_dst
  ON memory_index_edge (dst_id);

CREATE INDEX IF NOT EXISTS idx_edge_type
  ON memory_index_edge (edge_type);

-- Enforce undirected uniqueness: (A,B,type) == (B,A,type)
CREATE UNIQUE INDEX IF NOT EXISTS uq_edge_undirected
ON memory_index_edge (
  LEAST(src_id, dst_id),
  GREATEST(src_id, dst_id),
  edge_type
);