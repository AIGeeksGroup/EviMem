import json
from pathlib import Path
from datetime import datetime, timezone

from db_managers.db_managers import RawMemoryManager, IndexManager, LinkManager

from db.db_conn import get_conn


DATA_PATH = Path("data/locomo10.json")

def parse_locomo_session_time(session_dt_hint: str):
    """Parse LOCOMO session datetime strings like '4:04 pm on 20 January, 2023'.

    Returns a naive datetime (no tzinfo) or None if parsing fails.

    Note: LOCOMO time strings do not include an explicit timezone. We treat them as wall-clock
    times without timezone info to avoid date shifts.
    """
    if not isinstance(session_dt_hint, str):
        return None
    s = session_dt_hint.strip()
    if not s:
        return None

    # Fast-path: ISO-ish timestamps
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        # Treat as wall-clock time; strip timezone info to avoid cross-timezone date shifts.
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        pass

    # Normalize am/pm variants
    s2 = s
    s2 = s2.replace("a.m.", "AM").replace("p.m.", "PM")
    s2 = s2.replace(" am", " AM").replace(" pm", " PM")
    s2 = s2.replace("am", "AM").replace("pm", "PM")
    s2 = " ".join(s2.split())

    # Common LOCOMO pattern: '4:04 PM on 20 January, 2023'
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
        try:
            dt = datetime.strptime(s2, fmt)
            return dt
        except Exception:
            continue

    return None


def iter_session_turns(sample: dict):
    """Yield (dia_id, speaker, text, session_datetime_hint) following LOCOMO10 format.

    Supported conversation formats:
      A) sample["conversation"] is a List[turn]
         where turn is {"dia_id": str, "speaker": str, "text": str}

      B) sample["conversation"] is a Dict[str, Any] containing session blocks, e.g.
         {
           "session_1_date_time": "4:04 pm on 20 January, 2023",
           "session_1": [ {"dia_id": "D1:1", "speaker": ..., "text": ...}, ... ],
           "session_2_date_time": ...,
           "session_2": [...],
           ...
         }

    For each yielded turn, `session_datetime_hint` is the best available session-level
    time hint string for the session the turn belongs to.
    """

    conv = sample.get("conversation")

    # ----------------------------
    # Format A: conversation is a list of turns
    # ----------------------------
    if isinstance(conv, list):
        # LOCOMO10 provides per-session times (session_{n}_date_time). If the dataset
        # is flattened into a single list, there is no reliable session time to attach.
        # We require the dict-of-sessions format for LOCOMO10 ingestion.
        raise ValueError(
            "Unsupported conversation format for LOCOMO10 ingestion in this harness. "
            "Expected dict-of-sessions with per-session 'session_{n}' and 'session_{n}_date_time'."
        )

    # ----------------------------
    # Format B: conversation is a dict of sessions
    # ----------------------------
    if isinstance(conv, dict):
        # Find all session numbers present, based on keys like "session_1".
        # We intentionally ignore keys ending with "_date_time" here.
        session_nums = []
        for k, v in conv.items():
            if not isinstance(k, str):
                continue
            if not k.startswith("session_"):
                continue
            if k.endswith("_date_time"):
                continue
            # Accept only list-valued session blocks
            if not isinstance(v, list):
                continue
            # Parse the number after "session_"
            suffix = k[len("session_") :]
            try:
                n = int(suffix)
            except Exception:
                continue
            session_nums.append(n)

        if not session_nums:
            raise ValueError(
                "Unsupported conversation format for LOCOMO10. "
                "Expected list at sample['conversation'] or dict with session_* lists; "
                f"got dict with keys={list(conv.keys())}"
            )

        for n in sorted(set(session_nums)):
            if n != 1 and n !=2:
                continue
            turns = conv.get(f"session_{n}")
            if not isinstance(turns, list):
                continue

            session_dt_hint = conv.get(f"session_{n}_date_time")

            for i, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                speaker = turn.get("speaker")
                dia_id = turn.get("dia_id") or f"D{n}:{i+1}"
                text = turn.get("text") or turn.get("utterance")
                if not text:
                    continue
                yield dia_id, speaker, text, session_dt_hint
        return

    # ----------------------------
    # Unknown conversation format
    # ----------------------------
    raise ValueError(
        "Unsupported conversation format for LOCOMO10. "
        f"Expected list or dict at sample['conversation'], got {type(conv)}. "
        f"sample keys={list(sample.keys())}"
    )

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Dataset JSON must be a non-empty list.")

    # 只跑指定 sample_id（避免用 data[i] 这种不稳定下标）
    target_sample_id = "conv-26"
    matches = [s for s in data if s.get("sample_id") == target_sample_id]
    if not matches:
        raise ValueError(f"No sample found with sample_id={target_sample_id!r}. "
                         f"Available sample_ids={[s.get('sample_id') for s in data[:10]]} ...")

    # 通常同一个 sample_id 只有一条；如果有多条，我们默认取第一条并提示
    if len(matches) > 1:
        print(f"[WARN] Found {len(matches)} samples with sample_id={target_sample_id!r}; using the first one.")

    sample = matches[0]
    sample_id = sample.get("sample_id", "unknown")
    print(f"[TEST] sample_id={sample_id} (matched={len(matches)})")

    raw_manager = RawMemoryManager()
    index_manager = IndexManager()
    link_manager = LinkManager()

    # Phase counters
    inserted = 0
    skipped = 0
    indexed = 0
    index_skipped = 0
    edges_created = 0

    with get_conn() as conn:
        # -------------------------
        # Phase A: Fill memory DB
        # -------------------------
        for dia_id, speaker, text, session_dt_hint in iter_session_turns(sample):
            session_time = parse_locomo_session_time(session_dt_hint) if session_dt_hint else None
            result = raw_manager(
                conn=conn,
                user_text=text,
                speaker=speaker,
                dia_id=dia_id,
                session_time=session_time,
            )

            if result:
                inserted += 1
                print(f"[INSERTED] {dia_id} speaker={speaker} text={text[:60]!r}")

                # Build index rows for this raw row
                try:
                    idx_result = index_manager(
                        conn=conn,
                        raw_row=result,
                        memory_type="raw",
                    )
                except TypeError:
                    idx_result = index_manager(
                        conn=conn,
                        raw_row=result,
                    )

                if idx_result:
                    indexed += 1
                    print(f"[INDEXED ] raw_id={result.get('id')} -> {idx_result}")

                    # Build edges for newly created index node(s)
                    new_rows = idx_result if isinstance(idx_result, list) else [idx_result]
                    edge_summary = link_manager(
                        conn=conn,
                        new_index_rows=new_rows,
                        enable_clustering=True,
                    )
                    edges_created += int(edge_summary.get("edges_created", 0))
                    print(f"[EDGES  ] {edge_summary}")
                else:
                    index_skipped += 1
                    print(f"[NOINDEX ] raw_id={result.get('id')}")
            else:
                skipped += 1
                print(f"[SKIPPED ] {dia_id} speaker={speaker} text={text[:60]!r}")

        print(
            f"\n[INGEST SUMMARY] raw_inserted={inserted}, raw_skipped={skipped}, "
            f"indexed={indexed}, index_skipped={index_skipped}, edges_created={edges_created}\n"
        )


if __name__ == "__main__":
    main()