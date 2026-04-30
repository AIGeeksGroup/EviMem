"""
Embedding precomputation script.

Generates embeddings for all MemoryIndex entries in one shot and stores them in the database.
At query time embeddings are read directly, with no further API calls.

Usage:
    python precompute_embeddings.py --db path/to/memory.db
"""

import os
import sys
import json
import argparse
import time
from typing import List

from sqlalchemy import create_engine, text, Column, Text
from sqlalchemy.orm import sessionmaker

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def add_embedding_column_if_not_exists(engine):
    """Add the `embedding` column if it does not exist."""
    with engine.connect() as conn:
        # Check whether the column exists
        result = conn.execute(text("PRAGMA table_info(memory_index)"))
        columns = [row[1] for row in result.fetchall()]

        if "embedding" not in columns:
            print("[Precompute] Adding 'embedding' column to memory_index table...")
            conn.execute(text("ALTER TABLE memory_index ADD COLUMN embedding TEXT"))
            conn.commit()
            print("[Precompute] Column added.")
        else:
            print("[Precompute] 'embedding' column already exists.")


def get_openai_client():
    """Get an OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        print("Please run: export OPENAI_API_KEY=your_key")
        sys.exit(1)

    try:
        import openai
        return openai.OpenAI(api_key=api_key)
    except ImportError:
        print("Error: openai package not installed")
        print("Please run: pip install openai")
        sys.exit(1)


def batch_embed(client, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Batch-generate embeddings."""
    response = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]


def precompute_embeddings(
    db_path: str,
    batch_size: int = 100,
    model: str = "text-embedding-3-small",
    force: bool = False,
):
    """
    Precompute embeddings for all MemoryIndex entries.

    Args:
        db_path: database path
        batch_size: batch size
        model: OpenAI embedding model
        force: whether to force recomputing all embeddings
    """
    print(f"\n{'='*60}")
    print("PRECOMPUTE EMBEDDINGS")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print(f"Force recompute: {force}")

    # Initialize
    engine = create_engine(db_path, echo=False)
    Session = sessionmaker(bind=engine)
    client = get_openai_client()

    # Ensure the embedding column exists
    add_embedding_column_if_not_exists(engine)

    with Session() as session:
        # Get records to process
        if force:
            result = session.execute(text("SELECT id, subject, predicate, object FROM memory_index"))
        else:
            result = session.execute(text(
                "SELECT id, subject, predicate, object FROM memory_index WHERE embedding IS NULL"
            ))

        rows = result.fetchall()
        total = len(rows)

        if total == 0:
            print("\n[Precompute] All embeddings already computed. Use --force to recompute.")
            return

        print(f"\n[Precompute] Processing {total} records...")

        # Process in batches
        processed = 0
        start_time = time.time()

        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]

            # Build texts
            texts = []
            ids = []
            for row in batch:
                idx_id, subject, predicate, obj = row
                text_content = f"{subject} {predicate} {obj or ''}".strip()
                texts.append(text_content)
                ids.append(idx_id)

            # Generate embeddings
            try:
                embeddings = batch_embed(client, texts, model)

                # Update the database
                for idx_id, embedding in zip(ids, embeddings):
                    embedding_json = json.dumps(embedding)
                    session.execute(
                        text("UPDATE memory_index SET embedding = :emb WHERE id = :id"),
                        {"emb": embedding_json, "id": idx_id}
                    )

                session.commit()
                processed += len(batch)

                # Progress
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0

                print(f"  [{processed}/{total}] {processed/total*100:.1f}% "
                      f"({rate:.1f} items/sec, ETA: {eta:.0f}s)")

            except Exception as e:
                print(f"  [Error] Batch {i}-{i+batch_size}: {e}")
                continue

        # Done
        elapsed = time.time() - start_time
        print(f"\n[Precompute] Done! Processed {processed} records in {elapsed:.1f}s")


def verify_embeddings(db_path: str):
    """Verify whether embeddings have been generated."""
    engine = create_engine(db_path, echo=False)

    with engine.connect() as conn:
        # Total
        total = conn.execute(text("SELECT COUNT(*) FROM memory_index")).scalar()

        # Number with embedding
        with_emb = conn.execute(text(
            "SELECT COUNT(*) FROM memory_index WHERE embedding IS NOT NULL"
        )).scalar()

        # Sample check
        sample = conn.execute(text(
            "SELECT id, subject, predicate, embedding FROM memory_index WHERE embedding IS NOT NULL LIMIT 1"
        )).fetchone()

        print(f"\n{'='*60}")
        print("VERIFICATION")
        print(f"{'='*60}")
        print(f"Total records: {total}")
        print(f"With embedding: {with_emb} ({with_emb/total*100:.1f}%)")
        print(f"Without embedding: {total - with_emb}")

        if sample:
            emb = json.loads(sample[3])
            print(f"\nSample:")
            print(f"  ID: {sample[0]}")
            print(f"  Text: {sample[1]} {sample[2]}")
            print(f"  Embedding dim: {len(emb)}")
            print(f"  Embedding preview: [{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute embeddings for MemoryIndex")
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Database path (SQLAlchemy format, e.g., sqlite:///path/to/db)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation (default: 100)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute all embeddings"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing embeddings"
    )

    args = parser.parse_args()

    # Make sure path format is correct
    db_path = args.db
    if not db_path.startswith("sqlite:///"):
        db_path = f"sqlite:///{db_path}"

    if args.verify:
        verify_embeddings(db_path)
    else:
        precompute_embeddings(db_path, args.batch_size, args.model, args.force)
        verify_embeddings(db_path)
