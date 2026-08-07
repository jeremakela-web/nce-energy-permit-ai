"""
retrieval_trace.py — TASO 1: additive retrieval-tracing layer for the RAG pipeline.

Purely additive logging alongside the existing generation pipeline. Does NOT alter
source_policy.py's relevance rules and does NOT touch RAQS scoring logic — it only
records what the pipeline already computes, for debugging, provenance ("which
sources did this document draw on?"), and as a foundation for later cost-tracking /
rollback work.

Storage: SQLite (same pattern already used in this codebase for `raqs_reviews.db`,
see `_RAQS_DB` in generate_application.py — no PostgreSQL exists in this stack, and
no new dependency is added here; sqlite3 is stdlib). File lives on the same
persistent-disk directory as the ChromaDB index / raqs_reviews.db, so it survives
Render deploys.

Two tables:
  retrieval_trace       — one row per `_rag_context()` call (chunk-level detail as JSON)
  retrieval_trace_raqs  — one row per RAQS review, linked by generation_id

IMPORTANT: full chunk text is never stored here — only chunk IDs, similarity scores,
and lightweight metadata (source_type). Full chunk text stays in ChromaDB, referenced
by chunk_id, to avoid duplicating storage.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

_HERE     = os.path.dirname(os.path.abspath(__file__))
_DB_DIR   = os.path.join(_HERE, "embeddings")  # same dir as raqs_reviews.db / chroma
_TRACE_DB = os.path.join(_DB_DIR, "retrieval_trace.db")

_lock = threading.Lock()


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_trace (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id        TEXT    NOT NULL,
            created_at           TEXT    NOT NULL,
            query_text           TEXT,
            country              TEXT,
            hanketyyppi_tag      TEXT,
            avg_score            REAL,
            min_score_threshold  REAL,
            chunks_json          TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_trace_raqs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id  TEXT    NOT NULL,
            created_at     TEXT    NOT NULL,
            scores_json    TEXT    NOT NULL,
            overall        REAL,
            flagged_json   TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_retrieval_trace_gen ON retrieval_trace(generation_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_retrieval_trace_raqs_gen ON retrieval_trace_raqs(generation_id)"
    )


def _connect() -> sqlite3.Connection:
    os.makedirs(_DB_DIR, exist_ok=True)
    conn = sqlite3.connect(_TRACE_DB)
    _init_db(conn)
    return conn


def log_retrieval(
    *,
    generation_id: str,
    query_text: str,
    country: str,
    hanketyyppi_tag: str,
    chunks: list[dict],
    avg_score: float,
    min_score_threshold: Optional[float] = None,
) -> None:
    """Record one `_rag_context()` retrieval call. Never raises — a tracing failure
    must never block report generation.

    `chunks` items: {"chunk_id": str, "score": float, "source_type": str}.
    No chunk text is accepted or stored here by design.
    """
    if not generation_id:
        return
    try:
        safe_chunks = [
            {
                "chunk_id":    str(c.get("chunk_id", ""))[:200],
                "score":       round(float(c.get("score", 0.0)), 4),
                "source_type": str(c.get("source_type") or "unknown")[:100],
            }
            for c in (chunks or [])
        ]
        with _lock, _connect() as conn:
            conn.execute(
                "INSERT INTO retrieval_trace "
                "(generation_id, created_at, query_text, country, hanketyyppi_tag, "
                " avg_score, min_score_threshold, chunks_json) VALUES (?,?,?,?,?,?,?,?)",
                (
                    generation_id,
                    datetime.now(timezone.utc).isoformat(),
                    (query_text or "")[:2000],
                    country,
                    hanketyyppi_tag,
                    round(float(avg_score), 4) if avg_score is not None else None,
                    round(float(min_score_threshold), 4) if min_score_threshold is not None else None,
                    json.dumps(safe_chunks, ensure_ascii=False),
                ),
            )
    except Exception:
        pass  # never block generation


def log_raqs_outcome(
    *,
    generation_id: str,
    review: dict,
    raqs_order: list[str],
    flagged: Optional[list[dict]] = None,
) -> None:
    """Record the RAQS agent's final 5-criteria scores for this generation.

    `flagged`: RAQS reviews generated text sections, not individual chunks, so
    there is no chunk-ID-level flag in the current scoring mechanism — pass the
    criteria the agent itself flagged as low-confidence (if any), which is the
    closest honest equivalent. Defaults to [] when nothing was flagged.
    """
    if not generation_id or not review:
        return
    try:
        scores = {k: review[k] for k in raqs_order if k in review}
        pisteet = [v.get("pisteet") for v in scores.values() if isinstance(v, dict) and "pisteet" in v]
        overall = round(sum(pisteet) / len(pisteet), 2) if pisteet else None
        with _lock, _connect() as conn:
            conn.execute(
                "INSERT INTO retrieval_trace_raqs "
                "(generation_id, created_at, scores_json, overall, flagged_json) VALUES (?,?,?,?,?)",
                (
                    generation_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(scores, ensure_ascii=False),
                    overall,
                    json.dumps(flagged or [], ensure_ascii=False),
                ),
            )
    except Exception:
        pass  # never block generation


def get_trace(generation_id: str) -> dict:
    """Read-only fetch: all retrieval calls + the latest RAQS outcome for one generation_id."""
    conn = _connect()
    try:
        conn.row_factory = sqlite3.Row
        retrievals = [
            dict(row) for row in conn.execute(
                "SELECT * FROM retrieval_trace WHERE generation_id = ? ORDER BY id",
                (generation_id,),
            )
        ]
        for r in retrievals:
            r["chunks"] = json.loads(r.pop("chunks_json") or "[]")

        raqs_rows = [
            dict(row) for row in conn.execute(
                "SELECT * FROM retrieval_trace_raqs WHERE generation_id = ? ORDER BY id",
                (generation_id,),
            )
        ]
        raqs = None
        if raqs_rows:
            raqs = raqs_rows[-1]
            raqs["scores"] = json.loads(raqs.pop("scores_json") or "{}")
            raqs["flagged"] = json.loads(raqs.pop("flagged_json") or "[]")

        return {"generation_id": generation_id, "retrievals": retrievals, "raqs": raqs}
    finally:
        conn.close()


def _print_trace(generation_id: str) -> None:
    """CLI / internal-debugging helper: print a human-readable trace summary."""
    trace = get_trace(generation_id)
    if not trace["retrievals"] and not trace["raqs"]:
        print(f"No trace found for generation_id={generation_id!r}")
        return

    print(f"=== Retrieval trace: {generation_id} ===")
    for i, r in enumerate(trace["retrievals"], 1):
        print(f"\n--- Retrieval #{i} ({r['created_at']}) ---")
        print(f"  country: {r['country']}   hanketyyppi_tag: {r['hanketyyppi_tag']}")
        print(f"  query: {r['query_text']}")
        thr = r.get("min_score_threshold")
        thr_note = f"  (country threshold: {thr})" if thr is not None else ""
        print(f"  avg_score: {r['avg_score']}{thr_note}")
        chunks = sorted(r["chunks"], key=lambda c: c["score"], reverse=True)
        print(f"  retrieved sources, ranked by score ({len(chunks)}):")
        for c in chunks:
            print(f"    {c['score']:.3f}  [{c['source_type']}]  {c['chunk_id']}")

    if trace["raqs"]:
        raqs = trace["raqs"]
        print(f"\n--- RAQS outcome ({raqs['created_at']}) ---")
        print(f"  overall: {raqs['overall']}")
        for k, v in raqs["scores"].items():
            if isinstance(v, dict):
                print(f"    {k}: {v.get('pisteet')} — {v.get('perustelu', '')}")
        if raqs["flagged"]:
            print(f"  flagged: {raqs['flagged']}")
    else:
        print("\n(no RAQS outcome recorded for this generation_id)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch a RAG retrieval trace by generation_id (read-only, internal debugging).",
    )
    parser.add_argument("generation_id", help="generation_id / job_id to look up")
    args = parser.parse_args()
    _print_trace(args.generation_id)
