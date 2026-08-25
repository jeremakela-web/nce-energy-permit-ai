"""
retrieval_trace.py — TASO 1: additive tracing + cost-visibility layer for the RAG
generation pipeline.

Purely additive logging alongside the existing generation pipeline. Does NOT alter
source_policy.py's relevance rules and does NOT touch RAQS scoring logic — it only
records what the pipeline already computes, for debugging, provenance ("which
sources did this document draw on?"), and cost/resource visibility.

Storage: SQLite (same pattern already used in this codebase for `raqs_reviews.db`,
see `_RAQS_DB` in generate_application.py — no PostgreSQL exists in this stack, and
no new dependency is added here; sqlite3 is stdlib). File lives on the same
persistent-disk directory as the ChromaDB index / raqs_reviews.db, so it survives
Render deploys.

Five tables:
  retrieval_trace       — one row per `_rag_context()` call (chunk-level detail as JSON)
  retrieval_trace_raqs  — one row per RAQS review, linked by generation_id
  generation_cost       — one row per Claude API call (draft / proofread / RAQS),
                           with token usage + an estimated USD cost
  guardrail_log         — one row per time a per-generation cap (see generate_
                           application.py's GenerationCapError) actually tripped
  generation_checkpoint — one row per major pipeline step (retrieval / draft /
                           proofread / raqs_final), enough state to resume from
                           or discard that step before the human approval gate

`generation_cost` and `guardrail_log` are new tables rather than reuse of
`retrieval_trace`/`retrieval_trace_raqs` — their natural grain is "one row per
Claude API call" / "one row per cap trip", neither of which matches "one row per
RAG retrieval" or "one row per RAQS outcome". They share this module and this
DB file (not a separate module) because they're keyed by the same generation_id
and read together via get_trace() / the same CLI.

`generation_checkpoint` is likewise a new table rather than an ALTER TABLE on
`retrieval_trace`/`retrieval_trace_raqs` — those tables already ship in
production and adding a status column would be a schema migration on live data
for no real benefit. There is deliberate small redundancy: the retrieval and
raqs_final checkpoints re-log state that `retrieval_trace`/`retrieval_trace_raqs`
already capture (chunk IDs/scores, RAQS scores) — cheap (a few KB), and the two
copies serve different consumers (provenance/audit vs. rollback state with a
discard verb). Draft/proofread checkpoints store the actual generated section
text, which has no other persistent home anywhere in this codebase today (not
even the final PDF bytes are stored outside an in-memory job dict) — this is
NOT the same "never store full text" rule as chunk text: that rule exists to
avoid duplicating ChromaDB, which has no equivalent for generated output.

IMPORTANT: full chunk text is never stored here — only chunk IDs, similarity scores,
and lightweight metadata (source_type). Full chunk text stays in ChromaDB, referenced
by chunk_id, to avoid duplicating storage. Likewise, prompt/response text is never
stored here — only token counts and an estimated cost.

This module never raises and never blocks generation — every public writer function
swallows its own exceptions. It also never enforces anything: counting rows and
deciding to fail a generation over a cap is generate_application.py's job (see
GenerationCapError there) — this module only answers "how many so far" and records
the outcome.
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

# ── Model pricing (USD per token) — confirmed live 2026-08-07 against the two
# models this pipeline actually calls (_MODEL_ID = draft/proofread, _MODEL_ID_FAST
# = RAQS review). Keyed by both the alias and the dated ID actually used in
# generate_application.py so a lookup never misses because of which form was
# passed. 5-minute cache-write rate only — this codebase never sets ttl:"1h"
# anywhere (verified: every cache_control in generate_application.py is the
# bare {"type": "ephemeral"} default). Update this table if pricing changes or
# a new model is wired in — it is not fetched live.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    # Claude Sonnet 4.5 — draft generation + proofread
    "claude-sonnet-4-5": {
        "input":                3.00 / 1_000_000,
        "output":              15.00 / 1_000_000,
        "cache_write_5m":       3.75 / 1_000_000,
        "cache_read":           0.30 / 1_000_000,
    },
    "claude-sonnet-4-5-20250929": {
        "input":                3.00 / 1_000_000,
        "output":              15.00 / 1_000_000,
        "cache_write_5m":       3.75 / 1_000_000,
        "cache_read":           0.30 / 1_000_000,
    },
    # Claude Haiku 4.5 — RAQS review
    "claude-haiku-4-5": {
        "input":                1.00 / 1_000_000,
        "output":               5.00 / 1_000_000,
        "cache_write_5m":       1.25 / 1_000_000,
        "cache_read":           0.10 / 1_000_000,
    },
    "claude-haiku-4-5-20251001": {
        "input":                1.00 / 1_000_000,
        "output":               5.00 / 1_000_000,
        "cache_write_5m":       1.25 / 1_000_000,
        "cache_read":           0.10 / 1_000_000,
    },
}


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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generation_cost (
            id                           INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id                TEXT    NOT NULL,
            created_at                   TEXT    NOT NULL,
            call_type                    TEXT    NOT NULL,  -- draft | proofread | raqs
            model                        TEXT    NOT NULL,
            input_tokens                 INTEGER NOT NULL DEFAULT 0,
            cache_creation_input_tokens  INTEGER NOT NULL DEFAULT 0,
            cache_read_input_tokens      INTEGER NOT NULL DEFAULT 0,
            output_tokens                INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd           REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS guardrail_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id  TEXT    NOT NULL,
            created_at     TEXT    NOT NULL,
            guard_type     TEXT    NOT NULL,  -- retrieval_iteration_cap | claude_call_cap
            count_at_trip  INTEGER NOT NULL,
            cap            INTEGER NOT NULL,
            detail         TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generation_checkpoint (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id   TEXT    NOT NULL,
            step            TEXT    NOT NULL,  -- retrieval | draft | proofread | raqs_final
            created_at      TEXT    NOT NULL,
            status          TEXT    NOT NULL DEFAULT 'valid',  -- valid | discarded
            discarded_at    TEXT,
            discard_reason  TEXT,
            state_json      TEXT    NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_retrieval_trace_gen ON retrieval_trace(generation_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_retrieval_trace_raqs_gen ON retrieval_trace_raqs(generation_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_generation_cost_gen ON generation_cost(generation_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_generation_cost_created ON generation_cost(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_guardrail_log_gen ON guardrail_log(generation_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_generation_checkpoint_gen ON generation_checkpoint(generation_id)"
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


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    output_tokens: int = 0,
) -> Optional[float]:
    """Estimate USD cost for one Claude API call from its usage counts.

    Returns None (not 0.0) when `model` isn't in `_MODEL_PRICING` — callers should
    treat None as "unpriced", not "free", so a future new model doesn't silently
    log $0 costs.
    """
    rates = _MODEL_PRICING.get(model)
    if rates is None:
        return None
    return (
        input_tokens                * rates["input"]
        + cache_creation_input_tokens * rates["cache_write_5m"]
        + cache_read_input_tokens     * rates["cache_read"]
        + output_tokens                * rates["output"]
    )


def log_api_call(
    *,
    generation_id: str,
    call_type: str,
    model: str,
    input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record one Claude API call's token usage + estimated cost. Never raises.

    `call_type`: "draft" | "proofread" | "raqs" — matches the three call sites in
    generate_application.py (RAQS runs twice per generation, once per PDF build;
    both are logged as "raqs", not distinguished further).
    """
    if not generation_id:
        return
    try:
        cost = estimate_cost_usd(
            model,
            input_tokens=input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            output_tokens=output_tokens,
        )
        with _lock, _connect() as conn:
            conn.execute(
                "INSERT INTO generation_cost "
                "(generation_id, created_at, call_type, model, input_tokens, "
                " cache_creation_input_tokens, cache_read_input_tokens, output_tokens, "
                " estimated_cost_usd) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    generation_id,
                    datetime.now(timezone.utc).isoformat(),
                    call_type,
                    model,
                    int(input_tokens or 0),
                    int(cache_creation_input_tokens or 0),
                    int(cache_read_input_tokens or 0),
                    int(output_tokens or 0),
                    cost,
                ),
            )
    except Exception:
        pass  # never block generation


def log_guardrail_hit(
    *,
    generation_id: str,
    guard_type: str,
    count_at_trip: int,
    cap: int,
    detail: str = "",
) -> None:
    """Record that a per-generation cap actually tripped. Never raises.

    Enforcement (raising to fail the generation) happens in generate_application.py
    (GenerationCapError) — this only records that it happened, tagged to
    generation_id, per the task requirement.
    """
    if not generation_id:
        return
    try:
        with _lock, _connect() as conn:
            conn.execute(
                "INSERT INTO guardrail_log "
                "(generation_id, created_at, guard_type, count_at_trip, cap, detail) "
                "VALUES (?,?,?,?,?,?)",
                (
                    generation_id,
                    datetime.now(timezone.utc).isoformat(),
                    guard_type,
                    int(count_at_trip),
                    int(cap),
                    detail or "",
                ),
            )
    except Exception:
        pass  # never block generation


_CHECKPOINT_STEPS = ("retrieval", "draft", "proofread", "raqs_final")


def save_checkpoint(
    *,
    generation_id: str,
    step: str,
    state: dict,
) -> Optional[int]:
    """Record a checkpoint for one pipeline step, valid by default. Never raises
    — a checkpointing failure must never block report generation.

    `step` must be one of _CHECKPOINT_STEPS — the four steps where a bad state
    can propagate forward into the next computation (confirmed by investigation:
    the draft-PDF RAQS pass is NOT one of these — it feeds nothing downstream).
    `state` is step-shaped: retrieval -> chunk IDs + metadata (no chunk text,
    same rule as log_retrieval); draft/proofread -> the actual sections dict
    (no other persistent copy exists anywhere); raqs_final -> the scores dict.

    Returns the new row's id (needed for discard_checkpoint), or None if nothing
    was written (empty generation_id, unknown step, or any failure).
    """
    if not generation_id or step not in _CHECKPOINT_STEPS:
        return None
    try:
        with _lock, _connect() as conn:
            cur = conn.execute(
                "INSERT INTO generation_checkpoint "
                "(generation_id, step, created_at, status, state_json) VALUES (?,?,?,?,?)",
                (
                    generation_id,
                    step,
                    datetime.now(timezone.utc).isoformat(),
                    "valid",
                    json.dumps(state, ensure_ascii=False),
                ),
            )
            return cur.lastrowid
    except Exception:
        return None


def get_checkpoints(generation_id: str) -> list[dict]:
    """Read-only: every checkpoint (valid or discarded) for one generation_id,
    step order (retrieval -> draft -> proofread -> raqs_final), each with its
    parsed state.
    """
    if not generation_id:
        return []
    try:
        conn = _connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM generation_checkpoint WHERE generation_id = ? ORDER BY id",
                    (generation_id,),
                )
            ]
            for r in rows:
                r["state"] = json.loads(r.pop("state_json") or "{}")
            return rows
        finally:
            conn.close()
    except Exception:
        return []


def discard_checkpoint(checkpoint_id: int, reason: str = "") -> bool:
    """Explicitly discard one checkpoint by id (soft mark only — the row is never
    deleted, so a future audit can still see a bad step was caught and flagged;
    see the module docstring's rationale). Does not touch any other checkpoint,
    for this generation_id or any other. Never raises — returns True if a row
    was actually updated, False otherwise (no such id, already discarded, or
    any failure).

    This flips status only — it does not trigger any re-run. No automated
    re-run/resume engine exists yet (deliberately out of scope here); discard is
    a signal for a human or a future agent that this step's state should not be
    trusted, not an executable action.
    """
    try:
        with _lock, _connect() as conn:
            cur = conn.execute(
                "UPDATE generation_checkpoint SET status = 'discarded', discarded_at = ?, "
                "discard_reason = ? WHERE id = ? AND status = 'valid'",
                (datetime.now(timezone.utc).isoformat(), reason or "", int(checkpoint_id)),
            )
            return cur.rowcount > 0
    except Exception:
        return False


def count_claude_calls(generation_id: str) -> int:
    """How many Claude API calls (draft + proofread + raqs) already logged for
    this generation_id. Used by generate_application.py's cap check — pure read,
    never raises (returns 0 on any failure, which is the safe direction: a
    tracing failure must never falsely trip the cap and block generation).
    """
    if not generation_id:
        return 0
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM generation_cost WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def count_retrieval_calls(generation_id: str) -> int:
    """How many `_rag_context()` calls already logged for this generation_id.
    Same safe-on-failure contract as count_claude_calls().
    """
    if not generation_id:
        return 0
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM retrieval_trace WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def get_generation_cost(generation_id: str) -> dict:
    """Read-only: every logged Claude API call + total estimated cost for one
    generation_id.
    """
    conn = _connect()
    try:
        conn.row_factory = sqlite3.Row
        calls = [
            dict(row) for row in conn.execute(
                "SELECT * FROM generation_cost WHERE generation_id = ? ORDER BY id",
                (generation_id,),
            )
        ]
        priced = [c["estimated_cost_usd"] for c in calls if c["estimated_cost_usd"] is not None]
        total = round(sum(priced), 6) if priced else (0.0 if calls else None)
        unpriced_calls = sum(1 for c in calls if c["estimated_cost_usd"] is None)
        return {
            "generation_id": generation_id,
            "calls": calls,
            "total_cost_usd": total,
            "unpriced_calls": unpriced_calls,
        }
    finally:
        conn.close()


def get_cost_for_range(start_date: str, end_date: Optional[str] = None) -> dict:
    """Read-only: total estimated cost across all generations in [start_date, end_date]
    (inclusive, both as "YYYY-MM-DD" — compared against `created_at`'s UTC date).
    `end_date` defaults to `start_date` (single-day query).
    """
    end_date = end_date or start_date
    # created_at is stored as an ISO 8601 UTC timestamp (e.g. "2026-08-07T13:27:30...");
    # its first 10 chars are the UTC date, directly comparable to "YYYY-MM-DD" strings.
    conn = _connect()
    try:
        conn.row_factory = sqlite3.Row
        rows = [
            dict(row) for row in conn.execute(
                "SELECT generation_id, call_type, model, estimated_cost_usd, created_at "
                "FROM generation_cost "
                "WHERE substr(created_at, 1, 10) BETWEEN ? AND ? "
                "ORDER BY created_at",
                (start_date, end_date),
            )
        ]
        by_generation: dict[str, float] = {}
        by_call_type: dict[str, float] = {}
        unpriced = 0
        for r in rows:
            cost = r["estimated_cost_usd"]
            if cost is None:
                unpriced += 1
                continue
            by_generation[r["generation_id"]] = by_generation.get(r["generation_id"], 0.0) + cost
            by_call_type[r["call_type"]] = by_call_type.get(r["call_type"], 0.0) + cost
        total = round(sum(by_generation.values()), 6) if by_generation else 0.0
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_cost_usd": total,
            "generation_count": len(by_generation),
            "call_count": len(rows),
            "unpriced_calls": unpriced,
            "by_generation": [
                {"generation_id": gid, "cost_usd": round(c, 6)}
                for gid, c in sorted(by_generation.items(), key=lambda x: -x[1])
            ],
            "by_call_type": {k: round(v, 6) for k, v in by_call_type.items()},
        }
    finally:
        conn.close()


def _fetch_raqs_row(conn: sqlite3.Connection, generation_id: str) -> Optional[dict]:
    """Shared by get_trace() and get_raqs_outcome(): the latest
    retrieval_trace_raqs row for one generation_id, with scores_json/
    flagged_json unpacked into `scores`/`flagged`. None if RAQS never ran
    (or logging failed) for this generation_id. `conn` must already have
    row_factory = sqlite3.Row set.
    """
    raqs_rows = [
        dict(row) for row in conn.execute(
            "SELECT * FROM retrieval_trace_raqs WHERE generation_id = ? ORDER BY id",
            (generation_id,),
        )
    ]
    if not raqs_rows:
        return None
    raqs = raqs_rows[-1]
    raqs["scores"] = json.loads(raqs.pop("scores_json") or "{}")
    raqs["flagged"] = json.loads(raqs.pop("flagged_json") or "[]")
    return raqs


def get_trace(generation_id: str) -> dict:
    """Read-only fetch: all retrieval calls + the latest RAQS outcome + cost +
    any guardrail trips for one generation_id.
    """
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

        raqs = _fetch_raqs_row(conn, generation_id)

        guardrail_hits = [
            dict(row) for row in conn.execute(
                "SELECT * FROM guardrail_log WHERE generation_id = ? ORDER BY id",
                (generation_id,),
            )
        ]

        return {
            "generation_id": generation_id,
            "retrievals": retrievals,
            "raqs": raqs,
            "cost": get_generation_cost(generation_id),
            "guardrail_hits": guardrail_hits,
            "checkpoints": get_checkpoints(generation_id),
        }
    finally:
        conn.close()


def get_raqs_outcome(generation_id: str) -> Optional[dict]:
    """Read-only fetch: just the latest RAQS outcome for one generation_id —
    the same shape as get_trace()'s "raqs" key, without the cost of also
    querying retrievals/guardrail_hits/checkpoints. Added for the customer-
    facing /api/proofread/{job_id} endpoint (backend/main.py), which needs
    only this piece, not the full internal trace get_trace() assembles for
    /api/admin/retrieval-trace.

    Returns None if RAQS never ran (or its logging failed) for this
    generation_id — callers should treat None as "not available yet", not
    as an error.
    """
    if not generation_id:
        return None
    conn = _connect()
    try:
        conn.row_factory = sqlite3.Row
        return _fetch_raqs_row(conn, generation_id)
    finally:
        conn.close()


def _print_trace(generation_id: str) -> None:
    """CLI / internal-debugging helper: print a human-readable trace summary."""
    trace = get_trace(generation_id)
    if not trace["retrievals"] and not trace["raqs"] and not trace["cost"]["calls"]:
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

    cost = trace["cost"]
    print(f"\n--- Cost ({len(cost['calls'])} Claude API call(s)) ---")
    for c in cost["calls"]:
        usd = f"${c['estimated_cost_usd']:.4f}" if c["estimated_cost_usd"] is not None else "unpriced"
        print(
            f"  [{c['call_type']:<9}] {c['model']:<28} "
            f"in={c['input_tokens']} cache_w={c['cache_creation_input_tokens']} "
            f"cache_r={c['cache_read_input_tokens']} out={c['output_tokens']}  {usd}"
        )
    if cost["total_cost_usd"] is not None:
        print(f"  TOTAL: ${cost['total_cost_usd']:.4f}" + (
            f"  ({cost['unpriced_calls']} unpriced call(s) excluded)" if cost["unpriced_calls"] else ""
        ))

    if trace["guardrail_hits"]:
        print(f"\n--- Guardrail trips ({len(trace['guardrail_hits'])}) ---")
        for g in trace["guardrail_hits"]:
            print(f"  {g['created_at']}  {g['guard_type']}: {g['count_at_trip']}/{g['cap']}  {g['detail']}")

    if trace["checkpoints"]:
        print(f"\n--- Checkpoints ({len(trace['checkpoints'])}) ---")
        for c in trace["checkpoints"]:
            tag = f"[{c['status'].upper()}]"
            print(f"  id={c['id']}  {tag}  step={c['step']}  ({c['created_at']})")
            if c["status"] == "discarded":
                print(f"      discarded_at={c['discarded_at']}  reason={c['discard_reason']!r}")
            state_preview = json.dumps(c["state"], ensure_ascii=False)
            if len(state_preview) > 200:
                state_preview = state_preview[:200] + "…"
            print(f"      state: {state_preview}")


def _print_cost_range(start_date: str, end_date: Optional[str]) -> None:
    """CLI helper: print a human-readable cost summary for a date range."""
    r = get_cost_for_range(start_date, end_date)
    print(f"=== Cost report: {r['start_date']} to {r['end_date']} ===")
    print(f"  total: ${r['total_cost_usd']:.4f}  across {r['generation_count']} generation(s), "
          f"{r['call_count']} Claude API call(s)")
    if r["unpriced_calls"]:
        print(f"  ({r['unpriced_calls']} unpriced call(s) excluded from total)")
    if r["by_call_type"]:
        print("  by call type:")
        for k, v in sorted(r["by_call_type"].items(), key=lambda x: -x[1]):
            print(f"    {k}: ${v:.4f}")
    if r["by_generation"]:
        print("  by generation (highest cost first):")
        for g in r["by_generation"]:
            print(f"    {g['generation_id']}: ${g['cost_usd']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrieval-trace / cost-visibility CLI (read-only, internal debugging).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_trace = sub.add_parser("trace", help="Fetch a full trace (retrieval + RAQS + cost) by generation_id.")
    p_trace.add_argument("generation_id")

    p_cost = sub.add_parser("cost", help="Total cost for a date (or date range).")
    p_cost.add_argument("start_date", help="YYYY-MM-DD")
    p_cost.add_argument("end_date", nargs="?", default=None, help="YYYY-MM-DD (defaults to start_date)")

    p_discard = sub.add_parser("discard", help="Discard (soft-mark invalid) one checkpoint by id.")
    p_discard.add_argument("checkpoint_id", type=int)
    p_discard.add_argument("reason", nargs="?", default="", help="Why this checkpoint is being discarded")

    args = parser.parse_args()
    if args.cmd == "trace":
        _print_trace(args.generation_id)
    elif args.cmd == "cost":
        _print_cost_range(args.start_date, args.end_date)
    elif args.cmd == "discard":
        ok = discard_checkpoint(args.checkpoint_id, args.reason)
        print(f"discarded checkpoint id={args.checkpoint_id}: {'OK' if ok else 'FAILED (no such valid checkpoint id?)'}")
