"""
manual_source_freshness.py — TASO 1: read-only staleness report over the
`source_type:"manual"` + `last_verified` metadata added by PR #50.

Purely additive, purely read-only. Does not touch source_policy.py, RAQS
scoring, chunk tagging/backfill logic (PR #50), or retrieval_trace.py/the cost
guardrail/rollback checkpoints (PR #47/#48/#49) — this reads ChromaDB metadata
directly and has nothing to do with any of those.

No new storage: this queries `source_type`/`last_verified` chunk metadata live
on every call. There is no event to log (no generation_id, no per-call state)
and nothing to persist between runs, so — confirmed before implementing, same
as the original layer-4 task's reasoning before it was paused — a SQLite table
would add nothing here. If a future need arises to track when a reminder was
last shown to a human, that would be a real, separate reason to add one; it
doesn't exist today.

Staleness buckets (confirmed 2026-08-08, plain day-counts, not calendar months):
    overdue   : days_since_last_verified > 365  (past 12 months)
    due_soon  : 270 < days_since_last_verified <= 365  (9-12 months)
    fresh     : days_since_last_verified <= 270  (under 9 months)

This is visibility only. Acting on a reminder means a human re-runs the normal
tagging (lithuania_ingestion.py / ingest_fi_env.py / ingest_datakeskus.py /
backfill_manual_source_tags.py, per PR #50) — no automated re-fetch mechanism
exists or is added here.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Optional

_HERE   = os.path.dirname(os.path.abspath(__file__))
_DB_DIR = os.path.join(_HERE, "embeddings")

_COLLECTIONS = ["permit_docs", "permit_docs_v2"]

OVERDUE_DAYS  = 365  # past 12 months
DUE_SOON_DAYS = 270  # 9 months — the due-soon window is (DUE_SOON_DAYS, OVERDUE_DAYS]


def _bucket(days_since: int) -> str:
    if days_since > OVERDUE_DAYS:
        return "overdue"
    if days_since > DUE_SOON_DAYS:
        return "due_soon"
    return "fresh"


def _days_since(last_verified: str, today: date) -> Optional[int]:
    """Parse a 'YYYY-MM-DD' last_verified value. Returns None (not 0) for a
    missing/malformed value — callers must treat that as "unknown", not "fresh",
    so a tagging bug can never silently hide a chunk from the report.
    """
    try:
        d = datetime.strptime(last_verified, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    return (today - d).days


def _connect_collection(name: str):
    import chromadb
    client = chromadb.PersistentClient(path=_DB_DIR)
    return client.get_collection(name)  # raises if it doesn't exist — caller handles


def get_manual_source_report(*, today: Optional[date] = None) -> dict:
    """Read-only: every source_type='manual' chunk across both ChromaDB
    collections, grouped by (country, source), bucketed by staleness.

    `today` is injectable for testing the bucket boundaries against a synthetic
    date without needing to wait months or fake a chunk's last_verified value —
    production callers should never pass it (defaults to real UTC today).

    Never writes anything — every ChromaDB call here is a read (`.get()`).
    """
    today = today or datetime.now(timezone.utc).date()

    # source_name -> aggregated state. Two chunks of the same source can live in
    # both collections (see PR #50's lt_pav_istatymas) — collapse them into one
    # row per source, since a human acts on "the source", not "the collection".
    by_source: dict[tuple[str, str], dict] = {}  # (country, source) -> {...}
    unknown_date_chunks = 0

    for cname in _COLLECTIONS:
        try:
            col = _connect_collection(cname)
        except Exception:
            continue  # collection doesn't exist locally — not an error, just absent
        got = col.get(where={"source_type": "manual"}, include=["metadatas"])
        for chunk_id, meta in zip(got["ids"], got["metadatas"]):
            country = meta.get("country") or "?"
            source  = meta.get("source") or "?"
            key = (country, source)
            row = by_source.setdefault(key, {
                "country": country,
                "source": source,
                "chunk_count": 0,
                "last_verified": None,
                "last_verified_inconsistent": False,
                "collections": set(),
            })
            row["chunk_count"] += 1
            row["collections"].add(cname)
            lv = meta.get("last_verified")
            if row["last_verified"] is None:
                row["last_verified"] = lv
            elif row["last_verified"] != lv:
                # A source's chunks should all share one last_verified value
                # (set once per ingestion run) — if they don't, that's a real
                # data anomaly worth surfacing, not silently averaging over.
                row["last_verified_inconsistent"] = True
                # Keep the older (more conservative) of the two for bucketing.
                d1 = _days_since(row["last_verified"], today)
                d2 = _days_since(lv, today)
                if d2 is not None and (d1 is None or d2 > d1):
                    row["last_verified"] = lv
            if lv is None:
                unknown_date_chunks += 1

    buckets: dict[str, list[dict]] = {"overdue": [], "due_soon": [], "fresh": [], "unknown": []}
    by_country: dict[str, dict[str, list[dict]]] = {}

    for (country, source), row in sorted(by_source.items()):
        days_since = _days_since(row["last_verified"], today)
        bucket = "unknown" if days_since is None else _bucket(days_since)
        entry = {
            "country": row["country"],
            "source": row["source"],
            "chunk_count": row["chunk_count"],
            "last_verified": row["last_verified"],
            "days_since_verified": days_since,
            "bucket": bucket,
            "collections": sorted(row["collections"]),
            "last_verified_inconsistent": row["last_verified_inconsistent"],
        }
        buckets[bucket].append(entry)
        by_country.setdefault(country, {"overdue": [], "due_soon": [], "fresh": [], "unknown": []})
        by_country[country][bucket].append(entry)

    total_chunks = sum(r["chunk_count"] for r in by_source.values())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": today.isoformat(),
        "thresholds": {"overdue_days": OVERDUE_DAYS, "due_soon_days": DUE_SOON_DAYS},
        "total_manual_chunks": total_chunks,
        "total_sources": len(by_source),
        "unknown_date_chunks": unknown_date_chunks,
        "summary": {
            b: {"sources": len(entries), "chunks": sum(e["chunk_count"] for e in entries)}
            for b, entries in buckets.items()
        },
        "by_bucket": buckets,
        "by_country": by_country,
    }


def _print_report(report: dict) -> None:
    """CLI / internal-debugging helper: human-readable staleness report."""
    print(f"=== Manual-source freshness report (as of {report['as_of_date']}) ===")
    print(f"  thresholds: overdue > {report['thresholds']['overdue_days']}d, "
          f"due_soon > {report['thresholds']['due_soon_days']}d")
    print(f"  {report['total_manual_chunks']} manual chunks across {report['total_sources']} sources")
    if report["unknown_date_chunks"]:
        print(f"  ⚠ {report['unknown_date_chunks']} chunk(s) with missing/malformed last_verified")

    print("\n--- Summary ---")
    for b in ["overdue", "due_soon", "fresh", "unknown"]:
        s = report["summary"][b]
        if s["sources"]:
            print(f"  {b:<9}: {s['sources']} source(s), {s['chunks']} chunk(s)")

    print("\n--- By country ---")
    for country, buckets in sorted(report["by_country"].items()):
        counts = {b: len(entries) for b, entries in buckets.items() if entries}
        print(f"\n[{country}]")
        for b in ["overdue", "due_soon", "fresh", "unknown"]:
            entries = buckets[b]
            if not entries:
                continue
            print(f"  {b} ({len(entries)} source(s)):")
            for e in entries:
                flag = "  ⚠ inconsistent last_verified across chunks" if e["last_verified_inconsistent"] else ""
                print(f"    {e['source']}: {e['chunk_count']} chunks, "
                      f"last_verified={e['last_verified']} "
                      f"({e['days_since_verified']} days ago){flag}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Read-only manual-source freshness report (internal debugging).",
    )
    parser.add_argument(
        "--as-of", metavar="YYYY-MM-DD",
        help="Evaluate staleness as of this date instead of today (for testing bucket boundaries).",
    )
    args = parser.parse_args()

    as_of = None
    if args.as_of:
        as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()

    report = get_manual_source_report(today=as_of)
    _print_report(report)
