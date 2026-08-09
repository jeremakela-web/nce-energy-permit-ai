"""
source_drift.py — TASO 1 (PR A): domain/regulatory-drift detection. Detects when
a source document that chunks were originally ingested from has since changed on
the authority's website, so stale chunks/sources can be flagged for re-review.

Design confirmed with user 2026-08-09, following the same investigation pattern
as PR #48/#49/#50/#51 (report real numbers before building):

  - Real corpus survey (production, both collections): only ~2861 chunks across
    ~91 sources carry a structured `url` metadata field; another 64 chunks across
    64 sources embed a "SOURCE: <url>" line in the chunk text instead (no url
    metadata) — these two groups are this module's checkable surface, roughly
    100 unique (source, url) pairs across ~15-21 domains. The remaining ~160+
    sources are static uploaded documents (IAEA/Polish/Latvian national reports
    etc.) with no live URL recorded at all — EXPLICITLY OUT OF SCOPE here (see
    `excluded_sourceless_count` in check_all_sources()'s return value — reported,
    not silently dropped).
  - Check by unique (source, url) pair, not per-chunk: many chunks share one
    source URL (e.g. IAEA's 586 chunks are a handful of PDF URLs), so per-chunk
    fetching would be ~30x more requests than the problem actually needs.
  - No cron/worker in this deployment (render.yaml: only redis + web; a second
    worker is blocked because the ChromaDB persistent disk can only mount on one
    service — see render.yaml's own comment). So this is admin-triggered only,
    same as every other admin endpoint in this stack (PR #47-51's x-admin-secret
    pattern) — POST runs real fetches, GET reads the last-known status for free.
  - Flags at SOURCE level, not per-chunk (matches PR #51's freshness dashboard
    grouping) and NOT per-generation (that cross-reference against PR #49's
    checkpoints was explicitly deferred as a stretch goal, out of scope here).
  - Three-state result per check: changed / unchanged / check_failed. A blocked
    fetch (Finlex is fully JS-rendered, e-seimas.lrs.lt blocks automated fetches
    — both per PR #50's own investigation notes) must show up as check_failed,
    never silently coerced into either changed or unchanged.
  - A 4th, internal-only status "baseline" exists for a (source, url) pair's very
    first successful check: there is nothing yet stored to diff against, so
    "unchanged" would be a claim we never actually verified. This is a deliberate
    documented deviation from the strict three-state description — every check
    AFTER the first one for a given pair is a real three-way changed/unchanged/
    check_failed result, which is what matters for ongoing drift detection.

Own module, own SQLite file (permit_ai/embeddings/source_drift.db) — does not
import or modify source_policy.py, the RAQS logic, or any of PRs #47-51's code
or tables (retrieval_trace.py is untouched; a small helper to open a ChromaDB
collection is duplicated here rather than imported, to keep this module fully
independent).

Usage (CLI, internal debugging):
    python3 permit_ai/source_drift.py [--source NAME] [--dry-run-targets]
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

_HERE      = os.path.dirname(os.path.abspath(__file__))
_DB_DIR    = os.path.join(_HERE, "embeddings")
_DRIFT_DB  = os.path.join(_DB_DIR, "source_drift.db")

_COLLECTIONS = ["permit_docs", "permit_docs_v2"]

_SOURCE_LINE_RE = re.compile(r"^SOURCE:\s*(\S+)", re.MULTILINE)

_HTTP_TIMEOUT     = 20.0
_MAX_CONCURRENCY  = 5   # be a reasonable citizen — these are ~15-21 gov/authority domains
_USER_AGENT = "Mozilla/5.0 (compatible; NCEPermitAI-SourceDriftCheck/1.0)"


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_drift_check (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            source         TEXT    NOT NULL,
            url            TEXT    NOT NULL,
            checked_at     TEXT    NOT NULL,
            status         TEXT    NOT NULL,  -- changed | unchanged | check_failed | baseline
            http_status    INTEGER,
            content_kind   TEXT,              -- 'pdf' | 'html' — how it was hashed
            content_hash   TEXT,              -- sha256 hex; NULL if the fetch failed
            previous_hash  TEXT,              -- what content_hash was compared against (NULL for baseline/check_failed)
            error_detail   TEXT               -- exception / HTTP-status detail; NULL if ok
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_drift_check_source ON source_drift_check(source)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_drift_check_checked_at ON source_drift_check(checked_at)"
    )


def _connect() -> sqlite3.Connection:
    os.makedirs(_DB_DIR, exist_ok=True)
    conn = sqlite3.connect(_DRIFT_DB)
    _init_db(conn)
    return conn


def _last_successful_hash(conn: sqlite3.Connection, source: str) -> Optional[str]:
    row = conn.execute(
        "SELECT content_hash FROM source_drift_check "
        "WHERE source = ? AND content_hash IS NOT NULL "
        "ORDER BY id DESC LIMIT 1",
        (source,),
    ).fetchone()
    return row[0] if row else None


def _record_check(conn: sqlite3.Connection, *, source: str, url: str, status: str,
                   http_status: Optional[int], content_kind: Optional[str],
                   content_hash: Optional[str], previous_hash: Optional[str],
                   error_detail: Optional[str]) -> None:
    conn.execute(
        "INSERT INTO source_drift_check "
        "(source, url, checked_at, status, http_status, content_kind, content_hash, previous_hash, error_detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source, url, datetime.now(timezone.utc).isoformat(), status, http_status,
         content_kind, content_hash, previous_hash, error_detail),
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Target discovery — which (source, url) pairs are even checkable
# ─────────────────────────────────────────────────────────────────────────────

def _connect_collection(name: str):
    import chromadb
    client = chromadb.PersistentClient(path=_DB_DIR)
    return client.get_collection(name)  # raises if it doesn't exist — caller handles


def discover_check_targets() -> tuple[dict[str, dict], int, list[str]]:
    """Read-only: scan both ChromaDB collections' metadata (+ text, for the
    inline-SOURCE: fallback) to build the checkable (source -> {url, country,
    hanketyyppi_tag}) map.

    Returns (targets, excluded_sourceless_count, inconsistent_sources).
    `excluded_sourceless_count` is the number of distinct sources that have
    neither a structured `url` field nor an inline "SOURCE: <url>" text line —
    these are static uploaded documents with no live page to check at all, and
    are deliberately excluded, not silently dropped (see module docstring).
    """
    targets: dict[str, dict] = {}
    all_sources: set[str] = set()
    sources_with_link: set[str] = set()
    inconsistent: list[str] = []

    for cname in _COLLECTIONS:
        try:
            col = _connect_collection(cname)
        except Exception:
            continue
        got = col.get(include=["metadatas", "documents"])
        for meta, doc in zip(got["metadatas"], got["documents"]):
            meta = meta or {}
            source = meta.get("source")
            if not source:
                continue
            all_sources.add(source)

            url = meta.get("url")
            if not url and doc:
                m = _SOURCE_LINE_RE.search(doc[:300])
                if m:
                    url = m.group(1)

            if not url:
                continue
            sources_with_link.add(source)

            existing = targets.get(source)
            if existing is None:
                targets[source] = {
                    "url": url,
                    "country": meta.get("country"),
                    "hanketyyppi_tag": meta.get("hanketyyppi_tag"),
                }
            elif existing["url"] != url:
                if source not in inconsistent:
                    inconsistent.append(source)
                # Keep the first URL seen — a source disagreeing with itself on
                # URL is a real anomaly worth surfacing, not silently averaging.

    excluded_sourceless_count = len(all_sources - sources_with_link)
    return targets, excluded_sourceless_count, sorted(inconsistent)


# ─────────────────────────────────────────────────────────────────────────────
# Fetch + hash
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(html: str) -> str:
    """Lightweight main-content extraction — same simple heuristic as
    ingest_web.py (find main/article/content, else body), not reproduced from
    it verbatim to keep this module dependency-free of the ingestion scripts.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    main = (soup.find("main") or soup.find("article") or soup.find(id="content")
            or soup.find(class_=re.compile(r"content|main|article", re.I)))
    target = main if main else (soup.body if soup.body else soup)
    text = target.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


async def _fetch_and_hash(client, url: str) -> dict:
    """One fetch attempt. Never raises — every failure mode (timeout, DNS,
    connection refused, non-2xx status, TLS error) comes back as ok=False with
    an error_detail string, so the caller can record check_failed instead of
    guessing.
    """
    try:
        resp = await client.get(
            url, timeout=_HTTP_TIMEOUT, follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
    except Exception as exc:
        return {"ok": False, "http_status": None, "content_kind": None,
                "content_hash": None, "error": f"{type(exc).__name__}: {exc}"}

    if resp.status_code >= 400:
        return {"ok": False, "http_status": resp.status_code, "content_kind": None,
                "content_hash": None, "error": f"HTTP {resp.status_code}"}

    content_type = (resp.headers.get("content-type") or "").lower()
    is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")

    try:
        if is_pdf:
            content_kind = "pdf"
            content_hash = hashlib.sha256(resp.content).hexdigest()
        else:
            content_kind = "html"
            text = _extract_text(resp.text)
            content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    except Exception as exc:
        # Fetch succeeded but we couldn't make sense of the body — still a
        # real check_failed, not a false "changed".
        return {"ok": False, "http_status": resp.status_code, "content_kind": None,
                "content_hash": None, "error": f"parse error: {type(exc).__name__}: {exc}"}

    return {"ok": True, "http_status": resp.status_code, "content_kind": content_kind,
            "content_hash": content_hash, "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

async def check_all_sources(sources: Optional[list[str]] = None) -> dict:
    """Runs real fetches. Admin-triggered only — no scheduler calls this.

    `sources`: optional list of exact source names to check (subset), instead
    of every discovered target. Unknown names are reported, not silently
    ignored.
    """
    import asyncio
    import httpx

    targets, excluded_sourceless_count, inconsistent = discover_check_targets()

    unknown_requested: list[str] = []
    if sources:
        wanted = set(sources)
        unknown_requested = sorted(wanted - set(targets))
        targets = {s: t for s, t in targets.items() if s in wanted}

    conn = _connect()
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)
    results: list[dict] = []

    async def _run_one(client, source: str, info: dict) -> None:
        url = info["url"]
        async with sem:
            fetch = await _fetch_and_hash(client, url)

        checked_at = datetime.now(timezone.utc).isoformat()
        if not fetch["ok"]:
            status = "check_failed"
            _record_check(conn, source=source, url=url, status=status,
                           http_status=fetch["http_status"], content_kind=None,
                           content_hash=None, previous_hash=None, error_detail=fetch["error"])
        else:
            prev = _last_successful_hash(conn, source)
            if prev is None:
                status = "baseline"
            elif prev == fetch["content_hash"]:
                status = "unchanged"
            else:
                status = "changed"
            _record_check(conn, source=source, url=url, status=status,
                           http_status=fetch["http_status"], content_kind=fetch["content_kind"],
                           content_hash=fetch["content_hash"], previous_hash=prev, error_detail=None)

        results.append({
            "source": source, "url": url,
            "country": info.get("country"), "hanketyyppi_tag": info.get("hanketyyppi_tag"),
            "status": status, "http_status": fetch["http_status"],
            "content_kind": fetch.get("content_kind"), "checked_at": checked_at,
            "error": fetch.get("error"),
        })

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(_run_one(client, s, info) for s, info in targets.items()))
    conn.close()

    results.sort(key=lambda r: r["source"])
    summary: dict[str, int] = {}
    for r in results:
        summary[r["status"]] = summary.get(r["status"], 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets_checked": len(results),
        "excluded_sourceless_count": excluded_sourceless_count,
        "inconsistent_url_sources": inconsistent,
        "unknown_requested_sources": unknown_requested,
        "summary": summary,
        "results": results,
    }


def get_latest_drift_status() -> dict:
    """Read-only, no fetching: the most recent check result per source that has
    EVER been checked (i.e. read from source_drift_check history only — does
    NOT re-scan ChromaDB for new targets). Cheap; safe to call often.
    """
    conn = _connect()
    rows = conn.execute("""
        SELECT t1.source, t1.url, t1.checked_at, t1.status, t1.http_status,
               t1.content_kind, t1.error_detail
        FROM source_drift_check t1
        INNER JOIN (
            SELECT source, MAX(id) AS max_id FROM source_drift_check GROUP BY source
        ) t2 ON t1.source = t2.source AND t1.id = t2.max_id
        ORDER BY t1.source
    """).fetchall()
    conn.close()

    entries = [
        {"source": r[0], "url": r[1], "checked_at": r[2], "status": r[3],
         "http_status": r[4], "content_kind": r[5], "error": r[6]}
        for r in rows
    ]
    summary: dict[str, int] = {}
    for e in entries:
        summary[e["status"]] = summary.get(e["status"], 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources_ever_checked": len(entries),
        "summary": summary,
        "sources": entries,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI (internal debugging)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import asyncio as _asyncio
    import json as _json

    parser = argparse.ArgumentParser(description="Source drift check (internal debugging).")
    parser.add_argument("--source", action="append", help="Check only this source (repeatable)")
    parser.add_argument("--dry-run-targets", action="store_true",
                         help="Only discover and print checkable targets — no fetching")
    args = parser.parse_args()

    if args.dry_run_targets:
        targets, excluded, inconsistent = discover_check_targets()
        print(f"Checkable (source, url) pairs: {len(targets)}")
        print(f"Excluded (no url at all, static docs): {excluded}")
        if inconsistent:
            print(f"⚠ Sources with inconsistent url across chunks: {inconsistent}")
        for s, info in sorted(targets.items()):
            print(f"  {s}: {info['url']}")
    else:
        report = _asyncio.run(check_all_sources(sources=args.source))
        print(_json.dumps(report, indent=2, ensure_ascii=False))
