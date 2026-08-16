"""
FastAPI-backend BESS-kaavoituskartoitustyökalulle.
Pöytyä, kiinteistötunnus 636-439-4-711.

Käynnistys:
    cd bess_tool/backend && uvicorn main:app --reload --port 8000
"""

# TODO: domain muutos ncepermit.ai kun NCE Global perustettu

import asyncio
import base64
import dataclasses
import email.mime.multipart
import email.mime.text
import io
import json
import logging
import os
import re
import secrets
import smtplib
import time
import unicodedata
import uuid
from collections import defaultdict
from threading import Thread, Lock
from typing import Optional

import requests as _requests

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from mml_api import (
    get_land_use, get_natura_areas, get_property_boundaries,
    get_zoning_info, get_groundwater_areas, infer_zoning_from_osm,
    get_flood_risk,
)
from finnish_authorities import get_pelastuslaitos, get_ely, genitive
from fingrid_api import (
    get_transmission_lines, get_buildings, get_highways, get_substations,
    nearest_line_distance_m, nearest_point_distance_m, nearest_substation_info,
)
from heritage_api import get_heritage_sites
from gtk_api import get_soil_type
from ai_strategy import get_lupaprosessi_strategy
from report import generate_bess_report
from permit_ai import query_permit_ai, query_permit_ai_chat
import permit_ai as _permit_ai_module
import rtb_store as _rtb

# permit_ai-moduuli on ~/bess_tool/permit_ai/ — lisätään polkuun
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "permit_ai"))
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate_application import (
    generate_application, generate_application_draft, apply_proofread_to_pdf,
    ApplicationInput, _get_embed_model, _get_chroma_col,
    InsufficientSourcesError, GenerationCapError,
)
import generate_application as _gen_app_module
import retrieval_trace as _retrieval_trace
import manual_source_freshness as _manual_source_freshness
import source_drift as _source_drift
from tenant_endpoints import router as _tenant_router
try:
    from optimizer import NCEOptimizer, EnergySite
    _OPTIMIZER_OK = True
except ImportError:
    _OPTIMIZER_OK = False
    print("[startup] optimizer.py ei löydy — /api/optimize-bess palauttaa 501")

from entsoe_prices import refresh_all_prices as _refresh_entsoe_prices

# ── V2 re-index constants ──────────────────────────────────────────────────────
_V2_COL        = "permit_docs_v2"
_V2_MODEL      = "paraphrase-multilingual-mpnet-base-v2"
_V2_MIN_CHUNKS = 200            # buildCommand produces ~300-600 FI chunks; background reindex produces ~10k
_DB_PATH       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings"))
_reindex_log   = logging.getLogger("reindex")


def _v2_is_ready() -> bool:
    """Return True if permit_docs_v2 exists and has enough chunks."""
    try:
        import chromadb as _chroma
        c = _chroma.PersistentClient(path=_DB_PATH)
        col = c.get_collection(_V2_COL)
        return col.count() >= _V2_MIN_CHUNKS
    except Exception:
        return False


def _activate_all_v2() -> None:
    """Switch both RAG modules to V2 collection + mpnet model (no restart needed)."""
    _permit_ai_module.activate_v2()
    _gen_app_module.activate_v2()
    logging.getLogger("startup").info(
        "[rag] Switched to permit_docs_v2 + paraphrase-multilingual-mpnet-base-v2 (768-dim)"
    )


def _run_background_reindex() -> None:
    """
    Background thread: re-embeds all chunks from permit_docs → permit_docs_v2
    using paraphrase-multilingual-mpnet-base-v2 (768-dim, multilingual, 512-tok).
    Logs progress every 500 chunks. Calls _activate_all_v2() on completion.
    On error: logs and exits — app keeps serving from V1 collection.
    """
    import warnings
    warnings.filterwarnings("ignore")

    _reindex_log.info("[reindex] Starting background re-index → permit_docs_v2 (mpnet 768-dim)")

    try:
        import chromadb as _chroma
        from sentence_transformers import SentenceTransformer

        _reindex_log.info(f"[reindex] Loading model: {_V2_MODEL}")
        model = SentenceTransformer(_V2_MODEL)
        _reindex_log.info(f"[reindex] Model loaded, dim={model.get_sentence_embedding_dimension()}")

        client = _chroma.PersistentClient(path=_DB_PATH)
        src    = client.get_collection("permit_docs")
        total  = src.count()
        _reindex_log.info(f"[reindex] Source: {total} chunks in permit_docs")

        # Delete and recreate target (handles partial previous runs)
        try:
            client.delete_collection(_V2_COL)
            _reindex_log.info(f"[reindex] Deleted partial '{_V2_COL}'")
        except Exception:
            pass

        tgt = client.create_collection(
            name=_V2_COL,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

        PAGE_SIZE   = 500
        EMBED_BATCH = 32
        WRITE_BATCH = 500

        offset      = 0
        total_added = 0
        errors: list = []
        t0 = time.time()

        while offset < total:
            page  = src.get(limit=PAGE_SIZE, offset=offset,
                            include=["documents", "metadatas"])
            ids   = page["ids"]
            docs  = page["documents"]
            metas = page["metadatas"]
            if not ids:
                break

            embs = model.encode(docs, batch_size=EMBED_BATCH,
                                show_progress_bar=False, normalize_embeddings=True)

            for i in range(0, len(ids), WRITE_BATCH):
                try:
                    tgt.add(
                        ids       = ids[i:i+WRITE_BATCH],
                        documents = docs[i:i+WRITE_BATCH],
                        metadatas = metas[i:i+WRITE_BATCH],
                        embeddings= embs[i:i+WRITE_BATCH].tolist(),
                    )
                    total_added += len(ids[i:i+WRITE_BATCH])
                except Exception as e:
                    errors.append(str(e))
                    _reindex_log.warning(f"[reindex] Write error at offset {offset+i}: {e}")

            elapsed = time.time() - t0
            rate    = total_added / elapsed if elapsed > 0 else 0
            eta     = (total - total_added) / rate if rate > 0 else 0
            _reindex_log.info(
                f"[reindex] {total_added}/{total} chunks "
                f"({100*total_added//total}%) | "
                f"{elapsed/60:.0f}min elapsed | "
                f"ETA {eta/60:.0f}min"
            )

            offset += len(ids)
            if len(ids) < PAGE_SIZE:
                break

        final_count = tgt.count()
        elapsed_total = time.time() - t0
        _reindex_log.info(
            f"[reindex] Done: {final_count}/{total} chunks in "
            f"{elapsed_total/60:.1f}min, errors={len(errors)}"
        )

        if final_count >= _V2_MIN_CHUNKS:
            _activate_all_v2()
            _reindex_log.info("[reindex] ✓ Auto-switched to permit_docs_v2 — no restart needed")
        else:
            _reindex_log.error(
                f"[reindex] ✗ Only {final_count} chunks written (need {_V2_MIN_CHUNKS}) — "
                "staying on V1 collection"
            )

    except Exception as exc:
        _reindex_log.exception(f"[reindex] Fatal error: {exc} — app continues on V1 collection")


def _db_needs_index() -> bool:
    """
    Return True if the embeddings directory has no ChromaDB data.

    Primary check: count rows in chroma.sqlite3 directly. This is reliable
    across ChromaDB versions regardless of whether binary HNSW segment dirs
    exist — ChromaDB 1.5.x stores all data in SQLite; UUID subdirs may not
    be present after Shell-based ingestion or partial rebuilds.

    Fallback: UUID subdir check (legacy behaviour, kept for safety).
    """
    import sqlite3 as _sql
    db_file = os.path.join(_DB_PATH, "chroma.sqlite3")
    if not os.path.exists(db_file):
        return True
    try:
        _con = _sql.connect(db_file, check_same_thread=False)
        count = _con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        _con.close()
        return count == 0
    except Exception:
        # SQLite unreadable — fall back to UUID subdir check
        from pathlib import Path as _Path
        db_dir = _Path(_DB_PATH)
        if not db_dir.exists():
            return True
        return not any(child.is_dir() for child in db_dir.iterdir())


def _run_startup_fallback_index() -> None:
    """
    Background thread: build the FI index when the DB is empty at startup.

    CRITICAL INVARIANT: this thread must be started only when NO ChromaDB
    PersistentClient for DB_PATH has been created yet in this process.
    build_index.build() calls shutil.rmtree internally; if any live client
    exists before the rmtree the module-level segment state becomes stale
    and subsequent query() calls return 0 even though count() returns 886.

    After build() completes we clear lru_caches (evicting any clients that
    may have been created by requests arriving mid-build) then create the
    single definitive client that all subsequent queries will use.
    """
    _log = logging.getLogger("startup-fallback")
    try:
        import build_index as _build_index
        _log.info("[startup-fallback] Building FI index from permit_ai/docs/")
        try:
            _build_index.build()
        except SystemExit as exc:
            if exc.code != 0:
                _log.error(f"[startup-fallback] build_index.build() exited {exc.code} — staying empty")
                return
        # Evict any stale clients created by requests that arrived mid-build
        _get_chroma_col.cache_clear()
        _get_embed_model.cache_clear()
        _permit_ai_module._get_collection.cache_clear()
        _permit_ai_module._get_embed_model.cache_clear()
        count = _get_chroma_col().count()
        _log.info(f"[startup-fallback] Done — {count} chunks in permit_docs, ready for queries")
        print(f"[startup-fallback] ✓ {count} chunkkia indeksoitu — RAG valmis (malli ladataan laiskasti ensimmäisellä kyselyllä)")
    except Exception as exc:
        logging.getLogger("startup-fallback").exception(f"[startup-fallback] Unexpected error: {exc}")


# Startup: check DB state with filesystem ops first, open ChromaDB only after
# any necessary rebuild — so there is never a live client during rmtree.
try:
    if _db_needs_index():
        # No collection on disk yet. Start the fallback indexer in background
        # WITHOUT opening any ChromaDB client here — the thread will create the
        # single definitive client after build() completes.
        print("[startup] permit_docs tyhjä — käynnistetään taustalla FI-indeksointi")
        Thread(target=_run_startup_fallback_index, daemon=True, name="startup-fallback").start()
    else:
        # DB has collections — safe to open clients, check V2, and warm up.
        if _v2_is_ready():
            _activate_all_v2()
            logging.getLogger("startup").info("[startup] permit_docs_v2 ready — using mpnet 768-dim")
        elif os.getenv("ENABLE_REINDEX", "").lower() == "true":
            logging.getLogger("startup").info(
                "[startup] permit_docs_v2 not ready — background V2 reindex starts in 5s"
            )
            def _delayed_reindex():
                time.sleep(5)
                _run_background_reindex()
            Thread(target=_delayed_reindex, daemon=True, name="reindex-v2").start()
        else:
            logging.getLogger("startup").info(
                "[startup] permit_docs_v2 not ready — set ENABLE_REINDEX=true to enable"
            )
        count = _get_chroma_col().count()
        print(f"[startup] ChromaDB ladattu ({count} chunkkia) — malli ladataan laiskasti ensimmäisellä kyselyllä")
except Exception as _e:
    print(f"[startup] Varoitus: RAG-lataus epäonnistui: {_e}")

# Payment + B2B key DB init (NOOP when respective env vars are false)
try:
    from stripe_payments import init_db as _init_payments_db
    from api_keys import init_api_keys_db as _init_api_keys_db
    _init_payments_db()
    _init_api_keys_db()
except Exception as _e:
    print(f"[startup] Payments/API-keys init: {_e}")

# LinkedIn agent DB init
try:
    from linkedin_agent import init_post_db as _init_post_db
    _init_post_db()
except Exception as _e:
    print(f"[startup] LinkedIn post DB init: {_e}")

limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])

app = FastAPI(
    title="BESS-kaavoituskartoitus API",
    description="Pöytyä 636-439-4-711 – akkuvarastohankkeen sijaintianalyysi",
    version="2.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# PR A (tenant architecture) — session cookies for the new magic-link auth
# surface. NOTE: CORS above is wildcard-origin with no allow_credentials, so
# per the CORS spec a browser will only actually deliver this session cookie
# on SAME-origin requests (the app's own frontend calling its own API on
# ai.ncenergy.fi) — cross-origin API consumers won't get a working session
# here. That's an existing CORS posture, not something changed by this PR;
# flagged for awareness, not fixed as part of it.
_SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "")
if not _SESSION_SECRET_KEY:
    import secrets as _secrets_mod
    _SESSION_SECRET_KEY = _secrets_mod.token_urlsafe(32)
    logging.getLogger("tenant_auth").warning(
        "[session] SESSION_SECRET_KEY not set — using a random per-process key "
        "(all sessions invalidate on restart). Fine for local dev; set a real "
        "value in production so sessions survive a redeploy."
    )
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(
    SessionMiddleware, secret_key=_SESSION_SECRET_KEY,
    session_cookie="nce_session", same_site="lax",
    # Render sets RENDER=true on every service automatically — reused here
    # rather than inventing a new env var, same reasoning as _AUTH_PASS
    # being empty = local dev elsewhere in this file. https_only=True would
    # silently drop the cookie over plain http:// (local dev), so it's only
    # enabled where we're actually confirmed to be running on Render.
    https_only=bool(os.getenv("RENDER")),
)
app.include_router(_tenant_router)

# ── ARQ job queue (Redis-backed, single-service) ──────────────────────────────
_ARQ_POOL = None          # arq.ArqRedis | None — None = Redis unavailable, fall back to Thread
_ARQ_WORKER_TASK = None   # asyncio.Task | None — the supervisor task, tracked for clean shutdown


# TEMPORARY (2026-08-16) — see arq_task_cron_diagnostic's docstring. Flip
# both this and the cron_jobs entry below off together once root cause is
# found; also remove the watchdog task creation in _arq_worker_supervisor.
_CRON_WATCHDOG_ENABLED = True


def _build_arq_worker(redis_settings):
    from arq import cron
    from arq.worker import Worker

    cron_jobs = [
        cron(arq_task_refresh_entsoe_prices, hour={3}, minute={0}),
        cron(arq_task_source_drift_check, weekday='sun', hour={3}, minute={0}),
    ]
    if _CRON_WATCHDOG_ENABLED:
        cron_jobs.append(cron(arq_task_cron_diagnostic))  # fires every minute

    # TEMPORARY diagnostic (2026-08-16): confirm registration itself
    # succeeded with the parameters we think we passed, independent of
    # whether dispatch later works — a malformed/silently-dropped
    # registration would be invisible without this.
    for cj in cron_jobs:
        print(f"[arq-registration] {cj.name}: month={cj.month} day={cj.day} "
              f"weekday={cj.weekday} hour={cj.hour} minute={cj.minute} "
              f"second={cj.second} run_at_startup={cj.run_at_startup}", flush=True)

    return Worker(
        functions=[arq_task_generate_permit],
        # Daily ENTSO-E day-ahead price refresh — see
        # arq_task_refresh_entsoe_prices above and permit_ai/entsoe_prices.py.
        # 03:00 UTC: off-peak, well clear of the ~00:00 CET SDAC market close
        # that publishes next-day prices, and clear of typical daytime PDF-
        # generation load on this single-service instance.
        #
        # Weekly source-drift sweep (2026-08-14) — see
        # arq_task_source_drift_check above and permit_ai/source_drift.py.
        # Sunday 03:00 UTC: same off-peak reasoning as the ENTSO-E job.
        # Weekly (not daily) because regulatory source text realistically
        # drifts on the order of weeks/months, not days — a daily full
        # sweep would just write ~121 near-identical "unchanged" rows/day
        # for no real benefit, see the 2026-08-13/14 investigation report.
        cron_jobs=cron_jobs,
        redis_settings=redis_settings,
        max_jobs=2,           # max 2 concurrent permit generations
        handle_signals=False,  # uvicorn owns SIGTERM — don't let ARQ shadow it
        poll_delay=0.5,
        job_timeout=900,       # 15 min — covers RAG+Claude+proofread+PDF
    )


async def _cron_watchdog(worker) -> None:
    """
    TEMPORARY (2026-08-16) — see arq_task_cron_diagnostic's docstring.
    Logs each registered cron job's name, current next_run, and whether
    it looks overdue, every 30s. Directly answers: is the schedule being
    computed/advanced at all, is it stuck, or is it correct-but-not-
    resulting-in-dispatch. Reads worker.cron_jobs directly — no
    monkeypatching, no dependency on arq's own (separately-confirmed-
    invisible-to-us) internal logger.
    """
    from datetime import datetime, timezone
    while True:
        await asyncio.sleep(30)
        now = datetime.now(timezone.utc)
        for cj in worker.cron_jobs:
            overdue = cj.next_run is not None and cj.next_run < now
            print(f"[cron-watchdog] {cj.name}: next_run={cj.next_run} "
                  f"now={now.isoformat()} overdue={overdue}", flush=True)


async def _arq_worker_supervisor(redis_settings) -> None:
    """
    Runs the ARQ worker with automatic restart-on-crash.

    Worker.main() runs an infinite poll loop and is never expected to
    return on its own; ANY unhandled exception inside it (a transient
    Redis network blip, anything) currently propagates straight out of
    the fire-and-forget asyncio.create_task() this used to be wrapped in,
    which kills the task silently — job processing then stays dead for
    the rest of the process's life. That's a real, standing gap
    independent of any specific change to this file: nothing here
    supervises or restarts a dead worker task. This wraps main() in a
    restart loop with exponential backoff so a single crash, from
    whatever cause, is self-healing instead of a silent, indefinite
    outage — the failure mode that matters here is "job processing never
    resumes until the next full deploy," and this directly prevents that,
    regardless of what actually triggers a given crash.
    """
    backoff = 2.0
    while True:
        worker = _build_arq_worker(redis_settings)
        watchdog_task = (
            asyncio.create_task(_cron_watchdog(worker), name="cron-watchdog")
            if _CRON_WATCHDOG_ENABLED else None
        )
        try:
            print("[arq] Worker (re)started", flush=True)
            backoff = 2.0  # reset after a clean (re)start
            await worker.main()
            # main() returning at all (not raising) is unexpected in normal
            # operation (burst=False here) — treat it the same as a crash.
            print("[arq] Worker.main() returned unexpectedly — restarting", flush=True)
        except asyncio.CancelledError:
            print("[arq] Worker supervisor cancelled — shutting down cleanly", flush=True)
            raise
        except Exception as exc:
            import traceback as _tb
            print(f"[arq] WORKER CRASHED: {exc!r} — restarting in {backoff:.0f}s", flush=True)
            print(_tb.format_exc(), flush=True)
        finally:
            if watchdog_task is not None:
                watchdog_task.cancel()
            try:
                await worker.close()
            except Exception:
                pass
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60.0)  # exponential backoff, capped at 60s


@app.on_event("startup")
async def _arq_startup() -> None:
    global _ARQ_POOL, _ARQ_WORKER_TASK
    _redis_url = os.getenv("REDIS_URL", "")
    if not _redis_url:
        print("[arq] REDIS_URL not set — job queue disabled, fallback to daemon threads")
        return
    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        _rs = RedisSettings.from_dsn(_redis_url)
        _ARQ_POOL = await create_pool(_rs)
        _ARQ_WORKER_TASK = asyncio.create_task(
            _arq_worker_supervisor(_rs), name="arq-worker-supervisor"
        )
        print(f"[arq] Worker supervisor started — max_jobs=2  redis={_redis_url[:40]}")
    except Exception as _exc:
        print(f"[arq] Startup failed ({_exc}) — fallback to daemon threads")
        _ARQ_POOL = None


@app.on_event("shutdown")
async def _arq_shutdown() -> None:
    """
    Cancel the worker supervisor cleanly on process shutdown. Prevents
    "Task was destroyed but it is pending!" noise on every deploy restart —
    harmless on its own, but exactly the kind of log line that's easy to
    mistake for a startup crash when read in isolation (as happened during
    the 2026-08-06 incident investigation this supervisor was added for);
    worth cleaning up regardless of whether it was ever the actual cause
    of anything.
    """
    global _ARQ_WORKER_TASK
    if _ARQ_WORKER_TASK is not None:
        _ARQ_WORKER_TASK.cancel()
        try:
            await _ARQ_WORKER_TASK
        except (asyncio.CancelledError, Exception):
            pass


@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    # Auth disabled when BASIC_AUTH_PASS not set (local dev)
    if not _AUTH_PASS:
        return await call_next(request)

    # Only enforce auth on the tool subdomain (ai.ncenergy.fi).
    # ncenergy.fi landing page and localhost pass through unconditionally.
    host = request.headers.get("host", "")
    if "ai.ncenergy" not in host:
        return await call_next(request)

    # Three paths remain public on the tool domain
    if request.url.path in _TOOL_EXEMPT:
        return await call_next(request)

    # All other requests — including /, /static/*, /api/* — require credentials
    auth = request.headers.get("authorization", "")
    _401 = HTMLResponse(
        content=_401_HTML,
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="NCE Permit AI"'},
    )
    if not auth.startswith("Basic "):
        return _401
    try:
        decoded  = base64.b64decode(auth[6:]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        return _401
    ok = secrets.compare_digest(username.encode(), _AUTH_USER.encode()) and \
         secrets.compare_digest(password.encode(), _AUTH_PASS.encode())
    if not ok:
        return _401
    return await call_next(request)


@app.middleware("http")
async def add_charset(request, call_next):
    response = await call_next(request)
    if "application/json" in response.headers.get("content-type", ""):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


@app.middleware("http")
async def head_as_get(request: Request, call_next):
    """HEAD requests must behave like GET with no body (RFC 7231 §4.3.2).
    FastAPI/Starlette routes registered with @app.get() do not auto-handle HEAD,
    causing crawlers and Search Console to receive 405."""
    if request.method != "HEAD":
        return await call_next(request)
    request.scope["method"] = "GET"
    response = await call_next(request)
    if hasattr(response, "body_iterator"):
        async for _ in response.body_iterator:
            pass
    return Response(
        status_code=response.status_code,
        headers=dict(response.headers),
    )
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR    = os.path.dirname(_BACKEND_DIR)
_STATIC_DIR  = os.path.join(_BACKEND_DIR, "static")
_LANDING_DIR = _REPO_DIR  # root index.html lives here
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Root assets/ for the landing page (ncenergy.fi / www.ncenergy.fi).
# Mount only if the folder exists so the app still starts without it.
_ASSETS_DIR = os.path.join(_REPO_DIR, "assets")
if os.path.isdir(_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

MML_API_KEY   = os.getenv("MML_API_KEY", "")
PORT          = int(os.environ.get("PORT", 8000))
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
ALERT_EMAIL   = os.getenv("ALERT_EMAIL", "jere@ncenergy.fi")

_AUTH_USER   = os.getenv("BASIC_AUTH_USER", "nce")
_AUTH_PASS   = os.getenv("BASIC_AUTH_PASS", "")  # empty = auth disabled (local dev)
# Paths that remain public even on ai.ncenergy.fi (landing page counter, contact form, health check)
_TOOL_EXEMPT = {"/api/stats", "/api/access-request", "/api/health", "/api/rag-status"}

SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
_INGEST_SECRET = os.getenv("INGEST_SECRET", "")

_401_HTML = """<!doctype html>
<html lang="fi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kirjautuminen vaaditaan</title>
<style>
  body{margin:0;font-family:system-ui,sans-serif;background:#fff;
       display:flex;align-items:center;justify-content:center;min-height:100vh;text-align:center}
  .box{padding:40px 24px}
  h1{font-size:22px;font-weight:500;color:#111;margin:0 0 8px}
  p{font-size:15px;color:#666;margin:0 0 28px}
  a{color:#00B4A0;text-decoration:none;font-size:15px}
  a:hover{text-decoration:underline}
</style>
</head>
<body>
<div class="box">
  <h1>Kirjautuminen vaaditaan</h1>
  <p>Authentication required</p>
  <a href="https://ncenergy.fi">&#8592; Palaa etusivulle &nbsp;/&nbsp; Return to homepage</a>
</div>
</body>
</html>"""

# ── Usage monitoring ──────────────────────────────────────────────────────────
_usage_logger = logging.getLogger("usage")
_usage_logger.setLevel(logging.INFO)
_ip_window: dict[str, list[float]] = defaultdict(list)   # ip → [timestamps]
_ip_lock = Lock()
_ALERT_WINDOW_SEC  = 600   # 10 min
_ALERT_THRESHOLD   = 3     # max calls per window before alert
_alerted_ips: set[str] = set()  # avoid duplicate alerts per server lifetime


def _log_usage(ip: str, hanketyyppi: str, country: str, phase: str,
               job_id: str, status: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _usage_logger.info(
        "[USAGE] timestamp=%s ip=%s type=%s country=%s phase=%s job_id=%s status=%s",
        ts, ip, hanketyyppi, country, phase, job_id, status,
    )
    if not ip or ip in ("testclient", "127.0.0.1"):
        return
    now = time.monotonic()
    with _ip_lock:
        calls = [t for t in _ip_window[ip] if now - t < _ALERT_WINDOW_SEC]
        calls.append(now)
        _ip_window[ip] = calls
        should_alert = len(calls) > _ALERT_THRESHOLD and ip not in _alerted_ips
        if should_alert:
            _alerted_ips.add(ip)
    if should_alert:
        Thread(target=_send_alert, args=(ip, len(calls), ts), daemon=True).start()


def _send_alert(ip: str, count: int, ts: str) -> None:
    if not RESEND_API_KEY:
        _usage_logger.warning("[USAGE] ALERT: ip=%s count=%d — RESEND_API_KEY puuttuu, sähköposti lähettämättä", ip, count)
        return
    try:
        _requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json={
                "from": "NCE Permit AI <noreply@ncenergy.fi>",
                "to": [ALERT_EMAIL],
                "subject": f"[NCE Permit AI] Hälytys: {count} kutsua 10 min — IP {ip}",
                "text": (
                    f"Epäilyttävä käyttö havaittu.\n\n"
                    f"IP: {ip}\n"
                    f"Kutsuja viimeisen 10 min aikana: {count}\n"
                    f"Aika: {ts}\n\n"
                    f"Tarkista Render-lokit lisätietoja varten."
                ),
            },
            timeout=10,
        )
        _usage_logger.info("[USAGE] ALERT lähetetty: ip=%s count=%d", ip, count)
    except Exception as exc:
        _usage_logger.warning("[USAGE] ALERT-lähetys epäonnistui: %s", exc)

if not MML_API_KEY:
    print("[startup] VAROITUS: MML_API_KEY ei asetettu — maankäyttöselvityksen WFS-haut eivät toimi. "
          "Aseta ympäristömuuttuja tai lisää Render-palveluun. Ks. README.md.")


# ── Pydantic-mallit ───────────────────────────────────────────────────────────

class AccessRequestModel(BaseModel):
    yritys: str
    yhteyshenkilo: str
    sahkoposti: str
    puhelin: str = ""
    kuvaus: str


class PermitAIRequest(BaseModel):
    question: str
    n_results: int = 5


class PermitChatRequest(BaseModel):
    question:    str
    session_id:  str = ""
    n_results:   int = 6
    hanketyyppi: str = ""
    country:     str = "FI"


class OptimizeRequest(BaseModel):
    bbox: list          # [lat_min, lon_min, lat_max, lon_max]
    project_type: str   = "bess"
    power_mw:     float = 5.0
    min_area_ha:  float = 2.0


class ApplicationRequest(BaseModel):
    hanketyyppi:                  str
    kiinteistotunnus:             str
    teho_mw:                      Optional[float] = 0.0
    kapasiteetti_mwh:             Optional[float] = 0.0
    y_tunnus:                     Optional[str]   = None
    osoite:                       Optional[str]   = None
    kunta:                        str
    hakija:                       str
    sijainti_ymparistovaikutukset: Optional[str]   = None
    hankkeen_vaihe:               Optional[str]   = None
    kohdeviranomainen:            Optional[str]   = None
    lang:                         Optional[str]   = "FI"
    country:                      Optional[str]   = "FI"
    session_id:                   Optional[str]   = ""
    hanke_id:                     Optional[str]   = ""   # RTB cockpit linkitys
    # IFC esitäyttö (valinnainen)
    ifc_floor_area:               Optional[float] = 0.0
    ifc_building_height:          Optional[float] = 0.0
    ifc_fire_rating:              Optional[str]   = ""
    ifc_materials:                Optional[str]   = ""
    ifc_storeys:                  Optional[int]   = 0
    ifc_compliance_flags:         Optional[str]   = ""


# ── Oikolukutehtävien in-memory-varasto ──────────────────────────────────────
# {job_id: {status: pending|running|done|error, pdf_bytes: bytes|None, error: str|None}}
_proofread_store: dict = {}

# ── Admin ingest -tehtävien in-memory-varasto ─────────────────────────────────
_ingest_jobs: dict = {}


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kiinteistotunnus: str
    hanke_id:         Optional[str] = ""   # RTB cockpit linkitys
    title: Optional[str] = None
    map_image: Optional[str] = None          # base64 PNG (vanhentunut, käytetään vain fallbackina)
    property_geojson: Optional[dict] = None  # GeoJSON frontendilta – vältetään kaksoisnouto
    analysis_result: Optional[dict] = None   # Esikäsitelty analyysi – sama arvo UI:hin ja PDF:ään
    project_owner:   str = "Carbon Zero Finland"
    project_name:    str = "Standalone BESS 1 MW"
    power_mw:        float = 1.0
    grid_connection: str = "Jakeluverkko 20 kV (Caruna)"
    market:          str = "FCR (Frequency Containment Reserve)"
    # Manuaaliset syötteet
    manual_kaavoitus:    Optional[str]   = None  # asemakaava|yleiskaava|ei_kaavaa|ei_tietoa
    manual_tulvavaara:   Optional[str]   = None  # ei|kyllä|ei_tietoa
    manual_maapera:      Optional[str]   = None  # kallio|moreeni|hiekka|savi|turve|ei_tietoa
    manual_pinta_ala_ha: Optional[float] = None
    lang: Optional[str] = "FI"


# ── Endpointit ────────────────────────────────────────────────────────────────

@app.get("/")
async def root(request: Request):
    host = request.headers.get("host", "")
    if "ai.ncenergy" in host:
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))
    # ncenergy.fi, www.ncenergy.fi, localhost → landing page
    landing = os.path.join(_LANDING_DIR, "index.html")
    if os.path.isfile(landing):
        return FileResponse(landing)
    # fallback: tool (landing page not deployed yet)
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/sitemap.xml")
async def sitemap():
    path = os.path.join(_LANDING_DIR, "sitemap.xml")
    return FileResponse(path, media_type="application/xml")


@app.get("/robots.txt")
async def robots():
    path = os.path.join(_LANDING_DIR, "robots.txt")
    return FileResponse(path, media_type="text/plain")


@app.get("/privacy")
async def privacy():
    return FileResponse(os.path.join(_STATIC_DIR, "privacy.html"))


@app.get("/tietosuoja")
async def tietosuoja():
    return FileResponse(os.path.join(_STATIC_DIR, "privacy.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "mml_key_set": bool(MML_API_KEY)}


@app.get("/api/rag-status")
async def rag_status():
    """Returns which RAG collection + model is currently active. Public endpoint."""
    import chromadb as _chroma
    active_col   = _permit_ai_module._COLLECTION
    active_model = _permit_ai_module._EMBED_MODEL
    v1_count = None
    v2_count = None
    db_error = None
    try:
        client = _chroma.PersistentClient(path=_DB_PATH)
        try:
            v1_count = client.get_collection("permit_docs").count()
        except Exception:
            pass
        try:
            v2_count = client.get_collection("permit_docs_v2").count()
        except Exception:
            pass
    except Exception as exc:
        db_error = str(exc)
    db_path_exists = os.path.isdir(_DB_PATH)
    db_path_files  = os.listdir(_DB_PATH) if db_path_exists else []
    return {
        "active_collection": active_col,
        "active_model":      active_model,
        "v2_ready":          (v2_count or 0) >= _V2_MIN_CHUNKS,
        "permit_docs_count": v1_count,
        "permit_docs_v2_count": v2_count,
        "db_path": _DB_PATH,
        "db_path_exists": db_path_exists,
        "db_path_files": db_path_files,
        **({"db_error": db_error} if db_error else {}),
    }


@app.post("/api/access-request")
async def access_request(req: AccessRequestModel):
    # PR A (tenant architecture, 2026-08-09): this endpoint used to be pure
    # email-and-forget — nothing survived if the email was missed or SMTP
    # failed. Now it ALSO auto-drafts a tenant + owner user (both
    # pending_approval — admin-gated, see tenant_auth.py) so the request is
    # durable and shows up in GET /api/admin/tenants?status=pending_approval
    # even if the notification email below never arrives. Wrapped in
    # try/except and logged-not-raised: a DB outage must not take down
    # request intake entirely — the email channel below still works as a
    # fallback either way, matching this endpoint's original behavior.
    try:
        from tenant_db.base import get_session
        import tenant_auth as _ta
        _session = get_session()
        try:
            _ta.create_access_request_and_draft(
                _session, company_name=req.yritys, contact_name=req.yhteyshenkilo,
                email=req.sahkoposti, phone=req.puhelin, description=req.kuvaus,
            )
        finally:
            _session.close()
    except Exception as exc:
        logging.getLogger("usage").error("[ACCESS-REQUEST] tenant draft creation failed: %s", exc)

    def _send():
        msg = email.mime.multipart.MIMEMultipart()
        msg["From"]    = SMTP_USER or "info@ncenergy.fi"
        msg["To"]      = "info@ncenergy.fi"
        msg["Subject"] = "Käyttöoikeuspyyntö — NCE Permit AI"
        body = (
            "Käyttöoikeuspyyntö — NCE Permit AI\n"
            "=====================================\n\n"
            f"Yritys:           {req.yritys}\n"
            f"Yhteyshenkilö:    {req.yhteyshenkilo}\n"
            f"Sähköposti:       {req.sahkoposti}\n"
            f"Puhelin:          {req.puhelin or '—'}\n\n"
            "Kuvaus toiminnasta:\n"
            "-------------------\n"
            f"{req.kuvaus}\n"
        )
        msg.attach(email.mime.text.MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)

    if not SMTP_USER or not SMTP_PASSWORD:
        logging.getLogger("usage").warning(
            "[ACCESS-REQUEST] SMTP not configured — yritys=%s email=%s",
            req.yritys, req.sahkoposti,
        )
        raise HTTPException(status_code=503, detail="Email service not configured")

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _send)
    except Exception as exc:
        logging.getLogger("usage").error("[ACCESS-REQUEST] SMTP error: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to send email")

    return {"ok": True}


@app.get("/api/debug-raw")
async def debug_raw():
    """Palauttaa viimeisimmän Claude-vastauksen /tmp/debug_raw_claude.txt."""
    try:
        with open("/tmp/debug_raw_claude.txt", encoding="utf-8") as f:
            content = f.read()
        return {"content": content[:3000]}
    except FileNotFoundError:
        return {"content": "Ei debug-tiedostoa — aja ensin generaatio."}


@app.get("/api/debug-encoding")
def debug_encoding():
    data = {"raw": "testiäö", "ae": "ä", "oe": "ö"}
    return JSONResponse(
        content=data,
        media_type="application/json; charset=utf-8",
    )


@app.get("/api/property/{kiinteistotunnus}")
async def property_boundaries(
    kiinteistotunnus: str,
    api_key: Optional[str] = Query(default=None),
):
    """Kiinteistörajat MML INSPIRE WFS:stä (ei API-avainta tarvita)."""
    try:
        return await get_property_boundaries(kiinteistotunnus, api_key=api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"INSPIRE WFS -virhe: {exc}")


@app.get("/api/fingrid/lines")
async def fingrid_lines(
    bbox: str = Query(default="22.5,60.6,23.0,60.9"),
    min_voltage_kv: int = Query(default=0),
):
    """Voimajohdot OSM Overpass -rajapinnasta (ei API-avainta)."""
    try:
        coords = [float(x) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox: minlon,minlat,maxlon,maxlat")
    return await get_transmission_lines(tuple(coords))


@app.get("/api/groundwater")
async def groundwater(bbox: str = Query(default="22.5,60.6,23.0,60.9")):
    """Pohjavesialueet SYKE Hakku -rajapinnasta (ei API-avainta)."""
    try:
        coords = [float(x) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox: minlon,minlat,maxlon,maxlat")
    return await get_groundwater_areas(tuple(coords))


@app.get("/api/buildings/nearest")
async def nearest_building(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(default=1.0),
):
    """Lähin rakennus OSM:sta – palauttaa etäisyyden metreinä ja GeoJSON."""
    delta = radius_km / 111.0
    bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
    data = await get_buildings(bbox)
    dist = nearest_point_distance_m(lat, lon, data)
    return {
        "nearest_building_m": round(dist) if dist >= 0 else None,
        "buildings_found": len(data.get("features", [])),
        "geojson": data,
    }


@app.get("/api/natura")
async def natura(bbox: str = Query(default="22.5,60.6,23.0,60.9")):
    """Natura 2000 -alueet SYKE:ltä."""
    try:
        coords = [float(x) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox: minlon,minlat,maxlon,maxlat")
    return await get_natura_areas(tuple(coords))


@app.get("/api/bess/analysis/{kiinteistotunnus}")
async def bess_analysis(
    kiinteistotunnus: str,
    api_key: Optional[str] = Query(default=None),
    grid_connection: Optional[str] = Query(default=None),
):
    """
    Kokonaisvaltainen BESS-soveltuvuusanalyysi – hakee kaikki datat rinnakkain.
    Pisteytyskriteerit:
      Jakeluverkon etäisyys: <500 m = 30p, 500 m–2 km = 20p, >2 km = 5p
      Ei pohjavettä:         20p  (ei dataa = 0p)
      Ei Natura:             20p
      Ei asemakaavaa:        15p
      Asutus >300 m:         15p
    """
    return await _run_analysis(
        kiinteistotunnus,
        api_key=api_key or MML_API_KEY,
        grid_connection=grid_connection or "",
    )


@app.get("/api/map/static/{kiinteistotunnus}")
async def static_map_image(
    kiinteistotunnus: str,
    zoom: int = Query(default=16),
):
    """
    Generoi staattinen karttakuva kiinteistöstä (staticmap + OSM-tiilet).
    Piirtää kiinteistörajan punaisena viivana. Palauttaa base64 PNG.
    """
    try:
        prop = await get_property_boundaries(kiinteistotunnus)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Property lookup failed: {exc}")

    center_lat, center_lon = _centroid(prop)

    try:
        png_bytes = await asyncio.to_thread(
            _render_static_map, prop, center_lat, center_lon, zoom
        )
        return {"image_b64": base64.b64encode(png_bytes).decode()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Map image generation failed: {exc}")


@app.post("/api/report/generate")
async def generate_report(req: ReportRequest):
    """PDF-raportti. Käyttää frontendilta tullutta analyysiä – ei kaksoisajoa."""
    kt = req.kiinteistotunnus

    # Käytä frontendilta tullutta geometriaa; hae WFS:stä vain jos puuttuu
    prop = req.property_geojson
    if not prop:
        try:
            prop = await get_property_boundaries(kt, api_key=MML_API_KEY)
        except Exception:
            prop = None

    # Generoi karttakuva samasta geometriasta
    map_image_b64 = req.map_image
    if map_image_b64 is None and prop:
        center_lat, center_lon = _centroid(prop)
        try:
            png_bytes = await asyncio.to_thread(
                _render_static_map, prop, center_lat, center_lon, 16
            )
            map_image_b64 = base64.b64encode(png_bytes).decode()
        except Exception:
            pass

    # Analyysi: käytä frontendilta tullutta (UI:ssa näytetty arvo = PDF:n arvo)
    analysis = req.analysis_result
    if not analysis:
        try:
            analysis = await _run_analysis(
                kt, api_key=MML_API_KEY, prop=prop,
                grid_connection=req.grid_connection,
            )
        except HTTPException:
            analysis = {}

    # Sovella manuaaliset syötteet — päivittää pisteytyksen
    has_manual = any([
        req.manual_kaavoitus, req.manual_tulvavaara,
        req.manual_maapera, req.manual_pinta_ala_ha,
    ])
    if has_manual:
        analysis = _apply_manual_overrides(analysis, req)

    prop_meta = {
        "area_ha":  analysis.get("area_ha"),
        "kuntanimi": analysis.get("kuntanimi", "–"),
        "kylanimi":  analysis.get("kylanimi", "–"),
    }

    pdf_bytes = generate_bess_report(
        kiinteistotunnus=kt,
        property_data=prop_meta,
        analysis_data=analysis,
        map_image_b64=map_image_b64,
        project_owner=req.project_owner,
        project_name=req.project_name,
        power_mw=req.power_mw,
        grid_connection=req.grid_connection,
        market=req.market,
        lang=req.lang or "FI",
    )
    # RTB tracking — record land use completion
    _rtb_id = (req.hanke_id or "").strip() or _rtb.make_hanke_id("", kt)
    if _rtb_id:
        try:
            _rtb.update_land_use(
                _rtb_id,
                kiinteistotunnus=kt,
                hanketyyppi=getattr(req, "hanketyyppi", "") or "",
                maa="FI",
            )
        except Exception:
            pass

    filename = f"BESS_raportti_{kt.replace('-', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Permit AI ────────────────────────────────────────────────────────────────

@app.post("/api/generate-application")
@limiter.limit("5/hour")
async def generate_application_endpoint(request: Request, req: ApplicationRequest):
    """Käynnistä lupahakemus-PDF:n generointi taustasäikeessä. Palauttaa job_id heti (202)."""
    # Payment gate — NOOP when PAYMENT_ENABLED=false (default)
    from stripe_payments import PAYMENT_ENABLED as _PAY_ON, get_payment_status as _pay_status
    if _PAY_ON:
        _session_id = req.session_id or ""
        if not _session_id or _pay_status(_session_id) != "paid":
            raise HTTPException(status_code=402, detail="Payment required")

    allowed = {"BESS", "tuulivoima_maa", "tuulivoima_meri", "aurinkovoima", "SMR",
               "smr_bess", "vesivoima", "hybridi",
               "asuinrakennus", "teollisuus", "maatalous", "liikerakennus", "muu",
               "ymparistolupa", "datakeskus",
               "smr_se", "smr_no", "smr_da", "smr_de", "smr_ee",
               "egs", "offshore_wind"}
    if req.hanketyyppi not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"hanketyyppi oltava: {', '.join(sorted(allowed))}")

    # Phase-Lock: tarkista onko edellinen vaihe suoritettu
    if _PHASE_LOCK_OK and req.session_id and req.hankkeen_vaihe:
        ok, err = _check_phase(req.session_id, req.hanketyyppi, req.hankkeen_vaihe)
        if not ok:
            raise HTTPException(status_code=400, detail=err)

    inp = ApplicationInput(
        hanketyyppi                   = req.hanketyyppi,
        kiinteistotunnus              = req.kiinteistotunnus,
        teho_mw                       = req.teho_mw or 0.0,
        kapasiteetti_mwh              = req.kapasiteetti_mwh or 0.0,
        y_tunnus                      = req.y_tunnus or "",
        osoite                        = req.osoite or "",
        kunta                         = req.kunta,
        hakija                        = req.hakija,
        sijainti_ymparistovaikutukset = req.sijainti_ymparistovaikutukset or "",
        hankkeen_vaihe                = req.hankkeen_vaihe or "",
        kohdeviranomainen             = req.kohdeviranomainen or "",
        lang                          = req.lang or "FI",
        country                       = req.country or "FI",
        ifc_floor_area                = req.ifc_floor_area or 0.0,
        ifc_building_height           = req.ifc_building_height or 0.0,
        ifc_fire_rating               = req.ifc_fire_rating or "",
        ifc_materials                 = req.ifc_materials or "",
        ifc_storeys                   = req.ifc_storeys or 0,
        ifc_compliance_flags          = req.ifc_compliance_flags or "",
    )

    # White-label: NCE logo by default; B2B customers override via api_key branding
    # (This route uses the NCE default — white-label is fully active on /api/b2b/generate-report)
    from white_label import NCE_LOGO_PATH as _NCE_LOGO
    inp.logo_path = _NCE_LOGO

    job_id = uuid.uuid4().hex[:10]
    inp.generation_id = job_id  # links this generation's retrieval_trace rows (see retrieval_trace.py)

    # PR B (tenant architecture, 2026-08-09): best-effort Layer 1 tracking,
    # inert unless TENANT_TRACKING_ENABLED=true AND a tenant session cookie
    # is present — neither is true for any real traffic yet (Basic Auth
    # still gates this route entirely; see tenant_db/layer1.py's docstring).
    from tenant_db.layer1 import record_generation_start as _record_gen_start
    _tenant_id = request.session.get("tenant_id")
    _project_id = _record_gen_start(
        _tenant_id, hanketyyppi=req.hanketyyppi, country=req.country or "FI",
        phase=req.hankkeen_vaihe or "",
    )

    _proofread_store[job_id] = {
        "status": "pending", "pdf_bytes": None, "error": None,
        "lang":          req.lang or "FI",
        "hanketyyppi":   req.hanketyyppi or "doc",
        "kunta":         req.kunta or "hanke",
        "session_id":    req.session_id or "",
        "hankkeen_vaihe": req.hankkeen_vaihe or "",
    }

    _client_ip = get_remote_address(request)
    _log_usage(_client_ip, req.hanketyyppi, req.country or "FI",
               req.hankkeen_vaihe or "", job_id, "started")

    def _bg_generate():
        try:
            _proofread_store[job_id]["status"] = "running"
            print(f"[bg] {job_id} START hanke={req.hanketyyppi} country={req.country or 'FI'}", flush=True)
            draft_bytes, sections, sources = generate_application_draft(inp)
            print(f"[bg] {job_id} draft done, sections={list(sections.keys())}", flush=True)
            _proofread_store[job_id]["debug_sections"] = {k: len(v) for k, v in sections.items() if isinstance(v, str)}
            pdf = apply_proofread_to_pdf(inp, sections, sources)
            print(f"[bg] {job_id} pdf done len={len(pdf) if pdf else 0}", flush=True)
            _proofread_store[job_id]["pdf_bytes"] = pdf
            _proofread_store[job_id]["status"] = "done"
            _log_usage(_client_ip, req.hanketyyppi, req.country or "FI",
                       req.hankkeen_vaihe or "", job_id, "done")
            # PR B: best-effort Layer 1 tracking (see the note above where
            # _project_id was created) — pdf_url/raqs_score are None: no
            # durable PDF storage/URL exists yet (bytes live only in
            # _proofread_store, served on demand) and RAQS scores aren't
            # currently returned out of generate_application.py to here.
            # Recorded as None rather than invented, honestly reflecting
            # what this pipeline actually exposes today.
            from tenant_db.layer1 import record_report as _record_report
            _record_report(
                _tenant_id, _project_id, phase=req.hankkeen_vaihe or "",
                pdf_url=None, raqs_score=None,
            )
            # Auto-complete phase when PDF is generated (no user click required)
            if _PHASE_LOCK_OK and req.session_id and req.hankkeen_vaihe:
                # 2026-08-13: added kayttolupa/purku — this dict predates P3-2's
                # generalization of the phase-lock endpoints and was missed then;
                # without these, a real SMR kayttolupa/purku generation would
                # complete fine but silently fail to auto-advance the phase lock.
                _phase_num = {"esiselvitys": 1, "lupavaihe": 2, "rakentaminen": 3,
                              "rakentamisvaihe": 3, "kayttolupa": 4, "purku": 5,
                              }.get(req.hankkeen_vaihe.lower().strip(), 0)
                if _phase_num:
                    _phase_status = _unlock_next_phase(
                        req.session_id, req.hanketyyppi, _phase_num, "generated"
                    )
                    _proofread_store[job_id]["phase_status"] = _phase_status
            # RTB tracking — record permit doc completion
            _rtb_id = (req.hanke_id or "").strip() or _rtb.make_hanke_id(
                req.y_tunnus or "", req.kiinteistotunnus or ""
            )
            if _rtb_id:
                try:
                    _rtb.update_permit_doc(
                        _rtb_id,
                        job_id=job_id,
                        phase=req.hankkeen_vaihe or "",
                        y_tunnus=req.y_tunnus or "",
                        kiinteistotunnus=req.kiinteistotunnus or "",
                        hanketyyppi=req.hanketyyppi or "",
                        maa=req.country or "FI",
                    )
                    _proofread_store[job_id]["hanke_id"] = _rtb_id
                except Exception:
                    pass
        except InsufficientSourcesError as exc:
            _proofread_store[job_id]["status"] = "insufficient_sources"
            _proofread_store[job_id]["error"] = str(exc)
            _proofread_store[job_id]["chunks_found"] = exc.chunks_found
            _proofread_store[job_id]["avg_relevance"] = round(exc.avg_relevance, 2)
            _log_usage(_client_ip, req.hanketyyppi, req.country or "FI",
                       req.hankkeen_vaihe or "", job_id, f"RAG_FAIL:chunks={exc.chunks_found}")
        except GenerationCapError as exc:
            # TASO 1 cost & resource guardrail — a per-generation cap tripped.
            # Already logged to guardrail_log (retrieval_trace.py) with generation_id
            # by the raising code; surface it here as a distinct, clean status too.
            _proofread_store[job_id]["status"] = "cap_exceeded"
            _proofread_store[job_id]["error"] = str(exc)
            _proofread_store[job_id]["cap_kind"] = exc.kind
            _proofread_store[job_id]["cap_count"] = exc.count
            _proofread_store[job_id]["cap_limit"] = exc.cap
            _log_usage(_client_ip, req.hanketyyppi, req.country or "FI",
                       req.hankkeen_vaihe or "", job_id, f"CAP_HIT:{exc.kind}={exc.count}/{exc.cap}")
        except Exception as exc:
            import traceback as _tb
            _err = f"{type(exc).__name__}: {exc}"
            print(f"[bg] {job_id} ERROR {_err}", flush=True)
            print(_tb.format_exc(), flush=True)
            _proofread_store[job_id]["status"] = "error"
            _proofread_store[job_id]["error"] = _err
            _log_usage(_client_ip, req.hanketyyppi, req.country or "FI",
                       req.hankkeen_vaihe or "", job_id, f"error:{_err[:60]}")
        except BaseException as exc:
            import traceback as _tb
            _err = f"{type(exc).__name__}: {exc}"
            print(f"[bg] {job_id} FATAL {_err}", flush=True)
            print(_tb.format_exc(), flush=True)
            try:
                _proofread_store[job_id]["status"] = "error"
                _proofread_store[job_id]["error"] = _err
            except Exception:
                pass

    if _ARQ_POOL is not None:
        await _ARQ_POOL.enqueue_job(
            "arq_task_generate_permit",
            job_id        = job_id,
            inp_dict      = dataclasses.asdict(inp),
            client_ip     = _client_ip,
            hanke_id      = req.hanke_id or "",
            session_id    = req.session_id or "",
            hankkeen_vaihe = req.hankkeen_vaihe or "",
            hanketyyppi   = req.hanketyyppi or "",
            country       = req.country or "FI",
            tenant_id     = _tenant_id or "",   # PR B — see the note above
            project_id    = _project_id or "",  # this function, same reasoning
        )
    else:
        Thread(target=_bg_generate, daemon=True).start()

    return Response(
        content    = json.dumps({"job_id": job_id}),
        status_code = 202,
        media_type = "application/json",
        headers    = {
            "X-Job-Id":                      job_id,
            "Access-Control-Expose-Headers": "X-Job-Id",
        },
    )


async def arq_task_generate_permit(
    ctx: dict,
    *,
    job_id: str,
    inp_dict: dict,
    client_ip: str,
    hanke_id: str,
    session_id: str,
    hankkeen_vaihe: str,
    hanketyyppi: str,
    country: str,
    tenant_id: str = "",
    project_id: str = "",
) -> None:
    """
    ARQ task — runs permit generation concurrently without blocking the event loop.

    Replaces Thread(target=_bg_generate) when REDIS_URL is set.
    Sync blocking work (Claude API + PDF render) is off-loaded to
    asyncio.to_thread() so other ARQ jobs and FastAPI requests run freely.
    max_jobs=2 caps concurrent generations at 2 (prevents OOM on 512MB Render).

    2026-08-13: the very first _proofread_store write used to sit OUTSIDE
    this try/except (before it even started), unlike _bg_generate()'s
    equivalent write below, which has always been inside its own try. That
    gap was found while diagnosing a real production incident: SMR
    kayttolupa/purku generations occasionally got stuck at status="pending"
    forever, invisible to the client, with zero error surfaced anywhere.
    The exact low-level trigger for the underlying KeyError was never
    pinned down with certainty despite substantial investigation (real
    production log forensics + 170+ local repro trials matching this
    arq version) — but regardless of the trigger, a bare
    `_proofread_store[job_id][...]` outside any exception handler is a
    real defect: if it ever raises, for ANY reason, ARQ's own internal
    job-failure handling silently swallows it into a Redis result blob
    nothing in this app reads, and the job hangs forever from the
    client's point of view. Moving the write inside try/except (mirroring
    _bg_generate exactly) plus the setdefault() self-heal below closes
    that failure mode unconditionally, independent of whatever the exact
    trigger turns out to be.
    """
    try:
        _pre_existing = job_id in _proofread_store
        _proofread_store.setdefault(
            job_id, {"status": "pending", "pdf_bytes": None, "error": None}
        )
        if not _pre_existing:
            # This is the specific scenario the investigation couldn't fully
            # reproduce: _proofread_store[job_id] not yet visible when the
            # ARQ worker started this task, despite the HTTP handler writing
            # it before enqueueing. Self-healed via setdefault() above so
            # the job runs normally instead of dying silently — but log it
            # loudly so a recurrence finally gives us forensic proof of the
            # exact mechanism, instead of the silent hang this used to cause.
            print(f"[arq] {job_id} WARNING: _proofread_store entry missing at "
                  f"task start (self-healed, job proceeding normally) — this "
                  f"is the dispatch race from the 2026-08-13 investigation; "
                  f"if you see this, please flag it, it's the first direct "
                  f"evidence of the actual trigger", flush=True)
        _proofread_store[job_id]["status"] = "running"
        print(f"[arq] {job_id} START hanke={hanketyyppi} country={country}", flush=True)

        inp = ApplicationInput(**inp_dict)

        draft_bytes, sections, sources = await asyncio.to_thread(
            generate_application_draft, inp
        )
        print(f"[arq] {job_id} draft done, sections={list(sections.keys())}", flush=True)
        _proofread_store[job_id]["debug_sections"] = {
            k: len(v) for k, v in sections.items() if isinstance(v, str)
        }

        pdf = await asyncio.to_thread(apply_proofread_to_pdf, inp, sections, sources)
        print(f"[arq] {job_id} pdf done len={len(pdf) if pdf else 0}", flush=True)

        _proofread_store[job_id]["pdf_bytes"] = pdf
        _proofread_store[job_id]["status"] = "done"
        _log_usage(client_ip, hanketyyppi, country, hankkeen_vaihe, job_id, "done")

        # PR B: best-effort Layer 1 tracking — same reasoning as
        # _bg_generate()'s equivalent call (Thread-dispatch path above);
        # this is the ARQ-dispatch path's counterpart.
        from tenant_db.layer1 import record_report as _record_report
        _record_report(
            tenant_id or None, project_id or None, phase=hankkeen_vaihe or "",
            pdf_url=None, raqs_score=None,
        )

        # Auto-complete phase
        # 2026-08-13: added kayttolupa/purku — same fix as
        # generate_application_endpoint()'s equivalent dict above (Thread
        # dispatch path); this is the ARQ dispatch path's counterpart, found
        # missing the same two entries during the same investigation.
        if _PHASE_LOCK_OK and session_id and hankkeen_vaihe:
            _phase_num = {
                "esiselvitys": 1, "lupavaihe": 2,
                "rakentaminen": 3, "rakentamisvaihe": 3,
                "kayttolupa": 4, "purku": 5,
            }.get(hankkeen_vaihe.lower().strip(), 0)
            if _phase_num:
                _phase_status = _unlock_next_phase(
                    session_id, hanketyyppi, _phase_num, "generated"
                )
                _proofread_store[job_id]["phase_status"] = _phase_status

        # RTB tracking
        _rtb_id = hanke_id.strip() or _rtb.make_hanke_id(
            inp.y_tunnus or "", inp.kiinteistotunnus or ""
        )
        if _rtb_id:
            try:
                _rtb.update_permit_doc(
                    _rtb_id,
                    job_id          = job_id,
                    phase           = hankkeen_vaihe,
                    y_tunnus        = inp.y_tunnus or "",
                    kiinteistotunnus = inp.kiinteistotunnus or "",
                    hanketyyppi     = hanketyyppi,
                    maa             = country,
                )
                _proofread_store[job_id]["hanke_id"] = _rtb_id
            except Exception:
                pass

    except InsufficientSourcesError as exc:
        _proofread_store[job_id]["status"] = "insufficient_sources"
        _proofread_store[job_id]["error"] = str(exc)
        _proofread_store[job_id]["chunks_found"] = exc.chunks_found
        _proofread_store[job_id]["avg_relevance"] = round(exc.avg_relevance, 2)
        _log_usage(client_ip, hanketyyppi, country, hankkeen_vaihe, job_id,
                   f"RAG_FAIL:chunks={exc.chunks_found}")

    except GenerationCapError as exc:
        # TASO 1 cost & resource guardrail — a per-generation cap tripped. Already
        # logged to guardrail_log (retrieval_trace.py) with generation_id by the
        # raising code; surface it here as a distinct, clean status too.
        _proofread_store[job_id]["status"] = "cap_exceeded"
        _proofread_store[job_id]["error"] = str(exc)
        _proofread_store[job_id]["cap_kind"] = exc.kind
        _proofread_store[job_id]["cap_count"] = exc.count
        _proofread_store[job_id]["cap_limit"] = exc.cap
        _log_usage(client_ip, hanketyyppi, country, hankkeen_vaihe, job_id,
                   f"CAP_HIT:{exc.kind}={exc.count}/{exc.cap}")

    except Exception as exc:
        import traceback as _tb
        _err = f"{type(exc).__name__}: {exc}"
        print(f"[arq] {job_id} ERROR {_err}", flush=True)
        print(_tb.format_exc(), flush=True)
        _proofread_store[job_id]["status"] = "error"
        _proofread_store[job_id]["error"] = _err
        _log_usage(client_ip, hanketyyppi, country, hankkeen_vaihe, job_id,
                   f"error:{_err[:60]}")

    except BaseException as exc:
        _err = f"{type(exc).__name__}: {exc}"
        print(f"[arq] {job_id} CANCELLED/FATAL {_err}", flush=True)
        _proofread_store[job_id]["status"] = "error"
        _proofread_store[job_id]["error"] = _err
        raise  # re-raise so ARQ marks the job as failed


async def arq_task_refresh_entsoe_prices(ctx: dict) -> None:
    """
    ARQ daily cron job (see cron_jobs= in _arq_startup below) — refreshes the
    ENTSO-E day-ahead price cache in Redis for all 9 countries. See
    permit_ai/entsoe_prices.py for the full fetch/aggregate/cache design.

    Failure here (including ENTSOE_API_TOKEN simply not being set yet) is
    logged, never fatal — feasibility.py's calculate_feasibility() falls
    back to the static IRENA/NREL/BloombergNEF benchmark table whenever the
    Redis cache has no fresh entry, so a skipped or failed refresh degrades
    solar/wind pricing accuracy, not availability.
    """
    token = os.getenv("ENTSOE_API_TOKEN", "")
    if not token:
        print("[arq] entsoe-price-refresh SKIPPED — ENTSOE_API_TOKEN not set", flush=True)
        return
    try:
        summary = await _refresh_entsoe_prices(token=token)
        ok = [c for c, v in summary["countries"].items() if v.get("lo_avg_eur_mwh") is not None]
        print(f"[arq] entsoe-price-refresh OK — {len(ok)}/{len(summary['countries'])} countries refreshed: {ok}", flush=True)
    except Exception as exc:
        print(f"[arq] entsoe-price-refresh FAILED: {exc}", flush=True)


async def arq_task_source_drift_check(ctx: dict) -> None:
    """
    ARQ weekly cron job (see cron_jobs= in _arq_startup below) — runs the
    full source-drift sweep (permit_ai/source_drift.py) across every
    checkable (source, url) pair (~121 as of 2026-08-14) and logs a
    summary. source_drift.py has existed since PR #52 (admin-triggered
    only by original design — no cron/worker previously called it, see
    that module's own docstring) but was never actually wired to run on a
    schedule until now. Confirmed safe to automate via the 2026-08-13/14
    investigation: a full manual sweep took ~30s wall-clock, negligible
    CPU, zero observed impact on the live process.

    Failure classification avoids re-alerting on sources that have NEVER
    once succeeded (e.g. a domain-wide SSL misconfig, a long-dead URL —
    real examples found in that investigation's first full sweep): a
    source only counts as a genuine regression if it has at least one
    prior successful (non-check_failed) result in its history AND this
    run's result is check_failed. A source with zero successful history
    ever is still recorded in the DB (nothing hidden), just not flagged
    as urgent — it's a known, standing problem, not new information.
    """
    print("[drift-cron] weekly sweep starting", flush=True)
    try:
        report = await _source_drift.check_all_sources()
    except Exception as exc:
        print(f"[drift-cron] sweep FAILED to run at all: {type(exc).__name__}: {exc}", flush=True)
        return

    changed = [r["source"] for r in report["results"] if r["status"] == "changed"]
    failed = [r["source"] for r in report["results"] if r["status"] == "check_failed"]
    prior_success = _source_drift.get_prior_success_flags(failed) if failed else {}
    regressions = sorted(s for s in failed if prior_success.get(s))
    known_unreachable = sorted(s for s in failed if not prior_success.get(s))

    print(
        f"[drift-cron] weekly check complete: {report['targets_checked']} sources checked, "
        f"{len(changed)} newly changed, {len(regressions)} newly-broken (regressions), "
        f"{len(known_unreachable)} known-unreachable (excluded_sourceless={report['excluded_sourceless_count']})",
        flush=True,
    )
    if changed:
        print(f"[drift-cron] CHANGED: {changed}", flush=True)
    if regressions:
        print(f"[drift-cron] REGRESSIONS (previously succeeded, now failing): {regressions}", flush=True)


async def arq_task_cron_diagnostic(ctx: dict) -> None:
    """
    TEMPORARY (2026-08-16) — investigating why arq_task_refresh_entsoe_prices
    (previously reliable) and arq_task_source_drift_check (new) both show
    zero confirmed firings since the 2026-08-14 17:24 UTC deploy. Fires
    every minute (bare cron() default: no month/day/weekday/hour
    restriction, second=0) so a real production cron failure is
    observable within ~1-2 minutes of this deploy, instead of waiting for
    the next real 03:00 UTC window. Remove once root cause is found — see
    _CRON_WATCHDOG_ENABLED below, which also needs turning off with this.
    """
    from datetime import datetime, timezone
    print(f"[cron-diagnostic] tick fired at {datetime.now(timezone.utc).isoformat()}", flush=True)


@app.get("/api/proofread/{job_id}")
async def proofread_status(job_id: str):
    """Oikolukutehtävän tila: pending | running | done | error | insufficient_sources."""
    job = _proofread_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Task not found")
    _status = job.get("status", "running")
    if _status == "insufficient_sources":
        raise HTTPException(
            status_code=422,
            detail={
                "error":         "insufficient_sources",
                "message":       (
                    "Riittämätön lähdeaineisto — RAG-tietokanta ei palauttanut riittävästi "
                    "relevantteja lähteitä luotettavan lupahakemusluonnoksen tuottamiseen. "
                    "Kokeile eri hanketyyppiä tai ota yhteyttä info@ncenergy.fi."
                ),
                "chunks_found":  job.get("chunks_found", 0),
                "avg_relevance": job.get("avg_relevance", 0.0),
            },
        )
    return {
        "status": _status,
        "error": job.get("error"),
        "debug_sections": job.get("debug_sections"),
        "phase_status": job.get("phase_status"),
    }


_FILE_PREFIX = {"FI": "hakemus", "EN": "application", "SE": "ansökan",
                "DA": "ansøgning", "NO": "søknad", "PL": "wniosek", "LV": "pieteikums"}


def _fn(s: str) -> str:
    """Sanitize a string for use in Content-Disposition filename (ASCII-safe)."""
    nfkd = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-zA-Z0-9]", "_", nfkd.encode("ascii", "ignore").decode("ascii"))


@app.get("/api/proofread/{job_id}/download")
async def proofread_download(job_id: str):
    """Lataa oikoluvun jälkeinen PDF."""
    job = _proofread_store.get(job_id)
    if job is None or job["status"] != "done" or not job["pdf_bytes"]:
        raise HTTPException(status_code=404, detail="PDF not ready yet")
    prefix  = _FILE_PREFIX.get(job.get("lang", "FI"), "hakemus")
    _kt     = _fn(job.get("hanketyyppi", "doc"))
    _kunta  = _fn(job.get("kunta", "hanke"))
    return Response(
        content    = job["pdf_bytes"],
        media_type = "application/pdf",
        headers    = {"Content-Disposition": f'attachment; filename="{prefix}_{_kt}_{_kunta}.pdf"'},
    )


@app.post("/api/permit-ai")
@limiter.limit("50/hour")
async def permit_ai(request: Request, req: PermitAIRequest):
    """RAG-pohjainen lupaprosessikysely. Hakee Fingrid/Pelastusopisto/Tukes-dokumenteista."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = await asyncio.to_thread(
            query_permit_ai, req.question, req.n_results
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Permit AI -virhe: {exc}")


@app.post("/api/chat")
@limiter.limit("60/hour")
async def permit_chat(request: Request, req: PermitChatRequest):
    """
    Stateful RAG chat with per-session conversation history.
    Filters RAG results by hanketyyppi + country for relevance.
    Body: {question, session_id, n_results, hanketyyppi, country}
    Returns: {answer, sources}
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    session_id = req.session_id or get_remote_address(request)
    try:
        result = await asyncio.to_thread(
            query_permit_ai_chat,
            req.question, session_id, req.n_results,
            req.hanketyyppi, req.country,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat-virhe: {exc}")


# ── Site Optimizer ───────────────────────────────────────────────────────────

def _lcg(s: int) -> int:
    return (1664525 * s + 1013904223) % (2 ** 32)


def _rng01(seed: int, offset: int = 0) -> float:
    s = seed
    for _ in range(offset + 1):
        s = _lcg(s)
    return s / (2 ** 32)


@app.post("/api/optimize-bess")
@limiter.limit("20/hour")
async def optimize_sites(request: Request, req: OptimizeRequest):
    """
    Sijaintioptimointityökalu — pisteyttää kandidaattisijainteja hanketyypeittäin.
    Hanketyypit: bess, tuulivoima, aurinkovoima, smr
    """
    if not _OPTIMIZER_OK:
        raise HTTPException(status_code=501, detail="optimizer.py not found")

    _allowed = {"bess", "tuulivoima", "aurinkovoima", "smr"}
    if req.project_type not in _allowed:
        raise HTTPException(status_code=400,
                            detail=f"project_type oltava: {', '.join(sorted(_allowed))}")
    if len(req.bbox) != 4:
        raise HTTPException(status_code=400,
                            detail="bbox: [lat_min, lon_min, lat_max, lon_max]")

    lat_min, lon_min, lat_max, lon_max = req.bbox

    # Suomen maa-alueen karkea bounding box (manner + saaret, ei Itämeri/ulkomaat)
    _FI_LAT_MIN, _FI_LAT_MAX = 59.5, 70.1
    _FI_LON_MIN, _FI_LON_MAX = 19.5, 31.6

    # Tarkista että bbox on Suomen sisällä — hylkää jos täysin ulkopuolella
    if (lat_max < _FI_LAT_MIN or lat_min > _FI_LAT_MAX or
            lon_max < _FI_LON_MIN or lon_min > _FI_LON_MAX):
        raise HTTPException(
            status_code=400,
            detail=(
                "bbox on Suomen maa-alueen ulkopuolella. "
                f"Sallittu alue: lat {_FI_LAT_MIN}–{_FI_LAT_MAX}, "
                f"lon {_FI_LON_MIN}–{_FI_LON_MAX}."
            ),
        )

    # Leikkaa bbox Suomen rajoihin (jos käyttäjä antoi osittain ulkopuolisen alueen)
    lat_min = max(lat_min, _FI_LAT_MIN)
    lat_max = min(lat_max, _FI_LAT_MAX)
    lon_min = max(lon_min, _FI_LON_MIN)
    lon_max = min(lon_max, _FI_LON_MAX)

    def _inside_finland(lat: float, lon: float) -> bool:
        """Karkea maa-alue-check Suomelle. Hylkää ilmiselvästi meren tai ulkomaan pisteet."""
        if not (_FI_LAT_MIN <= lat <= _FI_LAT_MAX and _FI_LON_MIN <= lon <= _FI_LON_MAX):
            return False
        # Poista Suomenlahden eteläinen meri-alue (Viro/Latvia): lat<59.8 + lon<27
        if lat < 59.8 and lon < 27.0:
            return False
        # Poista Ruotsin puoli (Merenkurkku + Pohjanlahti): lon<20.5 ja lat<65
        if lon < 20.5 and lat < 65.0:
            return False
        return True

    # Generoi 16 kandidaattisijaintia 4×4-gridillä bbox:n sisältä
    _rows, _cols = 4, 4
    sites: list = []
    skipped_sea: list = []
    _col_labels = "ABCDE"
    for i in range(_rows):
        for j in range(_cols):
            lat = lat_min + (lat_max - lat_min) * (i + 0.5) / _rows
            lon = lon_min + (lon_max - lon_min) * (j + 0.5) / _cols
            if not _inside_finland(lat, lon):
                skipped_sea.append(f"{_col_labels[j]}{i + 1}")
                continue
            _seed = int(abs(lat * 1e4)) * 99991 + int(abs(lon * 1e4)) * 31337
            r = lambda off: _rng01(_seed, off)
            sites.append(EnergySite(
                site_id      = f"{_col_labels[j]}{i + 1}",
                lat          = round(lat, 5),
                lon          = round(lon, 5),
                solar_irradiance    = 700 + r(1) * 400,
                wind_resource       = 4.0 + r(2) * 5.0,
                grid_distance_km    = 0.5 + r(3) * 44.5,
                land_area_ha        = max(req.min_area_ha, 2 + r(4) * 58),
                zoning_score        = 0.15 + r(5) * 0.85,
                protected_area_score= 0.10 + r(6) * 0.90,
                water_access_score  = 0.30 + r(7) * 0.70,
                land_cost_eur_ha    = 5000 + r(8) * 30000,
            ))

    if not sites:
        raise HTTPException(
            status_code=400,
            detail="Kaikki kandidaattisijannit osuivat meri- tai ulkomaa-alueelle. Tarkista bbox.",
        )

    optimizer = NCEOptimizer(req.project_type)
    result    = optimizer.optimize(sites)

    top5 = [
        {
            "site_id":              site.site_id,
            "lat":                  site.lat,
            "lon":                  site.lon,
            "score":                score,
            "score_pct":            f"{score:.0%}",
            "grid_distance_km":     round(site.grid_distance_km, 1),
            "zoning_score":         round(site.zoning_score, 2),
            "protected_area_score": round(site.protected_area_score, 2),
            "solar_irradiance":     round(site.solar_irradiance),
            "wind_resource":        round(site.wind_resource, 1),
            "land_area_ha":         round(site.land_area_ha, 1),
        }
        for site, score in zip(result.ranked_sites[:5], result.scores[:5])
    ]

    resp: dict = {
        "results":          top5,
        "optimizer_used":   result.optimizer_used,
        "project_type":     result.project_type,
        "total_candidates": len(sites),
    }
    if skipped_sea:
        resp["skipped_outside_finland"] = skipped_sea
    return resp


# ── Sisäinen analyysilogiikka ─────────────────────────────────────────────────

async def _run_analysis(
    kiinteistotunnus: str,
    api_key: str = "",
    prop: Optional[dict] = None,
    grid_connection: str = "",
) -> dict:
    """
    Kokonaisanalyysi. prop=None → haetaan WFS:stä kerran.
    grid_connection: ohjaa verkkoetäisyyden suodatusta
      "Fingrid 110" → vain ≥100 kV johdot
      "Fingrid 400" → vain ≥380 kV johdot
      muu (jakeluverkko) → ≤25 kV tai tagittomat johdot
    """
    key = api_key or MML_API_KEY
    muni_code = kiinteistotunnus.split("-")[0].zfill(3)

    if prop is None:
        try:
            prop = await get_property_boundaries(kiinteistotunnus, api_key=key)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Property lookup failed: {exc}")

    kuntanimi  = _prop_kuntanimi(prop)
    kylanimi   = _prop_kylanimi(prop)
    kuntanimi_gen = genitive(kuntanimi) if kuntanimi and not kuntanimi.startswith("Kunta ") else kuntanimi
    pelastuslaitos_name = get_pelastuslaitos(muni_code)
    ely_center_name     = get_ely(muni_code)

    center_lat, center_lon = _centroid(prop)
    area_m2 = _area(prop)
    wide_bbox   = (center_lon - 0.20, center_lat - 0.15, center_lon + 0.20, center_lat + 0.15)
    narrow_bbox = (center_lon - 0.005, center_lat - 0.005, center_lon + 0.005, center_lat + 0.005)
    bldg_delta  = 0.010
    bldg_bbox   = (center_lon - bldg_delta, center_lat - bldg_delta,
                   center_lon + bldg_delta, center_lat + bldg_delta)

    substation_wide = (center_lon - 0.30, center_lat - 0.20, center_lon + 0.30, center_lat + 0.20)

    (grid_data, natura_data, gw_data, bldg_data,
     zoning_data, heritage_data, highway_data,
     flood_data, substation_data) = await asyncio.gather(
        get_transmission_lines(wide_bbox),
        get_natura_areas(wide_bbox),
        get_groundwater_areas(wide_bbox),
        get_buildings(bldg_bbox),
        get_zoning_info(wide_bbox, api_key=key),
        get_heritage_sites(wide_bbox),
        get_highways(wide_bbox),
        get_flood_risk(wide_bbox),
        get_substations(substation_wide),
        return_exceptions=True,
    )

    grid_data       = grid_data       if isinstance(grid_data,       dict) else {"features": []}
    natura_data     = natura_data     if isinstance(natura_data,     dict) else {"features": []}
    gw_data         = gw_data         if isinstance(gw_data,         dict) else {"features": [], "unavailable": True}
    bldg_data       = bldg_data       if isinstance(bldg_data,       dict) else {"features": []}
    zoning_data     = zoning_data     if isinstance(zoning_data,     dict) else {"features": [], "unavailable": True}
    heritage_data   = heritage_data   if isinstance(heritage_data,   dict) else {"features": [], "unavailable": True}
    highway_data    = highway_data    if isinstance(highway_data,    dict) else {"features": []}
    flood_data      = flood_data      if isinstance(flood_data,      dict) else {"flood_overlap": False, "unavailable": True}
    substation_data = substation_data if isinstance(substation_data, dict) else {"features": []}

    # Etäisyyssuodatus verkkotypin mukaan
    if "Fingrid 400" in grid_connection:
        nearest_grid_m = nearest_line_distance_m(center_lat, center_lon, grid_data, min_voltage_kv=380)
    elif "Fingrid 110" in grid_connection or "110 kV" in grid_connection:
        nearest_grid_m = nearest_line_distance_m(center_lat, center_lon, grid_data, min_voltage_kv=100)
    else:
        # Jakeluverkko: ≤25 kV tai tagittomat (voltage_kv=0)
        nearest_grid_m = nearest_line_distance_m(center_lat, center_lon, grid_data, max_voltage_kv=25)

    natura_overlap      = len(natura_data.get("features", [])) > 0
    gw_overlap          = len(gw_data.get("features", [])) > 0
    gw_unavailable      = gw_data.get("unavailable", False)
    gw_class            = _worst_gw_class(gw_data.get("features", [])) if gw_overlap else ""
    heritage_overlap    = len(heritage_data.get("features", [])) > 0
    heritage_unavailable= heritage_data.get("unavailable", False)
    heritage_source     = heritage_data.get("source", "none")
    heritage_note       = heritage_data.get("note", "")
    zoning_unavailable  = zoning_data.get("unavailable", False)
    nearest_bldg_m      = nearest_point_distance_m(center_lat, center_lon, bldg_data)
    nearest_road_m      = nearest_line_distance_m(center_lat, center_lon, highway_data)
    road_protection_ok  = nearest_road_m < 0 or nearest_road_m >= 20.0
    nearest_road_name   = _nearest_road_name(center_lat, center_lon, highway_data)
    if zoning_data.get("unavailable"):
        osm_zone = await infer_zoning_from_osm(center_lat, center_lon)
        zoning_data["osm_inference"] = osm_zone
    zoning_status, zoning_ok = _eval_zoning(zoning_data)
    land_use            = _dominant_land_use(await get_land_use(narrow_bbox, api_key=key))
    grid_type           = _best_line_type(center_lat, center_lon, grid_data)
    powerline_buffer_ok = nearest_grid_m < 0 or nearest_grid_m >= 25.0

    # Maaperä (GTK)
    soil_data           = await get_soil_type(center_lat, center_lon)
    maaperalaaji        = soil_data.get("maaperalaaji", "Ei tiedossa")
    soil_score_pts      = soil_data.get("score_pts")   # None = N/A

    # Tulvavaara (SYKE)
    flood_overlap       = flood_data.get("flood_overlap", False)
    flood_unavailable   = flood_data.get("unavailable", False)

    # Sähköasema (lähin, OSM)
    sub_info            = nearest_substation_info(center_lat, center_lon, substation_data)
    nearest_substation_m = sub_info.get("distance_m")
    nearest_substation_name = sub_info.get("name")

    # Lupapiste-URL
    lupapiste_url = f"https://www.lupapiste.fi/?municipality={muni_code}"

    scores = _score(
        nearest_grid_m=nearest_grid_m,
        gw_overlap=gw_overlap,
        gw_unavailable=gw_unavailable,
        gw_class=gw_class,
        natura_overlap=natura_overlap,
        zoning_ok=zoning_ok,
        zoning_unavailable=zoning_unavailable,
        nearest_bldg_m=nearest_bldg_m,
        heritage_overlap=heritage_overlap,
        heritage_unavailable=heritage_unavailable,
        road_protection_ok=road_protection_ok,
        flood_overlap=flood_overlap,
        flood_unavailable=flood_unavailable,
        soil_score_pts=soil_score_pts,
    )

    # Lupaprosessianalyysi generoidaan PDF:ssä datapohjaisen templaten kautta —
    # Claude API:ta ei kutsuta enää tässä vaiheessa.
    ai_result = {}

    return {
        "kiinteistotunnus": kiinteistotunnus,
        "kuntanimi":        kuntanimi,
        "kuntanimi_gen":    kuntanimi_gen,
        "kylanimi":         kylanimi,
        "muni_code":        muni_code,
        "pelastuslaitos":   pelastuslaitos_name,
        "ely_center":       ely_center_name,
        "lupapiste_url":    lupapiste_url,
        "center_lat": round(center_lat, 6),
        "center_lon": round(center_lon, 6),
        "area_m2": area_m2,
        "area_ha": round(area_m2 / 10_000, 2) if area_m2 else None,
        "nearest_grid_m": round(nearest_grid_m) if nearest_grid_m >= 0 else None,
        "powerline_buffer_ok": powerline_buffer_ok,
        "grid_status": _grid_status(nearest_grid_m),
        "grid_type": grid_type,
        "nearest_substation_m":    nearest_substation_m,
        "nearest_substation_name": nearest_substation_name,
        "groundwater_overlap": gw_overlap,
        "groundwater_unavailable": gw_unavailable,
        "groundwater_class": gw_class,
        "natura_overlap": natura_overlap,
        "heritage_overlap": heritage_overlap,
        "heritage_unavailable": heritage_unavailable,
        "heritage_source": heritage_source,
        "heritage_note": heritage_note,
        "nearest_road_m": round(nearest_road_m) if nearest_road_m >= 0 else None,
        "nearest_road_name": nearest_road_name,
        "road_protection_ok": road_protection_ok,
        "zoning_status": zoning_status,
        "zoning_unavailable": zoning_unavailable,
        "zoning_ok": zoning_ok,
        "nearest_building_m": round(nearest_bldg_m) if nearest_bldg_m >= 0 else None,
        "land_use": land_use,
        "maaperalaaji":       maaperalaaji,
        "maaperalaaji_source": soil_data.get("source", "unavailable"),
        "flood_overlap":      flood_overlap,
        "flood_unavailable":  flood_unavailable,
        "ai_strategy":        ai_result.get("strategy"),
        "ai_strategy_error":  ai_result.get("error"),
        "bess_score":           scores["total"],
        "score_grid":           scores["grid"],
        "score_groundwater":    scores["gw"],
        "score_natura":         scores["natura"],
        "score_zoning":         scores["zoning"],
        "score_settlement":     scores["settlement"],
        "score_heritage":       scores["heritage"],
        "score_road":           scores["road"],
        "score_flood":          scores["flood"],
        "score_soil":           scores["soil"],
    }


# ── Apufunktiot ───────────────────────────────────────────────────────────────

_SOIL_MAP: dict[str, tuple[str, int]] = {
    "kallio":  ("Kallio",  5),
    "moreeni": ("Moreeni", 4),
    "hiekka":  ("Hiekka",  3),
    "savi":    ("Savi",    1),
    "turve":   ("Turve",   0),
}


def _apply_manual_overrides(analysis: dict, req: "ReportRequest") -> dict:
    """
    Soveltaa manuaaliset syötteet analyysidict:iin ja laskee pisteytyksen uudelleen.
    Käytetään vain PDF-raportin generoinnissa.
    """
    a = dict(analysis)

    if req.manual_kaavoitus and req.manual_kaavoitus not in ("", "ei_tietoa"):
        a["zoning_unavailable"] = False
        if req.manual_kaavoitus == "asemakaava":
            a["zoning_ok"]     = False
            a["zoning_status"] = "Asemakaava (manuaalinen syöte)"
        elif req.manual_kaavoitus == "yleiskaava":
            a["zoning_ok"]     = True
            a["zoning_status"] = "Yleiskaava (manuaalinen syöte)"
        else:  # ei_kaavaa
            a["zoning_ok"]     = True
            a["zoning_status"] = "Ei kaavaa (manuaalinen syöte)"
        a["manual_kaavoitus"] = req.manual_kaavoitus

    if req.manual_tulvavaara and req.manual_tulvavaara not in ("", "ei_tietoa"):
        a["flood_unavailable"] = False
        a["flood_overlap"]     = (req.manual_tulvavaara == "kyllä")
        a["manual_tulvavaara"] = req.manual_tulvavaara

    soil_score_override: Optional[int] = None
    if req.manual_maapera and req.manual_maapera not in ("", "ei_tietoa") and req.manual_maapera in _SOIL_MAP:
        nimi, pts = _SOIL_MAP[req.manual_maapera]
        a["maaperalaaji"]        = nimi
        a["maaperalaaji_source"] = "manual"
        a["manual_maapera"]      = req.manual_maapera
        soil_score_override      = pts

    if req.manual_pinta_ala_ha is not None and req.manual_pinta_ala_ha > 0:
        a["area_ha"]             = req.manual_pinta_ala_ha
        a["manual_pinta_ala_ha"] = req.manual_pinta_ala_ha

    # Pisteytys uudelleen
    soil_pts = soil_score_override if soil_score_override is not None else a.get("score_soil")
    scores = _score(
        nearest_grid_m   = a.get("nearest_grid_m") if a.get("nearest_grid_m") is not None else -1,
        gw_overlap       = a.get("groundwater_overlap", False),
        gw_unavailable   = a.get("groundwater_unavailable", False),
        gw_class         = a.get("groundwater_class", ""),
        natura_overlap   = a.get("natura_overlap", False),
        zoning_ok        = a.get("zoning_ok", True),
        zoning_unavailable = a.get("zoning_unavailable", False),
        nearest_bldg_m   = a.get("nearest_building_m") if a.get("nearest_building_m") is not None else -1,
        heritage_overlap = a.get("heritage_overlap", False),
        heritage_unavailable = a.get("heritage_unavailable", False),
        road_protection_ok = a.get("road_protection_ok", True),
        flood_overlap    = a.get("flood_overlap", False),
        flood_unavailable = a.get("flood_unavailable", False),
        soil_score_pts   = soil_pts,
    )
    a.update({
        "bess_score":        scores["total"],
        "score_grid":        scores["grid"],
        "score_groundwater": scores["gw"],
        "score_natura":      scores["natura"],
        "score_zoning":      scores["zoning"],
        "score_settlement":  scores["settlement"],
        "score_heritage":    scores["heritage"],
        "score_road":        scores["road"],
        "score_flood":       scores["flood"],
        "score_soil":        scores["soil"],
    })
    return a


def _centroid(geojson: dict) -> tuple[float, float]:
    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords:
            continue
        ring = coords[0] if geom.get("type") == "Polygon" else (coords[0][0] if coords else None)
        if ring:
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            return sum(lats) / len(lats), sum(lons) / len(lons)
    return 60.6833, 22.5333   # Pöytyä default


def _area(geojson: dict) -> Optional[float]:
    for feat in geojson.get("features", []):
        v = (feat.get("properties") or {}).get("pinta_ala")
        if v:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


def _eval_zoning(zoning_data: dict) -> tuple[str, bool]:
    if not zoning_data.get("unavailable"):
        feats = zoning_data.get("features", [])
        if not feats:
            return "Ei asemakaavaa — tarkistettu MML WFS", True
        types = [(f.get("properties") or {}).get("kaavatyyppi", "") for f in feats]
        if any("asemakaava" in t.lower() for t in types):
            return "Asemakaava-alue", False
        return "Yleiskaava / maakuntakaava", True
    # MML ei saatavilla — käytetään OSM-päättelyä
    osm = zoning_data.get("osm_inference", {})
    inferred = osm.get("inferred", "unknown")
    if inferred == "asemakaava":
        return "Todennäköisesti asemakaava-alue (OSM-päättely)", False
    if inferred == "rural":
        return "Haja-asutusalue (OSM-päättely — tarkista MML WFS)", True
    return "Ei saatavilla (MML API-avain puuttuu)", True


def _nearest_road_name(lat: float, lon: float, highway_geojson: dict) -> str:
    from fingrid_api import _haversine_m, _point_to_segment_m
    best_name, best_dist = "", float("inf")
    for feat in highway_geojson.get("features", []):
        coords = feat.get("geometry", {}).get("coordinates", [])
        p = feat.get("properties") or {}
        name = p.get("name") or p.get("ref") or p.get("highway") or ""
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            d = _point_to_segment_m(lat, lon, lat1, lon1, lat2, lon2)
            if d < best_dist:
                best_dist = d
                best_name = name
    return best_name


def _parse_gw_class(luokka_text: str) -> str:
    """'Vedenhankintaa... (1E)' → '1E'.  Palauttaa '' jos ei tunnisteta."""
    m = re.search(r'\(([12E]+E?)\)\s*$', luokka_text.strip())
    return m.group(1) if m else ""


def _worst_gw_class(features: list) -> str:
    """Pahin pohjavesiluokka (1 > 1E > 2E > 2 > E) annetuista piirtein."""
    classes = {_parse_gw_class((f.get("properties") or {}).get("luokka", ""))
               for f in features} - {""}
    for prio in ("1", "1E", "2E", "2", "E"):
        if prio in classes:
            return prio
    return ""


_LANDUSE_FI_MAIN: dict[str, str] = {
    "farmland":    "Peltoalue",
    "forest":      "Metsäalue",
    "meadow":      "Niitty / laidun",
    "residential": "Asuinalue",
    "commercial":  "Kaupallinen alue",
    "industrial":  "Teollisuusalue",
    "retail":      "Vähittäiskauppa",
}


def _dominant_land_use(landuse_data: dict) -> str:
    feats = landuse_data.get("features", [])
    if not feats:
        return "Maatalousmaa / metsä (oletus)"
    p = (feats[0].get("properties") or {})
    raw = p.get("kohdeluokka") or p.get("luokka") or ""
    # kohdeluokka on jo suomeksi jos tuli OSM-fallbackista (mml_api käänsi)
    return raw or _LANDUSE_FI_MAIN.get(p.get("kohdeluokka_osm", ""), "Maatalousmaa / metsä")


def _prop_kuntanimi(prop: dict) -> str:
    for feat in (prop or {}).get("features", []):
        v = (feat.get("properties") or {}).get("kuntanimi", "")
        if v:
            return v
    return "–"


def _prop_kylanimi(prop: dict) -> str:
    for feat in (prop or {}).get("features", []):
        v = (feat.get("properties") or {}).get("kylanimi", "")
        if v and v != "–":
            return v
    return "–"


def _best_line_type(lat: float, lon: float, grid_geojson: dict) -> str:
    """Palauttaa lähimmän power-elementin tyypin (johto tai pylväs)."""
    from fingrid_api import _haversine_m, _extract_line_coords, _point_to_segment_m
    best_type, best_dist = "–", float("inf")
    for feat in grid_geojson.get("features", []):
        props = feat.get("properties") or {}
        geom  = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        if gtype == "Point":
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                d = _haversine_m(lat, lon, coords[1], coords[0])
                if d < best_dist:
                    best_dist = d
                    best_type = props.get("line_type", "–")
        else:
            for seg in _extract_line_coords(geom):
                for i in range(len(seg) - 1):
                    d = _point_to_segment_m(lat, lon, seg[i][1], seg[i][0], seg[i+1][1], seg[i+1][0])
                    if d < best_dist:
                        best_dist = d
                        best_type = props.get("line_type", "–")
    return best_type


def _grid_status(nearest_grid_m: float) -> str:
    if nearest_grid_m < 0:
        return "Ei dataa"
    if nearest_grid_m < 1_000:
        return "Erinomainen ✓"
    if nearest_grid_m < 2_000:
        return "Hyvä ✓"
    return "Tarkista — pyydä liityntätarjous Carunalta"


def _score(
    nearest_grid_m: float,
    gw_overlap: bool,
    gw_unavailable: bool,
    gw_class: str = "",
    natura_overlap: bool = False,
    zoning_ok: bool = True,
    zoning_unavailable: bool = False,
    nearest_bldg_m: float = -1,
    heritage_overlap: bool = False,
    heritage_unavailable: bool = False,
    road_protection_ok: bool = True,
    flood_overlap: bool = False,
    flood_unavailable: bool = True,
    soil_score_pts: Optional[int] = None,
) -> dict:
    """
    Pisteytys (max 110p, normalisoidaan 100:aan käytettävissä olevilla kriteereillä):
      Verkkoliityntä:   30p  (<1km=30, 1-2km=20, >2km=5)
      Pohjavesiluokka:  20p  (ei=20, luokka2/E=8, luokka1=0, N/A pois)
      Natura 2000:      15p  (ei=15, on=0)
      Ei asemakaavaa:   10p  (maaseutu=10, asemakaava=3, N/A pois)
      Asutus >300m:     10p  (>300m=10, 150-300m=5, <150m=0)
      Ei muinaismuistoja:10p (ei=10, on=0, N/A pois)
      Tiesuoja-alue OK:  5p  (ok=5, ei ok=0)
      Tulvavaara:        5p  (tulossa – ei dataa = N/A, pois indeksistä)
      Maaperä:           5p  (tulossa – ei dataa = N/A, pois indeksistä)
    """
    # Verkkoliityntä (30p)
    if nearest_grid_m < 0:
        grid = 15
    elif nearest_grid_m < 1_000:
        grid = 30
    elif nearest_grid_m < 2_000:
        grid = 20
    else:
        grid = 5

    # Pohjavesiluokka (20p) — N/A kun SYKE offline
    if gw_unavailable:
        gw = None
    elif not gw_overlap:
        gw = 20
    elif gw_class in ("1", "1E"):
        gw = 0
    else:
        gw = 8

    # Natura (15p)
    natura = 0 if natura_overlap else 15

    # Kaavoitus (10p) — N/A kun MML-avain puuttuu
    if zoning_unavailable:
        zoning = None
    else:
        zoning = 10 if zoning_ok else 3

    # Asutus (10p)
    if nearest_bldg_m < 0:
        settlement = 5
    elif nearest_bldg_m > 300:
        settlement = 10
    elif nearest_bldg_m > 150:
        settlement = 5
    else:
        settlement = 0

    # Muinaismuistot (10p) — N/A kun kumpikaan rajapinta ei vastaa
    if heritage_unavailable:
        heritage = None
    else:
        heritage = 0 if heritage_overlap else 10

    # Tiesuoja-alue (5p)
    road = 5 if road_protection_ok else 0

    # Tulvavaara (5p) — N/A kun SYKE offline
    if flood_unavailable:
        flood = None
    else:
        flood = 0 if flood_overlap else 5

    # Maaperä (5p) — N/A kun GTK ei saatavilla
    soil = soil_score_pts  # jo laskettu gtk_api:ssa (None = N/A)

    # Normalisointi: lasketaan vain niiden kriteerien yli joille on dataa
    components = [
        (grid,       30),
        (gw,         20),
        (natura,     15),
        (zoning,     10),
        (settlement, 10),
        (heritage,   10),
        (road,        5),
        (flood,       5),
        (soil,        5),
    ]
    achieved = sum(v for v, _ in components if v is not None)
    max_pts  = sum(m for v, m in components if v is not None)
    total    = min(round(achieved / max_pts * 100), 100) if max_pts else 0

    return {
        "total": total,
        "grid": grid, "gw": gw, "natura": natura,
        "zoning": zoning, "settlement": settlement,
        "heritage": heritage, "road": road,
        "flood": flood, "soil": soil,
    }


def _render_static_map(
    prop: dict,
    center_lat: float,
    center_lon: float,
    zoom: int = 16,
) -> bytes:
    """
    Synkroninen apufunktio (ajetaan to_thread:ssa).
    Käyttää staticmap-kirjastoa OSM-tiilejä vasten.
    Piirtää kiinteistörajan punaisena viivana.
    """
    from staticmap import StaticMap, Line, CircleMarker

    m = StaticMap(
        800, 500,
        headers={"User-Agent": "bess-tool/1.0 (BESS planning tool)"},
    )

    # Piirretään kaikki renkaat (Polygon / MultiPolygon)
    for feat in prop.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        rings: list = []
        if gtype == "Polygon":
            rings = geom.get("coordinates", [])
        elif gtype == "MultiPolygon":
            for polygon in geom.get("coordinates", []):
                rings.extend(polygon)
        for ring in rings:
            pts = [[c[0], c[1]] for c in ring if len(c) >= 2]
            if len(pts) >= 2:
                m.add_line(Line(pts, "#e94560", 4))

    # Centroid-piste
    m.add_marker(CircleMarker([center_lon, center_lat], "#e94560", 10))

    img = m.render(zoom=zoom, center=[center_lon, center_lat])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Phase-Lock ───────────────────────────────────────────────────────────────
PHASE_LOCK_ENABLED = os.getenv("PHASE_LOCK_ENABLED", "false").lower() == "true"
try:
    from phase_lock import (
        check_phase_allowed as _check_phase,
        get_phase_status as _get_phase_status,
        unlock_next_phase as _unlock_next_phase,
        skip_phases as _skip_phases,
        get_max_phase as _get_max_phase,
        PHASE_NAMES as _PHASE_NAMES,
    )
    _PHASE_LOCK_OK = PHASE_LOCK_ENABLED
except Exception as _pl_err:
    _PHASE_LOCK_OK = False

# ── IFC parser imports (optional — graceful if missing) ───────────────────────
# sys.path already has bess_tool/ root (line 50 above), so permit_ai namespace works
try:
    from ifc_parser import extract_ifc_data as _extract_ifc_data
    from ifc_to_permit import map_to_permit as _map_to_permit
    _IFC_OK = True
except Exception as _ifc_err:
    _IFC_OK = False
    _ifc_err_msg = str(_ifc_err)

_IFC_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@app.post("/api/parse-ifc")
@limiter.limit("20/hour")
async def parse_ifc(
    request: Request,
    file: UploadFile = File(...),
    project_type: str = Query(default="BESS"),
    country: str = Query(default="FI"),
):
    """
    Parse an IFC file and return permit-relevant fields, missing fields,
    and compliance flags. Accepts multipart/form-data, max 50 MB.
    """
    if not _IFC_OK:
        raise HTTPException(status_code=501, detail=f"ifcopenshell ei saatavilla: {_ifc_err_msg}")

    if not file.filename or not file.filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Tiedoston tulee olla .ifc-muodossa")

    content = await file.read()
    if len(content) > _IFC_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Tiedosto liian suuri (max 50 MB)")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    allowed_project_types = {
        "BESS", "AURINKO", "TUULI", "SMR", "DATAKESKUS",
        "SCO2", "VESIVOIMA", "YVA", "VERKKO",
    }
    if project_type not in allowed_project_types:
        raise HTTPException(
            status_code=400,
            detail=f"project_type oltava: {', '.join(sorted(allowed_project_types))}",
        )
    if country not in {"FI", "SE", "DA", "NO", "PL", "DE", "LV", "EE"}:
        raise HTTPException(status_code=400, detail="country oltava: FI, SE, DA, NO, PL, DE, LV, EE")

    ifc_data = _extract_ifc_data(content)
    permit_map = _map_to_permit(ifc_data, project_type=project_type, country=country)

    # PR C (tenant architecture, 2026-08-09): best-effort lifecycle
    # logging, inert unless TENANT_TRACKING_ENABLED=true AND a tenant
    # session cookie is present — same reasoning as PR B's tenant_db/
    # layer1.py wiring. detail carries the filename/project_type/country
    # for context; no generation_id exists at this point in the flow (IFC
    # parsing happens before, not during, a generation).
    from tenant_db.events import record_user_event
    record_user_event(
        request.session.get("tenant_id"), request.session.get("user_id"),
        event_type="ifc_upload",
        detail={"filename": file.filename, "project_type": project_type, "country": country},
    )

    # Add confidence score per field to response
    prefilled_with_conf = {
        field: {
            "value": info["value"],
            "confidence": info["confidence"],
        }
        for field, info in permit_map["prefilled_fields"].items()
    }

    return JSONResponse({
        "prefilled_fields":  prefilled_with_conf,
        "missing_fields":    permit_map["missing_fields"],
        "compliance_flags":  permit_map["compliance_flags"],
        "summary":           permit_map["summary"],
        "parse_errors":      ifc_data.get("parse_errors", []),
        "ifc_schema":        ifc_data.get("ifc_schema"),
        "filename":          file.filename,
    })


# ── Phase-Lock endpointit ─────────────────────────────────────────────────────

@app.get("/api/phase-status")
async def phase_status(
    session_id: str = Query(...),
    hanketyyppi: str = Query(...),
):
    """Palauttaa vaiheen tilan sessiolle ja hanketyypille."""
    if not _PHASE_LOCK_OK:
        # Phase lock disabled (demo mode) — all phases open, signal frontend to skip locks.
        # Derived from phase_lock.py (2026-08-12, P3-2) instead of a hand-copied
        # literal, so this stays correct per-hanketyyppi instead of silently
        # capping everyone at 3 regardless of what phase_lock.py now supports.
        # hanketyyppi is a required Query(...) param, always populated here
        max_phase = _get_max_phase(hanketyyppi)
        return JSONResponse({"completed_phase": 0, "next_phase": 1, "phase_lock_disabled": True, "phases": [
            {"name": _PHASE_NAMES[n], "phase": n, "state": "active"}
            for n in range(1, max_phase + 1)
        ]})
    if not session_id or not hanketyyppi:
        raise HTTPException(status_code=400, detail="session_id ja hanketyyppi vaaditaan")
    return JSONResponse(_get_phase_status(session_id, hanketyyppi))


class CompletePhaseRequest(BaseModel):
    session_id:  str
    hanketyyppi: str
    phase:       int   # 1..N, N = phase_lock.get_max_phase(hanketyyppi) — 3 for
                        # everyone except SMR (5), see phase_lock.py


@app.post("/api/complete-phase")
@limiter.limit("60/hour")
async def complete_phase(request: Request, req: CompletePhaseRequest):
    """Merkitsee vaiheen valmiiksi ja avaa seuraavan."""
    if not _PHASE_LOCK_OK:
        return JSONResponse({"ok": True, "next_phase": req.phase + 1})
    # 2026-08-12 (P3-2): generalized from a hardcoded `not in (1, 2, 3)` --
    # that literal would have silently blocked SMR from ever completing
    # phase 4/5 through the real API even after phase_lock.py itself (P3-1)
    # already supports it. Bound is per-hanketyyppi now, same as everywhere
    # else in this sequence.
    max_phase = _get_max_phase(req.hanketyyppi)
    if req.phase < 1 or req.phase > max_phase:
        raise HTTPException(status_code=400, detail=f"phase oltava 1-{max_phase} tälle hanketyypille")
    status = _unlock_next_phase(req.session_id, req.hanketyyppi, req.phase, "generated")
    return JSONResponse({"ok": True, **status})


class SkipPhaseRequest(BaseModel):
    session_id:         str
    hanketyyppi:        str
    skip_through_phase: int   # 1..N, N = phase_lock.get_max_phase(hanketyyppi) —
                               # merkitsee vaiheet 1..N ohitetuiksi


@app.post("/api/skip-phase")
@limiter.limit("30/hour")
async def skip_phase(request: Request, req: SkipPhaseRequest):
    """Merkitsee aiemmat vaiheet 'skipped' (asiakas liittyy kesken matkan)."""
    if not _PHASE_LOCK_OK:
        return JSONResponse({"ok": True, "next_phase": req.skip_through_phase + 1})
    # Same generalization/reasoning as /api/complete-phase above.
    max_phase = _get_max_phase(req.hanketyyppi)
    if req.skip_through_phase < 1 or req.skip_through_phase > max_phase:
        raise HTTPException(status_code=400, detail=f"skip_through_phase oltava 1-{max_phase} tälle hanketyypille")
    status = _skip_phases(req.session_id, req.hanketyyppi, req.skip_through_phase)
    return JSONResponse({"ok": True, **status})


# ── RTB / Compliance Cockpit ─────────────────────────────────────────────────

@app.get("/api/rtb/{hanke_id}")
async def rtb_status(hanke_id: str):
    """Palauttaa RTB-projektin molempien moduulien tilan ja valmius-indikaattorin."""
    return JSONResponse(_rtb.rtb_summary(hanke_id))


@app.get("/rtb")
async def rtb_cockpit():
    """RTB Compliance Cockpit -sivu."""
    path = os.path.join(_STATIC_DIR, "rtb.html")
    return FileResponse(path)


# ── Admin: RAG-indeksointi ────────────────────────────────────────────────────

def _check_ingest_auth(request: Request) -> None:
    if not _INGEST_SECRET:
        raise HTTPException(status_code=503, detail="INGEST_SECRET ei asetettu")
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization: Bearer <token> vaaditaan")
    if not secrets.compare_digest(auth[7:].encode(), _INGEST_SECRET.encode()):
        raise HTTPException(status_code=401, detail="Väärä Bearer-token")


@app.post("/api/admin/ingest")
async def admin_ingest(request: Request):
    """
    Käynnistää RAG-indeksoinnin taustasäikeessä.
    Authorization: Bearer <INGEST_SECRET>
    Body: {"countries": ["SE", "DA", "NO", "PL"], "reindex": false}
    """
    _check_ingest_auth(request)

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    _valid = {"SE", "DA", "NO", "PL", "EE", "DE"}
    raw_countries = body.get("countries", list(_valid))
    countries = [c.upper() for c in raw_countries if c.upper() in _valid]
    if not countries:
        raise HTTPException(status_code=400, detail=f"countries oltava jokin: {', '.join(sorted(_valid))}")
    reindex = bool(body.get("reindex", False))

    job_id = uuid.uuid4().hex[:10]
    _ingest_jobs[job_id] = {
        "status":      "running",
        "countries":   countries,
        "reindex":     reindex,
        "started_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": None,
        "result":      None,
        "error":       None,
        "log":         [],
    }

    def _bg_ingest():
        import contextlib
        import io as _sio
        log = _ingest_jobs[job_id]["log"]
        buf = _sio.StringIO()
        try:
            import ingest_countries as _ic
            with contextlib.redirect_stdout(buf):
                result = _ic.ingest(countries, dry_run=False, reindex=reindex)
            log.extend(buf.getvalue().splitlines())
            _ingest_jobs[job_id]["status"] = "done"
            _ingest_jobs[job_id]["result"] = result
        except SystemExit as exc:
            log.extend(buf.getvalue().splitlines())
            log.append(f"[VIRHE] sys.exit({exc.code})")
            _ingest_jobs[job_id]["status"] = "error"
            _ingest_jobs[job_id]["error"] = f"sys.exit({exc.code})"
        except Exception as exc:
            log.extend(buf.getvalue().splitlines())
            log.append(f"[VIRHE] {exc}")
            _ingest_jobs[job_id]["status"] = "error"
            _ingest_jobs[job_id]["error"] = str(exc)
        _ingest_jobs[job_id]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    Thread(target=_bg_ingest, daemon=True).start()

    return Response(
        content=json.dumps({"job_id": job_id, "countries": countries, "reindex": reindex}),
        status_code=202,
        media_type="application/json",
    )


@app.get("/api/admin/ingest/{job_id}")
async def admin_ingest_status(job_id: str, request: Request):
    """Tarkistaa ingest-tehtävän tilan. Vaatii saman Bearer-tokenin."""
    _check_ingest_auth(request)
    job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Tehtävää ei löydy")
    return JSONResponse({
        "job_id":      job_id,
        "status":      job["status"],
        "countries":   job["countries"],
        "started_at":  job["started_at"],
        "finished_at": job.get("finished_at"),
        "result":      job.get("result"),
        "error":       job.get("error"),
        "log_tail":    job["log"][-40:],
    })


_meta_update_jobs: dict = {}


@app.post("/api/admin/update-metadata")
async def admin_update_metadata(request: Request):
    """
    Päivittää permit_docs_v2:n chunk-metadatan in-place (doc_type + hanketyyppi_tag)
    ilman kokoelman poistoa — turvallinen käyttää tuotannossa.
    Authorization: Bearer <ADMIN_SECRET>
    """
    _check_ingest_auth(request)
    import uuid as _uuid
    job_id = _uuid.uuid4().hex[:12]
    _meta_update_jobs[job_id] = {"status": "running", "log": [],
                                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def _bg_update():
        from source_policy import get_doc_type as _get_doc_type, get_hanketyyppi_tag as _get_ht
        log = _meta_update_jobs[job_id]["log"]
        try:
            import chromadb as _chroma
            client = _chroma.PersistentClient(path=_DB_PATH)
            col = client.get_collection(_V2_COL)
            total = col.count()
            log.append(f"[update-meta] {total} chunks in {_V2_COL}")
            PAGE = 500
            updated = 0
            for offset in range(0, total, PAGE):
                page = col.get(limit=PAGE, offset=offset, include=["metadatas"])
                ids = page["ids"]
                metas = page["metadatas"] or [{}] * len(ids)
                new_metas = []
                for meta in metas:
                    m = dict(meta or {})
                    src = m.get("source", "")
                    if "doc_type" not in m or m["doc_type"] == "?":
                        m["doc_type"] = _get_doc_type(src)
                    if "hanketyyppi_tag" not in m:
                        # migrate: prefer existing project_types field, else look up by source name
                        pt = m.get("project_types", "")
                        m["hanketyyppi_tag"] = pt if pt and pt != "all" else _get_ht(src) or "general"
                    new_metas.append(m)
                col.update(ids=ids, metadatas=new_metas)
                updated += len(ids)
                log.append(f"[update-meta] {updated}/{total} updated")
            _meta_update_jobs[job_id]["status"] = "done"
            _meta_update_jobs[job_id]["result"] = f"{updated} chunks updated"
            log.append(f"[update-meta] Done: {updated} chunks updated")
        except Exception as exc:
            _meta_update_jobs[job_id]["status"] = "error"
            _meta_update_jobs[job_id]["error"] = str(exc)
            log.append(f"[update-meta] ERROR: {exc}")
        _meta_update_jobs[job_id]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    Thread(target=_bg_update, daemon=True).start()
    return JSONResponse({"job_id": job_id, "status": "running",
                         "message": f"Metadata update started for {_V2_COL}"})


@app.get("/api/admin/update-metadata/{job_id}")
async def admin_update_metadata_status(job_id: str, request: Request):
    _check_ingest_auth(request)
    job = _meta_update_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Tehtävää ei löydy")
    return JSONResponse({"job_id": job_id, "status": job["status"],
                         "result": job.get("result"), "error": job.get("error"),
                         "log_tail": job["log"][-20:]})


@app.post("/api/admin/rtb/seed")
async def admin_rtb_seed(request: Request):
    """
    Luo tai päivittää RTB-tietueen suoraan testikäyttöön.
    Authorization: Bearer <INGEST_SECRET>
    Body: {"hanke_id": "...", "permit_done": true, "land_use_done": true,
           "y_tunnus": "...", "kiinteistotunnus": "...", "hanketyyppi": "...", "maa": "FI"}
    """
    _check_ingest_auth(request)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    hanke_id = (body.get("hanke_id") or "").strip()
    if not hanke_id:
        raise HTTPException(status_code=400, detail="hanke_id vaaditaan")
    y_tunnus        = body.get("y_tunnus", "")
    kiinteistotunnus = body.get("kiinteistotunnus", "")
    hanketyyppi     = body.get("hanketyyppi", "")
    maa             = body.get("maa", "FI")
    updated = {}
    if body.get("permit_done", False):
        updated["permit_doc"] = _rtb.update_permit_doc(
            hanke_id, job_id="admin-seed", phase="admin",
            y_tunnus=y_tunnus, kiinteistotunnus=kiinteistotunnus,
            hanketyyppi=hanketyyppi, maa=maa,
        )
    if body.get("land_use_done", False):
        updated["land_use"] = _rtb.update_land_use(
            hanke_id, kiinteistotunnus=kiinteistotunnus,
            hanketyyppi=hanketyyppi, maa=maa,
        )
    return JSONResponse({"hanke_id": hanke_id, "summary": _rtb.rtb_summary(hanke_id)})


# ── IFC parser ────────────────────────────────────────────────────────────────

class IFCApprovalRequest(BaseModel):
    """Insinöörin hyväksymät IFC-kentät + hakemuksen perustiedot."""
    # Hakemuksen perustiedot
    hanketyyppi:       str
    kiinteistotunnus:  str
    teho_mw:           float = 0.0
    kapasiteetti_mwh:  float = 0.0
    kunta:             str
    hakija:            str
    lang:              str = "FI"
    country:           str = "FI"
    hankkeen_vaihe:    str = ""
    kohdeviranomainen: str = ""
    # Hyväksytyt IFC-kentät (insinööri on tarkistanut)
    approved_fields:   dict = {}
    # Audit trail
    reviewer_name:     str
    review_notes:      Optional[str] = None


@app.post("/api/approve-ifc")
@limiter.limit("10/hour")
async def approve_ifc(request: Request, req: IFCApprovalRequest):
    """
    Insinööri lähettää hyväksytyt IFC-kentät → generoi final PDF + audit trail.
    Palauttaa PDF binäärinä (application/pdf).
    """
    import datetime

    approved = req.approved_fields

    # Rakenna ApplicationInput IFC-esitäyttöarvoilla
    inp = ApplicationInput(
        hanketyyppi                   = req.hanketyyppi,
        kiinteistotunnus              = req.kiinteistotunnus,
        teho_mw                       = req.teho_mw,
        kapasiteetti_mwh              = req.kapasiteetti_mwh,
        kunta                         = req.kunta,
        hakija                        = req.hakija,
        lang                          = req.lang,
        country                       = req.country,
        hankkeen_vaihe                = req.hankkeen_vaihe,
        kohdeviranomainen             = req.kohdeviranomainen,
        ifc_floor_area                = float(approved.get("floor_area_total") or 0),
        ifc_building_height           = float(approved.get("building_height") or 0),
        ifc_fire_rating               = str(approved.get("fire_rating_walls") or ""),
        ifc_materials                 = ", ".join(approved.get("materials") or []),
        ifc_storeys                   = len(approved.get("storeys") or []),
        ifc_compliance_flags          = "\n".join(approved.get("compliance_flags") or []),
        generation_id                 = uuid.uuid4().hex[:10],
    )

    # PR B (tenant architecture, 2026-08-09): best-effort Layer 1 tracking,
    # inert unless TENANT_TRACKING_ENABLED=true AND a tenant session cookie
    # is present — see tenant_db/layer1.py's docstring.
    from tenant_db.layer1 import record_generation_start as _record_gen_start, record_report as _record_report
    _tenant_id = request.session.get("tenant_id")
    _project_id = _record_gen_start(
        _tenant_id, hanketyyppi=req.hanketyyppi, country=req.country, phase=req.hankkeen_vaihe,
    )

    # Generoi PDF taustasäikeessä (blocking — approve on harvinainen operaatio)
    loop = asyncio.get_event_loop()
    try:
        draft_bytes, sections, sources = await loop.run_in_executor(
            None, generate_application_draft, inp
        )
        pdf_bytes = await loop.run_in_executor(
            None, lambda: apply_proofread_to_pdf(inp, sections, sources)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    # PR B: same call as above, now that pdf_bytes succeeded — pdf_url/
    # raqs_score are None for the same reason noted at the other call sites
    # (tenant_db/layer1.py's docstring / the /api/generate-application site).
    _record_report(_tenant_id, _project_id, phase=req.hankkeen_vaihe, pdf_url=None, raqs_score=None)

    # Audit trail — lisätään PDF:n metatietoihin (ei sisältöön)
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    audit = {
        "timestamp":        timestamp,
        "reviewer_name":    req.reviewer_name,
        "review_notes":     req.review_notes or "",
        "approved_fields":  list(approved.keys()),
        "hanketyyppi":      req.hanketyyppi,
        "country":          req.country,
    }

    filename = (
        f"NCE_{req.hanketyyppi}_{req.kunta}_approved_"
        f"{timestamp[:10].replace('-','')}.pdf"
    )

    resp = Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-NCE-Audit-Timestamp":    audit["timestamp"],
            "X-NCE-Audit-Reviewer":     audit["reviewer_name"],
            "X-NCE-Audit-Fields":       ",".join(audit["approved_fields"]),
        },
    )
    return resp


@app.get("/api/permits")
async def get_permits(
    type: Optional[str] = Query(default=None, alias="type"),
    country: Optional[str] = Query(default=None),
):
    """
    Permit/authority configuration per project type and country.
    GET /api/permits                     → full config keyed by country
    GET /api/permits?type=bess&country=FI → resolved single entry (with FI fallback)
    """
    data_file = os.path.join(_BACKEND_DIR, "permits_data.json")
    with open(data_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    fi_base = all_data.get("FI", {})

    if type:
        country = (country or "FI").upper()
        country_data = all_data.get(country, {})
        resolved = country_data.get(type) or fi_base.get(type)
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Tyyppiä '{type}' ei löydy")
        kasittelyaika = fi_base.get(type, {}).get("kasittelyaika")
        return JSONResponse({"type": type, "country": country, "kasittelyaika": kasittelyaika, **resolved})

    # Full config: FI base + overrides keyed by country
    result = {"FI": fi_base}
    for cc, overrides in all_data.items():
        if cc != "FI":
            # Enrich each override entry with kasittelyaika from FI if not set
            enriched = {}
            for t, cfg in overrides.items():
                entry = dict(cfg)
                if "kasittelyaika" not in entry:
                    entry["kasittelyaika"] = fi_base.get(t, {}).get("kasittelyaika")
                enriched[t] = entry
            result[cc] = enriched
    return JSONResponse(result)


@app.get("/api/stats")
async def get_stats():
    # Direct SQLite read — bypasses the lru_cached ChromaDB client so count is
    # always current even after Shell ingest writes to the same persistent disk.
    try:
        import sqlite3 as _sqlite3
        _db_file = os.path.join(
            os.path.dirname(__file__),
            "..", "permit_ai", "embeddings", "chroma.sqlite3"
        )
        _db_file = os.path.normpath(_db_file)
        _con = _sqlite3.connect(_db_file, check_same_thread=False)
        chunk_count = _con.execute(
            "SELECT COUNT(*) FROM embeddings e"
            " JOIN segments s ON e.segment_id = s.id"
            " JOIN collections c ON s.collection = c.id"
            " WHERE c.name = 'permit_docs'"
        ).fetchone()[0]
        _con.close()
    except Exception:
        try:
            chunk_count = _get_chroma_col().count()
        except Exception:
            chunk_count = 0
    return {
        "chunks_total":  chunk_count,
        "countries":     6,
        "project_types": 20,
        "languages":     7,
    }


# ── Stripe payment routes ─────────────────────────────────────────────────────

class _CheckoutRequest(BaseModel):
    customer_email: str
    mode: str = "payment"   # "payment" | "subscription"


@app.post("/api/payments/checkout")
async def payments_checkout(req: _CheckoutRequest):
    """Create a Stripe Checkout Session. Returns {url, session_id}."""
    from stripe_payments import PAYMENT_ENABLED, create_checkout_session
    if not PAYMENT_ENABLED:
        raise HTTPException(status_code=503, detail="Payment system not enabled")
    try:
        return create_checkout_session(req.customer_email, req.mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/payments/webhook")
async def payments_webhook(request: Request, stripe_signature: str = Header(..., alias="stripe-signature")):
    """Stripe webhook endpoint. Set webhook URL to /api/payments/webhook in Stripe dashboard."""
    from stripe_payments import handle_webhook
    payload = await request.body()
    try:
        result = handle_webhook(payload, stripe_signature)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/payments/status/{session_id}")
async def payments_status(session_id: str):
    """Return payment status for a Checkout Session ID."""
    from stripe_payments import get_payment_status
    return {"session_id": session_id, "status": get_payment_status(session_id)}


# ── B2B API key authenticated report generation ───────────────────────────────

_ADMIN_SECRET = os.getenv("ADMIN_SECRET") or os.getenv("INGEST_SECRET", "")


def _require_admin(x_admin_secret: str = Header(..., alias="x-admin-secret")):
    if not _ADMIN_SECRET or x_admin_secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


class _B2BReportRequest(BaseModel):
    hanketyyppi:                  str
    kiinteistotunnus:             str
    teho_mw:                      float = 0.0
    kapasiteetti_mwh:             float = 0.0
    kunta:                        str
    hakija:                       str
    sijainti_ymparistovaikutukset: str = ""
    hankkeen_vaihe:               str = ""
    kohdeviranomainen:            str = ""
    lang:                         str = "FI"
    country:                      str = "FI"
    y_tunnus:                     str = ""
    osoite:                       str = ""


@app.post("/api/b2b/generate-report")
@limiter.limit("20/hour")
async def b2b_generate_report(
    request: Request,
    req: _B2BReportRequest,
    authorization: str = Header(...),
):
    """
    B2B synchronous report generation with API key auth.
    Returns PDF bytes immediately (no polling — designed for server-to-server calls).
    Pass API key as: Authorization: Bearer nce_<key>
    """
    from api_keys import verify_api_key
    from white_label import get_customer_logo_path, NCE_LOGO_PATH

    # Authenticate
    raw_key = authorization.removeprefix("Bearer ").strip()
    customer = verify_api_key(raw_key)
    if customer is None:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    # Resolve white-label assets
    logo_url    = customer.get("logo_url") or ""
    footer_name = customer.get("footer_name") or ""
    logo_path   = get_customer_logo_path(logo_url) if logo_url else NCE_LOGO_PATH

    allowed = {"BESS", "tuulivoima_maa", "tuulivoima_meri", "aurinkovoima", "SMR",
               "smr_bess", "vesivoima", "hybridi",
               "asuinrakennus", "teollisuus", "maatalous", "liikerakennus", "muu",
               "ymparistolupa", "datakeskus",
               "smr_se", "smr_no", "smr_da", "smr_de", "smr_ee",
               "egs", "offshore_wind"}
    if req.hanketyyppi not in allowed:
        raise HTTPException(status_code=400, detail=f"hanketyyppi oltava: {', '.join(sorted(allowed))}")

    inp = ApplicationInput(
        hanketyyppi                   = req.hanketyyppi,
        kiinteistotunnus              = req.kiinteistotunnus,
        teho_mw                       = req.teho_mw,
        kapasiteetti_mwh              = req.kapasiteetti_mwh,
        kunta                         = req.kunta,
        hakija                        = req.hakija,
        sijainti_ymparistovaikutukset = req.sijainti_ymparistovaikutukset,
        hankkeen_vaihe                = req.hankkeen_vaihe,
        kohdeviranomainen             = req.kohdeviranomainen,
        lang                          = req.lang,
        country                       = req.country,
        y_tunnus                      = req.y_tunnus,
        osoite                        = req.osoite,
        logo_path                     = logo_path,
        footer_name                   = footer_name or None,
        generation_id                 = uuid.uuid4().hex[:10],
    )

    try:
        import asyncio
        loop = asyncio.get_event_loop()
        draft_bytes, sections, sources = await loop.run_in_executor(
            None, generate_application_draft, inp
        )
        pdf = await loop.run_in_executor(
            None, lambda: apply_proofread_to_pdf(inp, sections, sources)
        )
    except InsufficientSourcesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    kt = req.kiinteistotunnus.replace("-", "_")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="NCE_{kt}.pdf"'},
    )


# ── Admin: API key management ─────────────────────────────────────────────────

class _CreateKeyRequest(BaseModel):
    company_name: str
    email:        str
    logo_url:     str = ""
    footer_name:  str = ""


@app.get("/api/admin/api-keys", dependencies=[Depends(_require_admin)])
async def admin_list_keys():
    """List all B2B API keys (no raw key values)."""
    from api_keys import list_api_keys
    return list_api_keys()


@app.post("/api/admin/api-keys", dependencies=[Depends(_require_admin)])
async def admin_create_key(req: _CreateKeyRequest):
    """Create a new B2B API key. Raw key shown once — store it securely."""
    from api_keys import create_api_key
    try:
        return create_api_key(req.company_name, req.email, req.logo_url, req.footer_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.delete("/api/admin/api-keys/{key_id}", dependencies=[Depends(_require_admin)])
async def admin_revoke_key(key_id: str):
    """Revoke a B2B API key by key_id."""
    from api_keys import revoke_api_key
    found = revoke_api_key(key_id)
    if not found:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"revoked": key_id}


# ── Caruna grid-capacity ingestion ───────────────────────────────────────────

@app.post("/api/admin/ingest-poland", dependencies=[Depends(_require_admin)])
async def admin_ingest_poland():
    """Download Polish regulatory PDFs/HTML and upsert chunks into ChromaDB. Admin only."""
    try:
        from poland_ingestion import ingest_poland_sources
        count = ingest_poland_sources()
        return {"status": "ok", "chunks_indexed": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/admin/ingest-poland-full", dependencies=[Depends(_require_admin)])
async def admin_ingest_poland_full():
    """
    Full Poland regulatory RAG ingestion — 10 sources, 280+ chunks expected.
    Uses requests.Session with browser UA to bypass ISAP Incapsula protection.
    ISAP PDFs only accessible from Render's Frankfurt IP (not local Mac).
    Admin only.
    """
    try:
        from poland_rag_full import ingest_poland_sources as _ingest
        count = _ingest()
        return {"status": "ok", "chunks_indexed": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/admin/refresh-entsoe-prices", dependencies=[Depends(_require_admin)])
async def admin_refresh_entsoe_prices():
    """
    Manually trigger the ENTSO-E day-ahead price refresh (normally runs
    automatically via the 03:00 UTC ARQ cron job — see
    arq_task_refresh_entsoe_prices / _arq_startup above). Admin only.
    Useful for an immediate first population of the Redis cache without
    waiting for the next scheduled run, and for verifying ENTSOE_API_TOKEN
    is actually valid after it's set.
    """
    token = os.getenv("ENTSOE_API_TOKEN", "")
    if not token:
        raise HTTPException(status_code=503, detail="ENTSOE_API_TOKEN not set")
    try:
        summary = await _refresh_entsoe_prices(token=token)
        ok = {c: v for c, v in summary["countries"].items() if v.get("lo_avg_eur_mwh") is not None}
        failed = {c: v for c, v in summary["countries"].items() if v.get("lo_avg_eur_mwh") is None}
        return {"status": "ok", "refreshed": list(ok.keys()), "failed": list(failed.keys()), "detail": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/admin/ingest-latvia", dependencies=[Depends(_require_admin)])
async def admin_ingest_latvia():
    """Download Latvian regulatory HTML/PDFs and upsert chunks into ChromaDB. Admin only."""
    try:
        from latvia_ingestion import ingest_latvia_sources
        count = ingest_latvia_sources()
        return {"status": "ok", "chunks_indexed": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/admin/ingest-lithuania", dependencies=[Depends(_require_admin)])
async def admin_ingest_lithuania():
    """Download Lithuanian regulatory HTML/PDFs and upsert chunks into ChromaDB. Admin only."""
    try:
        from lithuania_ingestion import ingest_lithuania_sources
        count = ingest_lithuania_sources()
        return {"status": "ok", "chunks_indexed": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/admin/ingest-caruna", dependencies=[Depends(_require_admin)])
async def admin_ingest_caruna():
    """Download Caruna PDFs and upsert chunks into ChromaDB. Admin only."""
    try:
        from caruna_ingestion import ingest_caruna_sources
        count = ingest_caruna_sources()
        return {"status": "ok", "chunks_indexed": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/admin/ingest-playwright", dependencies=[Depends(_require_admin)])
async def admin_ingest_playwright():
    """
    Re-run the Playwright headless ingest (JS-rendered SPA sources, incl. EU
    BAT-conclusions / EIA-directive content) into permit_docs_v2. Admin only.

    Previously CLI-only (permit_ai/ingest_playwright.py), no admin endpoint —
    its content was never re-ingested after the mpnet v2 embedding migration,
    unlike every other country's ingestion path. This endpoint closes that gap.
    """
    try:
        from ingest_playwright import ingest_playwright, SOURCES
        # ingest_playwright() calls asyncio.run() internally (Playwright needs
        # its own event loop) — cannot be called directly from this already-
        # running async endpoint. Run it in a worker thread instead.
        summary = await asyncio.to_thread(ingest_playwright, SOURCES)
        count = sum(row.get("chunks_added", 0) for row in summary)
        return {"status": "ok", "chunks_indexed": count, "summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/admin/ingest-precedent", dependencies=[Depends(_require_admin)])
async def admin_ingest_precedent():
    """
    Re-run the precedent/case-law ingest (incl. EU BAT-conclusions, EIA
    directive, EU BIM content) into permit_docs_v2. Admin only.

    Previously CLI-only (permit_ai/ingest_precedent.py), no admin endpoint —
    its content was never re-ingested after the mpnet v2 embedding migration,
    unlike every other country's ingestion path. This endpoint closes that gap.
    """
    try:
        from ingest_precedent import ingest_precedent, SOURCES
        summary = ingest_precedent(SOURCES)
        count = sum(row.get("chunks_added", 0) for row in summary)
        return {"status": "ok", "chunks_indexed": count, "summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/admin/reindex-ee-v2", dependencies=[Depends(_require_admin)])
async def admin_reindex_ee_v2():
    """
    Re-embed all EE chunks from permit_docs (v1/MiniLM) into permit_docs_v2 (mpnet).

    The original EE ingestion only upserted into permit_docs (v1). Production uses
    permit_docs_v2 (mpnet 768-dim), so EE queries return 0 chunks until this runs.
    This endpoint reads the 79 EE chunks, re-embeds with mpnet, and upserts into v2.
    Admin only. Runs in a thread so it doesn't block the event loop (~15–30 s).
    """
    def _run_reindex() -> dict:
        import gc
        import chromadb
        from sentence_transformers import SentenceTransformer

        log = logging.getLogger("reindex-ee")

        log.info("[ee-reindex] Reading EE chunks from permit_docs (v1)…")
        client  = chromadb.PersistentClient(path=_DB_PATH)
        col_v1  = client.get_collection("permit_docs")
        result  = col_v1.get(where={"country": "EE"}, include=["documents", "metadatas"])
        ids, docs, metas = result["ids"], result["documents"], result["metadatas"]
        log.info(f"[ee-reindex] Found {len(ids)} EE chunks in v1")

        if not ids:
            return {"status": "no_chunks", "chunks_reindexed": 0}

        log.info(f"[ee-reindex] Loading {_V2_MODEL}…")
        model = SentenceTransformer(_V2_MODEL)
        log.info(f"[ee-reindex] Model loaded — dim={model.get_sentence_embedding_dimension()}")

        log.info(f"[ee-reindex] Embedding {len(docs)} chunks (batch_size=32)…")
        embeddings = model.encode(docs, batch_size=32, show_progress_bar=False).tolist()
        log.info("[ee-reindex] Embeddings done — releasing model")
        del model
        gc.collect()

        col_v2 = client.get_or_create_collection(_V2_COL, metadata={"hnsw:space": "cosine"})
        batch_size = 50
        upserted = 0
        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            col_v2.upsert(
                ids=ids[sl],
                documents=docs[sl],
                embeddings=embeddings[sl],
                metadatas=metas[sl],
            )
            upserted += len(ids[sl])
            log.info(f"[ee-reindex] Upserted {upserted}/{len(ids)}")

        ee_in_v2 = len(col_v1.get(where={"country": "EE"}, include=[])["ids"])
        # verify via v2 col (re-fetch count after upsert)
        ee_v2_count = len(col_v2.get(where={"country": "EE"}, include=[])["ids"])
        v2_total    = col_v2.count()
        log.info(f"[ee-reindex] DONE — permit_docs_v2 EE={ee_v2_count}  total={v2_total}")
        return {
            "status":          "ok",
            "chunks_reindexed": upserted,
            "ee_in_v1":        len(ids),
            "ee_in_v2":        ee_v2_count,
            "v2_total":        v2_total,
        }

    try:
        result = await asyncio.to_thread(_run_reindex)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Bulk re-embed ALL countries into permit_docs_v2 ───────────────────────────
# EE is skipped (already complete). Processes one country at a time in
# 100-chunk batches with 0.5 s pauses to avoid OOM on the Render instance.

_REINDEX_ALL_COUNTRIES = ["FI", "SE", "DA", "NO", "PL", "EU", "DE", "LV"]
_BULK_REINDEX_JOB: dict = {}


def _run_bulk_reindex(job: dict) -> None:
    """Background thread: re-embed all v1 chunks into permit_docs_v2 (mpnet)."""
    import gc
    import chromadb
    from sentence_transformers import SentenceTransformer
    log = logging.getLogger("reindex-bulk")

    try:
        job["status"] = "running"
        log.info("[reindex-bulk] START — loading %s", _V2_MODEL)

        client = chromadb.PersistentClient(path=_DB_PATH)
        col_v1 = client.get_collection("permit_docs")
        col_v2 = client.get_or_create_collection(_V2_COL, metadata={"hnsw:space": "cosine"})

        model = SentenceTransformer(_V2_MODEL)
        log.info("[reindex-bulk] Model loaded — dim=%d", model.get_sentence_embedding_dimension())
        job["model_loaded"] = True

        chunk_batch = 100   # chunks per visible progress step
        countries_result: dict = {}

        for cc in _REINDEX_ALL_COUNTRIES:
            log.info("[reindex-bulk] %s — fetching v1 chunks…", cc)
            res   = col_v1.get(where={"country": cc}, include=["documents", "metadatas"])
            ids   = res["ids"]
            docs  = res["documents"]
            metas = res["metadatas"]
            total = len(ids)
            log.info("[reindex-bulk] %s — %d chunks to embed", cc, total)

            job["current_country"] = cc
            job["country_total"]   = total
            job["country_done"]    = 0

            if total == 0:
                countries_result[cc] = {"upserted": 0, "v1_count": 0, "v2_count": 0}
                job["countries_done"].append(cc)
                continue

            total_batches = (total + chunk_batch - 1) // chunk_batch
            upserted = 0
            for b, start in enumerate(range(0, total, chunk_batch)):
                sl      = slice(start, start + chunk_batch)
                b_ids   = ids[sl]
                b_docs  = docs[sl]
                b_metas = metas[sl]

                embs = model.encode(b_docs, batch_size=32, show_progress_bar=False).tolist()
                col_v2.upsert(ids=b_ids, documents=b_docs, embeddings=embs, metadatas=b_metas)
                upserted                += len(b_ids)
                job["country_done"]     = upserted
                job["total_upserted"]   = job.get("total_upserted", 0) + len(b_ids)

                log.info("[reindex-bulk] %s batch %d/%d — %d/%d done",
                         cc, b + 1, total_batches, upserted, total)
                time.sleep(0.5)

            v2_count = len(col_v2.get(where={"country": cc}, include=[])["ids"])
            countries_result[cc] = {"upserted": upserted, "v1_count": total, "v2_count": v2_count}
            job["countries_done"].append(cc)
            log.info("[reindex-bulk] %s DONE — v2_count=%d", cc, v2_count)

        log.info("[reindex-bulk] All countries done — releasing model")
        del model
        gc.collect()

        v2_total = col_v2.count()
        job["status"]           = "done"
        job["v2_total"]         = v2_total
        job["countries_result"] = countries_result
        log.info("[reindex-bulk] COMPLETE — v2_total=%d", v2_total)

    except Exception as exc:
        import traceback as _tb
        job["status"] = "error"
        job["error"]  = f"{type(exc).__name__}: {exc}"
        log.error("[reindex-bulk] ERROR: %s\n%s", exc, _tb.format_exc())


@app.post("/api/admin/reindex-all-v2", dependencies=[Depends(_require_admin)])
async def admin_reindex_all_v2():
    """
    Re-embed FI, SE, DA, NO, PL, EU, DE chunks from permit_docs (v1/MiniLM) into
    permit_docs_v2 (mpnet 768-dim). EE is skipped — already complete.

    Processes 100 chunks at a time with 0.5 s pauses to avoid OOM.
    Returns immediately with a job_id; poll via GET /api/admin/reindex-all-v2/status.
    """
    global _BULK_REINDEX_JOB
    if _BULK_REINDEX_JOB.get("status") == "running":
        return {"status": "already_running", **_BULK_REINDEX_JOB}

    job_id = str(uuid.uuid4())[:8]
    _BULK_REINDEX_JOB = {
        "job_id":           job_id,
        "status":           "starting",
        "model_loaded":     False,
        "current_country":  None,
        "country_total":    0,
        "country_done":     0,
        "countries_done":   [],
        "total_upserted":   0,
        "v2_total":         None,
        "countries_result": {},
        "error":            None,
    }
    Thread(target=_run_bulk_reindex, args=(_BULK_REINDEX_JOB,),
           daemon=True, name="bulk-reindex").start()
    return {"status": "started", "job_id": job_id, "countries": _REINDEX_ALL_COUNTRIES}


@app.get("/api/admin/reindex-all-v2/status", dependencies=[Depends(_require_admin)])
async def admin_reindex_all_v2_status():
    """Poll the in-progress bulk reindex. Returns current country, batch progress, totals."""
    if not _BULK_REINDEX_JOB:
        return {"status": "no_job"}
    return _BULK_REINDEX_JOB


@app.get("/api/admin/rag-test")
async def admin_rag_test(country: str = "FI", hanketyyppi: str = "BESS", secret: str = ""):
    """Quick RAG confidence check for a country+hanketyyppi pair — no PDF, no LLM, no rate limit."""
    if not secret or secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"status": "ping", "country": country, "hanketyyppi": hanketyyppi}


@app.get("/api/admin/last-generation-timing")
async def admin_last_generation_timing(secret: str = ""):
    """
    TEMPORARY diagnostic (2026-07-25) — investigating the recurring ~350-380s
    Claude API timeout. Returns wall-clock checkpoints (seconds since job start)
    for the single most recent generate_application_draft() call. Only meaningful
    when tested one generation at a time (not job_id-keyed). Remove once root-caused.
    """
    if not secret or secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    t = dict(_gen_app_module._LAST_TIMING)
    base = t.get("t0_start")
    if base is not None:
        for k, v in list(t.items()):
            if isinstance(v, float) and k != "t0_start":
                t[k] = round(v - base, 2)
            elif k == "t0_start":
                t[k] = 0.0
    return t


@app.get("/api/admin/retrieval-trace/{generation_id}")
async def admin_retrieval_trace(generation_id: str, secret: str = ""):
    """
    Internal debugging only — NOT user-facing. Read-only fetch of a generation's
    retrieval trace: retrieved chunk IDs + similarity scores + source_type per RAG
    call, the final RAQS outcome (5 criteria + overall + any low-confidence flags),
    estimated cost per Claude API call, any guardrail cap trips, and rollback
    checkpoints (retrieval / draft / proofread / raqs_final, each valid or
    discarded) — see retrieval_trace.py.
    """
    if not secret or secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    return _retrieval_trace.get_trace(generation_id)


@app.post("/api/admin/discard-checkpoint/{checkpoint_id}")
async def admin_discard_checkpoint(checkpoint_id: int, reason: str = "", secret: str = ""):
    """
    Internal debugging only — NOT user-facing. Explicitly discard (soft-mark
    invalid, never delete) one rollback checkpoint by id, e.g. after a human
    reviewing /api/admin/retrieval-trace spots a bad retrieval or draft. The row
    is kept forever with status=discarded for audit — see retrieval_trace.py's
    discard_checkpoint(). Does not touch any other checkpoint. Does not trigger
    any re-run — no automated resume engine exists yet (out of scope for TASO 1).
    checkpoint_id comes from GET /api/admin/retrieval-trace/{generation_id}'s
    "checkpoints" list.
    """
    if not secret or secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    ok = _retrieval_trace.discard_checkpoint(checkpoint_id, reason)
    if not ok:
        raise HTTPException(status_code=404, detail="No such valid checkpoint id")
    return {"checkpoint_id": checkpoint_id, "status": "discarded", "reason": reason}


@app.get("/api/admin/generation-cost")
async def admin_generation_cost(date: str, end_date: str = "", secret: str = ""):
    """
    Internal debugging only — NOT user-facing. Read-only total estimated Claude
    API cost for a day (`date`, YYYY-MM-DD) or an inclusive date range (`date` to
    `end_date`, both YYYY-MM-DD). TASO 1 cost & resource guardrail — visibility
    only, see retrieval_trace.py's generation_cost table.
    """
    if not secret or secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    return _retrieval_trace.get_cost_for_range(date, end_date or None)


@app.get("/api/admin/manual-source-freshness")
async def admin_manual_source_freshness(secret: str = ""):
    """
    Internal debugging only — NOT user-facing. Read-only staleness report over
    the source_type:"manual" + last_verified chunk metadata added by PR #50 —
    grouped by country and source, bucketed overdue (>365 days) / due_soon
    (270-365 days) / fresh (<=270 days). Visibility only: acting on a reminder
    means a human re-runs the normal tagging scripts, not anything this endpoint
    does — see permit_ai/manual_source_freshness.py.
    """
    if not secret or secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    return _manual_source_freshness.get_manual_source_report()


@app.get("/api/admin/source-drift")
async def admin_source_drift_status(secret: str = ""):
    """
    Internal debugging only — NOT user-facing. Read-only: the most recent
    drift-check result per source, from history only — does NOT fetch anything
    or re-scan ChromaDB for new targets (cheap, safe to call often). Use
    POST /api/admin/source-drift to actually run checks. See
    permit_ai/source_drift.py.
    """
    if not secret or secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    return _source_drift.get_latest_drift_status()


@app.post("/api/admin/source-drift")
async def admin_source_drift_run(secret: str = "", source: str = ""):
    """
    Internal debugging only — NOT user-facing. Fetches each checkable source's
    live URL, hashes the content, and compares it against the last stored hash
    for that source (three-state: changed / unchanged / check_failed — a
    blocked fetch, e.g. Finlex's JS-rendered pages or e-seimas.lrs.lt's bot
    blocking, reports check_failed, never a false changed/unchanged). Admin-
    triggered only — no cron/scheduler in this deployment calls this (see
    permit_ai/source_drift.py's module docstring for why). Pass `source` to
    check a single source by exact name instead of the full ~100-URL sweep.
    """
    if not secret or secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    sources = [source] if source else None
    return await _source_drift.check_all_sources(sources=sources)


@app.get("/api/admin/rag-check-all")
async def admin_rag_check_all(secret: str = ""):
    """
    Run RAG confidence check for all 8 countries × BESS in parallel.
    Auth via ?secret=ADMIN_SECRET query param (browser-friendly).
    Returns structured JSON: status, chunks_found, avg_relevance, pass/fail per country.
    """
    if not secret or secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden — pass ?secret=ADMIN_SECRET")

    import datetime
    _gen_app_module.activate_v2()   # idempotent; uses already-loaded module, no re-import

    TESTS = [
        ("FI", "BESS"),
        ("FI", "tuulivoima_maa"),
        ("SE", "BESS"),
        ("DA", "BESS"),
        ("NO", "BESS"),
        ("PL", "BESS"),
        ("EU", "BESS"),
        ("EU", "datakeskus"),
        ("EE", "BESS"),
        ("DE", "BESS"),
        ("LV", "BESS"),
        ("LT", "BESS"),
    ]
    MIN_SCORE_FI     = 0.65
    MIN_SCORE_NON_FI = 0.60

    # Semaphore: ChromaDB PersistentClient is not concurrency-safe across many threads;
    # limit to 2 simultaneous RAG calls to avoid lock contention.
    _sem = asyncio.Semaphore(2)

    async def _run_one(country: str, hanketyyppi: str) -> dict:
        min_score = MIN_SCORE_FI if country == "FI" else MIN_SCORE_NON_FI
        async with _sem:
            try:
                ctx, sources, warn, prec, _, avg = await asyncio.to_thread(
                    _gen_app_module._rag_context, hanketyyppi, country
                )
                ctx_chunks = ctx.split("\n\n---\n\n") if ctx else []
                n = len(ctx_chunks)
                return {
                    "country":         country,
                    "hanketyyppi":     hanketyyppi,
                    "status":          "PASS" if not warn else "PASS/WARN",
                    "chunks_found":    n,
                    "avg_relevance":   avg,
                    "min_score":       min_score,
                    "warning":         warn,
                    "sources":         len(sources),
                    "precedent_chunks": len(prec),
                    "top3_sources":    [s.get("display", "?")[:45] for s in sources[:3]],
                }
            except InsufficientSourcesError as exc:
                return {
                    "country":      country,
                    "hanketyyppi":  hanketyyppi,
                    "status":       "FAIL",
                    "chunks_found": exc.chunks_found,
                    "avg_relevance": round(exc.avg_relevance, 3),
                    "min_score":    min_score,
                    "warning":      None,
                    "sources":      0,
                    "top3_sources": [],
                }
            except Exception as exc:
                return {
                    "country":     country,
                    "hanketyyppi": hanketyyppi,
                    "status":      "ERROR",
                    "error":       f"{type(exc).__name__}: {exc}",
                }

    results = await asyncio.gather(*[_run_one(cc, ht) for cc, ht in TESTS])
    passed  = sum(1 for r in results if r["status"].startswith("PASS"))
    failed  = sum(1 for r in results if r["status"] == "FAIL")
    errors  = sum(1 for r in results if r["status"] == "ERROR")

    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "summary":   {"total": len(results), "passed": passed, "failed": failed, "errors": errors},
        "threshold": {"FI": MIN_SCORE_FI, "non_FI": MIN_SCORE_NON_FI},
        "results":   list(results),
    }


# ── LinkedIn posting agent ────────────────────────────────────────────────────

from linkedin_agent import (
    generate_post_draft as _li_generate,
    get_pending_posts   as _li_queue,
    approve_post        as _li_approve,
    reject_post         as _li_reject,
    mark_published      as _li_publish,
)


def _require_admin_header(x_admin_secret: str = Header(None, alias="x-admin-secret")):
    if not _ADMIN_SECRET or x_admin_secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


class _LinkedInGenerateRequest(BaseModel):
    post_type:     str = "thought_leadership"
    topic:         str
    extra_context: str = ""
    language:      str = "en"


@app.post("/api/linkedin/generate", dependencies=[Depends(_require_admin_header)])
async def linkedin_generate(req: _LinkedInGenerateRequest):
    """Generate a LinkedIn post draft via Claude. Admin only."""
    try:
        return _li_generate(req.post_type, req.topic, req.extra_context, req.language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/linkedin/queue", dependencies=[Depends(_require_admin_header)])
async def linkedin_queue():
    """List all pending posts awaiting approval."""
    return {"posts": _li_queue()}


class _LinkedInApproveRequest(BaseModel):
    edited_text: str | None = None


@app.post("/api/linkedin/approve/{post_id}", dependencies=[Depends(_require_admin_header)])
async def linkedin_approve(post_id: str, req: _LinkedInApproveRequest = _LinkedInApproveRequest()):
    """Approve a post, optionally with edited text."""
    return _li_approve(post_id, req.edited_text)


@app.post("/api/linkedin/reject/{post_id}", dependencies=[Depends(_require_admin_header)])
async def linkedin_reject(post_id: str):
    """Reject a pending post."""
    return _li_reject(post_id)


class _LinkedInPublishedRequest(BaseModel):
    linkedin_url: str | None = None


@app.post("/api/linkedin/published/{post_id}", dependencies=[Depends(_require_admin_header)])
async def linkedin_published(post_id: str, req: _LinkedInPublishedRequest):
    """Mark a post as published after manual LinkedIn posting."""
    return _li_publish(post_id, req.linkedin_url)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
