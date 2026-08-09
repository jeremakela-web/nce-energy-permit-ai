"""
tenant_db/base.py — PR 0: SQLAlchemy engine/session plumbing for the new
tenant/user Postgres database. Infra only — no tables defined here.

This is a SEPARATE database from everything else in this codebase. ChromaDB
(vectors), retrieval_trace.db / raqs_reviews.db / post_queue.db (SQLite, on
the persistent disk), and payments.db (SQLite, dormant Stripe/api_keys
scaffolding) are all untouched by this and stay exactly as they are — see
the design report (2026-08-09): Postgres+RLS is scoped narrowly to the new
tenant-scoped tables (users, tenants, projects, reports, rag_queries,
raqs_audit, consent/erasure records), not a migration of existing state.

DATABASE_URL is required to import models or run migrations against this DB.
It is intentionally NOT read with a hardcoded fallback (unlike most other
env vars in this codebase) — a silent fallback to sqlite or to no database at
all would be exactly the kind of mistake that defeats RLS-based tenant
isolation. Fail loudly instead.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set — the tenant/user Postgres database is "
            "required for backend.tenant_db, no fallback exists by design "
            "(see this module's docstring)."
        )
    # Render provides `postgres://`; SQLAlchemy 2.x + psycopg2 want `postgresql://`.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


class Base(DeclarativeBase):
    """Declarative base for every tenant-scoped table. PR A adds the first
    real models (Tenant, User) on top of this — none exist yet."""
    pass


_engine = None
_SessionLocal = None


def get_engine():
    """Lazy singleton — importing this module must not require DATABASE_URL
    to be set (e.g. local dev without Postgres configured yet); only actually
    using the DB does."""
    global _engine
    if _engine is None:
        _engine = create_engine(_database_url(), pool_pre_ping=True)
    return _engine


def get_session():
    """Returns a new SQLAlchemy Session bound to the tenant/user database."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal()
