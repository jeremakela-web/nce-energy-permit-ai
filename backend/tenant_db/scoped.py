"""
tenant_db/scoped.py — PR B: the RLS-safe, tenant-scoped database connection.

Postgres RLS does not apply to a table's OWNING role by default. Every
table in this database is owned by nce_tenant_db_user (it ran every
migration), so RLS policies alone would be silently bypassed for every
query the app makes via that connection. `FORCE ROW LEVEL SECURITY` would
fix that for the owner — but PR A's admin endpoints (list_tenants,
approve_tenant, etc., in tenant_auth.py) legitimately need cross-tenant
visibility through that same owner connection, and forcing RLS would break
them. Left un-forced deliberately — PR A's admin code is unchanged by PR B.

The fix: a SECOND, low-privilege Postgres role (nce_app_scoped) that owns
nothing and only has SELECT/INSERT/UPDATE/DELETE grants on the RLS-
protected tables (see the PR B migration). RLS applies to it automatically
— no FORCE needed — and it cannot see across tenants even if application
code has a bug, because the DATABASE enforces the boundary, not the Python
code calling it.

Connection safety — the part most likely to hide a subtle bug, tested
explicitly and adversarially under concurrency (see PR B's self-test
report): every tenant-scoped session issues
`SET LOCAL app.current_tenant_id = '<uuid>'` as the FIRST statement of its
transaction — never plain SET. SET LOCAL is scoped to the current
transaction only and is automatically reset by Postgres at COMMIT/ROLLBACK,
which is what makes it safe to reuse pooled connections across different
tenants' requests. Plain SET would leak the tenant context to the NEXT
request that happens to reuse the same pooled connection — a real
cross-tenant data leak. Never use plain SET here.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _scoped_database_url() -> str:
    url = os.environ.get("TENANT_SCOPED_DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "TENANT_SCOPED_DATABASE_URL is not set — required for RLS-protected "
            "tenant-scoped queries (Layer 1: projects/reports/rag_queries). No "
            "fallback to DATABASE_URL by design: that connection is the table "
            "OWNER and bypasses RLS entirely, which would defeat the whole point "
            "of this module existing."
        )
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


_scoped_engine = None
_ScopedSessionLocal = None


def _get_scoped_engine():
    global _scoped_engine
    if _scoped_engine is None:
        _scoped_engine = create_engine(_scoped_database_url(), pool_pre_ping=True)
    return _scoped_engine


@contextmanager
def tenant_scoped_session(tenant_id: str):
    """Yields a SQLAlchemy Session whose Postgres connection has
    `app.current_tenant_id` set for the lifetime of THIS TRANSACTION ONLY
    (SET LOCAL) — every query made through this session sees, and can only
    write, rows belonging to `tenant_id`, enforced by RLS policies at the
    database layer, not by application code remembering to filter.

    Commits on clean exit, rolls back on exception — which also guarantees
    the SET LOCAL value never survives past this call, even on an error
    path, before the connection returns to the pool.
    """
    if not tenant_id or not _UUID_RE.match(tenant_id):
        raise ValueError(f"tenant_id must be a UUID string, got: {tenant_id!r}")

    global _ScopedSessionLocal
    if _ScopedSessionLocal is None:
        _ScopedSessionLocal = sessionmaker(bind=_get_scoped_engine(), autoflush=False, autocommit=False)

    session = _ScopedSessionLocal()
    try:
        # Parameterized SET isn't supported by Postgres (SET/SET LOCAL don't
        # accept bind parameters) — but tenant_id is validated as a strict
        # UUID above, so string-formatting it here is safe: not raw external
        # input reaching SQL, a value that has already been shape-checked.
        session.execute(text(f"SET LOCAL app.current_tenant_id = '{tenant_id}'"))
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
