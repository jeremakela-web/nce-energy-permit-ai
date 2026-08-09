"""
tenant_db/layer1.py — PR B: best-effort recording of generation activity
into the RLS-protected Layer 1 tables (projects, reports, rag_queries),
gated behind TENANT_TRACKING_ENABLED (default false).

Wired into 2 of the 4 confirmed ApplicationInput call sites in
backend/main.py (POST /api/generate-application and POST /api/approve-ifc)
plus the ARQ background task that POST /api/generate-application hands off
to when REDIS_URL is set. Basic Auth and the existing generation flow are
completely unchanged either way -- every function here is a no-op unless
BOTH the flag is on AND a tenant_id is actually present, and every call site
wraps it in try/except so a failure here can never block or fail a real
generation.

The 4th call site (POST /api/b2b/generate-report) is deliberately NOT wired
here: it authenticates via API key (backend/api_keys.py), a different,
still-dormant identity mechanism with no tenant_id/session concept at all
today. Forcing tenant tracking onto it would mean inventing an API-key-to-
tenant mapping that hasn't been designed or approved -- flagged as an
explicit scope decision, not a silent omission.

No real tenant-authenticated traffic reaches ANY of these paths yet (PR A's
magic-link session system doesn't gate the actual permit-generation routes
-- only its own new /api/auth/* and /api/admin/tenants* surface does, per
the already-approved PR 0-E sequence: "a future PR decides if/when [the new
auth] replaces Basic Auth for real app routes"). So today, with
TENANT_TRACKING_ENABLED=false (the default) OR simply no request.session
tenant_id present, every function below is inert. The wiring exists and is
tested; it has nothing to do yet.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .scoped import tenant_scoped_session
from .models import Project, Report

log = logging.getLogger("tenant_layer1")

TENANT_TRACKING_ENABLED: bool = os.getenv("TENANT_TRACKING_ENABLED", "false").lower() == "true"


def record_generation_start(
    tenant_id: Optional[str], *, hanketyyppi: str, country: str, phase: str,
) -> Optional[str]:
    """Best-effort: creates a new Project row for this generation via the
    RLS-scoped connection. Returns the new project_id, or None if tracking
    is disabled, no tenant_id is present, or anything goes wrong — never
    raises, generation must never be blocked by this.
    """
    if not TENANT_TRACKING_ENABLED or not tenant_id:
        return None
    try:
        with tenant_scoped_session(tenant_id) as session:
            project = Project(
                tenant_id=tenant_id, type=hanketyyppi, country=country or "FI",
                phase=phase or "esiselvitys", status="active",
            )
            session.add(project)
            session.flush()
            return project.project_id
    except Exception as exc:
        log.error("[layer1] record_generation_start failed (non-fatal): %s", exc)
        return None


def record_report(
    tenant_id: Optional[str], project_id: Optional[str], *,
    phase: str, pdf_url: Optional[str] = None, raqs_score: Optional[dict] = None,
) -> None:
    """Best-effort: records a completed report. Same never-raises contract
    as record_generation_start()."""
    if not TENANT_TRACKING_ENABLED or not tenant_id or not project_id:
        return
    try:
        with tenant_scoped_session(tenant_id) as session:
            session.add(Report(
                tenant_id=tenant_id, project_id=project_id, phase=phase,
                pdf_url=pdf_url, raqs_score=raqs_score,
            ))
    except Exception as exc:
        log.error("[layer1] record_report failed (non-fatal): %s", exc)
