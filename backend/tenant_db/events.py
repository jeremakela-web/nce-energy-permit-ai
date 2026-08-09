"""
tenant_db/events.py — PR C: best-effort recording of user lifecycle events
into the RLS-protected user_events table, gated behind
TENANT_TRACKING_ENABLED (same flag PR B's layer1.py uses — one on/off
switch for "is any tenant-layer database writing active", not a second
flag, per the minimal-surface reasoning already established in PR B).

Scope approved 2026-08-09: ONLY 'login' (POST /api/auth/verify, on
successful magic-link verification) and 'ifc_upload' (POST /api/parse-ifc).
Explicitly NOT consent acceptance -- consent_records (PR A) already covers
that with proper document-version tracking; a mirrored entry here would be
redundant, not complementary.

Same never-raises contract as tenant_db/layer1.py: a DB/config failure here
must never block or fail the real request it's attached to.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .scoped import tenant_scoped_session
from .models import UserEvent

log = logging.getLogger("tenant_events")

# Same flag PR B's layer1.py reads — one on/off switch for "is any
# tenant-layer database writing active", not a second flag.
TENANT_TRACKING_ENABLED: bool = os.getenv("TENANT_TRACKING_ENABLED", "false").lower() == "true"

_VALID_EVENT_TYPES = {"login", "ifc_upload"}


def record_user_event(
    tenant_id: Optional[str], user_id: Optional[str], *,
    event_type: str, detail: Optional[dict] = None,
) -> None:
    """Best-effort: records one lifecycle event via the RLS-scoped
    connection. No-op (never raises) if tracking is disabled, tenant_id/
    user_id is missing, or event_type isn't one of the approved types --
    an unexpected event_type is a caller bug, logged loudly rather than
    silently written as junk data.
    """
    if not TENANT_TRACKING_ENABLED:
        return
    if not tenant_id or not user_id:
        return
    if event_type not in _VALID_EVENT_TYPES:
        log.error(
            "[events] record_user_event called with unapproved event_type=%r "
            "(valid: %s) — not recorded, this is a caller bug",
            event_type, sorted(_VALID_EVENT_TYPES),
        )
        return
    try:
        with tenant_scoped_session(tenant_id) as session:
            session.add(UserEvent(
                tenant_id=tenant_id, user_id=user_id,
                event_type=event_type, detail=detail,
            ))
    except Exception as exc:
        log.error("[events] record_user_event failed (non-fatal): %s", exc)
