"""
tenant_auth.py — PR A: admin-gated tenant/user creation, magic-link login,
and consent recording. Business logic behind backend/main.py's new endpoints
— kept out of main.py itself, same separation main.py already uses for
retrieval_trace/manual_source_freshness/source_drift.

Account creation is admin-gated, not self-service (design refinement
2026-08-09): the existing POST /api/access-request flow auto-drafts a
tenant + owner user, both status='pending_approval' — nothing is usable
until an admin explicitly approves via approve_tenant(). This mirrors the
RAQS philosophy already established in this codebase (AI/system drafts,
human approves, never auto-publishes).

Magic-link auth (approved over email+password — no reset-flow/password-
storage surface to build): request_magic_link() always returns the same
{"ok": true} shape to the caller regardless of whether the email matched an
active account, to avoid user enumeration. Emails go via the existing Resend
HTTP API (backend/main.py's RESEND_API_KEY, already used for usage alerts) —
not the Gmail SMTP path (SMTP_USER/PASSWORD), which stays reserved for the
internal access-request notification to info@ncenergy.fi as before.

RLS: not enabled on these tables in PR A — see backend/tenant_db/models.py's
module docstring for why (deferred to PR B alongside the tables RLS was
originally designed for). Every function here is either called from an
admin-gated endpoint (inherently cross-tenant) or scoped to exactly one
user/tenant by the caller (application-level scoping, not yet DB-enforced).
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from tenant_db.models import (
    AccessRequest, ConsentRecord, MagicLinkToken, RaqsAudit, Tenant, User,
)
# NOTE on import style: this module is imported both (a) at app runtime,
# where uvicorn runs as `cd backend && uvicorn main:app` — backend/ IS the
# import root there, matching the existing `import rtb_store as _rtb` style
# in main.py — and (b) from alembic/env.py, run from the repo root, which
# needs the `backend.`-prefixed form instead (see that file). Two different
# execution roots for the same package is pre-existing in this codebase, not
# introduced here; each context's imports are written to match its own root.

log = logging.getLogger("tenant_auth")

_MAGIC_LINK_TTL = timedelta(minutes=15)
_RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
_APP_BASE_URL = os.getenv("APP_BASE_URL", "https://ai.ncenergy.fi")


# ─────────────────────────────────────────────────────────────────────────────
# Access request -> auto-drafted tenant + owner user
# ─────────────────────────────────────────────────────────────────────────────

def create_access_request_and_draft(
    session: Session, *, company_name: str, contact_name: str,
    email: str, phone: str, description: str,
) -> tuple[AccessRequest, Tenant, User]:
    """The (b) option from the design review: auto-draft a tenant + owner
    user from the request data, present for one-click admin approve/reject.
    Both start status='pending_approval' — approve_tenant() is the only way
    either becomes usable.
    """
    req = AccessRequest(
        company_name=company_name, contact_name=contact_name,
        email=email, phone=phone, description=description,
    )
    session.add(req)
    session.flush()  # populate req.request_id before referencing it

    tenant = Tenant(
        company_name=company_name, status="pending_approval",
        request_source="access_request", request_id=req.request_id,
    )
    session.add(tenant)
    session.flush()

    owner = User(
        tenant_id=tenant.tenant_id, email=email, contact_name=contact_name,
        phone=phone, role="owner", status="pending_approval",
    )
    session.add(owner)

    req.status = "drafted"

    session.commit()
    session.refresh(req)
    session.refresh(tenant)
    session.refresh(owner)
    return req, tenant, owner


# ─────────────────────────────────────────────────────────────────────────────
# Admin: list / approve / reject / edit tenants, add / remove users
# ─────────────────────────────────────────────────────────────────────────────

def list_tenants(session: Session, *, status: Optional[str] = None) -> list[Tenant]:
    stmt = select(Tenant).order_by(Tenant.created_at.desc())
    if status:
        stmt = stmt.where(Tenant.status == status)
    return list(session.scalars(stmt))


def get_tenant(session: Session, tenant_id: str) -> Optional[Tenant]:
    return session.get(Tenant, tenant_id)


def approve_tenant(session: Session, *, tenant_id: str, admin_actor: str) -> Optional[Tenant]:
    """Activates the tenant AND every pending_approval user under it (in
    practice just the owner, drafted alongside it) — a tenant with no usable
    login would be a pointless approval. Logs to raqs_audit."""
    tenant = session.get(Tenant, tenant_id)
    if tenant is None:
        return None
    tenant.status = "active"
    tenant.approved_by = admin_actor
    tenant.approved_at = datetime.now(timezone.utc)

    for user in session.scalars(select(User).where(User.tenant_id == tenant_id)):
        if user.status == "pending_approval":
            user.status = "active"

    session.add(RaqsAudit(
        subject_type="tenant_approval", subject_id=tenant_id,
        result="approved", actor=admin_actor,
    ))
    session.commit()
    session.refresh(tenant)
    return tenant


def reject_tenant(session: Session, *, tenant_id: str, admin_actor: str, reason: str = "") -> Optional[Tenant]:
    tenant = session.get(Tenant, tenant_id)
    if tenant is None:
        return None
    tenant.status = "rejected"
    session.add(RaqsAudit(
        subject_type="tenant_approval", subject_id=tenant_id,
        result="rejected", actor=admin_actor, correction=reason or None,
    ))
    # Mirror the rejection back onto the originating access_request, if any,
    # so it isn't left dangling at status='drafted' forever.
    if tenant.request_id:
        req = session.get(AccessRequest, tenant.request_id)
        if req:
            req.status = "rejected"
    session.commit()
    session.refresh(tenant)
    return tenant


def update_tenant(
    session: Session, *, tenant_id: str,
    company_name: Optional[str] = None, status: Optional[str] = None,
) -> Optional[Tenant]:
    """Editable in place, per the point-4 design confirmation — never
    requires delete+recreate."""
    tenant = session.get(Tenant, tenant_id)
    if tenant is None:
        return None
    if company_name is not None:
        tenant.company_name = company_name
    if status is not None:
        tenant.status = status
    session.commit()
    session.refresh(tenant)
    return tenant


def add_user(
    session: Session, *, tenant_id: str, email: str,
    contact_name: str = "", phone: str = "", role: str = "member",
) -> Optional[User]:
    tenant = session.get(Tenant, tenant_id)
    if tenant is None:
        return None
    # New users under an already-active tenant go straight to active — the
    # admin gate is on the TENANT's first creation, not on every subsequent
    # teammate an already-approved company adds.
    status = "active" if tenant.status == "active" else "pending_approval"
    user = User(
        tenant_id=tenant_id, email=email, contact_name=contact_name or None,
        phone=phone or None, role=role, status=status,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def remove_user(session: Session, *, user_id: str) -> Optional[User]:
    """SOFT delete only (status='removed') — never a hard DELETE. A removed
    user's past actions still need attribution in the audit trail, per the
    point-4 design confirmation."""
    user = session.get(User, user_id)
    if user is None:
        return None
    user.status = "removed"
    session.commit()
    session.refresh(user)
    return user


# ─────────────────────────────────────────────────────────────────────────────
# Magic-link login
# ─────────────────────────────────────────────────────────────────────────────

def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def request_magic_link(session: Session, *, email: str) -> bool:
    """Always call this and always treat its return the same way from the
    endpoint (generic {"ok": true}) regardless of the boolean — the return
    value is for logging/testing only, never surfaced to the caller, so a
    probe can't distinguish 'no such account' from 'link sent'.
    """
    user = session.scalar(select(User).where(User.email == email, User.status == "active"))
    if user is None:
        log.info("[magic-link] no active user for %s — silently no-op", email)
        return False
    tenant = session.get(Tenant, user.tenant_id)
    if tenant is None or tenant.status != "active":
        log.info("[magic-link] user %s's tenant not active — silently no-op", email)
        return False

    raw_token = secrets.token_urlsafe(32)
    session.add(MagicLinkToken(
        token_hash=_hash_token(raw_token), user_id=user.user_id,
        expires_at=datetime.now(timezone.utc) + _MAGIC_LINK_TTL,
    ))
    session.commit()

    _send_magic_link_email(to_email=user.email, raw_token=raw_token)
    return True


def _send_magic_link_email(*, to_email: str, raw_token: str) -> None:
    link = f"{_APP_BASE_URL}/api/auth/verify?token={raw_token}"
    if not _RESEND_API_KEY:
        log.warning("[magic-link] RESEND_API_KEY not set — email not sent. Link: %s", link)
        return
    import requests
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {_RESEND_API_KEY}", "Content-Type": "application/json"},
            json={
                "from": "NCE Permit AI <noreply@ncenergy.fi>",
                "to": [to_email],
                "subject": "Kirjautumislinkki — NCE Permit AI",
                "text": (
                    f"Kirjaudu sisään klikkaamalla tästä (linkki voimassa 15 minuuttia):\n\n{link}\n\n"
                    f"Jos et pyytänyt tätä, voit jättää viestin huomiotta."
                ),
            },
            timeout=10,
        )
    except Exception as exc:
        log.error("[magic-link] Resend send failed for %s: %s", to_email, exc)


def verify_magic_link(session: Session, *, raw_token: str) -> Optional[User]:
    """Validates + consumes a token (single use). Returns the User on
    success, None on any failure (missing, expired, already used) — the
    endpoint should treat all failure modes identically (generic error),
    same enumeration-avoidance reasoning as request_magic_link()."""
    token_hash = _hash_token(raw_token)
    tok = session.get(MagicLinkToken, token_hash)
    if tok is None:
        return None
    now = datetime.now(timezone.utc)
    if tok.used_at is not None or tok.expires_at < now:
        return None
    tok.used_at = now
    session.commit()
    return session.get(User, tok.user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Consent
# ─────────────────────────────────────────────────────────────────────────────

def record_consent(
    session: Session, *, tenant_id: str, user_id: str,
    consent_type: str, document_ref: str,
) -> ConsentRecord:
    rec = ConsentRecord(
        tenant_id=tenant_id, user_id=user_id,
        consent_type=consent_type, document_ref=document_ref,
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def has_consented(session: Session, *, user_id: str, consent_type: str) -> bool:
    return session.scalar(
        select(ConsentRecord.consent_id).where(
            ConsentRecord.user_id == user_id,
            ConsentRecord.consent_type == consent_type,
        )
    ) is not None
