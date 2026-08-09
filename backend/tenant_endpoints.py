"""
tenant_endpoints.py — PR A: HTTP surface for tenant_auth.py's business logic.
Kept as its own APIRouter, mounted from backend/main.py, to keep main.py's
diff to an import + app.add_middleware(SessionMiddleware) + one
app.include_router() call — same "separate module" pattern main.py already
uses for retrieval_trace/manual_source_freshness/source_drift.

Admin endpoints reuse the exact x-admin-secret pattern from PR #47-51's
admin routes (backend/main.py's _ADMIN_SECRET) — imported, not duplicated.

Nothing here touches the EXISTING shared-Basic-Auth-gated app routes (permit
generation etc.) — those keep working completely unchanged. This is a
parallel, additive auth surface; a future PR decides if/when it replaces
Basic Auth for real app routes (explicitly out of scope for PR A/B/C/D/E's
already-approved sequence).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from tenant_db.base import get_session
import tenant_auth as ta
from tenant_db.models import Tenant, User
# Import style matches main.py's runtime root (backend/) — see the note in
# tenant_auth.py.

router = APIRouter()


def _require_admin(request: Request, secret: str) -> None:
    # Imported lazily to avoid a circular import (main.py imports this
    # router; this module must not import main.py at module load time).
    import main as _main
    if not secret or secret != _main._ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


def _tenant_out(t: Tenant) -> dict:
    return {
        "tenant_id": t.tenant_id, "company_name": t.company_name, "status": t.status,
        "request_source": t.request_source, "request_id": t.request_id,
        "approved_by": t.approved_by,
        "approved_at": t.approved_at.isoformat() if t.approved_at else None,
        "created_at": t.created_at.isoformat(), "updated_at": t.updated_at.isoformat(),
    }


def _user_out(u: User) -> dict:
    return {
        "user_id": u.user_id, "tenant_id": u.tenant_id, "email": u.email,
        "contact_name": u.contact_name, "phone": u.phone, "role": u.role,
        "status": u.status, "created_at": u.created_at.isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Admin: tenant/user management (x-admin-secret, matching PR #47-51's pattern)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/admin/tenants")
async def admin_list_tenants(request: Request, secret: str = "", status: str = ""):
    _require_admin(request, secret)
    session = get_session()
    try:
        return {"tenants": [_tenant_out(t) for t in ta.list_tenants(session, status=status or None)]}
    finally:
        session.close()


@router.post("/api/admin/tenants/{tenant_id}/approve")
async def admin_approve_tenant(tenant_id: str, request: Request, secret: str = "", actor: str = ""):
    _require_admin(request, secret)
    if not actor:
        raise HTTPException(status_code=400, detail="actor (admin email) is required for the audit trail")
    session = get_session()
    try:
        tenant = ta.approve_tenant(session, tenant_id=tenant_id, admin_actor=actor)
        if tenant is None:
            raise HTTPException(status_code=404, detail="No such tenant")
        return _tenant_out(tenant)
    finally:
        session.close()


@router.post("/api/admin/tenants/{tenant_id}/reject")
async def admin_reject_tenant(tenant_id: str, request: Request, secret: str = "", actor: str = "", reason: str = ""):
    _require_admin(request, secret)
    if not actor:
        raise HTTPException(status_code=400, detail="actor (admin email) is required for the audit trail")
    session = get_session()
    try:
        tenant = ta.reject_tenant(session, tenant_id=tenant_id, admin_actor=actor, reason=reason)
        if tenant is None:
            raise HTTPException(status_code=404, detail="No such tenant")
        return _tenant_out(tenant)
    finally:
        session.close()


@router.patch("/api/admin/tenants/{tenant_id}")
async def admin_update_tenant(
    tenant_id: str, request: Request, secret: str = "",
    company_name: str = "", status: str = "",
):
    _require_admin(request, secret)
    session = get_session()
    try:
        tenant = ta.update_tenant(
            session, tenant_id=tenant_id,
            company_name=company_name or None, status=status or None,
        )
        if tenant is None:
            raise HTTPException(status_code=404, detail="No such tenant")
        return _tenant_out(tenant)
    finally:
        session.close()


@router.post("/api/admin/tenants/{tenant_id}/users")
async def admin_add_user(
    tenant_id: str, request: Request, secret: str = "",
    email: str = "", contact_name: str = "", phone: str = "", role: str = "member",
):
    _require_admin(request, secret)
    if not email:
        raise HTTPException(status_code=400, detail="email is required")
    session = get_session()
    try:
        user = ta.add_user(
            session, tenant_id=tenant_id, email=email,
            contact_name=contact_name, phone=phone, role=role,
        )
        if user is None:
            raise HTTPException(status_code=404, detail="No such tenant")
        return _user_out(user)
    finally:
        session.close()


@router.post("/api/admin/users/{user_id}/remove")
async def admin_remove_user(user_id: str, request: Request, secret: str = ""):
    _require_admin(request, secret)
    session = get_session()
    try:
        user = ta.remove_user(session, user_id=user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="No such user")
        return _user_out(user)
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Public: magic-link auth
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/auth/request-link")
async def auth_request_link(email: str):
    """Always the same response regardless of whether the email matched an
    active account — see tenant_auth.request_magic_link()'s docstring for
    why (enumeration avoidance)."""
    if not email:
        raise HTTPException(status_code=400, detail="email is required")
    session = get_session()
    try:
        ta.request_magic_link(session, email=email)
    finally:
        session.close()
    return {"ok": True}


@router.get("/api/auth/verify")
async def auth_verify(request: Request, token: str = ""):
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    session = get_session()
    try:
        user = ta.verify_magic_link(session, raw_token=token)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid, expired, or already-used link")
        request.session["user_id"] = user.user_id
        request.session["tenant_id"] = user.tenant_id
        return {"ok": True, "user": _user_out(user)}
    finally:
        session.close()


@router.post("/api/auth/logout")
async def auth_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/api/auth/me")
async def auth_me(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not logged in")
    session = get_session()
    try:
        user = session.get(User, user_id)
        if user is None or user.status != "active":
            request.session.clear()
            raise HTTPException(status_code=401, detail="Session no longer valid")
        return _user_out(user)
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Consent
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/consent/accept")
async def consent_accept(request: Request, consent_type: str, document_ref: str):
    user_id = request.session.get("user_id")
    tenant_id = request.session.get("tenant_id")
    if not user_id or not tenant_id:
        raise HTTPException(status_code=401, detail="Not logged in")
    if consent_type not in ("privacy_policy", "ai_disclosure"):
        raise HTTPException(status_code=400, detail="consent_type must be privacy_policy or ai_disclosure")
    session = get_session()
    try:
        rec = ta.record_consent(
            session, tenant_id=tenant_id, user_id=user_id,
            consent_type=consent_type, document_ref=document_ref,
        )
        return {
            "consent_id": rec.consent_id, "consent_type": rec.consent_type,
            "document_ref": rec.document_ref, "accepted_at": rec.accepted_at.isoformat(),
        }
    finally:
        session.close()
