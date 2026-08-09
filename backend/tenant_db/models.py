"""
tenant_db/models.py — PR A: the first real tables on the Postgres database
provisioned in PR 0. Design approved by user 2026-08-09 (two review rounds —
initial schema, then the admin-gated-signup refinement).

Five tables:
  - tenants           the company-level account
  - users             individual people under a tenant
  - access_requests   durable landing table for the EXISTING /api/access-
                       request flow (backend/main.py) -- that endpoint sends
                       an email and persists NOTHING today; this fills that
                       gap and is the source of the auto-drafted tenant below
  - consent_records   what was accepted, referencing WHICH document version
                       -- not just a boolean, since policy text changes
  - magic_link_tokens the passwordless-login mechanism (approved over email
                       +password -- no reset-flow/password-storage surface
                       to build). Only the token's hash is stored, same
                       pattern as backend/api_keys.py's key_hash.

Account creation is admin-gated (2026-08-09 refinement), not self-service:
/api/access-request auto-drafts a tenant + owner user, both status=
'pending_approval' -- no account is usable until an admin explicitly
approves it (mirrors this codebase's RAQS philosophy: AI/system drafts,
human approves, nothing auto-publishes). Login (magic-link) is unrelated to
that gate -- it only works once an account is already active.

RLS is intentionally NOT enabled on these tables in PR A -- deferred to
PR B per the approved 6-PR sequence, alongside the Layer 1 tables
(projects/reports/rag_queries) it was designed for. Every endpoint touching
these tables in PR A is either admin-gated (inherently cross-tenant, RLS
would not apply the same way) or session-scoped to exactly one row (a user's
own tenant_id, enforced in application code, not yet at the DB layer).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class AccessRequest(Base):
    """Durable landing table for POST /api/access-request. Fills the
    'persists nothing today' gap found while designing this schema — the
    endpoint's existing email-to-info@ncenergy.fi behavior is UNCHANGED,
    this is purely additive alongside it.

    `status` mirrors the drafted tenant's fate rather than being a second
    source of truth: new -> drafted (once auto-draft creates a tenant) ->
    approved | rejected (once an admin acts on that tenant).
    """
    __tablename__ = "access_requests"

    request_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=_new_uuid)
    company_name: Mapped[str] = mapped_column(Text, nullable=False)      # req.yritys
    contact_name: Mapped[str] = mapped_column(Text, nullable=False)      # req.yhteyshenkilo
    email: Mapped[str] = mapped_column(Text, nullable=False)             # req.sahkoposti
    phone: Mapped[Optional[str]] = mapped_column(Text)                   # req.puhelin
    description: Mapped[str] = mapped_column(Text, nullable=False)       # req.kuvaus
    status: Mapped[str] = mapped_column(Text, nullable=False, default="new")
        # new | drafted | approved | rejected
    # No drafted_tenant_id column here: that would be a redundant reverse
    # pointer to the same relationship tenants.request_id already expresses,
    # and the two FKs pointing at each other created an unresolvable table-
    # creation-order cycle (caught by Alembic's autogenerate warning before
    # this ever reached a real migration) — query tenants WHERE request_id =
    # this row's request_id instead.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Tenant(Base):
    """The company-level account. Never active until an admin approves it —
    see the module docstring."""
    __tablename__ = "tenants"

    tenant_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=_new_uuid)
    company_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending_approval")
        # pending_approval | active | rejected | suspended | erasure_requested | erased
    request_source: Mapped[str] = mapped_column(Text, nullable=False, default="access_request")
        # 'access_request' | 'admin_manual' -- not DB-constrained (plain TEXT,
        # no CHECK) precisely so 'admin_manual' needs no schema change later,
        # confirmed with user 2026-08-09 — only the manual-entry endpoint
        # itself (not built in PR A) is the remaining work for that path.
    request_id: Mapped[Optional[str]] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("access_requests.request_id")
    )
    approved_by: Mapped[Optional[str]] = mapped_column(Text)      # admin identifier (email)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    users: Mapped[list["User"]] = relationship(back_populates="tenant")


class User(Base):
    """An individual person under a tenant. Same pending_approval gate as
    the tenant — approving a tenant approves its owner user(s) too (see
    tenant_auth.py). `status='removed'` is a SOFT delete (per the point-4
    design confirmation): a removed user's past actions/generations still
    need attribution in the audit trail, so rows are never hard-deleted here.
    """
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=_new_uuid)
    tenant_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("tenants.tenant_id"), nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    contact_name: Mapped[Optional[str]] = mapped_column(Text)
    phone: Mapped[Optional[str]] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text, nullable=False, default="member")
        # 'owner' | 'member' -- simple v1, not a full RBAC system
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending_approval")
        # pending_approval | active | removed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    tenant: Mapped["Tenant"] = relationship(back_populates="users")


class ConsentRecord(Base):
    """What was accepted, referencing WHICH version of the document — not
    just a timestamp/boolean. Policy text changes over time; a consent
    record that doesn't pin the version becomes ambiguous once the document
    is later edited (confirmed as a design improvement with user 2026-08-09).
    """
    __tablename__ = "consent_records"

    consent_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=_new_uuid)
    tenant_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("tenants.tenant_id"), nullable=False)
    user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("users.user_id"), nullable=False)
    consent_type: Mapped[str] = mapped_column(Text, nullable=False)
        # 'privacy_policy' | 'ai_disclosure'
    document_ref: Mapped[str] = mapped_column(Text, nullable=False)
        # e.g. '/privacy#2026-08-09' -- a version tag or content reference,
        # never just a bare boolean
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RaqsAudit(Base):
    """The mandatory audit trail across all layers (per the source plan's
    own table list). First real use, in PR A: logging admin tenant approve/
    reject actions (approved with user 2026-08-09, over inventing a separate
    approval-log table).

    The plan's own column list is `(report_id, criterion, result, correction,
    actor, timestamp)` — tied to a specific generated report's RAQS scoring,
    which is a PR B/C concern (reports don't exist as a table until PR B).
    Generalized `report_id` to `subject_type` + `subject_id` here so this one
    table can carry BOTH a tenant-approval event now (subject_type=
    'tenant_approval', subject_id=tenant_id, criterion=NULL) and a future
    report-RAQS event (subject_type='report_raqs', subject_id=report_id,
    criterion=<RAQS criterion name>) without a schema change when PR B/C
    needs the table for its originally-intended purpose. `correction` is
    kept nullable per the plan's own shape, for whatever amendment-tracking
    a future criterion-level entry needs.
    """
    __tablename__ = "raqs_audit"

    audit_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=_new_uuid)
    subject_type: Mapped[str] = mapped_column(Text, nullable=False)
        # 'tenant_approval' (PR A) | 'report_raqs' (future, PR B/C)
    subject_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), nullable=False)
        # tenant_id for tenant_approval; report_id for report_raqs later
    criterion: Mapped[Optional[str]] = mapped_column(Text)      # NULL for tenant_approval
    result: Mapped[str] = mapped_column(Text, nullable=False)   # 'approved' | 'rejected' | ...
    correction: Mapped[Optional[str]] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(Text, nullable=False)    # admin identifier (email)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MagicLinkToken(Base):
    """Passwordless login. Only the SHA-256 hash of the token is stored
    (same pattern as backend/api_keys.py's key_hash) — the raw token exists
    only in the emailed link and the requester's browser, never at rest."""
    __tablename__ = "magic_link_tokens"

    token_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("users.user_id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
