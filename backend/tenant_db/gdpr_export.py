"""
tenant_db/gdpr_export.py — PR D: admin-triggered GDPR data-export ZIP for a
single tenant. Design approved 2026-08-09 (see the PR D report for full
per-table reasoning). Read-only — this module never deletes or modifies
anything (erasure is a separate, not-yet-approved PR in the 0-E sequence).

Export scope (all rows scoped to ONE tenant_id):
  - tenants                                   } RLS-protected tables —
  - users                                     } queried via
  - projects                                  } tenant_scoped_session() as
  - reports                                   } defense-in-depth, even
  - rag_queries                               } though the app code below
  - user_events                               } also filters by tenant_id
                                                 explicitly. See scoped.py.

  - access_requests (0-1 row, via tenants.request_id)     } NOT RLS-
  - raqs_audit (subject_type='tenant_approval',           } protected —
                subject_id=tenant_id)                     } filtered by
  - consent_records (tenant_id=tenant_id)                 } explicit WHERE
                                                             only, no DB-
                                                             level backstop.
                                                             Tested
                                                             adversarially
                                                             (see self-test
                                                             report).

  BUG FOUND DURING SELF-TEST, not hidden: consent_records was originally
  assumed to be RLS-protected (it has a tenant_id column and was created
  alongside tenants/users in PR A) and was queried via the scoped role.
  A real query against production failed with "permission denied for
  table consent_records" — checking PR B's migration (the one that
  actually enables RLS) shows consent_records was never added to
  _RLS_TABLES = ["tenants", "users", "projects", "reports", "rag_queries"],
  and PR C's migration only ever extended grants to user_events, not
  consent_records either. So consent_records has NEVER been RLS-protected
  or granted to nce_app_scoped since it was created — a pre-existing gap
  in PR B/C, only surfaced now because this is the first code to query it
  through the scoped role. Fixed here by moving it to the owner-session,
  explicit-WHERE-only group (same as access_requests/raqs_audit). Whether
  consent_records SHOULD gain real RLS protection is a separate decision
  for a future PR — not implemented here, out of scope for "GDPR export".

  - magic_link_tokens: EXCLUDED — SHA-256 hashes + expiry timestamps only,
    no informational value to the data subject (confirmed with user
    2026-08-09).

Format: one JSON file per table + manifest.json, inside an in-memory ZIP
(io.BytesIO + stdlib zipfile) — nothing ever touches disk, so there is no
temp-file cleanup step. JSON chosen over CSV because several columns are
already JSONB (Report.raqs_score, RagQuery.sources_used, UserEvent.detail)
— CSV would force those into escaped strings inside cells; JSON preserves
their native structure, matching this codebase's existing structured-not-
flattened convention for those columns.

Each successful export is logged to raqs_audit (subject_type='gdpr_export',
subject_id=tenant_id, actor=<admin email>, result='exported') — that table
already exists specifically as an extensible subject_type/subject_id audit
log (see its own docstring). NOT logged to user_events — PR C explicitly
scoped that table to exactly 'login'/'ifc_upload', and an admin-triggered
export is not a tenant user's own lifecycle event.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, date
from typing import Any, Optional

from sqlalchemy import inspect, select

from .base import get_session
from .scoped import tenant_scoped_session
from .models import (
    Tenant, User, ConsentRecord, Project, Report, RagQuery, UserEvent,
    AccessRequest, RaqsAudit,
)

# Tables queried through the RLS-protected scoped role — defense in depth,
# even though every query below also filters by tenant_id explicitly in
# application code. See tenant_db/scoped.py. NOTE: consent_records is
# deliberately NOT here despite having a tenant_id column — see this
# module's docstring for the "BUG FOUND DURING SELF-TEST" note; it was
# never actually granted to nce_app_scoped/RLS-enabled by PR B or PR C.
_RLS_MODELS = [
    (Tenant, "tenant_id"),
    (User, "tenant_id"),
    (Project, "tenant_id"),
    (Report, "tenant_id"),
    (RagQuery, "tenant_id"),
    (UserEvent, "tenant_id"),
]


def _row_to_dict(obj: Any) -> dict:
    """Generic column-to-dict via SQLAlchemy's own mapper — stays correct
    automatically if a model gains/loses columns later, rather than a
    hand-maintained per-model field list."""
    out = {}
    for col in inspect(obj).mapper.column_attrs:
        val = getattr(obj, col.key)
        if isinstance(val, (datetime, date)):
            val = val.isoformat()
        out[col.key] = val
    return out


def build_export_zip(tenant_id: str, *, actor: str) -> Optional[bytes]:
    """Returns the ZIP file's raw bytes, or None if no such tenant exists.
    Logs a raqs_audit 'gdpr_export' entry as a side effect on success only
    (a lookup miss is not an export event)."""
    owner_session = get_session()
    try:
        tenant = owner_session.get(Tenant, tenant_id)
        if tenant is None:
            return None

        tables: dict[str, list[dict]] = {}

        # RLS-protected tables, via the scoped low-privilege role.
        with tenant_scoped_session(tenant_id) as scoped:
            for model, tenant_col in _RLS_MODELS:
                rows = scoped.scalars(
                    select(model).where(getattr(model, tenant_col) == tenant_id)
                ).all()
                tables[model.__tablename__] = [_row_to_dict(r) for r in rows]

        # Non-RLS tables — explicit WHERE only, via the owner session.
        access_requests = []
        if tenant.request_id:
            ar = owner_session.get(AccessRequest, tenant.request_id)
            if ar is not None:
                access_requests = [_row_to_dict(ar)]
        tables["access_requests"] = access_requests

        raqs_rows = owner_session.scalars(
            select(RaqsAudit).where(
                RaqsAudit.subject_type == "tenant_approval",
                RaqsAudit.subject_id == tenant_id,
            )
        ).all()
        tables["raqs_audit"] = [_row_to_dict(r) for r in raqs_rows]

        # consent_records: NOT actually RLS-protected (see docstring's "BUG
        # FOUND DURING SELF-TEST" note) — explicit WHERE only, same as the
        # two tables above.
        consent_rows = owner_session.scalars(
            select(ConsentRecord).where(ConsentRecord.tenant_id == tenant_id)
        ).all()
        tables["consent_records"] = [_row_to_dict(r) for r in consent_rows]

        manifest = {
            "tenant_id": tenant_id,
            "company_name": tenant.company_name,
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "exported_by": actor,
            "tables": {name: len(rows) for name, rows in tables.items()},
        }

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            for name, rows in tables.items():
                zf.writestr(f"{name}.json", json.dumps(rows, indent=2, ensure_ascii=False))

        owner_session.add(RaqsAudit(
            subject_type="gdpr_export", subject_id=tenant_id,
            result="exported", actor=actor,
        ))
        owner_session.commit()

        return buf.getvalue()
    finally:
        owner_session.close()
