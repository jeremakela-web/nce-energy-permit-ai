"""
SMR change-management log — PoC-tason JSON-varasto, sama muoto kuin rtb_store.py
ja phase_lock.py.

Origin: STUK meeting 2026-08-25 (Antti Tynkkynen, written confirmation),
referencing YVL A.3 SS305-306. Real, verified text (fetched directly from
stuklex.fi, not recalled from the meeting summary):

  SS305: "Procedures shall be in place to identify and assess the multiplier
  effects of minor modifications."
  SS306: "The licensee shall submit safety-significant changes to the
  management system to STUK for approval before their implementation. Minor
  changes shall be submitted to STUK for information before their
  implementation."

Two real, distinct workflows, not two labels on one workflow:
  - significant -> STUK APPROVAL required before implementation
  - minor       -> STUK NOTIFICATION required before implementation (no
                    approval wait, but still before-not-after)

IMPORTANT SCOPE CAVEAT (flag to the user before this is presented to STUK
or built into UI copy): SS306's literal text is scoped to "changes to the
management system" -- SS301-330 is the "Management system" chapter. YVL A.3
SS643-647 ("Managing organisational changes", chapter 6.7) is a more
detailed, separate section on organisational-change handling specifically.
Neither section is about *design/technical* changes to the plant in the way
"a change to this permit application" might be read. This module's
`significance` field and the SS305/306 approve-vs-notify workflow are built
as a *faithful implementation of the real regulatory mechanic STUK
described* (the pre-approval/notify-before split), applied here to changes
recorded against a project's permit application -- this is a reasonable,
STUK-confirmed-in-writing extension of that mechanic, not a claim that
SS306 itself governs permit-application document changes verbatim. Keep
that distinction in mind if this is ever cited back to STUK by section
number.

SS326a lists the real (qualitative, not bright-line) factors STUK names for
judging safety significance -- kept here as reference/help text, not as
executable classification logic. There is no formula: a human (the SMR
project's responsible engineer/approver) makes the significant/minor call
per change; this module records that judgement and enforces the two
resulting workflows, it does not make the judgement itself:

  SS326a: "The assessment of safety significance shall take into account,
  for example, the following: safety significance and complexity of the
  organisation and operation; safety significance, exactingness, complexity,
  uniqueness and novelty of the product or function and the resulting lack
  of experience; risks related to the plant or operation, including the
  probabilistic risk assessment (PRA)."

Scope: SMR / smr_bess, FI only (country=="FI") -- same reasoning as the YVL
Compliance Memo integration-point investigation: YVL is STUK's own
guideline series, meaningless for smr_se/smr_no/smr_da/smr_de/smr_ee/
smr_lv's entirely different national nuclear-safety authorities. This
module itself does not enforce that restriction (same convention as
rtb_store.py, which is hanketyyppi-agnostic) -- gate it at whichever
endpoint/caller wires this up, matching how every other hanketyyppi-scoped
rule in this codebase lives at the call site, not inside a generic store.

Avain: hanke_id (reuses rtb_store.make_hanke_id -- same project identity as
the RTB cockpit, so this composes with what's already there).
Tiedosto: permit_ai/embeddings/smr_change_log.json (persistent disk).
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

# Same persistent-disk convention as rtb_store.py / phase_lock.py.
_PERSISTENT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "permit_ai", "embeddings"
)
_FILE = (
    os.path.join(_PERSISTENT_DIR, "smr_change_log.json")
    if os.path.isdir(_PERSISTENT_DIR)
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "smr_change_log.json")
)
_lock = threading.Lock()

SIGNIFICANCE_VALUES = ("significant", "minor")

# Per-significance workflow states -- see module docstring for the SS305/306
# basis. "pending_stuk_approval"/"approved" for significant changes;
# "pending_stuk_notification"/"notified" for minor ones. Implementation may
# only be recorded once the change has reached the terminal state for its
# own significance level -- see mark_implemented().
_INITIAL_STATUS = {
    "significant": "pending_stuk_approval",
    "minor":       "pending_stuk_notification",
}
_READY_STATUS = {
    "significant": "approved",
    "minor":       "notified",
}


class ChangeLogError(Exception):
    """Raised for any invalid change-log operation (unknown hanke_id/change_id,
    wrong significance for the operation, implementation attempted before the
    required approval/notification step, etc.) -- never a generic ValueError,
    matching this codebase's convention of typed exceptions for domain errors
    (see generate_application.py's InsufficientSourcesError etc.)."""
    pass


def _load() -> dict:
    try:
        with open(_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    with open(_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blank_log(hanke_id: str, hanketyyppi: str = "") -> dict:
    return {
        "hanke_id":    hanke_id,
        "hanketyyppi": hanketyyppi,
        "created_at":  _now(),
        "changes":     [],
    }


def _find(data: dict, hanke_id: str, change_id: str) -> dict:
    log = data.get(hanke_id)
    if log is None:
        raise ChangeLogError(f"No change log for hanke_id={hanke_id!r}")
    for rec in log["changes"]:
        if rec["change_id"] == change_id:
            return rec
    raise ChangeLogError(f"No change_id={change_id!r} in hanke_id={hanke_id!r}'s log")


def add_change(
    hanke_id: str, *,
    change_description: str,
    significance: str,
    dependencies: str = "",
    hanketyyppi: str = "",
) -> dict:
    """Record a new change. `significance` must be "significant" or "minor"
    (a human judgement call by the project's responsible engineer -- see
    module docstring's SS326a factors for reference, this function does not
    classify anything itself). `version` is assigned as this hanke_id's Nth
    recorded change (1-indexed), monotonic, never reused even if an earlier
    change is later disputed -- there is no delete operation in this module,
    matching rtb_store.py's own "never delete, only update" convention.

    Returns the created record. approval_status starts at the correct
    initial state for the given significance (see _INITIAL_STATUS) --
    neither "approved" nor "notified" can be set at creation time, only via
    approve_change()/notify_change() below, so there is always an explicit,
    separately-timestamped record of when STUK actually approved or was
    notified, not just when the change was drafted.
    """
    if significance not in SIGNIFICANCE_VALUES:
        raise ChangeLogError(
            f"significance must be one of {SIGNIFICANCE_VALUES}, got {significance!r}"
        )
    if not (change_description or "").strip():
        raise ChangeLogError("change_description is required")

    with _lock:
        data = _load()
        if hanke_id not in data:
            data[hanke_id] = _blank_log(hanke_id, hanketyyppi)
        log = data[hanke_id]
        if hanketyyppi:
            log["hanketyyppi"] = hanketyyppi

        version = len(log["changes"]) + 1
        record = {
            "change_id":          uuid.uuid4().hex[:12],
            "version":            version,
            "date":               _now(),
            "change_description": change_description.strip(),
            "dependencies":       dependencies.strip(),
            "significance":       significance,
            "approval_status":    _INITIAL_STATUS[significance],
            "approver":           None,
            "approved_at":        None,
            "implemented_at":     None,
        }
        log["changes"].append(record)
        _save(data)
        return record


def approve_change(hanke_id: str, change_id: str, *, approver: str) -> dict:
    """Record STUK's pre-approval for a *significant* change (SS306, first
    sentence). Raises if the change is "minor" -- minor changes go through
    notify_change() instead, they are never "approved" in STUK's own
    two-track language."""
    if not (approver or "").strip():
        raise ChangeLogError("approver is required")
    with _lock:
        data = _load()
        rec = _find(data, hanke_id, change_id)
        if rec["significance"] != "significant":
            raise ChangeLogError(
                f"change_id={change_id!r} is 'minor' -- minor changes are notified, "
                "not approved (see notify_change())"
            )
        rec["approval_status"] = "approved"
        rec["approver"] = approver.strip()
        rec["approved_at"] = _now()
        _save(data)
        return rec


def notify_change(hanke_id: str, change_id: str, *, approver: str) -> dict:
    """Record that STUK was notified of a *minor* change (SS306, second
    sentence) -- `approver` here means whoever confirmed the notification
    was sent, not a STUK approval (minor changes are never approved, only
    notified). Raises if the change is "significant"."""
    if not (approver or "").strip():
        raise ChangeLogError("approver is required")
    with _lock:
        data = _load()
        rec = _find(data, hanke_id, change_id)
        if rec["significance"] != "minor":
            raise ChangeLogError(
                f"change_id={change_id!r} is 'significant' -- significant changes "
                "require approval, not notification (see approve_change())"
            )
        rec["approval_status"] = "notified"
        rec["approver"] = approver.strip()
        rec["approved_at"] = _now()
        _save(data)
        return rec


def mark_implemented(hanke_id: str, change_id: str) -> dict:
    """Record implementation. This is the actual enforcement of SS306's
    "before their implementation" clause for both tracks: raises unless the
    change has already reached the ready state for its own significance
    (approved for significant, notified for minor) -- implementation can
    never be recorded before that, for either track."""
    with _lock:
        data = _load()
        rec = _find(data, hanke_id, change_id)
        _ready = _READY_STATUS[rec["significance"]]
        if rec["approval_status"] != _ready:
            raise ChangeLogError(
                f"change_id={change_id!r} ({rec['significance']}) is "
                f"{rec['approval_status']!r}, not {_ready!r} -- cannot record "
                "implementation before STUK's approval/notification, per YVL A.3 SS306"
            )
        rec["implemented_at"] = _now()
        _save(data)
        return rec


def get_log(hanke_id: str) -> dict:
    """Read-only: the full change log for one hanke_id. Never raises --
    returns an empty-changes shape (found=False) if nothing recorded yet,
    matching rtb_store.rtb_summary()'s own not-found convention."""
    with _lock:
        log = _load().get(hanke_id)
    if log is None:
        return {"found": False, "hanke_id": hanke_id, "hanketyyppi": "", "changes": []}
    return {"found": True, **log}
