# RAQS-adjacent change-management section — investigation & proposal

STUK priority 3. Investigation and proposal only, per instruction — **nothing in this
document has been built.**

## Headline finding: this is mostly already built, and unwired

Before designing anything new, I searched the repo for prior art and found
**`backend/smr_change_log.py`**, committed 2026-08-25 — the same day as the STUK
meeting (Antti Tynkkynen, written confirmation) that the module's own docstring cites
as its origin. It is a complete, STUK-sourced schema + business-logic module:

- Real quoted text from YVL A.3, fetched directly from stuklex.fi (not recalled from
  the meeting summary): SS305 (assess multiplier effects of minor modifications) and
  SS306 (significant changes → STUK approval *before* implementation; minor changes →
  STUK notification *before* implementation).
- Two distinct workflows, not two labels on one workflow: `significant` →
  `pending_stuk_approval` → `approved`; `minor` → `pending_stuk_notification` →
  `notified`. `mark_implemented()` enforces SS306's "before implementation" clause for
  real — it raises if called before the change has reached its track's ready state.
- `version` is assigned per hanke_id, monotonic, 1-indexed, never reused — there is no
  delete, only append, matching `rtb_store.py`'s own convention.
- Storage: a JSON file on persistent disk (`permit_ai/embeddings/smr_change_log.json`),
  same pattern as `rtb_store.py` / `phase_lock.py` — not a SQL table.
- Scope, stated explicitly in the docstring: SMR / smr_bess, **FI only** — same
  reasoning as the YVL Compliance Memo (YVL is STUK's own series, meaningless for
  smr_se/smr_no/smr_da/smr_de/smr_ee/smr_lv's different national authorities). Keyed
  on `hanke_id`, reusing `rtb_store.make_hanke_id()` — designed from day one to compose
  with the RTB cockpit's existing project identity.
- A real, self-flagged scope caveat worth preserving verbatim if this ever reaches
  STUK or UI copy: SS306's literal text is scoped to "changes to the management
  system" (SS301-330 is the Management System chapter); it does not literally govern
  *design/technical* changes to a permit application. The module is built as a
  faithful implementation of the real approve-vs-notify mechanic STUK described,
  applied to permit-application changes as a reasonable, STUK-confirmed-in-writing
  extension — not a claim that SS306 itself governs document changes verbatim.

I then grepped the entire repo for any reference to `smr_change_log` outside the file
itself: **zero matches.** No import in `backend/main.py`, no API endpoint, no frontend
usage, no PDF integration. It's finished, correct-looking, regulator-grounded code
sitting completely dormant for 5 days.

This changes the shape of the ask. The task isn't "design a `report_versions` table
from scratch" — the data model and workflow logic already exist and match the agreed
shape (separate mechanism, not a 6th RAQS criterion). **The actual gap is wiring: API
endpoints, a PDF report section, and a cockpit UI panel.**

## Why NOT the tenant_db Postgres route

`backend/tenant_db/models.py`'s `Project`/`Report`/`RaqsAudit` tables look tempting —
`RaqsAudit` even has a `subject_type='report_raqs'` value explicitly reserved for
"future" per-criterion RAQS audit entries. But this whole subsystem is currently
**fully inert**: gated behind `TENANT_TRACKING_ENABLED` (default `false`), and even
when enabled, `record_generation_start()`/`record_report()` only fire when a real
`tenant_id` is present on the request — and per `layer1.py`'s own docstring, no
tenant-authenticated traffic reaches any permit-generation route today (Basic Auth
still gates the real app routes; the new tenant auth only covers its own
`/api/auth/*` and `/api/admin/tenants*` surface). Building change-management on top of
that would mean building it on top of dead infrastructure — it'd be dormant too, for
the same reason.

`smr_change_log.py`, by contrast, uses the same lightweight JSON-on-disk pattern as
`rtb_store.py`, which **is** live in production today — real generation call sites at
`backend/main.py:1136/1287/1515` already derive an `_rtb_id` via
`_rtb.make_hanke_id(...)` on every generation. Recommendation: build on
`smr_change_log.py`, not on `tenant_db`. No new SQL table, no migration, no RLS
policy — just wiring existing, working code.

## Proposed wiring (not built — for review)

**1. API endpoints in `backend/main.py`**, mirroring the existing `rtb_store` pattern
(`import smr_change_log as _change_log`, `@app.get("/api/rtb/{hanke_id}")` as the
template):

- `GET  /api/smr-change-log/{hanke_id}` → `_change_log.get_log(hanke_id)` (read-only,
  never raises, matches `rtb_summary()`'s not-found convention).
- `POST /api/smr-change-log/{hanke_id}/changes` → `add_change()`. Body:
  `change_description`, `significance` (`significant`/`minor`), `dependencies`.
- `POST /api/smr-change-log/{hanke_id}/changes/{change_id}/approve` → `approve_change()`
  (significant only). `approver` from the authenticated caller, not free text, so this
  can't be spoofed client-side.
- `POST /api/smr-change-log/{hanke_id}/changes/{change_id}/notify` → `notify_change()`
  (minor only), same `approver` handling.
- `POST /api/smr-change-log/{hanke_id}/changes/{change_id}/implement` →
  `mark_implemented()`.
- All under existing Basic Auth, same as the rest of `main.py`'s admin-ish surface.
  `ChangeLogError` → HTTP 400 with the module's own message (it's already
  human-readable, e.g. "cannot record implementation before STUK's
  approval/notification, per YVL A.3 SS306").
- `hanke_id` derivation should reuse whatever `_rtb.make_hanke_id(...)` call already
  produced for that project at generation time, not a separately-invented id — the
  module was explicitly designed for this composition.

**2. PDF report section** — new `_change_log_page(log, st, lang)` function, same shape
as `_yvl_memo_page()`/`_raqs_page()`, wired at the exact same call site
(`generate_application.py:11525-11544`), right alongside the YVL memo:

```python
_yvl_memo = _yvl_compliance_memo(inp)
if _yvl_memo:
    for _elem in _yvl_memo_page(_yvl_memo, st, lang):
        story.append(_elem)

# proposed:
if inp.hanketyyppi in ("SMR", "smr_bess") and country == "FI":
    _change_log = _change_log_mod.get_log(_hanke_id)
    if _change_log["found"] and _change_log["changes"]:
        for _elem in _change_log_page(_change_log, st, lang):
            story.append(_elem)
```

Gated identically to the YVL memo (`SMR`/`smr_bess`, `country == "FI"`) — matches
`smr_change_log.py`'s own documented scope, and reuses a precedent already reviewed
and approved once for the YVL memo, rather than inventing a new gating convention.
Content: a table of recorded changes (version, date, description, significance,
approval status, approver, implemented date) — this is a direct rendering of the
existing JSON shape, no new data needed. The SS306 scope caveat from the module
docstring should appear as a footnote on this page, not just live in code comments,
so the same "don't over-claim regulatory coverage" discipline applied to the YVL memo
carries through to what STUK actually reads.

**3. Cockpit UI** — `backend/static/rtb.html` already exists as the RTB cockpit page
and is the natural home for a change-log panel (add change, mark
significant/minor, approve/notify, mark implemented) rather than a new page. Not
scoped in detail here since UI work wasn't asked for yet — flagging that it exists as
the logical integration point once/if this is approved to build.

**4. Relationship to RAQS**: none, by design — confirmed this matches your instruction.
This is a separate report section entirely; no RAQS criterion, prompt, or scoring
touches it. The only shared surface is the PDF page-assembly call site, where it sits
next to (not inside) the RAQS page.

## The "insufficient material / too early to file" warning — recommendation

Left as "your call" — my recommendation is to treat it as a **separate, later
decision**, not bundled into this wiring work:

- The closest existing precedent is `InsufficientSourcesError`
  (`generate_application.py:64`, raised at line 3570) — but that's a RAG-retrieval
  sufficiency check (chunk count / avg similarity score) for the *current* generation,
  not a "is this change-managed project mature enough to file" signal. Different
  question, same non-blocking-warning spirit as `citation_gap_flags`.
- If wanted later, the naive version this data model could support cheaply: a
  generation-time check against `smr_change_log.get_log(hanke_id)` for any
  `significant` change still stuck in `pending_stuk_approval` — surfaced as a
  non-blocking warning banner (never a hard block, matching every other
  supplementary check in this codebase: `citation_gap_flags`, the YVL memo). But this
  needs its own scope discussion — what "too early to file" actually means is a
  business judgement, not something inferable from the change log alone — so I'd
  rather not fold it into this proposal's build scope until that's discussed
  separately.

## Summary / ask

- No new SQL table. Don't build on `tenant_db` — it's dormant.
- Build on the already-existing, already-STUK-grounded `smr_change_log.py`: add API
  endpoints (mirroring `rtb_store`'s pattern), a PDF page (mirroring
  `_yvl_memo_page`), and flag the RTB cockpit (`rtb.html`) as where UI would eventually
  live.
- Scope stays SMR/smr_bess, FI-only, matching the module's own documented reasoning
  and the YVL memo precedent.
- "Insufficient material to file" kept as a separate, later, non-blocking-warning
  design question — not in this build's scope.

Waiting for review before building any of this.
