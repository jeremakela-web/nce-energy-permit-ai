# Coverage gaps backlog

Consolidated output of the first real run of `scripts/validate_country_coverage.py`
and `scripts/validate_frontend_coverage.py` (2026-08-26, right after both were
built — see `BUG_CONSOLIDATION_ARCHITECTURE_PROPOSAL.md` and
`NEW_COUNTRY_ONBOARDING_CHECKLIST.md` for the tooling and process these came
from), plus one related finding turned up while investigating the frontend
side. Tracked here in one place instead of scattered across session
messages. Nothing in this file has been fixed — it's the "becomes backlog,
not a blocker" list, ready to work whenever it's prioritized.

To re-verify any item, re-run the validators. As of this writing the local
dev Python (3.14, macOS/arm64) can't build `chromadb`'s dependency chain
from source, so the backend list below was produced via direct AST-literal
extraction against `generate_application.py` (same checking logic, real
data, just not through the actual `import` path) rather than
`python scripts/validate_country_coverage.py` directly — re-run the real
script once that's resolved (see the Python-version note in
`BUG_CONSOLIDATION_ARCHITECTURE_PROPOSAL.md`/this repo's deploy setup) to
confirm nothing changed in the meantime. The frontend list ran for real —
`validate_frontend_coverage.py` has no heavy dependencies.

---

## 1. Backend — `_LAW_TRANS` / `_LAW_CITATION_REPLACEMENT` (21 statute-rows)

The long tail beyond the ~13-14 rows each country's law-citation research
pass (SE, PL, LT, DA, NO, LV, EE, DE) has covered so far. These are
genuinely un-researched, not oversights within an already-worked row —
same category of work as this sprint's per-country passes, just organized
by statute instead of by country.

| Statute row | Missing |
|---|---|
| `Sähkömarkkinalaki 588/2013` | DE, EE, LT, LV |
| `Maa-aineslaki 555/1981` | DE, EE, LT, LV |
| `YVA-laki 252/2017 (kynnykset ylittyessä)` | DE, EE, LV |
| `YVA-laki 252/2017 (≥50 ha hankkeet)` | DE, EE, LV |
| `MRL 132/1999 § 137` | DE, EE, LV |
| `MRL 197 §` | DE, EE, LT, LV |
| `MRL 132/1999 § 91a` | DE, EE, LV |
| `MRL 132/1999 § 9` | DE, EE, LV |
| `Ilmailulaki 864/2014` | DE, EE, LT, LV |
| `Maakaari 540/1995` | DE, EE, LT, LV |
| `Merilaki 674/1994` | DE, EE, LT, LV |
| `Merenkulkulaki 1672/2009` | DE, EE, LT, LV |
| `Laki alueiden käytöstä` | DE, EE, LT, LV |
| `Rakentamislaki 751/2023 / MRL 132/1999 § 125–126` | DE, EE, LV |
| `Rakentamislaki 751/2023 / MRL 132/1999 § 126` | DE, EE, LV |
| `Kalastuslaki 379/2015` | DE, EE, LT, LV |
| `Säteilylaki 859/2018` | DE, EE, LT, LV |
| `Kemikaaliturvallisuuslaki 390/2005` | DE, EE, LT, LV |
| `Kemikaaliturvallisuuslaki 390/2005 (BESS)` | DE, EE, LT, LV |
| `Luonnonsuojelulaki 9/2023` | DE, EE, LT, LV |
| `Maantielaki 503/2005 (tiealueet)` | DE, EE, LT, LV |

Note the consistent pattern: DE, EE, LV are missing from nearly every row,
LT from most. SE/PL/DA/NO are complete for all 21 (their law-citation
passes went deeper into this list than EE/LV/DE/LT's did so far).

## 2. Backend — whole-dict gaps

| Dict | Missing | Notes |
|---|---|---|
| `_PDF_STRINGS` | ET, LV | Already independently flagged in `LV_LAW_CITATION_RESEARCH.md` §5 as a real, separate, larger gap — LV/ET reports fall back to the EN card text for every static UI string. Not fixed there deliberately (different-shaped task from law-citation research); still open. |
| `_CRITICAL_EXTRA` | DE, ET, LV | The "⚠️ expert-review-required" guardrail paragraph is silently absent for these three on any `_CRITICAL_HANKE_TYPES` hanketyyppi (SMR, smr_bess, ymparistolupa). LT and EE's country code itself were fixed since the original 2026-08-23 LT investigation found five languages missing here; DE/ET/LV remain. |
| `_NATIONAL_SUPERVISORS` | LT, LV | Feeds a prompt line naming the country's supervisory authority. Narrowed since the 2026-08-23 investigation (which also had ET) — ET is now fixed, LT/LV remain. |
| `_BESS_MARKET_DATA` | DE, LT | Already reported this session: falls back to Finland's own Clean Horizon market-index figure (110 €k/MW/year), cited unconditionally as a real source in the PDF's "Lähteet" section for every BESS report in these two countries. |

## 3. Frontend — `LUPA_I18N` (13 rows missing DE/ET/LT/LV entirely)

Found via `scripts/validate_frontend_coverage.py`'s first real run against
`backend/static/index.html`. `TRANSLATIONS` itself is fully clean (the
earlier ET-completion work holds up). These 13 permit-name rows render as
raw Finnish text for DE/ET/LT/LV users — same shape as the SE/DA/NO/PL
`LUPA_I18N` leak fixed urgently pre-Ecogain-demo (PR #95), just a different,
never-completed set of rows:

- Käyttölupa (ydinlaitos)
- Maankäyttösopimus
- Maankäyttösopimus / kaavoitus
- Naapurikuuleminen
- Osayleiskaava tai asemakaava
- Periaatepäätös (VN)
- Rakentamislupa (ydinlaitos)
- Verkkoliityntäsopimus
- Vesilupa (jäähdytysvesi)
- Vesilupa (padotus, rakentaminen)
- YVA-menettely
- YVA-menettely (tarvitt.)
- YVA-menettely (≥10 MW / ≥5 voimalaa)
- Ympäristölupa (tarvitt.)

## 4. Frontend — `_archI18n` IIFE, ET/LV/LT fall back to Finnish

Found while investigating item 3, not by the validator itself (this table
isn't in the coverage manifest yet — worth adding). In
`backend/static/index.html`, an IIFE merges a small `archKeys` table (the
architecture-drawing sidebar tool's labels — "Piirrä rakennuksen ääriviivat"
etc.) into `TRANSLATIONS` at runtime:

```js
for (const [lang, block] of Object.entries(TRANSLATIONS)) {
  for (const [key, vals] of Object.entries(archKeys)) {
    if (!block[key]) block[key] = vals[lang] || vals.FI;
  }
}
```

`archKeys` only defines FI/EN/SE/DA/NO/PL/DE — never ET/LV/LT. So
`vals[lang]` is `undefined` for those three, and every one of the ~9
architecture-tool labels (`h3_arch`, `arch_desc`, `arch_tool_polygon`,
`arch_tool_polyline`, `arch_tool_marker`, `arch_start`, `arch_stop`,
`arch_clear`, `arch_export`) silently falls back to Finnish for any
Estonian, Latvian, or Lithuanian user. Same bug shape as everything else in
this document, live today, previously undocumented.

## 5. Security hygiene — `/api/admin/retrieval-trace/{generation_id}` takes `secret` as a query parameter

Found 2026-08-30 while live-verifying the YVL memo parallelization work.
`GET /api/admin/retrieval-trace/{generation_id}?secret=...` (`backend/main.py`,
`admin_retrieval_trace()`) authenticates via `ADMIN_SECRET` passed as a query
string parameter, not a header. Confirmed the real consequence, not just a
theoretical one: every call landed in Render's own plaintext access logs with
the full secret value visible in the URL
(`GET /api/admin/retrieval-trace/{id}?secret=<real value> HTTP/1.1`) —
anyone with log access (or anything downstream that ingests/forwards access
logs, proxies, browser history if ever called from a browser, etc.) sees the
real `ADMIN_SECRET` in plaintext, unlike a header-based value which normal
access-log formats don't capture. Not exploited, not urgent — this endpoint
is admin-only diagnostic tooling, not customer-facing — but a real gap
between "requires a secret" and "keeps that secret out of logs by
construction." Fix: same `Authorization: Bearer <token>` header pattern
already used by `_check_ingest_auth()` for `INGEST_SECRET` elsewhere in
`main.py` — that one is already correct, this one isn't consistent with it.
Same fix pattern is worth checking against any other admin endpoint using
`secret: str = ""` as a query param (not audited yet — this was found
incidentally, not via a deliberate sweep).

## 6. Reliability/cost — ARQ `job_timeout` doesn't stop the underlying generation thread; it keeps running (and spending) after the client sees "failed"

Found 2026-08-30/31 while live-verifying the YVL memo parallelization work
(4 real timeout hits across 2 days made this visible). `arq_task_generate_permit`
(`backend/main.py`) does `pdf = await asyncio.to_thread(apply_proofread_to_pdf, ...)`.
When ARQ's `job_timeout` expires, `asyncio.wait_for()` cancels the *asyncio Task*
awaiting that thread — but `asyncio.to_thread()` runs on a real OS thread via
`concurrent.futures.ThreadPoolExecutor`, and Python cannot force-kill a running
thread. The thread keeps executing to completion regardless, making real,
billed Claude API calls the whole way, while the client-visible job status is
already frozen at `"status": "error", "error": "CancelledError: "` the moment
the timeout fires — permanently, nothing ever updates it afterward even though
the thread keeps making genuine progress.

Confirmed with real data, not inferred: two 2026-08-30 test jobs (`ef3ff6af34`,
`2cbfcd04c2`) both show Claude API calls completing in `retrieval_trace`'s cost
log **minutes after** their reported kill time — `2cbfcd04c2` is the clearest
case: killed at 1200.00s, yet `yvl_memo_A.1` and the entire RAQS review both
completed successfully ~3 minutes later. The orphaned thread ran the *whole
pipeline* to real completion — very likely including full PDF assembly — and
none of it was ever surfaced to the client; the result is silently discarded
when the thread's `run_in_executor` future tries to resolve against an
already-cancelled asyncio Task. `b56fa4719c` shows only partial orphaned
completion (one YVL guide, not all three) — its A.1 call completed at
07:35:47, itself already 5m24s *after* ARQ had marked the job terminally
failed at 07:30:23 — and the next deploy (PR #134) landed 12 minutes later
at 07:47:21. Plausible (not proven by a direct log line, unlike the rest of
this entry) that this specific deploy's process restart cut off that
orphaned thread mid-B.1, which a real deploy (unlike ARQ's own soft
cancellation) genuinely can do -- real wasted work either way, whichever
guide it was mid-generating when torn down.

Real cost impact, not theoretical: summed real `retrieval_trace` cost data
across 5 test attempts (2026-08-30/31) = **$7.51**, most of it from orphaned
work that produced nothing usable. This isn't unique to the YVL-memo case —
every hanketyyppi's generation goes through the same `asyncio.to_thread()`
call, so any timeout on any generation, past or future, has silently spent
money on discarded work with zero record of it beyond digging through
`retrieval_trace` by hand.

Checked and ruled out for this specific window, not just asserted: ARQ's
`retry_jobs=True`/`max_tries=5` defaults are unoverridden in
`_build_arq_worker()`, and its retry path *does* fire on a bare
`asyncio.CancelledError` (distinct from the `TimeoutError`-wrapped kind a
`job_timeout` expiry produces, which ARQ's retry check excludes and which
is confirmed via real ARQ source + zero duplicate "START" log lines across
all 5 test jobs to never have retried). A bare `CancelledError` — e.g. the
worker process being torn down by a deploy while `run_job()` is still
*actively* executing, before its own `job_timeout` has fired — genuinely
would trigger a retry, which combined with this same orphaned-thread issue
could double-bill for real (orphaned thread completes AND a retry re-runs
the whole thing from scratch). Cross-checked every deploy from PR #135's
revert (2026-08-30) through today against every test job's real ARQ-active
window (start → its own `job_timeout` firing, confirmed via real
timestamps, not the later orphaned-completion tail): zero overlaps, margins
15s-23min. This specific compounding scenario did not occur in any of the 5
real test attempts — the $7.51 total is genuinely complete, no hidden
retry-driven duplicate calls hiding in it. Still a real latent risk for
whenever a deploy DOES land mid-job in the future, sharing the same root
cause as the orphaned-thread issue above and worth fixing together.

Not fixed here — tracked for whenever it's prioritized. Real fix needs
actual engineering thought (a cooperative-cancellation mechanism checked
periodically inside `apply_proofread_to_pdf()`/`generate_pdf()` — similar in
spirit to this session's own `cap_event` mechanism in `_yvl_memo_one_guide()`,
just triggered by the ARQ-level timeout instead of the cost cap — or,
more simply, treating a large `job_timeout` as acceptable and instead making
sure a thread that DOES finish after the "official" deadline still gets its
result surfaced somewhere retrievable, rather than silently discarded).

## Suggested next step

Once this backlog is picked up: run both validators for real first
(`python scripts/validate_country_coverage.py`, `python scripts/validate_frontend_coverage.py`)
to reconfirm nothing above has drifted, then work through in whatever order
makes sense — the `_LAW_TRANS`/`_LAW_CITATION_REPLACEMENT` table (section 1)
is the largest single item and could reasonably follow the same
per-country-pass rhythm as the rest of this sprint's law-citation work.
