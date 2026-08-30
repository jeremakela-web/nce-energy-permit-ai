# YVL Compliance Memo: parallelize the 3 per-guide Claude calls

Investigation and proposal only, per instruction — **nothing here has been built.**
Follow-up to `RAQS_CHANGE_MANAGEMENT_PROPOSAL.md` (STUK priority 3), triggered by
real timeout failures found while verifying that work.

## What real data showed

Two live SMR + full YVL Compliance Memo test runs, 2026-08-30 (verifying PR #133):

- Run 1 (`job_timeout=900s`): retrieval+draft+proofread checkpoints all completed
  in 4m2s; job killed by the 900s ceiling ~11 min later with **zero** YVL-memo
  Claude calls yet logged as complete.
- Run 2 (`job_timeout=1500s`, temporary): same start, but this time the trace
  captured one full YVL-memo call before the kill: **`yvl_memo_A.1` alone took
  12m1s (37,550 output tokens, ~52 tokens/sec)**.

`_yvl_compliance_memo()` (`permit_ai/generate_application.py:10681-10802`) runs
its 3 per-guide calls (A.1, B.1, C.1) in a plain `for` loop — sequential, not
concurrent. B.1's source is the same size as A.1 (~190K chars each, per the
function's own 2026-08-25 comments); C.1's source is smaller (~44K chars) but
documented as needing *more* output room, not less, to avoid truncation. There's
no basis to expect either remaining call to be meaningfully faster than A.1's.
**Realistic total for the 3-call sequential path: 30-45+ minutes**, before RAQS
or PDF assembly. `job_timeout=900` (set before this 3-call design existed, PR
#119, 2026-08-25) and even the tested `1500` both fall well short.

`job_timeout=900` has been reverted (was never a real fix, and left raised it's
a live worker-capacity risk: `max_jobs=2`, so one long SMR job can tie up half
of total generation capacity for up to 25+ min). This means, as of now, **every
SMR/smr_bess generation that reaches the full YVL memo path will keep failing**
until either this is fixed or the timeout is raised again. Confirmed via
production log search: this has hit **zero real customer jobs** so far (see
below) — but it will start hitting them the moment real SMR/smr_bess traffic
arrives, so this isn't low-urgency, just not yet damaging.

`smr_bess` triggers the identical path: `_YVL_MEMO_HANKE_TYPES = {"SMR",
"smr_bess"}` gates entry, and `_YVL_MEMO_COVERED` (the fixed A.1/B.1/C.1 set) is
a single module-level dict with no hanketyyppi-based variation — smr_bess is not
a lighter path.

## Proposal: run the 3 calls concurrently instead of sequentially

The 3 per-guide calls are independent — each takes one guide's own source text
and produces one memo section, no dependency between them. Running them
concurrently instead of sequentially should cut this phase's wall-clock time
roughly 3x (bounded by whichever single call is slowest, not the sum of all
three) — plausibly bringing the whole generation back under the original 900s
ceiling, or at least close enough that a modest bump (not a 45-60 min one)
covers it.

**Mechanism**: `_yvl_compliance_memo()` is a plain synchronous function, already
invoked from inside a background thread (`arq_task_generate_permit` calls
`asyncio.to_thread(apply_proofread_to_pdf, ...)`, which eventually reaches this
function synchronously). The simplest fit for that shape is
`concurrent.futures.ThreadPoolExecutor`, not `asyncio.gather` — introducing
asyncio here would mean spinning up a nested event loop inside an
already-backgrounded thread for no real benefit, since each per-guide Claude
call is I/O-bound (waiting on the API) and threads already release the GIL
during that wait. `ThreadPoolExecutor(max_workers=len(_YVL_MEMO_COVERED))`,
submit one worker per guide, matches the existing sync architecture directly.

**Things that need explicit handling, not just "wrap it in a pool"**:

1. **Cost guardrail cap** (`_CLAUDE_CALL_CAP`, checked via
   `_retrieval_trace.count_claude_calls()` before each call today): the current
   sequential loop stops launching further guide calls the instant the cap
   trips mid-loop (real cost savings on a capped generation). Parallel launch
   loses that fail-fast property — all 3 would fire before any of them could
   observe another's completion. Proposed fix: a single up-front check before
   submitting any worker (`count_claude_calls(generation_id) + len(pending) >
   _CLAUDE_CALL_CAP` → raise `GenerationCapError` immediately, same as today,
   just checked once instead of per-iteration) rather than trying to preserve
   per-call fail-fast under concurrency, which would be racy anyway.
2. **Order preservation**: `memo_sections`/`actual_covered` are currently built
   in `_YVL_MEMO_COVERED` iteration order (A.1, B.1, C.1), and the final
   `memo_text` join depends on that order for a coherent read. Parallel
   completion order is not the same as submission order — needs futures
   collected into a list indexed by original position and reassembled in that
   order before joining, not just appended as each one finishes.
3. **Per-guide failure isolation**: today, one guide's API failure is caught
   (`except Exception ... continue`) without affecting the other two — this
   must survive parallelization (each future's exception handled
   independently; a failed future just means that guide falls through to the
   pending list, exactly as today).
4. **`retrieval_trace.py` thread-safety** — checked, not assumed: all write
   functions this touches (`log_api_call`, `log_guardrail_hit`) already use a
   module-level `threading.Lock()` around every write
   (`permit_ai/retrieval_trace.py:71` + `with _lock, _connect() as conn:` at
   each call site). `count_claude_calls()` is a lock-free read, which is fine —
   same race tolerance that already exists today, not something concurrency
   changes qualitatively. No new thread-safety work needed here; this was a
   real risk worth checking before proposing, and it checks out clean.
5. **Anthropic API rate limits** — the one real unknown, not yet checked: 3
   large (`max_tokens=48000`) requests fired near-simultaneously could hit a
   per-minute token-rate limit that naturally-spaced sequential calls don't.
   This needs a real live test to confirm, not just code review — can't be
   verified by reading the code alone.
6. **Cost**: unchanged — same 3 calls, same tokens, just concurrent instead of
   sequential. No cost-guardrail semantics change beyond the cap-check timing
   in point 1.

## Not proposing here

- No change to `_CLAUDE_CALL_CAP` itself, RAQS, PDF assembly, or anything
  outside `_yvl_compliance_memo()`'s own loop.
- No permanent `job_timeout` change yet — that's a downstream decision once
  real parallel-run timing data exists. A modest margin above whatever the
  parallel path actually measures at is the likely right call, not a return to
  guessing.

## Prior-incident check (your priority-2 ask, answered)

Cross-checked three independent log signals across the full production log
window, 2026-08-25 (when the 3-call YVL memo shipped) through now:

- `enqueued OK` (every job accepted): **4 total**
- `START hanke=` (every job that actually started): **4 total** — matches
  exactly, no dropped/lost jobs
- `arq_task_generate_permit failed...TimeoutError`: **2 total**

All 4 real generation jobs in this entire window, by timestamp:

| time | job | hanketyyppi/country | outcome |
|---|---|---|---|
| 2026-08-27 10:13 | `1edb18e1a3` | BESS/EE | succeeded (this session's earlier EE/ET verification) |
| 2026-08-27 10:20 | `11e9348e1e` | BESS/EE | succeeded (same) |
| 2026-08-30 07:15 | `b56fa4719c` | SMR/FI | **timed out at 900s** — this session's own test |
| 2026-08-30 07:47 | `d4d140add7` | SMR/FI | **timed out at 1500s** — this session's own test |

**Zero real customer/production SMR or smr_bess generations have been attempted
since the YVL memo shipped on 2026-08-25 — let alone failed.** Both TimeoutErrors
are this session's own test runs from today. This isn't "ordinary jobs have been
silently failing" — there have been no other SMR/smr_bess jobs of any kind in
this window to fail. The 900s ceiling's mismatch with the 3-call YVL memo design
is real and will bite the first real SMR/smr_bess customer who reaches this
path, but it hasn't yet.

## PDF-render verification status

Parked, as instructed — explicitly incomplete, not blocking. The API-level
wiring (add/approve/notify/implement, all 5 endpoints) is fully verified live
and correct. The PDF-render half needs either this parallelization fix or a
much larger timeout to complete a real end-to-end SMR test; revisit once one of
those lands.
