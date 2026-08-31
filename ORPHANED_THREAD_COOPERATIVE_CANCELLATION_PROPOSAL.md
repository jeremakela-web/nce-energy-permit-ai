# Orphaned generation threads: cooperative cancellation proposal

Investigation and proposal only, per instruction — **nothing here has been built.**
Follow-up to `COVERAGE_GAPS_BACKLOG.md` item 6.

## The real constraint, confirmed by reading Python's own semantics

`asyncio.to_thread()` (used twice in `arq_task_generate_permit`, `backend/main.py`)
is a thin wrapper: `await loop.run_in_executor(None, func)`. When ARQ's own
`job_timeout` fires, it cancels our task, which propagates a `CancelledError`
into that `await` — but `concurrent.futures.Future.cancel()` only succeeds
if the work **hasn't started yet** (Python's own documented behavior). Once
a thread is running, nothing external can stop it. This isn't a bug in this
codebase's use of `asyncio.to_thread` — it's a hard Python limitation. The
only real fix is **cooperative**: the code running *inside* the thread has
to periodically check "should I stop?" and voluntarily exit at a clean
point, on its own.

## Where the risk actually concentrates — and the lucky part

Checked real timing data across all 6 SMR test runs: `generate_application_draft`
(retrieval+draft+proofread) is fast and reliable everywhere, ~2-4 min, never
the cause of a timeout. The entire risk is in `_yvl_compliance_memo()`'s
concurrent phase — one guide alone has been observed taking 12-15+ minutes.

The lucky part: **this is exactly the code I already touched for the
parallelization fix.** `_yvl_memo_one_guide()` already has a shared
`cap_event: threading.Event`, checked before starting, before each
rate-limit retry, and every 50 chunks mid-stream — closing an in-flight
stream via `stream.close()` if set. That mechanism is currently wired to
one specific trigger (`_CLAUDE_CALL_CAP`). The proposal below is to make it
respond to a *second*, independent trigger — a deadline — using the exact
same checkpoints, not new plumbing.

## Proposed design

**1. One shared signal object, two possible triggers.** Instead of a bare
`threading.Event`, a tiny wrapper so downstream code/logs can tell *why* a
worker stopped:

```python
@dataclass
class AbortSignal:
    event: threading.Event
    reason: str = ""   # set once, alongside event.set(): "cost_cap" | "deadline"
```

`_yvl_memo_one_guide()`'s existing checks change from `cap_event.is_set()`
to `signal.event.is_set()` — same checkpoints, same `stream.close()`
behavior, no new logic there. At the end of `_yvl_compliance_memo()`, branch
on `signal.reason` to raise the right exception (`GenerationCapError` for
cost, a new `GenerationDeadlineExceeded` for the timeout case — see below).

**2. An internal watchdog, set with real margin before ARQ's own ceiling.**
In `arq_task_generate_permit`, before the two `asyncio.to_thread()` calls:

```python
_JOB_START = time.monotonic()
_DEADLINE_BUFFER_S = 90   # tunable — real margin for the mid-stream check to catch up
signal = AbortSignal(event=threading.Event())

async def _watchdog():
    remaining = _GENERATION_JOB_TIMEOUT_S - _DEADLINE_BUFFER_S - (time.monotonic() - _JOB_START)
    if remaining > 0:
        await asyncio.sleep(remaining)
    signal.event.set()
    signal.reason = signal.reason or "deadline"

watchdog_task = asyncio.create_task(_watchdog())
try:
    draft_bytes, sections, sources = await asyncio.to_thread(generate_application_draft, inp, signal=signal)
    pdf = await asyncio.to_thread(apply_proofread_to_pdf, inp, sections, sources, signal=signal)
finally:
    watchdog_task.cancel()
```

`_GENERATION_JOB_TIMEOUT_S` needs to become a single module-level constant
referenced both here and in `_build_arq_worker()`'s `job_timeout=` kwarg —
today that number only exists once, hardcoded into the `Worker(...)` call;
introducing a second hardcoded copy for the watchdog would be a real
single-source-of-truth risk worth avoiding from the start.

**3. `signal` threaded through the existing call chain**, as an optional,
default-`None` parameter so nothing else that calls these functions
directly (tests, any future caller) breaks: `apply_proofread_to_pdf()` →
`generate_pdf()` → `_yvl_compliance_memo()` → `_yvl_memo_one_guide()`. Five
function signatures across two files — mechanical, not risky, but worth
being upfront that it's not a one-line change. `generate_application_draft`
gets the same optional parameter too, checked once before each of its ~2-3
Claude calls — draft/proofread aren't the real risk, but it's cheap
defense-in-depth and keeps the mechanism consistent everywhere a Claude
call happens in this path.

**4. Handling the new exception — confirmed to slot directly into an
existing, well-structured ladder, not something novel.** Checked
`arq_task_generate_permit`'s real exception handling: it already has
`except InsufficientSourcesError`, `except GenerationCapError`,
`except Exception`, `except BaseException` — in that order, each setting a
clean `_proofread_store[job_id]["status"]` without re-raising, *except* the
final `except BaseException`, which is what currently catches
`asyncio.CancelledError` (it doesn't inherit from `Exception`), sets
status, and **re-raises** — that re-raise is exactly why ARQ ends up seeing
a hard failure today. A new `except GenerationDeadlineExceeded as exc:`
block, placed before the generic `except Exception`, would set a clean,
honest status (e.g. `"status": "timeout_soft_abort"`, with real
elapsed/guide-completion info in `error`) and **not re-raise**. That's the
actual fix's payoff: the coroutine returns normally, ARQ sees a genuinely
completed job (not a timeout), no `CancelledError` ever reaches the thread,
no orphaned execution, no silently discarded work, no "! ... failed,
TimeoutError:" log noise, and the client gets a real, specific, immediate
answer instead of the current opaque black hole.

ARQ's own `job_timeout` stays as-is (1800s) — it becomes a backstop that
should essentially never fire once this ships, not the primary mechanism.
Worth keeping, not removing: if the cooperative check is ever wrong (a bug,
or a genuinely uncovered slow section), the hard ceiling is still there.

## Question 2: capturing late/orphaned results anyway

Worth doing, and small — not worth skipping given how cheap it is once (1)
exists, though it becomes a true rare-edge-case once the watchdog is in
place (the mid-stream check already fires every 50 chunks, well inside the
90s buffer for any realistic chunk rate).

Mechanism: swap `asyncio.to_thread(func, ...)` for the equivalent explicit
`loop.run_in_executor(None, functools.partial(func, ...))`, and attach
`future.add_done_callback(...)` **before** awaiting it. This callback fires
whenever the underlying thread actually finishes — genuinely independent of
whether the *awaiting* coroutine got cancelled elsewhere (confirmed: this is
standard `concurrent.futures`/`asyncio` behavior, not something that needs
new machinery). On late completion, the callback writes into
`_proofread_store[job_id]["late_completion"] = {"pdf_bytes": ..., "completed_at": ...}`
rather than letting the result vanish. `_proofread_store` is a plain
in-process dict already mutated from multiple contexts without an explicit
lock elsewhere in this codebase — a single-key assignment here is
consistent with that existing (implicit) concurrency assumption, not a new
risk.

Would need one small additional surface: exposing `late_completion` via
`GET /api/proofread/{job_id}` (or a tiny dedicated admin lookup) so it's
actually retrievable, not just recorded. Scoped small deliberately — this
is insurance for the rare case the primary mechanism doesn't catch it, not
a second feature to build out.

## Not proposing here

- No change to ARQ itself, `max_jobs`, or `retry_jobs`/`max_tries` (separate,
  already-investigated, already-ruled-out-as-currently-happening risk from
  `COVERAGE_GAPS_BACKLOG.md` item 6).
- No change to `job_timeout`'s value (1800s stays, confirmed with real
  margin from the successful `60a92f2509` run).
- Not extending the watchdog/signal mechanism to other hanketyyppi's
  generation paths beyond what already flows through
  `generate_application_draft`/`apply_proofread_to_pdf` — every hanketyyppi
  already goes through this same call chain, so this fix is automatically
  general, not SMR-specific, without extra work.

## Summary / ask

- Primary fix: a shared `AbortSignal` (event + reason), a watchdog task with
  a real safety buffer, threaded through the existing call chain via a new
  optional parameter, reusing `_yvl_memo_one_guide()`'s already-built
  checkpoints. A new `GenerationDeadlineExceeded` exception slots directly
  into `arq_task_generate_permit`'s existing, already-well-structured
  exception ladder.
- Secondary, small: capture late-arriving results via `run_in_executor` +
  `add_done_callback` instead of `asyncio.to_thread`, so a rare miss isn't a
  total loss.
- Five function signatures change across `backend/main.py` and
  `permit_ai/generate_application.py`. Mechanical, not risky, but real work
  — flagging the actual scope rather than understating it.

Waiting for review before building any of this.
