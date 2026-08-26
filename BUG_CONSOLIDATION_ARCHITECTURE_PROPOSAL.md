# Bug-consolidation architecture proposal: a country/language coverage registry

Proposal only — no code built here, per standing practice. Written on request
after confirming no prior version of this proposal exists anywhere reachable
(full repo search: docs, git log `--all`, every local/remote branch, open
PRs/issues, memory; two peer sessions pinged, no reply as of writing).

## 1. The pattern, named across its seven real instances

Every instance below is structurally the same bug: **a country- or
language-keyed dict (or doc) is supposed to have one entry per supported
country/language, some subset actually does, and nothing detects the gap
before a customer — or a manual QA pass — hits it live.**

| # | Instance | Where | What broke |
|---|---|---|---|
| 1 | Traficom/STUK/ELY-keskus leak | `_COUNTRY_LIITTEET` (several countries missing SMR/smr_bess overrides), plus hardcoded entity names in template prose | PR #36-39: DA/NO reports showed Finland's aviation authority *alongside* the correct national one — a general replace-instruction wasn't enough against a PAKOLLINEN directive elsewhere in the same prompt |
| 2 | LUPA_I18N | `index.html` (frontend JS) | PR #95: `_COUNTRY_LUVAT` grew from a 14-string vocabulary to 185 once PR #94 wired the permit-name box to real per-country data; the translation table it fed through was never grown to match, so SE/DA/NO/PL customers saw raw Finnish permit-type strings the day before a live customer demo |
| 3 | Coverage-box translations | `TRANSLATIONS.ET` and siblings | `TRANSLATIONS.ET` was 63/398 keys (~16%) before a dedicated audit found and fixed 348 missing keys — a language-selector option existed in the UI pointing at a translation table that was mostly empty |
| 4 | Wrong-country authority names in attachment lists | `_COUNTRY_LIITTEET` | Same shape as #1, recurring: a hanketyyppi override exists for some countries and not others, so the missing ones silently render the Finnish base liite list (Finnish authorities, Finnish law names) as if it were correct for that country |
| 5 | RAQS language leaks | RAQS self-assessment / structured-criterion generation | PRs #89, #103: self-assessment page and RAQS "source humanization" both needed dedicated fixes to stop Finnish leaking into non-Finnish generation output — same missing-language-branch shape, different subsystem |
| 6 | SE law citations | `_LAW_TRANS`, `_LAW_CITATION_REPLACEMENT`, `_COUNTRY_LIITTEET` | PR #106: Swedish reports were citing bare Finnish statute numbers with no Swedish gloss, because the citation-translation table `_t_law()` routes through simply had no `"SE"` key for several statutes |
| 7 | Stale `FRONTEND_API_REFERENCE.md` | Documentation, not runtime | PR #117: the doc claimed 20 hanketyyppi / 6 countries; the real code had 23 hanketyyppi / 9 countries. A "registry of truth" document had itself drifted from the thing it was documenting — same failure mode one level up |
| — | `_BESS_MARKET_DATA` (reported separately this session) | `_BESS_MARKET_DATA` | Missing DE/LT at the country level, falls back to Finland's own market index, cited unconditionally as a real source in the PDF's "Lähteet" section |

Same root cause every time, at two different layers: **(a)** a country/language
axis is expressed as N independent, uncoordinated dicts/tables instead of one
canonical list with per-artifact coverage checked against it, and **(b)** the
failure mode on a missing key is silent fallback to Finnish content (or, for
docs, nothing at all) rather than a loud, pre-merge failure.

This list is itself evidence for the proposal's core claim: fixing instance
#N doesn't prevent instance #N+1, because each fix has been local (add the
missing key to *this* dict) rather than structural (make it impossible for
*any* dict to silently omit a supported country again).

## 2. Proposed architecture: a coverage registry, not a rewrite

**Explicitly not proposing**: migrating the ~15-20 existing country/language
dicts into one mega-structure. That's a large, high-regression-risk change
for zero behavior difference, and it would touch every one of the ~30 PRs'
worth of carefully-researched content built this sprint. Rewrite risk is
disproportionate to the actual problem, which is *detection*, not storage.

**Proposed instead**: a small, additive registry + validator that sits
alongside the existing dicts and knows about them, without owning their data.

### 2.1 One canonical list

```python
# country_registry.py (new file)
_SUPPORTED_COUNTRIES: tuple[str, ...] = ("FI", "SE", "DA", "NO", "PL", "DE", "EE", "LV", "LT")
_SUPPORTED_LANGUAGES: tuple[str, ...] = ("FI", "EN", "SE", "DA", "NO", "PL", "DE", "ET", "LV", "LT")
```
Everything else derives from this — today, "how many countries does this app
support" has to be answered by reading `_COUNTRY_CONFIG`'s keys and hoping
every other dict agrees (exactly the gap PR #117 found in
`FRONTEND_API_REFERENCE.md`).

### 2.2 A coverage manifest, one line per existing dict

Full classification of every country/language-keyed dict currently in
`permit_ai/generate_application.py` (15 backend dicts; the 2 frontend tables
are covered separately in §2.4 since they're JS, not Python). **HARD** means
every supported country/language must have a real entry or the validator
fails; **SOFT** means partial coverage can be legitimate, but only when the
specific missing keys are individually justified — never as a blanket "this
whole dict is allowed to be incomplete."

| Dict | Axis | Severity | Why |
|---|---|---|---|
| `_LANG_INSTRUCTIONS` | language | **HARD** | Missing = model gets zero "write in X" instruction at all (Class 2 in the LT investigation — a void, not even a leak). No legitimate reason for a supported language to lack this. |
| `_WRITE_INSTRUCTION` | language | **HARD** | Missing = Finnish write-instruction text leaks into the prompt verbatim. |
| `_PROMPT_HEADERS` | language | **HARD** | Missing = Finnish section headers/labels leak into prompt construction. |
| `_PDF_STRINGS` | language | **HARD** | Missing = up to 78 Finnish UI labels leak directly into the customer-visible PDF — the single most customer-visible instance of this bug class found so far. |
| `_CRITICAL_EXTRA` | language | **HARD** | Missing = the "⚠️ expert-review-required" guardrail paragraph silently disappears for that language on any `_CRITICAL_HANKE_TYPES` hanketyyppi (SMR, smr_bess, ymparistolupa) — a real content-quality safeguard, not just cosmetic. |
| `_HANKE_NIMI_TRANS` | language | **HARD** | Missing = the hanketyyppi's own display name doesn't translate — same "leaks into visible UI/PDF" shape as `_PDF_STRINGS`. |
| `_HUOM_LABEL` | language | **SOFT — verified legitimate** | Missing falls back to `"[Note] "`, a short, generic, safe English marker, confirmed not Finnish text. This is the one clean example of a fallback that's actually fine as shipped; kept SOFT deliberately, not because nobody's checked it. |
| `_COUNTRY_CONFIG` | country | **HARD** | Missing = the country doesn't exist in the app at all (authorities, key_laws, prompt_prefix). Not an optional field by definition. |
| `_COUNTRY_LUVAT` | country | **HARD**, but see caveat below | Missing hanketyyppi-for-country = falls through to the raw Finnish base permit table (Finnish authorities, Finnish law names) — the exact shape of instances #1 and #4. **Caveat**: today this is genuinely incomplete for legitimate reasons too (e.g. `egs`/geothermal is DE-only because DE is the only country with real geothermal project activity researched so far) — so the real check has to be per-(country × hanketyyppi actually offered to that country in the UI), not a flat "every country needs all 21 hanketyyppi." Flagging this as the one dict where the manifest entry needs a small allowlist of legitimately-not-offered combinations, not a plain HARD/SOFT toggle — worth designing carefully rather than assuming HARD=simple. |
| `_COUNTRY_LIITTEET` | country | **HARD**, same caveat as `_COUNTRY_LUVAT` | Same failure shape (#1, #4) and same nuance: a hanketyyppi genuinely not offered for a country isn't a gap. |
| `_NATIONAL_SUPERVISORS` | country | **HARD** | Missing = Finland's own supervisory authorities (Tukes, Pelastuslaitos) get named as that country's regulator in prompt context — confirmed real risk given the Traficom/STUK precedent, not just theoretical. |
| `_BESS_MARKET_DATA` | country | **HARD** | Missing = Finland's own market index figure gets cited as a real source in that country's PDF, unconditionally, for every BESS report. Already confirmed live for DE/LT (reported separately this session). |
| `_LAW_TRANS` | country | **HARD, but per-row, not per-dict** | This is where my first draft was too coarse, and it matters: `_LAW_TRANS` isn't one country-keyed dict, it's ~35 independent statute-keyed dicts, each separately needing full country coverage. A whole-dict SOFT/"allow_partial" check would have missed PR #106 entirely — the SE law-citation bug was exactly "some statute-rows have an SE entry and others don't," which only shows up if you check every row, not the dict's aggregate key set. Corrected design: each of the ~35 rows gets its own HARD coverage check, each with its own `verified_absent` set (e.g. `Patoturvallisuuslaki`'s row legitimately excludes PL/LT — no separate dam-safety act exists there, confirmed via research — but that's a per-row fact about *that statute*, not a property of the whole dict). |
| `_LAW_CITATION_REPLACEMENT` | country | **HARD, per-row** | Same correction as `_LAW_TRANS`, same reason — it's the same statute-keyed shape. |
| `_STUK_REPLACEMENT` | country | **SOFT, but conditional, not unconditional** | My first draft marked this flatly SOFT ("only needed where FI nuclear text could leak") — too permissive as stated, because whether that's true depends on another dict's state, not a fixed fact. Corrected: a country's absence from `_STUK_REPLACEMENT` is only legitimate if `_COUNTRY_LIITTEET`'s SMR/smr_bess entries for that country are *confirmed* to contain zero literal `"STUK"` substring (already true for SE/DA/NO/PL/EE/DE per the 2026-07-28 audit) — the validator should check that condition directly, not just trust a static allowlist that could go stale the next time someone edits `_COUNTRY_LIITTEET`. |

The manifest is the one place a developer registers "this new dict needs
coverage checking" — writing a new per-country/language dict without adding
it here is itself a checklist item (§3, step 5).

### 2.3 A validator that fails loudly, not a fallback that fails silently

```python
# scripts/validate_country_coverage.py
def validate() -> list[CoverageGap]:
    gaps = []
    for spec in _COVERAGE_MANIFEST:
        supported = _SUPPORTED_COUNTRIES if spec.axis == "country" else _SUPPORTED_LANGUAGES
        missing = set(supported) - set(spec.dict_obj.keys())
        # SOFT + allow_partial specs need an explicit "verified absent" marker,
        # not just silence, to count as intentionally incomplete -- see 2.5.
        if spec.severity == "HARD" and missing:
            gaps.append(CoverageGap(spec, missing))
        elif spec.severity == "SOFT" and missing - spec.verified_absent:
            gaps.append(CoverageGap(spec, missing - spec.verified_absent, soft=True))
    return gaps
```

Run two ways:
- **As a script**, so it can be run by hand right now against current `main`
  — this alone would enumerate every one of the 7 instances above, today,
  before any of them needed a live customer or a manual QA pass to surface.
- **As a pytest test** (`test_country_coverage.py`) that fails CI on any HARD
  gap. There's currently no CI on this repo (`gh pr checks` returns nothing
  configured) — this would be the first real regression test the repo gets,
  scoped narrowly to the one bug class that's recurred seven times.

### 2.4 The frontend half

`LUPA_I18N`, `TRANSLATIONS.<LANG>`, and `index.html`'s country/language
selector wiring live in JS, not Python. Two real instances (#2, #3) are on
this side. Proposed: a small companion script
(`scripts/validate_frontend_coverage.py`) that parses `index.html`'s JS
objects with a regex/AST-lite approach (same technique already used
successfully for the `TRANSLATIONS.ET` key-diff audit) and checks them
against the same `_SUPPORTED_COUNTRIES`/`_SUPPORTED_LANGUAGES` list — kept as
a second script rather than trying to share one Python validator across two
languages.

### 2.5 The "verified absent" problem — don't let this become false-positive noise

Several existing gaps are *intentional and already documented*, not bugs:
`_LAW_CITATION_REPLACEMENT["Patoturvallisuuslaki"]` deliberately has no PL/LT
entry (no separate dam-safety act exists there, confirmed via research, see
`PL_LAW_CITATION_RESEARCH.md`/`LT_LAW_CITATION_RESEARCH.md`) and no DA entry
(genuinely unconfirmed after two searches, explicitly left open rather than
guessed). A validator that flags these as gaps every run would train
everyone to ignore its output — worse than not having it.

Fix: `verified_absent: dict[str, str]` per SOFT spec — a country code mapped
to a one-line reason, sourced directly from the research docs' own
"deliberately no entry" comments (several already exist verbatim in the code
as Python comments, e.g. the Patoturvallisuuslaki block above
`_LAW_CITATION_REPLACEMENT`). Only *undocumented* absence counts as a gap.
This also means writing "verified absent, reason: X" becomes part of the
onboarding/research workflow itself, not an afterthought.

## 3. New-country onboarding checklist

Derived directly from mapping each of the 7 incidents to where it actually
lived, ordered the way a developer would naturally encounter it. This is the
literal artifact I'd add as `NEW_COUNTRY_ONBOARDING_CHECKLIST.md`. Each step
is annotated with exactly which of the seven named incidents (§1's table) it
would have caught pre-merge, so the mapping is explicit rather than implied:

1. **Add the country code to `_SUPPORTED_COUNTRIES`** in `country_registry.py`
   (and `_SUPPORTED_LANGUAGES` if the country brings a new language). Nothing
   else in this checklist works until this line exists.
   *Catches: none directly — this is the precondition every other step's
   detection depends on.*
2. **Run `scripts/validate_country_coverage.py`.** It now enumerates every
   dict missing the new country by name.
   *Catches: #2 (LUPA_I18N), #3 (coverage-box/TRANSLATIONS.ET), #6 (SE law
   citations), and the `_BESS_MARKET_DATA` leak — all four were exactly
   "a dict silently missing a country's key," which is precisely what this
   step makes impossible to overlook. This single step is the biggest lever
   in the whole checklist: 4 of 8 tracked incidents map to it directly.*
3. **Frontend**: `TRANSLATIONS.<LANG>` full completion (reuse the key-diff
   technique from the ET audit), `LUPA_I18N`, `index.html` country/language
   selector wiring, run `validate_frontend_coverage.py`.
   *Catches: #2 (LUPA_I18N) and #3 (coverage-box translations) specifically —
   these are the two JS-side incidents; step 2's Python validator can't see
   them, so this step is where their frontend companion actually closes the
   gap step 2 only flagged in principle (§2.4).*
4. **Backend prompt-construction dicts**: `_LANG_INSTRUCTIONS`,
   `_WRITE_INSTRUCTION`, `_PROMPT_HEADERS`, `_CRITICAL_EXTRA`, `_HUOM_LABEL`,
   `_PDF_STRINGS`, `_HANKE_NIMI_TRANS`.
   *Catches: #5 (RAQS language leaks) most directly — the self-assessment
   page and structured-criterion generation both leaked Finnish because a
   language-instruction dict was incomplete for that code path, same shape
   as this step's whole dict list. Also the general precondition that
   prevents a fresh instance of #5 in any new subsystem.*
5. **Backend country-specific legal/regulatory dicts** (the actual research
   work, same shape as this sprint's EE/DE/LV/SE/PL/LT/DA/NO passes):
   `_COUNTRY_CONFIG` (authorities/key_laws/prompt_prefix),
   `_COUNTRY_LUVAT`, `_COUNTRY_LIITTEET`, `_LAW_TRANS`,
   `_LAW_CITATION_REPLACEMENT`, `_STUK_REPLACEMENT` (if a nuclear framework
   needs one), `_NATIONAL_SUPERVISORS`, `_BESS_MARKET_DATA`. For any
   deliberately-absent entry, add it to the relevant row's `verified_absent`
   set with a one-line reason, not just silence.
   *Catches: #1 (Traficom/STUK/ELY-keskus leak) and #4 (wrong-country
   authority names in attachment lists) — both were `_COUNTRY_LIITTEET`/
   `_COUNTRY_LUVAT` rows missing for specific countries, which this step's
   research work (plus step 2's validator re-run) directly fills before
   first ship rather than after a live QA pass finds it. Also re-confirms
   #6 (SE law citations) at the per-row level per §2.2's correction.*
6. **Documentation**: `FRONTEND_API_REFERENCE.md`, `COVERAGE_MATRIX.md`.
   *Catches: #7 (stale FRONTEND_API_REFERENCE.md) directly — this step
   exists specifically because that doc drifted from live code for months
   before PR #117 caught it; updating it as a mandatory onboarding step
   (not an occasional audit) is the fix.*
7. **Live verification**: a real (not simulated) generation test in the new
   country/language, `rag-check-all` if RAG content applies, re-run
   `validate_country_coverage.py` one more time as the actual gate before
   calling the country "done."
   *Catches: whatever steps 1-6 structurally can't — anything that only
   shows up in real generated output (e.g. the original Traficom leak was
   two entities appearing *together*, which a static coverage check alone
   wouldn't have caught; it needed a real generated document to spot the
   PAKOLLINEN-directive interaction). This step is the backstop for
   emergent bugs, not a redundant re-check of steps 1-6.*

Steps 1-2 and 7 are new (they don't exist today); steps 3-6 are exactly the
work already being done per-country this sprint — the checklist doesn't
change *what* gets researched, only makes sure nothing gets silently skipped.

## 4. Scope and sequencing

Proposal only. If approved, the buildable pieces are:
- `country_registry.py` + `_COVERAGE_MANIFEST` (new, additive)
- `scripts/validate_country_coverage.py` + `test_country_coverage.py`
- `scripts/validate_frontend_coverage.py`
- `NEW_COUNTRY_ONBOARDING_CHECKLIST.md`

None of this touches existing dict content — running the validator for the
first time will surface a real, precise list of every current HARD gap
across all 9 countries (a superset of the 7 instances above, likely,
since some may not have been individually found yet). That list becomes a
new, better-organized version of the existing per-country backlog items,
not a reason to block this proposal on fixing them all first.

Not proposing to build anything for a 10th country — per instruction, this
is about making the *next* country onboarding safe, not starting new
geography work.
