# New-country onboarding checklist

Run through this in order whenever adding a 10th (or Nth) supported country.
Derived from `BUG_CONSOLIDATION_ARCHITECTURE_PROPOSAL.md`'s analysis of seven
real, named incidents of the same bug class — a country/language-keyed dict
silently missing an entry for one country, undetected until a customer or a
manual QA pass hit it live. Each step below names exactly which of those
incidents it would have caught, so the mapping is explicit, not assumed.

Not a substitute for the actual per-country legal/regulatory research — this
checklist doesn't change *what* gets researched, only makes sure nothing
gets silently skipped once it has been.

---

## 1. Add the country code

Add the new country code to `_SUPPORTED_COUNTRIES` in
`permit_ai/country_registry.py` (and to `_SUPPORTED_LANGUAGES` too, if the
country brings a language the app doesn't already support).

*Catches: nothing directly — this is the precondition every later step's
detection depends on. Nothing else in this checklist works until this line
exists.*

## 2. Run the backend coverage validator

```
python3 scripts/validate_country_coverage.py
```

This enumerates every backend dict missing the new country/language, by
name, with the exact row where relevant (e.g. `_LAW_TRANS['YSL 527/2014']`
specifically, not just "`_LAW_TRANS` is incomplete somewhere").

*Catches: LUPA_I18N-shaped gaps, coverage-box/TRANSLATIONS-shaped gaps, SE
law-citation-shaped gaps, and `_BESS_MARKET_DATA`-shaped gaps — all of these
were exactly "a dict silently missing a country's key," which is precisely
what this step makes impossible to overlook. This is the single biggest
lever in the whole checklist.*

## 3. Frontend

- Complete `TRANSLATIONS.<LANG>` in `backend/static/index.html` (reuse the
  key-diff technique from the historical ET completion: compare against
  `TRANSLATIONS.FI`'s key set).
- Complete `LUPA_I18N` for the new language across every permit-name row.
- Wire the new country/language into the UI's country/language selector.
- Run:
  ```
  python3 scripts/validate_frontend_coverage.py
  ```

*Catches: the two JS-side incidents (LUPA_I18N, coverage-box translations)
specifically — step 2's Python validator can't see either of these; this
step is where their frontend companion actually closes the gap.*

## 4. Backend prompt-construction dicts

`_LANG_INSTRUCTIONS`, `_WRITE_INSTRUCTION`, `_PROMPT_HEADERS`,
`_CRITICAL_EXTRA`, `_HUOM_LABEL` (optional — safe generic fallback),
`_PDF_STRINGS`, `_HANKE_NIMI_TRANS`.

*Catches: RAQS-language-leak-shaped bugs most directly — those happened
because a language-instruction dict was incomplete for a specific
generation code path; this step (plus re-running the validator) is the
general precondition that prevents a fresh instance of that shape in any
new subsystem.*

## 5. Backend country-specific legal/regulatory dicts

The actual research work — same shape as this sprint's EE/DE/LV/SE/PL/LT/
DA/NO passes, each documented in its own `<CC>_LAW_CITATION_RESEARCH.md`:

- `_COUNTRY_CONFIG` (authorities, key_laws, prompt_prefix — explicit FI→
  target-country statute mapping, not just a translated law name)
- `_COUNTRY_LUVAT` (per-hanketyyppi permit/authority/law table)
- `_COUNTRY_LIITTEET` (per-hanketyyppi attachment checklist)
- `_LAW_TRANS` and `_LAW_CITATION_REPLACEMENT` (per-statute-row country
  entries — real, verified statute names and dates, not templated off
  another country)
- `_STUK_REPLACEMENT`, if the country has a real nuclear framework the
  generic Finnish "STUK" backstop needs to be replaced for
- `_NATIONAL_SUPERVISORS`, `_BESS_MARKET_DATA`

For any statute or dict row where the target country genuinely has no
equivalent (e.g. no separate dam-safety act), add it to that row's
`verified_absent` set in `country_registry.py` with a one-line reason —
never leave it as silent absence, and never fabricate a plausible-sounding
law name to fill the gap.

*Catches: the Traficom/STUK/ELY-keskus-leak shape and the wrong-country-
authority-in-attachment-list shape — both were `_COUNTRY_LIITTEET`/
`_COUNTRY_LUVAT` rows missing for specific countries, which this step's
research work (plus a step-2 re-run) fills before first ship rather than
after a live QA pass finds it.*

## 6. Documentation

Update `FRONTEND_API_REFERENCE.md` and `COVERAGE_MATRIX.md` to reflect the
new country/language and its real hanketyyppi count.

*Catches: the stale-FRONTEND_API_REFERENCE.md shape directly — that doc
drifted from live code for months before anyone caught the mismatch between
its claimed and its real country/hanketyyppi counts. Making this a
mandatory onboarding step, not an occasional audit, is the fix.*

## 7. Live verification

- A real (not simulated) generation test in the new country and its
  language.
- `rag-check-all` if the country has RAG-ingested content.
- Re-run `python3 scripts/validate_country_coverage.py` one more time as the
  actual gate before calling the country "done."

*Catches: whatever steps 1–6 structurally can't. Some bugs only show up in
real generated output — the original Traficom leak was two entities
appearing *together* in one document, which a static coverage check alone
would never catch; it needed a real generated document to reveal the
interaction with another prompt directive. This step is the backstop for
emergent bugs, not a redundant re-check of steps 1–6.*

---

## Known open item

`_COUNTRY_LUVAT` and `_COUNTRY_LIITTEET` are checked by
`validate_country_coverage.py` at the "does this country exist as a
top-level key" level only. Whether every individual hanketyyppi is
*legitimately* offered to a given country (vs. genuinely missing) isn't
checked yet — that needs a real source of truth for which hanketyyppi each
country is meant to offer, which is still in flux pending the in-progress
frontend intake-form work. Do not build that finer allowlist against
today's live state; confirm the intended offered-set first (see
`permit_ai/country_registry.py`'s module docstring).
