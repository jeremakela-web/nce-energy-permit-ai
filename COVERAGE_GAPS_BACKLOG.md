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

## Suggested next step

Once this backlog is picked up: run both validators for real first
(`python scripts/validate_country_coverage.py`, `python scripts/validate_frontend_coverage.py`)
to reconfirm nothing above has drifted, then work through in whatever order
makes sense — the `_LAW_TRANS`/`_LAW_CITATION_REPLACEMENT` table (section 1)
is the largest single item and could reasonably follow the same
per-country-pass rhythm as the rest of this sprint's law-citation work.
