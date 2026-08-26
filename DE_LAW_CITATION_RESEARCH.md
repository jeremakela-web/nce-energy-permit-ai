# Germany — real statutory equivalents + `_COUNTRY_LIITTEET` build-out

Research findings for the backlog's priority item 1, third and final leg of the
EE/DE/LV pass (LV: PR #122, EE: PR #123). DE started from a genuinely
different baseline than EE/LV: `_COUNTRY_LUVAT["DE"]` was **already
comprehensive** (16 hanketyyppi entries, including `vesivoima` and `egs`
geothermal, which EE/LV both lacked entirely) and largely accurate — Germany's
federal structure (BauGB/Landesbauordnung, BImSchG, UVPG, EnWG, WHG, BNatSchG)
was already correctly mapped. So this pass is narrower and deeper rather than
broad: verify what's there, and specifically chase the two things that were
either missing or actively wrong.

Gap state before this pass (re-verified directly against the file, not
assumed):

| | value |
|---|---|
| `_LAW_TRANS` DE entries | 0 |
| `_LAW_CITATION_REPLACEMENT` DE entries | 0 |
| `_STUK_REPLACEMENT["DE"]` | absent |
| `_COUNTRY_LIITTEET["DE"]` hanketyyppi coverage | 7 keys: `tuulivoima_maa`, `tuulivoima_meri`, `BESS`, `aurinkovoima`, `SMR`, `smr_bess`, `datakeskus` — missing `vesivoima`, `hybridi`, `ymparistolupa` (same target list as LV/EE) |
| `_COUNTRY_LUVAT["DE"]` | already 16/~21 hanketyyppi, including `vesivoima` and `egs` — this pass corrects content within it, doesn't build it from scratch |

Every finding below verified via live web search against primary/near-primary
German sources (Wikipedia DE cross-checked against BMUKN's own site,
base.bund.de, Bundestag.de, planradar.de/baunetzwissen.de for the
Landesbauordnung question), each cross-checked across 2+ independent hits.

---

## 1. The two real findings

**Nuclear — a live, active bug, same severity class as Denmark's outright ban,
found in the existing content, not just missing.**

Germany's nuclear phase-out (Atomausstieg) was completed **15 April 2023** —
the last three plants shut down. The amended Atomgesetz (AtG) now **explicitly
bans new commercial nuclear power plants for electricity generation** ("die
gewerbliche Erzeugung von Elektrizität durch Kernspaltung" is prohibited).
There is no licensing pathway for a new commercial power-generating reactor —
same structural shape as Denmark's statutory ban (`_LAW_CITATION_REPLACEMENT`'s
existing DA entry), not a normal AtG §7 licensing process.

The existing `_COUNTRY_LUVAT["DE"]["SMR"]`/`["smr_de"]`/`["smr_bess"]` rows
and `_COUNTRY_LIITTEET["DE"]["SMR"]`/`["smr_bess"]` currently present this as
a real, if difficult, permit pathway ("Genehmigungsantrag nach Atomgesetz
(AtG § 7)") with only a vague "⚠️ Bundestagsbeschluss ggf. erforderlich"
hedge — not the actual, stronger fact that new commercial power-generating
plants are statutorily prohibited outright. **Fixed here** — same
"state the real severity, don't soften it" discipline as DA's ban and LV's
prior wrong "no framework at all" claim (opposite direction, same principle:
say exactly what's true).

One real nuance, kept in the correction rather than dropped: fission
installations for non-electricity purposes (process heat, hydrogen,
industrial steam) are **not** explicitly banned by the current AtG text and
would in principle be licensable if atomic-law requirements are met — but no
such facility has ever been licensed in Germany, so this is a theoretical
opening, not a demonstrated pathway. State both facts together, same
discipline as LT's VATESI/Ignalina framing (real regulator, but current remit
doesn't cover what's being asked).

Also live in 2026: active political debate (AfD pushing for a legislative
reversal to re-enter nuclear power) — real context, explicitly **not** a law
change; the ban is current law as of today.

**Ministry rename — BMUV → BMUKN, confirmed current, code was citing the
pre-2025 name.**

The federal ministry these rows attribute nuclear oversight to was cited as
**BMUV** (Bundesministerium für Umwelt, Naturschutz, nukleare Sicherheit und
Verbraucherschutz). Under the Merz cabinet (minister Carsten Schneider, SPD,
since 6 May 2025), the ministry was reorganised: consumer-protection
responsibilities moved to BMJV, and the remaining portfolio was renamed
**BMUKN** (Bundesministerium für Umwelt, Klimaschutz, Naturschutz und nukleare
Sicherheit) — confirmed via the ministry's own site (bundesumweltministerium.de,
now branded BMUKN) and German Wikipedia. "BMUV" has been stale for over a
year as of today (2026-08-26). Corrected everywhere it appeared.

The same 06.05.2025 reorganisation also renamed the economy ministry: **BMWi**
(cited in `_NATIONAL_SUPERVISORS["DE"]["SMR"]`) had already become BMWK in
2021 and was renamed again to **BMWE** (Bundesministerium für Wirtschaft und
Energie, minister Katherina Reiche) on the same date as the BMUV→BMUKN
change — confirmed via the same search pass, fixed in the same edit.

The real federal nuclear-oversight body for what remains active (repository
licensing/supervision, not new-build) is **BASE** (Bundesamt für die
Sicherheit der nuklearen Entsorgung — Federal Office for the Safety of
Nuclear Waste Management), a real Bundesoberbehörde under BMUKN, confirmed
via base.bund.de. Its actual current remit: Atomaufsicht des Bundes over the
Morsleben and Konrad repositories and the Asse II shaft — decommissioning/
waste-management supervision, not new commercial reactor licensing (there is
none to license). Same "real regulator, current remit doesn't cover new-build"
shape as LT's VATESI.

## 2. Confirmed, no correction needed (existing content was already right)

- **Fire safety**: confirmed no separate federal Brandschutzgesetz exists —
  Landesbauordnungen (per-Bundesland, based on the Musterbauordnung/MBO
  template) are the real legal basis, varying by Land. Matches the existing
  `_COUNTRY_LUVAT`/`_COUNTRY_LIITTEET` content's implicit treatment
  ("Bauordnungsrecht der Länder" / "Brandschutzkonzept... Bauordnungsrecht").
  Same "no separate act, folded into general building law" shape as PL/LT's
  Patoturvallisuuslaki finding, just for fire safety instead of dams. No fix
  needed to the underlying law — but `_COUNTRY_LUVAT["DE"]["BESS"]` had no
  explicit fire-safety line item at all (the liite checklist did, the permit
  table didn't) — added one for consistency, see build section below.
- **Dam safety**: confirmed no separate federal law either — governed by
  Wasserhaushaltsgesetz (WHG, the general water law, already correctly cited
  throughout) plus the **DIN 19700** technical-standard series (a real, named,
  specific standard — "Stauanlagen," Part 11 covers Talsperren specifically)
  plus Land-level Landeswassergesetze for size/class-based supervision rules.
  Different shape from PL/LT (no law *or* named standard) and from
  NO/LV (a dedicated law) — DE sits in between: no dedicated law, but a real
  named technical standard worth citing explicitly rather than just "WHG."
- MRL→BauGB, Rakentamislaki→Landesbauordnung, YVA-laki→UVPG, YSL→BImSchG,
  Sähkömarkkinalaki→EnWG: all already correctly used throughout the existing
  `_COUNTRY_LUVAT["DE"]` content — verified consistent with the sources above,
  no corrections needed, just wired into `_LAW_TRANS`/`_LAW_CITATION_REPLACEMENT`
  for the first time (both were previously empty for DE).

## 3. Scope boundary

Patoturvallisuuslaki gets a real answer this pass (WHG + DIN 19700, see
above) — unlike EE, where it was left unresearched. Not independently
re-verified this pass: Maa-aineslaki equivalent, and the `_PDF_STRINGS["DE"]`/
`kaava_*` card-text audit (the check LV's report did in its own §5) — same
scope boundary as EE's report, flagged rather than silently assumed done.

---

**Build shape**: same 6 steps as LV/EE — `_LAW_TRANS` (13 entries),
`_LAW_CITATION_REPLACEMENT` (8 entries, including Patoturvallisuuslaki this
time), `_STUK_REPLACEMENT["DE"]` (BASE, with the remit caveat), `key_laws`/
`prompt_prefix` corrections (BMUV→BMUKN, explicit nuclear-ban framing),
`_COUNTRY_LUVAT["DE"]` fixes (BMUV→BMUKN in SMR/smr_de/smr_bess, explicit ban
language, add missing fire-safety line to BESS), and `_COUNTRY_LIITTEET["DE"]`
build-out (7/10 → 10/10: add `vesivoima`, `hybridi`, `ymparistolupa`).
