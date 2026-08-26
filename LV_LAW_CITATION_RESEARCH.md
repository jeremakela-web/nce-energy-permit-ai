# Latvia — real statutory equivalents + _COUNTRY_LIITTEET build-out

Research findings for the backlog's priority item 1 (2026-08-26), first of
the EE/DE/LV pass. LV picked first via real gap-sizing, not assumption:

| | EE | DE | LV |
|---|---|---|---|
| `_LAW_TRANS` entries | 0 | 0 | 0 |
| `_LAW_CITATION_REPLACEMENT` entries | 0 | 0 | 0 |
| `_COUNTRY_LIITTEET` hanketyyppi coverage | 5/10 | 7/10 | **0/10** |
| `context_extra` branch markers | 5 | 7 | 4 |

Every statute verified via live web search against primary/near-primary
sources (likumi.lv — Latvia's official legal database, Latvijas Vēstnesis —
the official gazette, vvd.gov.lv, iem.gov.lv, lvportals.lv), not recalled.

Scope: matches the established "phase 1" pattern (law citations, correct
authorities, `_COUNTRY_LIITTEET`) — the deeper `context_extra` pass is a
separate, later follow-on, same sequencing SE got, not folded in here.

---

## 1. Verified statute mapping

| FI statute | LV equivalent | Verified detail |
|---|---|---|
| MRL 132/1999 | **Teritorijas attīstības plānošanas likums** (2011) | Confirmed — matches pre-existing content, good cross-check. Adopted 13.10.2011, in force 01.12.2011, Latvijas Vēstnesis Nr. 173. |
| Rakentamislaki 751/2023 | **Būvniecības likums** (2013) | Confirmed — matches pre-existing content. Published Latvijas Vēstnesis Nr. 146, 30.07.2013. |
| YVA-laki 252/2017 | **Likums "Par ietekmes uz vidi novērtējumu"** (1998) | Confirmed real, still in force and amended (2026 amendment filings found live). |
| YSL 527/2014 | **Likums "Par piesārņojumu"** (2001) | **Correction** — see section 2. |
| Vesilaki 587/2011 | **Ūdens apsaimniekošanas likums** (2002) | New — not in pre-existing content at all. Published Latvijas Vēstnesis Nr. 140, 01.10.2002. |
| Pelastuslaki 379/2011 | **Ugunsdrošības, ugunsdzēsības un glābšanas darbu likums** (in force since 13.11.2025) | **Correction** — see section 2. |
| Patoturvallisuuslaki 494/2009 | **Likums "Par hidroelektrostaciju hidrotehnisko būvju drošumu"** (2000/2001) | New — real, dedicated law, unlike PL/LT's "no separate act." Adopted 07.12.2000, in force 01.04.2001, still in force ("Spēkā esošs"). |
| Ydinenergialaki 990/1987 | **Likums "Par radiācijas drošību un kodoldrošību"** (2000, amended) | **Major correction** — see section 3. |
| Sähkömarkkinalaki 588/2013 | Elektroenerģijas tirgus likums (ETL, 2005) | Confirmed — matches pre-existing content. Enacted 05.05.2005, in force 08.06.2005. |

## 2. Two real corrections to pre-existing (wrong) content

**YSL 527/2014's equivalent was wrong.** Pre-existing `key_laws`/
`prompt_prefix` cited "Vides aizsardzības likums (1997)" as the environmental-
permit law. Verified: that law is actually from **2006** (Latvijas Vēstnesis
Nr. 183, 15.11.2006), not 1997 — and it's a general framework act, not the
permit-granting one. The real operative A/B/C-category pollution-permit law
(the actual YSL analog) is **Likums "Par piesārņojumu"** (2001): A-category
facilities listed in the law's own Annex 1; B/C categories in MK noteikumi
Nr. 1082's own annexes; permits issued by Valsts vides dienests (VVD).
Confirmed via vvd.gov.lv directly. Both laws are real and kept in
`key_laws`, but "Par piesārņojumu" is now correctly used as the primary
`_LAW_TRANS`/backstop mapping.

**Pelastuslaki 379/2011's equivalent had changed and pre-existing content
never had it.** The old "Ugunsdrošības un ugunsdzēsības likums" (2002) is
now repealed. The current law entered into force **13 November 2025**
(confirmed via lvportals.lv and iem.gov.lv) — "Ugunsdrošības, ugunsdzēsības
un glābšanas darbu likums." Would have been a stale citation if I'd stopped
at the first search result showing the old law's name.

## 3. Nuclear framework — a real, active bug found, not just missing content

Both the pre-existing `prompt_prefix` ("Latvia has NO nuclear power plants
and no nuclear regulatory framework") and `_STUK_REPLACEMENT["LV"]` (a
`[Vaatii tarkistuksen]` hedge, with an inline comment asserting the same
"no framework exists" claim) were **factually wrong**, not just
conservative placeholders.

Real findings:
- A real law: **Likums "Par radiācijas drošību un kodoldrošību"** (2000,
  amended — confirmed via likumi.lv id 12484, still in force; 2024
  amendments found live via EUR-Lex).
- A real regulator: the **Radiācijas drošības centrs** (Radiation Safety
  Centre), under Valsts vides dienests (VVD) — oversees licensing,
  operation, and decommissioning of nuclear/radioactive facilities
  (confirmed via ENSREG's country profile).
- Genuine current commercial/governmental interest: Latvia is participating
  in the real **U.S.-Latvia FIRST project** (Foundational Infrastructure
  for Responsible Use of Small Modular Reactor Technology, a real U.S. DOE
  program), with officials on record stating full SMR readiness could take
  ~15 years given the regulatory-framework development still needed.

**Same shape as Norway's case from the prior sprint** — real law + real
regulator + genuine current interest, but no demonstrated first-of-kind
commercial licensing process — explicitly NOT copied from NO's or any other
country's template; built from LV's own real facts throughout (own law
name, own regulator, own program name).

## 4. `_COUNTRY_LIITTEET` — built from 0/10

Same energy-first sequencing as every prior country's first pass: BESS,
tuulivoima_maa, tuulivoima_meri (+ offshore_wind alias), aurinkovoima,
vesivoima, hybridi, ymparistolupa, datakeskus. **SMR/smr_bess included from
the start** (unlike LT's initial pass, which deferred them for careful
nuclear-caveat treatment) — LV's nuclear situation is unambiguous once
researched (NO-style, not the LT-era ambiguity that justified deferring it
there). `smr_lv` aliased to the new `SMR` entry, matching the SE/DA/NO/DE
pattern; the old comment explaining LV's prior exclusion from that alias
loop (repeating the same wrong "no nuclear framework" claim) corrected.

`tuulivoima_meri` includes an honest `[Vaatii tarkistuksen]` hedge on the
exact offshore-wind permitting procedure — not independently verified this
pass, flagged rather than guessed.

## 5. `kaava_SMR`/`kaava_aurinkovoima` — checked, one real fix applied, one gap surfaced

`_PDF_STRINGS` has no `"LV"` entry at all (only FI/EN/SE/DA/NO/PL/DE/LT) —
LV reports fall back to the EN card text for every static UI string,
including `kaava_SMR`/`kaava_aurinkovoima`, via `_s()`'s existing EE/ET/LV→EN
fallback. This is a real, separate, larger gap (the same shape LT needed a
dedicated fix for, historically) — **not fixed here, flagged as a
follow-on**, since building a full `_PDF_STRINGS["LV"]` entry is a
different-shaped task from law-citation/liite research.

What IS fixed: the `kaava_SMR` render call site already passes `country`
through to `_s()` (confirmed in code), so the `_fix_hardcoded_stuk` backstop
already applies to LV's inherited EN card — the `_STUK_REPLACEMENT["LV"]`
fix above means LV's kaava_SMR card now correctly shows the real regulator
instead of "STUK", verified directly.

What's NOT fixed (residual, smaller gap): the EN card's *substance* ("the
Council of State's decision-in-principle... STUK's pre-licensing procedure")
describes Finland's own institutional process, which isn't caught by the
existing backstops (they match literal Finnish statute names, not
already-translated English prose) — a genuinely LV-specific kaava_SMR card
would need the same `_PDF_STRINGS["LV"]` work flagged above.

`kaava_aurinkovoima`'s EN card is short and generic (no fabricated
"Finland's 2025 reform as fact" narrative, unlike SE/DA/NO's original bug) —
lower residual risk than kaava_SMR here.

---

**Build complete, all 6 steps** (matching the SE/PL/LT/DA/NO shape):
1. `_LAW_TRANS["LV"]` — 14 entries (13 matching LT's own scope + Patoturvallisuuslaki, which LT/PL lack but LV has a real citation for)
2. `_LAW_CITATION_REPLACEMENT` — 8 entries (same 8 statute families)
3. `_STUK_REPLACEMENT["LV"]` — fixed from a wrong hedge to the real regulator
4. `_COUNTRY_CONFIG["LV"]["prompt_prefix"]` — corrected + explicit FI→LV statute mapping added
5. `kaava_SMR`/`kaava_aurinkovoima` — checked; STUK-name fix confirmed propagating via the existing backstop; `_PDF_STRINGS["LV"]` gap flagged as a separate follow-on
6. `_COUNTRY_LIITTEET["LV"]` — built from 0/10 to 12/12 (10 base hanketyyppis + offshore_wind + smr_lv aliases)

Verified: `python3 -c "import ast; ast.parse(...)"`, real dict-count checks
for every new entry, a line-by-line diff-additivity check confirming all 56
modified `_LAW_TRANS`/`_LAW_CITATION_REPLACEMENT` lines are purely additive
(every removed line has a matching added line that's identical plus the new
LV key — zero regression to SE/PL/DA/NO/LT/EE/DE), and direct value
spot-checks confirming other countries' real content is untouched.
