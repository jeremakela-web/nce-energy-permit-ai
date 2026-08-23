# Poland — real statutory equivalents + _COUNTRY_LIITTEET expansion

Research findings for item 5(B) + item 4 remainder, Poland (second in the
pilot-activity-ordered combined queue, after SE). Every statute verified via
live web search against primary sources (Sejm/ISAP — Poland's official legal
database, GDOŚ, PAA, gov.pl), not recalled. Investigation/research only —
nothing built into the codebase yet, per standing practice (same cadence as
SE: report findings, then build once directed).

---

## 1. Verified statute mapping

| FI statute (currently cited) | Real PL equivalent | Verified detail |
|---|---|---|
| MRL 132/1999 (land use/zoning) | **Ustawa o planowaniu i zagospodarowaniu przestrzennym** (Dz.U. 2003 nr 80 poz. 717) | Unlike Sweden, Poland genuinely DOES split zoning/spatial-planning from building permits into two separate acts — same shape as Finland's system *before* the 2023 Rakentamislaki reform. So this is a **two-law mapping for PL**, not the one-law collapse used for SE. [Sejm/ISAP](https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20030800717), [eli.gov.pl](https://eli.gov.pl/eli/DU/2003/717/ogl) |
| Rakentamislaki 751/2023 (building permits) | **Prawo budowlane** (Dz.U. 1994 nr 89 poz. 414) | Separate from the zoning act above — confirms the two-law split. [Sejm/ISAP](https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=wdu19940890414) |
| YVA-laki 252/2017 (EIA) | **Ustawa OOŚ** (Ustawa z 3.10.2008 r. o udostępnianiu informacji o środowisku..., Dz.U. 2008 nr 199 poz. 1227) | Already correctly used as a short name elsewhere in the existing PL `_LAW_TRANS`/`_LIITE_TRANS` tables — this confirms that existing usage was accurate, just missing the real Dz.U. number. Administered by GDOŚ (national) / RDOŚ (regional). [Sejm/ISAP](https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=wdu20081991227), [GDOŚ](http://www.gdos.gov.pl/system-oos) |
| YSL 527/2014 (environmental permit) | **Prawo ochrony środowiska** "POŚ" (Dz.U. 2001 nr 62 poz. 627) | Covers integrated permits (pozwolenie zintegrowane) — for larger installations this REPLACES separate sectoral permits including, in some cases, water permits. [eli.gov.pl](https://eli.gov.pl/eli/DU/2001/627/ogl) |
| Vesilaki 587/2011 (water act) | **Prawo wodne** (Dz.U. 2017 poz. 1566, ustawa z 20.07.2017) | Administered by Wody Polskie (Państwowe Gospodarstwo Wodne Wody Polskie). [eli.gov.pl](https://eli.gov.pl/eli/DU/2017/1566/ogl) |
| Pelastuslaki 379/2011 (rescue services) | **Ustawa o ochronie przeciwpożarowej** (Dz.U. 1991 nr 81 poz. 351, ustawa z 24.08.1991) | Administered by Państwowa Straż Pożarna (PSP). [Sejm/ISAP](https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU19910810351) |
| Ydinenergialaki 990/1987 (nuclear) | **Prawo atomowe** (ustawa z 29.11.2000, Dz.U. 2001 nr 3 poz. 18) | Administered by PAA (Państwowa Agencja Atomistyki) — already correctly used in the existing PL `SMR` liite entry, confirming that entry is accurate. [eli.gov.pl](https://eli.gov.pl/eli/DU/2001/18/ogl) |
| Patoturvallisuuslaki 494/2009 (dam safety) | **No separate act.** Folded into Prawo wodne itself | Poland's "State Service for Dam Safety" (Państwowa służba ds. bezpieczeństwa budowli piętrzących) is performed by IMGW-PIB (Instytut Meteorologii i Gospodarki Wodnej) via Centrum Technicznej Kontroli Zapór, under Prawo wodne — there is no standalone dam-safety law to cite the way Sweden has Förordning 2014:214. Any PL dam-safety liite entry should cite Prawo wodne + name IMGW-PIB/CTKZ, not invent a separate act. [Inżynier Budownictwa](https://inzynierbudownictwa.pl/bezpieczenstwo-budowli-hydrotechnicznych-pietrzacych-wode/) |

## 2. Authority mapping

| Authority | Role |
|---|---|
| **RDOŚ** (Regionalna Dyrekcja Ochrony Środowiska) / **GDOŚ** (national) | Environmental permits, EIA (OOŚ) — already correctly used elsewhere in this codebase as the PL replacement for ELY-keskus (PR #103) |
| **Starostwo / gmina** (local administration) | Prawo budowlane building permits (pozwolenie na budowę) |
| **Wody Polskie** | Water-law permits (Prawo wodne) |
| **PAA** (Państwowa Agencja Atomistyki) | Nuclear regulator (Prawo atomowe) — already correctly used in the existing SMR liite entry |
| **IMGW-PIB** / Centrum Technicznej Kontroli Zapór | Dam safety technical oversight (under Prawo wodne, not a separate law) |
| **PSP** (Państwowa Straż Pożarna) | Fire safety (Ustawa o ochronie przeciwpożarowej) |
| **PSE S.A.** (Polskie Sieci Elektroenergetyczne) | Transmission grid connection — already correctly used in the existing SMR/datakeskus liite entries |

Good cross-check, same as with SE: the existing PL `SMR`/`smr_bess`/`datakeskus` liite entries already correctly use PAA, RDOŚ/GDOŚ, and PSE S.A. — confirms those entries were accurate, just (like SE's) missing the real Dz.U. reference numbers on the law citations themselves, and one uses a vague placeholder ("Ustawa środowiskowa" instead of the precise "Ustawa OOŚ" already used consistently elsewhere).

## 3. `_LAW_TRANS`'s PL column has the exact same bug pattern as SE had

Confirmed via direct query — same two sub-patterns as SE:

- **Real law name, wrong (Finnish) number attached**: `"Ustawa o zagospodarowaniu przestrzennym (132/1999)"`, `"Prawo budowlane 751/2023"`, `"Ustawa OOŚ (252/2017)"`, `"Prawo wodne (587/2011)"`, `"Ustawa o ochronie przeciwpożarowej 379/2011, § 15"` — all five of these are **genuinely real Polish law names** (matches my research exactly) but every one has Finland's own statute number glued on instead of the real Dz.U. reference.
- **Fabricated name entirely**: `"Ustawa o ochronie środowiska (YSL 527/2014)"` (real name is **Prawo ochrony środowiska**, a different, specific title — "ochrona środowiska" vs "ochrony środowiska" is not just a typo, it's citing a generic descriptive phrase rather than the actual act title), `"Ustawa o energii jądrowej (990/1987)"` (real name is **Prawo atomowe**), `"Ustawa o bezpieczeństwie budowli piętrzących (494/2009)"` (not a real standalone act at all — per the research above, dam safety has no separate PL law).

Same fix shape as SE: correct all affected `_LAW_TRANS["PL"]` values to the real citation (dropping the Finnish paragraph/section suffix, same reasoning as SE — no verified section-level correspondence), extend `_fix_hardcoded_law_citations()`'s `_LAW_CITATION_REPLACEMENT` dict with a `"PL"` entry per statute, and check `_COUNTRY_CONFIG["PL"]["prompt_prefix"]` for the same "real laws listed but no explicit per-statute mapping" gap already found and fixed for SE.

## 4. Task #31 items folded in: PL's `kaava_SMR` / `kaava_aurinkovoima`

Checked directly (not deferred):

- **`kaava_SMR`** has the full bug pattern: "STUK" named twice as the reviewing
  authority (should be **PAA**), the citation "ustawa o energii jądrowej
  990/1987, § 11" (fabricated name + FI number — should be **Prawo atomowe,
  Dz.U. 2001 nr 3 poz. 18**), and "decyzja zasadnicza Rady Stanu" ("Council
  of State's decision-in-principle") — Finland-specific governmental
  terminology; Poland's actual decision-making body for this would need its
  own check (likely Rada Ministrów or a ministerial-level decision, not
  "Rada Stanu," which isn't the correct Polish state body for this).
- **`kaava_aurinkovoima`** does NOT have SE/DA/NO's "describes Finland's 2025
  reform as fact" narrative bug — the underlying Polish administrative
  concept it describes (zgłoszenie robót budowlanych instead of a full
  pozwolenie na budowę for small installations) is real and sensible. Only
  the citation itself needs fixing: `"Prawo budowlane 751/2023 / 132/1999,
  § 126"` → **Prawo budowlane (Dz.U. 1994 nr 89 poz. 414)**. Simpler fix
  than SE's version — citation swap only, no narrative rewrite needed.

## 5. `_COUNTRY_LIITTEET` — PL currently at 3/21 (SMR, smr_bess, datakeskus)

Same energy-first sequencing as SE, pending your go-ahead: BESS, tuulivoima_maa, tuulivoima_meri/offshore_wind (aliased), aurinkovoima, vesivoima, hybridi, ymparistolupa. Not drafted yet in this note — given the SE cadence (research reported first, drafts + actual build happen together once you confirm), I'll draft these alongside the `_LAW_TRANS` fix and backstop entry once you confirm the direction, rather than duplicating the drafting effort across two messages.

---

**Ready to build once confirmed** — same shape as the SE PR: fix `_LAW_TRANS`'s PL column at the source, add a `"PL"` entry to `_LAW_CITATION_REPLACEMENT`, enhance `_COUNTRY_CONFIG["PL"]["prompt_prefix"]` with the explicit mapping, fix `kaava_SMR`'s STUK→PAA (and `kaava_aurinkovoima` if it has the same reform-narrative bug), and expand `_COUNTRY_LIITTEET["PL"]` for the energy-project hanketyyppis. Report back with the diff before merging, as always.
