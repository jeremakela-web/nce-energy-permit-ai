# Lithuania — real statutory equivalents + _COUNTRY_LIITTEET expansion

Research findings for item 5(B) + item 4 remainder, Lithuania (third in the
pilot-activity-ordered combined queue, after SE and PL). Every statute
verified via live web search against primary/near-primary sources (VATESI,
PAGD, e-seimas/LRS — Lithuania's official legal database), not recalled.
Investigation/research only — same cadence as SE/PL: report first, build on
go-ahead.

---

## 0. LT's gap is structurally different from SE/PL — bigger, not just wrong

SE and PL both had `_LAW_TRANS` entries with wrong/fabricated content.
**`_LAW_TRANS` has no "LT" column at all** — confirmed directly (`_LAW_TRANS`
langs are `['DA', 'EN', 'NO', 'PL', 'SE']`, no LT/DE/ET/LV). This means
`_t_law()` currently returns the **raw, untranslated Finnish citation**
unchanged for every LT report — a more direct, visible leak into the
always-present "Lakiviitteet" section than SE/PL had (which at least showed
*something*, just wrong).

## 1. A good surprise: `_COUNTRY_CONFIG["LT"]["prompt_prefix"]` is already excellent

Unlike SE/PL's original prefixes (which needed real correction), LT's
existing prefix is already extensive and well-researched — real authorities
(VERT, LITGRID, ESO, AAA, PAGD, ANCO, Lietuvos kariuomenė), real thresholds
(PAV mandatory for wind >5 turbines or >30 MW; solar >50 MW), real
timelines (statybos leidimas 20–40 working days), and an already-correct,
honest note: *"Lithuania has NO operating nuclear power plants (Ignalina
NPP shut down 2009) — SMR projects would require entirely new primary
legislation before any permit path exists."* It only needs the same kind of
enhancement SE/PL got: an explicit per-statute FI→LT mapping sentence (the
real laws are already listed, just not mapped 1:1 to the Finnish citations
that need replacing).

## 2. Verified statute mapping

| FI statute | Real LT equivalent | Verified detail |
|---|---|---|
| MRL 132/1999 (zoning) | **Teritorijų planavimo įstatymas** | Already correctly named in the existing prompt_prefix's `key_laws` list — confirms that entry was accurate. |
| Rakentamislaki 751/2023 (building) | **Statybos įstatymas** | Same — already correctly listed. Lithuania keeps zoning and building permits as **two separate acts**, same as Poland — confirmed by the prompt_prefix listing both as distinct key laws, not merged the way Sweden's PBL is. |
| YVA-laki 252/2017 (EIA) | **PAV įstatymas** (Planuojamos ūkinės veiklos poveikio aplinkai vertinimo įstatymas) | Already correctly named and already used as the bracket-citation example in `_WRITE_INSTRUCTION["LT"]` (PR #105) — good consistency check. |
| YSL 527/2014 (environmental permit) | **Aplinkos apsaugos įstatymas** | Already correctly listed (taršos leidimas/TIPK). |
| Vesilaki 587/2011 (water) | **Vandens įstatymas** | Confirmed real via e-seimas/LRS references (amendment Law No. IX-1388 among others). Not previously listed in the prompt_prefix's `key_laws` — a real, small gap in the otherwise-thorough existing entry. |
| Pelastuslaki 379/2011 (fire/rescue) | **Priešgaisrinės saugos įstatymas** | Confirmed real (full text found via geslita.lt/pakruojospt.lt mirrors of the official act). Note: search results indicate the law has been redrafted with an updated name ("Fire Safety and Rescue Operations Law" in English coverage) — I could not confirm the exact current Lithuanian title with full confidence, so using the long-established, still-valid "Priešgaisrinės saugos įstatymas" rather than guessing at an unverified exact new title. |
| Ydinenergialaki 990/1987 (nuclear) | **Branduolinės energijos įstatymas** (1996, as amended) — WITH the existing caveat kept | Real law confirmed (Law No. I-1613 of 1996 on nuclear energy, amended by Law No. XIII-287), and VATESI is a real, active regulator under it — this is why `_STUK_REPLACEMENT["LT"]` (PR #103) correctly names VATESI. But VATESI's actual current remit is Ignalina NPP safety, decommissioning, and radioactive waste/material control — **not** a demonstrated new-build licensing pathway. This *confirms* rather than contradicts the existing prompt_prefix's "no permit path exists" note — a real law and a real regulator exist, but not for the "build a new SMR" case specifically. Recommend keeping both facts together (real law + real regulator + the existing "would need new legislation for new-build" caveat), not simplifying to one or the other. |
| Patoturvallisuuslaki 494/2009 (dam safety) | **No separate law.** Hydrotechnical structures (hidrotechniniai statiniai) regulated under **Statybos įstatymas** + technical construction regulations (STR 2.02.06:2004, STR 2.05.14:2005) | Same "folded into general framework, not a standalone act" pattern as Poland — confirmed, not assumed. `Patoturvallisuuslaki` should get the same "no entry" treatment in the backstop as PL got. |

## 3. Self-correction: my own PR #105 translations need the same fix

Building `_PDF_STRINGS["LT"]` fresh in PR #105, I already instinctively
glossed the FI citations with real Lithuanian law names in `kaava_tuuli`
("Žemės naudojimo ir statybos įstatymas") and `kaava_aurinkovoima`
("Statybos įstatymą") — but kept Finland's own numbers attached (132/1999,
751/2023), same bug class as SE/PL's original `_LAW_TRANS` problem, just
introduced fresh by me rather than inherited.

Also, **"Žemės naudojimo ir statybos įstatymas"** (a combined term I
invented, mirroring the English "Land Use and Building Act" literally) turns
out to be wrong for Lithuania specifically — per the verified mapping above,
Lithuania splits this into two separate acts (Teritorijų planavimo įstatymas
+ Statybos įstatymas), matching Poland's split, not Sweden's merge. Needs
correcting to use the right one depending on which Finnish statute (MRL vs.
Rakentamislaki) is actually being cited in each instance.

`kaava_SMR` still says "STUK" three times (untranslated — I hadn't done LT
nuclear research at the time of PR #105) and cites "Branduolinės energijos
įstatymas 990/1987, § 11" (right name, wrong number, same pattern). Needs
the same STUK→VATESI + citation fix SE/PL got, plus the nuclear-caveat
nuance from section 2 above (VATESI's real remit vs. no new-build pathway).

`kaava_aurinkovoima` needs checking against the SE/DA/NO "describes
Finland's reform as fact" narrative bug before assuming it's citation-only
— to be confirmed when building (same as I did for PL, where it turned out
not to have that bug).

## 4. `_COUNTRY_LIITTEET` — LT at 0/21, no baseline anywhere

Confirmed: LT has zero coverage for any hanketyyppi, so every liite list
currently falls through entirely to the raw Finnish base list — a bigger
gap than SE/PL's partial coverage. Same energy-first sequencing proposed as
SE/PL, pending go-ahead: BESS, tuulivoima_maa, tuulivoima_meri/offshore_wind
(aliased), aurinkovoima, vesivoima, ymparistolupa. **`hybridi` and `SMR`/
`smr_bess` deliberately excluded from this pass** — SMR/smr_bess need the
nuclear-caveat treatment worked out carefully (real law + real regulator +
honest "no new-build pathway" hedge, not a simple citation swap), better
handled as its own considered piece rather than folded into the general
energy-project batch.

---

**Ready to build once confirmed** — same shape as SE/PL: add the `_LAW_TRANS`
LT column for the 7 statutes with real single-law equivalents (all except
Patoturvallisuuslaki), add an `"LT"` entry to `_LAW_CITATION_REPLACEMENT`,
enhance `_COUNTRY_CONFIG["LT"]["prompt_prefix"]` with the explicit mapping
(adding Vandens įstatymas, which was missing even from the otherwise-good
existing entry), fix `kaava_tuuli`/`kaava_aurinkovoima`'s citations (my own
PR #105 work), handle `kaava_SMR`/nuclear separately and carefully per
section 3, and expand `_COUNTRY_LIITTEET["LT"]` for the energy-project types
(excluding SMR/smr_bess/hybridi, per section 4). Report back with the diff
before merging, as always.
