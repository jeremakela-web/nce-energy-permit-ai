# Sweden — real statutory equivalents + _COUNTRY_LIITTEET expansion

Research findings for item 5(B) + item 4's remainder, Sweden first (per the
agreed pilot-activity order). Every statute below was verified via live web
search against primary/near-primary sources (Riksdagen, Boverket, MSB,
Naturvårdsverket, Svenska kraftnät), not recalled from training data alone —
sources listed per item. Investigation/research only — nothing built into
the codebase yet, per standing practice.

---

## 1. Verified statute mapping (the 8 distinct FI citations in scope)

The FI statutes bare-cited across the ~13 SE-relevant hanketyyppis (from the
earlier scoping sweep), and their real Swedish equivalents:

| FI statute (currently cited) | Real SE equivalent | Verified detail |
|---|---|---|
| MRL 132/1999 (land use/zoning) | **Plan- och bygglagen (2010:900)** "PBL" | Covers land-use planning (översiktsplan, detaljplan) — Sweden doesn't split this from building law the way Finland's 2023 reform did. [Riksdagen](https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/plan-och-bygglag-2010900_sfs-2010-900/), [Boverket](https://www.boverket.se/contentassets/d6136e8e4ff143728ce52bdb20b6148f/plan--och-bygglag-2010-900-med-planprocessen-mellan-2011-2014.pdf) |
| Rakentamislaki 751/2023 (building permits) | **Plan- och bygglagen (2010:900)**, same act | Building permit (bygglov) provisions are in the *same* PBL as zoning — no separate Swedish "building act." Note as of 1 Dec 2025 a major PBL amendment took effect, exempting more measures from bygglov requirements. [Tihinen law firm](https://www.tihinen.se/post/new-building-permit-regulations-rules-sweden-2025), [climate-laws.org](https://climate-laws.org/document/the-building-and-planning-act-sfs-2010-900_7349) |
| YVA-laki 252/2017 (EIA) | **Miljöbalken (1998:808), kap. 6** (miljöbedömningar) | Confirmed: ch. 6 specifically governs environmental assessment; required for activities needing a permit under ch. 9 or 11 that may cause significant environmental impact. Matches exactly what the existing SE `SMR` liite entry already uses ("Miljökonsekvensbeskrivning (MKB) — Miljöbalken kap. 6") — cross-verified against real code, not just search. [Riksdagen](https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/miljobalk-1998808_sfs-1998-808/), [Naturvårdsverket](https://www.naturvardsverket.se/vagledning-och-stod/miljobalken/miljobedomningar/specifik-miljobedomning/) |
| YSL 527/2014 (environmental permit) | **Miljöbalken (1998:808), kap. 9** (miljöfarlig verksamhet) | Environmentally hazardous activities — permit (tillstånd) or notification (anmälan) depending on scale, reviewed by Länsstyrelsen's Miljöprövningsdelegation (MPD) for larger cases, kommun for smaller. [Riksdagen](https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/miljobalk-1998808_sfs-1998-808/) |
| Vesilaki 587/2011 (water act) | **Miljöbalken (1998:808), kap. 11** (vattenverksamhet) | Water operations chapter — confirmed via the same source as ch. 6/9 above. |
| Pelastuslaki 379/2011 (rescue services) | **Lag (2003:778) om skydd mot olyckor** "LSO" | Protection-against-accidents act; rescue services (räddningstjänst) framework, administered with MSB. [Riksdagen](https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/lag-2003778-om-skydd-mot-olyckor_sfs-2003-778/), [IFRC](https://disasterlaw.ifrc.org/media/3434?language_content_entity=sv) |
| Ydinenergialaki 990/1987 (nuclear) | **Lag (1984:3) om kärnteknisk verksamhet** "Kärntekniklagen" | Already correctly used in the existing SE `SMR`/`smr_bess` liite entries ("Kärntekniklag SFS 1984:3") — this row confirms that existing entry is accurate, not a new find. SSM (Strålsäkerhetsmyndigheten) is the reviewing authority. [Riksdagen](https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/lag-19843-om-karnteknisk-verksamhet_sfs-1984-3/), [Kärnavfallsrådet](https://www.karnavfallsradet.mkg.se/contentassets/54ebbfc0517a431b88ee759a32e43279/the-act-on-nuclear-activities-19843.pdf) |
| Patoturvallisuuslaki 494/2009 (dam safety) | **Förordning (2014:214) om dammsäkerhet** + Miljöbalken kap. 11 | Dam safety ordinance, county administrative boards (Länsstyrelsen) supervise; dams themselves are also "vattenverksamhet" under MB ch. 11 requiring tillstånd. [Riksdagen](https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/forordning-2014214-om-dammsakerhet_sfs-2014-214/), [Svenska kraftnät](https://www.svk.se/49e304/siteassets/english/dam-safety/dam-safety---legislation-guidance-and-supporting-documents.pdf) |

**Bonus find, relevant to BESS specifically** (not a direct FI-citation swap,
but fills a real gap): **Lag (2010:1011) om brandfarliga och explosiva
varor** "LBE" — governs fire/explosion safety for handling flammable/
explosive goods, permit required for anyone handling such goods "in large
quantities," administered by MSB. This is the right citation for BESS fire
safety (the existing SE `SMR`/`smr_bess` liite list has no equivalent — worth
adding when I get to those). [Riksdagen](https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/lag-20101011-om-brandfarliga-och-explosiva_sfs-2010-1011/), [MSB](https://www.msb.se/sv/amnesomraden/skydd-mot-olyckor-och-farliga-amnen/brandfarligt-och-explosivt/regler-for-brandfarliga-och-explosiva-varor/)

**Also verified, wind-specific**: **Miljöbalken 16 kap. 4 §** — the
"kommunalt veto": a wind power permit cannot be granted unless the host
municipality has approved it, and the municipality's decision needs no
justification and cannot be appealed. This is a real, currently very
consequential rule (municipalities rejected ~75% of onshore wind projects in
H1 2024) — worth surfacing explicitly in `tuulivoima_maa`/`offshore_wind`'s SE
liite list or context, since it materially affects approval likelihood in a
way the current FI-only content has no equivalent for. [Svensk Vindenergi](https://svenskvindenergi.org/wp-content/uploads/2025/04/Kommunala-vetot-landbaserat-2020-2024-2025-04-07.pdf), [Energimyndigheten](https://www.energimyndigheten.se/globalassets/fornybart/framjande-av-vindkraft/vagledning-om-kommunal-tillstyrkan_2015-02-02.pdf)

## 2. Authority mapping (for liite/lupa lists)

| Authority | Role |
|---|---|
| **Länsstyrelsen** (County Administrative Board), via **Miljöprövningsdelegationen** (MPD) | Main permitting authority for Miljöbalken-based permits: environmental permits (ch. 9), EIA (ch. 6), water operations (ch. 11), dam safety supervision |
| **Kommun / Byggnadsnämnden** (Municipality / Building Committee) | PBL-based building permits (bygglov), detaljplan; also holds the wind-power "kommunalt veto" |
| **Naturvårdsverket** (Swedish EPA) | National environmental guidance, some appeals |
| **Energimyndigheten** (Swedish Energy Agency) | Large-scale wind permits guidance/coordination, energy-project support schemes |
| **Svenska kraftnät** | Transmission grid connection; also publishes dam-safety guidance |
| **MSB** (Myndigheten för samhällsskydd och beredskap) | Fire/explosion safety (LBE), civil protection coordination (LSO) |
| **SSM** (Strålsäkerhetsmyndigheten) | Nuclear/radiation safety — already used correctly in the existing SMR entry |
| **Bolagsverket** | Company registration — already used correctly in the existing SMR entry |

All of this is internally consistent with, and confirms the accuracy of, the
one entry that already existed (`SE`/`SMR`) before this research — a good
sign the mapping above is sound.

## 3. A mechanism question needs your input before I build anything

There are two different things wrong with the current FI-citation leak, and
they need different fixes:

- **The citation itself is wrong** (bare Finnish statute number, or — after
  PR #104 — glossed with a Finnish-law name). This is fixable with a
  **deterministic backstop**: a citation-swap table + regex, same shape as
  the Traficom/STUK/ELY-keskus backstops already shipped. Cheap, reusable
  (one entry per FI statute, applies across every hanketyyppi that cites it),
  fast to build and verify.
- **The surrounding factual claims may not hold for Sweden even with the
  citation fixed.** E.g. `tuulivoima_maa`'s KERROS-3 text says EIA is
  mandatory "for projects ≥10 MW or ≥5 turbines" — that's Finland's YVA-laki
  threshold; Sweden's MB ch. 6 EIA trigger and process (plus the kommunalt
  veto, which has no FI equivalent at all) are genuinely different. A
  citation-only backstop would produce grammatically correct Swedish prose
  citing the *right law* but still describing *Finland's* thresholds and
  process — a real fix. only fully solving this needs country-aware
  `context_extra` content (an override block per hanketyyppi × country,
  same pattern as `_COUNTRY_LIITTEET` today), which is a much bigger
  content-authoring effort — real Swedish-process prose, not just a citation
  swap.

Given "no rush, do it right": my recommendation is to build the citation
backstop first (fast, fixes the literal bug reported, same proven pattern as
PR #104) and treat the deeper context_extra content work as its own
follow-up phase once we've done this for a few countries and can see the
real shape of it — but this is your call, not mine, since it changes the
shape of the rest of the "combined queue." Let me know which you want before
I build the backstop.

## 4. Draft _COUNTRY_LIITTEET entries for SE's remaining 13 hanketyyppis

SE currently covers `SMR`, `smr_bess`, `datakeskus` (+ `smr_se` aliased in
PR #104). Drafts below for the rest, grounded in the verified mapping above,
matching the existing `SMR` entry's format and quality bar. Not yet written
into the codebase — for your review first.

**BESS**
- Sijaintikartta / Lägesbeskrivning (skala 1:20 000)
- Maankäyttöselvitys PDF (NCE)
- Anmälan/tillstånd miljöfarlig verksamhet — Miljöbalken kap. 9 (anmälan till kommun eller tillstånd från Länsstyrelsen beroende på storlek)
- Brandskyddsdokumentation — Lag (2010:1011) om brandfarliga och explosiva varor (LBE)
- Bygglov / Detaljplan — Plan- och bygglagen (2010:900)
- Nätanslutningsplan (Svenska kraftnät / lokalt elnätsbolag)
- Bolagsregistreringsutdrag (Bolagsverket)
- Fullmakt (om ombud företräder sökanden)

**tuulivoima_maa** (onshore wind)
- Sijaintikartta / Lägesbeskrivning
- Maankäyttöselvitys PDF (NCE)
- Miljökonsekvensbeskrivning (MKB) — Miljöbalken kap. 6
- Tillståndsansökan miljöfarlig verksamhet — Miljöbalken kap. 9 (prövas av Miljöprövningsdelegationen, Länsstyrelsen)
- Kommunalt tillstyrkande (kommunalt veto — Miljöbalken 16 kap. 4 §)
- Bygglov / Detaljplan — Plan- och bygglagen (2010:900)
- Nätanslutningsplan (Svenska kraftnät / lokalt elnätsbolag)
- Bolagsregistreringsutdrag (Bolagsverket)
- Fullmakt

**tuulivoima_meri / offshore_wind** (offshore wind — same real project, shared entry per the PR #103 aliasing precedent)
- Sijaintikartta / Lägesbeskrivning (sjökort)
- Maankäyttöselvitys PDF (NCE)
- Miljökonsekvensbeskrivning (MKB) — Miljöbalken kap. 6
- Tillståndsansökan vattenverksamhet — Miljöbalken kap. 11 (prövas av mark- och miljödomstol för havsbaserad vindkraft)
- Kommunalt tillstyrkande (kommunalt veto — Miljöbalken 16 kap. 4 §, gäller även kustnära havsbaserad vindkraft)
- Nätanslutningsplan (Svenska kraftnät)
- Bolagsregistreringsutdrag (Bolagsverket)
- Fullmakt

**aurinkovoima** (solar)
- Sijaintikartta / Lägesbeskrivning
- Maankäyttöselvitys PDF (NCE)
- Bygglov / Detaljplan — Plan- och bygglagen (2010:900)
- Anmälan miljöfarlig verksamhet (tarvitt., stora anläggningar) — Miljöbalken kap. 9
- Nätanslutningsplan (lokalt elnätsbolag)
- Bolagsregistreringsutdrag (Bolagsverket)
- Fullmakt

**vesivoima** (hydropower)
- Sijaintikartta / Lägesbeskrivning
- Maankäyttöselvitys PDF (NCE)
- Miljökonsekvensbeskrivning (MKB) — Miljöbalken kap. 6
- Tillståndsansökan vattenverksamhet — Miljöbalken kap. 11 (prövas av mark- och miljödomstol)
- Dammsäkerhetsdokumentation — Förordning (2014:214) om dammsäkerhet
- Nätanslutningsplan (Svenska kraftnät)
- Bolagsregistreringsutdrag (Bolagsverket)
- Fullmakt

**hybridi** (BESS + wind/solar hybrid)
- Union of the BESS and tuulivoima_maa/aurinkovoima lists above (MKB, MB kap. 9 tillstånd, LBE brandskydd, PBL bygglov, kommunalt veto if wind component present, nätanslutningsplan, bolagsregistreringsutdrag, fullmakt) — needs the real per-component judgment applied when actually written, not just a mechanical union.

**asuinrakennus / liikerakennus / maatalous / teollisuus** (construction permit types — residential/commercial/agricultural/industrial)
- Sijaintikartta / Lägesbeskrivning
- Bygglov — Plan- och bygglagen (2010:900)
- Detaljplan / förhandsbesked (tarvitt.)
- Grannhörande — Plan- och bygglagen (2010:900)
- Anmälan/tillstånd miljöfarlig verksamhet (tarvitt., industri/lantbruk) — Miljöbalken kap. 9
- Bolagsregistreringsutdrag (Bolagsverket)
- Fullmakt

**ymparistolupa** (environmental permit)
- Tillståndsansökan miljöfarlig verksamhet — Miljöbalken kap. 9
- Miljökonsekvensbeskrivning (MKB, tarvitt.) — Miljöbalken kap. 6
- Bolagsregistreringsutdrag (Bolagsverket)
- Fullmakt

**muu** (other) — left as the generic FI fallback; not hanketyyppi-specific enough to draft real content for without knowing what "other" actually covers case by case.

These are drafts for review, not final — in particular `hybridi`'s entry
needs real per-instance judgment rather than a mechanical union, and I'd
want your read on whether the construction-type entries (asuinrakennus etc.)
are worth the effort now versus later given they're likely lower business
priority than the energy-project types.
