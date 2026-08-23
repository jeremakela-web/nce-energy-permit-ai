# Denmark + Norway — real statutory equivalents + _COUNTRY_LIITTEET expansion

Research findings for item 5(B) + item 4 remainder + task #31 (STUK-naming,
rakentamislupa-leak), Denmark and Norway (fourth/fifth in the pilot-priority
queue). Researched together since investigation is low-risk and efficient to
batch, each verified independently — not assumed identical just because
they're neighboring Nordic countries, per standing practice. Every statute
verified via live web search against primary/near-primary sources
(retsinformation.dk, lovdata.no, DSA, NVE, DEMA), not recalled. Investigation
only — build on go-ahead, same cadence as SE/PL/LT.

---

## 1. Verified statute mapping

| FI statute | Denmark (DA) | Norway (NO) |
|---|---|---|
| MRL 132/1999 (zoning) | **Planloven** (1992, consolidated) — separate from building law | **Plan- og bygningsloven** (PBA, 2008) — merged with building, see below |
| Rakentamislaki 751/2023 (building) | **Byggeloven** (Building Act) — confirmed separate from Planloven, same split as Poland/Lithuania | Same law as MRL: **Plan- og bygningsloven** — Norway merges zoning+building like Sweden's PBL, NOT split like Denmark |
| YVA-laki 252/2017 (EIA) | **Miljøvurderingsloven** (Environmental Assessment Act, governs EIA since 2016) | Same law, **PBA kapittel 14** (konsekvensutredning) — a chapter of PBA, not a separate act, mirroring how Sweden's EIA is Miljöbalken ch. 6 within a broader code |
| YSL 527/2014 (environmental permit) | **Miljøbeskyttelsesloven** (Environmental Protection Act — pollution/miljøgodkendelse) | **Forurensningsloven** (Pollution Control Act, 1981) |
| Vesilaki 587/2011 (water) | **Vandløbsloven** (Watercourse Act) — confirmed the primary framework for hydropower (vandkraft) permits and watercourse modifications specifically (real example found: Gudenå river hydropower maintenance obligations). NOT Vandforsyningsloven, which is drinking-water/groundwater supply specifically — a different, narrower law. | **Vannressursloven** (Water Resources Act, 2000) |
| Pelastuslaki 379/2011 (fire/rescue) | **Beredskabsloven** (Emergency Preparedness Act) — DEMA administers | **Brann- og eksplosjonsvernloven** (Fire and Explosion Prevention Act, 2002) — DSB administers |
| Ydinenergialaki 990/1987 (nuclear) | Denmark has an **active statutory ban** on nuclear power generation (1985 law) — more clear-cut than a "no permit path" situation, this is an explicit prohibition. A separate Nuclear Installations Act (1962) exists for historical/materials regulation (Risø's decommissioned research reactors). | Real, active framework: **Atomenergiloven** (Act on Nuclear Energy Activities, No. 28 of 1972) + **Strålevernloven** (Radiation Protection Act, No. 36 of 2000), regulated by **DSA**. Norway has never had a commercial nuclear power plant (only 2 research reactors, both permanently shut 2018/2019, now in decommissioning) — but unlike Denmark's ban or Lithuania's ambiguity, Norway has **genuine, current commercial SMR proposals** (real 2023-2025 news: companies submitting siting proposals for first reactors). See section 2 below — this needs its own careful framing, distinct from both DA and LT. |
| Patoturvallisuuslaki 494/2009 (dam safety) | **Not found** after two targeted searches. No dedicated Danish dam-safety act or regulation surfaced. Vandløbsloven governs the underlying hydropower/watercourse installations broadly but I found no confirmation dam safety specifically is regulated under it (unlike Poland/Lithuania, where I could positively confirm "folded into X"). Recommend: no DA entry in the backstop, same treatment as PL/LT's "no separate law," but flagged as a genuine open question rather than a confirmed "folded into Y" — worth a follow-up if it ever becomes relevant. | **Damsikkerhetsforskriften** (Regulation on safety at watercourse facilities, 2009, no. 1600) — a real, specific, *named* dam-safety regulation (unlike PL/LT/possibly-DA), based on Vannressursloven + Vassdragsreguleringsloven (1917) + Energiloven (1990). **NVE** is the supervisory authority. |

## 2. Norway's nuclear situation needs its own careful framing — a third distinct case

This is now the third different "nuclear reality" found across the queue, and
each is genuinely different — worth being precise rather than reusing LT's
template:

- **Denmark**: nuclear power generation is **legally banned** (1985 law) —
  the most clear-cut of the three, an explicit prohibition, not just an
  absent framework.
- **Lithuania**: real law + real active regulator (VATESI), but VATESI's
  current remit is decommissioning/waste only — no demonstrated new-build
  path.
- **Norway**: real law (Atomenergiloven, 1972) + real active regulator
  (DSA), **and** genuine current commercial interest — real companies have
  submitted SMR siting proposals in 2023-2025, with news coverage describing
  "plans for first reactors within 10 years." Norway has never operated a
  commercial plant, and Atomenergiloven predates SMR technology by decades,
  so exactly how a new-build commercial license would work under the
  existing 1972 Act is genuinely unclear from what I've verified — but
  unlike Denmark or Lithuania, there is no reason to state "no permit path
  could exist" here; the honest statement is closer to "a licensing
  framework exists and real commercial proposals are being pursued, but the
  specific process for a first-of-its-kind commercial license hasn't been
  demonstrated yet."

Recommend keeping all three countries' nuclear treatment distinct rather
than reusing one template — same "don't simplify, state what's actually
true" principle as LT's SMR/smr_bess treatment.

## 3. Task #31 folded in — kaava_SMR / kaava_aurinkovoima for both

Not yet checked directly (deferred until build phase, matching the SE/PL/LT
pattern of checking exact card text right before fixing it) — but the
known bugs from task #31 (STUK-naming, rakentamislupa-leak) apply to both
DA and NO's kaava_SMR/kaava_aurinkovoima per the original sweep. Given
section 2 above, DA's kaava_SMR should state the legal ban directly (not a
"no permit path" hedge — a ban is a stronger, more definite statement); NO's
kaava_SMR needs the "real law + real regulator + genuine current interest,
but no demonstrated first-of-kind process" framing.

## 4. `_COUNTRY_LIITTEET` — DA and NO currently at 4/21 each (SMR, smr_bess, datakeskus, smr_da/smr_no aliased in PR #103/#104)

Same energy-first sequencing as SE/PL/LT, pending go-ahead: BESS,
tuulivoima_maa, tuulivoima_meri/offshore_wind (aliased), aurinkovoima,
vesivoima, hybridi, ymparistolupa. SMR/smr_bess already have real entries
for both countries (per the existing coverage) — worth checking those
existing entries for the same citation-accuracy issues found in SE/PL's
pre-existing SMR entries before assuming they're clean.

---

**Ready to build once confirmed** — same shape as SE/PL/LT: `_LAW_TRANS`
fix/addition at the source for both DA and NO columns, `_LAW_CITATION_REPLACEMENT`
entries, `_COUNTRY_CONFIG["DA"/"NO"]["prompt_prefix"]` enhancement with the
explicit mapping, kaava_* fixes (with the distinct nuclear framing per
country), and `_COUNTRY_LIITTEET` expansion for the energy types. Your call
whether to build DA and NO together in one PR or split them — the research
is genuinely intertwined efficiently, but the actual code changes are
naturally separable per country if you'd rather review them separately.
