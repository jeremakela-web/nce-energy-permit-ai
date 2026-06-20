# NCE Permit AI — RAG Coverage Matrix

Auto-generated | 2026-06-20 | Sources: `_COUNTRY_LUVAT`, `_HANKE_CFG`, `ingest_web.py`, `ingest_iaea.py`, `ingest_precedent.py`, `ingest_playwright.py`

Legend: ✅ = `_COUNTRY_LUVAT` entry + adequate RAG · ⚠️ = entry exists but RAG thin or config aliased · ❌ = not covered · 🚧 = regWarning (legal framework incomplete)

---

## RAG Chunk Counts per Country

| Country | Code | Chunks | RAG level |
|---------|------|-------:|-----------|
| Finland | FI | 1,042 | Full |
| Sweden | SE | 1,563 | Full |
| Denmark | DA | 471 | Partial — low coverage |
| Norway | NO | 1,270 | Full |
| Poland | PL | 2,799 | Full |
| Germany | DE | 2,433 | Full |
| **Estonia** | **EE** | **~450 est.** | **New — 6 source docs (2026-06-20); run ingest to confirm count** |
| EU/IAEA | EU | 738 | Partial — IAEA + EIA Directive indexed; GSR docs missing |
| **Total** | | **~10,766** | |

---

## Coverage Matrix (project types × countries)

| Project type | ID | FI | SE | DA | NO | PL | DE | EE |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BESS (battery storage) | `BESS` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Wind — onshore | `tuulivoima_maa` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Wind — offshore | `tuulivoima_meri` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Solar / PV | `aurinkovoima` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| SMR (nuclear) | `SMR` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | 🚧 |
| Hybridivoimala (BESS+wind/solar) | `hybridi` | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Data centre | `datakeskus` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Asuinrakennus (residential) | `asuinrakennus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Kaupallinen (commercial) | `liikerakennus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Teollisuus (industrial) | `teollisuus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Maatalous (agriculture) | `maatalous` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Vesivoima (hydropower) | `vesivoima` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ❌ |
| EGS (deep geothermal) | `egs` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ❌ |
| sCO₂ turbine | — | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**EE notes:**
- **SMR (🚧 regWarning)**: No dedicated nuclear power plant law in Estonia. Draft `Tuumaenergia seadus` under development (not yet adopted as of 2024). Reports correctly show nuclear law caveat and regWarning status.
- **BESS**: No dedicated BESS regulation — general Electricity Market Act (Elektrituruseadus) + Ehitusseadustik applies. Reports note this gap explicitly.
- **Offshore wind (✅)**: Uses combined permit (ühisluba) — a distinguishing feature unique to Estonia's post-2023 reformed process. EU-approved state aid €2.6B (2024) documented.
- **Onshore wind (✅)**: National 1,200 MW target by 2030; wind priority development areas (tuuleenergia arendusalad); RRP streamlining measures documented.
- **Solar PV (✅)**: Dual-track permitting documented — simplified path for rooftop <15 kW (teatis only) vs. full process for utility-scale ground-mounted.
- **Vesivoima, EGS (❌)**: Not added — Estonia has no significant hydropower capacity and deep geothermal is not a realistic near-term project type for the market.

---

## doc_type × permit_phase Coverage (Step 7 additions)

New metadata dimensions added 2026-06-12 via `ingest_precedent.py`.

| doc_type | permit_phase | FI | SE | DA | NO | PL | DE | EU | EE |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| case_law | all | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ | — | ❌ |
| bat_principles | lupavaihe | ⚠️ | ⚠️ | ❌ | ⚠️ | ✅ | ✅ | ⚠️ | ❌ |
| eia_guidance | esiselvitys | ⚠️ | ❌ | ⚠️ | ❌ | ❌ | ❌ | ✅ | ❌ |
| bim_standard | rakentaminen | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| noise_standard | lupavaihe | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | — | ❌ |

**Legend (doc_type table):** ✅ = well-indexed (≥25 chunks) · ⚠️ = thin (1–24 chunks, landing page only) · ❌ = not indexed

**Steps 7+8 chunks indexed:**

| Country | Source | doc_type | Method | Chunks |
|---------|--------|----------|--------|-------:|
| FI | KHO | case_law | requests | 2 |
| FI | HAO | case_law | requests | 1 |
| FI | YVA_guidance | eia_guidance | Playwright | 1 |
| FI | SYKE_BAT | bat_principles | Playwright | 1 |
| FI | AVI | case_law | Playwright | 1 |
| SE | MÖD | case_law | Playwright | 1 |
| SE | Naturvardsverket_BAT | bat_principles | Playwright | 1 |
| DA | Energiklagenaevnet | case_law | Playwright | 1 |
| DA | Miljoeklagenaevnet | case_law | Playwright | 1 |
| DA | Miljostyrelsen_VVM | eia_guidance | Playwright | 2 |
| NO | NVE_vedtak | case_law | Playwright | 1 |
| NO | Miljodirektoratet_vedtak | case_law | Playwright | 1 |
| NO | Miljodirektoratet_BAT | bat_principles | Playwright | 1 |
| PL | NSA | case_law | requests | 127 |
| PL | GIOS_BAT | bat_principles | requests | 99 |
| DE | UBA_BAT | bat_principles | requests | 25 |
| DE | BNetzA_beschlusskammern | case_law | Playwright | 1 |
| EU | EUR_Lex_BAT | bat_principles | Playwright | 1 |
| EU | EIA_Directive | eia_guidance | Playwright | 36 |
| **Total** | | | | **304** |

**Remaining gaps (Steps 7+8):**

| Country | Source | Reason |
|---------|--------|--------|
| FI | STUK, SYKE, ymparisto.fi/melu | URL changed / 404 |
| SE | EI_beslut, Boverket_BIM | URL changed / 404 |
| NO | — | All sources indexed (thin) |
| DA | — | All sources indexed (thin) |
| PL | URE_decyzje | URL changed / 404 |
| DE | BVerwG | Playwright timeout (JS-heavy) |
| DE | BMWSB_BIM | URL changed / 404 |
| EU | EU_BIM | 0 text extracted |
| **EE** | **Riigikohus (Supreme Court), Keskkonnaamet case decisions** | **Not yet indexed — future task** |

---

## Estonia (EE) — New Country Summary (2026-06-20)

### RAG Source Documents Indexed (permit_ai/rag_docs/EE/)

| File | Content | Scope |
|------|---------|-------|
| `elektrituruseadus_electricity_market_act.txt` | Electricity Market Act (ETS) — production license, grid connection, renewable support, BESS framework | All energy project types |
| `keskkonnaseadustiku_yldosa_seadus_eia.txt` | KeÜS general environmental code + KMH-KSH EIA Act + Planeerimisseadus + Ehitusseadustik | All project types |
| `elering_grid_connection_requirements.txt` | Elering TSO grid connection procedure, technical requirements (RfG/DCC/HVDC), connection timeline | All energy + data centre types |
| `energiamajanduse_korralduse_seadus_sector_organisation.txt` | Energy Sector Organisation Act — NECP targets, renewable auctions, offshore wind state aid €2.6B | Policy + energy project types |
| `offshore_wind_combined_permit_uhlisluba.txt` | Combined offshore permit (ühisluba) — replaces 3-step process; EU state aid SA.110117; maritime spatial plan | Offshore wind |
| `tuuleenergia_wind_energy_permitting.txt` | Onshore wind permitting process — priority development areas, military radar, EIA, noise, grid | Onshore wind |
| `paikeseenergia_solar_pv_permitting.txt` | Solar PV — dual-track (rooftop simplified vs. utility-scale full); agricultural land rules | Solar PV |
| `smr_nuclear_regulatory_estonia_draft.txt` | Nuclear regulatory gap analysis — no nuclear law; Kiirgusseadus limitations; IAEA standards; draft law status | SMR (regWarning) |
| `bess_datakeskus_teollisuus_ehitus_permitting.txt` | BESS (no dedicated law), data centres, industrial, residential, commercial, agricultural buildings | BESS, DC, teollisuus, asuinrak., liikerak., maatalous |

### Config Changes in generate_application.py

| Dict | Key Added | Content |
|------|-----------|---------|
| `_COUNTRY_CONFIG` | `"EE"` | Full country config with authorities, key laws, prompt prefix |
| `_COUNTRY_LUVAT` | `"EE"` | 12 project types: tuulivoima_maa, aurinkovoima, tuulivoima_meri, offshore_wind, BESS, SMR, smr_ee, datakeskus, teollisuus, asuinrakennus, liikerakennus, maatalous, hybridi |
| `_COUNTRY_LIITTEET` | `"EE"` | Attachment lists for SMR, smr_ee, tuulivoima_maa, tuulivoima_meri |
| `_NATIONAL_SUPERVISORS` | `"EE"` | 13 project type supervisors |
| `_BESS_MARKET_DATA` | `"EE"` | Storage index 120 €k/MW/year (estimated, Q1/2026) |
| `_HANKE_CFG` | `"smr_ee"` alias | Maps to base SMR config |
| `_LANG_INSTRUCTIONS` | `"ET"` | Estonian language output instruction (fallback if lang=ET requested) |
| `ingest_countries.py` | `"EE": "et"` | Registers Estonia in COUNTRY_LANG dict |

### Key Regulatory Distinctions (Estonia-specific)

1. **Offshore wind combined permit (ühisluba)**: Single procedure replacing 3 separate permits. Lead authority: MKM. 36-month statutory max. Unique to Estonia in this corpus.
2. **EU offshore wind state aid €2.6B**: SA.110117, approved 2024. CfD mechanism, 20-year support. Documented in corpus.
3. **Wind priority development areas**: ~1,000 MW designated in county plans under RRP. Streamlined planning for projects in these areas.
4. **Small solar simplified path**: Rooftop <15 kW (household) and <50 kW (commercial) require only ehitusteatis + Konkurentsiamet notification. No EIA, no production license.
5. **BESS — no dedicated law**: Explicitly flagged. General ETS + Ehitusseadustik applies.
6. **SMR — regWarning**: No nuclear power plant licensing law. Draft Tuumaenergia seadus under development. Terviseamet has limited nuclear capacity. All SMR entries show ⚠️ disclaimer.
7. **Elering = Fingrid equivalent**: Estonia's TSO. Grid connection via liitumisleping. Equivalent role in EE corpus to Fingrid in FI corpus.
8. **Military radar**: Kaitseministeerium clearance is mandatory for all wind projects. Documented in wind and offshore wind permit lists.

---

## Notes

- **DA**: all ⚠️ regardless of config — low chunk count (~467) means RAG answers will be thin
- **DE**: upgraded to Full — BauGB (476 chunks) + EnWG (900 chunks) indexed 2026-06-12; BImSchG already present
- **EU/IAEA**: SMR safety standards indexed 2026-06-12 — SSR-2/1 Rev.1 (275 chunks), SSG-52 (171 chunks), NS-R-5 Rev.1 (183 chunks); GSR Part 1/3/4 not yet indexed
- **EE**: 6 new source documents added 2026-06-20; run `python3 permit_ai/ingest_countries.py --country EE` to index and confirm chunk count
- **Steps 7+8 SPA sources**: Playwright resolved the 404/202 blocking; all 15 targeted SPA sources now have at least a landing-page chunk indexed. Most yield 1–2 chunks (minimal text on index pages) — actual decision lists/documents are behind dynamic pagination or search UIs requiring deeper interaction. High-value exception: EU EIA Directive (36 chunks, full text).
- **FI EGS**: aliased to `aurinkovoima` config — EGS-specific guidance is thin
- **FI asuinrakennus/teollisuus/maatalous/liikerakennus**: generic `_HANKE_CFG` entries, limited RAG depth
- **DA hybridi**: no entry in `_COUNTRY_LUVAT`; falls through to FI base config
- **sCO₂**: not yet in `_HANKE_CFG`; planned feature only

---

## Known Gaps

| Gap | Countries | Priority | Action needed |
|-----|-----------|----------|---------------|
| Thin SPA sources (1–2 chunks) need deeper crawl | All | High | Playwright ingest added (Step 8) but index pages only; needs interaction-based crawl (clicking pagination, search) for actual decision texts from domstol.se, nve.no, naevneneshus.dk, bverwg.de etc. |
| DA RAG depth (~467 chunks) | DA | High | Index retsinformation.dk, Energistyrelsen, Miljøstyrelsen |
| EE case law not yet indexed | EE | Medium | Riigikohus decisions, Keskkonnaamet EIA decisions — Riigiteataja case law search |
| IAEA GSR Part 1/3/4 + TECDOC series | EU | Medium | SSR-2/1, SSG-52, NS-R-5 now indexed; GSR docs still missing |
| Bauordnungsrecht (Landesbauordnungen) | DE | Medium | Index BayBO, LBO BW and other state building codes |
| Offshore wind (Ustawa offshore 2021) | PL | Medium | Index Polish offshore wind act + GDOŚ/URE guidance |
| EE vesivoima + EGS configs | EE | Low | Not realistic for current EE market; add if demand arises |
| FI EGS dedicated config | FI | Low | Remove `egs` alias to `aurinkovoima`; write EGS-specific permits |
| DA hybridi config | DA | Low | Add `hybridi` entry in `_COUNTRY_LUVAT["DA"]` |
| sCO₂ turbine (all countries) | All | Low | Not yet in `_HANKE_CFG`; feature planned |
| SMR nuclear law adoption tracking | EE | Ongoing | Monitor Riigikogu legislative calendar for Tuumaenergia seadus |

---

> **Update this file after every RAG indexing session.**
