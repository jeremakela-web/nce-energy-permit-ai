# NCE Permit AI — RAG Coverage Matrix

Auto-generated | 2026-06-12 | Sources: `_COUNTRY_LUVAT`, `_HANKE_CFG`, `ingest_web.py`, `ingest_iaea.py`, `ingest_precedent.py`

Legend: ✅ = `_COUNTRY_LUVAT` entry + adequate RAG · ⚠️ = entry exists but RAG thin or config aliased · ❌ = not covered

---

## RAG Chunk Counts per Country

| Country | Code | Chunks | RAG level |
|---------|------|-------:|-----------|
| Finland | FI | 1,039 | Full |
| Sweden | SE | 1,561 | Full |
| Denmark | DA | 467 | Partial — low coverage |
| Norway | NO | 1,267 | Full |
| Poland | PL | 2,827 | Full |
| Germany | DE | 2,432 | Full |
| EU/IAEA | EU | 701 | Partial — IAEA SMR standards indexed; GSR docs missing |
| **Total** | | **10,266** | |

---

## Coverage Matrix (project types × countries)

| Project type | ID | FI | SE | DA | NO | PL | DE |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BESS (battery storage) | `BESS` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Wind — onshore | `tuulivoima_maa` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Wind — offshore | `tuulivoima_meri` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Solar / PV | `aurinkovoima` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| SMR (nuclear) | `SMR` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Hybridivoimala (BESS+wind/solar) | `hybridi` | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Data centre | `datakeskus` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Asuinrakennus (residential) | `asuinrakennus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Kaupallinen (commercial) | `liikerakennus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Teollisuus (industrial) | `teollisuus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Maatalous (agriculture) | `maatalous` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Vesivoima (hydropower) | `vesivoima` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| EGS (deep geothermal) | `egs` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| sCO₂ turbine | — | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## doc_type × permit_phase Coverage (Step 7 additions)

New metadata dimensions added 2026-06-12 via `ingest_precedent.py`.

| doc_type | permit_phase | FI | SE | DA | NO | PL | DE | EU |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| case_law | all | ⚠️ | ❌ | ❌ | ❌ | ✅ | ❌ | — |
| bat_principles | lupavaihe | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| eia_guidance | esiselvitys | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| bim_standard | rakentaminen | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| noise_standard | lupavaihe | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | — |

**Legend (doc_type table):** ✅ = indexed · ⚠️ = indexed but thin (<5 chunks) · ❌ = not indexed (source returned 404 or JS-only)

**Step 7 chunks indexed:**

| Country | Source | doc_type | Chunks |
|---------|--------|----------|-------:|
| FI | KHO | case_law | 2 |
| FI | HAO | case_law | 1 |
| PL | NSA | case_law | 127 |
| PL | GIOS_BAT | bat_principles | 99 |
| DE | UBA_BAT | bat_principles | 25 |
| **Total** | | | **254** |

**Step 7 failures (JS-rendered SPAs, not accessible via HTTP):**

| Country | Source | URL | Reason |
|---------|--------|-----|--------|
| FI | AVI | avi.fi/ymparistoluvat | SSL error |
| FI | STUK | stuk.fi/paatokset | 404 |
| FI | SYKE_BAT | syke.fi/BREF | 404 |
| FI | YVA_guidance | ymparisto.fi/YVA | 404 |
| FI | ymparisto_melu | ymparisto.fi/tuulivoimamelu | 404 |
| SE | MÖD | domstol.se | 404 |
| SE | EI_beslut | ei.se | 404 |
| SE | Naturvardsverket_BAT | naturvardsverket.se | 404 |
| SE | Boverket_BIM | boverket.se/digitalt | 404 |
| NO | NVE_vedtak | nve.no/konsesjonsvedtak | 404 |
| NO | Miljodirektoratet_vedtak | miljodirektoratet.no | 404 |
| NO | Miljodirektoratet_BAT | miljodirektoratet.no | 404 |
| DA | Energiklagenaevnet | naevneneshus.dk | 404 |
| DA | Miljoeklagenaevnet | naevneneshus.dk | 404 |
| DA | Miljostyrelsen_VVM | mst.dk | 404 |
| PL | URE_decyzje | ure.gov.pl/decyzje | 404 |
| DE | BVerwG | bverwg.de | 404 |
| DE | BNetzA | bundesnetzagentur.de | 404 |
| DE | BMWSB_BIM | bmwsb.bund.de | 404 |
| EU | EUR_Lex_BAT | eur-lex.europa.eu | HTTP 202 (async) |
| EU | EIA_Directive | eur-lex.europa.eu | HTTP 202 (async) |
| EU | EU_BIM | eubim.eu | 0 text extracted |

---

## Notes

- **DA**: all ⚠️ regardless of config — low chunk count (~467) means RAG answers will be thin
- **DE**: upgraded to Full — BauGB (476 chunks) + EnWG (900 chunks) indexed 2026-06-12; BImSchG already present
- **EU/IAEA**: SMR safety standards indexed 2026-06-12 — SSR-2/1 Rev.1 (275 chunks), SSG-52 (171 chunks), NS-R-5 Rev.1 (183 chunks); GSR Part 1/3/4 not yet indexed
- **Step 7 JS-rendered sources**: 19 of 27 requested sources return 404 or HTTP 202 — these use React/Angular SPAs and require headless browser (Playwright) to access; re-indexing requires a Playwright-based ingest script
- **FI EGS**: aliased to `aurinkovoima` config — EGS-specific guidance is thin
- **FI asuinrakennus/teollisuus/maatalous/liikerakennus**: generic `_HANKE_CFG` entries, limited RAG depth
- **DA hybridi**: no entry in `_COUNTRY_LUVAT`; falls through to FI base config
- **sCO₂**: not yet in `_HANKE_CFG`; planned feature only

---

## Known Gaps

| Gap | Countries | Priority | Action needed |
|-----|-----------|----------|---------------|
| JS-rendered case law / BAT sites (19 sources) | All | High | Build Playwright-based ingest for domstol.se, naturvardsverket.se, ymparisto.fi, nve.no/vedtak, naevneneshus.dk, mst.dk, bverwg.de, eur-lex.europa.eu etc. |
| DA RAG depth (~467 chunks) | DA | High | Index retsinformation.dk, Energistyrelsen, Miljøstyrelsen |
| IAEA GSR Part 1/3/4 + TECDOC series | EU | Medium | SSR-2/1, SSG-52, NS-R-5 now indexed; GSR docs still missing |
| Bauordnungsrecht (Landesbauordnungen) | DE | Medium | Index BayBO, LBO BW and other state building codes |
| Offshore wind (Ustawa offshore 2021) | PL | Medium | Index Polish offshore wind act + GDOŚ/URE guidance |
| FI EGS dedicated config | FI | Low | Remove `egs` alias to `aurinkovoima`; write EGS-specific permits |
| DA hybridi config | DA | Low | Add `hybridi` entry in `_COUNTRY_LUVAT["DA"]` |
| sCO₂ turbine (all countries) | All | Low | Not yet in `_HANKE_CFG`; feature planned |

---

> **Update this file after every RAG indexing session.**
