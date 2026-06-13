# NCE Permit AI — RAG Coverage Matrix

Auto-generated | 2026-06-13 | Sources: `_COUNTRY_LUVAT`, `_HANKE_CFG`, `ingest_web.py`, `ingest_iaea.py`, `ingest_precedent.py`, `ingest_playwright.py`

Legend: ✅ = `_COUNTRY_LUVAT` entry + adequate RAG · ⚠️ = entry exists but RAG thin or config aliased · ❌ = not covered

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
| EU/IAEA | EU | 738 | Partial — IAEA + EIA Directive indexed; GSR docs missing |
| **Total** | | **10,316** | |

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
| case_law | all | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ | — |
| bat_principles | lupavaihe | ⚠️ | ⚠️ | ❌ | ⚠️ | ✅ | ✅ | ⚠️ |
| eia_guidance | esiselvitys | ⚠️ | ❌ | ⚠️ | ❌ | ❌ | ❌ | ✅ |
| bim_standard | rakentaminen | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| noise_standard | lupavaihe | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | — |

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

---

## Notes

- **DA**: all ⚠️ regardless of config — low chunk count (~467) means RAG answers will be thin
- **DE**: upgraded to Full — BauGB (476 chunks) + EnWG (900 chunks) indexed 2026-06-12; BImSchG already present
- **EU/IAEA**: SMR safety standards indexed 2026-06-12 — SSR-2/1 Rev.1 (275 chunks), SSG-52 (171 chunks), NS-R-5 Rev.1 (183 chunks); GSR Part 1/3/4 not yet indexed
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
| IAEA GSR Part 1/3/4 + TECDOC series | EU | Medium | SSR-2/1, SSG-52, NS-R-5 now indexed; GSR docs still missing |
| Bauordnungsrecht (Landesbauordnungen) | DE | Medium | Index BayBO, LBO BW and other state building codes |
| Offshore wind (Ustawa offshore 2021) | PL | Medium | Index Polish offshore wind act + GDOŚ/URE guidance |
| FI EGS dedicated config | FI | Low | Remove `egs` alias to `aurinkovoima`; write EGS-specific permits |
| DA hybridi config | DA | Low | Add `hybridi` entry in `_COUNTRY_LUVAT["DA"]` |
| sCO₂ turbine (all countries) | All | Low | Not yet in `_HANKE_CFG`; feature planned |

---

> **Update this file after every RAG indexing session.**
