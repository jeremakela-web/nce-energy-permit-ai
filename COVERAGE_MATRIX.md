# NCE Permit AI — RAG Coverage Matrix

Auto-generated | 2026-06-12 | Source: `_COUNTRY_LUVAT`, `_HANKE_CFG` in `permit_ai/generate_application.py`; IAEA PDFs indexed via `permit_ai/ingest_iaea.py`

Legend: ✅ = `_COUNTRY_LUVAT` entry + adequate RAG · ⚠️ = entry exists but RAG thin or config aliased · ❌ = not covered

---

## RAG Chunk Counts per Country

| Country | Code | Chunks | RAG level |
|---------|------|-------:|-----------|
| Finland | FI | 1,036 | Full |
| Sweden | SE | 1,561 | Full |
| Denmark | DA | 467 | Partial — low coverage |
| Norway | NO | 1,267 | Full |
| Poland | PL | 2,573 | Full |
| Germany | DE | 2,407 | Full |
| EU/IAEA | EU | 701 | Partial — IAEA SMR standards indexed; GSR docs missing |
| **Total** | | **10,012** | |

---

## Coverage Matrix

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

**Notes:**
- **DA**: all ⚠️ regardless of config — low chunk count (~467) means RAG answers will be thin
- **DE**: upgraded to Full — BauGB (476 chunks) + EnWG (900 chunks) indexed 2026-06-12; BImSchG already present
- **EU/IAEA**: SMR safety standards indexed 2026-06-12 — SSR-2/1 Rev.1 (275 chunks), SSG-52 (171 chunks), NS-R-5 Rev.1 (183 chunks); GSR Part 1/3/4 and IAEA-TECDOC series not yet indexed
- **FI EGS**: aliased to `aurinkovoima` config — EGS-specific guidance is thin
- **FI asuinrakennus/teollisuus/maatalous/liikerakennus**: generic `_HANKE_CFG` entries, limited RAG depth
- **DA hybridi**: no entry in `_COUNTRY_LUVAT`; falls through to FI base config
- **sCO₂**: not yet in `_HANKE_CFG`; planned feature only

---

## Known Gaps

| Gap | Countries | Priority | Action needed |
|-----|-----------|----------|---------------|
| DA RAG depth (~467 chunks) | DA | High | Index retsinformation.dk, Energistyrelsen, Miljøstyrelsen |
| IAEA GSR Part 1/3/4 + TECDOC series | EU | Medium | SSR-2/1, SSG-52, NS-R-5 now indexed; GSR docs still missing |
| Bauordnungsrecht (Landesbauordnungen) | DE | Medium | Index BayBO, LBO BW and other state building codes |
| Offshore wind (Ustawa offshore 2021) | PL | Medium | Index Polish offshore wind act + GDOŚ/URE guidance |
| BAT reference documents (BREFs) | All | Medium | Index EU BAT principles |
| FI EGS dedicated config | FI | Low | Remove `egs` alias to `aurinkovoima`; write EGS-specific permits |
| DA hybridi config | DA | Low | Add `hybridi` entry in `_COUNTRY_LUVAT["DA"]` |
| sCO₂ turbine (all countries) | All | Low | Not yet in `_HANKE_CFG`; feature planned |

---

> **Update this file after every RAG indexing session.**
