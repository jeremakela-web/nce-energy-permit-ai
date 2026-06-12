# NCE Permit AI — RAG Coverage Matrix

Auto-generated | 2026-06-12 | Source: `_COUNTRY_LUVAT`, `_HANKE_CFG` in `permit_ai/generate_application.py`

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
| Germany | DE | 1,017 | Partial |
| EU/IAEA | EU | 72 | Partial |
| **Total** | | **7,993** | |

---

## Coverage Matrix

| Project type | ID | FI | SE | DA | NO | PL | DE |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BESS (battery storage) | `BESS` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| Wind — onshore | `tuulivoima_maa` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| Wind — offshore | `tuulivoima_meri` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| Solar / PV | `aurinkovoima` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| SMR (nuclear) | `SMR` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| Hybridivoimala (BESS+wind/solar) | `hybridi` | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Data centre | `datakeskus` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| Asuinrakennus (residential) | `asuinrakennus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| Kaupallinen (commercial) | `liikerakennus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| Teollisuus (industrial) | `teollisuus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| Maatalous (agriculture) | `maatalous` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| Vesivoima (hydropower) | `vesivoima` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| EGS (deep geothermal) | `egs` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| sCO₂ turbine | — | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Notes:**
- **DA**: all ⚠️ regardless of config — low chunk count (~467) means RAG answers will be thin
- **DE**: all ⚠️ — partial RAG coverage (~1,017 chunks); Bauordnungsrecht (state building codes) not yet indexed
- **FI EGS**: aliased to `aurinkovoima` config — EGS-specific guidance is thin
- **FI asuinrakennus/teollisuus/maatalous/liikerakennus**: generic `_HANKE_CFG` entries, limited RAG depth
- **SE/NO/PL hybridi**: no entry in `_COUNTRY_LUVAT` — falls through to FI base config, unreliable
- **sCO₂**: not yet in `_HANKE_CFG`; planned feature only

---

## Known Gaps

| Gap | Countries | Priority | Action needed |
|-----|-----------|----------|---------------|
| DA RAG depth (~467 chunks) | DA | High | Index retsinformation.dk, Energistyrelsen, Miljøstyrelsen |
| Bauordnungsrecht (Landesbauordnungen) | DE | High | Index BayBO, LBO BW and other state building codes |
| IAEA SSR-2/1, NS-R-5, GSR documents | EU | High | Index IAEA safety standards for SMR chapters |
| Offshore wind (Ustawa offshore 2021) | PL | Medium | Index Polish offshore wind act + GDOŚ/URE guidance |
| hybridi config for SE, NO, PL | SE/NO/PL | Medium | Add country-specific entries in `_COUNTRY_LUVAT` |
| BAT reference documents (BREFs) | All | Medium | Index EU BAT principles |
| FI EGS dedicated config | FI | Low | Remove `egs` alias to `aurinkovoima`; write EGS-specific permits |
| sCO₂ turbine (all countries) | All | Low | Not yet in `_HANKE_CFG`; feature planned |

---

> **Update this file after every RAG indexing session.**
