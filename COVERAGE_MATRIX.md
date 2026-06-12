# NCE Permit AI — RAG Coverage Matrix

Auto-generated | 2026-06-12 | Source: `_COUNTRY_LUVAT`, `_HANKE_CFG` in `permit_ai/generate_application.py`

---

## Summary

| Country | Code | Chunks | Status | Primary sources |
|---------|------|-------:|--------|-----------------|
| Finland | FI | 1,036 | Full | Fingrid, STUK YVL, Luova, Tukes, Pelastuslaki, MRL, Rakentamislaki |
| Sweden | SE | 1,561 | Full | Energimyndigheten, SSM, Mark- och miljödomstolen, Elsäkerhetsverket |
| Denmark | DA | 467 | Partial | Energistyrelsen, Planloven, Bekendtgørelse om nukleare anlæg |
| Norway | NO | 1,267 | Full | NVE, DSA, Energidepartementet, Plan- og bygningsloven, Atomenergiloven |
| Poland | PL | 2,573 | Full | PAA, URE, UDT, Ustawa OZE, Prawo budowlane, Prawo wodne |
| Germany | DE | 1,017 | Partial | BImSchG, EEG, WHG, BauGB, BetrSichV, AtG, WindSeeG, BBergG |
| EU/IAEA | EU | 72 | Partial | EU taxonomy, RED III, IAEA safety standards (incomplete) |
| **Total** | | **7,993** | | |

---

## Project Type Coverage

Legend: ✅ Full config + RAG context · ⚠️ Partial (low chunks or aliased config) · ❌ Not defined

| Project type | ID | FI | SE | DA | NO | PL | DE |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BESS (battery storage) | `BESS` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Wind — onshore | `tuulivoima_maa` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Wind — offshore (fixed) | `tuulivoima_meri` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Wind — offshore (floating) | `offshore_wind` | ✅ | ✅ | ⚠️ | ✅ | ❌ | ✅ |
| Solar / PV | `aurinkovoima` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| SMR (generic) | `SMR` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| SMR — Sweden variant | `smr_se` | — | ✅ | — | — | — | — |
| SMR — Denmark variant | `smr_da` | — | — | ⚠️ | — | — | — |
| SMR — Norway variant | `smr_no` | — | — | — | ✅ | — | — |
| SMR — Germany variant | `smr_de` | — | — | — | — | — | ⚠️ |
| SMR + BESS hybrid | `smr_bess` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Hybrid (BESS + wind/solar) | `hybridi` | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Hydropower | `vesivoima` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Data centre | `datakeskus` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| EGS / deep geothermal | `egs` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Residential building | `asuinrakennus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Industrial facility | `teollisuus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Agricultural building | `maatalous` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Commercial building | `liikerakennus` | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Environmental permit (YSL) | `ymparistolupa` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Other / generic | `muu` | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |

**Notes:**
- FI `egs` is aliased to `aurinkovoima` config — EGS-specific guidance is thin.
- SE/NO/PL `hybridi` falls through to FI base config — unreliable for those countries.
- PL `offshore_wind` uses `tuulivoima_meri` base; Polish offshore framework (2021) not yet indexed.
- DA all types rated ⚠️ regardless of config due to low chunk count (~467).

---

## Known Gaps

| Gap | Country | Priority | Action needed |
|-----|---------|----------|---------------|
| retsinformation.dk coverage incomplete | DA | High | Index Energistyrelsen VE-love, Planloven, Miljøvurderingsloven |
| Bauordnungsrecht (Landesbauordnungen) not indexed | DE | High | Index state-level building codes (BayBO, LBO BW, etc.) |
| IAEA SSR-2/1, NS-R-5, GSR Part 4 missing | EU | High | Index IAEA safety standards for SMR chapters |
| Offshore wind framework (Ustawa offshore 2021) | PL | Medium | Index Polish offshore wind act + GDOŚ/URE guidance |
| BAT principles, best-practice maintenance manuals | All | Medium | Index EU BAT reference documents (BREFs) |
| E-value / energy performance calculations | All | Low | Index EN ISO 52000, national energy performance regs |
| `hybridi` config for SE, NO, PL | SE/NO/PL | Medium | Add country-specific entries in `_COUNTRY_LUVAT` |
| FI EGS dedicated config | FI | Low | Remove `egs` alias to `aurinkovoima`; write EGS-specific config |
| sCO₂ turbine (all countries) | All | Low | Not yet in `_HANKE_CFG`; planned feature |

---

> **Update this file after every RAG indexing session.**
