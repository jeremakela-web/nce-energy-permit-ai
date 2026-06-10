# NCE Permit AI — Coverage Matrix

Generated: 2026-06-10  
Source: `_COUNTRY_LUVAT`, `_HANKE_CFG`, `_COUNTRY_COVERAGE` in `permit_ai/generate_application.py` and `backend/static/index.html`.

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Permit entries defined in `_COUNTRY_LUVAT` / `_HANKE_CFG` + adequate RAG context |
| ⚠️ | Defined but partial — low RAG chunk count, aliased config, or only some phases tested |
| ❌ | Not defined / not covered |

## RAG Chunk Counts per Country

| Country | Chunks | Level |
|---------|--------|-------|
| 🇫🇮 FI | ~5 000+ | Full (Fingrid, STUK YVL, Luova, Tukes, Pelastuslaki, MRL) |
| 🇸🇪 SE | ~1 560 | Partial (Energimyndigheten, Mark- och miljödomstolen, Elsäkerhetsverket) |
| 🇩🇰 DA | ~470 | Partial — low coverage (Energistyrelsen, Planloven) |
| 🇳🇴 NO | ~1 270 | Partial (NVE, Energidepartementet, Plan- og bygningsloven) |
| 🇵🇱 PL | ~2 190 | Partial (UDT, URE, Ustawa OZE, Prawo budowlane) |
| 🇩🇪 DE | ~920 | Partial (BImSchG, EEG, WHG, BauGB, BetrSichV) |

## Coverage Matrix

| Project type | 🇫🇮 FI | 🇸🇪 SE | 🇩🇰 DA | 🇳🇴 NO | 🇵🇱 PL | 🇩🇪 DE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BESS** (akkuvarasto) | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Tuulivoima — maa** (onshore) | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Tuulivoima — meri** (offshore) | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Aurinkovoima** (solar) | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **SMR** (pienydinreaktori) | ✅ | ✅ (smr_se) | ⚠️ (smr_da) | ✅ (smr_no) | ✅ | ✅ (smr_de) |
| **SMR + BESS** (hybridi ydin+akku) | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Hybridivoimala** (BESS+tuuli/aurinko) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Vesivoima** (hydropower) | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Datakeskus** | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Asuinrakennus** (residential) | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Teollisuus** (industrial) | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Maatalous** (agriculture) | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Liikerakennus** (commercial) | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **EGS** (geothermal / enhanced) | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Offshore Wind** (floating) | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **sCO₂ turbine** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Ympäristölupa** (YSL 527/2014) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

## Notes

### FI
- Full RAG: Finnish regulatory database is the primary training corpus.
- `asuinrakennus`, `teollisuus`, `maatalous`, `liikerakennus` use generic configs in `_HANKE_CFG` with limited RAG depth compared to energy project types — marked ⚠️.
- `egs` is currently aliased to the `aurinkovoima` base config (limited EGS-specific guidance).
- `ymparistolupa` is FI-only (YSL 527/2014).

### SE (~1 560 chunks)
- Country-specific SMR variant: `smr_se` (Kärntillstånd / MKB under Miljöbalken).
- `hybridi` not yet defined in `_COUNTRY_LUVAT["SE"]` — falls through to FI base config, unreliable.
- `offshore_wind` aliased to `tuulivoima_meri`.

### DA (~470 chunks)
- Lowest RAG coverage of all active countries — all project types marked ⚠️.
- Country-specific SMR variant: `smr_da` (Bekendtgørelse om nukleare anlæg).
- Priority: ingest more Danish regulatory documents (Energistyrelsen, Natur- og Miljøklagenævnet).

### NO (~1 270 chunks)
- Country-specific SMR variant: `smr_no` (NVE konsesjon / Atomenergiloven).
- `hybridi` not defined for NO — falls through to FI base config.

### PL (~2 190 chunks)
- Best non-FI RAG coverage.
- `hybridi` not defined for PL — falls through to FI base config.
- `smr_bess` defined; standalone SMR uses generic SMR config.

### DE (~920 chunks — BImSchG / EEG / WHG)
- Most complete `_COUNTRY_LUVAT` entry: all 15 types including `hybridi` (added 2026-06-09).
- Country-specific SMR variant: `smr_de` (AtG-Genehmigung, BMUV).
- `offshore_wind` uses WindSeeG (BSH) permits.
- `egs` uses BBergG Betriebsplanzulassung.

## Missing Coverage / Gaps

| Gap | Priority | Action needed |
|-----|----------|---------------|
| DA RAG depth (~470 chunks) | High | Ingest Energistyrelsen, Miljøstyrelsen, Planloven docs |
| SE `hybridi` in `_COUNTRY_LUVAT` | Medium | Add SE hybridi entry (BImSchG equivalent: MB + koncession) |
| NO `hybridi` in `_COUNTRY_LUVAT` | Medium | Add NO hybridi entry (Energiloven + Plan- og bygningsloven) |
| PL `hybridi` in `_COUNTRY_LUVAT` | Medium | Add PL hybridi entry (Prawo energetyczne + Prawo budowlane) |
| sCO₂ turbine (all countries) | Low | Feature planned — not yet in `_HANKE_CFG` |
| FI EGS dedicated config | Low | Remove `egs` alias to `aurinkovoima`; write FI-specific EGS config |
