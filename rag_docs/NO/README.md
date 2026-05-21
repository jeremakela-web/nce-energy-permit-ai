# RAG-dokumentit — Norja (NO)

Lisää tähän kansioon norjalaiset viranomaisohjeistukset PDF-muodossa.

## Suositellut dokumentit

### NVE — Konsesjonsbehandling
- **Lähde**: https://www.nve.no/konsesjon/
- **Dokumentit**:
  - Vindkraft på land — konsesjonsbehandling (veileder)
  - Solkraft — veileder for konsesjonsprosessen
  - Nettanlegg — konsesjonsbehandling
  - Batterilagringsanlegg — NVE-veileder

### Kommunal- og distriktsdepartementet — Plan- og bygningsloven
- **Lähde**: https://www.regjeringen.no/no/tema/plan-by-og-eiendom/plan-og-bygningsloven/
- **Dokumentit**:
  - Veileder for kommunal planlegging (PBL kap. 11)
  - Veileder for reguleringsplan (PBL kap. 12)
  - Konsekvensutredning — veileder (T-1493)

### DSB — Direktoratet for samfunnssikkerhet og beredskap
- **Lähde**: https://www.dsb.no/lover-og-forskrifter/veiledere/
- **Dokumentit**:
  - Veileder for risikovurdering av batterianlegg (BESS)
  - Brann- og eksplosjonsvernloven veileder
  - Storulykkeforskriften veileder

### Statsforvalteren — Miljøvurdering
- **Lähde**: https://www.statsforvalteren.no/
- **Dokumentit**:
  - Veileder for konsekvensutredning (KU) for vindkraft
  - Miljøvurdering av planer og tiltak

### Olje- og energidepartementet — Energiloven
- **Lähde**: https://www.regjeringen.no/
- **Dokumentit**:
  - Energiloven med forskrifter
  - Veileder for tilknytningsplikt (§ 3-3)

## Tiedostonimikäytäntö

```
nve_vindkraft_konsesjonsbehandling.pdf
nve_solkraft_veileder.pdf
pbl_reguleringsplan_veileder.pdf
dsb_batterianlegg_risikovurdering.pdf
energiloven_med_forskrifter.pdf
```

## Indeksointi

```bash
python3 permit_ai/ingest_countries.py --country NO
```
