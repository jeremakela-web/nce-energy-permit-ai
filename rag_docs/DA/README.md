# RAG-dokumentit — Tanska (DA)

Lisää tähän kansioon tanskalaiset viranomaisohjeistukset PDF-muodossa.

## Suositellut dokumentit

### Energistyrelsen — Vindmøller & VE-tilladelser
- **Lähde**: https://ens.dk/ansvarsomraader/vindmoeller
- **Dokumentit**:
  - Vejledning til bekendtgørelse om vindmøller
  - Vejledning om VVM for vindmøller på land
  - Solceller og solvarmeanlæg — vejledning

### Erhvervsstyrelsen — Planloven
- **Lähde**: https://planinfo.erhvervsstyrelsen.dk/planloven
- **Dokumentit**:
  - Vejledning om lokalplaner
  - Vejledning om kommuneplanlægning
  - Planlægning for vindmøller (cirkulære)

### Miljøstyrelsen — Miljøvurdering
- **Lähde**: https://www.mst.dk/erhverv/anlaeg-og-miljoegodkendelser/
- **Dokumentit**:
  - Vejledning om miljøvurdering (VVM)
  - Vejledning om miljøgodkendelse af anlæg
  - Bekendtgørelse om miljøvurdering (BEK nr. 1976/2021)

### Energinet — Nettilslutning
- **Lähde**: https://energinet.dk/el/elmarkedet/nettilslutning/
- **Dokumentit**:
  - Tilslutningsbetingelser for elproducenter
  - Tekniske forskrifter for elproducerende anlæg

### Indenrigsministeriet — Byggeloven
- **Lähde**: https://www.retsinformation.dk/
- **Dokumentit**:
  - Byggeloven (seneste konsoliderede version)
  - Bygningsreglementet BR18 — vejledning

## Tiedostonimikäytäntö

```
energistyrelsen_vindmoeller_vejledning.pdf
erhvervsstyrelsen_planloven_vejledning.pdf
mst_miljoevurdering_vvm_vejledning.pdf
energinet_nettilslutning_betingelser.pdf
byggeloven_br18_vejledning.pdf
```

## Indeksointi

```bash
python3 permit_ai/ingest_countries.py --country DA
```
