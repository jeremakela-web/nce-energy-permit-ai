# RAG-dokumentit — Ruotsi (SE)

Lisää tähän kansioon ruotsalaiset viranomaisohjeistukset PDF-muodossa.

## Suositellut dokumentit

### Boverket — Plan- och bygglag (PBL)
- **Lähde**: https://www.boverket.se/sv/PBL-kunskapsbanken/
- **Dokumentit**:
  - PBL-handbok (byggsanktionsavgifter, lov och anmälan)
  - Vägledning detaljplaneläggning
  - Vägledning bygglov och anmälan

### Energimyndigheten — Vindkraft & solenergi
- **Lähde**: https://www.energimyndigheten.se/fornybart/
- **Dokumentit**:
  - Handbok för vindkraft på land (tillståndsprocessen)
  - Vägledning för solenergi och nätanslutning
  - Elcertifikatsystemet vägledning

### Naturvårdsverket — Miljökonsekvensbeskrivning
- **Lähde**: https://www.naturvardsverket.se/vagledning-och-stod/miljokonsekvensbeskrivningar/
- **Dokumentit**:
  - Vägledning om MKB för planer och program
  - Vägledning om MKB för verksamheter och åtgärder

### Länsstyrelsen — Vindkraft riktlinjer
- **Lähde**: Länsstyrelsernas gemensamma vägledning
- **Dokumentit**:
  - Riktlinjer för prövning av vindkraft (2012:xx)
  - Vindkraft i havet — planeringsunderlag

### Energimarknadsinspektionen (Ei) — Nätkoncessioner
- **Lähde**: https://www.ei.se/sv/for-bransch/elnatsforetag/koncessioner/
- **Dokumentit**:
  - Vägledning för nätkoncession

### Mark- och miljödomstolen
- **Lähde**: https://www.domstol.se/mark-och-miljodomstolen/
- **Dokumentit**:
  - Prövningsordning för vindkraft och solkraft

## Tiedostonimikäytäntö

```
boverket_pbl_handbok_lov.pdf
energimyndigheten_vindkraft_handbok.pdf
naturvardsverket_mkb_vagledning.pdf
lansstyrelsen_vindkraft_riktlinjer.pdf
ei_natkoncession_vagledning.pdf
```

## Indeksointi

```bash
python3 permit_ai/ingest_countries.py --country SE
```
