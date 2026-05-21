# RAG-dokumentit — Puola (PL)

Lisää tähän kansioon puolalaiset viranomaisohjeistukset PDF-muodossa.

## Suositellut dokumentit

### PAA — Państwowa Agencja Atomistyki (ydinenergia)
- **Lähde**: https://www.paa.gov.pl/
- **Dokumentit**:
  - Wytyczne dotyczące pre-licencjonowania SMR
  - Wymagania dla wniosku o zezwolenie na budowę obiektu jądrowego
  - Procedura udzielania zezwoleń — poradnik

### URE — Urząd Regulacji Energetyki (OZE-luvat)
- **Lähde**: https://www.ure.gov.pl/
- **Dokumentit**:
  - Poradnik dla wytwórców energii z OZE — warunki przyłączenia
  - Wzory wniosków o koncesję na wytwarzanie energii
  - Prawo energetyczne — komentarz

### GDOŚ — Generalna Dyrekcja Ochrony Środowiska (ympäristövaikutukset)
- **Lähde**: https://www.gdos.gov.pl/ocena-oddzialywania-na-srodowisko
- **Dokumentit**:
  - Wytyczne do przeprowadzania OOŚ (ocena oddziaływania na środowisko)
  - Poradnik dla wnioskodawców — raport OOŚ
  - Dyrektywa OOŚ — implementacja w Polsce

### GUNB — Główny Urząd Nadzoru Budowlanego (rakennusluvat)
- **Lähde**: https://www.gunb.gov.pl/
- **Dokumentit**:
  - Prawo budowlane (tekst jednolity)
  - Vademecum inwestora budowlanego — pozwolenie na budowę
  - Wymagania dla farm fotowoltaicznych i wiatrowych

### PSE — Polskie Sieci Elektroenergetyczne (verkkoon liittyminen)
- **Lähde**: https://www.pse.pl/
- **Dokumentit**:
  - Instrukcja Ruchu i Eksploatacji Sieci Przesyłowej (IRiESP)
  - Warunki przyłączenia do sieci przesyłowej

### Ministerstwo Klimatu i Środowiska
- **Lähde**: https://www.gov.pl/web/klimat
- **Dokumentit**:
  - Ustawa o odnawialnych źródłach energii (ustawa OZE, Dz.U. 2015 poz. 478)
  - Ustawa o inwestycjach w zakresie elektrowni wiatrowych (10H, Dz.U. 2016 poz. 961)
  - Polityka energetyczna Polski do 2040 roku (PEP2040)

## Tiedostonimikäytäntö

```
paa_pre_licencjonowanie_smr_wytyczne.pdf
ure_oze_warunki_przylaczenia.pdf
gdos_oos_poradnik_wnioskodawcy.pdf
gunb_prawo_budowlane_tekst_jednolity.pdf
pse_iriesp_warunki_przylaczenia.pdf
ustawa_oze_dzu2015.pdf
ustawa_wiatrowa_10h_dzu2016.pdf
```

## Indeksointi

```bash
python3 permit_ai/ingest_countries.py --country PL
```
