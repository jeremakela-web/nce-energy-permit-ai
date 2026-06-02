# NCE Energy — RAG Knowledge Base Coverage Report
**Päivitetty:** 2026-06-02  
**Indeksi:** 5 716 chunkkia · 5 maata · 9 hanketyyppiä  
**Työkalu:** `python3 backend/rag_coverage_report.py`

---

## Executive Summary

> *Lyhyt yhteenveto sijoittajille ja johdolle.*

NCE Permit AI:n tietopohja kattaa **35 % hanketyyppi–maa-yhdistelmistä täysin** (Full-taso ≥5 dokumenttilohkoa). Keskimääräinen Gap Score on **58.9 %**, mikä tarkoittaa, että noin 41 % relevanteista viranomaisohjeista on vielä indeksoimatta.

### Vahvuudet
- **Ruotsi (SE) 72 %** — paras kattavuus; tuuli-, aurinko-, data- ja vesivoima täysin katettu
- **BESS**: katettu FI · SE · PL — kolme päämarkkina-aluetta kunnossa
- **Verkko/liityntä**: FI ja NO täysin; SVK (SE) osittain
- **Tekninen laatu**: 0 metadatavirhettä, 0 liian lyhyttä chunkkia, ei maiden välistä sekaannusta

### Kriittiset aukot
| Maa | Hanketyyppi | Tila | Vaikutus |
|-----|-------------|------|----------|
| FI | Aurinkovoima | ❌ Puuttuu | Suomen suurin kasvava hanketyyppi |
| FI | Tuulivoima | ❌ Puuttuu | Merkittävä lupaprosessi |
| FI | Vesivoima | ❌ Puuttuu | Lupaketju tuntematon |
| NO | BESS | ❌ Puuttuu | Norja strateginen markkina |
| NO | Vesivoima | ❌ Puuttuu | NO:n tärkein energiamuoto |
| SE | SMR/Ydinvoima | ❌ Puuttuu | Ruotsin ydinvoimaohjelma |
| DA | Verkkoliityntä | ❌ Puuttuu | ENS-viranomainen kattamatta |
| PL | Tuulivoima | ❌ Puuttuu | Puolan kasvava sektori |

### Gap Score per maa
| Maa | Gap Score | Taso |
|-----|-----------|------|
| SE | 72.2 % | Hyvä |
| FI | 61.1 % | Kohtalainen |
| PL | 61.1 % | Kohtalainen |
| DA | 50.0 % | Heikko |
| NO | 50.0 % | Heikko |

**Suositeltu toimenpide:** Täydennä FI/Aurinko ja NO/BESS ensin — ne ovat liiketaloudellisesti kriittisimpiä.

---

## Technical Detail

> *Tekninen yksityiskohta kehittäjille ja RAG-ylläpidolle.*

### Coverage Matrix (Full / Partial / Weak / Missing)

```
Hanke           FI      SE      DA      NO      PL
─────────────────────────────────────────────────
BESS            ✅      ✅      ⚡      ❌      ✅
AURINKO         ❌      ✅      ⚡      ⚡      ✅
TUULI           ❌      ✅      ✅      ✅      ❌
SMR/YVA         ✅      ❌      ⚡      ⚡      ✅
DATAKESKUS      ✅      ✅      ⚡      ⚡      ⚡
SCO2            ⚡      ⚡      ⚡      ⚡      ⚡
VESIVOIMA       ❌      ✅      ⚡      ❌      ⚡
YVA/MKB         ✅      ⚡      ⚡      ⚡      ⚡
VERKKO          ✅      ⚡      ❌      ✅      ⚡

✅ Full ≥5ch   ⚡ Partial 2–4ch   ⚠️ Weak 1ch   ❌ Missing 0ch
```

### Indeksin tilastot
| Maa | Chunkkeja | Lähteitä | Gap Score |
|-----|-----------|----------|-----------|
| FI  | 886       | 17       | 61.1 %   |
| SE  | 1 350     | 20       | 72.2 %   |
| DA  | 265       | 18       | 50.0 %   |
| NO  | 1 029     | 26       | 50.0 %   |
| PL  | 2 186     | 61       | 61.1 %   |
| **Yht.** | **5 716** | **142** | **58.9 %** |

> **Huom PL:** 2 186 chunkista ~90 % on IAEA/ydinturvallisuusraportteja (Joint Convention kansalliset raportit). PL:n BESS/aurinko/tuuli-kattavuus on todellisuudessa paljon heikompi kuin Gap Score antaa ymmärtää.

### Puuttuvat kriittiset dokumentit — prioriteettijärjestys

#### KRIITTINEN (0 chunkkia, välitön toimenpide)

| Prioriteetti | Maa/Hanke | Puuttuva dokumentti | Lähde |
|---|---|---|---|
| 1 | FI/AURINKO | Aurinkovoimalan lupaohjeet | Energiavirasto / MML |
| 2 | FI/TUULI | Tuulivoimalan ympäristölupa-ohjeet | ELY-keskus / Luova |
| 3 | FI/VESIVOIMA | Vesilain mukainen lupaprosessi | AVI / SYKE |
| 4 | NO/BESS | NVE batterianlegg søknadsveileder | NVE.no |
| 5 | NO/VESIVOIMA | NVE konsesjonssøknad vannkraft | NVE.no |
| 6 | SE/SMR | SSM kärntillstånd fullständig guide | SSM.se |
| 7 | DA/VERKKO | ENS nettilslutning og systemtilladelse | ENS.dk |
| 8 | PL/TUULI | URE wiatr pozwolenia procedury | URE.gov.pl |

#### OSITTAINEN (2–4 chunkkia, täydennettävä)

| Maa/Hanke | Nykyiset ch | Tavoite | Dokumentti |
|---|---|---|---|
| DA/BESS | 3 | ≥10 | ENS elproduktion over 25 MW lisäohjeet |
| DA/AURINKO | 2 | ≥10 | ENS solenergi tilladelse |
| DA/SMR/YVA | 2 | ≥10 | SIS nukleær regulering |
| DA/DATAKESKUS | 2 | ≥5 | ENS datacenter energikrav |
| DA/SCO2 | 4 | ≥5 | SIK trykbærende udstyr |
| FI/SCO2 | 2 | ≥5 | TUKES painelaitedirektiivi täydennys |
| NO/AURINKO | 4 | ≥10 | NVE solkraft konsesjon |
| NO/SMR/YVA | 3 | ≥10 | DSA nuclear regulatory guide |
| NO/DATAKESKUS | 2 | ≥5 | Statsforvalteren datasenter |
| NO/SCO2 | 3 | ≥5 | DSB trykksatte anlegg |
| SE/BESS | 5 ✅ | — | OK (energilager täynnä) |
| SE/SCO2 | 2 | ≥5 | AV tryckkärl fullständig guide |

### SCO2 — kaikkialla Partial (rakenteellinen aukko)

SCO2-kattavuus on kaikissa maissa vain Partial (2–4 chunkkia), koska:
- Teknologia on uusi — viranomaisohjeet hajanaisina eri säädösten alla (painelaitedirektiivi, kemikaaliasetus)
- Yksittäiset ohjesivu-lataukset tuottavat vain 1–2 lyhyttä chunkkia
- **Toimenpide:** Lataa täydelliset painelaitedirektiivi-PDF:t kaikille maille

### Indeksin tekninen laatu

| Mittari | Arvo | Tila |
|---|---|---|
| Chunkkeja yhteensä | 5 716 | — |
| Chunkkeja ilman metadata | 0 | ✅ |
| Liian lyhyet chunkit (<100ch) | 0 | ✅ |
| Maiden väliset duplikaatit | 0 | ✅ |
| Poistetut PL-duplikaatit | 203 | ✅ Korjattu 2026-06-02 |
| Chunkkien mediaanipituus | 1 500 ch | ✅ |
| Chunkeista täysiä (≥1 500 ch) | 69 % | ✅ |

> **Duplikaatit:** 203 PL-chunkkia poistettu 2026-06-02. Syy: `7th_national_report_to_the_Joint_Convention_(2020)` ja `NATIONAL_REPORT_OF_REPUBLIC_OF_POLAND` jakoivat identtisiä tekstikappaleita (IAEA-vaatimusten boilerplate).

### Seuraavat RAG-kehitystoimenpiteet

1. **FI: Lataa aurinko- ja tuulivoima-ohjeet** (energiavirasto.fi, ymparisto.fi)  
2. **NO/BESS:** NVE:n akkuvarasto-hakuohje (`nve.no/konsesjon/batterianlegg`)  
3. **DA:** Täydennä kaikki Partial-kategoriat ENS.dk-sivulta  
4. **PL:** Lisää URE:n tuulivoima + aurinkovoima -lupaohjeet (`ure.gov.pl`)  
5. **SCO2 kaikki maat:** Lataa kansalliset painelaitedirektiivi-PDF:t  
6. **SE/SMR:** SSM:n ydinlupaohjeet täydellisesti (`ssm.se`)  

---

*Raportti generoitu: `python3 backend/rag_coverage_report.py` · NCE Energy Permit AI*
