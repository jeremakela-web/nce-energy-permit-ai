# NCE Energy — Lupa-asiakirjapalvelu

AI-pohjainen lupa-asiakirjojen generointipalvelu ja maankäyttöselvitystyökalu energiahankkeille.

## Pikaopas (lokaali kehitys)

```bash
# 1. Asenna riippuvuudet
pip install -r backend/requirements.txt

# 2. Aseta ympäristömuuttujat
export ANTHROPIC_API_KEY="sk-ant-..."
export MML_API_KEY="..."          # Ks. alla

# 3. Käynnistä
cd backend && uvicorn main:app --reload --port 8000
```

Avaa selaimella: http://localhost:8000

## Ympäristömuuttujat

| Muuttuja | Pakollinen | Kuvaus |
|---|---|---|
| `ANTHROPIC_API_KEY` | Kyllä | Claude API -avain (claude.ai / console.anthropic.com) |
| `MML_API_KEY` | Suositellaan | Maanmittauslaitoksen WFS-avain maankäyttöselvitykseen |

### MML_API_KEY hankkiminen

1. Mene osoitteeseen https://www.maanmittauslaitos.fi/rajapinnat/api-avaimen-hallinta
2. Kirjaudu sisään tai luo tili
3. Luo uusi API-avain palvelulle **"Kiinteistötietojärjestelmä (KTJ)"** ja **"Maastotiedot WFS"**
4. Kopioi avain ympäristömuuttujaan

> **Huom:** Ilman MML_API_KEY:tä maankäyttöselvitys ja kiinteistörajapalvelu eivät toimi. Lupahakemusgeneraattori toimii silti normaalisti.

## Render-deployaus

### Uusi palvelu

1. **New → Web Service** Render-dashboardissa
2. Yhdistä GitHub-repositorio
3. Asetukset:
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3

### Ympäristömuuttujat Renderissä

Mene **Environment**-välilehdelle ja lisää:

```
ANTHROPIC_API_KEY = sk-ant-...
MML_API_KEY       = ...
```

> Renderissä `PORT`-muuttuja asetetaan automaattisesti — ei tarvitse lisätä manuaalisesti.

### Jatkuva deployaus

Render deployaa automaattisesti joka kerta kun `main`-haara päivittyy GitHubissa.

## Arkkitehtuuri

```
bess_tool/
├── backend/
│   ├── main.py              # FastAPI-backend, API-endpointit
│   ├── report.py            # Maankäyttöselvitys-PDF (ReportLab)
│   ├── mml_api.py           # MML WFS -kyselyt
│   ├── fingrid_api.py       # Fingrid-verkkodata
│   └── static/
│       └── index.html       # Single-page frontend
└── permit_ai/
    ├── generate_application.py  # Lupahakemusgeneraattori (RAG + Claude)
    ├── chroma_db/               # RAG-vektoritietokanta (ChromaDB)
    └── docs/                    # Lähdedokumentit (PDF)
```

## RAG-tietokanta

Lupa-asiakirjapalvelu käyttää RAG-hakua (Retrieval-Augmented Generation). Tietokanta sisältää ~8 000 chunkkia lupa- ja rakennusohjeistuksista — 6 maa (FI, SE, DA, NO, PL, DE), 7 kieltä (FI, EN, SE, DA, NO, PL, DE), 20+ hanketyyppiä.

Tietokannan uudelleenindeksointi:
```bash
cd permit_ai && python3 index_docs.py
```
