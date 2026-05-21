# RAG-dokumentit — kansainväliset viranomaisohjeet

Tähän hakemistoon lisätään manuaalisesti PDF-dokumentit maakohtaisiin kansioihin.
Dokumentit indeksoidaan ChromaDB-vektoritietokantaan `ingest_countries.py`-skriptillä.

## Hakemistorakenne

```
rag_docs/
├── SE/   — Ruotsi   (lang: sv)
├── DA/   — Tanska   (lang: da)
├── NO/   — Norja    (lang: no)
└── PL/   — Puola    (lang: pl)
```

## Indeksointi

```bash
# Kaikki maat
python3 permit_ai/ingest_countries.py

# Vain yksi maa
python3 permit_ai/ingest_countries.py --country SE

# Testaa ilman kirjoitusta
python3 permit_ai/ingest_countries.py --dry-run

# Poista vanhat ja indeksoi uudelleen
python3 permit_ai/ingest_countries.py --country SE --reindex
```

## Huomiot

- PDF:t eivät kuulu git-repoon (`.gitignore`)
- `ingest_countries.py` EI tyhjennä olemassaolevaa indeksiä
- Suomalaiset FI-dokumentit ovat `permit_ai/docs/` -kansiossa
- FI-indeksi rakennetaan erikseen: `python3 permit_ai/build_index.py`

---

## Suositellut dokumentit per maa

Katso maakohtaiset README-tiedostot:
- [SE/README.md](SE/README.md)
- [DA/README.md](DA/README.md)
- [NO/README.md](NO/README.md)
- [PL/README.md](PL/README.md)
