"""
Rakenna ChromaDB-vektori-indeksi permit_ai/docs/ -kansion PDF-tiedostoista.

Indeksoi kaikki FI-dokumentit metadatalla country="FI", lang="fi".
Kansainväliset dokumentit (SE/DA/NO/PL) lisätään erikseen:
    python3 permit_ai/ingest_countries.py

Käyttö:
    python3 permit_ai/build_index.py

Ajetaan automaattisesti Render-buildissa (render.yaml buildCommand).
"""

import os
import shutil
import sys
from pathlib import Path

HERE     = Path(__file__).parent
DB_DIR   = HERE / "embeddings"
DOCS_DIR = HERE / "docs"

EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION   = "permit_docs"
CHUNK_CHARS  = 1500   # merkkiä per chunkkia (≈300 sanaa)
OVERLAP      = 200    # päällekkäisyys chunks välillä
BATCH        = 64     # embedding-batch koko


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunks.append(text[start:end].strip())
        start += CHUNK_CHARS - OVERLAP
    return [c for c in chunks if len(c) > 100]


def build(force: bool = False) -> None:
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer
    import chromadb

    # Persistent-disk safety: if an index already exists and force=False, skip rmtree.
    # This prevents wiping a pre-seeded disk on every deploy restart or startup fallback.
    if not force and DB_DIR.exists():
        uuid_dirs = [c for c in DB_DIR.iterdir() if c.is_dir()]
        if uuid_dirs:
            print(
                f"[build_index] Existing index found ({DB_DIR}, "
                f"{len(uuid_dirs)} segment(s)) — skipping rebuild. "
                f"Call build(force=True) or delete {DB_DIR} to force a fresh index."
            )
            return

    pdfs = sorted(DOCS_DIR.rglob("*.pdf"))
    txts = sorted(DOCS_DIR.rglob("*.txt"))
    if not pdfs and not txts:
        print(f"[build_index] Ei PDF- tai TXT-tiedostoja hakemistossa {DOCS_DIR}")
        sys.exit(1)

    print(f"[build_index] Löytyi {len(pdfs)} PDF:ää, {len(txts)} TXT-tiedostoa")

    # Tyhjennetään vanha FI-indeksi ja rakennetaan uudelleen
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    DB_DIR.mkdir(parents=True)

    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    all_docs:  list[str]  = []
    all_ids:   list[str]  = []
    all_metas: list[dict] = []

    for pdf in pdfs:
        try:
            reader = PdfReader(str(pdf))
            text   = "\n".join(p.extract_text() or "" for p in reader.pages)
            chunks = _chunk(text)
            for i, chunk in enumerate(chunks):
                all_docs.append(chunk)
                all_ids.append(f"{pdf.name}_{i}")
                all_metas.append({
                    "country": "FI",
                    "lang":    "fi",
                    "source":  pdf.stem,
                })
            print(f"  {pdf.name}: {len(chunks)} chunkkia")
        except Exception as exc:
            print(f"  VIRHE {pdf.name}: {exc}")

    for txt in txts:
        try:
            text   = txt.read_text(encoding="utf-8", errors="replace")
            chunks = _chunk(text)
            for i, chunk in enumerate(chunks):
                all_docs.append(chunk)
                all_ids.append(f"{txt.name}_{i}")
                all_metas.append({
                    "country": "FI",
                    "lang":    "fi",
                    "source":  txt.stem,
                })
            print(f"  {txt.name}: {len(chunks)} chunkkia")
        except Exception as exc:
            print(f"  VIRHE {txt.name}: {exc}")

    if not all_docs:
        print("[build_index] Ei tekstiä indeksoitavaksi — tarkista PDF:t")
        sys.exit(1)

    print(f"[build_index] Lisätään {len(all_docs)} chunkkia ChromaDB:hen...")
    for i in range(0, len(all_docs), BATCH):
        batch_docs  = all_docs[i : i + BATCH]
        batch_ids   = all_ids[i : i + BATCH]
        batch_metas = all_metas[i : i + BATCH]
        embeddings  = model.encode(batch_docs, show_progress_bar=False).tolist()
        col.add(
            documents=batch_docs,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metas,
        )
        pct = min(100, (i + len(batch_docs)) * 100 // len(all_docs))
        print(f"  {i + len(batch_docs)}/{len(all_docs)} ({pct}%)")

    print(f"[build_index] ✅ Valmis — {col.count()} chunkkia tallennettu: {DB_DIR}")
    print(f"[build_index]    Kansainväliset dokumentit: python3 permit_ai/ingest_countries.py")


if __name__ == "__main__":
    build()
