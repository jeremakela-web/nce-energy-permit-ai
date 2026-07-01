"""
Indeksoi datakeskus-lupasisältö ChromaDB-kokoelmaan (permit_docs).

Lisää vain tiedostot permit_ai/docs/datakeskus/ -kansiosta sekä
ym_datakeskukset.txt ja datakeskus_luvat_suomi.txt.
Tarkistaa olemassa olevat ID:t — ei ota uudelleen jo indeksoituja chunkkeja.

Käyttö:
    python3 permit_ai/ingest_datakeskus.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).parent
DB_DIR = HERE / "embeddings"

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION  = "permit_docs"
CHUNK_CHARS = 1500
OVERLAP     = 200
BATCH       = 32

# Specific files to ingest (relative to permit_ai/)
TARGET_FILES = [
    HERE / "docs" / "datakeskus" / "bios_datakeskus_sijoittamislupa.txt",
    HERE / "docs" / "datakeskus" / "microsoft_espoo_yva_selostus.txt",
    HERE / "docs" / "datakeskus" / "rakentamislaki_sijoittamislupa_datakeskus.txt",
    HERE / "docs" / "datakeskus" / "ymparistolupa_datakeskus_ysl.txt",
    HERE / "docs" / "ym_datakeskukset.txt",
    HERE / "docs" / "datakeskus_luvat_suomi.txt",
]


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunk = text[start:end].strip()
        if len(chunk) > 150:
            chunks.append(chunk)
        start += CHUNK_CHARS - OVERLAP
    return chunks


def _safe_id(stem: str, idx: int) -> str:
    h = hashlib.md5(f"dc__{stem}__{idx}".encode()).hexdigest()[:8]
    return f"dc__{stem}__{idx}__{h}"


def ingest(dry_run: bool = False) -> None:
    from sentence_transformers import SentenceTransformer
    import chromadb

    if not DB_DIR.exists():
        print(f"[ingest_datakeskus] ERROR: {DB_DIR} missing — run build_index.py first.")
        sys.exit(1)

    print(f"[ingest_datakeskus] Connecting to ChromaDB: {DB_DIR}")
    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    existing_ids: set[str] = set(col.get()["ids"])
    print(f"[ingest_datakeskus] Existing chunks: {len(existing_ids)}")
    print()

    grand_new = 0

    for path in TARGET_FILES:
        if not path.exists():
            print(f"  SKIP (missing): {path}")
            continue

        stem  = path.stem
        text  = path.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk(text)

        already = sum(1 for id_ in existing_ids if id_.startswith(f"dc__{stem}__"))
        print(f"[{stem}]")
        print(f"  Size: {len(text):,} chars → {len(chunks)} chunks | Already indexed: {already}")

        new_docs:  list[str]  = []
        new_ids:   list[str]  = []
        new_metas: list[dict] = []

        for i, chunk in enumerate(chunks):
            id_ = _safe_id(stem, i)
            if id_ in existing_ids:
                continue
            new_docs.append(chunk)
            new_ids.append(id_)
            new_metas.append({
                "country":         "FI",
                "lang":            "fi",
                "source":          stem,
                "source_type":     "text",
                "hanketyyppi_tag": "datakeskus",
            })

        print(f"  New chunks: {len(new_docs)}")

        if not new_docs:
            print(f"  → Already fully indexed, skipping\n")
            continue

        if dry_run:
            print(f"  DRY-RUN: would add {len(new_docs)} chunks\n")
            grand_new += len(new_docs)
            continue

        print(f"  Embedding and writing to ChromaDB…")
        for i in range(0, len(new_docs), BATCH):
            b = slice(i, i + BATCH)
            embs = model.encode(new_docs[b], show_progress_bar=False).tolist()
            col.add(
                documents=new_docs[b],
                embeddings=embs,
                ids=new_ids[b],
                metadatas=new_metas[b],
            )
            pct = min(100, (i + len(new_docs[b])) * 100 // len(new_docs))
            print(f"  {i + len(new_docs[b])}/{len(new_docs)} ({pct}%)")

        existing_ids.update(new_ids)
        grand_new += len(new_docs)
        print(f"  ✅ {len(new_docs)} chunks added\n")

    print("─" * 55)
    print(f"Summary:")
    print(f"  New chunks added: {grand_new}")
    print(f"  Total index size: {col.count()}")
    print("─" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ingest(dry_run=args.dry_run)
