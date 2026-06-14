"""
Hae ja indeksoi IAEA-turvallisuusstandardit ChromaDB-vektoritietokantaan.

Lähde-PDFit (public domain, www-pub.iaea.org):
  SSR-2/1 Rev.1 — Safety of Nuclear Power Plants: Design          (Pub1682)
  SSG-52        — Design of Reactor Containment Systems           (Pub1946)
  NS-R-5 Rev.1  — Safety of Nuclear Fuel Cycle Facilities         (Pub1641)

Metadata jokaisessa chunkissa:
    country     : "EU"
    lang        : "en"
    source      : "IAEA_SSR-2_1" | "IAEA_SSG-52" | "IAEA_NS-R-5"
    url         : PDF URL
    source_type : "pdf"

Käyttö:
    python3.9 permit_ai/ingest_iaea.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DB_DIR = HERE / "embeddings"

EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION  = "permit_docs_v2"
CHUNK_CHARS = 2000
OVERLAP     = 200
BATCH       = 32

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NCE-RAG-Ingest/1.0)"}

DOCS = [
    {
        "code":   "SSR-2_1",
        "label":  "SSR-2/1 Rev.1 — Safety of Nuclear Power Plants: Design",
        "url":    "https://www-pub.iaea.org/MTCD/Publications/PDF/Pub1682_web.pdf",
        "source": "IAEA_SSR-2_1",
    },
    {
        "code":   "SSG-52",
        "label":  "SSG-52 — Design of Reactor Containment Systems",
        "url":    "https://www-pub.iaea.org/MTCD/Publications/PDF/Pub1946_web.pdf",
        "source": "IAEA_SSG-52",
    },
    {
        "code":   "NS-R-5",
        "label":  "NS-R-5 Rev.1 — Safety of Nuclear Fuel Cycle Facilities",
        "url":    "https://www-pub.iaea.org/MTCD/Publications/PDF/Pub1641_web.pdf",
        "source": "IAEA_NS-R-5",
    },
]


def _fetch_pdf(url: str) -> bytes | None:
    import requests
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
        if r.status_code == 200 and "pdf" in r.headers.get("content-type", ""):
            return r.content
        print(f"  HTTP {r.status_code}: {url}")
    except Exception as exc:
        print(f"  fetch error ({exc.__class__.__name__}): {url}")
    return None


def _extract_text_pypdf(data: bytes) -> list[tuple[int, str]]:
    """Return list of (page_number, page_text) from PDF bytes."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Normalise whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()
        if text:
            pages.append((i + 1, text))
    return pages


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunk = text[start:end].strip()
        if len(chunk) > 150:
            chunks.append(chunk)
        start += CHUNK_CHARS - OVERLAP
    return chunks


def _safe_id(source: str, page: int, idx: int) -> str:
    h = hashlib.md5(f"{source}:{page}:{idx}".encode()).hexdigest()[:8]
    return f"iaea__{source}__{page}__{idx}__{h}"


def ingest_iaea(dry_run: bool = False) -> None:
    import requests
    from sentence_transformers import SentenceTransformer
    import chromadb

    if not DB_DIR.exists():
        print(f"[ingest_iaea] ERROR: {DB_DIR} missing. Run build_index.py first.")
        sys.exit(1)

    print(f"[ingest_iaea] Connecting to ChromaDB: {DB_DIR}")
    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION)

    existing_ids: set[str] = set(col.get()["ids"])
    print(f"[ingest_iaea] Existing chunks: {len(existing_ids)}")
    print()

    grand_new = 0

    for doc in DOCS:
        source = doc["source"]
        url    = doc["url"]
        label  = doc["label"]

        # Check if already indexed (any ID with this source prefix)
        already = sum(1 for id_ in existing_ids if id_.startswith(f"iaea__{source}__"))
        print(f"[{source}] {label}")
        print(f"  Already indexed: {already} chunks")

        if already > 0:
            print(f"  → Skipping (already present)\n")
            continue

        print(f"  Fetching: {url}")
        pdf_data = _fetch_pdf(url)
        if not pdf_data:
            print(f"  → FAILED to fetch, skipping\n")
            continue
        print(f"  Downloaded: {len(pdf_data):,} bytes")

        pages = _extract_text_pypdf(pdf_data)
        print(f"  Pages with text: {len(pages)}")

        new_docs:  list[str]  = []
        new_ids:   list[str]  = []
        new_metas: list[dict] = []

        for page_no, page_text in pages:
            for idx, chunk in enumerate(_chunk(page_text)):
                id_ = _safe_id(source, page_no, idx)
                if id_ in existing_ids:
                    continue
                new_docs.append(chunk)
                new_ids.append(id_)
                new_metas.append({
                    "country":     "EU",
                    "lang":        "en",
                    "source":      source,
                    "url":         url,
                    "source_type": "pdf",
                })

        print(f"  New chunks: {len(new_docs)}")

        if not new_docs:
            print(f"  → Nothing to add\n")
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

    print(f"{'─'*55}")
    print(f"Summary (IAEA ingest):")
    print(f"  New chunks added:  {grand_new}")
    print(f"  Total index size:  {col.count()}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index IAEA safety standards into ChromaDB")
    parser.add_argument("--dry-run", action="store_true", help="Show chunk counts, do not write")
    args = parser.parse_args()
    ingest_iaea(dry_run=args.dry_run)
