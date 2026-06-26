"""
Caruna network capacity PDF ingestion for NCE Permit AI RAG.

Downloads public Caruna PDFs, extracts text with pdfplumber, chunks by word count,
and upserts into both ChromaDB collections:
  - permit_docs    (MiniLM-L12-v2, 384-dim) — v1, so chunks survive future reindex
  - permit_docs_v2 (mpnet 768-dim)           — active production collection

Run standalone for testing:
    python3 backend/caruna_ingestion.py
Or trigger via API:
    POST /api/admin/ingest-caruna  (x-admin-secret header required)
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
import time
from typing import Any

log = logging.getLogger(__name__)

# Persistent ChromaDB path — matches main.py _DB_PATH
_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings")
)

_COL_V1    = "permit_docs"
_COL_V2    = "permit_docs_v2"
_MODEL_V1  = "paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_V2  = "paraphrase-multilingual-mpnet-base-v2"

CHUNK_WORDS = 800
OVERLAP_WORDS = 100

CARUNA_SOURCES = [
    {
        "name": "caruna_network_development_plan_2026",
        "url": (
            "https://caruna.fi/sites/default/files/docs/"
            "Jakeluverkon%20kehitt%C3%A4missuunnitelma%20Caruna%20Oy%202026.pdf"
        ),
        "country": "FI",
        "category": "grid_capacity",
        "language": "fi",
        "description": (
            "Caruna Oy verkkoalueen kehittamissuunnitelma 2026. "
            "Kapasiteettitiedot, liitantamahdollisuudet, kehitysvyohykkeet."
        ),
    },
    {
        "name": "caruna_high_voltage_capacity_2025",
        "url": (
            "https://caruna.fi/sites/default/files/docs/"
            "Suurjannitteisen_jakeluverkon_kapasiteetti.pdf"
        ),
        "country": "FI",
        "category": "grid_capacity",
        "language": "fi",
        "description": (
            "Carunan suurjanniteisen jakeluverkon laskennallinen vapaa kapasiteetti. "
            "Paivitetty 30.6.2025."
        ),
    },
]


def _download_pdf(url: str) -> bytes:
    """Download URL, return raw bytes. Raises on HTTP error."""
    import requests
    resp = requests.get(url, timeout=60, headers={"User-Agent": "NCEPermitAI/1.0"})
    resp.raise_for_status()
    return resp.content


def _extract_text(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes using pdfplumber."""
    import pdfplumber
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-count chunks. Skips chunks under 50 words."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end]).strip()
        if len(chunk.split()) >= 50:
            chunks.append(chunk)
        start += chunk_words - overlap
    return chunks


def _chunk_id(source_name: str, idx: int) -> str:
    """Stable, collision-resistant chunk ID."""
    h = hashlib.sha256(f"caruna__{source_name}__{idx}".encode()).hexdigest()[:10]
    return f"caruna__{source_name}__{idx}__{h}"


def _upsert_to_collection(
    col: Any,
    model: Any,
    ids: list[str],
    docs: list[str],
    metas: list[dict],
    batch: int = 64,
) -> int:
    """Embed and upsert in batches. Returns count upserted."""
    total = 0
    for i in range(0, len(ids), batch):
        b_ids   = ids[i:i + batch]
        b_docs  = docs[i:i + batch]
        b_metas = metas[i:i + batch]
        embs = model.encode(
            b_docs, batch_size=batch,
            show_progress_bar=False, normalize_embeddings=True,
        )
        col.upsert(
            ids=b_ids,
            documents=b_docs,
            metadatas=b_metas,
            embeddings=embs.tolist(),
        )
        total += len(b_ids)
    return total


def ingest_caruna_sources(sources: list[dict] | None = None) -> int:
    """
    Download, chunk, embed and upsert all Caruna sources.
    Returns total chunk count upserted across both collections.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    sources = sources or CARUNA_SOURCES

    if not os.path.exists(_DB_PATH):
        raise RuntimeError(f"ChromaDB path not found: {_DB_PATH}")

    log.info("[caruna] Connecting to ChromaDB at %s", _DB_PATH)
    client = chromadb.PersistentClient(path=_DB_PATH)

    col_v1 = client.get_or_create_collection(_COL_V1, metadata={"hnsw:space": "cosine"})
    col_v2 = client.get_or_create_collection(_COL_V2, metadata={"hnsw:space": "cosine"})

    log.info("[caruna] Loading embedding models …")
    model_v1 = SentenceTransformer(_MODEL_V1)
    model_v2 = SentenceTransformer(_MODEL_V2)

    ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_upserted = 0

    for src in sources:
        name = src["name"]
        url  = src["url"]
        log.info("[caruna] Downloading: %s", name)
        print(f"[caruna] Downloading: {name}")

        try:
            pdf_bytes = _download_pdf(url)
        except Exception as exc:
            log.warning("[caruna] Download failed for %s: %s", name, exc)
            print(f"  ERROR download: {exc}")
            continue

        print(f"  Downloaded {len(pdf_bytes):,} bytes — extracting text …")
        try:
            text = _extract_text(pdf_bytes)
        except Exception as exc:
            log.warning("[caruna] Text extraction failed for %s: %s", name, exc)
            print(f"  ERROR extract: {exc}")
            continue

        if not text.strip():
            print(f"  WARN: no text extracted from {name}")
            continue

        chunks = _chunk_text(text)
        print(f"  {len(text):,} chars → {len(chunks)} chunks")

        ids   = [_chunk_id(name, i) for i in range(len(chunks))]
        metas = [
            {
                "source":       name,
                "country":      src["country"],
                "category":     src["category"],
                "lang":         src["language"],
                "description":  src["description"],
                "ingested_at":  ingested_at,
                "source_type":  "pdf",
            }
            for _ in chunks
        ]

        # Upsert into v1 (MiniLM — survives future reindex from v1 → v2)
        n1 = _upsert_to_collection(col_v1, model_v1, ids, chunks, metas)
        # Upsert into v2 (mpnet — immediately queryable in production)
        n2 = _upsert_to_collection(col_v2, model_v2, ids, chunks, metas)
        total_upserted += n2
        print(f"  Upserted {n1} chunks → permit_docs, {n2} chunks → permit_docs_v2")
        log.info("[caruna] %s: upserted %d v1 + %d v2 chunks", name, n1, n2)

    log.info("[caruna] Done. Total v2 chunks upserted: %d", total_upserted)
    return total_upserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = ingest_caruna_sources()
    print(f"\n[caruna] Total chunks upserted to permit_docs_v2: {count}")
