"""
Indeksoi kansainväliset viranomaisohjeet ChromaDB-vektoritietokantaan.

Odottaa PDF-tiedostoja kansioissa (projektin juuressa):
    rag_docs/SE/   — Ruotsi   (sv)
    rag_docs/DA/   — Tanska   (da)
    rag_docs/NO/   — Norja    (no)
    rag_docs/PL/   — Puola    (pl)
    rag_docs/EE/   — Viro     (et)
    rag_docs/DE/   — Saksa    (de)

Käyttö:
    # Kaikki maat
    python3 permit_ai/ingest_countries.py

    # Vain yksi tai useampi maa
    python3 permit_ai/ingest_countries.py --country SE NO

    # Näytä mitä tehtäisiin ilman varsinaista kirjoitusta
    python3 permit_ai/ingest_countries.py --dry-run

    # Poista maan chunkit ja indeksoi uudelleen
    python3 permit_ai/ingest_countries.py --country SE --reindex

Metadata jokaisessa chunkissa:
    country  : "SE" | "DA" | "NO" | "PL"
    lang     : "sv" | "da" | "no" | "pl"
    source   : tiedoston kantanimi (ilman .pdf)

EI tyhjennä olemassaolevaa indeksiä — lisää ainoastaan uusia dokumentteja.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Single source of truth for project-type tagging — same as build_index.py
sys.path.insert(0, str(Path(__file__).parent))
from source_policy import get_hanketyyppi_tag as _get_tag

HERE = Path(__file__).parent
ROOT = HERE.parent          # bess_tool/
DB_DIR   = HERE / "embeddings"
# Primary: permit_ai/rag_docs/ (committed to git, available on Render)
# Fallback: bess_tool/rag_docs/ (local-only manual additions)
RAG_ROOT_PRIMARY  = HERE / "rag_docs"
RAG_ROOT_FALLBACK = ROOT / "rag_docs"
RAG_ROOT = RAG_ROOT_PRIMARY  # used only for legacy reference below

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION  = "permit_docs"
CHUNK_CHARS = 1500
OVERLAP     = 200
BATCH       = 64

# Added 2026-09-03: dual-write to permit_docs_v2. This is the active,
# regularly-used ingestion path for all six countries (POST /api/admin/ingest)
# and was writing v1 (MiniLM) only -- invisible to real production retrieval,
# which has queried permit_docs_v2 (mpnet) exclusively since before this gap
# was first found in PR #88's 2026-08-17 audit. That audit's own
# /api/admin/reindex-all-v2 has since caught this file's existing content up
# to v2 (confirmed via a real collection-source-diff call, 2026-09-03: zero
# v1-only sources in production right now) -- but that was a one-time manual
# catch-up, not a standing fix. Without this change, the next real ingest run
# through this file's actual admin endpoint would silently reopen the same
# gap for whatever it adds. Matches the dual-write pattern already used by
# backend/latvia_ingestion.py, lithuania_ingestion.py, poland_ingestion.py,
# caruna_ingestion.py, and permit_ai/ingest_maatalous_vesivoima.py (PR #77).
EMBED_MODEL_V2 = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION_V2  = "permit_docs_v2"

COUNTRY_LANG: dict[str, str] = {
    "SE": "sv",
    "DA": "da",
    "NO": "no",
    "PL": "pl",
    "EE": "et",
    "DE": "de",
}

ALL_COUNTRIES = list(COUNTRY_LANG.keys())


# ── Apufunktiot ───────────────────────────────────────────────────────────────

def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunks.append(text[start:end].strip())
        start += CHUNK_CHARS - OVERLAP
    return [c for c in chunks if len(c) > 100]


def _safe_id(country: str, stem: str, idx: int) -> str:
    """ChromaDB-turvallinen ID: ei erikoismerkkejä, max 512 merkkiä."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)[:60]
    return f"{country}__{safe}__{idx}"


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ── Päälogiikka ───────────────────────────────────────────────────────────────

def ingest(
    countries: list[str],
    dry_run: bool = False,
    reindex: bool = False,
) -> dict[str, int]:
    from sentence_transformers import SentenceTransformer
    import chromadb

    if not DB_DIR.exists():
        print(
            f"[ingest] VIRHE: indeksihakemistoa {DB_DIR} ei ole.\n"
            f"         Aja ensin: python3 permit_ai/build_index.py"
        )
        sys.exit(1)

    print(f"[ingest] Yhdistetään ChromaDB:hen: {DB_DIR}")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION,    metadata={"hnsw:space": "cosine"})
    col_v2 = client.get_or_create_collection(COLLECTION_V2, metadata={"hnsw:space": "cosine"})

    # Models loaded lazily -- a dry-run, or a run where v1/v2 are already
    # fully synced, should not pay the model-load cost.
    model_v1 = model_v2 = None

    def _model_v1():
        nonlocal model_v1
        if model_v1 is None:
            model_v1 = SentenceTransformer(EMBED_MODEL)
        return model_v1

    def _model_v2():
        nonlocal model_v2
        if model_v2 is None:
            model_v2 = SentenceTransformer(EMBED_MODEL_V2)
        return model_v2

    existing_ids:    set[str] = set(col.get()["ids"])
    existing_ids_v2: set[str] = set(col_v2.get()["ids"])
    print(f"[ingest] Olemassaolevia chunkkeja: v1={len(existing_ids)}  v2={len(existing_ids_v2)}")

    totals: dict[str, int] = {}

    for country in countries:
        # Check primary (permit_ai/rag_docs/, committed to git) then fallback (bess_tool/rag_docs/)
        country_dir = RAG_ROOT_PRIMARY / country
        if not country_dir.exists():
            country_dir = RAG_ROOT_FALLBACK / country
        if not country_dir.exists():
            print(f"\n[{country}] Kansio puuttuu molemmista sijainneista — ohitetaan")
            totals[country] = 0
            continue

        # Combine files from both locations (dedup by stem)
        seen_stems: set[str] = set()
        all_files: list[tuple] = []
        for search_root in [RAG_ROOT_PRIMARY / country, RAG_ROOT_FALLBACK / country]:
            if not search_root.exists():
                continue
            for p in sorted(search_root.rglob("*.pdf")):
                if p.stem not in seen_stems:
                    all_files.append((p, "pdf"))
                    seen_stems.add(p.stem)
            for t in sorted(search_root.rglob("*.txt")):
                if t.stem not in seen_stems:
                    all_files.append((t, "txt"))
                    seen_stems.add(t.stem)
        if not all_files:
            print(f"\n[{country}] Ei tiedostoja kansiossa {country_dir}")
            totals[country] = 0
            continue

        n_pdf = sum(1 for _, ft in all_files if ft == "pdf")
        n_txt = sum(1 for _, ft in all_files if ft == "txt")
        print(f"\n[{country}] Löytyi {n_pdf} PDF:ää, {n_txt} TXT:tä")
        lang = COUNTRY_LANG[country]

        # Poista vanhat chunkit jos --reindex -- mirrored to both collections
        # so a reindex can't leave v1/v2 out of sync with each other.
        if reindex and not dry_run:
            old = [id_ for id_ in existing_ids if id_.startswith(f"{country}__")]
            if old:
                col.delete(ids=old)
                existing_ids -= set(old)
                print(f"  [reindex] Poistettu {len(old)} vanhaa chunkkia (v1)")
            old_v2 = [id_ for id_ in existing_ids_v2 if id_.startswith(f"{country}__")]
            if old_v2:
                col_v2.delete(ids=old_v2)
                existing_ids_v2 -= set(old_v2)
                print(f"  [reindex] Poistettu {len(old_v2)} vanhaa chunkkia (v2)")

        all_ids:   list[str]  = []
        all_docs:  list[str]  = []
        all_metas: list[dict] = []

        for fpath, ftype in all_files:
            try:
                if ftype == "pdf":
                    text = _read_pdf(fpath)
                else:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                chunks = _chunk(text)
                for i, chunk in enumerate(chunks):
                    all_ids.append(_safe_id(country, fpath.stem, i))
                    all_docs.append(chunk)
                    all_metas.append({
                        "country":         country,
                        "lang":            lang,
                        "source":          fpath.stem,
                        "hanketyyppi_tag": _get_tag(fpath.stem),
                    })
                print(f"  {fpath.name}: {len(chunks)} chunkkia")
            except Exception as exc:
                print(f"  VIRHE {fpath.name}: {exc}")

        # Two independent "new" sets -- v1 and v2 can be out of sync in
        # either direction. Most commonly: content ingested before this
        # dual-write existed is v1-complete but missing from v2 (the exact
        # gap PR #88 found and this change closes for good going forward).
        def _new_for(existing: set[str]) -> tuple[list[str], list[str], list[dict]]:
            ids, docs, metas = [], [], []
            for id_, doc, meta in zip(all_ids, all_docs, all_metas):
                if id_ not in existing:
                    ids.append(id_)
                    docs.append(doc)
                    metas.append(meta)
            return ids, docs, metas

        new_ids,    new_docs,    new_metas    = _new_for(existing_ids)
        new_ids_v2, new_docs_v2, new_metas_v2 = _new_for(existing_ids_v2)

        if not new_ids and not new_ids_v2:
            print(f"  → Kaikki chunkit jo indeksoitu (v1 ja v2)")
            totals[country] = 0
            continue

        if dry_run:
            print(f"  DRY-RUN: v1: {len(new_ids)} uutta  |  v2: {len(new_ids_v2)} uutta")
            totals[country] = len(new_ids)
            continue

        if new_ids:
            print(f"  Lisätään {len(new_ids)} chunkkia -> permit_docs (v1)...")
            model = _model_v1()
            for i in range(0, len(new_ids), BATCH):
                b_docs  = new_docs[i : i + BATCH]
                b_ids   = new_ids[i : i + BATCH]
                b_metas = new_metas[i : i + BATCH]
                embs    = model.encode(b_docs, show_progress_bar=False).tolist()
                col.add(documents=b_docs, embeddings=embs, ids=b_ids, metadatas=b_metas)
                pct = min(100, (i + len(b_docs)) * 100 // len(new_ids))
                print(f"  {i + len(b_docs)}/{len(new_ids)} ({pct}%)")
            existing_ids.update(new_ids)

        if new_ids_v2:
            print(f"  Lisätään {len(new_ids_v2)} chunkkia -> permit_docs_v2 (mpnet)...")
            model2 = _model_v2()
            for i in range(0, len(new_ids_v2), BATCH):
                b_docs  = new_docs_v2[i : i + BATCH]
                b_ids   = new_ids_v2[i : i + BATCH]
                b_metas = new_metas_v2[i : i + BATCH]
                embs    = model2.encode(b_docs, show_progress_bar=False, normalize_embeddings=True).tolist()
                col_v2.add(documents=b_docs, embeddings=embs, ids=b_ids, metadatas=b_metas)
                pct = min(100, (i + len(b_docs)) * 100 // len(new_ids_v2))
                print(f"  {i + len(b_docs)}/{len(new_ids_v2)} ({pct}%)")
            existing_ids_v2.update(new_ids_v2)

        totals[country] = len(new_ids)
        print(f"  ✅ v1: {len(new_ids)} chunkkia  |  v2: {len(new_ids_v2)} chunkkia  ({country})")
        # Force WAL checkpoint so data lands in the main DB file before process exits
        _wal_checkpoint(DB_DIR)

    # Final checkpoint after all countries
    _wal_checkpoint(DB_DIR)

    # Yhteenveto
    print(f"\n{'─'*50}")
    print(f"Yhteenveto:")
    grand = 0
    for c in ALL_COUNTRIES:
        n = totals.get(c, 0)
        print(f"  {c}: {n} uutta chunkkia")
        grand += n
    print(f"  Yhteensä uutta: {grand}")
    print(f"  Koko indeksi:   v1={col.count()}  v2={col_v2.count()} chunkkia")
    print(f"{'─'*50}")

    return totals


def _wal_checkpoint(db_dir: Path) -> None:
    """Force SQLite WAL checkpoint so chunks survive process exit / container restart."""
    import sqlite3 as _sql
    db_file = db_dir / "chroma.sqlite3"
    if not db_file.exists():
        return
    try:
        con = _sql.connect(str(db_file), check_same_thread=False)
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        print("  [WAL] checkpoint OK")
    except Exception as e:
        print(f"  [WAL] checkpoint epäonnistui: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indeksoi kansainväliset RAG-dokumentit ChromaDB:hen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--country",
        nargs="+",
        choices=ALL_COUNTRIES + ["ALL"],
        default=["ALL"],
        metavar="CC",
        help="Maa/maat (SE DA NO PL) tai ALL (oletus)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Näytä mitä tehtäisiin — ei kirjoita indeksiin",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Poista maan vanhat chunkit ennen lisäystä",
    )
    args = parser.parse_args()

    target = ALL_COUNTRIES if "ALL" in args.country else args.country
    ingest(target, dry_run=args.dry_run, reindex=args.reindex)
