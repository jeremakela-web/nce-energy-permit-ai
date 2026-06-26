"""
Poland regulatory RAG ingestion for NCE Permit AI.

Downloads public Polish legal PDFs and HTML pages, extracts text,
chunks by word count, and upserts into both ChromaDB collections:
  - permit_docs    (MiniLM-L12-v2, 384-dim) — v1, survives future reindex
  - permit_docs_v2 (mpnet 768-dim)           — active production collection

Run standalone for testing:
    python3 backend/poland_ingestion.py
Or trigger via API:
    POST /api/admin/ingest-poland  (x-admin-secret header required)
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

_DB_PATH   = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings")
)
_COL_V1   = "permit_docs"
_COL_V2   = "permit_docs_v2"
_MODEL_V1 = "paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_V2 = "paraphrase-multilingual-mpnet-base-v2"

CHUNK_WORDS   = 800
OVERLAP_WORDS = 100
MIN_WORDS     = 50

POLAND_SOURCES = [
    {
        "name": "poland_dus_environmental_decision_ustawa_2008",
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20081991227/U/D20081227Lj.pdf",
        "type": "pdf",
        "category": "environmental_permit",
        "description": (
            "Ustawa z 3 pazdziernika 2008 r. o udostepnianiu informacji o srodowisku. "
            "Podstawa prawna DUS - decyzji o srodowiskowych uwarunkowaniach dla projektow energetycznych."
        ),
    },
    {
        "name": "poland_rzadzenie_przedsiewziecia_2019",
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20190002093/O/D20192093.pdf",
        "type": "pdf",
        "category": "environmental_permit",
        "description": (
            "Rozporzadzenie RM z 10 wrzesnia 2019 r. w sprawie przedsiewziec mogacych znaczaco "
            "oddzialywac na srodowisko. Katalog projektow wymagajacych DUS (Grupa I i II)."
        ),
    },
    {
        "name": "poland_ustawa_energetyczna_2024",
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970540348/U/D19970348Lj.pdf",
        "type": "pdf",
        "category": "energy_law",
        "description": (
            "Ustawa Prawo energetyczne (tekst jednolity). "
            "Reguluje koncesje URE, rejestr magazynow energii, warunki przylaczenia do sieci."
        ),
    },
    {
        "name": "poland_ure_bess_register_guide_2025",
        "url": "https://www.ure.gov.pl/pl/urzad/informacje-ogolne/aktualnosci/11234,Przewodnik-po-rejestrze-magazynow-energii.html",
        "type": "html",
        "category": "energy_storage_register",
        "description": (
            "URE przewodnik po rejestrze magazynow energii elektrycznej 2025. "
            "Obowiazki rejestracyjne dla BESS powyzej 50 kW, koncesja URE powyzej 10 MW."
        ),
    },
    {
        "name": "poland_pse_grid_connection_rules_2025",
        "url": "https://www.pse.pl/obszary-dzialalnosci/krajowy-system-elektroenergetyczny/przylaczanie-do-sieci",
        "type": "html",
        "category": "grid_connection",
        "description": (
            "PSE zasady przylaczania do sieci przesylowej 2025. "
            "Nowa metodologia od 1 sierpnia 2025. Grid Act marzec 2026 — nowe zasady przylaczania OZE i magazynow energii."
        ),
    },
    {
        "name": "poland_offshore_wind_act_2025",
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20210001873/U/D20211873Lj.pdf",
        "type": "pdf",
        "category": "offshore_wind",
        "description": (
            "Ustawa o promowaniu wytwarzania energii elektrycznej w morskich farmach wiatrowych. "
            "Nowelizacja pazdziernik 2025. Aukcje 2025 (2.5 GW) i 2027 (2.5 GW)."
        ),
    },
    {
        "name": "poland_wind_energy_distance_act_2023",
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20160000961/U/D20160961Lj.pdf",
        "type": "pdf",
        "category": "wind_energy",
        "description": (
            "Ustawa o inwestycjach w zakresie elektrowni wiatrowych (ustawa 10H). "
            "Zasada odleglosci 10H od zabudowan, zmieniona w 2023 na 700m minimum."
        ),
    },
    {
        "name": "poland_bess_permitting_guide_2025",
        "url": "https://www.dudkowiak.com/blog/battery-energy-storage-in-poland-legal-requirements-and-investment-risks-in-2025/",
        "type": "html",
        "category": "bess_permitting",
        "description": (
            "Legal requirements for BESS in Poland 2025: zoning, DUS environmental permit thresholds "
            "(0.5ha protected / 1ha elsewhere), grid connection >50kW, URE register, concession >10MW."
        ),
    },
    {
        "name": "poland_building_law_amendment_2026_bess",
        "url": "https://www.gramwzielone.pl/magazynowanie-energii/20360841/nowe-przepisy-dla-magazynow-energii-co-sie-zmieni-2026",
        "type": "html",
        "category": "bess_building_permit",
        "description": (
            "Nowe przepisy dla magazynow energii 2026. Prawo budowlane 7 stycznia 2026: "
            "BESS >6.5 kWh wymaga konsultacji z rzeczoznawca ds. pozarnictwa. "
            "Pelna sciezka pozwolenia na budowe dla 5-200 MWh. "
            "Skrocenie waznosci warunkow przylaczenia do 1 roku od 30 kwietnia 2026."
        ),
    },
    {
        "name": "poland_repowering_exemption_dus_2025",
        "url": "https://offshorewindpoland.pl/regulacje-ulatwia-modernizacje-ladowych-farm-wiatrowych-nowe-rozporzadzenie-w-iv-kwartale-2025-roku/",
        "type": "html",
        "category": "wind_repowering",
        "description": (
            "Nowe rozporzadzenie repowering 2025/2026. Zwolnienie z DUS dla modernizacji turbin wiatrowych: "
            "max +30% mocy, brak nowych turbin, relokacja max 250m, lacznie max 100 MW."
        ),
    },
]


# ── Text extraction ───────────────────────────────────────────────────────────

def _download(url: str, timeout: int = 30) -> bytes:
    import requests
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "NCEPermitAI/1.0"})
    resp.raise_for_status()
    return resp.content


def _extract_pdf(data: bytes) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def _extract_html(data: bytes) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start: start + CHUNK_WORDS]).strip()
        if len(chunk.split()) >= MIN_WORDS:
            chunks.append(chunk)
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def _chunk_id(name: str, idx: int) -> str:
    h = hashlib.sha256(f"pl__{name}__{idx}".encode()).hexdigest()[:10]
    return f"pl__{name}__{idx}__{h}"


# ── ChromaDB upsert ───────────────────────────────────────────────────────────

def _upsert(col: Any, model: Any, ids: list, docs: list, metas: list, batch: int = 64) -> int:
    total = 0
    for i in range(0, len(ids), batch):
        b_ids, b_docs, b_metas = ids[i:i+batch], docs[i:i+batch], metas[i:i+batch]
        embs = model.encode(b_docs, batch_size=batch, show_progress_bar=False, normalize_embeddings=True)
        col.upsert(ids=b_ids, documents=b_docs, metadatas=b_metas, embeddings=embs.tolist())
        total += len(b_ids)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest_poland_sources(sources: list[dict] | None = None) -> int:
    """
    Ingest all Poland sources into permit_docs + permit_docs_v2.
    Returns total v2 chunks upserted. Never raises — logs failures and continues.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    sources = sources or POLAND_SOURCES

    if not os.path.exists(_DB_PATH):
        raise RuntimeError(f"ChromaDB path not found: {_DB_PATH}")

    log.info("[poland] Connecting to ChromaDB at %s", _DB_PATH)
    client  = chromadb.PersistentClient(path=_DB_PATH)
    col_v1  = client.get_or_create_collection(_COL_V1, metadata={"hnsw:space": "cosine"})
    col_v2  = client.get_or_create_collection(_COL_V2, metadata={"hnsw:space": "cosine"})

    log.info("[poland] Loading embedding models …")
    model_v1 = SentenceTransformer(_MODEL_V1)
    model_v2 = SentenceTransformer(_MODEL_V2)

    ingested_at    = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_v2       = 0
    summary: list[dict] = []

    for src in sources:
        name = src["name"]
        url  = src["url"]
        kind = src["type"]
        print(f"\n[poland] {name}")
        print(f"  URL: {url[:80]}")

        # Download
        try:
            raw = _download(url)
            print(f"  Downloaded {len(raw):,} bytes")
        except Exception as exc:
            msg = f"download failed: {exc}"
            log.warning("[poland] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})
            continue

        # Extract text
        try:
            text = _extract_pdf(raw) if kind == "pdf" else _extract_html(raw)
        except Exception as exc:
            msg = f"text extraction failed: {exc}"
            log.warning("[poland] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})
            continue

        if len(text.split()) < 200:
            msg = f"too short ({len(text.split())} words)"
            log.warning("[poland] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "SKIP", "chunks": 0, "reason": msg})
            continue

        chunks = _chunk_text(text)
        print(f"  {len(text.split()):,} words → {len(chunks)} chunks")

        ids   = [_chunk_id(name, i) for i in range(len(chunks))]
        metas = [
            {
                "source":      name,
                "url":         url,
                "country":     "PL",
                "category":    src["category"],
                "lang":        "pl",
                "description": src["description"],
                "ingested_at": ingested_at,
                "source_type": kind,
            }
            for _ in chunks
        ]

        try:
            n1 = _upsert(col_v1, model_v1, ids, chunks, metas)
            n2 = _upsert(col_v2, model_v2, ids, chunks, metas)
            total_v2 += n2
            print(f"  Upserted {n1} → permit_docs  |  {n2} → permit_docs_v2")
            log.info("[poland] %s: v1=%d v2=%d", name, n1, n2)
            summary.append({"source": name, "status": "OK", "chunks": n2, "reason": ""})
        except Exception as exc:
            msg = f"upsert failed: {exc}"
            log.warning("[poland] %s: %s", name, msg)
            print(f"  ERROR: {msg}")
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Source':<48} {'Status':^6} {'Chunks':>6}")
    print(f"{'-'*70}")
    for r in summary:
        flag = r["reason"] and f"  ({r['reason'][:40]})" or ""
        print(f"{r['source'][:48]:<48} {r['status']:^6} {r['chunks']:>6}{flag}")
    print(f"{'-'*70}")
    print(f"{'TOTAL permit_docs_v2':<48} {'':^6} {total_v2:>6}")

    log.info("[poland] Done. Total v2 chunks: %d", total_v2)
    return total_v2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = ingest_poland_sources()
    print(f"\n[poland] Done — {count} chunks added to permit_docs_v2")
