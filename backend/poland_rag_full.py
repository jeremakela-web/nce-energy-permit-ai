"""
Poland full regulatory RAG ingestion for NCE Permit AI.

Key finding: ISAP sejm.gov.pl PDFs require requests.Session with browser UA
from a non-residential IP (Render Frankfurt works; local Mac is Incapsula-blocked).
HTML sources work from any IP.

Downloads/scrapes all sources, chunks 800 words / 100-word overlap,
upserts into BOTH ChromaDB collections:
  - permit_docs    (MiniLM-L12-v2, 384-dim) — survives reindex
  - permit_docs_v2 (mpnet 768-dim)           — immediately live

Run via API (from Render where ISAP is accessible):
    POST /api/admin/ingest-poland-full  (x-admin-secret header)

Do NOT run locally if ISAP PDFs are needed — Incapsula blocks non-server IPs.
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

# Browser session headers — required for ISAP sejm.gov.pl (Incapsula CDN)
_ISAP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
    "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
    "Referer": "https://isap.sejm.gov.pl/",
}

POLAND_SOURCES = [
    # ── CONFIRMED ISAP PDFs (work from Render IP via requests.Session) ─────────
    # Expected chunks: prawo_energetyczne ~204, ustawa_oos ~61, 10h ~13, rozp ~4
    {
        "name": "poland_prawo_energetyczne_1997",
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970540348/U/D19970348Lj.pdf",
        "type": "pdf",
        "category": "energy_law",
        "description": (
            "Ustawa Prawo energetyczne 1997 tekst jednolity. Koncesje URE: wytwarzanie (WEE), "
            "przesyl, dystrybucja, obrot. Warunki przylaczenia do sieci. Rejestr magazynow energii. "
            "Podstawa prawna dla wszystkich projektow energetycznych w Polsce — BESS, OZE, SMR."
        ),
    },
    {
        "name": "poland_ustawa_oos_2008",
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20081991227/U/D20081227Lj.pdf",
        "type": "pdf",
        "category": "environmental_permit",
        "description": (
            "Ustawa z 3.10.2008 o udostepnianiu informacji o srodowisku i jego ochronie oraz "
            "o ocenach oddzialywania na srodowisko. Podstawa prawna DUS — decyzji o srodowiskowych "
            "uwarunkowaniach. Procedura OOS dla BESS, farm wiatrowych, SMR i innych inwestycji."
        ),
    },
    {
        "name": "poland_ustawa_10h_2016",
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20160000961/U/D20160961Lj.pdf",
        "type": "pdf",
        "category": "wind_energy",
        "description": (
            "Ustawa o inwestycjach w zakresie elektrowni wiatrowych (ustawa 10H) 2016 tekst jednolity. "
            "Minimalna odleglosc turbiny od zabudowy: 10-krotnosc calkowitej wysokosci (min 700m po "
            "nowelizacji 2023). Wymog uchwalenia MPZP. Kluczowe dla wszystkich projektow wiatrowych lad."
        ),
    },
    {
        "name": "poland_rozp_przedsiewziecia_2019",
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20190002093/O/D20192093.pdf",
        "type": "pdf",
        "category": "environmental_permit",
        "description": (
            "Rozporzadzenie RM z 10.09.2019 w sprawie przedsiewziec mogacych znaczaco oddzialywac "
            "na srodowisko. Katalog przedsiewziec Grupy I (zawsze OOS) i Grupy II (potencjalnie OOS). "
            "Progi mocy dla farm wiatrowych, BESS, elektrowni jadrowych i innych instalacji."
        ),
    },
    # ── HTML sources (work from any IP) ────────────────────────────────────────
    {
        "name": "poland_gramwzielone_bess_2026",
        "url": "https://www.gramwzielone.pl/magazynowanie-energii/20360841/nowe-przepisy-dla-magazynow-energii-co-sie-zmieni-2026",
        "type": "html",
        "category": "bess_building_permit",
        "description": (
            "Nowe przepisy dla magazynow energii 2026 — Gramwzielone.pl. Prawo budowlane 7.01.2026: "
            "BESS >6.5 kWh wymaga konsultacji z rzeczoznawca ds. pozarnictwa. "
            "Pelna sciezka pozwolenia na budowe dla 5–200 MWh. "
            "Skrocenie waznosci warunkow przylaczenia do 1 roku od 30.04.2026."
        ),
    },
    {
        "name": "poland_dudkowiak_bess_legal_2025",
        "url": "https://www.dudkowiak.com/blog/battery-energy-storage-in-poland-legal-requirements-and-investment-risks-in-2025/",
        "type": "html",
        "category": "bess_permitting",
        "description": (
            "Dudkowiak & Co: Legal requirements for BESS in Poland 2025. "
            "DUS thresholds (0.5ha protected / 1ha elsewhere), grid connection >50kW, "
            "URE register obligations >50kW, concession >10MW. Investment risks and timelines."
        ),
    },
    {
        "name": "poland_ure_bess_register_przewodnik",
        "url": "https://www.ure.gov.pl/pl/urzad/informacje-ogolne/aktualnosci/11234,Przewodnik-po-rejestrze-magazynow-energii.html",
        "type": "html",
        "category": "bess_storage",
        "description": (
            "URE przewodnik po rejestrze magazynow energii elektrycznej. "
            "Obowiazki rejestracyjne dla instalacji BESS powyzej 50 kW. "
            "Koncesja URE wymagana powyzej 10 MW. Procedura wpisu, terminy, wymagane dokumenty."
        ),
    },
    {
        "name": "poland_ure_bess_wytyczne",
        "url": "https://www.ure.gov.pl/pl/urzad/informacje-ogolne/aktualnosci/11183,Magazyny-energii-elektrycznej-wytyczne-URE.html",
        "type": "html",
        "category": "bess_storage",
        "description": (
            "URE wytyczne dla magazynow energii elektrycznej. "
            "Rejestracja instalacji magazynowania energii elektrycznej — wymogi formalne i techniczne. "
            "Podstawa: Prawo energetyczne art. 43d."
        ),
    },
    {
        "name": "poland_offshorewind_regulacje",
        "url": "https://offshorewindpoland.pl/regulacje/",
        "type": "html",
        "category": "offshore_wind",
        "description": (
            "Offshorewindpoland.pl — regulacje morskiej energetyki wiatrowej w Polsce. "
            "Ustawa offshore 2021, aukcje CfD 2025 (2.5 GW) i 2027 (2.5 GW), "
            "warunki przydomowe, wspolpraca z PSE i Urzedem Morskim."
        ),
    },
    {
        "name": "poland_offshorewind_repowering_2025",
        "url": "https://offshorewindpoland.pl/regulacje-ulatwia-modernizacje-ladowych-farm-wiatrowych-nowe-rozporzadzenie-w-iv-kwartale-2025-roku/",
        "type": "html",
        "category": "wind_repowering",
        "description": (
            "Repowering ladowych farm wiatrowych — nowe rozporzadzenie 2025/2026. "
            "Zwolnienie z DUS przy modernizacji: max +30% mocy, brak nowych turbin, "
            "relokacja max 250m, lacznie max 100 MW. Uproszczona procedura MPZP."
        ),
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session() -> Any:
    import requests
    s = requests.Session()
    s.headers.update(_ISAP_HEADERS)
    return s


def _download(session: Any, url: str, timeout: int = 60) -> bytes:
    resp = session.get(url, timeout=timeout)
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
    h = hashlib.sha256(f"pl_full__{name}__{idx}".encode()).hexdigest()[:10]
    return f"pl_full__{name}__{idx}__{h}"


def _upsert(col: Any, model: Any, ids: list, docs: list, metas: list, batch: int = 32) -> int:
    total = 0
    for i in range(0, len(ids), batch):
        b_ids, b_docs, b_metas = ids[i:i+batch], docs[i:i+batch], metas[i:i+batch]
        embs = model.encode(
            b_docs, batch_size=batch,
            show_progress_bar=False, normalize_embeddings=True,
        )
        col.upsert(ids=b_ids, documents=b_docs, metadatas=b_metas, embeddings=embs.tolist())
        total += len(b_ids)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest_poland_sources(sources: list[dict] | None = None) -> int:
    """
    Download, chunk, embed and upsert all Poland sources.
    Returns total v2 chunks upserted. Logs per source, never raises on partial failure.

    NOTE: ISAP PDF sources require Render's server IP (Incapsula blocks residential/Mac IPs).
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    sources = sources or POLAND_SOURCES

    if not os.path.exists(_DB_PATH):
        raise RuntimeError(f"ChromaDB path not found: {_DB_PATH}")

    log.info("[poland_full] Connecting to ChromaDB at %s", _DB_PATH)
    client  = chromadb.PersistentClient(path=_DB_PATH)
    col_v1  = client.get_or_create_collection(_COL_V1, metadata={"hnsw:space": "cosine"})
    col_v2  = client.get_or_create_collection(_COL_V2, metadata={"hnsw:space": "cosine"})

    log.info("[poland_full] Loading embedding models …")
    model_v1 = SentenceTransformer(_MODEL_V1)
    model_v2 = SentenceTransformer(_MODEL_V2)

    session      = _make_session()
    ingested_at  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_v2     = 0
    summary: list[dict] = []

    for src in sources:
        name = src["name"]
        url  = src["url"]
        kind = src["type"]
        print(f"\n[poland_full] {name}", flush=True)
        print(f"  {url[:90]}", flush=True)

        # Download
        try:
            raw = _download(session, url)
            print(f"  Downloaded {len(raw):,} bytes", flush=True)
        except Exception as exc:
            msg = f"download failed: {exc}"
            log.warning("[poland_full] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping", flush=True)
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg[:60]})
            continue

        # Extract text
        try:
            text = _extract_pdf(raw) if kind == "pdf" else _extract_html(raw)
        except Exception as exc:
            msg = f"text extraction failed: {exc}"
            log.warning("[poland_full] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping", flush=True)
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg[:60]})
            continue

        word_count = len(text.split())
        if word_count < 200:
            msg = f"too short ({word_count} words)"
            log.warning("[poland_full] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping", flush=True)
            summary.append({"source": name, "status": "SKIP", "chunks": 0, "reason": msg})
            continue

        chunks = _chunk_text(text)
        print(f"  {word_count:,} words → {len(chunks)} chunks", flush=True)

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
            print(f"  Upserted {n1} → permit_docs  |  {n2} → permit_docs_v2", flush=True)
            log.info("[poland_full] %s: v1=%d v2=%d", name, n1, n2)
            summary.append({"source": name, "status": "OK", "chunks": n2, "reason": ""})
        except Exception as exc:
            msg = f"upsert failed: {exc}"
            log.warning("[poland_full] %s: %s", name, msg)
            print(f"  ERROR: {msg}", flush=True)
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg[:60]})

    # Summary table
    print(f"\n{'='*72}")
    print(f"{'Source':<46} {'Status':^6} {'Chunks':>6}  Reason")
    print(f"{'-'*72}")
    for r in summary:
        reason = f"  ({r['reason'][:28]})" if r["reason"] else ""
        print(f"{r['source'][:46]:<46} {r['status']:^6} {r['chunks']:>6}{reason}")
    print(f"{'-'*72}")
    print(f"{'TOTAL permit_docs_v2':<46} {'':^6} {total_v2:>6}")

    log.info("[poland_full] Done. Total v2 chunks: %d", total_v2)
    return total_v2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("NOTE: ISAP PDF sources require Render server IP — local run will skip them.")
    count = ingest_poland_sources()
    print(f"\n[poland_full] Done — {count} chunks added to permit_docs_v2")
