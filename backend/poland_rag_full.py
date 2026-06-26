"""
Poland full regulatory RAG ingestion — memory-safe version.

Memory constraints on Render (512MB RAM):
  - pdfplumber is called page-by-page via a temp file to avoid parse-tree explosion
  - Models are loaded ONE AT A TIME: v1 first (all sources), then v2 (all sources)
  - gc.collect() after every source and after each model pass
  - psutil memory check before each source; skip if > 400MB used
  - prawo_energetyczne (2.6MB PDF) is excluded from default sources (OOM risk)
    → re-add as LARGE_SOURCES if RAM budget improves or RAM is upgraded

ISAP PDF URLs require requests.Session with browser UA from Render's Frankfurt IP.
HTML sources work from any IP.

Route: POST /api/admin/ingest-poland-full  (x-admin-secret header)
"""
from __future__ import annotations

import gc
import hashlib
import io
import logging
import os
import tempfile
import time
from typing import Any

log = logging.getLogger(__name__)

_DB_PATH  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings")
)
_COL_V1   = "permit_docs"
_COL_V2   = "permit_docs_v2"
_MODEL_V1 = "paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_V2 = "paraphrase-multilingual-mpnet-base-v2"

CHUNK_WORDS    = 800
OVERLAP_WORDS  = 100
MIN_WORDS      = 50
MEM_LIMIT_MB   = 1400  # psutil RSS on Render includes shared libs (~920MB baseline); real limit ~512MB anon

_ISAP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
    "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
    "Referer": "https://isap.sejm.gov.pl/",
}

# ── Sources ────────────────────────────────────────────────────────────────────
# prawo_energetyczne (2.6MB PDF → ~204 chunks) is excluded from defaults —
# it causes OOM on 512MB Render containers. Kept in LARGE_SOURCES below.

POLAND_SOURCES = [
    # ISAP PDFs — confirmed accessible from Render IP (Incapsula blocks local Mac)
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
    # HTML sources — work from any IP
    {
        "name": "poland_gramwzielone_bess_2026",
        "url": "https://www.gramwzielone.pl/magazynowanie-energii/20360841/nowe-przepisy-dla-magazynow-energii-co-sie-zmieni-2026",
        "type": "html",
        "category": "bess_building_permit",
        "description": (
            "Nowe przepisy dla magazynow energii 2026 — Gramwzielone.pl. Prawo budowlane 7.01.2026: "
            "BESS >6.5 kWh wymaga konsultacji z rzeczoznawca ds. pozarnictwa. "
            "Pelna sciezka pozwolenia na budowe dla 5-200 MWh. "
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

# 2.6MB PDF — excluded from defaults due to OOM risk on 512MB Render containers.
# Re-enable by passing LARGE_SOURCES + POLAND_SOURCES if container RAM is upgraded.
LARGE_SOURCES = [
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
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


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
    """Extract text page-by-page via a temp file to limit pdfplumber's parse-tree memory."""
    import pdfplumber
    parts = []
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
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


def _upsert_batched(col: Any, model: Any, ids: list, docs: list, metas: list,
                    batch: int = 16) -> int:
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


# ── Main ───────────────────────────────────────────────────────────────────────

def ingest_poland_sources(sources: list[dict] | None = None) -> int:
    """
    Download, chunk, embed and upsert all Poland sources.
    Returns total v2 chunks upserted.

    Memory strategy:
      1. Download + chunk ALL sources first (no models loaded yet).
      2. Load MiniLM → upsert all into permit_docs → del model → gc.collect().
      3. Load mpnet  → upsert all into permit_docs_v2 → del model → gc.collect().
    Peak RAM = baseline + ONE model + chunk text (~1–2MB) — never two models at once.

    ISAP PDFs require Render's Frankfurt IP (Incapsula blocks local Mac).
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    if sources is None:
        sources = POLAND_SOURCES

    if not os.path.exists(_DB_PATH):
        raise RuntimeError(f"ChromaDB path not found: {_DB_PATH}")

    client = chromadb.PersistentClient(path=_DB_PATH)
    col_v1 = client.get_or_create_collection(_COL_V1, metadata={"hnsw:space": "cosine"})
    col_v2 = client.get_or_create_collection(_COL_V2, metadata={"hnsw:space": "cosine"})

    session     = _make_session()
    ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary: list[dict] = []

    # ── Phase 1: download + chunk all sources (no embedding models in memory) ──
    all_batches: list[tuple[str, list, list, list]] = []

    for src in sources:
        name = src["name"]
        url  = src["url"]
        kind = src["type"]
        print(f"\n[poland_full] {name}", flush=True)
        print(f"  {url[:90]}", flush=True)

        mem = _get_memory_mb()
        print(f"  RAM (RSS): {mem:.0f}MB", flush=True)

        try:
            raw = _download(session, url)
            print(f"  Downloaded {len(raw):,} bytes", flush=True)
        except Exception as exc:
            msg = f"download failed: {exc}"
            print(f"  WARN: {msg} — skipping", flush=True)
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg[:60]})
            gc.collect()
            continue

        try:
            text = _extract_pdf(raw) if kind == "pdf" else _extract_html(raw)
        except Exception as exc:
            msg = f"extraction failed: {exc}"
            print(f"  WARN: {msg} — skipping", flush=True)
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg[:60]})
            gc.collect()
            continue
        finally:
            del raw
            gc.collect()

        word_count = len(text.split())
        if word_count < 200:
            msg = f"too short ({word_count} words)"
            print(f"  WARN: {msg} — skipping", flush=True)
            summary.append({"source": name, "status": "SKIP", "chunks": 0, "reason": msg})
            del text
            gc.collect()
            continue

        chunks = _chunk_text(text)
        del text
        gc.collect()
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
        all_batches.append((name, ids, chunks, metas))
        summary.append({"source": name, "status": "OK", "chunks": len(chunks), "reason": ""})

    print(f"\n[poland_full] Phase 1 done — {len(all_batches)} sources, "
          f"{sum(len(b[2]) for b in all_batches)} total chunks", flush=True)

    if not all_batches:
        print("[poland_full] Nothing to embed.", flush=True)
        return 0

    # ── Phase 2: embed → permit_docs (MiniLM, small model first) ──────────────
    print("\n[poland_full] Phase 2: loading MiniLM for permit_docs …", flush=True)
    gc.collect()
    model_v1 = SentenceTransformer(_MODEL_V1)
    for name, ids, chunks, metas in all_batches:
        try:
            n = _upsert_batched(col_v1, model_v1, ids, chunks, metas)
            print(f"  v1  {n:>4} chunks  {name}", flush=True)
        except Exception as exc:
            print(f"  v1 ERR {name}: {exc}", flush=True)
    del model_v1
    gc.collect()
    print(f"  MiniLM unloaded  (RAM: {_get_memory_mb():.0f}MB)", flush=True)

    # ── Phase 3: embed → permit_docs_v2 (mpnet, larger model) ─────────────────
    print("\n[poland_full] Phase 3: loading mpnet for permit_docs_v2 …", flush=True)
    gc.collect()
    model_v2 = SentenceTransformer(_MODEL_V2)
    total_v2 = 0
    for name, ids, chunks, metas in all_batches:
        try:
            n = _upsert_batched(col_v2, model_v2, ids, chunks, metas)
            total_v2 += n
            print(f"  v2  {n:>4} chunks  {name}", flush=True)
        except Exception as exc:
            print(f"  v2 ERR {name}: {exc}", flush=True)
    del model_v2
    gc.collect()
    print(f"  mpnet unloaded  (RAM: {_get_memory_mb():.0f}MB)", flush=True)

    # Update summary with actual v2 counts
    name_to_v2 = {b[0]: len(b[2]) for b in all_batches}
    for r in summary:
        if r["status"] == "OK":
            r["chunks"] = name_to_v2.get(r["source"], r["chunks"])

    # ── Summary ────────────────────────────────────────────────────────────────
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
    print("NOTE: ISAP PDF sources require Render server IP — local Mac run will fail PDFs.")
    print("NOTE: prawo_energetyczne (2.6MB) excluded by default — add LARGE_SOURCES if RAM allows.")
    count = ingest_poland_sources()
    print(f"\n[poland_full] Done — {count} chunks in permit_docs_v2")
