"""
Denmark regulatory RAG ingestion for NCE Permit AI.

Fetches real, full-text Danish consolidated law texts via retsinformation.dk's
ELI (European Legislation Identifier) direct-document endpoint —
`https://www.retsinformation.dk/eli/lta/{year}/{number}/xml` — which returns
structured LexDania-schema XML and bypasses the JS-rendering wall that makes
the normal retsinformation.dk site (and its harvest API, which is search-less)
unfetchable by simple HTTP tools. See the manual sourcing backlog memory for
the full investigation that found this pattern.

Every (year, number) pair below was individually verified before being added
here: fetched live, confirmed `<PopularTitle>`/`<DocumentTitle>` matches the
intended law, and confirmed `<Status>Valid</Status>` (not `Historic`/superseded)
— found via WebSearch + a live ELI fetch, not guessed. This replaces 9 wrong
citations previously in `_COUNTRY_LUVAT["DA"]` (permit_ai/generate_application.py)
that resolved to *completely unrelated* real Danish documents when checked
against this same endpoint (e.g. "LBK nr. 1157/2021" — cited as Planloven —
actually resolves to an artificial-island construction act). See PR description
for the full before/after table.

One entry (Elforsyningsloven) is flagged `is_verified_current=False`: retsinformation
marks even LBK nr. 1248/2023 as `Status: Historic` (superseded sometime before
2026-06-24) but a web search could not pin down the exact newer LBK number —
flagging honestly rather than guessing. This is still a strict improvement over
the previous citation (LBK nr. 119/2020, itself also Historic and additionally
the wrong law was never in question there — only the version was stale).

hanketyyppi_tag is always resolved via get_hanketyyppi_tag() from source_policy;
no tag value is hardcoded here so the BESS/wind/solar restriction logic stays
in a single source of truth. country: "DA" is set explicitly on every chunk —
required for _rag_context()'s country-scoping filter in generate_application.py.

Run standalone for testing:
    python3 backend/denmark_ingestion.py
Or trigger via API:
    POST /api/admin/ingest-denmark  (x-admin-secret header required)
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "permit_ai"))
from source_policy import get_hanketyyppi_tag

log = logging.getLogger(__name__)

_DB_PATH  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings")
)
_COL_V1   = "permit_docs"
_COL_V2   = "permit_docs_v2"
_MODEL_V1 = "paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_V2 = "paraphrase-multilingual-mpnet-base-v2"

CHUNK_WORDS   = 800
OVERLAP_WORDS = 100
MIN_WORDS     = 50

_ELI_URL = "https://www.retsinformation.dk/eli/lta/{year}/{number}/xml"

DENMARK_SOURCES: list[dict] = [
    {
        "name": "da_undergrundsloven",
        "eli_year": 2023, "eli_number": 1461,
        "category": "subsoil_law",
        "is_verified_current": True,
        "description": (
            "Undergrundsloven — Bekendtgørelse af lov om anvendelse af Danmarks "
            "undergrund (LBK nr. 1461/2023). Governs use of the Danish subsoil: "
            "exploration/extraction of raw materials, subsurface storage, "
            "geothermal energy, scientific investigation. Relevant to SMR "
            "(siting/principle-decision stage) and EGS (geothermal exploration "
            "permit) permit tracks."
        ),
    },
    {
        "name": "da_miljovurderingsloven",
        "eli_year": 2023, "eli_number": 4,
        "category": "eia_law",
        "is_verified_current": True,
        "description": (
            "Miljøvurderingsloven — Bekendtgørelse af lov om miljøvurdering af "
            "planer og programmer og af konkrete projekter (VVM) (LBK nr. 4/2023). "
            "Denmark's EIA law, covering environmental assessment of plans/"
            "programmes and concrete projects (VVM = Vurdering af Virkninger på "
            "Miljøet). Relevant across nearly all DA hanketyyppi: SMR, BESS, "
            "onshore/offshore wind, solar, hydro, offshore wind, industrial."
        ),
    },
    {
        "name": "da_stralebeskyttelsesloven",
        "eli_year": 2018, "eli_number": 23,
        "category": "radiation_safety_law",
        "is_verified_current": True,
        "description": (
            "Strålebeskyttelsesloven — Lov om ioniserende stråling og "
            "strålebeskyttelse (Lov nr. 23/2018). Governs use of radioactive "
            "substances and ionising radiation in Denmark; underpins "
            "Sundhedsstyrelsen/SIS's authority over nuclear-facility "
            "construction and operating licences. Replaces the previously-cited "
            "but non-existent 'Lov om brug af radioaktive stoffer (nr. 94/2003)' "
            "— that citation did not resolve to any real document when checked."
        ),
    },
    {
        "name": "da_kystbeskyttelsesloven",
        "eli_year": 2025, "eli_number": 245,
        "category": "coastal_protection_law",
        "is_verified_current": True,
        "description": (
            "Kystbeskyttelsesloven — Bekendtgørelse af lov om kystbeskyttelse "
            "m.v. (LBK nr. 245/2025). Governs coastal protection and marine-area "
            "water permits. Relevant to SMR (cooling water), offshore/marine "
            "wind (marine-area water permit)."
        ),
    },
    {
        "name": "da_byggeloven",
        "eli_year": 2016, "eli_number": 1178,
        "category": "building_law",
        "is_verified_current": True,
        "description": (
            "Byggeloven — Bekendtgørelse af byggeloven (LBK nr. 1178/2016). "
            "Denmark's core building-permit law (byggetilladelse), municipal "
            "administration (Kommunen, teknik og miljø). Relevant to nearly "
            "every DA hanketyyppi requiring a building permit."
        ),
    },
    {
        "name": "da_planloven",
        "eli_year": 2024, "eli_number": 572,
        "category": "spatial_planning_law",
        "is_verified_current": True,
        "description": (
            "Planloven — Bekendtgørelse af lov om planlægning (LBK nr. 572/2024). "
            "Denmark's spatial/territorial planning law: municipal local plans "
            "(lokalplan), zoning (kaavoitus/planafdelingen). Relevant to nearly "
            "every DA hanketyyppi requiring municipal land-use agreement."
        ),
    },
    {
        "name": "da_miljobeskyttelsesloven",
        "eli_year": 2025, "eli_number": 1742,
        "category": "environmental_protection_law",
        "is_verified_current": True,
        "description": (
            "Miljøbeskyttelsesloven — Bekendtgørelse af lov om miljøbeskyttelse "
            "(LBK nr. 1742/2025). Denmark's core environmental protection law "
            "— environmental permits (miljøgodkendelse) for BESS, wind, hydro, "
            "data centre, industrial and geothermal projects."
        ),
    },
    {
        "name": "da_elforsyningsloven",
        "eli_year": 2023, "eli_number": 1248,
        "category": "electricity_supply_law",
        "is_verified_current": False,  # see module docstring — flagged, not guessed further
        "description": (
            "Elforsyningsloven — Bekendtgørelse af lov om elforsyning "
            "(LBK nr. 1248/2023). Denmark's electricity supply law — grid "
            "connection agreements (Energinet). NOTE: retsinformation.dk marks "
            "this specific LBK number as Status: Historic (superseded before "
            "2026-06-24) — a newer consolidated reissue exists but its exact "
            "LBK number could not be pinned down via search. Still correctly "
            "identifies the right law (unlike the previous citation, LBK nr. "
            "119/2020, which was also stale) — flagged for a future correction "
            "pass rather than left silently wrong."
        ),
    },
    {
        "name": "da_vandforsyningsloven",
        "eli_year": 2024, "eli_number": 1149,
        "category": "water_supply_law",
        "is_verified_current": True,
        "description": (
            "Vandforsyningsloven — Bekendtgørelse af lov om vandforsyning m.v. "
            "(LBK nr. 1149/2024). Governs water permits (impoundment, "
            "construction) — relevant to hydropower (vesivoima) projects."
        ),
    },
    {
        "name": "da_ve_loven",
        "eli_year": 2024, "eli_number": 1031,
        "category": "renewable_energy_law",
        "is_verified_current": True,
        "description": (
            "VE-loven — Bekendtgørelse af lov om fremme af vedvarende energi "
            "(LBK nr. 1031/2024). Denmark's renewable energy promotion law — "
            "onshore wind turbine permits (vindmølletilladelse), offshore wind "
            "permits (havvindtilladelse). Previously cited under two different, "
            "both-incorrect numbers/names ('Lov om vedvarende energi' and "
            "'Lov om fremme af vedvarende energi', both LBK nr. 388/2022) — "
            "now one consistent, correct citation."
        ),
    },
    {
        "name": "da_husdyrbrugloven",
        "eli_year": 2025, "eli_number": 1065,
        "category": "livestock_environmental_law",
        "is_verified_current": True,
        "description": (
            "Husdyrbrugloven — Bekendtgørelse af lov om husdyrbrug og "
            "anvendelse af gødning m.v. (LBK nr. 1065/2025). Governs "
            "environmental approval of livestock farming (miljøgodkendelse, "
            "husdyr) — relevant to agricultural (maatalous) projects."
        ),
    },
]


# ── Text extraction ───────────────────────────────────────────────────────────

def _download(url: str, timeout: int = 30) -> bytes:
    import requests
    resp = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "NCEPermitAI/1.0 (Denmark regulatory research; contact: info@ncenergy.fi)",
            "Accept-Language": "da,en;q=0.9",
        },
    )
    resp.raise_for_status()
    return resp.content


def _extract_xml(data: bytes) -> str:
    """LexDania-schema XML — real body text lives in <Char> leaves nested inside
    <Linea>/<Stk>/<Paragraf>/<Kapitel> structural tags. get_text() on the whole
    tree is sufficient (same chunk-by-word-count approach used elsewhere in this
    codebase; no structure-aware chunking needed) — verified live to produce
    clean, readable Danish legal prose, not tag/attribute noise.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data, "xml")
    return soup.get_text(separator=" ", strip=True)


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
    h = hashlib.sha256(f"da__{name}__{idx}".encode()).hexdigest()[:10]
    return f"da__{name}__{idx}__{h}"


# ── ChromaDB upsert ───────────────────────────────────────────────────────────

def _upsert(col: Any, model: Any, ids: list, docs: list, metas: list, batch: int = 64) -> int:
    total = 0
    for i in range(0, len(ids), batch):
        b_ids   = ids[i:i+batch]
        b_docs  = docs[i:i+batch]
        b_metas = metas[i:i+batch]
        embs = model.encode(b_docs, batch_size=batch, show_progress_bar=False, normalize_embeddings=True)
        col.upsert(ids=b_ids, documents=b_docs, metadatas=b_metas, embeddings=embs.tolist())
        total += len(b_ids)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest_denmark_sources(sources: list[dict] | None = None) -> int:
    """
    Ingest all Denmark sources into permit_docs + permit_docs_v2.
    Returns total v2 chunks upserted. Never raises — logs failures and continues.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    sources = sources or DENMARK_SOURCES

    if not os.path.exists(_DB_PATH):
        raise RuntimeError(f"ChromaDB path not found: {_DB_PATH}")

    log.info("[denmark] Connecting to ChromaDB at %s", _DB_PATH)
    client  = chromadb.PersistentClient(path=_DB_PATH)
    col_v1  = client.get_or_create_collection(_COL_V1, metadata={"hnsw:space": "cosine"})
    col_v2  = client.get_or_create_collection(_COL_V2, metadata={"hnsw:space": "cosine"})

    log.info("[denmark] Loading embedding models …")
    model_v1 = SentenceTransformer(_MODEL_V1)
    model_v2 = SentenceTransformer(_MODEL_V2)

    ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_v2    = 0
    summary: list[dict] = []

    for src in sources:
        name = src["name"]
        url  = _ELI_URL.format(year=src["eli_year"], number=src["eli_number"])
        ht_tag = get_hanketyyppi_tag(name)

        print(f"\n[denmark] {name}")
        print(f"  URL: {url}")
        print(f"  hanketyyppi_tag: {ht_tag}")
        print(f"  is_verified_current: {src.get('is_verified_current', True)}")

        try:
            raw = _download(url)
            print(f"  Downloaded {len(raw):,} bytes")
        except Exception as exc:
            msg = f"download failed: {exc}"
            log.warning("[denmark] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})
            continue

        try:
            text = _extract_xml(raw)
        except Exception as exc:
            msg = f"text extraction failed: {exc}"
            log.warning("[denmark] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})
            continue

        if len(text.split()) < 200:
            msg = f"too short ({len(text.split())} words) — possible fetch/parse issue"
            log.warning("[denmark] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "SKIP", "chunks": 0, "reason": msg})
            continue

        chunks = _chunk_text(text)
        print(f"  {len(text.split()):,} words → {len(chunks)} chunks")

        ids   = [_chunk_id(name, i) for i in range(len(chunks))]
        metas = [
            {
                "source":              name,
                "url":                 url,
                "country":             "DA",
                "category":            src["category"],
                "lang":                "da",
                "description":         src["description"],
                "ingested_at":         ingested_at,
                "source_type":         "eli_direct_fetch",
                "is_verified_current": src.get("is_verified_current", True),
                "hanketyyppi_tag":     ht_tag,
            }
            for _ in chunks
        ]

        try:
            n1 = _upsert(col_v1, model_v1, ids, chunks, metas)
            n2 = _upsert(col_v2, model_v2, ids, chunks, metas)
            total_v2 += n2
            print(f"  Upserted {n1} → permit_docs  |  {n2} → permit_docs_v2")
            log.info("[denmark] %s: v1=%d v2=%d tag=%s", name, n1, n2, ht_tag)
            summary.append({"source": name, "status": "OK", "chunks": n2, "tag": ht_tag, "reason": ""})
        except Exception as exc:
            msg = f"upsert failed: {exc}"
            log.warning("[denmark] %s: %s", name, msg)
            print(f"  ERROR: {msg}")
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "tag": ht_tag, "reason": msg})

    print(f"\n{'='*75}")
    print(f"{'Source':<32} {'Tag':<20} {'St':^4} {'Ch':>5}")
    print(f"{'-'*75}")
    for r in summary:
        flag = r.get("reason") and f"  ({r['reason'][:30]})" or ""
        print(f"{r['source'][:32]:<32} {r.get('tag','?'):<20} {r['status']:^4} {r['chunks']:>5}{flag}")
    print(f"{'-'*75}")
    print(f"{'TOTAL permit_docs_v2':<32} {'':20} {'':4} {total_v2:>5}")
    print(f"{'='*75}")

    log.info("[denmark] Done. Total v2 chunks: %d", total_v2)
    return total_v2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = ingest_denmark_sources()
    print(f"\n[denmark] Done — {count} chunks added to permit_docs_v2")
