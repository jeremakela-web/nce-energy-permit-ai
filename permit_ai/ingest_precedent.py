"""
Index regulatory precedent decisions and approval-criteria documents
(case law, BAT principles, EIA guidance, BIM standards, noise standards)
for all 6 countries + EU into ChromaDB permit_docs collection.

Metadata schema per chunk:
    country      : FI/SE/NO/DA/PL/DE/EU   (jurisdiction — which laws apply)
    source       : descriptive name (KHO, NVE, MÖD, …)
    doc_type     : case_law / bat_principles / eia_guidance /
                   bim_standard / noise_standard / nature_law
    language     : fi/sv/no/da/pl/de/en   (document language, independent of country)
    permit_phase : maankaytto / esiselvitys / lupavaihe / rakentaminen / all
    project_types: comma-separated list or 'all'

NOTE: country = jurisdiction (which laws apply)
      language  = document language (independent of country)
      RAG retrieval always filters by country + EU
      Report generation uses UI language selection separately

Usage:
    python3.9 permit_ai/ingest_precedent.py [--dry-run] [--country FI SE ...]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

HERE = Path(__file__).parent
DB_DIR = HERE / "embeddings"

EMBED_MODEL  = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION   = "permit_docs_v2"
CHUNK_CHARS  = 2000
OVERLAP      = 200
BATCH        = 32
MAX_PAGES    = 20
CRAWL_DEPTH  = 1
DELAY_S      = 1.0

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NCE-RAG-Ingest/1.0)"}

# Domains where we fetch only the single given URL (no crawling)
NO_CRAWL_DOMAINS = {
    "eur-lex.europa.eu",
    "www.finlex.fi",
    "lovdata.no",
    "www.nsa.gov.pl",
    "www.bverwg.de",
}

# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------
SOURCES: list[dict] = [
    # ── FI ──────────────────────────────────────────────────────────────────
    dict(country="FI", language="fi", source="KHO",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.finlex.fi/fi/oikeus/kho/"]),
    dict(country="FI", language="fi", source="HAO",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.finlex.fi/fi/oikeus/hao/"]),
    dict(country="FI", language="fi", source="AVI",
         doc_type="case_law", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.avi.fi/fi/luvat-ilmoitukset-rekisteroinnit/ymparistoluvat"]),
    dict(country="FI", language="fi", source="STUK_paatokset",
         doc_type="case_law", permit_phase="lupavaihe", project_types="SMR",
         urls=["https://www.stuk.fi/paatokset"]),
    dict(country="FI", language="fi", source="SYKE_BAT",
         doc_type="bat_principles", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.syke.fi/fi-FI/Tutkimus__kehittaminen/Ympariston_tila/Teollisuuden_paastot/Parhaan_tekniikan_asiakirjat_BREF"]),
    dict(country="FI", language="fi", source="YVA_guidance",
         doc_type="eia_guidance", permit_phase="esiselvitys",
         project_types="tuulivoima,BESS,SMR",
         urls=["https://www.ymparisto.fi/fi/Luvat_ilmoitukset_ja_ymparistoasiat/Ymparistovaikutusten_arviointi"]),
    dict(country="FI", language="fi", source="ymparisto_tuulivoimamelu",
         doc_type="noise_standard", permit_phase="lupavaihe",
         project_types="tuulivoima",
         urls=["https://www.ymparisto.fi/tuulivoimamelu"]),

    # ── SE ──────────────────────────────────────────────────────────────────
    dict(country="SE", language="sv", source="MÖD",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.domstol.se/mark-och-miljodomstolen/"]),
    dict(country="SE", language="sv", source="EI_beslut",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.ei.se/sv/Beslut-och-foreskrifter/"]),
    dict(country="SE", language="sv", source="Naturvardsverket_BAT",
         doc_type="bat_principles", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.naturvardsverket.se/vagledning-och-stod/industri-och-verksamheter/bat/"]),
    dict(country="SE", language="sv", source="Boverket_BIM",
         doc_type="bim_standard", permit_phase="rakentaminen", project_types="all",
         urls=["https://www.boverket.se/sv/byggande/digitalt-byggande/"]),

    # ── NO ──────────────────────────────────────────────────────────────────
    dict(country="NO", language="no", source="NVE_vedtak",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.nve.no/konsesjon/konsesjonsvedtak/"]),
    dict(country="NO", language="no", source="Miljodirektoratet_vedtak",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.miljodirektoratet.no/tjenester/arkiv/vedtak/"]),
    dict(country="NO", language="no", source="Miljodirektoratet_BAT",
         doc_type="bat_principles", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.miljodirektoratet.no/regelverk/bat/"]),

    # ── DA ──────────────────────────────────────────────────────────────────
    dict(country="DA", language="da", source="Energiklagenaevnet",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.naevneneshus.dk/start/energiklagenaevnet/"]),
    dict(country="DA", language="da", source="Miljoeklagenaevnet",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.naevneneshus.dk/start/miljoe-og-foedevareklagenaevnet/"]),
    dict(country="DA", language="da", source="Miljostyrelsen_VVM",
         doc_type="eia_guidance", permit_phase="esiselvitys", project_types="all",
         urls=["https://www.mst.dk/erhverv/industri/vvm/"]),

    # ── PL ──────────────────────────────────────────────────────────────────
    dict(country="PL", language="pl", source="NSA",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.nsa.gov.pl/"]),
    dict(country="PL", language="pl", source="URE_decyzje",
         doc_type="case_law", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.ure.gov.pl/pl/urzad/informacje-ogolne/decyzje-prezesa-ure"]),
    dict(country="PL", language="pl", source="GIOS_BAT",
         doc_type="bat_principles", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.gios.gov.pl/"]),

    # ── DE ──────────────────────────────────────────────────────────────────
    dict(country="DE", language="de", source="BVerwG",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.bverwg.de/entscheidungen"]),
    dict(country="DE", language="de", source="BNetzA_beschlusskammern",
         doc_type="case_law", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.bundesnetzagentur.de/DE/Beschlusskammern/beschlusskammern-node.html"]),
    dict(country="DE", language="de", source="UBA_BAT",
         doc_type="bat_principles", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.umweltbundesamt.de/themen/wirtschaft-konsum/beste-verfuegbare-techniken"]),
    dict(country="DE", language="de", source="BMWSB_BIM",
         doc_type="bim_standard", permit_phase="rakentaminen", project_types="all",
         urls=["https://www.bmwsb.bund.de/Webs/BMWSB/DE/themen/bauwesen/digitales-bauen/bim/bim-node.html"]),

    # ── EU ──────────────────────────────────────────────────────────────────
    dict(country="EU", language="en", source="EUR_Lex_BAT",
         doc_type="bat_principles", permit_phase="lupavaihe", project_types="all",
         urls=["https://eur-lex.europa.eu/search.html?qid=&text=BAT+conclusions+energy"]),
    dict(country="EU", language="en", source="EIA_Directive",
         doc_type="eia_guidance", permit_phase="esiselvitys", project_types="all",
         urls=["https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32014L0052"]),
    dict(country="EU", language="en", source="EU_BIM",
         doc_type="bim_standard", permit_phase="rakentaminen", project_types="all",
         urls=["https://www.eubim.eu/"]),
]


# ---------------------------------------------------------------------------
# Web fetching helpers (shared with ingest_web.py)
# ---------------------------------------------------------------------------

def _fetch(url: str, session) -> str | None:
    try:
        r = session.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if r.status_code == 200:
            ct = r.headers.get("content-type", "")
            if "html" in ct:
                return r.text
            print(f"    skip (content-type={ct}): {url}")
        else:
            print(f"    HTTP {r.status_code}: {url}")
    except Exception as exc:
        print(f"    error ({exc.__class__.__name__}): {url}")
    return None


def _extract_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "noscript", "meta", "link"]):
        tag.decompose()
    main = (soup.find("main") or soup.find("article")
            or soup.find(id="content")
            or soup.find(class_=re.compile(r"content|main|article", re.I)))
    target = main if main else (soup.body if soup.body else soup)
    text = target.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if l)


def _collect_links(html: str, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup
    parsed_base = urlparse(base_url)
    base_path   = parsed_base.path.rstrip("/")
    soup  = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0].split("?")[0]
        if not href:
            continue
        full = urljoin(base_url, href)
        p    = urlparse(full)
        if p.scheme not in ("http", "https"):
            continue
        if p.netloc != parsed_base.netloc:
            continue
        if not p.path.startswith(base_path):
            continue
        clean = p._replace(fragment="", query="").geturl()
        links.append(clean)
    seen: set[str] = set()
    result: list[str] = []
    for lnk in links:
        if lnk not in seen:
            seen.add(lnk)
            result.append(lnk)
    return result


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunk = text[start:end].strip()
        if len(chunk) > 150:
            chunks.append(chunk)
        start += CHUNK_CHARS - OVERLAP
    return chunks


def _safe_id(source: str, url: str, idx: int) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    path = urlparse(url).path
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", path)[:30]
    src_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", source)[:20]
    return f"prec__{src_safe}__{safe}__{url_hash}__{idx}"


# ---------------------------------------------------------------------------
# Main ingest logic
# ---------------------------------------------------------------------------

def ingest_precedent(
    sources: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    import requests
    from sentence_transformers import SentenceTransformer
    import chromadb

    if not DB_DIR.exists():
        print(f"[ingest_precedent] ERROR: {DB_DIR} missing.")
        sys.exit(1)

    print(f"[ingest_precedent] Connecting to ChromaDB: {DB_DIR}")
    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION)

    # Load all existing IDs and indexed URLs
    existing_data = col.get(include=["metadatas"])
    existing_ids:  set[str] = set(existing_data["ids"])
    existing_urls: set[str] = set()
    for m in existing_data["metadatas"]:
        if m and m.get("url"):
            existing_urls.add(m["url"])
    print(f"[ingest_precedent] Existing chunks: {len(existing_ids)}")
    print(f"[ingest_precedent] Existing indexed URLs: {len(existing_urls)}")
    print()

    session = requests.Session()
    summary: list[dict] = []

    for src in sources:
        country      = src["country"]
        language     = src["language"]
        source_name  = src["source"]
        doc_type     = src["doc_type"]
        permit_phase = src["permit_phase"]
        project_types = src["project_types"]
        urls         = src["urls"]

        print(f"[{country}/{source_name}] {doc_type} | phase={permit_phase}")

        new_docs:    list[str]  = []
        new_ids:     list[str]  = []
        new_metas:   list[dict] = []
        pending_ids: set[str]   = set()

        for start_url in urls:
            domain   = urlparse(start_url).netloc
            no_crawl = domain in NO_CRAWL_DOMAINS

            to_visit: list[str] = [start_url]
            visited:  set[str]  = set()
            page_count = 0

            while to_visit and page_count < MAX_PAGES:
                url = to_visit.pop(0)
                if url in visited:
                    continue
                visited.add(url)

                # Skip already-indexed URLs
                if url in existing_urls:
                    print(f"    skip (already indexed): {url}")
                    continue

                if page_count > 0:
                    time.sleep(DELAY_S)

                html = _fetch(url, session)
                page_count += 1

                if not html:
                    continue

                text   = _extract_text(html)
                chunks = _chunk(text)
                added  = 0

                for i, chunk in enumerate(chunks):
                    id_ = _safe_id(source_name, url, i)
                    if id_ in existing_ids or id_ in pending_ids:
                        continue
                    pending_ids.add(id_)
                    new_docs.append(chunk)
                    new_ids.append(id_)
                    new_metas.append({
                        "country":       country,
                        "source":        source_name,
                        "doc_type":      doc_type,
                        "language":      language,
                        "permit_phase":  permit_phase,
                        "project_types": project_types,
                        "url":           url,
                        "source_type":   "web",
                    })
                    added += 1

                short = url.replace("https://", "").replace("http://", "")[:70]
                print(f"    [{page_count:2d}] {short}  →  {len(chunks)} chunks, {added} new")

                if not no_crawl and CRAWL_DEPTH >= 1 and page_count < MAX_PAGES:
                    for link in _collect_links(html, start_url):
                        if link not in visited and link not in to_visit:
                            to_visit.append(link)

        added_count = len(new_docs)
        summary.append({
            "country":      country,
            "doc_type":     doc_type,
            "permit_phase": permit_phase,
            "source":       source_name,
            "chunks_added": added_count,
        })

        if not new_docs:
            print(f"  → 0 new chunks\n")
            continue

        if dry_run:
            print(f"  DRY-RUN: would add {added_count} chunks\n")
            continue

        print(f"  Writing {added_count} chunks to ChromaDB…")
        for i in range(0, len(new_docs), BATCH):
            b = slice(i, i + BATCH)
            embs = model.encode(new_docs[b], show_progress_bar=False).tolist()
            col.add(
                documents=new_docs[b],
                embeddings=embs,
                ids=new_ids[b],
                metadatas=new_metas[b],
            )
        existing_ids.update(new_ids)
        existing_urls.update(m["url"] for m in new_metas)
        print(f"  ✅ {added_count} chunks added\n")

    return summary


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(summary: list[dict], col=None) -> None:
    print(f"\n{'─'*85}")
    print(f"{'country':^8} {'doc_type':^16} {'permit_phase':^14} {'source':^28} {'chunks':>6}")
    print(f"{'─'*85}")
    grand = 0
    for row in summary:
        if row["chunks_added"] > 0:
            print(f"  {row['country']:6s}  {row['doc_type']:16s}  {row['permit_phase']:14s}  "
                  f"{row['source']:28s}  {row['chunks_added']:5d}")
            grand += row["chunks_added"]
    print(f"{'─'*85}")
    print(f"  Total new chunks: {grand}")
    if col:
        print(f"  Total index size: {col.count()}")
    print(f"{'─'*85}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index regulatory precedents and approval criteria into ChromaDB"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--country", nargs="+", metavar="CC",
        help="Limit to specific countries (FI SE NO DA PL DE EU)"
    )
    args = parser.parse_args()

    sources = SOURCES
    if args.country:
        cc = {c.upper() for c in args.country}
        sources = [s for s in SOURCES if s["country"] in cc]

    summary = ingest_precedent(sources, dry_run=args.dry_run)

    # Print final summary table
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(COLLECTION)
    _print_summary(summary, col=col)
