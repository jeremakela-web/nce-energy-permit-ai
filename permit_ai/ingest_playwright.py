"""
Playwright-based headless Chromium ingest for JS-rendered SPA sources.

These sites return 404/202 for static HTTP requests and require a real
browser to render. This script uses Playwright async API + Chromium.

Same metadata schema as ingest_precedent.py:
    country, source, doc_type, language, permit_phase, project_types

# NOTE: country  = jurisdiction (which laws apply)
#       language = document language (independent of country)
#       RAG retrieval always filters by country + EU
#       Report generation uses UI language selection separately

Usage:
    python3.9 permit_ai/ingest_playwright.py [--dry-run] [--country FI SE ...]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

HERE = Path(__file__).parent
DB_DIR = HERE / "embeddings"

EMBED_MODEL   = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION    = "permit_docs_v2"
CHUNK_CHARS   = 2000
OVERLAP       = 200
BATCH         = 32
MAX_PAGES     = 15
CRAWL_DEPTH   = 1
DELAY_S       = 1.5      # polite delay between page loads

# Domains to fetch only the root URL (no link crawling)
NO_CRAWL_DOMAINS = {
    "eur-lex.europa.eu",
}

# ---------------------------------------------------------------------------
# Source configuration — JS-SPA sources that need headless browser
# ---------------------------------------------------------------------------
SOURCES: list[dict] = [
    # ── FI ──────────────────────────────────────────────────────────────────
    dict(country="FI", language="fi", source="YVA_guidance",
         doc_type="eia_guidance", permit_phase="esiselvitys",
         project_types="tuulivoima,BESS,SMR",
         urls=["https://www.ymparisto.fi/fi/Luvat_ilmoitukset_ja_ymparistoasiat/Ymparistovaikutusten_arviointi"]),
    dict(country="FI", language="fi", source="SYKE_BAT",
         doc_type="bat_principles", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.syke.fi/fi-FI/Tutkimus__kehittaminen/Ympariston_tila/Teollisuuden_paastot/Parhaan_tekniikan_asiakirjat_BREF"]),
    dict(country="FI", language="fi", source="AVI",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.avi.fi/fi/luvat-ilmoitukset-rekisteroinnit/ymparistoluvat"]),

    # ── SE ──────────────────────────────────────────────────────────────────
    dict(country="SE", language="sv", source="MÖD",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.domstol.se/mark-och-miljodomstolen/"]),
    dict(country="SE", language="sv", source="Naturvardsverket_BAT",
         doc_type="bat_principles", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.naturvardsverket.se/vagledning-och-stod/industri-och-verksamheter/bat/"]),

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

    # ── DE ──────────────────────────────────────────────────────────────────
    dict(country="DE", language="de", source="BVerwG",
         doc_type="case_law", permit_phase="all", project_types="all",
         urls=["https://www.bverwg.de/entscheidungen"]),
    dict(country="DE", language="de", source="BNetzA_beschlusskammern",
         doc_type="case_law", permit_phase="lupavaihe", project_types="all",
         urls=["https://www.bundesnetzagentur.de/DE/Beschlusskammern/beschlusskammern-node.html"]),

    # ── EU ──────────────────────────────────────────────────────────────────
    dict(country="EU", language="en", source="EUR_Lex_BAT",
         doc_type="bat_principles", permit_phase="lupavaihe", project_types="all",
         urls=["https://eur-lex.europa.eu/search.html?qid=&text=BAT+conclusions+energy"]),
    dict(country="EU", language="en", source="EIA_Directive",
         doc_type="eia_guidance", permit_phase="esiselvitys", project_types="all",
         urls=["https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32014L0052"]),
]


# ---------------------------------------------------------------------------
# Text + link extraction helpers
# ---------------------------------------------------------------------------

def _clean_text(raw: str) -> str:
    lines = [l.strip() for l in raw.splitlines()]
    cleaned = "\n".join(l for l in lines if l)
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _collect_links(html: str, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup
    parsed_base = urlparse(base_url)
    base_path   = parsed_base.path.rstrip("/")
    soup   = BeautifulSoup(html, "html.parser")
    seen:   set[str] = set()
    result: list[str] = []
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
        if clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end   = start + CHUNK_CHARS
        chunk = text[start:end].strip()
        if len(chunk) > 150:
            chunks.append(chunk)
        start += CHUNK_CHARS - OVERLAP
    return chunks


def _safe_id(source: str, url: str, idx: int) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    path     = urlparse(url).path
    safe     = re.sub(r"[^a-zA-Z0-9_-]", "_", path)[:30]
    src_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", source)[:20]
    return f"pw__{src_safe}__{safe}__{url_hash}__{idx}"


# ---------------------------------------------------------------------------
# Playwright page fetch
# ---------------------------------------------------------------------------

async def _fetch_page(url: str, page) -> tuple[str, str]:
    """Return (text_content, full_page_html) after waiting for networkidle.

    text_content  — cleaned text from the main content area (for chunking)
    full_page_html — full page HTML (for link collection)
    """
    try:
        await page.goto(url, wait_until="networkidle", timeout=25000)
        # Extra wait for late-rendering SPAs
        await asyncio.sleep(2.0)

        # Full page HTML for link collection
        full_html = await page.content()

        # Extract text from content area
        text = ""
        for selector in ["main", "article", "#content", ".content", "body"]:
            el = await page.query_selector(selector)
            if el:
                candidate = await el.inner_text()
                if len(candidate.strip()) > 200:
                    text = candidate
                    break
        if not text:
            text = await page.inner_text("body")

        return _clean_text(text), full_html
    except Exception as exc:
        print(f"    playwright error ({exc.__class__.__name__}): {url}")
        return "", ""


# ---------------------------------------------------------------------------
# Main async ingest
# ---------------------------------------------------------------------------

async def ingest_playwright_async(
    sources: list[dict],
    dry_run: bool,
    existing_ids: set[str],
    existing_urls: set[str],
) -> list[dict]:
    from playwright.async_api import async_playwright

    summary: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = await context.new_page()

        for src in sources:
            country       = src["country"]
            language      = src["language"]
            source_name   = src["source"]
            doc_type      = src["doc_type"]
            permit_phase  = src["permit_phase"]
            project_types = src["project_types"]
            urls          = src["urls"]

            print(f"\n[{country}/{source_name}] {doc_type} | phase={permit_phase}")

            new_docs:    list[str]  = []
            new_ids:     list[str]  = []
            new_metas:   list[dict] = []
            pending_ids: set[str]   = set()

            for start_url in urls:
                domain   = urlparse(start_url).netloc
                no_crawl = domain in NO_CRAWL_DOMAINS

                to_visit:  list[str] = [start_url]
                visited:   set[str]  = set()
                page_count = 0

                while to_visit and page_count < MAX_PAGES:
                    url = to_visit.pop(0)
                    if url in visited:
                        continue
                    visited.add(url)

                    if url in existing_urls:
                        print(f"    skip (already indexed): {url}")
                        continue

                    if page_count > 0:
                        await asyncio.sleep(DELAY_S)

                    text, html = await _fetch_page(url, page)
                    page_count += 1

                    if not text:
                        continue

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

                    if not no_crawl and CRAWL_DEPTH >= 1 and page_count < MAX_PAGES and html:
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
                "new_docs":     new_docs,
                "new_ids":      new_ids,
                "new_metas":    new_metas,
            })

            if not new_docs:
                print(f"  → 0 new chunks")
            elif dry_run:
                print(f"  DRY-RUN: would add {added_count} chunks")
            else:
                print(f"  → {added_count} chunks queued for embedding")

        await context.close()
        await browser.close()

    return summary


def ingest_playwright(sources: list[dict], dry_run: bool = False) -> list[dict]:
    from sentence_transformers import SentenceTransformer
    import chromadb

    if not DB_DIR.exists():
        print(f"[ingest_playwright] ERROR: {DB_DIR} missing.")
        sys.exit(1)

    print(f"[ingest_playwright] Connecting to ChromaDB: {DB_DIR}")
    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION)

    existing_data = col.get(include=["metadatas"])
    existing_ids:  set[str] = set(existing_data["ids"])
    existing_urls: set[str] = {
        m["url"] for m in existing_data["metadatas"] if m and m.get("url")
    }
    print(f"[ingest_playwright] Existing chunks: {len(existing_ids)}")
    print(f"[ingest_playwright] Existing indexed URLs: {len(existing_urls)}")

    summary = asyncio.run(
        ingest_playwright_async(sources, dry_run, existing_ids, existing_urls)
    )

    if not dry_run:
        for row in summary:
            docs   = row.pop("new_docs",  [])
            ids    = row.pop("new_ids",   [])
            metas  = row.pop("new_metas", [])
            if not docs:
                continue
            print(f"\n  [{row['country']}/{row['source']}] Embedding {len(docs)} chunks…")
            for i in range(0, len(docs), BATCH):
                b    = slice(i, i + BATCH)
                embs = model.encode(docs[b], show_progress_bar=False).tolist()
                col.add(
                    documents=docs[b],
                    embeddings=embs,
                    ids=ids[b],
                    metadatas=metas[b],
                )
                pct = min(100, (i + len(docs[b])) * 100 // len(docs))
                print(f"    {i + len(docs[b])}/{len(docs)} ({pct}%)")
            existing_ids.update(ids)
            print(f"  ✅ {len(docs)} chunks added [{row['country']}/{row['source']}]")
    else:
        for row in summary:
            row.pop("new_docs",  None)
            row.pop("new_ids",   None)
            row.pop("new_metas", None)

    return summary


# ---------------------------------------------------------------------------
# Summary table + CLI
# ---------------------------------------------------------------------------

def _print_summary(summary: list[dict], col=None) -> None:
    print(f"\n{'─'*85}")
    print(f"{'country':^8} {'doc_type':^16} {'permit_phase':^14} {'source':^28} {'chunks':>6}")
    print(f"{'─'*85}")
    grand = 0
    for row in summary:
        n = row["chunks_added"]
        if n > 0:
            print(f"  {row['country']:6s}  {row['doc_type']:16s}  {row['permit_phase']:14s}  "
                  f"{row['source']:28s}  {n:5d}")
            grand += n
    print(f"{'─'*85}")
    print(f"  Total new chunks: {grand}")
    if col:
        print(f"  Total index size: {col.count()}")
    print(f"{'─'*85}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Playwright headless ingest for JS-rendered SPA sources"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--country", nargs="+", metavar="CC")
    args = parser.parse_args()

    srcs = SOURCES
    if args.country:
        cc   = {c.upper() for c in args.country}
        srcs = [s for s in SOURCES if s["country"] in cc]

    summary = ingest_playwright(srcs, dry_run=args.dry_run)

    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION)
    _print_summary(summary, col=col)
