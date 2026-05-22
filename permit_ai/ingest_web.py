"""
Hae ja indeksoi viranomaisverkkosivujen sisältö ChromaDB-vektoritietokantaan.

Hakee annetut URL:t (+ sisäiset linkit 1 taso syvemmälle) ja indeksoi
tekstin samaan ChromaDB-kokoelmaan kuin ingest_countries.py.

Käyttö:
    # Yksittäinen maa, useita URL:ja
    python3 permit_ai/ingest_web.py --country SE \\
        --url "https://www.boverket.se/sv/PBL-kunskapsbanken/" \\
              "https://www.energimyndigheten.se/fornybart/"

    # Kaikki maat konfiguraatiosta (WEB_SOURCES)
    python3 permit_ai/ingest_web.py --all

    # Vain dry-run (ei kirjoita indeksiin)
    python3 permit_ai/ingest_web.py --all --dry-run

Metadata jokaisessa chunkissa:
    country     : "SE" | "DA" | "NO" | "PL"
    lang        : "sv" | "da" | "no" | "pl"
    source      : domain (esim. "nve.no")
    url         : koko URL
    source_type : "web"
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

HERE = Path(__file__).parent
ROOT = HERE.parent
DB_DIR = HERE / "embeddings"

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION  = "permit_docs"
CHUNK_CHARS = 1500
OVERLAP     = 200
BATCH       = 64
MAX_PAGES   = 20       # max sivua per lähtö-URL (crawl)
CRAWL_DEPTH = 1        # 0 = vain annettu URL, 1 = + suorat linkit
DELAY_S     = 1.0      # kohteliaisuusviive pyyntöjen välillä (s)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NCE-RAG-Ingest/1.0; "
        "+https://github.com/jeremakela-web/nce-energy-permit-ai)"
    )
}

COUNTRY_LANG: dict[str, str] = {
    "SE": "sv",
    "DA": "da",
    "NO": "no",
    "PL": "pl",
}

# ── Valmiit lähdekohtaiset URL-konfiguraatiot ────────────────────────────────
WEB_SOURCES: dict[str, list[str]] = {
    "SE": [
        "https://www.boverket.se/sv/PBL-kunskapsbanken/",
        "https://www.energimyndigheten.se/fornybart/",
    ],
    "DA": [
        "https://ens.dk/",
        "https://energinet.dk/",
    ],
    "NO": [
        "https://www.nve.no/konsesjon/",
        "https://lovdata.no/dokument/NL/lov/2008-06-27-71",
    ],
    "PL": [
        "https://isap.sejm.gov.pl/",
    ],
}

# Sivustot joita ei crawlata linkkien kautta (vain annettu URL itse)
NO_CRAWL_DOMAINS = {"lovdata.no", "isap.sejm.gov.pl"}


# ── Tekstin poiminta ─────────────────────────────────────────────────────────

def _extract_text(html: str, url: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Poista nav, header, footer, script, style, aside
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "noscript", "meta", "link"]):
        tag.decompose()

    # Suosi main/article-elementtiä jos löytyy
    main = soup.find("main") or soup.find("article") or soup.find(id="content") \
           or soup.find(class_=re.compile(r"content|main|article", re.I))
    target = main if main else soup.body if soup.body else soup

    text = target.get_text(separator="\n")
    # Siivoa ylimääräiset tyhjät rivit
    lines = [l.strip() for l in text.splitlines()]
    cleaned = "\n".join(l for l in lines if l)
    return cleaned


def _collect_links(html: str, base_url: str) -> list[str]:
    """Kerää saman domainin ja saman polkuprefixin linkit."""
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
        # Vain sama polkuprefiksi (alihakemistot)
        if not p.path.startswith(base_path):
            continue
        clean = p._replace(fragment="", query="").geturl()
        links.append(clean)

    # Deduplicate säilyttäen järjestys
    seen: set[str] = set()
    result: list[str] = []
    for l in links:
        if l not in seen:
            seen.add(l)
            result.append(l)
    return result


# ── Sivun haku ───────────────────────────────────────────────────────────────

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
        print(f"    virhe ({exc.__class__.__name__}): {url}")
    return None


# ── Chunkkaaja ───────────────────────────────────────────────────────────────

def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunks.append(text[start:end].strip())
        start += CHUNK_CHARS - OVERLAP
    return [c for c in chunks if len(c) > 100]


def _safe_id(country: str, url: str, idx: int) -> str:
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    path = urlparse(url).path
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", path)[:40]
    return f"web__{country}__{safe}__{url_hash}__{idx}"


# ── Päälogiikka ──────────────────────────────────────────────────────────────

def ingest_web(
    country_urls: dict[str, list[str]],
    dry_run: bool = False,
) -> dict[str, int]:
    import requests
    from sentence_transformers import SentenceTransformer
    import chromadb

    if not DB_DIR.exists():
        print(
            f"[ingest_web] VIRHE: {DB_DIR} puuttuu.\n"
            f"             Aja ensin: python3 permit_ai/build_index.py"
        )
        sys.exit(1)

    print(f"[ingest_web] Yhdistetään ChromaDB:hen: {DB_DIR}")
    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION)

    existing_ids: set[str] = set(col.get()["ids"])
    print(f"[ingest_web] Olemassaolevia chunkkeja: {len(existing_ids)}")

    session = requests.Session()
    totals: dict[str, int] = {}

    for country, urls in country_urls.items():
        lang = COUNTRY_LANG.get(country, "en")
        print(f"\n[{country}] {len(urls)} lähtö-URL:a, kieli={lang}")

        new_docs:  list[str]  = []
        new_ids:   list[str]  = []
        new_metas: list[dict] = []

        for start_url in urls:
            domain   = urlparse(start_url).netloc
            no_crawl = domain in NO_CRAWL_DOMAINS

            # Sivut tältä lähtö-URL:lta
            to_visit: list[str] = [start_url]
            visited:  set[str]  = set()
            page_count = 0

            print(f"  → {start_url}")

            while to_visit and page_count < MAX_PAGES:
                url = to_visit.pop(0)
                if url in visited:
                    continue
                visited.add(url)

                if page_count > 0:
                    time.sleep(DELAY_S)

                html = _fetch(url, session)
                page_count += 1

                if not html:
                    continue

                text   = _extract_text(html, url)
                chunks = _chunk(text)
                added  = 0

                for i, chunk in enumerate(chunks):
                    id_ = _safe_id(country, url, i)
                    if id_ in existing_ids:
                        continue
                    new_docs.append(chunk)
                    new_ids.append(id_)
                    new_metas.append({
                        "country":     country,
                        "lang":        lang,
                        "source":      domain,
                        "url":         url,
                        "source_type": "web",
                    })
                    added += 1

                short = url.replace("https://", "").replace("http://", "")[:70]
                print(f"    [{page_count:2d}] {short}  →  {len(chunks)} chunkkia, {added} uutta")

                # Lisää linkit jonoon (1 taso)
                if not no_crawl and CRAWL_DEPTH >= 1 and page_count < MAX_PAGES:
                    for link in _collect_links(html, start_url):
                        if link not in visited and link not in to_visit:
                            to_visit.append(link)

        # Kirjoita ChromaDB
        if not new_docs:
            print(f"  → Ei uusia chunkkeja ({country})")
            totals[country] = 0
            continue

        if dry_run:
            print(f"  DRY-RUN [{country}]: {len(new_docs)} chunkkia lisättäisiin")
            totals[country] = len(new_docs)
            continue

        print(f"  Lisätään {len(new_docs)} chunkkia ChromaDB:hen ({country})…")
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
        totals[country] = len(new_docs)
        print(f"  ✅ {len(new_docs)} chunkkia lisätty ({country})")

    # Yhteenveto
    print(f"\n{'─'*55}")
    print("Yhteenveto (web-indeksointi):")
    grand = 0
    for c in ["SE", "DA", "NO", "PL"]:
        n = totals.get(c, 0)
        if n:
            print(f"  {c}: {n} uutta chunkkia")
        grand += n
    print(f"  Yhteensä uutta: {grand}")
    print(f"  Koko indeksi:   {col.count()} chunkkia")
    print(f"{'─'*55}")
    return totals


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hae ja indeksoi viranomaisverkkosivut ChromaDB:hen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--country",
        nargs="+",
        choices=list(COUNTRY_LANG.keys()),
        metavar="CC",
        help="Maa/maat (SE DA NO PL)",
    )
    parser.add_argument(
        "--url",
        nargs="+",
        metavar="URL",
        help="URL:t haettaville sivuille (vaatii --country)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Hae kaikki WEB_SOURCES-konfiguraation URL:t",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Näytä chunkit — ei kirjoita indeksiin",
    )
    args = parser.parse_args()

    if args.all:
        target = WEB_SOURCES
    elif args.country and args.url:
        # Kaikki annetut URL:t samalle maalle / maille
        target = {c: args.url for c in args.country}
    elif args.country:
        target = {c: WEB_SOURCES.get(c, []) for c in args.country}
    else:
        parser.print_help()
        sys.exit(0)

    ingest_web(target, dry_run=args.dry_run)
