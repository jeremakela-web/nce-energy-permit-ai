"""
RAG Coverage Matrix — NCE Energy Permit AI
Analysoi ChromaDB-indeksin kattavuuden per maa × hanketyyppi.

Ajo:
    python3 backend/rag_coverage_report.py
    python3 backend/rag_coverage_report.py --json   # koneluettava JSON
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_DIR = ROOT / "permit_ai" / "embeddings"
COLLECTION = "permit_docs"

# ── Hanketyypit ja hakusanat indeksissä ──────────────────────────────────────

HANKE_KEYWORDS: dict[str, list[str]] = {
    "BESS":         ["bess", "battery", "akku", "akkuvarasto", "energilager",
                     "batterianlaegg", "magazyn", "energy_storage", "lion"],
    "AURINKO":      ["sol", "aurinko", "solar", "fotovoltaik", "pv", "aurinkovoi"],
    "TUULI":        ["tuuli", "vind", "wind", "wiatr", "vindmoel", "vindkraft",
                     "vindkraftanlaegg"],
    "SMR/YVA":      ["nuclear", "smr", "atomi", "kaern", "yaederreaktori",
                     "yvl", "national_report", "radiation", "yadrovy"],
    "DATAKESKUS":   ["datakeskus", "datacenter", "datasenter", "data_center",
                     "datacent", "ym_data"],
    "SCO2":         ["sco2", "tryckkarlstillstyrning", "cisnieniowy", "painelaite",
                     "trykkpaavirket", "trykbaerende"],
    "VESIVOIMA":    ["vesivoima", "vattenkraft", "vandkraft", "hydropower",
                     "vattenverksamhet", "wody_polskie", "vandloeb"],
    "YVA/MKB":      ["yva", "yvl", "mkb", "eia", "vvm", "miljoevurdering",
                     "konsekvensutredning", "oos", "gdos"],
    "VERKKO":       ["fingrid", "pse_grid", "svk_anslutning", "nve_tilknytning",
                     "sjv", "vjv", "statnett"],
}

COUNTRIES = ["FI", "SE", "DA", "NO", "PL"]

# Thresholds per hanketyyppi × maa (min chunkkeja "Full" -luokitteluun)
FULL_THRESHOLD  = 5
PARTIAL_MIN     = 2

# Kriittiset dokumentit jotka pitäisi löytyä (source-avain sisältää string)
CRITICAL_DOCS: dict[str, dict[str, list[str]]] = {
    "FI": {
        "BESS":      ["lion_2025_bess", "tukes_liion"],
        "AURINKO":   ["aurinkovoima", "solar"],
        "TUULI":     ["tuuli", "wind"],
        "SMR/YVA":   ["YVL_A", "YVL_B"],
        "DATAKESKUS":["datakeskus", "ym_data"],
        "SCO2":      ["painelaite", "sco2"],
        "VESIVOIMA": ["vesivoima", "vattenkraft"],
        "VERKKO":    ["fingrid", "sjv", "vjv"],
    },
    "SE": {
        "BESS":      ["energilager", "batterianlaegg"],
        "AURINKO":   ["solenergi"],
        "TUULI":     ["vindkraft"],
        "SMR/YVA":   ["ssm_karnkraft", "nuclear"],
        "DATAKESKUS":["datacenter"],
        "SCO2":      ["tryckkarlstillstyrning", "av_sco2"],
        "VESIVOIMA": ["vattenkraft", "vattenverksamhet"],
        "VERKKO":    ["svk_anslutning"],
    },
    "DA": {
        "BESS":      ["bess", "elproduktion"],
        "AURINKO":   ["sol_ve"],
        "TUULI":     ["vindmoel"],
        "SMR/YVA":   ["nuclear", "smr"],
        "DATAKESKUS":["datacentre"],
        "SCO2":      ["sco2", "trykbaerende"],
        "VESIVOIMA": ["vandkraft", "vandloeb"],
        "YVA/MKB":   ["miljoevurdering", "vvm"],
    },
    "NO": {
        "BESS":      ["batterianlaegg", "soknadsveileder_batteri"],
        "AURINKO":   ["solkraft"],
        "TUULI":     ["vindkraft", "vindmoel"],
        "SMR/YVA":   ["smr_nuclear", "dsa_nuclear"],
        "DATAKESKUS":["datasenter", "dibk_data"],
        "SCO2":      ["trykkpaavirket", "dsb_sco2"],
        "VESIVOIMA": ["nve_konsesjon"],
        "YVA/MKB":   ["konsekvensutredning"],
    },
    "PL": {
        "BESS":      ["bess", "magazyn", "ure_bess"],
        "AURINKO":   ["solar", "fotovoltaik"],
        "TUULI":     ["wind", "wiatr"],
        "SMR/YVA":   ["national_report", "nuclear_safety"],
        "DATAKESKUS":["datacenter", "udt_ure_datacenter"],
        "SCO2":      ["cisnieniowy", "udt_sco2"],
        "VESIVOIMA": ["hydropower", "wody_polskie"],
        "YVA/MKB":   ["gdos", "oos"],
    },
}


def _load_index() -> tuple[dict[str, dict[str, int]], set[str]]:
    """Lataa ChromaDB ja palauta (by_country_source, missing_meta_ids)."""
    try:
        import chromadb
    except ImportError:
        print("VIRHE: chromadb ei asennettu. Aja: pip install chromadb", file=sys.stderr)
        sys.exit(1)

    if not DB_DIR.exists():
        print(f"VIRHE: ChromaDB ei löydy: {DB_DIR}", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(COLLECTION)

    all_data = col.get(include=["metadatas", "documents"])
    metas = all_data["metadatas"]
    ids = all_data["ids"]
    docs = all_data["documents"]

    by_country_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing_meta: set[str] = set()
    short_chunks: list[tuple[str, int]] = []
    all_docs: list[str] = docs or []

    for idx, m in enumerate(metas):
        doc_text = all_docs[idx] if idx < len(all_docs) else ""
        if not m or not m.get("country") or not m.get("source"):
            missing_meta.add(ids[idx])
        else:
            c = m["country"]
            s = m["source"]
            by_country_source[c][s] += 1
            if doc_text and len(doc_text) < 100:
                short_chunks.append((ids[idx], len(doc_text)))

    return dict(by_country_source), missing_meta, short_chunks, ids, docs


def _coverage_for(country: str, hanke: str, sources: dict[str, int]) -> tuple[str, int]:
    """Palauta (Full/Partial/Weak/Missing, n_chunks)."""
    keywords = HANKE_KEYWORDS.get(hanke, [])
    total = 0
    for src, cnt in sources.items():
        src_lower = src.lower()
        if any(kw in src_lower for kw in keywords):
            total += cnt
    if total >= FULL_THRESHOLD:
        return "Full", total
    if total >= PARTIAL_MIN:
        return "Partial", total
    if total == 1:
        return "Weak", total
    return "Missing", 0


def _gap_score(coverages: dict[str, str]) -> float:
    """Gap Score 0–100: 100 = täysin katettu, 0 = ei mitään."""
    weights = {"Full": 1.0, "Partial": 0.5, "Weak": 0.2, "Missing": 0.0}
    vals = [weights[v] for v in coverages.values()]
    return round(100 * sum(vals) / len(vals), 1) if vals else 0.0


def _missing_critical(country: str, sources: dict[str, int]) -> list[str]:
    """Palauta lista kriittisistä puuttuvista dokumenteista."""
    crit = CRITICAL_DOCS.get(country, {})
    missing = []
    for hanke, patterns in crit.items():
        for pat in patterns:
            found = any(pat.lower() in s.lower() for s in sources)
            if not found:
                missing.append(f"{country}/{hanke}: '{pat}' puuttuu")
    return missing


def run_report(as_json: bool = False) -> dict:
    by_country, missing_meta, short_chunks, all_ids, all_docs = _load_index()

    hanketyypit = list(HANKE_KEYWORDS.keys())
    matrix: dict[str, dict] = {}

    for country in COUNTRIES:
        sources = by_country.get(country, {})
        coverages = {}
        chunk_counts = {}
        for hanke in hanketyypit:
            status, n = _coverage_for(country, hanke, sources)
            coverages[hanke] = status
            chunk_counts[hanke] = n
        gap = _gap_score(coverages)
        missing = _missing_critical(country, sources)
        matrix[country] = {
            "total_chunks":  sum(sources.values()),
            "total_sources": len(sources),
            "gap_score":     gap,
            "coverages":     coverages,
            "chunk_counts":  chunk_counts,
            "missing_critical": missing,
        }

    # Priorisoi puuttuvat: Missing ensin, sitten Weak, sitten Partial
    priority_gaps: list[dict] = []
    for country in COUNTRIES:
        for hanke in hanketyypit:
            status = matrix[country]["coverages"][hanke]
            if status in ("Missing", "Weak", "Partial"):
                priority_gaps.append({
                    "priority": {"Missing": 1, "Weak": 2, "Partial": 3}[status],
                    "country": country,
                    "hanke": hanke,
                    "status": status,
                    "chunks": matrix[country]["chunk_counts"][hanke],
                })
    priority_gaps.sort(key=lambda x: (x["priority"], x["country"]))

    result = {
        "total_chunks":    sum(v["total_chunks"] for v in matrix.values()),
        "countries":       matrix,
        "priority_gaps":   priority_gaps,
        "missing_meta_ids": len(missing_meta),
        "short_chunks":    len(short_chunks),
    }

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    # ── Tulosta ihmisluettava raportti ──────────────────────────────────────

    W = {"Full": "✅", "Partial": "⚡", "Weak": "⚠️ ", "Missing": "❌"}

    print("=" * 70)
    print("  NCE ENERGY — RAG COVERAGE MATRIX")
    print(f"  Indeksi: {result['total_chunks']} chunkkia yhteensä")
    print("=" * 70)

    # Coverage-taulukko
    hanke_labels = [h.ljust(12) for h in hanketyypit]
    header = f"{'Hanke':13}" + "".join(f"{c:>8}" for c in COUNTRIES)
    print(f"\n{header}")
    print("-" * (13 + 8 * len(COUNTRIES)))
    for hanke in hanketyypit:
        row = f"{hanke:13}"
        for c in COUNTRIES:
            status = matrix[c]["coverages"][hanke]
            row += f"  {W[status]:5}"
        print(row)

    print(f"\nLegenda: ✅ Full(≥{FULL_THRESHOLD}ch)  ⚡ Partial(2-{FULL_THRESHOLD-1}ch)  ⚠️  Weak(1ch)  ❌ Missing")

    # Gap Score
    print(f"\n{'GAP SCORE (100=täysin katettu)':35}", end="")
    for c in COUNTRIES:
        print(f"  {matrix[c]['gap_score']:>4}%", end="")
    print()

    print(f"\n{'CHUNKKEJA YHTEENSÄ':35}", end="")
    for c in COUNTRIES:
        print(f"  {matrix[c]['total_chunks']:>5}", end="")
    print()

    print(f"\n{'LÄHTEITÄ YHTEENSÄ':35}", end="")
    for c in COUNTRIES:
        print(f"  {matrix[c]['total_sources']:>5}", end="")
    print()

    # Priority gaps
    print(f"\n{'─'*70}")
    print("PUUTTEET PRIORITEETTIJÄRJESTYKSESSÄ:")
    print(f"{'─'*70}")
    for g in priority_gaps[:20]:
        prio_label = {1: "KRIITTINEN", 2: "HEIKKO    ", 3: "OSITTAINEN"}[g["priority"]]
        print(f"  [{prio_label}] {g['country']}/{g['hanke']:<14} {g['chunks']:3}ch")

    # Kriittiset puuttuvat dokumentit
    all_missing_crit = []
    for c in COUNTRIES:
        all_missing_crit.extend(matrix[c]["missing_critical"])

    if all_missing_crit:
        print(f"\n{'─'*70}")
        print("KRIITTISET PUUTTUVAT DOKUMENTIT:")
        for m in all_missing_crit[:30]:
            print(f"  • {m}")

    # Metadata-ongelmat
    print(f"\n{'─'*70}")
    print("TEKNINEN LAATU:")
    print(f"  Chunkkeja ilman metadata:  {len(missing_meta)}")
    print(f"  Liian lyhyet chunkit:      {len(short_chunks)}")

    # Executive summary
    print(f"\n{'─'*70}")
    print("EXECUTIVE SUMMARY:")
    full_total = sum(
        1 for c in COUNTRIES for h in hanketyypit
        if matrix[c]["coverages"][h] == "Full"
    )
    total_cells = len(COUNTRIES) * len(hanketyypit)
    missing_total = sum(
        1 for c in COUNTRIES for h in hanketyypit
        if matrix[c]["coverages"][h] == "Missing"
    )
    avg_gap = round(sum(matrix[c]["gap_score"] for c in COUNTRIES) / len(COUNTRIES), 1)
    print(f"  Kattavuus: {full_total}/{total_cells} solu on Full ({full_total*100//total_cells}%)")
    print(f"  Puuttuvat: {missing_total} yhdistelmää ilman dokumentteja")
    gap_parts = " | ".join(f"{c}:{matrix[c]['gap_score']}%" for c in COUNTRIES)
    print(f"  Avg Gap Score: {avg_gap}% ({gap_parts})")

    best = max(COUNTRIES, key=lambda c: matrix[c]["gap_score"])
    worst = min(COUNTRIES, key=lambda c: matrix[c]["gap_score"])
    print(f"  Paras kattavuus: {best} ({matrix[best]['gap_score']}%)")
    print(f"  Heikoin:         {worst} ({matrix[worst]['gap_score']}%)")
    print(f"{'='*70}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Coverage Matrix raportti")
    parser.add_argument("--json", action="store_true", help="Tulosta JSON")
    args = parser.parse_args()
    run_report(as_json=args.json)
