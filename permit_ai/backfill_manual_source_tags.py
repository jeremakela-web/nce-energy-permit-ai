"""
One-time (idempotent) metadata backfill: adds source_type:"manual" + last_verified
to chunks from the 3 source groups confirmed — by tracing each back to the actual
ingestion script/commit that added them, not by assumption — as genuinely manually
sourced content with no automated fetch path:

  1. LT — lt_pav_istatymas (backend/lithuania_ingestion.py). The only one of 17 LT
     sources using a local_path bypass; in-code comment: "e-seimas.lrs.lt blocks
     automated fetches ... manually sourced". last_verified = 2026-07-25
     (commit 63aca77, PR #16 — when this was added, unchanged since).
  2. FI inline law texts (permit_ai/ingest_fi_env.py, _FI_ENV_LAW_DOCS — 4 docs:
     YSL 527/2014 luvantarve/hakeminen/BESS, YVA-laki 252/2017). Finlex is fully
     JS-rendered per that script's own docstring; this text is typed directly into
     the script, no fetch path exists. last_verified = 2026-05-28 (commit cbfbe74,
     confirmed unchanged since via `git log -S` on the text content).
  3. FI datakeskus files (permit_ai/ingest_datakeskus.py, 6 files). Each .txt file
     self-documents its origin with a "SOURCE: <url>" header; commit message:
     "add datakeskus RAG content to unblock InsufficientSourcesError" — human-
     extracted to fix a RAG_FAIL. last_verified = 2026-06-15 (commit dbfbefc).

Everything else investigated and explicitly ruled out (zero chunks exist for these
— nothing to backfill): DA/retsinformation.dk, LT's VERT/pagd.lrv.lt/e-tar.lt/4
e-seimas.lrs.lt sources (confirmed via a live re-run: 403/SSL-failure/too-short,
0 chunks each), LV wind gap (a retrieval-ranking issue, not a sourcing one), PL
isap.sejm.gov.pl. A larger, structurally-ambiguous category (ingest_countries.py's
SE/DA/NO/PL/EE/DE folder-based PDFs, build_index.py's bulk FI PDFs) was
deliberately left OUT of this backfill — see the task investigation notes;
tagging it would need per-file provenance research this pass didn't do.

Going forward, the 3 source ingestion scripts listed above now write these two
fields themselves at ingest time — this script only fixes chunks that were
already in the index before that code existed.

Safe to re-run (idempotent): only ever touches `source_type`/`last_verified` on
chunks whose `source` metadata exactly matches one of the confirmed values below.
Never touches document text, embeddings, or any other metadata field, and never
touches any chunk outside this exact list.

Usage:
    python3 permit_ai/backfill_manual_source_tags.py [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE   = Path(__file__).parent
DB_DIR = HERE / "embeddings"

# (source value, last_verified date, collections to check)
_TARGETS: list[tuple[str, str, list[str]]] = [
    ("lt_pav_istatymas", "2026-07-25", ["permit_docs", "permit_docs_v2"]),
    ("YSL 527/2014 — Ympäristölupa: luvantarve", "2026-05-28", ["permit_docs"]),
    ("YSL 527/2014 — Ympäristöluvan hakeminen: prosessi ja liitteet", "2026-05-28", ["permit_docs"]),
    ("YVA-laki 252/2017 — Ympäristövaikutusten arviointi", "2026-05-28", ["permit_docs"]),
    ("YSL 527/2014 — Akkuvarasto (BESS) ja energiantuotanto: ympäristölupatarve", "2026-05-28", ["permit_docs"]),
    ("bios_datakeskus_sijoittamislupa", "2026-06-15", ["permit_docs"]),
    ("microsoft_espoo_yva_selostus", "2026-06-15", ["permit_docs"]),
    ("rakentamislaki_sijoittamislupa_datakeskus", "2026-06-15", ["permit_docs"]),
    ("ymparistolupa_datakeskus_ysl", "2026-06-15", ["permit_docs"]),
    ("ym_datakeskukset", "2026-06-15", ["permit_docs"]),
    ("datakeskus_luvat_suomi", "2026-06-15", ["permit_docs"]),
]


def backfill(dry_run: bool = False) -> dict[str, int]:
    import chromadb

    if not DB_DIR.exists():
        raise RuntimeError(f"ChromaDB path not found: {DB_DIR}")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    totals: dict[str, int] = {}

    for source_value, last_verified, collections in _TARGETS:
        for cname in collections:
            try:
                col = client.get_collection(cname)
            except Exception:
                print(f"  SKIP ({cname} doesn't exist)")
                continue

            got = col.get(where={"source": source_value}, include=["metadatas"])
            ids, metas = got["ids"], got["metadatas"]
            if not ids:
                print(f"[{cname}] source={source_value!r}: 0 chunks found — nothing to backfill")
                continue

            already = sum(
                1 for m in metas
                if m.get("source_type") == "manual" and m.get("last_verified") == last_verified
            )
            to_update = len(ids) - already
            print(f"[{cname}] source={source_value!r}: {len(ids)} chunks, "
                  f"{already} already tagged correctly, {to_update} to update")

            if to_update == 0:
                totals[f"{cname}:{source_value}"] = 0
                continue

            if dry_run:
                print(f"  DRY-RUN: would update {to_update} chunk(s) — "
                      f"source_type=manual, last_verified={last_verified}")
                totals[f"{cname}:{source_value}"] = to_update
                continue

            update_ids: list[str] = []
            update_metas: list[dict] = []
            for id_, meta in zip(ids, metas):
                if meta.get("source_type") == "manual" and meta.get("last_verified") == last_verified:
                    continue  # already correct — idempotent skip
                # Merge, never replace wholesale — every other field is preserved untouched.
                merged = dict(meta)
                merged["source_type"] = "manual"
                merged["last_verified"] = last_verified
                update_ids.append(id_)
                update_metas.append(merged)

            col.update(ids=update_ids, metadatas=update_metas)
            print(f"  ✅ updated {len(update_ids)} chunk(s)")
            totals[f"{cname}:{source_value}"] = len(update_ids)

    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    print(f"{'DRY-RUN — ' if args.dry_run else ''}Backfilling source_type:manual + last_verified "
          f"for the 3 confirmed manually-sourced groups...\n")
    totals = backfill(dry_run=args.dry_run)
    grand = sum(totals.values())
    print(f"\n{'─'*60}")
    print(f"{'Would update' if args.dry_run else 'Updated'}: {grand} chunk(s) total")
    print(f"{'─'*60}")
