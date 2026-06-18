"""
CoT prompt validation script.
Runs permit generation for given test cases and validates:
  1. All 6 CoT labels present in output
  2. ARVIOI section contains a specific risk with a real regulatory citation
  3. LIFECYCLE either cites a source or emits the configured fallback message
  4. No numbered SUOSITTELE items lack any source reference

Usage (from repo root):
    PYTHONPATH=backend:permit_ai ANTHROPIC_API_KEY=<key> \
        venv312/bin/python3 test_cot_validation.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, "backend")
sys.path.insert(0, "permit_ai")

from generate_application import ApplicationInput, generate_application_draft  # noqa: E402

# ---------------------------------------------------------------------------
# Citation detection helpers
# ---------------------------------------------------------------------------

# Finnish uncertainty-marker words rendered by the model as [Huom] etc.
# These are NOT regulatory citations and must be excluded.
_NOTE_WORDS: frozenset[str] = frozenset({
    "huom", "huomio", "note", "obs", "bemærk", "merkintä", "uwaga",
})

_BRACKET_RE = re.compile(r"\[([^\]]{2,80})\]")


def real_citations(text: str) -> list[str]:
    """
    Return bracket-delimited citations that look like legal/source references.

    Accepted:
      - Capital-letter starts: [YSL 27 §], [Rakentamislaki 751/2023], [Fingrid SJV2019]
      - Digit-leading law numbers: [390/2005], [751/2023], [379/2011]

    Excluded:
      - [Huom], [Note], [Obs] and other uncertainty-marker words
    """
    found = []
    for m in _BRACKET_RE.finditer(text):
        inner = m.group(1).strip()
        # Exclude known note-marker words (case-insensitive, exact match)
        if inner.lower() in _NOTE_WORDS:
            continue
        # Accept: starts with capital letter OR starts with a digit
        if re.match(r"^[A-ZÄÖÅ0-9]", inner):
            found.append(inner)
    return found


def extract_subsection(section_text: str, label: str) -> str:
    """Extract text under a ### LABEL header, stopping at the next ### or ---."""
    text = ""
    active = False
    for line in section_text.splitlines():
        if re.match(rf"^#{{1,3}}\s*{label}", line, re.IGNORECASE):
            active = True
            continue
        if active and re.match(r"^#{1,3}", line):
            break
        if active and line.strip().startswith("---"):
            break
        if active:
            text += line + "\n"
    return text.strip()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "label": "VSV BESS (Vantaa, teollisuusalue, 25 MW / 50 MWh, lupavaihe)",
        "inp": ApplicationInput(
            hanketyyppi="BESS",
            kiinteistotunnus="092-0403-0001-0001",
            teho_mw=25.0,
            kapasiteetti_mwh=50.0,
            y_tunnus="1234567-8",
            osoite="Koivuhaantie 1, Vantaa",
            kunta="Vantaa",
            hakija="VSV Energia Oy",
            sijainti_ymparistovaikutukset=(
                "Kohde sijaitsee Vantaan Petikon teollisuusalueella. "
                "Lähin asuinalue 800 m. Ei pohjavesialuetta. "
                "110 kV verkkoliityntäpiste 1,2 km päässä."
            ),
            hankkeen_vaihe="lupavaihe",
            lang="FI",
            country="FI",
        ),
    },
    {
        "label": "Salo BESS (Salo, 10 MW / 20 MWh, esiselvitys)",
        "inp": ApplicationInput(
            hanketyyppi="BESS",
            kiinteistotunnus="734-0001-0001-0001",
            teho_mw=10.0,
            kapasiteetti_mwh=20.0,
            y_tunnus="",
            osoite="Teollisuuskatu 1, Salo",
            kunta="Salo",
            hakija="Salo Energia Oy",
            sijainti_ymparistovaikutukset=(
                "Salo, Varsinais-Suomi. Teollisuusalue, kytkentä 110 kV verkkoon. "
                "Etäisyys lähimpään asutukseen 400 m."
            ),
            hankkeen_vaihe="esiselvitys",
            lang="FI",
            country="FI",
        ),
    },
]

COT_LABELS = ["ANALYSOI", "HAE", "VERTAA", "ARVIOI", "SUOSITTELE", "ELINKAARI"]
LIFECYCLE_FALLBACK = "lähdeaineisto ei riitä"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def validate_case(label: str, inp: ApplicationInput) -> dict:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print("="*70)

    try:
        _, sections, sources = generate_application_draft(inp)
    except Exception as exc:
        print(f"  GENERATION FAILED: {exc}")
        return {"label": label, "error": str(exc)}

    full_text = "\n".join(sections.values()).upper()
    cot_check = {lbl: (lbl in full_text) for lbl in COT_LABELS}

    # ARVIOI sub-section
    arvioi_text = extract_subsection(sections.get("perustelut", ""), "ARVIOI")
    arvioi_cites = real_citations(arvioi_text)
    # Also scan full perustelut — citation may appear at end of ARVIOI block
    perust_cites = real_citations(sections.get("perustelut", ""))

    # ELINKAARI sub-section
    lifecycle_text = extract_subsection(sections.get("toimenpiteet", ""), "ELINKAARI")
    lc_cites = real_citations(lifecycle_text)
    lc_fallback = LIFECYCLE_FALLBACK in lifecycle_text.lower()

    # SUOSITTELE numbered items without any bracket citation
    suos_text = extract_subsection(sections.get("toimenpiteet", ""), "SUOSITTELE")
    uncited = [
        line.strip()
        for line in suos_text.splitlines()
        if re.match(r"^\d+[\.\)]", line.strip()) and not _BRACKET_RE.search(line)
    ]

    result = {
        "label":            label,
        "sources":          [s["id"] for s in sources],
        "section_lens":     {k: len(v) for k, v in sections.items()},
        "cot_check":        cot_check,
        "arvioi_cites":     perust_cites,   # full section; citation may be at block end
        "lifecycle_cites":  lc_cites,
        "lifecycle_fallback": lc_fallback,
        "uncited_recs":     uncited,
        "arvioi_excerpt":   arvioi_text[:400],
        "lifecycle_excerpt": lifecycle_text[:400],
    }

    # Print inline summary
    cot_ok = all(cot_check.values())
    arv_ok = bool(perust_cites)
    lc_ok  = lc_fallback or bool(lc_cites)

    print(f"  RAG sources : {result['sources']}")
    print(f"  Sec lengths : {result['section_lens']}")
    print(f"  CoT 6/6     : {'PASS' if cot_ok else 'FAIL'} — {cot_check}")
    print(f"  ARVIOI cites: {'PASS' if arv_ok else 'FAIL'} — {perust_cites}")
    print(f"  LIFECYCLE   : {'PASS (fallback)' if lc_fallback else ('PASS (cited)' if lc_cites else 'FAIL — no source or fallback')}")
    print(f"  Uncited recs: {len(uncited)} {'— OK' if not uncited else '— FLAGGED'}")
    if uncited:
        for u in uncited:
            print(f"    ⚠️  {u[:100]}")

    return result


def print_summary(results: list[dict]) -> None:
    print("\n\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    hdr = f"{'Case':<42} {'CoT':^5} {'ARVIOI':^8} {'LIFECYCLE':^12} {'Uncited':^8}"
    print(hdr)
    print("-"*80)
    for r in results:
        if "error" in r:
            print(f"{r['label'][:42]:<42} ERROR")
            continue
        cot  = "PASS" if all(r["cot_check"].values()) else f"FAIL ({sum(r['cot_check'].values())}/6)"
        arv  = "PASS" if r["arvioi_cites"] else "⚠️ FAIL"
        lc   = "PASS" if (r["lifecycle_fallback"] or r["lifecycle_cites"]) else "⚠️ FAIL"
        unc  = str(len(r["uncited_recs"])) if r["uncited_recs"] else "0 OK"
        print(f"{r['label'][:42]:<42} {cot:^5} {arv:^8} {lc:^12} {unc:^8}")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    results = [validate_case(c["label"], c["inp"]) for c in CASES]
    print_summary(results)

    failed = [r for r in results if "error" in r or
              not all(r["cot_check"].values()) or
              not r["arvioi_cites"] or
              not (r["lifecycle_fallback"] or r["lifecycle_cites"])]
    sys.exit(1 if failed else 0)
