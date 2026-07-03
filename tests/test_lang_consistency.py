#!/usr/bin/env python3
"""Automated language-consistency check for non-FI permit reports.

Usage:
  python tests/test_lang_consistency.py                  # generate + check FI/BESS/EN
  python tests/test_lang_consistency.py path/to/file.pdf # check existing PDF
  python tests/test_lang_consistency.py --live           # hit production API

Exits 0 if clean, 1 if Finnish markers found.
"""

import sys
import os
import re

# Make sure permit_ai package is importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from permit_ai.generate_application import check_fi_language_leak, _FI_LEAK_MARKERS


def extract_pdf_text(pdf_path: str) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        print("pypdf not available — install with: pip install pypdf")
        sys.exit(2)


def check_pdf(pdf_path: str) -> list[str]:
    print(f"Checking: {pdf_path}")
    text = extract_pdf_text(pdf_path)
    leaks = check_fi_language_leak(text)
    return leaks


def generate_and_check(use_live: bool = False) -> bool:
    """Generate FI/BESS/EN report via API and check for Finnish leaks.
    Returns True if clean."""
    import json
    import time
    import urllib.request

    base = "https://nce-energy-permit-ai.onrender.com" if use_live else "http://localhost:8000"

    payload = json.dumps({
        "hanketyyppi": "BESS",
        "country": "FI",
        "lang": "EN",
        "kiinteistotunnus": "049-001-0001-0001",
        "teho_mw": 10,
        "kapasiteetti_mwh": 20,
        "kunta": "Espoo",
        "hakija": "NCE Test Ltd",
    }).encode()

    print(f"Submitting FI/BESS/EN job to {base}...")
    req = urllib.request.Request(
        f"{base}/api/generate-application",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        job_id = json.loads(r.read())["job_id"]
    print(f"  job_id: {job_id}")

    for _ in range(72):
        time.sleep(5)
        with urllib.request.urlopen(f"{base}/api/proofread/{job_id}", timeout=10) as r:
            status = json.loads(r.read()).get("status", "?")
        if status == "done":
            break
        if status == "error":
            print("  ERROR during generation")
            return False

    pdf_path = f"/tmp/lang_check_{job_id}.pdf"
    with urllib.request.urlopen(f"{base}/api/proofread/{job_id}/download", timeout=30) as r:
        with open(pdf_path, "wb") as f:
            f.write(r.read())

    leaks = check_pdf(pdf_path)
    os.unlink(pdf_path)
    return _report(leaks)


def _report(leaks: list[str]) -> bool:
    print(f"\nFinnish leak markers checked ({len(_FI_LEAK_MARKERS)} markers):")
    if not leaks:
        print("  CLEAN — no Finnish markers found in non-FI output")
        return True
    print(f"  FAIL — {len(leaks)} match(es) found:")
    for hit in leaks[:20]:
        print(f"    {hit}")
    return False


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and os.path.isfile(args[0]):
        leaks = check_pdf(args[0])
        ok = _report(leaks)
        sys.exit(0 if ok else 1)

    use_live = "--live" in args
    ok = generate_and_check(use_live=use_live)
    sys.exit(0 if ok else 1)
