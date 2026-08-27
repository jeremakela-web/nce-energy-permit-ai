#!/usr/bin/env python3
"""Validate country/language coverage across permit_ai.generate_application's
country/language-keyed dicts, per the manifest in permit_ai/country_registry.py.

Catches the "some countries/languages have an entry, one doesn't, and it
silently falls back to Finnish content" bug class before merge -- see
BUG_CONSOLIDATION_ARCHITECTURE_PROPOSAL.md for the full rationale and the
seven real, named incidents this is meant to prevent a recurrence of.

Usage:
    python scripts/validate_country_coverage.py
    python scripts/validate_country_coverage.py --json   # machine-readable

Exits 0 if no HARD gaps found, 1 otherwise. SOFT gaps are reported but never
fail the run.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
# country_registry.py does a flat `import generate_application` (not
# `permit_ai.generate_application`) to match backend/main.py's own import
# convention and avoid double-loading the module under two different names
# -- see country_registry.py's own comment. permit_ai/ needs to be on
# sys.path directly for that flat import to resolve when this script runs
# standalone, same as backend/main.py arranges for itself at startup.
sys.path.insert(0, os.path.join(_REPO_ROOT, "permit_ai"))

from permit_ai.country_registry import (  # noqa: E402
    _COVERAGE_MANIFEST,
    _SUPPORTED_COUNTRIES,
    _SUPPORTED_LANGUAGES,
    CoverageSpec,
)


@dataclass
class Gap:
    spec_name: str
    row: str | None       # None for "whole" granularity, the row key for "per_row"
    missing: set[str]
    soft: bool


def _supported_for(spec: CoverageSpec) -> set[str]:
    base = _SUPPORTED_COUNTRIES if spec.axis == "country" else _SUPPORTED_LANGUAGES
    return set(base) - set(spec.exclude_from_required)


def _check_whole(spec: CoverageSpec) -> list[Gap]:
    supported = set(_supported_for(spec))
    present = set(spec.dict_obj.keys())
    missing = supported - present

    if spec.conditional_ok is not None:
        # Only codes where the live condition says "not actually legitimate"
        # count as real gaps.
        missing = {code for code in missing if not spec.conditional_ok(code)}
        if missing:
            return [Gap(spec.name, None, missing, soft=(spec.severity == "SOFT"))]
        return []

    verified_absent = spec.verified_absent if isinstance(spec.verified_absent, set) else set()
    real_missing = missing - verified_absent
    if not real_missing:
        return []
    return [Gap(spec.name, None, real_missing, soft=(spec.severity == "SOFT"))]


def _check_per_row(spec: CoverageSpec) -> list[Gap]:
    supported = set(_supported_for(spec))
    verified_absent_by_row = spec.verified_absent if isinstance(spec.verified_absent, dict) else {}
    gaps = []
    for row_key, row_dict in spec.dict_obj.items():
        if not isinstance(row_dict, dict):
            continue
        present = set(row_dict.keys())
        missing = supported - present
        real_missing = missing - verified_absent_by_row.get(row_key, set())
        if real_missing:
            gaps.append(Gap(spec.name, row_key, real_missing, soft=(spec.severity == "SOFT")))
    return gaps


def validate() -> list[Gap]:
    gaps: list[Gap] = []
    for spec in _COVERAGE_MANIFEST:
        checker = _check_per_row if spec.granularity == "per_row" else _check_whole
        gaps.extend(checker(spec))
    return gaps


def _print_report(gaps: list[Gap]) -> None:
    hard = [g for g in gaps if not g.soft]
    soft = [g for g in gaps if g.soft]

    print(f"Coverage check: {len(_COVERAGE_MANIFEST)} dicts, "
          f"{len(_SUPPORTED_COUNTRIES)} countries, {len(_SUPPORTED_LANGUAGES)} languages\n")

    if hard:
        print(f"HARD gaps ({len(hard)}) — must be fixed before merge:")
        for g in hard:
            where = f"{g.spec_name}[{g.row!r}]" if g.row else g.spec_name
            print(f"  {where}: missing {sorted(g.missing)}")
    else:
        print("No HARD gaps.")

    if soft:
        print(f"\nSOFT gaps ({len(soft)}) — not blocking, but not verified-absent either:")
        for g in soft:
            where = f"{g.spec_name}[{g.row!r}]" if g.row else g.spec_name
            print(f"  {where}: missing {sorted(g.missing)}")

    print(
        "\nNOTE: _COUNTRY_LUVAT / _COUNTRY_LIITTEET are checked at the "
        "'does this country exist as a top-level key' level only. Whether "
        "every hanketyyppi is legitimately offered to every country is a "
        "deliberate open TODO -- see permit_ai/country_registry.py's module "
        "docstring and BUG_CONSOLIDATION_ARCHITECTURE_PROPOSAL.md §2.2."
    )


def main() -> int:
    gaps = validate()
    if "--json" in sys.argv:
        print(json.dumps([
            {"spec": g.spec_name, "row": g.row, "missing": sorted(g.missing), "soft": g.soft}
            for g in gaps
        ], indent=2))
    else:
        _print_report(gaps)
    return 1 if any(not g.soft for g in gaps) else 0


if __name__ == "__main__":
    sys.exit(main())
