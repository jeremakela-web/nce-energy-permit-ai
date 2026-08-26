"""
supported_locales.py — the one canonical list of countries/languages this
app supports. Deliberately zero-dependency (no import of
generate_application or anything heavy) so both the backend coverage
validator and the frontend (JS-side) coverage validator can share it without
either one needing the other's dependency chain — the frontend validator in
particular has no reason to require chromadb/torch/etc. just to check
country codes.

Everything else in permit_ai/country_registry.py derives from this. See
BUG_CONSOLIDATION_ARCHITECTURE_PROPOSAL.md for the full rationale.
"""

from __future__ import annotations

_SUPPORTED_COUNTRIES: tuple[str, ...] = ("FI", "SE", "DA", "NO", "PL", "DE", "EE", "LV", "LT")
_SUPPORTED_LANGUAGES: tuple[str, ...] = ("FI", "EN", "SE", "DA", "NO", "PL", "DE", "ET", "LV", "LT")
