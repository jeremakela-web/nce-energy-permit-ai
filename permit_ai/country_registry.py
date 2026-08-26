"""
country_registry.py — canonical country/language support list + coverage
manifest for permit_ai.generate_application's country/language-keyed dicts.

Exists to catch the recurring "some countries/languages have an entry, one
doesn't, and it silently falls back to Finnish content" bug class before
merge, not after a customer or manual QA pass finds it live. Seven real,
named instances of this exact bug are documented in
BUG_CONSOLIDATION_ARCHITECTURE_PROPOSAL.md, which is the design rationale
for everything in this module — read that first if the "why" isn't obvious
from a comment here.

This module does NOT own or duplicate any data. It only references the real
dicts in generate_application.py and declares what "complete" means for each
one. Fixing a reported gap always means editing generate_application.py (or
the frontend JS, for the two dicts covered by validate_frontend_coverage.py)
— never this file.

One deliberate scope boundary: _COUNTRY_LUVAT and _COUNTRY_LIITTEET are
registered as HARD/country-axis (does the country exist as a top-level key
at all), but do NOT yet get a per-(country x hanketyyppi) "is this
hanketyyppi legitimately offered to this country" allowlist -- that needs a
real source of truth for which hanketyyppi each country is meant to offer,
which depends on the in-progress frontend intake-form work and isn't
resolved yet (see BUG_CONSOLIDATION_ARCHITECTURE_PROPOSAL.md §2.2's caveat
under those two dicts). Building that allowlist against today's live state
risks going stale the moment the form scope changes -- exactly the failure
mode this whole module exists to prevent, one level up. Left as a TODO,
flagged loudly in validate_country_coverage.py's own output rather than
silently guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from permit_ai import generate_application as ga
from permit_ai.supported_locales import _SUPPORTED_COUNTRIES, _SUPPORTED_LANGUAGES  # noqa: F401 (re-exported)

Axis = Literal["country", "language"]
Severity = Literal["HARD", "SOFT"]
Granularity = Literal["whole", "per_row"]


@dataclass
class CoverageSpec:
    name: str
    dict_obj: dict
    axis: Axis
    severity: Severity
    granularity: Granularity = "whole"
    # "whole": verified_absent is a flat set of codes for the whole dict.
    # "per_row": dict_obj's values are themselves code-keyed dicts (e.g.
    #   _LAW_TRANS's "MRL 132/1999" -> {"SE": ..., "EE": ...}); each row is
    #   checked independently, and verified_absent is {row_key: {codes}}.
    verified_absent: set | dict = field(default_factory=set)
    # For conditional SOFT specs whose legitimacy depends on another dict's
    # live content (currently only _STUK_REPLACEMENT) rather than a static
    # allowlist that could silently drift out of sync.
    conditional_ok: Callable[[str], bool] | None = None
    # Codes to drop from the "required" set for this whole spec. Needed for
    # _HANKE_NIMI_TRANS, _LAW_TRANS, and _LAW_CITATION_REPLACEMENT: each row
    # key in those three dicts IS the Finnish content itself (a hanketyyppi
    # slug or a bare Finnish statute citation), so there is no separate "FI"
    # value to check for -- checking for one would be a false-positive gap
    # on every single row. Confirmed directly against the real dicts before
    # writing this, not assumed.
    exclude_from_required: frozenset[str] = frozenset()


def _stuk_absence_is_legitimate(code: str) -> bool:
    """A country's absence from _STUK_REPLACEMENT is only legitimate if none
    of its _COUNTRY_LIITTEET SMR / smr_<cc> entries contain a literal "STUK"
    substring -- i.e. there's genuinely nothing left for the backstop to
    replace. Checked live against _COUNTRY_LIITTEET's *current* content on
    every run, not a static allowlist -- so it can't go stale the next time
    someone edits _COUNTRY_LIITTEET without knowing this dependency exists."""
    liite = ga._COUNTRY_LIITTEET.get(code, {})
    for hanke_key, items in liite.items():
        if hanke_key != "SMR" and not hanke_key.startswith("smr_"):
            continue
        for item in items:
            if "STUK" in item:
                return False
    return True


# ── Verified-absent sets, sourced from the code's own correction comments ───
# Each entry here mirrors a real, already-written comment in
# generate_application.py explaining why a specific country's row is
# deliberately incomplete. Kept here (not re-derived from the comments
# themselves) so this stays a plain, auditable data structure -- but every
# entry below must point at a real comment; do not add one without also
# adding the matching explanation in generate_application.py.
_LAW_CITATION_REPLACEMENT_VERIFIED_ABSENT: dict[str, set] = {
    # PL/LT: no separate dam-safety act exists (folded into general
    # water-law/construction-technical-regulation framework instead).
    # DA: genuinely unconfirmed after two targeted searches, left open
    # rather than guessed.
    # EE: explicitly out of scope for the 2026-08-26 research pass (see
    # EE_LAW_CITATION_RESEARCH.md's scope-boundary section).
    "Patoturvallisuuslaki": {"PL", "LT", "DA", "EE"},
}
_LAW_TRANS_VERIFIED_ABSENT: dict[str, set] = {
    # Same statute, same reasons, mirrored here because _LAW_TRANS and
    # _LAW_CITATION_REPLACEMENT are two independent tables covering
    # overlapping but not identical statute-citation keys.
    "Patoturvallisuuslaki 494/2009": {"PL", "LT", "DA", "EE"},
}


_COVERAGE_MANIFEST: list[CoverageSpec] = [
    # ── Language axis, whole-dict, HARD ─────────────────────────────────────
    CoverageSpec("_LANG_INSTRUCTIONS", ga._LANG_INSTRUCTIONS, "language", "HARD"),
    CoverageSpec("_WRITE_INSTRUCTION", ga._WRITE_INSTRUCTION, "language", "HARD"),
    CoverageSpec("_PROMPT_HEADERS", ga._PROMPT_HEADERS, "language", "HARD"),
    CoverageSpec("_PDF_STRINGS", ga._PDF_STRINGS, "language", "HARD"),
    CoverageSpec("_CRITICAL_EXTRA", ga._CRITICAL_EXTRA, "language", "HARD"),
    # SOFT and verified legitimate: falls back to "[Note] ", a short, safe,
    # generic English marker -- confirmed not Finnish text, kept incomplete
    # deliberately.
    CoverageSpec("_HUOM_LABEL", ga._HUOM_LABEL, "language", "SOFT",
                 verified_absent=set(_SUPPORTED_LANGUAGES)),

    # ── Language axis, PER-ROW, HARD ────────────────────────────────────────
    # _HANKE_NIMI_TRANS is keyed by hanketyyppi ("BESS", "tuulivoima_maa",
    # ...), each value a {lang: name} dict -- same per-row shape as
    # _LAW_TRANS below, needs the same per-row check, not a flat
    # whole-dict-language check (which would only see the union of
    # languages across all rows, not whether any single row is short one).
    CoverageSpec("_HANKE_NIMI_TRANS", ga._HANKE_NIMI_TRANS, "language", "HARD",
                 granularity="per_row", exclude_from_required=frozenset({"FI"})),

    # ── Country axis, whole-dict, HARD ──────────────────────────────────────
    CoverageSpec("_COUNTRY_CONFIG", ga._COUNTRY_CONFIG, "country", "HARD"),
    # See module docstring: HARD at "does the country key exist at all" --
    # the finer (country x hanketyyppi) allowlist is a deliberate TODO.
    # exclude_from_required={"FI"}: confirmed by running the validator for
    # real against generate_application.py's actual dicts (2026-08-26) --
    # both dicts exist SPECIFICALLY to hold non-FI country overrides on top
    # of Finland's own default content (built elsewhere, e.g. _HANKE_CFG);
    # "FI" is never meant to be a key here. Without this exclusion the first
    # real run reported a false-positive gap on both. Caught by dogfooding,
    # not assumed correct on the first pass.
    CoverageSpec("_COUNTRY_LUVAT", ga._COUNTRY_LUVAT, "country", "HARD",
                 exclude_from_required=frozenset({"FI"})),
    CoverageSpec("_COUNTRY_LIITTEET", ga._COUNTRY_LIITTEET, "country", "HARD",
                 exclude_from_required=frozenset({"FI"})),
    CoverageSpec("_NATIONAL_SUPERVISORS", ga._NATIONAL_SUPERVISORS, "country", "HARD"),
    CoverageSpec("_BESS_MARKET_DATA", ga._BESS_MARKET_DATA, "country", "HARD"),

    # ── Country axis, PER-ROW, HARD ─────────────────────────────────────────
    # Corrected from an earlier draft's whole-dict SOFT/"allow_partial"
    # classification: _LAW_TRANS and _LAW_CITATION_REPLACEMENT are each
    # ~35 independent statute-keyed dicts, not one country-keyed dict. A
    # whole-dict check would have missed PR #106 (SE law citations) exactly
    # -- some statute-rows had an SE entry, others didn't, which only shows
    # up checking every row. Per-row verified_absent sets (above) carry the
    # small number of genuinely-legitimate per-statute gaps (e.g.
    # Patoturvallisuuslaki's PL/LT/DA/EE absences); everything else in these
    # two dicts is HARD.
    # Both of these are also missing "FI" from every row, same reason as
    # _HANKE_NIMI_TRANS above -- the row key IS the Finnish statute citation.
    CoverageSpec("_LAW_TRANS", ga._LAW_TRANS, "country", "HARD",
                 granularity="per_row", verified_absent=_LAW_TRANS_VERIFIED_ABSENT,
                 exclude_from_required=frozenset({"FI"})),
    CoverageSpec("_LAW_CITATION_REPLACEMENT", ga._LAW_CITATION_REPLACEMENT, "country", "HARD",
                 granularity="per_row", verified_absent=_LAW_CITATION_REPLACEMENT_VERIFIED_ABSENT,
                 exclude_from_required=frozenset({"FI"})),

    # ── Country axis, whole-dict, SOFT, CONDITIONAL ─────────────────────────
    # Corrected from an earlier draft's unconditional SOFT: legitimacy
    # depends on _COUNTRY_LIITTEET's live content (see
    # _stuk_absence_is_legitimate above), not a fixed fact -- checked fresh
    # every run instead of trusted from a static allowlist.
    CoverageSpec("_STUK_REPLACEMENT", ga._STUK_REPLACEMENT, "country", "SOFT",
                 conditional_ok=_stuk_absence_is_legitimate),
]
