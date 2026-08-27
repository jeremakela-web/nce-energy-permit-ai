#!/usr/bin/env python3
"""Validate country/language coverage in the two frontend JS tables this
bug class has hit before: LUPA_I18N and TRANSLATIONS, both in
backend/static/index.html (the actual app frontend -- NOT the root
index.html, which is the marketing landing page; confirmed by grep before
writing this, both files exist and only one has these tables).

Companion to scripts/validate_country_coverage.py -- kept separate because
these two tables live in JS, not Python, and are parsed with a lightweight
brace-depth extractor rather than imported as real objects. See
BUG_CONSOLIDATION_ARCHITECTURE_PROPOSAL.md §2.4 for why.

Two different coverage shapes:
  - LUPA_I18N: keyed by Finnish permit-name string, each row a {LANG: text}
    object. Row-shaped, same as _LAW_TRANS/_HANKE_NIMI_TRANS on the Python
    side -- every row needs every non-FI supported language.
  - TRANSLATIONS: keyed by language code, each value a flat {key: text}
    object. The gap here isn't a missing top-level language (all of
    FI/EN/SE/DA/NO/PL/DE/ET/LV/LT exist as keys) -- it's individual
    translation KEYS missing inside a language's block relative to FI's
    key set. This is exactly the shape of the historical TRANSLATIONS.ET
    63/398 bug: present as a language, mostly empty inside.

Usage:
    python scripts/validate_frontend_coverage.py [path/to/index.html]

Exits 0 if no HARD gaps found, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
# permit_ai/ also needs to be on sys.path directly so this matches the flat
# `from supported_locales import ...` convention used throughout permit_ai/
# (dotted `permit_ai.X` imports break whenever backend/main.py is also on
# the path, since backend/permit_ai.py -- an unrelated RAG-query module --
# shadows the bare "permit_ai" name there; not a risk for this standalone
# script today, but matching the convention avoids relying on that).
sys.path.insert(0, str(_REPO_ROOT / "permit_ai"))

# Deliberately imports the lightweight, zero-dependency supported_locales
# module, NOT country_registry -- this is a pure-JS/HTML check and has no
# reason to require country_registry's generate_application import chain
# (chromadb, torch, etc.) just to read two constant tuples.
from supported_locales import _SUPPORTED_LANGUAGES  # noqa: E402

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "backend" / "static" / "index.html"


def _extract_braced_block(text: str, start_marker: str) -> str:
    """Return the full `{ ... }` block (braces included) immediately
    following start_marker, using brace-depth counting. Assumes no literal
    unescaped `{`/`}` characters appear inside string values -- confirmed
    true for both tables checked here before writing this."""
    idx = text.index(start_marker)
    brace_start = text.index("{", idx)
    depth = 0
    i = brace_start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1]
        i += 1
    raise ValueError(f"Unbalanced braces after {start_marker!r}")


def _split_top_level_entries(block: str) -> list[tuple[str, str]]:
    """Given a `{ key1: {...}, key2: {...}, ... }` block, return
    [(key, inner_block), ...] for each top-level entry, via brace-depth
    tracking (so nested braces inside a value don't confuse the split)."""
    inner = block[1:-1]  # strip outer { }
    entries = []
    i = 0
    n = len(inner)
    key_re = re.compile(r"""\s*(?://[^\n]*\n\s*)*(?:'([^']+)'|"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*))\s*:\s*""")
    while i < n:
        m = key_re.match(inner, i)
        if not m:
            i += 1
            continue
        key = m.group(1) or m.group(2) or m.group(3)
        j = m.end()
        if j >= n or inner[j] != "{":
            # Not an object value (e.g. a stray comment/const) -- skip to next comma at depth 0.
            i = j
            continue
        depth = 0
        k = j
        while k < n:
            if inner[k] == "{":
                depth += 1
            elif inner[k] == "}":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        entries.append((key, inner[j:k]))
        i = k
    return entries


def check_lupa_i18n(html: str) -> list[str]:
    """Every LUPA_I18N row must have every supported language except FI
    (the row key itself is the Finnish string)."""
    block = _extract_braced_block(html, "const LUPA_I18N = ")
    rows = _split_top_level_entries(block)
    needed = set(_SUPPORTED_LANGUAGES) - {"FI"}
    problems = []
    for row_key, row_block in rows:
        present = set(re.findall(r"\b([A-Z]{2}):", row_block))
        missing = needed - present
        if missing:
            problems.append(f"LUPA_I18N[{row_key!r}]: missing {sorted(missing)}")
    return problems


def check_translations(html: str) -> list[str]:
    """Every TRANSLATIONS[lang] block must have the same key set as
    TRANSLATIONS.FI (the reference/most-complete language by construction).
    Reports missing keys per language, not missing languages -- that's the
    bug this dict actually has."""
    block = _extract_braced_block(html, "const TRANSLATIONS = ")
    langs = _split_top_level_entries(block)
    lang_keys: dict[str, set[str]] = {}
    key_re = re.compile(r"^\s*(?:'([^']+)'|([A-Za-z_][A-Za-z0-9_]*))\s*:", re.M)
    for lang, lang_block in langs:
        keys = set(m.group(1) or m.group(2) for m in key_re.finditer(lang_block))
        lang_keys[lang] = keys

    if "FI" not in lang_keys:
        return ["TRANSLATIONS: no FI block found -- can't establish a reference key set"]

    reference = lang_keys["FI"]
    problems = []
    for lang in _SUPPORTED_LANGUAGES:
        if lang not in lang_keys:
            problems.append(f"TRANSLATIONS: missing language block entirely: {lang!r}")
            continue
        missing = reference - lang_keys[lang]
        if missing:
            pct = 100 * len(lang_keys[lang]) / len(reference) if reference else 0
            problems.append(
                f"TRANSLATIONS.{lang}: {len(lang_keys[lang])}/{len(reference)} keys "
                f"({pct:.0f}%) -- missing {len(missing)}, e.g. {sorted(missing)[:5]}"
            )
    return problems


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_PATH
    html = path.read_text(encoding="utf-8")

    problems = check_lupa_i18n(html) + check_translations(html)

    print(f"Frontend coverage check: {path}\n")
    if not problems:
        print("No gaps found.")
        return 0

    print(f"{len(problems)} gap(s) found:")
    for p in problems:
        print(f"  {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
