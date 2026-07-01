"""
Single source of truth for NCE Permit AI source relevance and doc_type rules.

All ingest scripts, build_index.py, and the runtime RAG filter import from here.
Adding a new restricted source = add one line to SOURCE_HANKETYYPPI_TAG.
"""
from __future__ import annotations

# ── Doc-type map: source stem → laki / viranomaisohje / ennakkotapaus ───────
DOC_TYPE_MAP: dict[str, str] = {
    # Lait ja asetukset
    "rakentamislaki_751_2023":            "laki",
    "kemikaaliturvallisuuslaki_390_2005": "laki",
    "pelastuslaki_379_2011":              "laki",
    # Viranomaisohjeet
    "fingrid_liittyminen_kantaverkkoon":  "viranomaisohje",
    "tukes_liion_opas":                   "viranomaisohje",
    "tukes_painelaitteet":                "viranomaisohje",
    "tukes_painelaitteet_sco2":           "viranomaisohje",
    "energiavirasto_energiatehokkuus":    "viranomaisohje",
    "ym_datakeskukset":                   "viranomaisohje",
    "datakeskus_luvat_suomi":             "viranomaisohje",
    "YVL_A.1":                            "viranomaisohje",
    "YVL_B.1":                            "viranomaisohje",
    "YVL_C.1":                            "viranomaisohje",
    "IAEA_NS-R-5":                        "viranomaisohje",
    "IAEA_SSG-52":                        "viranomaisohje",
    "IAEA_SSR-2_1":                       "viranomaisohje",
    "lion_2025_bess":                     "viranomaisohje",
    "lion_teollisuus_2025":               "viranomaisohje",
    "sjv2024_fingrid":                    "viranomaisohje",
    "vjv2024_fingrid":                    "viranomaisohje",
    "caruna_network_development_plan_2026": "viranomaisohje",
    "bios_datakeskus_sijoittamislupa":    "viranomaisohje",
    "microsoft_espoo_yva_selostus":       "viranomaisohje",
    "rakentamislaki_sijoittamislupa_datakeskus": "viranomaisohje",
    "ymparistolupa_datakeskus_ysl":       "viranomaisohje",
}

# ── Hanketyyppi tag map ───────────────────────────────────────────────────────
# Maps source stem → comma-separated list of project types that may use it.
# "general" (default when not in this map) = unrestricted, all project types.
# Use comma-separated values when a source is relevant for multiple but not all types.
SOURCE_HANKETYYPPI_TAG: dict[str, str] = {
    # Nuclear safety guides — SMR / smr_bess only
    "YVL_A.1":      "SMR,smr_bess",
    "YVL_B.1":      "SMR,smr_bess",
    "YVL_C.1":      "SMR,smr_bess",
    "IAEA_NS-R-5":  "SMR,smr_bess",
    "IAEA_SSG-52":  "SMR,smr_bess",
    "IAEA_SSR-2_1": "SMR,smr_bess",
    # Data-centre-specific documents
    "bios_datakeskus_sijoittamislupa":           "datakeskus",
    "microsoft_espoo_yva_selostus":              "datakeskus",
    "rakentamislaki_sijoittamislupa_datakeskus": "datakeskus",
    "ymparistolupa_datakeskus_ysl":              "datakeskus",
    "ym_datakeskukset":                          "datakeskus",
    "datakeskus_luvat_suomi":                    "datakeskus",
    # Fingrid transmission-grid (kantaverkko) connection — for projects that connect
    # directly to the 110/400 kV grid. BESS / aurinkovoima use distribution grid
    # (jakeluverkko, typically Carunan 20 kV).
    "fingrid_liittyminen_kantaverkkoon": "tuulivoima_maa,tuulivoima_meri,SMR,smr_bess,teollisuus",
}


def get_doc_type(source_name: str) -> str:
    """Return doc_type for a source stem. Defaults to 'viranomaisohje'."""
    return DOC_TYPE_MAP.get(source_name, "viranomaisohje")


def get_hanketyyppi_tag(source_name: str) -> str:
    """Return hanketyyppi_tag for a source stem. Defaults to 'general'."""
    return SOURCE_HANKETYYPPI_TAG.get(source_name, "general")


def is_chunk_relevant(chunk_meta: dict, current_hanketyyppi: str) -> bool:
    """Return True if a chunk should be included when generating a current_hanketyyppi report.

    Precedence:
    1. chunk_meta["hanketyyppi_tag"]  — set by all new/updated ingestion
    2. chunk_meta["project_types"]    — set by ingest_playwright / ingest_precedent (fallback)
    3. Name-based lookup via SOURCE_HANKETYYPPI_TAG (pre-migration chunks)
    4. Default True — unknown sources are not filtered out
    """
    tag = (
        chunk_meta.get("hanketyyppi_tag")
        or chunk_meta.get("project_types")
        or get_hanketyyppi_tag(chunk_meta.get("source", ""))
    )
    if not tag or tag in ("general", "all"):
        return True
    allowed = {t.strip() for t in tag.split(",")}
    return current_hanketyyppi in allowed
