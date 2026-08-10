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
    # Priority-2 maatalous/vesivoima content (2026-08-10) --
    # permit_ai/ingest_maatalous_vesivoima.py. "laki" for verbatim/
    # near-verbatim statutory text pulled from a primary source; asetukset
    # (ministerial decrees) are classified "laki" too since this map has no
    # separate "asetus" tier -- they're genuinely binding delegated
    # legislation, closer to "laki" than to non-binding "viranomaisohje".
    "ysl_527_2014_liite1_elainsuoja":        "laki",
    "mmm_610_2023_lypsykarjarakennukset":    "laki",
    "vesilaki_587_2011_kalatalousvelvoite":  "laki",
    "patoturvallisuuslaki_494_2009":         "laki",
    # NOT primary statute text -- see ingest_maatalous_vesivoima.py's
    # docstring and this PR's commit message. Genuine guidance content,
    # correctly categorized as viranomaisohje rather than laki so the
    # distinction stays visible in doc_type, not just the source label.
    "ruokavirasto_maatalouden_investointituet": "viranomaisohje",
    "nitraattiasetus_1250_2014_valvontaohje":   "viranomaisohje",
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
    # DA — nuclear/radiation safety (Sikkerhedsstyrelsen) and datacenter
    "sik_nuclear_smr_stralebebeskyttelse": "SMR",
    "sik_sco2_tryk_modulaert_kraftanlaeg": "SMR",
    "sik_stralebeskyttelse":               "SMR",
    "ens_datacentre_energikrav":           "datakeskus",
    # NO — nuclear regulatory (DSA/NVE)
    "dsa_nve_smr_nuclear_regulatory":      "SMR",
    # EE — nuclear regulatory draft and mixed BESS/datacenter/industrial permitting
    "smr_nuclear_regulatory_estonia_draft":          "SMR",
    "bess_datakeskus_teollisuus_ehitus_permitting":  "BESS,datakeskus,teollisuus",
    # LV — energy-type-specific sources (Latvia has no nuclear)
    # Transmission grid connection (AST) — energy projects only, not housing/agriculture
    "lv_ast_sistemas_pievienošanas_noteikumi":       "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,teollisuus,SMR",
    # Distribution grid (Sadales tīkls) — smaller energy projects, not SMR/industrial
    "lv_sadales_tikls_pievienošanas_kārtiba":        "BESS,aurinkovoima,tuulivoima_maa",
    # SPRK licensing — energy generation/storage projects only
    "lv_sprk_elektroenerģijas_ražošanas_licence":    "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima",
    "lv_mk_631_licencesanas_noteikumi":              "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,teollisuus",
    # Renewables law — only for renewable energy and co-located BESS
    "lv_atjaunojamas_energijas_likums":              "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima",
    # BESS market/balancing — BESS only
    "lv_bess_balansesanas_tirgus_ast":               "BESS",
    # Wind-specific regulations
    "lv_veja_energija_buvniecibas_noteikumi":        "tuulivoima_maa,tuulivoima_meri",
    # BVKB wind turbine regulatory overview — wind only
    "lv_bvkb_veja_elektrostacija":                  "tuulivoima_maa,tuulivoima_meri",
    # BESS fire safety — BESS and datakeskus (both use large Li-ion)
    "lv_ugunsdrošibas_bess_prasibas":               "BESS,datakeskus",
    # LT — energy-type-specific sources (Lithuania has no nuclear power plants)
    # Transmission grid connection (LITGRID) — energy projects only
    "lt_litgrid_prisijungimo_tvarka":                "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,teollisuus",
    # Distribution grid (ESO) — smaller energy projects, not SMR/industrial
    "lt_eso_gamintojams_prisijungimas":              "BESS,aurinkovoima,tuulivoima_maa",
    # VERT generation licensing — energy generation/storage projects only
    "lt_vert_elektros_gamybos_licencija":            "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima",
    # VERT OZE auctions — renewables and co-located BESS
    "lt_vert_oze_aukcionai_kvota":                   "tuulivoima_maa,tuulivoima_meri,aurinkovoima",
    # Renewables law — only for renewable energy and co-located BESS
    "lt_atsinaujinancio_resurso_energetikos_istatymas": "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima",
    # BESS balancing market — BESS only
    "lt_litgrid_balansavimo_rinka":                  "BESS",
    # Wind energy special plan — wind only
    "lt_vejo_energetikos_specialusis_planas":        "tuulivoima_maa,tuulivoima_meri",
    "lt_vert_vejo_energetika":                       "tuulivoima_maa,tuulivoima_meri",
    # BESS fire safety — BESS and datakeskus (both use large Li-ion batteries)
    "lt_priesgaisrines_saugos_taisykles":            "BESS,datakeskus",
    # EIA law — all energy projects requiring PAV (BESS, wind, solar, SMR)
    "lt_pav_istatymas":                              "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR",
    "lt_pav_kategoriju_sarasas":                     "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR",
    # SE — BESS fire safety (RISE 2023:117 + Boverket BFS 2024:7) — BESS and datakeskus
    "boverket_rise_bess_brandsakerhet":              "BESS,datakeskus",
    # DA — BESS fire safety (Beredskabsstyrelsen 2023 vejledning + DBI gap analysis 2025)
    "beredskabsstyrelsen_dbi_bess_brandsikkerhed":   "BESS,datakeskus",
    # NO — BESS fire safety (NEK 488:2024, NEK 487:2022, TEK17 kap. 11, DSB 2021)
    "nek_dsb_bess_brannsikkerhet":                   "BESS,datakeskus",
    # EE — BESS fire safety (Päästeamet Dec 2024 guidance, Tuleohutuse seadus, EVS 812-7)
    "paasteamet_bess_tuleohutus":                    "BESS,datakeskus",
    # DE — PR-TAG-1 mistagging fix (2026-08-09/10). All 12 DE sources were
    # previously untagged and fell through to "general" (visible to every
    # hanketyyppi regardless of relevance) -- see the platform coverage
    # audit. Classified from actual chunk content, not filenames alone.
    # Type-specific (bucket a):
    "bess_stromspeicher_genehmigung":                "BESS,hybridi",
    "rechenzentrum_datenzentrum_genehmigung":        "datakeskus",
    "smr_nuclear_regulatory_germany_regwarning":     "SMR",
    "solar_photovoltaik_genehmigung":                "aurinkovoima,hybridi",
    "windenergie_offshore_windseeG":                 "tuulivoima_meri,hybridi",
    "windenergie_onshore_genehmigung":               "tuulivoima_maa,hybridi",
    # Broad energy-facility-scoped law (bucket b) -- each chunk explicitly
    # self-scopes to "Energieanlagen"/"Energieprojekte" permitting, not
    # general-purpose German law, confirmed by reading actual chunk text.
    "umweltrecht_naturschutz_deutschland":           "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "baugb_planungsrecht_raumordnung":                "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "bimschg_immissionsschutzgesetz":                "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "bnetza_netzanschluss_anforderungen":            "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    # EEG 2023 — renewables-only by legal definition; excludes bare BESS
    # (standalone storage isn't EEG-eligible without co-located generation)
    # and SMR (nuclear isn't "erneuerbar").
    "eeg_2023_erneuerbare_energien_gesetz":          "tuulivoima_maa,tuulivoima_meri,aurinkovoima,vesivoima,hybridi",
    # wasserstoff_power_to_x_regulierung (bucket c) deliberately NOT tagged
    # here -- hydrogen/Power-to-X has no matching hanketyyppi in _HANKE_CFG.
    # Stays "general" (unchanged from before this fix), flagged to user as
    # a future product-scope question, not guessed at.
    # EE — PR-TAG-2 mistagging fix (2026-08-10). 7 sources previously
    # untagged (a further 3 EE sources already carried explicit tags and
    # were out of scope). Classified from actual chunk content.
    # Type-specific (bucket a):
    "offshore_wind_combined_permit_uhlisluba":       "tuulivoima_meri,hybridi",
    "paikeseenergia_solar_pv_permitting":            "aurinkovoima,hybridi",
    # Onshore-specific by content (maakonnaplaneering county plans, dwelling-
    # distance noise modeling, land-based framing) -- offshore is covered by
    # its own dedicated source above, not double-tagged here.
    "tuuleenergia_wind_energy_permitting":           "tuulivoima_maa,hybridi",
    # Broad energy-project law (bucket b), confirmed by reading actual chunk
    # text, not filenames:
    # keskkonnaseadustiku_yldosa_seadus_eia's mandatory-EIA threshold list
    # explicitly names wind/solar/hydro thresholds, "nuclear facilities of
    # any size", and "large industrial facilities >75MW thermal input"; a
    # later chunk walks through BESS's own EIA-screening path explicitly.
    # datakeskus excluded -- no explicit evidence found in any of its 10
    # chunks, unlike the other 7 types which are all directly named.
    "keskkonnaseadustiku_yldosa_seadus_eia":         "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi,teollisuus",
    "elering_grid_connection_requirements":          "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "energiamajanduse_korralduse_seadus_sector_organisation": "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    # ETS §14 Electricity Production License -- BESS included per explicit
    # user decision (2026-08-10), overriding the default EEG-style storage
    # exclusion used for Germany; unlike EEG, ETS's own licensing threshold
    # language doesn't hinge on "renewable" status the way EEG's does.
    "elektrituruseadus_electricity_market_act":      "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    # DA — PR-TAG-3 mistagging fix (2026-08-10). 11 of 14 previously-
    # untagged DA sources tagged here; 3 (bygningsreglementet,
    # ens_energilagring, mst_miljoevurdering) confirmed boilerplate/
    # navigation content with no real regulatory substance and deliberately
    # left general -- same treatment as rakentamislaki_751_2023.
    # Type-specific (bucket a):
    "ens_middelgrunden_vindmoellepark":              "tuulivoima_meri,hybridi",
    "ens_vejledning_vindmoeller_2":                  "tuulivoima_meri,hybridi",
    "ens_vejledning_vindmoeller_3":                  "tuulivoima_meri,hybridi",
    "ens_anmeldelse_skift_ejer_vindmoelle":          "tuulivoima_maa,tuulivoima_meri,hybridi",
    "mst_vandkraft_vandloebstilladelse":             "vesivoima",
    # Lillebælt Syd Vindmøllepark -- confirmed via external research
    # (user, 2026-08-10) as an officially-designated "kystnær
    # havvindmøllepark" (nearshore/coastal OFFSHORE wind farm; 11
    # turbines, 165MW, ~3km off Als, permitted by Energistyrelsen's
    # offshore-wind authority, not Miljøstyrelsen) -- offshore only,
    # not dual-tagged onshore despite the ambiguous filename/text.
    "ens_vejledning_vindmoeller_tilladelse":         "tuulivoima_meri,hybridi",
    # Broad/technology-neutral rules (bucket b), confirmed by reading
    # actual chunk text, not filenames:
    # Explicitly states the >25MW electricity-production-permit rule
    # "means a BESS... requires the SAME... permit as solar or wind
    # installations" -- not BESS-exclusive despite the source name. SMR
    # excluded -- nuclear has its own dedicated licensing pathway, this
    # generic electricity-market threshold rule doesn't apply to it.
    "ens_bess_elproduktion_over25mw":                "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,vesivoima,hybridi",
    # Also broader than the filename suggests -- general VE (renewable
    # energy) installation permit process (VE-tilladelsesprocesbekendt-
    # gørelsen), not solar-specific. BESS/SMR excluded, same
    # renewables-only-by-definition reasoning as Germany's EEG.
    "ens_sol_ve_tilladelse":                         "tuulivoima_maa,tuulivoima_meri,aurinkovoima,vesivoima,hybridi",
    # Offshore renewable-energy tender permit process -- technology-neutral
    # wording ("vedvarende energi-anlæg på havet") but Denmark's actual
    # offshore tender program has been wind-only in practice to date;
    # aurinkovoima deliberately excluded until the program expands
    # (user decision, 2026-08-10 -- don't tag ahead of actual practice).
    "ens_tilladelsesprocessen_vedvarende_energi_hav": "tuulivoima_meri,hybridi",
    # General Danish EIA framework (Miljøvurdering/VVM) -- same broad
    # energy-facility pattern as DE/EE's EIA docs.
    "sgav_vvm_miljoevurdering":                      "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    # General pressure-equipment product-safety page -- distinct from the
    # already-tagged sik_sco2_tryk_modulaert_kraftanlaeg (SMR-specific
    # sCO2 equipment).
    "sik_trykbaerende_udstyr":                       "BESS,SMR,smr_bess,teollisuus",
    # FI — PR-TAG-4 mistagging fix (2026-08-10). 14 of 18 previously-
    # untagged FI sources tagged here (the largest single batch so far,
    # 718 chunks); 4 (rakentamislaki_751_2023, pelastuslaki_379_2011,
    # energiavirasto_energiatehokkuus, lvv.fi) confirmed genuinely broad
    # national policy/law content and deliberately left general.
    # NOTE (flagged as a future follow-up, not acted on here): 3 of these
    # keys are full descriptive strings, not stems -- ingest_fi_env.py's
    # own deliberate design choice (human-readable citation labels), but
    # fragile as a SOURCE_HANKETYYPPI_TAG lookup key -- any future
    # relabeling of that source string silently breaks the tag mapping
    # with no error, just a quiet fallback to unfiltered/general. Worth
    # normalizing ingest_fi_env.py to stems like every other ingest
    # script, in a separate PR.
    # Type-specific (bucket a):
    "lion_2025_bess":                                "BESS,hybridi",
    # SJV = Fingrid's STORAGE-specific grid code (Sähkövarastojen
    # järjestelmätekniset vaatimukset) -- deliberate counterpart to
    # VJV below, which covers generation instead.
    "sjv2024_fingrid":                                "BESS,smr_bess,hybridi",
    "tukes_painelaitteet_sco2":                       "SMR,smr_bess",
    "YSL 527/2014 — Akkuvarasto (BESS) ja energiantuotanto: ympäristölupatarve": "BESS,hybridi",
    # Broad/technology-neutral or explicitly-scoped rules (bucket b),
    # confirmed by reading actual chunk text, not filenames:
    # VJV = Fingrid's POWER-PLANT (generation) grid code -- "Voimalaitosten
    # järjestelmätekniset vaatimukset". Bare BESS deliberately EXCLUDED --
    # that's what SJV (above) covers instead; the two documents are a
    # deliberate pair, confirmed by their own titles/scope.
    "vjv2024_fingrid":                                "tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    # Explicitly scoped in its own text to "ammattikäytössä oleviin
    # teollisuusakuiksi luokiteltaviin Li-akkuihin" (professional-use,
    # industrial-classified Li-ion batteries) -- distinct from
    # lion_2025_bess's grid-storage scope, same reasoning for both:
    "lion_teollisuus_2025":                           "teollisuus,datakeskus",
    "tukes_liion_opas":                                "teollisuus,datakeskus",
    "kemikaaliturvallisuuslaki_390_2005":             "BESS,teollisuus,datakeskus,hybridi",
    "tukes_painelaitteet":                            "BESS,SMR,smr_bess,teollisuus",
    "YVA-laki 252/2017 — Ympäristövaikutusten arviointi": "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi,teollisuus,datakeskus",
    # General YSL trigger-law overview (SS27 permit requirement, SS29
    # registration-only tier) -- confirmed genuinely broad by content,
    # not narrowed to a couple of types.
    "YSL 527/2014 — Ympäristölupa: luvantarve":       "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi,datakeskus,teollisuus,maatalous,ymparistolupa",
    # Purely procedural (application process/timelines/appeals) -- applies
    # to any hanketyyppi that ever needs a YSL permit, same breadth as
    # above.
    "YSL 527/2014 — Ympäristöluvan hakeminen: prosessi ja liitteet": "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi,datakeskus,teollisuus,maatalous,ymparistolupa",
    # Caruna distribution-grid (jakeluverkko) family -- matches the
    # existing fingrid_liittyminen_kantaverkkoon comment's own note that
    # BESS/aurinkovoima use distribution grid, not kantaverkko.
    "caruna_network_development_plan_2026":          "BESS,aurinkovoima,hybridi,datakeskus,teollisuus",
    "caruna_high_voltage_capacity_2025":              "BESS,aurinkovoima,hybridi,datakeskus,teollisuus",
    # Priority-2 (2026-08-10): maatalous + vesivoima previously had ZERO
    # dedicated content -- see permit_ai/ingest_maatalous_vesivoima.py.
    # No hybridi inheritance here -- hybridi is defined as BESS+wind/solar
    # only, never includes maatalous or vesivoima components.
    "ysl_527_2014_liite1_elainsuoja":           "maatalous",
    "ruokavirasto_maatalouden_investointituet": "maatalous",
    "mmm_610_2023_lypsykarjarakennukset":       "maatalous",
    "nitraattiasetus_1250_2014_valvontaohje":   "maatalous",
    "vesilaki_587_2011_kalatalousvelvoite":     "vesivoima",
    "patoturvallisuuslaki_494_2009":            "vesivoima",
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
