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
#
# "none" (2026-08-10) = excluded from EVERY hanketyyppi, including any added
# in the future -- a stronger statement than "general" (visible to all) and
# the opposite of a comma-list (visible to specific ones). Use only for
# content confirmed, by directly reading the actual chunk text (never the
# filename alone), to be irrelevant to every current use of this platform --
# e.g. content from a different regulatory domain entirely (medical/
# healthcare radiation rules swept up alongside genuine nuclear-power
# content because both happen to fall under the same umbrella law), or
# content that isn't real ("Ingested a dead 404 page as if it were a
# document"). Always pair a "none" entry with an inline comment stating
# why -- this is a stronger, easier-to-forget action than any other tag
# here (it hides content from every hanketyyppi, not just the wrong one),
# so it gets the same real-content-verification-then-explicit-user-sign-off
# discipline as every other entry in this file, never added unilaterally.
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
    # hybridi backfill (2026-08-11): already carried both BESS and wind/solar
    # tags before this remediation sequence began -- LV/LT were never part
    # of the 9-PR mistagging sequence (confirmed genuinely not mistagged),
    # so the hybridi-inheritance rule approved during PR-TAG-1..7c never
    # ran here. Mechanical backfill of the already-approved rule, not new
    # sourcing or a new judgment call -- see this PR's commit message.
    "lv_ast_sistemas_pievienošanas_noteikumi":       "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,teollisuus,SMR,hybridi",
    # Distribution grid (Sadales tīkls) — smaller energy projects, not SMR/industrial
    "lv_sadales_tikls_pievienošanas_kārtiba":        "BESS,aurinkovoima,tuulivoima_maa,hybridi",
    # SPRK licensing — energy generation/storage projects only
    "lv_sprk_elektroenerģijas_ražošanas_licence":    "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,hybridi",
    "lv_mk_631_licencesanas_noteikumi":              "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,teollisuus,hybridi",
    # Renewables law — only for renewable energy and co-located BESS
    "lv_atjaunojamas_energijas_likums":              "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,hybridi",
    # BESS market/balancing — BESS only, no wind/solar co-presence -- does NOT qualify for hybridi backfill
    "lv_bess_balansesanas_tirgus_ast":               "BESS",
    # Wind-specific regulations — wind only, no BESS -- does NOT qualify
    "lv_veja_energija_buvniecibas_noteikumi":        "tuulivoima_maa,tuulivoima_meri",
    # BVKB wind turbine regulatory overview — wind only, no BESS -- does NOT qualify
    "lv_bvkb_veja_elektrostacija":                  "tuulivoima_maa,tuulivoima_meri",
    # BESS fire safety — BESS and datakeskus, no wind/solar -- does NOT qualify
    "lv_ugunsdrošibas_bess_prasibas":               "BESS,datakeskus",
    # LT — energy-type-specific sources (Lithuania has no nuclear power plants)
    # Transmission grid connection (LITGRID) — energy projects only
    "lt_litgrid_prisijungimo_tvarka":                "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,teollisuus,hybridi",
    # Distribution grid (ESO) — smaller energy projects, not SMR/industrial
    "lt_eso_gamintojams_prisijungimas":              "BESS,aurinkovoima,tuulivoima_maa,hybridi",
    # VERT generation licensing — energy generation/storage projects only.
    # NOT YET IN PRODUCTION (checked 2026-08-11): defined in
    # backend/lithuania_ingestion.py's source list but never successfully
    # ingested -- its URL (vert.lt) is the exact domain already flagged in
    # the manual-sourcing-backlog memory as WAF-403-blocked for all paths.
    # Kept here (not deleted) because it's a real, still-intended source
    # blocked by a known, already-tracked issue, not stale/incorrect
    # tagging -- the tag is ready for whenever that block is resolved.
    "lt_vert_elektros_gamybos_licencija":            "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,hybridi",
    # VERT OZE auctions — renewables only, no BESS -- does NOT qualify
    "lt_vert_oze_aukcionai_kvota":                   "tuulivoima_maa,tuulivoima_meri,aurinkovoima",
    # Renewables law — only for renewable energy and co-located BESS.
    # NOT YET IN PRODUCTION (checked 2026-08-11): same situation as
    # lt_vert_elektros_gamybos_licencija above, but its URL is
    # e-seimas.lrs.lt -- the domain already flagged in the manual-
    # sourcing-backlog memory as JS-rendered/near-zero-text-extracted.
    # Kept here for the same reason.
    "lt_atsinaujinancio_resurso_energetikos_istatymas": "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,hybridi",
    # BESS balancing market — BESS only, no wind/solar co-presence -- does NOT qualify
    "lt_litgrid_balansavimo_rinka":                  "BESS",
    # Wind energy special plan — wind only, no BESS -- does NOT qualify
    "lt_vejo_energetikos_specialusis_planas":        "tuulivoima_maa,tuulivoima_meri",
    "lt_vert_vejo_energetika":                       "tuulivoima_maa,tuulivoima_meri",
    # BESS fire safety — BESS and datakeskus, no wind/solar -- does NOT qualify
    "lt_priesgaisrines_saugos_taisykles":            "BESS,datakeskus",
    "lt_priesgaisrines_saugos_istatymas":            "BESS,datakeskus",
    "lt_str_esminiai_gaisrine_sauga":                "BESS,datakeskus,teollisuus,asuinrakennus,liikerakennus,maatalous",
    "lt_gamybos_pramones_sandeliavimo_gaisrine_sauga": "BESS,datakeskus,teollisuus",
    # Gas systems installation rules (buildings) — building-code gas piping/appliance
    # safety, sourced via pagd.lrv.lt's Ministry of Energy acts list. Not applicable
    # to BESS/wind/solar/SMR/hydro (no gas systems involved) -- deliberately excluded
    # from those to avoid contamination; only building types that plausibly install
    # gas heating/appliances qualify.
    "lt_dujos_sistemu_pastatuose_taisykles":         "asuinrakennus,liikerakennus,teollisuus,maatalous,datakeskus",
    # EIA law — all energy projects requiring PAV (BESS, wind, solar, SMR)
    "lt_pav_istatymas":                              "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,hybridi",
    "lt_pav_kategoriju_sarasas":                     "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,hybridi",
    # SE — BESS fire safety (RISE 2023:117 + Boverket BFS 2024:7) — BESS and datakeskus
    "boverket_rise_bess_brandsakerhet":              "BESS,datakeskus",
    # DA — BESS fire safety (Beredskabsstyrelsen 2023 vejledning + DBI gap analysis 2025)
    "beredskabsstyrelsen_dbi_bess_brandsikkerhed":   "BESS,datakeskus",
    # DA — real full-text law ingestion via retsinformation.dk ELI endpoint,
    # replacing 9 previously-wrong _COUNTRY_LUVAT["DA"] citations (see
    # backend/denmark_ingestion.py). Tags mirror exactly which DA hanketyyppi
    # cite each law in _COUNTRY_LUVAT.
    "da_undergrundsloven":            "SMR,smr_bess,smr_da,egs",
    "da_miljovurderingsloven":        "SMR,BESS,tuulivoima_maa,aurinkovoima,smr_bess,vesivoima,tuulivoima_meri,teollisuus,smr_da,offshore_wind",
    "da_stralebeskyttelsesloven":     "SMR,smr_bess,smr_da",
    "da_kystbeskyttelsesloven":       "SMR,smr_bess,tuulivoima_meri,offshore_wind",
    "da_byggeloven":                  "SMR,BESS,tuulivoima_maa,aurinkovoima,smr_bess,vesivoima,datakeskus,teollisuus,asuinrakennus,maatalous,liikerakennus,smr_da,egs",
    "da_planloven":                   "SMR,BESS,tuulivoima_maa,aurinkovoima,smr_bess,datakeskus,teollisuus,asuinrakennus,maatalous,liikerakennus,smr_da,egs",
    "da_miljobeskyttelsesloven":      "BESS,tuulivoima_maa,vesivoima,tuulivoima_meri,datakeskus,teollisuus,offshore_wind,egs",
    "da_elforsyningsloven":           "BESS,tuulivoima_maa,aurinkovoima,smr_bess,vesivoima,tuulivoima_meri,datakeskus,offshore_wind",
    "da_vandforsyningsloven":         "vesivoima",
    "da_ve_loven":                    "tuulivoima_maa,tuulivoima_meri,offshore_wind",
    "da_husdyrbrugloven":             "maatalous",
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
    # NO — PR-TAG-5 mistagging fix (2026-08-10). 20 of 25 previously-
    # untagged NO sources tagged here (1,028 chunks, the largest per-
    # country total in the whole backlog); 5 confirmed thin/general
    # content deliberately left general; dsa_nve_smr_nuclear_regulatory
    # already carried an explicit tag and was out of scope.
    # Type-specific (bucket a):
    "regjeringen_planlegging_konsesjonsbehandling_vindkraft": "tuulivoima_maa,hybridi",
    "nve_konsesjonsveileder_vindkraft_land":         "tuulivoima_maa,hybridi",
    "nve_kommunal_mindre_vindkraftanlegg":           "tuulivoima_maa,hybridi",
    "nve_mta_vindkraftanlegg_2016":                  "tuulivoima_maa,hybridi",
    "nve_skyggekast_vindkraftverk_2014":             "tuulivoima_maa,hybridi",
    "nve_iskast_vindturbiner_2018":                  "tuulivoima_maa,tuulivoima_meri,hybridi",
    "nve_konsesjon_solkraft":                        "aurinkovoima,hybridi",
    "nve_konsesjonssoknad_solkraft_guide":           "aurinkovoima,hybridi",
    "nve_soknadsveileder_batterianlegg":             "BESS,hybridi",
    "dibk_datasenter_byggetillatelse":               "datakeskus",
    "dsb_sco2_trykkpaavirket_modulaert":             "SMR,smr_bess",
    # Broad/technology-neutral rules and confirmed judgment calls
    # (bucket b), confirmed by reading actual chunk text, not filenames:
    # Statnett's national grid functional-requirements code -- a single
    # unified document covering the whole power system, no internal
    # generation/storage split like Finland's VJV/SJV pair (confirmed by
    # user via external research, 2026-08-10) -- the largest single
    # source in the entire mistagging backlog across all countries.
    "statnett_nvf_2025":                             "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    # Onshore-specific -- confirmed via external research (user,
    # 2026-08-10): NVE maintains a SEPARATE offshore landscape-impact
    # framework ("Strategisk konsekvensutredning av vindkraft til havs",
    # veiledere.nve.no/havvind, its own "Landskap og verneområder" page).
    # This 2015 guide uses the Lista ONSHORE wind farm as its worked
    # example and is a distinct document from the offshore framework --
    # not dual-tagged.
    "nve_landskapsvirkninger_vindkraft_2015":        "tuulivoima_maa,hybridi",
    "nve_ik_energi_2018":                            "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "nve_personellsikkerhet_kraftforsyning_2024":    "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "nve_terrengbehandling_2021":                    "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "statsforvalteren_konsekvensutredning_energi":   "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "dsb_trykkpaavirket_utstyr":                     "BESS,SMR,smr_bess,teollisuus",
    # Explicitly enumerated in its own text -- "NVE behandler...konsesjon
    # til å bygge vindkraftverk, vannkraftverk og nettanlegg" -- kept
    # deliberately narrower than the broad-energy sets above since the
    # source names only these three.
    "nve_konsesjonsprosessen":                       "tuulivoima_maa,tuulivoima_meri,vesivoima,hybridi",
    # Thin homepage content, but genuinely DSA (Norway's radiation/
    # nuclear-safety authority)'s own page -- unlike the DA/NO thin-
    # content examples left general below, this one is topically
    # correct despite its thinness.
    "dsa_nuclear_safety":                            "SMR",
    # SE — PR-TAG-6 mistagging fix (2026-08-10). 11 of 20 previously-
    # untagged SE sources tagged here (1,350 chunks); 9 confirmed thin,
    # dead, or genuinely cross-sectoral content deliberately left general.
    # boverket_rise_bess_brandsakerhet already carried an explicit tag and
    # was out of scope.
    # Type-specific (bucket a):
    "energimyndigheten_havsbaserad_vindkraft_potential": "tuulivoima_meri,hybridi",
    # Explicitly "på land och till havs" (onshore AND offshore) in its
    # own title -- the source Jere named explicitly.
    "energimyndigheten_vagledning_nedmontering_vindkraft": "tuulivoima_maa,tuulivoima_meri,hybridi",
    "energimyndigheten_tillstand_vindkraft_land":    "tuulivoima_maa,hybridi",
    "av_sco2_tryckkarlstillstyrning":                "SMR,smr_bess",
    "energimyndigheten_datacenter_rapportering":     "datakeskus",
    "havochvatten_vattenkraft_tillstand":            "vesivoima",
    # Broad/technology-neutral rules and confirmed judgment calls
    # (bucket b), confirmed by reading actual chunk text, not filenames.
    # regeringen_nationell_energi_klimatplan_2021_2030 (531 chunks, the
    # largest single SE source) deliberately NOT tagged here -- Sweden's
    # official EU NECP (Regulation 2018/1999), genuinely cross-sectoral by
    # design (transport, buildings, industry, agriculture, energy).
    # Narrowing it to specific hanketyyppi would misrepresent its actual
    # scope -- confirmed with user, left general.
    "energimyndigheten_scenarier_energisystem_2023": "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "energimyndigheten_nationell_strategi_vindkraft_2021": "tuulivoima_maa,tuulivoima_meri,hybridi",
    "naturvardsverket_mkb_miljokonsekvensbeskrivning": "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    "svk_anslutning_inmatning":                      "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,SMR,smr_bess,vesivoima,hybridi",
    # PL — PR-TAG-7a mistagging fix (2026-08-10), first of 3 Poland sub-
    # batches (7a nuclear, 7b BESS/energy, 7c other -- PL is too large,
    # 76 sources, for one PR). 35 of the ~51-source PL nuclear cluster
    # tagged here as genuinely SMR-relevant, verified by reading actual
    # chunk text, not filenames. 16 sources deliberately NOT tagged and
    # left general -- see the precision distinction below, this PR's most
    # important judgment call.
    #
    # MEDICAL EXCLUSION (13 sources): all Minister-of-Health-issued,
    # healthcare-unit/medical-exposure-specific regulations -- X-ray
    # diagnostics, radiotherapy, nuclear medicine, clinical audits,
    # diagnostic reference levels, healthcare radiological-equipment
    # databases. Confirmed genuinely medical by reading the actual text
    # (issuing authority, healthcare-unit-specific scope language), not
    # nuclear-power-plant content. Tagging these SMR would have been a
    # real retrieval-precision error -- an SMR developer has no use for
    # hospital X-ray/radiotherapy equipment regulations. Left untagged/
    # general (there is no "healthcare facility" hanketyyppi in this
    # platform to tag them to instead): Regulation_reference_procedures_
    # nuclear_medicine_EXTRACT_DzUrzMZ201482, Regulation_requirements_
    # heathcare_units_radiotherapy_radiopharmaceutical_products_DzU20211890,
    # Regulation_minimum_requirements_healthcare_units_X-ray_diagnostics_
    # DzU20211725, B09Regulation_detailed_conditions_safe_use_
    # radiological_equipment_DzU2006_180_1325, Regulation_detailed_scope_
    # internal_external_clinical_audits_DzU20222683, Regulation_scope_
    # info_contained_Central_Database_Medical_Exposures__DzU20201051,
    # Regulation_diagnostic_reference_levels_DzU20222626, Regulation_
    # operational_tests_radiological_equipment_auxiliary_devices_pp1-11_
    # 56-67_DzU20222759, B10Regulation_supervision_control_compliance_
    # conditions_radiation_protection_organisational_units_using_X-ray_
    # equipment_DzU2007111, Regulation_categories_eligibility_criteria_
    # unintended_accidental_exposures_DzU20222700, Regulation_order_
    # perform_non-medical_exposures_employment_insurance_DzU20201568,
    # Regulation_radiation_protection_officer_authoriz_internal_
    # supervision_health__DzU20211908, Regulation_info_National_Database_
    # Radiological_Equipment_DzU20211959.
    #
    # GENERAL/NOT NUCLEAR-POWER-SPECIFIC (3 sources): Regulation_
    # building_materials_which_require_determining_activity_concern_
    # K-40_Ra-226_Th-232_DzU202133 (general construction-material
    # radioactivity, any building type), Regulation_determination_of_
    # entities_competent_to_inspect_maximum_permitted_levels_of_
    # radioactive_contamination_food_and_feed_DzU200498988 (public-health
    # food-safety emergency response, not facility-specific). B04Ordinance_
    # MoND_exercising_the_provisions_Atomic_Law_DzUrzMON200315161 --
    # RECLASSIFIED after direct content verification: confirmed this is a
    # Ministry of National Defence ordinance on radiation-exposure dose
    # limits for SOLDIERS/military personnel, not nuclear facility
    # regulation -- real content, wrong initial guess, caught before
    # tagging, not after.
    "NATIONAL_REPORT_OF_POLAND_ON_COMPLIANCE_WITH_THE_OBLIGATIONS_OF_THE_CONVENTION_ON_NUCLEAR_SAFETY": "SMR,smr_bess",
    "7th_national_report_to_the_Joint_Convention_(2020)": "SMR,smr_bess",
    "NATIONAL_REPORT_OF_REPUBLIC_OF_POLAND_ON_COMPLIANCE_WITH_OBLIGATIONS_OF_THE_JOINT_CONVENTION_ON_THE_SAFETY": "SMR,smr_bess",
    "7th_National_Report_of_Poland_for_CNS":         "SMR,smr_bess",
    "5th_national_report_to_the_Joint_Convention_2014": "SMR,smr_bess",
    "6th_national_report_to_the_Joint_Convention_2018": "SMR,smr_bess",
    "6th_national_report_PL":                        "SMR,smr_bess",
    "raport_NS_2010":                                "SMR,smr_bess",
    "Regulation_scope_radiat_monitoring_environ_org_entities_cat_I_II_hazards_Appen_1_and_3_DzU20222058": "SMR,smr_bess",
    "Regulation_documents_application_issuance_license__activity__exposure_ion_radiation_DzU20211667": "SMR,smr_bess",
    "National_Assessment_Report_Ageing_Management_Poland": "SMR,smr_bess",
    "raport_NS_EOM_2012":                             "SMR,smr_bess",
    "Regulation_detailed_conditions_safe_work_with_ionising_radiation_sources_DzU2022967": "SMR,smr_bess",
    "Regulation_requirements_commissioning_operation_nuclear_facilities_DzU2013281": "SMR,smr_bess",
    "Questions_and_answers_to_the_7th_National_Report_JC": "SMR,smr_bess",
    "Regulation_radiation_protection_officers_DzU2021640": "SMR,smr_bess",
    "Regulation_requirements_individual_dose_registration_DzU20211053": "SMR,smr_bess",
    "Regulation_nuclear_safety_radiological_protection_requirements_decommissioning_DzU2013270": "SMR,smr_bess",
    "Regulation_nuclear_regulatory_inspectors_DzU20211577": "SMR,smr_bess",
    "B02Regulation_stations_for_early_detection_of_radioactive_contamination_and_units_that_conduct_measurements_DzU20022392030": "SMR,smr_bess",
    "Regulation_physical_protection_nuclear_material_nuclear_facilities_DzU20082071295": "SMR,smr_bess",
    "Notice_consolidated_text_regulation_basic_requirements_supervised_controlled_areas_DzU_2022_poz_722": "SMR,smr_bess",
    "Regulation_intervention_levels_for_various_intervention_measures_and_criteria_for_cancelling_intervention_measures_DzU200498987": "SMR,smr_bess",
    "Regulation_indicators_determination_ionizing_radiation_doses_assessing_exposure_ion_radiat__pages1-5_DzU20211657": "SMR,smr_bess",
    "Regulation_allocated_special-purpose_subsidy_state-owned_public_utility_company_DzU20201624": "SMR,smr_bess",
    "Regulation_granting_license_permit_import_export_transit_radioactive_waste_spent_nuclear_fuel_DzU20082191402": "SMR,smr_bess",
    "B03Regulation_requirements_for_dosimetric_equipment_DzU20022392032": "SMR,smr_bess",
    "Regulation_prior_information_to_general_public_in_event_of_radiation_emergency_DzU20041021065": "SMR,smr_bess",
    "Regulation_periodical_safety_assessment_of_nuclear_facility_DzU2012556": "SMR,smr_bess",
    "Regulation_periodic_safety_review_radioactive_waste_repository_DzU201628": "SMR,smr_bess",
    "Regulation_psychiatric_psychological_tests_employees_perfm_act_import_nucl_safety_rad_prot_DzU20112201310": "SMR,smr_bess",
    "Regulation_types_protective_actions_external_zones_DzU20202247": "SMR,smr_bess",
    "Regulationa_method_carrying_out_supervision_inspection_ISA_FIA_CAB_nuclear_regulatory_authority_DzU2010855": "SMR,smr_bess",
    "Regulation_template_quarterly_report_decomissioning_fund__DzU2021393": "SMR,smr_bess",
    "Regulation_amount_contribution_cover_costs_final_management_spent_fuel_radioactive_waste_DzU20121213": "SMR,smr_bess",
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
    # "none" exclusions (2026-08-10) -- see the docstring above this dict.
    # 13 PL sources from PR-TAG-7a: confirmed by direct chunk-text reading
    # to be Minister-of-Health-issued, healthcare-unit/medical-exposure-
    # specific regulations (X-ray diagnostics, radiotherapy, nuclear
    # medicine, clinical audits) -- swept into the corpus alongside genuine
    # nuclear-power content because both fall under Poland's Atomic Law
    # umbrella, but not power-plant content. Empirically confirmed to leak
    # into real SMR-query top-10 results at ranks as high as #3 when left
    # as unfiltered/general (2026-08-10 retrieval test) -- "none" is the
    # correct fix, not just "not tagged SMR".
    "Regulation_reference_procedures_nuclear_medicine_EXTRACT_DzUrzMZ201482": "none",  # confirmed medical, not nuclear-power
    "Regulation_requirements_heathcare_units_radiotherapy_radiopharmaceutical_products_DzU20211890": "none",  # confirmed medical, not nuclear-power
    "Regulation_minimum_requirements_healthcare_units_X-ray_diagnostics_DzU20211725": "none",  # confirmed medical, not nuclear-power
    "B09Regulation_detailed_conditions_safe_use_radiological_equipment_DzU2006_180_1325": "none",  # confirmed medical, not nuclear-power
    "Regulation_detailed_scope_internal_external_clinical_audits_DzU20222683": "none",  # confirmed medical, not nuclear-power
    "Regulation_scope_info_contained_Central_Database_Medical_Exposures__DzU20201051": "none",  # confirmed medical, not nuclear-power
    "Regulation_diagnostic_reference_levels_DzU20222626": "none",  # confirmed medical, not nuclear-power
    "Regulation_operational_tests_radiological_equipment_auxiliary_devices_pp1-11_56-67_DzU20222759": "none",  # confirmed medical, not nuclear-power
    "B10Regulation_supervision_control_compliance_conditions_radiation_protection_organisational_units_using_X-ray_equipment_DzU2007111": "none",  # confirmed medical, not nuclear-power
    "Regulation_categories_eligibility_criteria_unintended_accidental_exposures_DzU20222700": "none",  # confirmed medical, not nuclear-power
    "Regulation_order_perform_non-medical_exposures_employment_insurance_DzU20201568": "none",  # confirmed medical, not nuclear-power
    "Regulation_radiation_protection_officer_authoriz_internal_supervision_health__DzU20211908": "none",  # confirmed medical, not nuclear-power
    "Regulation_info_National_Database_Radiological_Equipment_DzU20211959": "none",  # confirmed medical, not nuclear-power
    # SE — confirmed dead 404 link ingested as if it were a real document
    # (PR-TAG-6 finding): the actual chunk content is the literal Swedish
    # "Sidan kan inte hittas" (page not found) text, zero informational
    # value for any hanketyyppi.
    "ssm_karnkraft_tillstand_reglering": "none",  # confirmed dead 404 link
    # PL — PR-TAG-7b mistagging fix (2026-08-10), second of 3 Poland
    # sub-batches (7a nuclear, 7b BESS/energy, 7c other). Genuine BESS
    # content, confirmed by reading actual chunk text and URLs:
    "poland_building_law_amendment_2026_bess": "BESS,hybridi",
    "poland_gramwzielone_bess_2026":           "BESS,hybridi",  # exact-content duplicate of the above -- logged in manual-sourcing-backlog memory, not deduplicated here
    "poland_bess_permitting_guide_2025":       "BESS,hybridi",
    "poland_dudkowiak_bess_legal_2025":        "BESS,hybridi",  # exact-content duplicate of the above -- logged in manual-sourcing-backlog memory, not deduplicated here
    "ure_bess_energy_storage_licensing":       "BESS,hybridi",
    # ure_aktualnosci_magazyny_energii deliberately NOT tagged -- confirmed
    # genuine but mixed/generic URE energy-sector news, not BESS-specific
    # enough to narrow-tag despite the name. Stays general.
    #
    # "none" -- DIFFERENT reason than the medical/dead-link exclusions
    # above: these 3 have correct, BESS-relevant URLs (confirmed real
    # pages about the energy-storage register / URE storage guidelines)
    # but the actual scraped chunk content is the WRONG page entirely
    # (wholesale natural-gas-market monitoring data, and an unrelated
    # economic-summit patronage-events listing) -- a scraper/redirect
    # glitch at ingestion time, not a domain mismatch. The real content
    # these sources were meant to capture still doesn't exist in the
    # corpus at all -- tracked as a re-ingestion task in the manual-
    # sourcing-backlog memory, not just a tagging fix.
    "poland_ure_bess_register_guide_2025":     "none",  # confirmed wrong-page scraper content (gas-market data, not the storage register guide)
    "poland_ure_bess_register_przewodnik":     "none",  # confirmed wrong-page scraper content (same gas-market data as above)
    "poland_ure_bess_wytyczne":                "none",  # confirmed wrong-page scraper content (economic-summit event listing, not URE storage guidelines)
    # PL — PR-TAG-7c mistagging fix (2026-08-10), FINAL batch of 3 Poland
    # sub-batches (7a nuclear, 7b BESS/energy, 7c other: offshore/onshore
    # wind, hydro, EIA, grid connection, general energy law) -- this
    # closes the entire 9-PR Priority 1 mistagging sequence.
    "poland_offshorewind_regulacje":           "tuulivoima_meri",
    # Filename says "offshorewind" but the actual article content is
    # entirely about ONSHORE wind farm modernization ("modernizację
    # lądowych farm wiatrowych") -- tagged by real content, not the
    # misleading filename.
    "poland_offshorewind_repowering_2025":     "tuulivoima_maa",
    "poland_repowering_exemption_dus_2025":    "tuulivoima_maa",  # exact-content duplicate of the source above -- logged in manual-sourcing-backlog memory, not deduplicated here
    "poland_ustawa_10h_2016":                  "tuulivoima_maa",  # the "10H" turbine-to-building minimum-distance law, onshore-only throughout
    "udt_ure_datacenter_pozwolenia":           "datakeskus",
    "wody_polskie_hydropower_pozwolenie":      "vesivoima",
    # sCO2 pressure-vessel content -- same convention as every other
    # country's sCO2 source (DA sik_sco2_tryk_modulaert_kraftanlaeg, NO
    # dsb_sco2_trykkpaavirket_modulaert, SE av_sco2_tryckkarlstillstyrning,
    # FI tukes_painelaitteet_sco2 -- all SMR,smr_bess).
    "udt_sco2_urzadzenia_cisnieniowe":         "SMR,smr_bess",
    # Left general (confirmed genuinely broad/thin by reading real chunk
    # text, not assumed): gdos_oos_environmental_assessment (curated EIA
    # content, explicitly cross-technology), poland_prawo_energetyczne_1997
    # (foundational energy law, own text names BESS/OZE/SMR alike),
    # poland_ustawa_oos_2008 (master EIA procedural law, explicitly covers
    # nuclear facility investments alongside everything else),
    # pse_grid_connection (thin PSE homepage -- cookie banner, live
    # dashboard, news ticker -- but touches wind/PV curtailment
    # compensation generically), udt_dozor_techniczny (thin UDT homepage
    # nav, on-topic but not substantive enough to narrow-tag),
    # wody_polskie_pozwolenia_wodnoprawne (generic "types of water permit"
    # portal nav, not hydropower-specific despite sharing a URL with the
    # curated wody_polskie_hydropower_pozwolenie above -- see manual-
    # sourcing-backlog memory for that same-URL/different-content anomaly).
    #
    # "none" -- same wrong-content failure mode as PR-TAG-7b, two variants:
    "gdos_informacje_dla_przedsiebiorcow":     "none",  # URL promises business/investor info; actual content is a homepage news ticker (kids' events, peatland workshops), unrelated
    "gdos_oceny_oddzialywania_srodowisko":     "none",  # URL promises environmental impact assessment info; actual content is the IDENTICAL news-ticker junk as the source above
    # NEW variant of the wrong-content bug, worse than 7b's: this is a
    # wrong PDF fetched at the source, not just a wrong scraped webpage.
    # URL/description say this is the EIA-thresholds regulation
    # ("przedsięwzięcia mogące znacząco oddziaływać na środowisko"), but
    # the downloaded ISAP PDF is a Ministry of Finance excise-tax (akcyza)
    # declaration-form regulation entirely. Flagged in the manual-
    # sourcing-backlog memory: the ISAP download URL itself needs manual
    # re-verification before any future re-fetch attempt -- simply
    # re-running the same URL would reproduce the same wrong PDF.
    "poland_rozp_przedsiewziecia_2019":        "none",  # confirmed wrong PDF entirely (excise-tax form, not the EIA-thresholds regulation)
    # Hybrid co-location sourcing (2026-08-12): real, primary-source content
    # specifically about combining BESS with wind/solar at ONE shared grid
    # connection point -- genuinely hybridi-specific, not single-technology
    # background reading. Ingested via permit_ai/ingest_hybridi_colocation.py.
    # See that module's docstring for full retrieval/verification detail.
    "fi_fingrid_hybridivoimalaitos_ohje_2023":       "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,vesivoima,hybridi",
    "de_eeg_8a_flexible_netzanschluss":              "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,hybridi",
    "de_baugb_35_batteriespeicher_privilegierung":   "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,hybridi",
    "da_energinet_samplacerede_overplantede_krav":   "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,hybridi",
    "lv_mk_821_hibridatlauja":                       "BESS,tuulivoima_maa,tuulivoima_meri,aurinkovoima,hybridi",
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
    4. tag == "none" — excluded from every hanketyyppi, including future
       ones (see SOURCE_HANKETYYPPI_TAG's docstring for when this applies)
    5. Default True — unknown sources are not filtered out
    """
    tag = (
        chunk_meta.get("hanketyyppi_tag")
        or chunk_meta.get("project_types")
        or get_hanketyyppi_tag(chunk_meta.get("source", ""))
    )
    if tag == "none":
        return False
    if not tag or tag in ("general", "all"):
        return True
    allowed = {t.strip() for t in tag.split(",")}
    return current_hanketyyppi in allowed
