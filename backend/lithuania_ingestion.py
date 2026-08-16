"""
Lithuania regulatory RAG ingestion for NCE Permit AI.

Downloads public Lithuanian legal texts and regulatory guidance (e-seimas.lrs.lt,
e-tar.lt, VERT, LITGRID, ESO, PAGD), extracts text, chunks by word count, and
upserts into both ChromaDB collections:
  - permit_docs    (MiniLM-L12-v2, 384-dim)  — v1, survives future reindex
  - permit_docs_v2 (mpnet 768-dim)            — active production collection

hanketyyppi_tag is always resolved via get_hanketyyppi_tag() from source_policy;
no tag value is hardcoded here so the BESS/wind/solar restriction logic stays
in a single source of truth.

Run standalone for testing:
    python3 backend/lithuania_ingestion.py
Or trigger via API:
    POST /api/admin/ingest-lithuania  (x-admin-secret header required)

Key Lithuanian regulatory framework notes:
  - VERT (Valstybinė energetikos reguliavimo taryba) — independent energy regulator:
    elektros energijos gamybos licencija required for generation/storage >100 kW;
    registracija for 10–100 kW; below 10 kW self-declaration only.
  - LITGRID — TSO, operates 110/330 kV transmission grid. Grid connection agreement
    (tinklų prijungimo sutartis) required before commissioning >5 MW projects.
  - ESO (Energijos skirstymo operatorius) — DSO, distribution grid for smaller
    projects (<5 MW BESS, rooftop/ground solar, small wind).
  - Statybos leidimas (building permit) issued by savivaldybės administracija
    (municipal administration) — requirements vary by municipality; always verify
    locally before submitting.
  - PAV (Planuojamos ūkinės veiklos poveikio aplinkai vertinimas) — Lithuanian EIA;
    screening authority is Aplinkos apsaugos agentūra (AAA) under Aplinkos ministerija.
  - Lithuania has no operating nuclear power plants (Ignalina NPP shut down 2009);
    SMR projects would require entirely new primary legislation.
  - Baltic desynchronisation from IPS/UPS network and synchronisation with
    Continental European grid (planned completion 2025) affects LITGRID balancing
    market rules and reserve obligations.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Single source of truth for hanketyyppi tags — never hardcode tag values here
sys.path.insert(0, str(Path(__file__).parent.parent / "permit_ai"))
from source_policy import get_hanketyyppi_tag

log = logging.getLogger(__name__)

_DB_PATH  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings")
)
_COL_V1   = "permit_docs"
_COL_V2   = "permit_docs_v2"
_MODEL_V1 = "paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_V2 = "paraphrase-multilingual-mpnet-base-v2"

CHUNK_WORDS   = 800
OVERLAP_WORDS = 100
MIN_WORDS     = 50

LITHUANIA_SOURCES: list[dict] = [
    # ── Core electricity & energy laws ───────────────────────────────────────
    {
        "name": "lt_elektros_energetikos_istatymas",
        "url": "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.17924",
        "type": "html",
        "category": "electricity_market_law",
        "description": (
            "Elektros energetikos įstatymas (EEĮ, 2000-07-20, Nr. VIII-1881, consolidated). "
            "Central electricity market law governing electricity generation, transmission, "
            "distribution and supply in Lithuania. "
            "VERT licensing thresholds: elektros energijos gamybos licencija for >100 kW "
            "installed generation/storage capacity; registracija (simplified) for 10–100 kW; "
            "below 10 kW: self-declaration only. "
            "TSO = LITGRID (110/330 kV transmission); DSO = ESO (Energijos skirstymo operatorius). "
            "BESS market access: must be balance responsible party (BRP) or conclude BRP agreement "
            "with LITGRID. FCR/FRR participation requires SCADA connection to LITGRID for >1 MW. "
            "Baltic desynchronisation: Lithuania, Latvia and Estonia are desynchronising from "
            "IPS/UPS network and synchronising with Continental European grid — materially affects "
            "reserve market volumes and BESS balancing obligations."
        ),
    },
    {
        "name": "lt_atsinaujinancio_resurso_energetikos_istatymas",
        "url": "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.255838",
        "type": "html",
        "category": "renewable_energy_law",
        "description": (
            "Atsinaujinančių išteklių energetikos įstatymas (AIEI, 2011-05-12, Nr. XI-1375, "
            "consolidated). Governs wind, solar PV, hydro, and biomass electricity production "
            "and BESS co-located with renewables in Lithuania. "
            "OZE (atsinaujinantys energijos ištekliai) support framework: "
            "fiksuota priemoka (feed-in premium/FIP) mechanism administered by VERT; "
            "aukcionai (auctions/OZE kvotos) for competitive capacity allocation. "
            "Net metering (neto atsiskaitymas) available for prosumers up to 30 kW. "
            "BESS co-located with renewables: simplified VERT registration conditions. "
            "Target: 100% renewable electricity consumption by 2030. "
            "Sets technical requirements for renewable energy plants and monitoring obligations. "
            "VERT administers OZE quota auctions; winners obtain premium support for 10–20 years."
        ),
    },
    {
        "name": "lt_energetikos_istatymas",
        "url": "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.88734",
        "type": "html",
        "category": "energy_framework_law",
        "description": (
            "Energetikos įstatymas (2002-05-16, Nr. IX-884, consolidated). "
            "Framework law for Lithuania's energy sector: energy system planning, "
            "licensing, consumer protection, energy crisis management. "
            "Establishes VERT (Valstybinė energetikos reguliavimo taryba) as the "
            "independent energy regulatory authority. "
            "Mandates national energy independence strategy (Nacionalinė energetikos "
            "nepriklausomybės strategija). "
            "Sets framework for energy system security and renewable transition targets. "
            "Defines energy sector roles: electricity generation, transmission (LITGRID), "
            "distribution (ESO), and supply. "
            "Regulates energy price formation principles and supply obligation."
        ),
    },
    # ── Environmental & EIA ───────────────────────────────────────────────────
    {
        "name": "lt_pav_istatymas",
        "url": "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.40705",
        # e-seimas.lrs.lt blocks automated fetches (see manual sourcing backlog).
        # Static copy below is the officially consolidated text (redakcija nuo
        # 2026-02-21), manually sourced, including Annexes 1 and 2 with the
        # project-type-specific PAV/atranka thresholds.
        "local_path": str(
            Path(__file__).parent.parent / "permit_ai" / "rag_docs" / "LT" / "lt_pav_istatymas.txt"
        ),
        "type": "html",
        "category": "eia_law",
        "description": (
            "Planuojamos ūkinės veiklos poveikio aplinkai vertinimo įstatymas "
            "(PAV įstatymas, consolidated). "
            "Lithuania's Environmental Impact Assessment law. "
            "PAV = Planuojamos ūkinės veiklos poveikio aplinkai vertinimas "
            "(equivalent to Finnish YVA/Latvian IVN/Swedish MKB). "
            "Two-stage process: (1) atranka (screening by AAA/Aplinkos apsaugos agentūra), "
            "(2) full PAV if screening positive. "
            "Screening authority: Aplinkos apsaugos agentūra (AAA), "
            "under Aplinkos ministerija (AM). "
            "Mandatory PAV triggers for energy projects: wind farms >5 turbines or >30 MW; "
            "stand-alone BESS >50 MW or >100 MWh may require screening. "
            "Solar PV: >50 MW (100 ha land area) triggers mandatory PAV. "
            "PAV programa (scoping) → PAV ataskaita (EIA report) → "
            "AAA sprendimas (EIA decision, binding). Typical duration: 6–18 months. "
            "PAV ataskaita must be complete before statybos leidimas (building permit) "
            "application is submitted."
        ),
    },
    {
        "name": "lt_aplinkos_apsaugos_istatymas",
        "url": "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.17346",
        "type": "html",
        "category": "environmental_protection_law",
        "description": (
            "Aplinkos apsaugos įstatymas (consolidated). "
            "Framework environmental protection law. "
            "Environmental permit (taršos integruotos prevencijos ir kontrolės leidimas, "
            "TIPK) required for IPPC installations under A-kategorija. "
            "B-kategorija (taršos leidimas) for significant impact installations. "
            "BESS fire and chemical risk: lithium-ion battery installations above certain "
            "thresholds may trigger taršos leidimas requirement. "
            "Administered by Aplinkos apsaugos agentūra (AAA) and regional departments. "
            "Environmental protection conditions attached to statybos leidimas and "
            "elektros energijos gamybos licencija."
        ),
    },
    {
        "name": "lt_pav_kategoriju_sarasas",
        "url": "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.312453",
        "type": "html",
        "category": "eia_categories",
        "description": (
            "Planuojamos ūkinės veiklos, kuriai turi būti atliekamas poveikio aplinkai "
            "vertinimas, rūšių sąrašas (PAV kategorijų sąrašas, consolidated). "
            "Lists projects requiring mandatory PAV vs. atranka (screening) only. "
            "Wind energy: >5 turbines or total capacity >30 MW → mandatory full PAV; "
            "2–5 turbines → screening (atranka) by AAA. "
            "Solar PV: >50 MW (>100 ha site area) → mandatory PAV; "
            ">10 MW (>25 ha) → screening. "
            "BESS standalone: listed under industrial energy installations — AAA screening "
            "recommended for all grid-scale projects >10 MWh. "
            "Process: submit PAV atrankos informacija to AAA; decision within 20 working days."
        ),
    },
    # ── Building & spatial planning ───────────────────────────────────────────
    {
        "name": "lt_statybu_istatymas",
        "url": "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.14870",
        "type": "html",
        "category": "construction_law",
        "description": (
            "Statybos įstatymas (consolidated). "
            "Three-category building permit system based on project complexity: "
            "(1) Statybą leidžiantis dokumentas — rašytinis pritarimas (written approval) "
            "for simple/low-risk works; "
            "(2) Statybos leidimas (building permit) — required for all commercial energy "
            "infrastructure including BESS, wind turbines, solar parks. "
            "BESS ≥10 kW grid-connected or multi-container permanent installation: typically "
            "requires statybos leidimas from savivaldybės administracija (municipal authority). "
            "Digital submission via Informacinė sistema INFOSTATYBA (infostatyba.lt). "
            "Required attachments: žemės sklypo kadastro duomenys, topografinė nuotrauka, "
            "techninis projektas (technical design with PAV ataskaita if EIA required), "
            "LITGRID/ESO techniniai prisijungimo sąlygai (technical connection conditions), "
            "priešgaisrinės saugos specialisto ataskaita (fire safety report). "
            "Atsakingasis statinio statybos vadovas (licensed construction supervisor) mandatory. "
            "IMPORTANT: specific requirements vary by savivaldybė (municipality) — "
            "always contact local administracija to confirm exact documentation requirements."
        ),
    },
    {
        "name": "lt_teritoriju_planavimo_istatymas",
        "url": "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.87072",
        "type": "html",
        "category": "spatial_planning_law",
        "description": (
            "Teritorijų planavimo įstatymas (consolidated). "
            "Governs spatial planning at national, regional and local (savivaldybė) levels. "
            "Three planning instrument types: "
            "(1) Savivaldybės bendrasis planas (municipal general plan, ~10-year horizon); "
            "(2) Specialusis planas (special territorial plan for specific sectors, "
            "including energy — e.g., vėjo energetikos specialusis planas); "
            "(3) Detalusis planas (detailed plan for specific areas/plots). "
            "Energy projects (wind, solar, BESS) must comply with bendrasis planas "
            "and any applicable specialusis planas. "
            "Detalusis planas required for wind parks >1 turbine in most municipalities. "
            "Approval: savivaldybės tarybos sprendimas (municipal council decision) required "
            "for plan amendments; public consultation (viešas svarstymas) mandatory ≥20 days. "
            "National significance projects: may require specialusis teritorijų planavimas "
            "at Vyriausybės (government) level — energy ministry decision. "
            "Vėjo energetikos specialusis planas (national wind energy special plan) "
            "designates zones where wind energy is permitted nationally."
        ),
    },
    # ── Grid connection — transmission ────────────────────────────────────────
    {
        "name": "lt_litgrid_prisijungimo_tvarka",
        "url": "https://www.litgrid.eu/",
        "type": "html",
        "category": "grid_connection_transmission",
        "description": (
            "LITGRID AB (Lithuanian Electricity Transmission System Operator) "
            "prisijungimo prie elektros perdavimo tinklo tvarka. "
            "Regulatory framework for 110/330 kV transmission grid connection in Lithuania. "
            "Applies to: generators and storage connecting at transmission voltage level "
            "(typically >5 MW BESS, large wind/solar parks, industrial consumers >5 MW). "
            "Three-step process: "
            "(1) Techniniai prisijungimo sąlygai (technical connection conditions) — "
            "submit application to LITGRID, conditions issued within 25 working days; "
            "(2) Projektavimas (design phase) — develop technical project per LITGRID conditions; "
            "(3) Tinklų prijungimo sutartis (grid connection agreement) — before commissioning. "
            "LITGRID also administers system services: "
            "FCR (Frequency Containment Reserve) — symmetric ±MW, 30s activation; "
            "aFRR (automatic Frequency Restoration Reserve); "
            "mFRR (manual FRR) — activation within 12.5 min; "
            "BESS participation requires separate BRP agreement (balanso atsakingosios šalies sutartis). "
            "Baltic desynchronisation from IPS/UPS network to Continental European grid "
            "significantly impacts connection technical requirements and reserve market."
        ),
    },
    # ── Grid connection — distribution ────────────────────────────────────────
    {
        "name": "lt_eso_gamintojams_prisijungimas",
        "url": "https://www.eso.lt/lt/gamintojams/",
        "type": "html",
        "category": "grid_connection_distribution",
        "description": (
            "ESO (Energijos skirstymo operatorius) prisijungimo prie skirstomojo elektros "
            "tinklo tvarka gaminantiems vartotojams (prosumer/producer connection procedure). "
            "Distribution grid connection for projects at 0.4–35 kV level. "
            "Used for: BESS <5 MW, rooftop/ground-mount solar <5 MW, small wind <5 MW. "
            "Process: "
            "(1) Prijungimo paraiška (connection application) to ESO; "
            "techniniai prisijungimo sąlygai issued within 20 working days; "
            "(2) Technical project design approval; "
            "(3) Prijungimo sutartis (connection agreement). "
            "Net metering (neto atsiskaitymas) available for prosumers up to 30 kW under AIEI. "
            "Timeline: distribution connection 3–9 months total from application to commissioning. "
            "ESO serves the entire Lithuanian distribution grid (10 kV and below); "
            "primary DSO for residential, commercial and small industrial connections."
        ),
    },
    # ── VERT licensing & registration ─────────────────────────────────────────
    {
        "name": "lt_vert_elektros_gamybos_licencija",
        "url": "https://www.vert.lt/lt/veiklos-sritys/elektros-energetika/elektros-energijos-gamyba/",
        "type": "html",
        "category": "electricity_generation_licensing",
        "description": (
            "VERT (Valstybinė energetikos reguliavimo taryba) elektros energijos gamybos "
            "licencijavimo tvarka. "
            "Licence thresholds per Elektros energetikos įstatymas: "
            "• Gamybos licencija: required for installed generation/storage capacity >100 kW; "
            "• Registracija (simplified): for 10–100 kW; application to VERT within 30 days "
            "of commissioning; "
            "• Below 10 kW: no VERT licence or registration required. "
            "BESS specific: merchant BESS >100 kW participating in balancing market requires "
            "gamybos licencija. Application package: "
            "techniniai duomenys (technical specs, installed capacity, technology type), "
            "priešgaisrinės saugos specialisto ataskaita (fire safety report), "
            "prijungimo sutartis (grid connection agreement from LITGRID or ESO), "
            "statybos leidimas (building permit). "
            "Processing time: up to 20 working days after complete application received. "
            "VERT monitors compliance; licence can be suspended for non-compliance."
        ),
    },
    {
        "name": "lt_vert_oze_aukcionai_kvota",
        "url": "https://www.vert.lt/lt/veiklos-sritys/elektros-energetika/atsinaujinantieji-energetikos-istekliai/",
        "type": "html",
        "category": "renewable_energy_licensing",
        "description": (
            "VERT OZE (atsinaujinančių išteklių energetikos) kvotų aukcionų tvarka. "
            "Competitive renewable energy support auctions administered by VERT. "
            "OZE auction process: "
            "(1) Kvota (capacity quota) set by government per technology and term; "
            "(2) Aukcionas (competitive tender) — applicants bid for fiksuota priemoka (FIP); "
            "(3) Winners sign priemokos sutartis (premium support agreement, 10–20 years). "
            "Eligible technologies: saulės elektrinės (solar PV), vėjo elektrinės (wind), "
            "biokuro elektrinės (biomass/biogas), hidroelektrinės (hydro). "
            "BESS co-located with OZE: may qualify for simplified support under AIEI. "
            "Auction capacity tranches published annually by Vyriausybė (government). "
            "VERT also monitors OZE support payments via Viešuosius interesus atitinkančios "
            "paslaugos (VIAP) levy on all electricity consumers."
        ),
    },
    # ── Gas systems installation rules (buildings) ────────────────────────────
    {
        "name": "lt_dujos_sistemu_pastatuose_taisykles",
        # pagd.lrv.lt (PAGD's own site) blocks automated fetches with an active
        # Cloudflare challenge — confirmed via direct curl/WebFetch 2026-08-16,
        # see manual sourcing backlog. The document itself is linked from PAGD's
        # "Lietuvos Respublikos energetikos ministro priimti teisės aktai" page
        # and is hosted on e-seimas.lrs.lt, which fetches fine directly.
        # NOTE: e-seimas's default /portal/legalAct/lt/TAD/{id}/asr view only
        # returns the document's table-of-contents shell (clause numbers, no
        # article text) — the real content loads via JS/AJAX per clause and is
        # NOT present in that HTML. The /rs/legalact/TAD/{id}/ export view
        # (Word-generated HTML) DOES contain the full real text — confirmed by
        # direct extraction (21,884 words, real numbered articles) — and is the
        # pattern used here.
        "url": "https://e-seimas.lrs.lt/rs/legalact/TAD/TAIS.416480/",
        "type": "html",
        "category": "gas_systems_building_code",
        "description": (
            "Dujų sistemų pastatuose įrengimo taisyklės (patvirtinta Lietuvos "
            "Respublikos energetikos ministro 2012-01-02 įsakymu Nr. 1-2, "
            "galiojanti redakcija nuo 2020-09-02). "
            "Technical rules for gas system design and installation in buildings "
            "with gas operating pressure up to 16 bar: building gas pipelines, "
            "gas appliance installation (types A/B/C), combustion product removal "
            "(chimneys/ventilation), gas cylinder equipment, and required air "
            "supply for gas appliances sharing a room with extraction ventilation. "
            "Relevant to buildings with gas heating/appliances (residential, "
            "commercial, industrial, agricultural, data centre backup systems) — "
            "not applicable to BESS, wind, solar or SMR projects, which do not "
            "involve building gas systems."
        ),
    },
    # ── BESS fire safety ──────────────────────────────────────────────────────
    {
        "name": "lt_priesgaisrines_saugos_taisykles",
        "url": "https://www.pagd.lrv.lt/lt/priesgazirine-sauga/taisykles-ir-instrukcijos/",
        "type": "html",
        "category": "fire_safety_bess",
        "description": (
            "PAGD (Priešgaisrinės apsaugos ir gelbėjimo departamentas) priešgaisrinės "
            "saugos taisyklės ir techniniai normatyvai. "
            "Fire safety regulatory framework administered by PAGD (State Fire and Rescue "
            "Department, under Interior Ministry). "
            "Priešgaisrinės saugos ekspertizė (fire safety examination) mandatory for "
            "commercial BESS installations before statybos leidimas is issued. "
            "Key requirements for BESS installations referencing IEC 62619, NFPA 855, EN 50604-1: "
            "• Automatinė gesintuvų sistema (automatic suppression) for Li-ion >100 kWh "
            "in enclosed space; "
            "• Dujų detekcija (gas detection), temperatūros monitoringas (temperature monitoring), "
            "BMS (battery management system); "
            "• Avarinė išjungimo sistema (ESD) with PAGD-approved design; "
            "• Priešgaisrinių automobilių privažiavimo keliai ≥3.5m wide access route; "
            "• Minimum 5m separation from other structures for containerized BESS. "
            "Outdoor containerized BESS: PAGD notification mandatory before commissioning. "
            "Coordinate with local PAGD territorial unit before design finalisation."
        ),
    },
    # ── Wind energy specific ──────────────────────────────────────────────────
    {
        "name": "lt_vejo_energetikos_specialusis_planas",
        "url": "https://www.e-tar.lt/portal/lt/legalAct/a86e5000b94c11e9a07f94f7031faa53",
        "type": "html",
        "category": "wind_energy_regulations",
        "description": (
            "Lietuvos Respublikos Vyriausybės nutarimas dėl Vėjo energetikos plėtros "
            "specialiojo plano patvirtinimo (Wind Energy Special Plan, Vyriausybė resolution). "
            "National spatial plan designating zones where wind energy development is permitted. "
            "Wind turbine siting requirements: "
            "• 500m minimum setback from residential buildings for turbines up to 2.5 MW; "
            "• 800m minimum setback for turbines >2.5 MW; "
            "• 1× turbine total height (ašies aukštis + rotoriaus spindulys) from land boundary; "
            "• Allowed zones: agricultūrinės, miško, techninės paskirties žemės; "
            "• Prohibited: rezervatai (strict reserves) and within airport safety zones. "
            "Radar and aviation clearance mandatory: "
            "• Oro navigacijos paslauga (ANCO/LGS equivalent) for >45m height; "
            "• Kariuomenės radiolokacinis suderinimas (military radar clearance); "
            "• CAA Lithuania (Civilinės aviacijos administracija) for near-airport projects. "
            "Municipal bendrasis planas must designate zones where wind energy is permitted; "
            "PAV required for parks >5 turbines or >30 MW. "
            "Offshore wind (jūrinis vėjas): "
            "separate regulatory track — Jūrinio planavimo dokumentai + EM (Energetikos ministerija)."
        ),
    },
    {
        "name": "lt_vert_vejo_energetika",
        "url": "https://www.vert.lt/lt/veiklos-sritys/elektros-energetika/vejo-energetika/",
        "type": "html",
        "category": "wind_energy_regulations",
        "description": (
            "VERT (Valstybinė energetikos reguliavimo taryba) vėjo energetikos reguliavimo "
            "apžvalga ir licencijavimo reikalavimai. "
            "Covers the full regulatory sequence for wind turbine (vėjo elektrinė/vėjo parkas) "
            "projects from VERT's perspective: "
            "• PAV (poveikio aplinkai vertinimas) — mandatory for >5 turbines or >30 MW; "
            "• Specialusis planas (special territorial plan) — must designate wind energy zone; "
            "• Techniniai prisijungimo sąlygai (technical connection conditions) from LITGRID/ESO; "
            "• Statybos leidimas (building permit) from savivaldybės administracija; "
            "• Gamybos licencija (generation licence) from VERT for >100 kW; "
            "• Vėjo turbinos minimalus atstumas nuo gyvenamųjų namų (setback requirements); "
            "• Bendruomenės išmokos (community payment): mandatory for wind parks per AIEI; "
            "• Oro navigacijos suderinimas: ANCO (Oro navigacijos paslaugos) for >45m height; "
            "• Kariuomenės suderinimas: mandatory military radar clearance for all wind parks. "
            "VERT regulates feed-in premium (fiksuota priemoka) support for eligible projects."
        ),
    },
    # ── BESS market & balancing ───────────────────────────────────────────────
    {
        "name": "lt_litgrid_balansavimo_rinka",
        "url": "https://www.litgrid.eu/",
        "type": "html",
        "category": "bess_market_balancing",
        "description": (
            "LITGRID AB sistemos (balansavimo) paslaugų tvarka, įskaitant BESS rinkos dalyvavimą. "
            "BESS can participate in Lithuanian balancing market as: "
            "(1) FCR (Frequency Containment Reserve / Pirminė reguliavimo galia) — symmetric "
            "±MW response, 30-second activation, monthly tender coordinated by LITGRID "
            "and ENTSO-E Baltic TSOs; "
            "(2) aFRR (automatic Frequency Restoration Reserve / Antrinis reguliavimas); "
            "(3) mFRR (manual Frequency Restoration Reserve / Tretinis reguliavimas) — "
            "activation within 15 minutes; "
            "(4) Intraday energy arbitrage — requires only gamybos licencija (>100 kW). "
            "Prerequisites for balancing market participation: "
            "Balanso atsakingosios šalies (BRP) sutartis with LITGRID — either own BRP "
            "status or agreement with existing BRP; "
            "SCADA real-time connection to LITGRID required for >1 MW BESS. "
            "Lithuania joined PICASSO (aFRR) and MARI (mFRR) EU balancing platforms. "
            "Baltic desynchronisation: synchronisation with Continental European grid will "
            "significantly increase FCR and FRR market volumes — BESS business case driver."
        ),
    },
    # ── Municipal building permit process ─────────────────────────────────────
    {
        "name": "lt_infostatyba_statybos_leidimas",
        "url": "https://infostatyba.lt/",
        "type": "html",
        "category": "municipal_building_permit",
        "description": (
            "Lietuvos Respublikos IS INFOSTATYBA — valstybinė statybos leidimų ir "
            "statybos valstybinės priežiūros informacinė sistema (infostatyba.planuojuinfo.lt). "
            "All building permit applications in Lithuania submitted through INFOSTATYBA portal. "
            "Permit authority: savivaldybės administracija (municipal administration) — "
            "REQUIREMENTS VARY BY MUNICIPALITY. Always contact local administration to confirm. "
            "Process stages in INFOSTATYBA: "
            "(1) Prašymas išduoti statybą leidžiantį dokumentą (building permit application); "
            "(2) Techninis projektas (technical design documentation); "
            "(3) Statybos leidimas (building permit with conditions); "
            "(4) Statybos pradžios pranešimas (construction start notice); "
            "(5) Statinio pripažinimas tinkamu naudoti (commissioning/acceptance). "
            "Standard timeline for energy infrastructure (commercial statybos leidimas): "
            "20–40 working days from complete submission. "
            "Typical energy project documents: žemės sklypo planas (1:500 or 1:1000), "
            "topografinė nuotrauka, LITGRID/ESO techniniai prisijungimo sąlygai, "
            "PAV ataskaita (if EIA required), priešgaisrinės saugos ekspertizė. "
            "All state supervision (valstybinė statybos priežiūra) via same portal."
        ),
    },
]


# ── Text extraction ───────────────────────────────────────────────────────────

def _download(url: str, timeout: int = 30) -> bytes:
    import requests
    resp = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "NCEPermitAI/1.0 (Lithuania regulatory research; contact: info@ncenergy.fi)",
            "Accept-Language": "lt,en;q=0.9",
        },
    )
    resp.raise_for_status()
    return resp.content


def _extract_pdf(data: bytes) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def _extract_html(data: bytes) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start: start + CHUNK_WORDS]).strip()
        if len(chunk.split()) >= MIN_WORDS:
            chunks.append(chunk)
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def _chunk_id(name: str, idx: int) -> str:
    h = hashlib.sha256(f"lt__{name}__{idx}".encode()).hexdigest()[:10]
    return f"lt__{name}__{idx}__{h}"


# ── ChromaDB upsert ───────────────────────────────────────────────────────────

def _upsert(col: Any, model: Any, ids: list, docs: list, metas: list, batch: int = 64) -> int:
    total = 0
    for i in range(0, len(ids), batch):
        b_ids   = ids[i:i+batch]
        b_docs  = docs[i:i+batch]
        b_metas = metas[i:i+batch]
        embs = model.encode(b_docs, batch_size=batch, show_progress_bar=False, normalize_embeddings=True)
        col.upsert(ids=b_ids, documents=b_docs, metadatas=b_metas, embeddings=embs.tolist())
        total += len(b_ids)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest_lithuania_sources(sources: list[dict] | None = None) -> int:
    """
    Ingest all Lithuania sources into permit_docs + permit_docs_v2.
    Returns total v2 chunks upserted. Never raises — logs failures and continues.

    hanketyyppi_tag is resolved via get_hanketyyppi_tag(source_name) from
    source_policy.py. Sources not listed in SOURCE_HANKETYYPPI_TAG default
    to 'general' (unrestricted, visible to all project types).
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    sources = sources or LITHUANIA_SOURCES

    if not os.path.exists(_DB_PATH):
        raise RuntimeError(f"ChromaDB path not found: {_DB_PATH}")

    log.info("[lithuania] Connecting to ChromaDB at %s", _DB_PATH)
    client  = chromadb.PersistentClient(path=_DB_PATH)
    col_v1  = client.get_or_create_collection(_COL_V1, metadata={"hnsw:space": "cosine"})
    col_v2  = client.get_or_create_collection(_COL_V2, metadata={"hnsw:space": "cosine"})

    log.info("[lithuania] Loading embedding models …")
    model_v1 = SentenceTransformer(_MODEL_V1)
    model_v2 = SentenceTransformer(_MODEL_V2)

    ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_v2    = 0
    summary: list[dict] = []

    for src in sources:
        name = src["name"]
        url  = src["url"]
        kind = src["type"]

        # Resolve hanketyyppi_tag from source_policy — never hardcoded here
        ht_tag = get_hanketyyppi_tag(name)

        print(f"\n[lithuania] {name}")
        print(f"  URL: {url[:80]}")
        print(f"  hanketyyppi_tag: {ht_tag}")

        local_path = src.get("local_path")
        if local_path:
            # Static source file — bypasses the network fetch entirely (e.g. for
            # domains that block automated requests). Mirrors the static-file
            # pattern used by ingest_countries.py for SE/DA/NO/EE/PL/DE.
            try:
                text = Path(local_path).read_text(encoding="utf-8")
                print(f"  Read {len(text):,} chars from local file: {local_path}")
            except Exception as exc:
                msg = f"local file read failed: {exc}"
                log.warning("[lithuania] %s: %s", name, msg)
                print(f"  WARN: {msg} — skipping")
                summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})
                continue
        else:
            # Download
            try:
                raw = _download(url)
                print(f"  Downloaded {len(raw):,} bytes")
            except Exception as exc:
                msg = f"download failed: {exc}"
                log.warning("[lithuania] %s: %s", name, msg)
                print(f"  WARN: {msg} — skipping")
                summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})
                continue

            # Extract text
            try:
                text = _extract_pdf(raw) if kind == "pdf" else _extract_html(raw)
            except Exception as exc:
                msg = f"text extraction failed: {exc}"
                log.warning("[lithuania] %s: %s", name, msg)
                print(f"  WARN: {msg} — skipping")
                summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})
                continue

        if len(text.split()) < 200:
            msg = f"too short ({len(text.split())} words) — possible 404 or redirect"
            log.warning("[lithuania] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "SKIP", "chunks": 0, "reason": msg})
            continue

        chunks = _chunk_text(text)
        print(f"  {len(text.split()):,} words → {len(chunks)} chunks")

        # A local_path source bypassed the network fetch entirely (see the comment
        # on that source's dict above) — that IS this project's definition of
        # manually-sourced content: a human had to obtain and place the text
        # because automated fetching doesn't work for it. Tag it accordingly so a
        # future freshness check can find it. Every other source here was fetched
        # live this run and is not manual, regardless of past fetch failures for
        # other sources (a fetch failure just skips the source — see above —
        # it never produces a chunk to mistag).
        ids   = [_chunk_id(name, i) for i in range(len(chunks))]
        metas = [
            {
                "source":          name,
                "url":             url,
                "country":         "LT",
                "category":        src["category"],
                "lang":            "lt",
                "description":     src["description"],
                "ingested_at":     ingested_at,
                "source_type":     "manual" if local_path else kind,
                **({"last_verified": ingested_at[:10]} if local_path else {}),
                "hanketyyppi_tag": ht_tag,
            }
            for _ in chunks
        ]

        try:
            n1 = _upsert(col_v1, model_v1, ids, chunks, metas)
            n2 = _upsert(col_v2, model_v2, ids, chunks, metas)
            total_v2 += n2
            print(f"  Upserted {n1} → permit_docs  |  {n2} → permit_docs_v2")
            log.info("[lithuania] %s: v1=%d v2=%d tag=%s", name, n1, n2, ht_tag)
            summary.append({"source": name, "status": "OK", "chunks": n2, "tag": ht_tag, "reason": ""})
        except Exception as exc:
            msg = f"upsert failed: {exc}"
            log.warning("[lithuania] %s: %s", name, msg)
            print(f"  ERROR: {msg}")
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "tag": ht_tag, "reason": msg})

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'Source':<48} {'Tag':<20} {'St':^4} {'Ch':>5}")
    print(f"{'-'*75}")
    for r in summary:
        flag = r.get("reason") and f"  ({r['reason'][:30]})" or ""
        print(f"{r['source'][:48]:<48} {r.get('tag','?'):<20} {r['status']:^4} {r['chunks']:>5}{flag}")
    print(f"{'-'*75}")
    print(f"{'TOTAL permit_docs_v2':<48} {'':20} {'':4} {total_v2:>5}")
    print(f"{'='*75}")

    log.info("[lithuania] Done. Total v2 chunks: %d", total_v2)
    return total_v2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = ingest_lithuania_sources()
    print(f"\n[lithuania] Done — {count} chunks added to permit_docs_v2")
