"""
Latvia regulatory RAG ingestion for NCE Permit AI.

Downloads public Latvian legal texts and regulatory guidance (likumi.lv, SPRK,
AST, VARAM, Sadales tīkls), extracts text, chunks by word count, and upserts
into both ChromaDB collections:
  - permit_docs    (MiniLM-L12-v2, 384-dim)  — v1, survives future reindex
  - permit_docs_v2 (mpnet 768-dim)            — active production collection

hanketyyppi_tag is always resolved via get_hanketyyppi_tag() from source_policy;
no tag value is hardcoded here so the BESS/wind/solar restriction logic stays
in a single source of truth.

Run standalone for testing:
    python3 backend/latvia_ingestion.py
Or trigger via API:
    POST /api/admin/ingest-latvia  (x-admin-secret header required)

Key Latvian regulatory framework notes:
  - SPRK (Sabiedrisko pakalpojumu regulēšanas komisija) — public utilities
    regulator: ražošanas licence required for generation/storage >1 MW;
    reģistrācija for 50 kW–1 MW; below 50 kW no SPRK registration needed.
  - AST (Augstsprieguma tīkls) — TSO, 110/330 kV transmission connection.
  - Sadales tīkls AS — DSO, distribution grid for smaller projects (<1 MW BESS,
    rooftop solar, small wind).
  - Building permits (būvatļauja) issued by pašvaldības būvvalde (municipal
    building authority) — requirements vary by municipality; always verify
    locally before submitting.
  - IVN (Ietekmes uz vidi novērtējums) — Latvian EIA process; screening
    authority is Vides pārraudzības valsts birojs (VPVB) within VARAM.
  - Latvia has no nuclear power plants or nuclear regulatory framework;
    SMR projects would require entirely new legislation.
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

_DB_PATH   = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings")
)
_COL_V1   = "permit_docs"
_COL_V2   = "permit_docs_v2"
_MODEL_V1 = "paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_V2 = "paraphrase-multilingual-mpnet-base-v2"

CHUNK_WORDS   = 800
OVERLAP_WORDS = 100
MIN_WORDS     = 50

LATVIA_SOURCES: list[dict] = [
    # ── Core electricity & energy laws ───────────────────────────────────────
    {
        "name": "lv_elektroenergijas_tirgus_likums",
        "url": "https://likumi.lv/ta/id/108834",
        "type": "html",
        "category": "energy_market_law",
        "description": (
            "Elektroenerģijas tirgus likums (ETL, 2005, consolidated). "
            "Central electricity market law. Regulates electricity generation, "
            "transmission, distribution and supply. SPRK licensing thresholds: "
            "ražošanas licence (production licence) required for >1 MW installed capacity; "
            "simplified reģistrācija for 50 kW–1 MW; below 50 kW no SPRK action needed. "
            "TSO = Augstsprieguma tīkls (AST); DSO = Sadales tīkls AS. "
            "BESS market access (balancing, FCR, mFRR) requires balance responsible party "
            "agreement with AST per ETL §§73-83."
        ),
    },
    {
        "name": "lv_energetikas_likums",
        "url": "https://likumi.lv/ta/id/50979",
        "type": "html",
        "category": "energy_framework_law",
        "description": (
            "Enerģētikas likums (1998, consolidated). "
            "Framework law for Latvia's energy sector: energy system planning, "
            "licensing, consumer protection, energy crisis management. "
            "Establishes SPRK as the independent public utilities regulator. "
            "Mandates national energy strategy (Latvijas enerģētikas ilgtermiņa stratēģija). "
            "Sets framework for energy independence and renewable transition targets."
        ),
    },
    {
        "name": "lv_atjaunojamas_energijas_likums",
        "url": "https://likumi.lv/ta/id/317215",
        "type": "html",
        "category": "renewable_energy_law",
        "description": (
            "MK noteikumi Nr. 317215 'Noteikumi par elektroenerģijas ražošanu, izmantojot "
            "atjaunīgos energoresursus, kā arī par cenu noteikšanas kārtību un uzraudzību' "
            "(consolidated). Governs wind, solar PV, hydro, biomass electricity production "
            "and BESS co-located with renewables in Latvia. "
            "OZE support framework: obligātais iepirkums (mandatory purchase) and prēmija "
            "(feed-in premium) mechanisms administered by SPRK. "
            "Renewables auction system (izsoles): SPRK administers competitive tenders. "
            "Grid priority (tīkla prioritāte) for renewables dispatch per ETL. "
            "Net metering (neto norēķini) available for prosumers up to 0.25 MW. "
            "BESS co-located with renewables: conditions for simplified SPRK registration. "
            "Target: 80% renewables in electricity consumption by 2030. "
            "Sets technical requirements for renewable energy plants and monitoring obligations."
        ),
    },
    # ── Environmental & EIA ───────────────────────────────────────────────────
    {
        "name": "lv_ivn_likums",
        "url": "https://likumi.lv/ta/id/45393",
        "type": "html",
        "category": "eia_law",
        "description": (
            "Likums par ietekmes uz vidi novērtējumu (IVN likums, consolidated). "
            "Latvia's Environmental Impact Assessment law. "
            "IVN = Ietekmes uz vidi novērtējums (equivalent to Finnish YVA/Swedish MKB). "
            "Two-stage process: (1) sākotnējā izvērtēšana (initial screening by VPVB), "
            "(2) full IVN if screening positive. "
            "Screening authority: Vides pārraudzības valsts birojs (VPVB), under VARAM. "
            "Mandatory IVN triggers for energy projects: wind farms >2 turbines or total capacity "
            ">5 MW; stand-alone BESS: no explicit threshold — VPVB screening recommended for "
            ">10 MWh or where significant land use change occurs. "
            "IVN programma (scoping document) → public consultation → IVN ziņojums (EIA report) "
            "→ VPVB atzinums (EIA opinion, binding). Typical duration: 6–18 months."
        ),
    },
    {
        "name": "lv_vides_aizsardzibas_likums",
        "url": "https://likumi.lv/ta/id/51306",
        "type": "html",
        "category": "environmental_protection_law",
        "description": (
            "Vides aizsardzības likums (1997, consolidated). "
            "Framework environmental protection law. "
            "Three-category environmental permit system: "
            "A kategorija (complex IPPC installations, Valsts vides dienests + VPVB), "
            "B kategorija (significant environmental impact, Valsts vides dienests), "
            "C kategorija (minor impact, municipal level). "
            "BESS fire and chemical risk: lithium-ion battery installations >1 MWh in "
            "commercial/industrial setting may trigger B-kategorija atļauja. "
            "Administered by Valsts vides dienests (VVD), regional offices across Latvia."
        ),
    },
    {
        "name": "lv_mk_ivn_kategorijas_noteikumi",
        "url": "https://likumi.lv/ta/id/344494",
        "type": "html",
        "category": "eia_categories",
        "description": (
            "Ministru kabineta noteikumi par ietekmes uz vidi novērtējumu un sākotnējo "
            "izvērtēšanu (MK noteikumi, aktuālā redakcija). "
            "Lists projects requiring mandatory IVN vs. screening only. "
            "Wind energy: >2 turbines always triggers screening; parks >5 MW or >5 turbines "
            "→ mandatory full IVN. "
            "Solar PV: >50 ha site area triggers screening. "
            "BESS standalone: listed under industrial installations — VPVB screening "
            "recommended for all grid-scale projects. "
            "Process: submit IVN sākotnējās izvērtēšanas pieteikums to VPVB, decision within 30 days."
        ),
    },
    # ── Building & spatial planning ───────────────────────────────────────────
    {
        "name": "lv_buvniecibas_likums",
        "url": "https://likumi.lv/ta/id/258572",
        "type": "html",
        "category": "construction_law",
        "description": (
            "Būvniecības likums (2013, consolidated 2024). "
            "Three-tier building permit system based on project complexity and risk: "
            "(1) Apliecinājums (notification/certificate) — for minor works, simple structures; "
            "(2) Paskaidrojuma raksts (explanatory note) — medium complexity; "
            "(3) Būvatļauja (building permit) — required for complex/larger energy structures. "
            "BESS ≥50 kW installed capacity or multi-container/permanent installation: typically "
            "requires Būvatļauja from pašvaldības būvvalde (municipal building authority). "
            "Digital submission via BIS (Būvniecības informācijas sistēma, bis.gov.lv). "
            "Required attachments: zemes vienības kadastra apzīmējums, topogrāfiskā plāna "
            "izraksts, technical design documentation (tehniskie noteikumi from AST/Sadales tīkls), "
            "IVN atzinums (if EIA required), fire safety certificate (ugunsdrošības atzinums). "
            "Atbildīgais būvdarbu vadītājs (licensed construction supervisor) mandatory. "
            "IMPORTANT: specific requirements vary by pašvaldība (municipality) — "
            "always contact local būvvalde to confirm exact documentation requirements."
        ),
    },
    {
        "name": "lv_teritorijas_planosanas_likums",
        "url": "https://likumi.lv/ta/id/238807",
        "type": "html",
        "category": "spatial_planning_law",
        "description": (
            "Teritorijas attīstības plānošanas likums (2011, consolidated). "
            "Governs spatial planning at national, regional and local (novads) levels. "
            "Three planning instrument types: "
            "(1) Novada teritorijas plānojums (municipal land-use plan, ~10-year horizon); "
            "(2) Lokālplānojums (local detailed plan for specific area); "
            "(3) Detālplānojums (detailed plan for specific plot/development). "
            "Energy projects (wind, solar, BESS) must comply with novada teritorijas plānojums. "
            "Detālplānojums required for industrial/energy complexes typically >1 ha or when "
            "the current territorial plan does not allow the intended use. "
            "Approval: domes lēmums (municipal council decision) required for plan amendments; "
            "public consultation (sabiedriskā apspriešana) mandatory. "
            "National significance projects: may require valsts nozīmes teritorijas plānojums "
            "(national spatial plan layer) — Minister of Economics decision."
        ),
    },
    # ── Grid connection — transmission ────────────────────────────────────────
    {
        "name": "lv_ast_sistemas_pievienošanas_noteikumi",
        "url": "https://likumi.lv/ta/id/326536",
        "type": "html",
        "category": "grid_connection_transmission",
        "description": (
            "MK noteikumi 'Sistēmas pieslēguma noteikumi elektroenerģijas pārvades sistēmai' "
            "(ID 326536, likumi.lv). Regulatory framework for 110/330 kV transmission "
            "grid connection in Latvia, administered by AST (Augstsprieguma tīkls). "
            "Applies to: generators and storage connecting at transmission voltage level "
            "(typically >10 MW BESS, large wind/solar parks, industrial consumers >1 MW). "
            "Three-step process: "
            "(1) Tehniskie noteikumi (technical conditions) — submit application to AST, "
            "AST issues conditions within 30 working days; "
            "(2) Projektēšana (design phase) — develop technical design per AST conditions; "
            "(3) Pievienošanas līgums (connection agreement) — before commissioning. "
            "AST also administers system services: FCR (Frequency Containment Reserve), "
            "FRR (Frequency Restoration Reserve), mFRR — BESS participation requires "
            "separate atbildīgās puses līgums (balance responsible party agreement). "
            "Latvia–Lithuania–Estonia synchronisation with Continental European grid "
            "(desynchronisation from IPS/UPS) — impacts grid connection technical "
            "requirements and reserve market obligations."
        ),
    },
    # ── Grid connection — distribution ────────────────────────────────────────
    {
        "name": "lv_sadales_tikls_pievienošanas_kārtiba",
        "url": "https://sadalestikls.lv/lv/elektrostacijas-pieslegsana",
        "type": "html",
        "category": "grid_connection_distribution",
        "description": (
            "Sadales tīkls AS pieslēguma kārtība atjaunojamajiem un uzkrāšanas sistēmām. "
            "Distribution grid connection for projects at 0.4–35 kV level. "
            "Used for: BESS <10 MW, rooftop/ground-mount solar <5 MW, small wind <10 MW. "
            "Process: "
            "(1) Pieslēguma pieteikums (connection application) to Sadales tīkls, "
            "technical conditions issued within 20 working days; "
            "(2) Technical design approval; "
            "(3) Pieslēguma līgums (connection agreement). "
            "Net metering (neto norēķini) available for prosumers up to 250 kW under AEL. "
            "Timeline: distribution connection 3–9 months total from application to commissioning. "
            "Sadales tīkls serves ~99% of Latvia's electricity consumers; "
            "DSO area covers entire country (monopoly distribution operator)."
        ),
    },
    # ── SPRK licensing & registration ─────────────────────────────────────────
    {
        "name": "lv_sprk_elektroenerģijas_ražošanas_licence",
        "url": "https://www.sprk.gov.lv/content/registresanalicencesana-1",
        "type": "html",
        "category": "electricity_generation_licensing",
        "description": (
            "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija) elektroenerģijas "
            "ražošanas licencēšanas kārtība. "
            "Licence thresholds: "
            "• Ražošanas licence: required for installed generation/storage capacity >1 MW; "
            "• Reģistrācija (simplified): for 50 kW–1 MW; application to SPRK within 30 days "
            "of commissioning; "
            "• Below 50 kW: no SPRK licence or registration required. "
            "BESS specific: merchant BESS >1 MW participating in balancing market requires "
            "ražošanas licence. Application package: "
            "tehniskie parametri (technical specs), ugunsdrošības atzinums (fire safety "
            "certificate), pievienošanas līgums (grid connection agreement from AST/Sadales tīkls), "
            "būvatļauja (building permit) or equivalent. "
            "Processing time: up to 30 working days after complete application received. "
            "SPRK monitors compliance annually; licence can be suspended for non-compliance."
        ),
    },
    {
        "name": "lv_mk_631_licencesanas_noteikumi",
        "url": "https://likumi.lv/ta/id/348679",
        "type": "html",
        "category": "electricity_licensing_regulations",
        "description": (
            "MK noteikumi Nr. 348679 'Noteikumi par atļaujām jaunas elektroenerģijas "
            "ražošanas iekārtas ieviešanai vai elektroenerģijas ražošanas jaudas "
            "palielināšanai' (consolidated). "
            "Permit procedure for introducing new electricity generation capacity in Latvia. "
            "Applies to wind, solar, BESS, and other electricity generation technologies. "
            "Required documentation for atļauja (permit) application: "
            "1. Uzņēmuma reģistrācijas apliecība (company registration certificate, UR); "
            "2. Tehniskie parametri (installed capacity, voltage level, technology type); "
            "3. Būvatļauja or equivalent (building permit from pašvaldības būvvalde); "
            "4. Pievienošanas līgums (grid connection agreement from AST or Sadales tīkls); "
            "5. Ugunsdrošības atzinums (fire safety certificate — VUGD inspection); "
            "6. IVN atzinums (EIA opinion, if EIA was required). "
            "Processing: 30 working days. Issued by the Ministry of Economics (Ekonomikas ministrija). "
            "Licence valid indefinitely but subject to annual compliance verification by SPRK."
        ),
    },
    # ── VARAM / EIA process guidance ──────────────────────────────────────────
    {
        "name": "lv_varam_ivn_procedura",
        "url": "https://www.varam.gov.lv/lv/ietekmes-uz-vidi-novertejums",
        "type": "html",
        "category": "eia_process_guidance",
        "description": (
            "VARAM (Vides aizsardzības un reģionālās attīstības ministrija) IVN procedūras "
            "rokasgrāmata. "
            "Latvia EIA process overview: "
            "Step 1 — sākotnējā izvērtēšana (initial screening): submit pieteikums to VPVB; "
            "VPVB issues lēmums within 30 days — either IVN required or not. "
            "Step 2 — IVN programma (scoping): if IVN required, prepare and submit programma "
            "to VPVB and pašvaldība; public consultation (sabiedriskā apspriešana) ≥14 days. "
            "Step 3 — IVN ziņojums (EIA report): comprehensive impact assessment document. "
            "Step 4 — Atzinums (EIA opinion): VPVB binding decision on acceptability. "
            "IVN atzinums must be positive before building permit application is submitted. "
            "Timeline: complete IVN 12–24 months for large energy projects. "
            "Competent authority: VPVB (Vides pārraudzības valsts birojs) under VARAM."
        ),
    },
    # ── BESS market & balancing ───────────────────────────────────────────────
    {
        "name": "lv_bess_balansesanas_tirgus_ast",
        "url": "https://www.sprk.gov.lv/lv/elektroenergijas-nozare-0",
        "type": "html",
        "category": "bess_market_balancing",
        "description": (
            "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija) elektroenerģijas nozares "
            "regulatīvā informācija, tostarp par BESS tirgus dalību un balansēšanas pakalpojumiem. "
            "BESS can participate in Latvian balancing market as: "
            "(1) FCR provider (Frequency Containment Reserve) — symmetric ±MW response, "
            "30-second activation time, monthly tender coordinated by AST and ENTSO-E Baltic TSOs; "
            "(2) mFRR provider (manual Frequency Restoration Reserve) — activation within 12.5 min; "
            "(3) Intraday energy arbitrage — no specific licence beyond ražošanas licence. "
            "Prerequisites for balancing market: "
            "atbildīgā puse (balance responsible party) status — either own BRP or agree "
            "with existing BRP; SCADA connection to AST required for >1 MW BESS. "
            "Latvia joined PICASSO (FCR/aFRR) and MARI (mFRR) EU balancing platforms. "
            "Baltic desynchronisation from IPS/UPS and synchronisation with Continental European "
            "grid significantly impacts reserve sizing and balancing market volumes in Latvia. "
            "Note: AST (augstsprieguma-tikls.lv) direct pages use bot protection; "
            "regulatory framework available via SPRK and likumi.lv."
        ),
    },
    # ── Wind energy specific ──────────────────────────────────────────────────
    {
        "name": "lv_veja_energija_buvniecibas_noteikumi",
        "url": "https://likumi.lv/ta/id/256866",
        "type": "html",
        "category": "wind_energy_regulations",
        "description": (
            "MK noteikumi Nr. 256866 'Vispārīgie teritorijas plānošanas, izmantošanas un "
            "apbūves noteikumi' (consolidated). General territorial planning regulation "
            "that contains specific wind turbine (vēja elektrostacija) siting requirements. "
            "Wind turbine placement rules (§§covering enerģētikas objekti): "
            "• 500m minimum setback from residential buildings for turbines 20 kW–2 MW; "
            "• 800m minimum setback for larger wind parks; "
            "• Allowed in: industrial (R), technical (TA), agricultural (L), forest (M) zones; "
            "• Prohibited in nature reserves and near protected habitats per IVN atzinums. "
            "• 1× rotor diameter minimum from neighbouring land boundary. "
            "Radar and aviation clearance mandatory: "
            "• LGS (Latvijas gaisa satiksme) — aviation obstacle clearance for >60m height; "
            "• NBS (Nacionālie bruņotie spēki) — military radar clearance; "
            "• LAAAS radar clearance near airports. "
            "Municipal territory plan (novada teritorijas plānojums) must designate zones "
            "where wind energy is permitted; IVN required for parks >2 turbines. "
            "Offshore wind: separate regulatory track — Jūras kodekss + Ekonomikas ministrija."
        ),
    },
    # ── Municipal building permit process ─────────────────────────────────────
    {
        "name": "lv_buvniecibas_informacijas_sistema_bis",
        "url": "https://bis.gov.lv/lv/buvniecibas-process/buvniecibas-ieceres-iesniegums",
        "type": "html",
        "category": "municipal_building_permit",
        "description": (
            "Latvijas Būvniecības informācijas sistēma (BIS, bis.gov.lv) — "
            "digitālā būvatļaujas pieteikšanas platforma. "
            "All building permit applications in Latvia submitted through BIS portal. "
            "Permit authority: pašvaldības būvvalde (municipal building authority) — "
            "REQUIREMENTS VARY BY MUNICIPALITY. Always contact local būvvalde to confirm. "
            "Process stages in BIS: "
            "(1) Būvniecības ieceres iesniegums (building intent application); "
            "(2) Būvprojekts minimālā sastāvā (minimum design documentation); "
            "(3) Būvatļauja (building permit with conditions); "
            "(4) Atzīme par projektēšanas nosacījumu izpildi (design completion sign-off); "
            "(5) Atzīme par būvdarbu uzsākšanu (construction start notice); "
            "(6) Nodošana ekspluatācijā (commissioning/handover). "
            "Standard timeline for energy infrastructure (commercial Būvatļauja): "
            "30–60 working days from complete submission. "
            "Typical energy project documents: atrašanās vietas shēma, "
            "topogrāfija (1:500 or 1:1000), AST/Sadales tīkls tehniskie noteikumi, "
            "IVN atzinums (if EIA required), ugunsdrošības atzinums."
        ),
    },
    # ── BESS fire safety ──────────────────────────────────────────────────────
    {
        "name": "lv_ugunsdrošibas_bess_prasibas",
        "url": "https://likumi.lv/ta/id/364042",
        "type": "html",
        "category": "fire_safety_bess",
        "description": (
            "Ugunsdrošības, ugunsdzēsības un glābšanas darbu likums (likumi.lv ID 364042, "
            "in force 13.11.2025). New consolidated fire safety framework law administered "
            "by VUGD (Valsts ugunsdzēsības un glābšanas dienests). "
            "Replaces previous Ugunsdrošības un ugunsdzēsības likums (ID 68293). "
            "Ugunsdrošības atzinums (fire safety certificate) mandatory for commercial BESS. "
            "Requirements reference IEC 62619, NFPA 855, EN 50604-1 for Li-ion storage systems. "
            "VUGD inspection (ugunsdrošības pārbaude) required before commissioning. "
            "Key requirements for BESS installations: "
            "• Sprinkler system or equivalent suppression for Li-ion >100 kWh in enclosed space; "
            "• Gas detection, temperature monitoring, BMS (battery management system); "
            "• Emergency shutdown (ESD) system with VUGD-approved design; "
            "• Firefighter access route (ugunsdzēsēju piekļuves ceļš) ≥3.5m wide; "
            "• Minimum 5m separation from other structures. "
            "Outdoor containerized BESS: simplified requirements but mandatory VUGD notification. "
            "Coordinate with local VUGD territorial department before design finalisation."
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
            "User-Agent": "NCEPermitAI/1.0 (Latvia regulatory research; contact: info@ncenergy.fi)",
            "Accept-Language": "lv,en;q=0.9",
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
    h = hashlib.sha256(f"lv__{name}__{idx}".encode()).hexdigest()[:10]
    return f"lv__{name}__{idx}__{h}"


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

def ingest_latvia_sources(sources: list[dict] | None = None) -> int:
    """
    Ingest all Latvia sources into permit_docs + permit_docs_v2.
    Returns total v2 chunks upserted. Never raises — logs failures and continues.

    hanketyyppi_tag is resolved via get_hanketyyppi_tag(source_name) from
    source_policy.py. Sources not listed in SOURCE_HANKETYYPPI_TAG default
    to 'general' (unrestricted, visible to all project types).
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    sources = sources or LATVIA_SOURCES

    if not os.path.exists(_DB_PATH):
        raise RuntimeError(f"ChromaDB path not found: {_DB_PATH}")

    log.info("[latvia] Connecting to ChromaDB at %s", _DB_PATH)
    client  = chromadb.PersistentClient(path=_DB_PATH)
    col_v1  = client.get_or_create_collection(_COL_V1, metadata={"hnsw:space": "cosine"})
    col_v2  = client.get_or_create_collection(_COL_V2, metadata={"hnsw:space": "cosine"})

    log.info("[latvia] Loading embedding models …")
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

        print(f"\n[latvia] {name}")
        print(f"  URL: {url[:80]}")
        print(f"  hanketyyppi_tag: {ht_tag}")

        # Download
        try:
            raw = _download(url)
            print(f"  Downloaded {len(raw):,} bytes")
        except Exception as exc:
            msg = f"download failed: {exc}"
            log.warning("[latvia] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})
            continue

        # Extract text
        try:
            text = _extract_pdf(raw) if kind == "pdf" else _extract_html(raw)
        except Exception as exc:
            msg = f"text extraction failed: {exc}"
            log.warning("[latvia] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "FAIL", "chunks": 0, "reason": msg})
            continue

        if len(text.split()) < 200:
            msg = f"too short ({len(text.split())} words) — possible 404 or redirect"
            log.warning("[latvia] %s: %s", name, msg)
            print(f"  WARN: {msg} — skipping")
            summary.append({"source": name, "status": "SKIP", "chunks": 0, "reason": msg})
            continue

        chunks = _chunk_text(text)
        print(f"  {len(text.split()):,} words → {len(chunks)} chunks")

        ids   = [_chunk_id(name, i) for i in range(len(chunks))]
        metas = [
            {
                "source":          name,
                "url":             url,
                "country":         "LV",
                "category":        src["category"],
                "lang":            "lv",
                "description":     src["description"],
                "ingested_at":     ingested_at,
                "source_type":     kind,
                "hanketyyppi_tag": ht_tag,
            }
            for _ in chunks
        ]

        try:
            n1 = _upsert(col_v1, model_v1, ids, chunks, metas)
            n2 = _upsert(col_v2, model_v2, ids, chunks, metas)
            total_v2 += n2
            print(f"  Upserted {n1} → permit_docs  |  {n2} → permit_docs_v2")
            log.info("[latvia] %s: v1=%d v2=%d tag=%s", name, n1, n2, ht_tag)
            summary.append({"source": name, "status": "OK", "chunks": n2, "tag": ht_tag, "reason": ""})
        except Exception as exc:
            msg = f"upsert failed: {exc}"
            log.warning("[latvia] %s: %s", name, msg)
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

    log.info("[latvia] Done. Total v2 chunks: %d", total_v2)
    return total_v2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = ingest_latvia_sources()
    print(f"\n[latvia] Done — {count} chunks added to permit_docs_v2")
