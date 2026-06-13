"""
FastAPI-backend BESS-kaavoituskartoitustyökalulle.
Pöytyä, kiinteistötunnus 636-439-4-711.

Käynnistys:
    cd bess_tool/backend && uvicorn main:app --reload --port 8000
"""

# TODO: domain muutos ncepermit.ai kun NCE Global perustettu

import asyncio
import base64
import email.mime.multipart
import email.mime.text
import io
import json
import logging
import os
import re
import secrets
import smtplib
import time
import unicodedata
import uuid
from collections import defaultdict
from threading import Thread, Lock
from typing import Optional

import requests as _requests

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from mml_api import (
    get_land_use, get_natura_areas, get_property_boundaries,
    get_zoning_info, get_groundwater_areas, infer_zoning_from_osm,
    get_flood_risk,
)
from finnish_authorities import get_pelastuslaitos, get_ely, genitive
from fingrid_api import (
    get_transmission_lines, get_buildings, get_highways, get_substations,
    nearest_line_distance_m, nearest_point_distance_m, nearest_substation_info,
)
from heritage_api import get_heritage_sites
from gtk_api import get_soil_type
from ai_strategy import get_lupaprosessi_strategy
from report import generate_bess_report
from permit_ai import query_permit_ai

# permit_ai-moduuli on ~/bess_tool/permit_ai/ — lisätään polkuun
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "permit_ai"))
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate_application import (
    generate_application, generate_application_draft, apply_proofread_to_pdf,
    ApplicationInput, _get_embed_model, _get_chroma_col,
    InsufficientSourcesError,
)
try:
    from optimizer import NCEOptimizer, EnergySite
    _OPTIMIZER_OK = True
except ImportError:
    _OPTIMIZER_OK = False
    print("[startup] optimizer.py ei löydy — /api/optimize-bess palauttaa 501")

# Warmup: lataa embedding-malli ja ChromaDB heti käynnistyksen yhteydessä,
# ei ensimmäisen requestin yhteydessä.
try:
    _get_embed_model()
    _get_chroma_col()
    print("[startup] Embedding-malli ja ChromaDB ladattu")
except Exception as _e:
    print(f"[startup] Varoitus: RAG-lataus epäonnistui: {_e}")

limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])

app = FastAPI(
    title="BESS-kaavoituskartoitus API",
    description="Pöytyä 636-439-4-711 – akkuvarastohankkeen sijaintianalyysi",
    version="2.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    # Auth disabled when BASIC_AUTH_PASS not set (local dev)
    if not _AUTH_PASS:
        return await call_next(request)

    # Only enforce auth on the tool subdomain (ai.ncenergy.fi).
    # ncenergy.fi landing page and localhost pass through unconditionally.
    host = request.headers.get("host", "")
    if "ai.ncenergy" not in host:
        return await call_next(request)

    # Three paths remain public on the tool domain
    if request.url.path in _TOOL_EXEMPT:
        return await call_next(request)

    # All other requests — including /, /static/*, /api/* — require credentials
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Basic "):
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="NCE Permit AI"'},
        )
    try:
        decoded  = base64.b64decode(auth[6:]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="NCE Permit AI"'},
        )
    ok = secrets.compare_digest(username.encode(), _AUTH_USER.encode()) and \
         secrets.compare_digest(password.encode(), _AUTH_PASS.encode())
    if not ok:
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="NCE Permit AI"'},
        )
    return await call_next(request)


@app.middleware("http")
async def add_charset(request, call_next):
    response = await call_next(request)
    if "application/json" in response.headers.get("content-type", ""):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR    = os.path.dirname(_BACKEND_DIR)
_STATIC_DIR  = os.path.join(_BACKEND_DIR, "static")
_LANDING_DIR = _REPO_DIR  # root index.html lives here
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Root assets/ for the landing page (ncenergy.fi / www.ncenergy.fi).
# Mount only if the folder exists so the app still starts without it.
_ASSETS_DIR = os.path.join(_REPO_DIR, "assets")
if os.path.isdir(_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

MML_API_KEY   = os.getenv("MML_API_KEY", "")
PORT          = int(os.environ.get("PORT", 8000))
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
ALERT_EMAIL   = os.getenv("ALERT_EMAIL", "jere@ncenergy.fi")

_AUTH_USER   = os.getenv("BASIC_AUTH_USER", "nce")
_AUTH_PASS   = os.getenv("BASIC_AUTH_PASS", "")  # empty = auth disabled (local dev)
# Paths that remain public even on ai.ncenergy.fi (landing page counter, contact form, health check)
_TOOL_EXEMPT = {"/api/stats", "/api/access-request", "/api/health"}

SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# ── Usage monitoring ──────────────────────────────────────────────────────────
_usage_logger = logging.getLogger("usage")
_usage_logger.setLevel(logging.INFO)
_ip_window: dict[str, list[float]] = defaultdict(list)   # ip → [timestamps]
_ip_lock = Lock()
_ALERT_WINDOW_SEC  = 600   # 10 min
_ALERT_THRESHOLD   = 3     # max calls per window before alert
_alerted_ips: set[str] = set()  # avoid duplicate alerts per server lifetime


def _log_usage(ip: str, hanketyyppi: str, country: str, phase: str,
               job_id: str, status: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _usage_logger.info(
        "[USAGE] timestamp=%s ip=%s type=%s country=%s phase=%s job_id=%s status=%s",
        ts, ip, hanketyyppi, country, phase, job_id, status,
    )
    if not ip or ip in ("testclient", "127.0.0.1"):
        return
    now = time.monotonic()
    with _ip_lock:
        calls = [t for t in _ip_window[ip] if now - t < _ALERT_WINDOW_SEC]
        calls.append(now)
        _ip_window[ip] = calls
        should_alert = len(calls) > _ALERT_THRESHOLD and ip not in _alerted_ips
        if should_alert:
            _alerted_ips.add(ip)
    if should_alert:
        Thread(target=_send_alert, args=(ip, len(calls), ts), daemon=True).start()


def _send_alert(ip: str, count: int, ts: str) -> None:
    if not RESEND_API_KEY:
        _usage_logger.warning("[USAGE] ALERT: ip=%s count=%d — RESEND_API_KEY puuttuu, sähköposti lähettämättä", ip, count)
        return
    try:
        _requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json={
                "from": "NCE Permit AI <noreply@ncenergy.fi>",
                "to": [ALERT_EMAIL],
                "subject": f"[NCE Permit AI] Hälytys: {count} kutsua 10 min — IP {ip}",
                "text": (
                    f"Epäilyttävä käyttö havaittu.\n\n"
                    f"IP: {ip}\n"
                    f"Kutsuja viimeisen 10 min aikana: {count}\n"
                    f"Aika: {ts}\n\n"
                    f"Tarkista Render-lokit lisätietoja varten."
                ),
            },
            timeout=10,
        )
        _usage_logger.info("[USAGE] ALERT lähetetty: ip=%s count=%d", ip, count)
    except Exception as exc:
        _usage_logger.warning("[USAGE] ALERT-lähetys epäonnistui: %s", exc)

if not MML_API_KEY:
    print("[startup] VAROITUS: MML_API_KEY ei asetettu — maankäyttöselvityksen WFS-haut eivät toimi. "
          "Aseta ympäristömuuttuja tai lisää Render-palveluun. Ks. README.md.")


# ── Pydantic-mallit ───────────────────────────────────────────────────────────

class AccessRequestModel(BaseModel):
    yritys: str
    yhteyshenkilo: str
    sahkoposti: str
    puhelin: str = ""
    kuvaus: str


class PermitAIRequest(BaseModel):
    question: str
    n_results: int = 5


class OptimizeRequest(BaseModel):
    bbox: list          # [lat_min, lon_min, lat_max, lon_max]
    project_type: str   = "bess"
    power_mw:     float = 5.0
    min_area_ha:  float = 2.0


class ApplicationRequest(BaseModel):
    hanketyyppi:                  str
    kiinteistotunnus:             str
    teho_mw:                      Optional[float] = 0.0
    kapasiteetti_mwh:             Optional[float] = 0.0
    y_tunnus:                     Optional[str]   = None
    osoite:                       Optional[str]   = None
    kunta:                        str
    hakija:                       str
    sijainti_ymparistovaikutukset: Optional[str]   = None
    hankkeen_vaihe:               Optional[str]   = None
    kohdeviranomainen:            Optional[str]   = None
    lang:                         Optional[str]   = "FI"
    country:                      Optional[str]   = "FI"
    session_id:                   Optional[str]   = ""
    # IFC esitäyttö (valinnainen)
    ifc_floor_area:               Optional[float] = 0.0
    ifc_building_height:          Optional[float] = 0.0
    ifc_fire_rating:              Optional[str]   = ""
    ifc_materials:                Optional[str]   = ""
    ifc_storeys:                  Optional[int]   = 0
    ifc_compliance_flags:         Optional[str]   = ""


# ── Oikolukutehtävien in-memory-varasto ──────────────────────────────────────
# {job_id: {status: pending|running|done|error, pdf_bytes: bytes|None, error: str|None}}
_proofread_store: dict = {}


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kiinteistotunnus: str
    title: Optional[str] = None
    map_image: Optional[str] = None          # base64 PNG (vanhentunut, käytetään vain fallbackina)
    property_geojson: Optional[dict] = None  # GeoJSON frontendilta – vältetään kaksoisnouto
    analysis_result: Optional[dict] = None   # Esikäsitelty analyysi – sama arvo UI:hin ja PDF:ään
    project_owner:   str = "Carbon Zero Finland"
    project_name:    str = "Standalone BESS 1 MW"
    power_mw:        float = 1.0
    grid_connection: str = "Jakeluverkko 20 kV (Caruna)"
    market:          str = "FCR (Frequency Containment Reserve)"
    # Manuaaliset syötteet
    manual_kaavoitus:    Optional[str]   = None  # asemakaava|yleiskaava|ei_kaavaa|ei_tietoa
    manual_tulvavaara:   Optional[str]   = None  # ei|kyllä|ei_tietoa
    manual_maapera:      Optional[str]   = None  # kallio|moreeni|hiekka|savi|turve|ei_tietoa
    manual_pinta_ala_ha: Optional[float] = None
    lang: Optional[str] = "FI"


# ── Endpointit ────────────────────────────────────────────────────────────────

@app.get("/")
async def root(request: Request):
    host = request.headers.get("host", "")
    if "ai.ncenergy" in host:
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))
    # ncenergy.fi, www.ncenergy.fi, localhost → landing page
    landing = os.path.join(_LANDING_DIR, "index.html")
    if os.path.isfile(landing):
        return FileResponse(landing)
    # fallback: tool (landing page not deployed yet)
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/privacy")
async def privacy():
    return FileResponse(os.path.join(_STATIC_DIR, "privacy.html"))


@app.get("/tietosuoja")
async def tietosuoja():
    return FileResponse(os.path.join(_STATIC_DIR, "privacy.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "mml_key_set": bool(MML_API_KEY)}


@app.post("/api/access-request")
async def access_request(req: AccessRequestModel):
    def _send():
        msg = email.mime.multipart.MIMEMultipart()
        msg["From"]    = SMTP_USER or "info@ncenergy.fi"
        msg["To"]      = "info@ncenergy.fi"
        msg["Subject"] = "Käyttöoikeuspyyntö — NCE Permit AI"
        body = (
            "Käyttöoikeuspyyntö — NCE Permit AI\n"
            "=====================================\n\n"
            f"Yritys:           {req.yritys}\n"
            f"Yhteyshenkilö:    {req.yhteyshenkilo}\n"
            f"Sähköposti:       {req.sahkoposti}\n"
            f"Puhelin:          {req.puhelin or '—'}\n\n"
            "Kuvaus toiminnasta:\n"
            "-------------------\n"
            f"{req.kuvaus}\n"
        )
        msg.attach(email.mime.text.MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)

    if not SMTP_USER or not SMTP_PASSWORD:
        logging.getLogger("usage").warning(
            "[ACCESS-REQUEST] SMTP not configured — yritys=%s email=%s",
            req.yritys, req.sahkoposti,
        )
        raise HTTPException(status_code=503, detail="Email service not configured")

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _send)
    except Exception as exc:
        logging.getLogger("usage").error("[ACCESS-REQUEST] SMTP error: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to send email")

    return {"ok": True}


@app.get("/api/debug-raw")
async def debug_raw():
    """Palauttaa viimeisimmän Claude-vastauksen /tmp/debug_raw_claude.txt."""
    try:
        with open("/tmp/debug_raw_claude.txt", encoding="utf-8") as f:
            content = f.read()
        return {"content": content[:3000]}
    except FileNotFoundError:
        return {"content": "Ei debug-tiedostoa — aja ensin generaatio."}


@app.get("/api/debug-encoding")
def debug_encoding():
    data = {"raw": "testiäö", "ae": "ä", "oe": "ö"}
    return JSONResponse(
        content=data,
        media_type="application/json; charset=utf-8",
    )


@app.get("/api/property/{kiinteistotunnus}")
async def property_boundaries(
    kiinteistotunnus: str,
    api_key: Optional[str] = Query(default=None),
):
    """Kiinteistörajat MML INSPIRE WFS:stä (ei API-avainta tarvita)."""
    try:
        return await get_property_boundaries(kiinteistotunnus, api_key=api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"INSPIRE WFS -virhe: {exc}")


@app.get("/api/fingrid/lines")
async def fingrid_lines(
    bbox: str = Query(default="22.5,60.6,23.0,60.9"),
    min_voltage_kv: int = Query(default=0),
):
    """Voimajohdot OSM Overpass -rajapinnasta (ei API-avainta)."""
    try:
        coords = [float(x) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox: minlon,minlat,maxlon,maxlat")
    return await get_transmission_lines(tuple(coords))


@app.get("/api/groundwater")
async def groundwater(bbox: str = Query(default="22.5,60.6,23.0,60.9")):
    """Pohjavesialueet SYKE Hakku -rajapinnasta (ei API-avainta)."""
    try:
        coords = [float(x) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox: minlon,minlat,maxlon,maxlat")
    return await get_groundwater_areas(tuple(coords))


@app.get("/api/buildings/nearest")
async def nearest_building(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(default=1.0),
):
    """Lähin rakennus OSM:sta – palauttaa etäisyyden metreinä ja GeoJSON."""
    delta = radius_km / 111.0
    bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
    data = await get_buildings(bbox)
    dist = nearest_point_distance_m(lat, lon, data)
    return {
        "nearest_building_m": round(dist) if dist >= 0 else None,
        "buildings_found": len(data.get("features", [])),
        "geojson": data,
    }


@app.get("/api/natura")
async def natura(bbox: str = Query(default="22.5,60.6,23.0,60.9")):
    """Natura 2000 -alueet SYKE:ltä."""
    try:
        coords = [float(x) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox: minlon,minlat,maxlon,maxlat")
    return await get_natura_areas(tuple(coords))


@app.get("/api/bess/analysis/{kiinteistotunnus}")
async def bess_analysis(
    kiinteistotunnus: str,
    api_key: Optional[str] = Query(default=None),
    grid_connection: Optional[str] = Query(default=None),
):
    """
    Kokonaisvaltainen BESS-soveltuvuusanalyysi – hakee kaikki datat rinnakkain.
    Pisteytyskriteerit:
      Jakeluverkon etäisyys: <500 m = 30p, 500 m–2 km = 20p, >2 km = 5p
      Ei pohjavettä:         20p  (ei dataa = 0p)
      Ei Natura:             20p
      Ei asemakaavaa:        15p
      Asutus >300 m:         15p
    """
    return await _run_analysis(
        kiinteistotunnus,
        api_key=api_key or MML_API_KEY,
        grid_connection=grid_connection or "",
    )


@app.get("/api/map/static/{kiinteistotunnus}")
async def static_map_image(
    kiinteistotunnus: str,
    zoom: int = Query(default=16),
):
    """
    Generoi staattinen karttakuva kiinteistöstä (staticmap + OSM-tiilet).
    Piirtää kiinteistörajan punaisena viivana. Palauttaa base64 PNG.
    """
    try:
        prop = await get_property_boundaries(kiinteistotunnus)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Property lookup failed: {exc}")

    center_lat, center_lon = _centroid(prop)

    try:
        png_bytes = await asyncio.to_thread(
            _render_static_map, prop, center_lat, center_lon, zoom
        )
        return {"image_b64": base64.b64encode(png_bytes).decode()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Map image generation failed: {exc}")


@app.post("/api/report/generate")
async def generate_report(req: ReportRequest):
    """PDF-raportti. Käyttää frontendilta tullutta analyysiä – ei kaksoisajoa."""
    kt = req.kiinteistotunnus

    # Käytä frontendilta tullutta geometriaa; hae WFS:stä vain jos puuttuu
    prop = req.property_geojson
    if not prop:
        try:
            prop = await get_property_boundaries(kt, api_key=MML_API_KEY)
        except Exception:
            prop = None

    # Generoi karttakuva samasta geometriasta
    map_image_b64 = req.map_image
    if map_image_b64 is None and prop:
        center_lat, center_lon = _centroid(prop)
        try:
            png_bytes = await asyncio.to_thread(
                _render_static_map, prop, center_lat, center_lon, 16
            )
            map_image_b64 = base64.b64encode(png_bytes).decode()
        except Exception:
            pass

    # Analyysi: käytä frontendilta tullutta (UI:ssa näytetty arvo = PDF:n arvo)
    analysis = req.analysis_result
    if not analysis:
        try:
            analysis = await _run_analysis(
                kt, api_key=MML_API_KEY, prop=prop,
                grid_connection=req.grid_connection,
            )
        except HTTPException:
            analysis = {}

    # Sovella manuaaliset syötteet — päivittää pisteytyksen
    has_manual = any([
        req.manual_kaavoitus, req.manual_tulvavaara,
        req.manual_maapera, req.manual_pinta_ala_ha,
    ])
    if has_manual:
        analysis = _apply_manual_overrides(analysis, req)

    prop_meta = {
        "area_ha":  analysis.get("area_ha"),
        "kuntanimi": analysis.get("kuntanimi", "–"),
        "kylanimi":  analysis.get("kylanimi", "–"),
    }

    pdf_bytes = generate_bess_report(
        kiinteistotunnus=kt,
        property_data=prop_meta,
        analysis_data=analysis,
        map_image_b64=map_image_b64,
        project_owner=req.project_owner,
        project_name=req.project_name,
        power_mw=req.power_mw,
        grid_connection=req.grid_connection,
        market=req.market,
        lang=req.lang or "FI",
    )
    filename = f"BESS_raportti_{kt.replace('-', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Permit AI ────────────────────────────────────────────────────────────────

@app.post("/api/generate-application")
@limiter.limit("5/hour")
async def generate_application_endpoint(request: Request, req: ApplicationRequest):
    """Käynnistä lupahakemus-PDF:n generointi taustasäikeessä. Palauttaa job_id heti (202)."""
    allowed = {"BESS", "tuulivoima_maa", "tuulivoima_meri", "aurinkovoima", "SMR",
               "smr_bess", "vesivoima", "hybridi",
               "asuinrakennus", "teollisuus", "maatalous", "liikerakennus", "muu",
               "ymparistolupa", "datakeskus",
               "smr_se", "smr_no", "smr_da", "smr_de",
               "egs", "offshore_wind"}
    if req.hanketyyppi not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"hanketyyppi oltava: {', '.join(sorted(allowed))}")

    # Phase-Lock: tarkista onko edellinen vaihe suoritettu
    if _PHASE_LOCK_OK and req.session_id and req.hankkeen_vaihe:
        ok, err = _check_phase(req.session_id, req.hanketyyppi, req.hankkeen_vaihe)
        if not ok:
            raise HTTPException(status_code=400, detail=err)

    inp = ApplicationInput(
        hanketyyppi                   = req.hanketyyppi,
        kiinteistotunnus              = req.kiinteistotunnus,
        teho_mw                       = req.teho_mw or 0.0,
        kapasiteetti_mwh              = req.kapasiteetti_mwh or 0.0,
        y_tunnus                      = req.y_tunnus or "",
        osoite                        = req.osoite or "",
        kunta                         = req.kunta,
        hakija                        = req.hakija,
        sijainti_ymparistovaikutukset = req.sijainti_ymparistovaikutukset or "",
        hankkeen_vaihe                = req.hankkeen_vaihe or "",
        kohdeviranomainen             = req.kohdeviranomainen or "",
        lang                          = req.lang or "FI",
        country                       = req.country or "FI",
        ifc_floor_area                = req.ifc_floor_area or 0.0,
        ifc_building_height           = req.ifc_building_height or 0.0,
        ifc_fire_rating               = req.ifc_fire_rating or "",
        ifc_materials                 = req.ifc_materials or "",
        ifc_storeys                   = req.ifc_storeys or 0,
        ifc_compliance_flags          = req.ifc_compliance_flags or "",
    )

    job_id = uuid.uuid4().hex[:10]
    _proofread_store[job_id] = {
        "status": "pending", "pdf_bytes": None, "error": None,
        "lang":        req.lang or "FI",
        "hanketyyppi": req.hanketyyppi or "doc",
        "kunta":       req.kunta or "hanke",
    }

    _client_ip = get_remote_address(request)
    _log_usage(_client_ip, req.hanketyyppi, req.country or "FI",
               req.hankkeen_vaihe or "", job_id, "started")

    def _bg_generate():
        try:
            _proofread_store[job_id]["status"] = "running"
            draft_bytes, sections, sources = generate_application_draft(inp)
            _proofread_store[job_id]["debug_sections"] = {k: len(v) for k, v in sections.items() if isinstance(v, str)}
            pdf = apply_proofread_to_pdf(inp, sections, sources)
            _proofread_store[job_id]["pdf_bytes"] = pdf
            _proofread_store[job_id]["status"] = "done"
            _log_usage(_client_ip, req.hanketyyppi, req.country or "FI",
                       req.hankkeen_vaihe or "", job_id, "done")
        except InsufficientSourcesError as exc:
            _proofread_store[job_id]["status"] = "insufficient_sources"
            _proofread_store[job_id]["error"] = str(exc)
            _proofread_store[job_id]["chunks_found"] = exc.chunks_found
            _proofread_store[job_id]["avg_relevance"] = round(exc.avg_relevance, 2)
            _log_usage(_client_ip, req.hanketyyppi, req.country or "FI",
                       req.hankkeen_vaihe or "", job_id, f"RAG_FAIL:chunks={exc.chunks_found}")
        except Exception as exc:
            _proofread_store[job_id]["status"] = "error"
            _proofread_store[job_id]["error"] = str(exc)
            _log_usage(_client_ip, req.hanketyyppi, req.country or "FI",
                       req.hankkeen_vaihe or "", job_id, f"error:{str(exc)[:60]}")

    Thread(target=_bg_generate, daemon=True).start()

    return Response(
        content    = json.dumps({"job_id": job_id}),
        status_code = 202,
        media_type = "application/json",
        headers    = {
            "X-Job-Id":                      job_id,
            "Access-Control-Expose-Headers": "X-Job-Id",
        },
    )


@app.get("/api/proofread/{job_id}")
async def proofread_status(job_id: str):
    """Oikolukutehtävän tila: pending | running | done | error | insufficient_sources."""
    job = _proofread_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if job["status"] == "insufficient_sources":
        raise HTTPException(
            status_code=422,
            detail={
                "error":         "insufficient_sources",
                "message":       (
                    "Riittämätön lähdeaineisto — RAG-tietokanta ei palauttanut riittävästi "
                    "relevantteja lähteitä luotettavan lupahakemusluonnoksen tuottamiseen. "
                    "Kokeile eri hanketyyppiä tai ota yhteyttä info@ncenergy.fi."
                ),
                "chunks_found":  job.get("chunks_found", 0),
                "avg_relevance": job.get("avg_relevance", 0.0),
            },
        )
    return {"status": job["status"], "error": job.get("error"), "debug_sections": job.get("debug_sections")}


_FILE_PREFIX = {"FI": "hakemus", "EN": "application", "SE": "ansökan",
                "DA": "ansøgning", "NO": "søknad", "PL": "wniosek"}


def _fn(s: str) -> str:
    """Sanitize a string for use in Content-Disposition filename (ASCII-safe)."""
    nfkd = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-zA-Z0-9]", "_", nfkd.encode("ascii", "ignore").decode("ascii"))


@app.get("/api/proofread/{job_id}/download")
async def proofread_download(job_id: str):
    """Lataa oikoluvun jälkeinen PDF."""
    job = _proofread_store.get(job_id)
    if job is None or job["status"] != "done" or not job["pdf_bytes"]:
        raise HTTPException(status_code=404, detail="PDF not ready yet")
    prefix  = _FILE_PREFIX.get(job.get("lang", "FI"), "hakemus")
    _kt     = _fn(job.get("hanketyyppi", "doc"))
    _kunta  = _fn(job.get("kunta", "hanke"))
    return Response(
        content    = job["pdf_bytes"],
        media_type = "application/pdf",
        headers    = {"Content-Disposition": f'attachment; filename="{prefix}_{_kt}_{_kunta}.pdf"'},
    )


@app.post("/api/permit-ai")
@limiter.limit("50/hour")
async def permit_ai(request: Request, req: PermitAIRequest):
    """RAG-pohjainen lupaprosessikysely. Hakee Fingrid/Pelastusopisto/Tukes-dokumenteista."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = await asyncio.to_thread(
            query_permit_ai, req.question, req.n_results
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Permit AI -virhe: {exc}")


# ── Site Optimizer ───────────────────────────────────────────────────────────

def _lcg(s: int) -> int:
    return (1664525 * s + 1013904223) % (2 ** 32)


def _rng01(seed: int, offset: int = 0) -> float:
    s = seed
    for _ in range(offset + 1):
        s = _lcg(s)
    return s / (2 ** 32)


@app.post("/api/optimize-bess")
@limiter.limit("20/hour")
async def optimize_sites(request: Request, req: OptimizeRequest):
    """
    Sijaintioptimointityökalu — pisteyttää kandidaattisijainteja hanketyypeittäin.
    Hanketyypit: bess, tuulivoima, aurinkovoima, smr
    """
    if not _OPTIMIZER_OK:
        raise HTTPException(status_code=501, detail="optimizer.py not found")

    _allowed = {"bess", "tuulivoima", "aurinkovoima", "smr"}
    if req.project_type not in _allowed:
        raise HTTPException(status_code=400,
                            detail=f"project_type oltava: {', '.join(sorted(_allowed))}")
    if len(req.bbox) != 4:
        raise HTTPException(status_code=400,
                            detail="bbox: [lat_min, lon_min, lat_max, lon_max]")

    lat_min, lon_min, lat_max, lon_max = req.bbox

    # Suomen maa-alueen karkea bounding box (manner + saaret, ei Itämeri/ulkomaat)
    _FI_LAT_MIN, _FI_LAT_MAX = 59.5, 70.1
    _FI_LON_MIN, _FI_LON_MAX = 19.5, 31.6

    # Tarkista että bbox on Suomen sisällä — hylkää jos täysin ulkopuolella
    if (lat_max < _FI_LAT_MIN or lat_min > _FI_LAT_MAX or
            lon_max < _FI_LON_MIN or lon_min > _FI_LON_MAX):
        raise HTTPException(
            status_code=400,
            detail=(
                "bbox on Suomen maa-alueen ulkopuolella. "
                f"Sallittu alue: lat {_FI_LAT_MIN}–{_FI_LAT_MAX}, "
                f"lon {_FI_LON_MIN}–{_FI_LON_MAX}."
            ),
        )

    # Leikkaa bbox Suomen rajoihin (jos käyttäjä antoi osittain ulkopuolisen alueen)
    lat_min = max(lat_min, _FI_LAT_MIN)
    lat_max = min(lat_max, _FI_LAT_MAX)
    lon_min = max(lon_min, _FI_LON_MIN)
    lon_max = min(lon_max, _FI_LON_MAX)

    def _inside_finland(lat: float, lon: float) -> bool:
        """Karkea maa-alue-check Suomelle. Hylkää ilmiselvästi meren tai ulkomaan pisteet."""
        if not (_FI_LAT_MIN <= lat <= _FI_LAT_MAX and _FI_LON_MIN <= lon <= _FI_LON_MAX):
            return False
        # Poista Suomenlahden eteläinen meri-alue (Viro/Latvia): lat<59.8 + lon<27
        if lat < 59.8 and lon < 27.0:
            return False
        # Poista Ruotsin puoli (Merenkurkku + Pohjanlahti): lon<20.5 ja lat<65
        if lon < 20.5 and lat < 65.0:
            return False
        return True

    # Generoi 16 kandidaattisijaintia 4×4-gridillä bbox:n sisältä
    _rows, _cols = 4, 4
    sites: list = []
    skipped_sea: list = []
    _col_labels = "ABCDE"
    for i in range(_rows):
        for j in range(_cols):
            lat = lat_min + (lat_max - lat_min) * (i + 0.5) / _rows
            lon = lon_min + (lon_max - lon_min) * (j + 0.5) / _cols
            if not _inside_finland(lat, lon):
                skipped_sea.append(f"{_col_labels[j]}{i + 1}")
                continue
            _seed = int(abs(lat * 1e4)) * 99991 + int(abs(lon * 1e4)) * 31337
            r = lambda off: _rng01(_seed, off)
            sites.append(EnergySite(
                site_id      = f"{_col_labels[j]}{i + 1}",
                lat          = round(lat, 5),
                lon          = round(lon, 5),
                solar_irradiance    = 700 + r(1) * 400,
                wind_resource       = 4.0 + r(2) * 5.0,
                grid_distance_km    = 0.5 + r(3) * 44.5,
                land_area_ha        = max(req.min_area_ha, 2 + r(4) * 58),
                zoning_score        = 0.15 + r(5) * 0.85,
                protected_area_score= 0.10 + r(6) * 0.90,
                water_access_score  = 0.30 + r(7) * 0.70,
                land_cost_eur_ha    = 5000 + r(8) * 30000,
            ))

    if not sites:
        raise HTTPException(
            status_code=400,
            detail="Kaikki kandidaattisijannit osuivat meri- tai ulkomaa-alueelle. Tarkista bbox.",
        )

    optimizer = NCEOptimizer(req.project_type)
    result    = optimizer.optimize(sites)

    top5 = [
        {
            "site_id":              site.site_id,
            "lat":                  site.lat,
            "lon":                  site.lon,
            "score":                score,
            "score_pct":            f"{score:.0%}",
            "grid_distance_km":     round(site.grid_distance_km, 1),
            "zoning_score":         round(site.zoning_score, 2),
            "protected_area_score": round(site.protected_area_score, 2),
            "solar_irradiance":     round(site.solar_irradiance),
            "wind_resource":        round(site.wind_resource, 1),
            "land_area_ha":         round(site.land_area_ha, 1),
        }
        for site, score in zip(result.ranked_sites[:5], result.scores[:5])
    ]

    resp: dict = {
        "results":          top5,
        "optimizer_used":   result.optimizer_used,
        "project_type":     result.project_type,
        "total_candidates": len(sites),
    }
    if skipped_sea:
        resp["skipped_outside_finland"] = skipped_sea
    return resp


# ── Sisäinen analyysilogiikka ─────────────────────────────────────────────────

async def _run_analysis(
    kiinteistotunnus: str,
    api_key: str = "",
    prop: Optional[dict] = None,
    grid_connection: str = "",
) -> dict:
    """
    Kokonaisanalyysi. prop=None → haetaan WFS:stä kerran.
    grid_connection: ohjaa verkkoetäisyyden suodatusta
      "Fingrid 110" → vain ≥100 kV johdot
      "Fingrid 400" → vain ≥380 kV johdot
      muu (jakeluverkko) → ≤25 kV tai tagittomat johdot
    """
    key = api_key or MML_API_KEY
    muni_code = kiinteistotunnus.split("-")[0].zfill(3)

    if prop is None:
        try:
            prop = await get_property_boundaries(kiinteistotunnus, api_key=key)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Property lookup failed: {exc}")

    kuntanimi  = _prop_kuntanimi(prop)
    kylanimi   = _prop_kylanimi(prop)
    kuntanimi_gen = genitive(kuntanimi) if kuntanimi and not kuntanimi.startswith("Kunta ") else kuntanimi
    pelastuslaitos_name = get_pelastuslaitos(muni_code)
    ely_center_name     = get_ely(muni_code)

    center_lat, center_lon = _centroid(prop)
    area_m2 = _area(prop)
    wide_bbox   = (center_lon - 0.20, center_lat - 0.15, center_lon + 0.20, center_lat + 0.15)
    narrow_bbox = (center_lon - 0.005, center_lat - 0.005, center_lon + 0.005, center_lat + 0.005)
    bldg_delta  = 0.010
    bldg_bbox   = (center_lon - bldg_delta, center_lat - bldg_delta,
                   center_lon + bldg_delta, center_lat + bldg_delta)

    substation_wide = (center_lon - 0.30, center_lat - 0.20, center_lon + 0.30, center_lat + 0.20)

    (grid_data, natura_data, gw_data, bldg_data,
     zoning_data, heritage_data, highway_data,
     flood_data, substation_data) = await asyncio.gather(
        get_transmission_lines(wide_bbox),
        get_natura_areas(wide_bbox),
        get_groundwater_areas(wide_bbox),
        get_buildings(bldg_bbox),
        get_zoning_info(wide_bbox, api_key=key),
        get_heritage_sites(wide_bbox),
        get_highways(wide_bbox),
        get_flood_risk(wide_bbox),
        get_substations(substation_wide),
        return_exceptions=True,
    )

    grid_data       = grid_data       if isinstance(grid_data,       dict) else {"features": []}
    natura_data     = natura_data     if isinstance(natura_data,     dict) else {"features": []}
    gw_data         = gw_data         if isinstance(gw_data,         dict) else {"features": [], "unavailable": True}
    bldg_data       = bldg_data       if isinstance(bldg_data,       dict) else {"features": []}
    zoning_data     = zoning_data     if isinstance(zoning_data,     dict) else {"features": [], "unavailable": True}
    heritage_data   = heritage_data   if isinstance(heritage_data,   dict) else {"features": [], "unavailable": True}
    highway_data    = highway_data    if isinstance(highway_data,    dict) else {"features": []}
    flood_data      = flood_data      if isinstance(flood_data,      dict) else {"flood_overlap": False, "unavailable": True}
    substation_data = substation_data if isinstance(substation_data, dict) else {"features": []}

    # Etäisyyssuodatus verkkotypin mukaan
    if "Fingrid 400" in grid_connection:
        nearest_grid_m = nearest_line_distance_m(center_lat, center_lon, grid_data, min_voltage_kv=380)
    elif "Fingrid 110" in grid_connection or "110 kV" in grid_connection:
        nearest_grid_m = nearest_line_distance_m(center_lat, center_lon, grid_data, min_voltage_kv=100)
    else:
        # Jakeluverkko: ≤25 kV tai tagittomat (voltage_kv=0)
        nearest_grid_m = nearest_line_distance_m(center_lat, center_lon, grid_data, max_voltage_kv=25)

    natura_overlap      = len(natura_data.get("features", [])) > 0
    gw_overlap          = len(gw_data.get("features", [])) > 0
    gw_unavailable      = gw_data.get("unavailable", False)
    gw_class            = _worst_gw_class(gw_data.get("features", [])) if gw_overlap else ""
    heritage_overlap    = len(heritage_data.get("features", [])) > 0
    heritage_unavailable= heritage_data.get("unavailable", False)
    heritage_source     = heritage_data.get("source", "none")
    heritage_note       = heritage_data.get("note", "")
    zoning_unavailable  = zoning_data.get("unavailable", False)
    nearest_bldg_m      = nearest_point_distance_m(center_lat, center_lon, bldg_data)
    nearest_road_m      = nearest_line_distance_m(center_lat, center_lon, highway_data)
    road_protection_ok  = nearest_road_m < 0 or nearest_road_m >= 20.0
    nearest_road_name   = _nearest_road_name(center_lat, center_lon, highway_data)
    if zoning_data.get("unavailable"):
        osm_zone = await infer_zoning_from_osm(center_lat, center_lon)
        zoning_data["osm_inference"] = osm_zone
    zoning_status, zoning_ok = _eval_zoning(zoning_data)
    land_use            = _dominant_land_use(await get_land_use(narrow_bbox, api_key=key))
    grid_type           = _best_line_type(center_lat, center_lon, grid_data)
    powerline_buffer_ok = nearest_grid_m < 0 or nearest_grid_m >= 25.0

    # Maaperä (GTK)
    soil_data           = await get_soil_type(center_lat, center_lon)
    maaperalaaji        = soil_data.get("maaperalaaji", "Ei tiedossa")
    soil_score_pts      = soil_data.get("score_pts")   # None = N/A

    # Tulvavaara (SYKE)
    flood_overlap       = flood_data.get("flood_overlap", False)
    flood_unavailable   = flood_data.get("unavailable", False)

    # Sähköasema (lähin, OSM)
    sub_info            = nearest_substation_info(center_lat, center_lon, substation_data)
    nearest_substation_m = sub_info.get("distance_m")
    nearest_substation_name = sub_info.get("name")

    # Lupapiste-URL
    lupapiste_url = f"https://www.lupapiste.fi/?municipality={muni_code}"

    scores = _score(
        nearest_grid_m=nearest_grid_m,
        gw_overlap=gw_overlap,
        gw_unavailable=gw_unavailable,
        gw_class=gw_class,
        natura_overlap=natura_overlap,
        zoning_ok=zoning_ok,
        zoning_unavailable=zoning_unavailable,
        nearest_bldg_m=nearest_bldg_m,
        heritage_overlap=heritage_overlap,
        heritage_unavailable=heritage_unavailable,
        road_protection_ok=road_protection_ok,
        flood_overlap=flood_overlap,
        flood_unavailable=flood_unavailable,
        soil_score_pts=soil_score_pts,
    )

    # Lupaprosessianalyysi generoidaan PDF:ssä datapohjaisen templaten kautta —
    # Claude API:ta ei kutsuta enää tässä vaiheessa.
    ai_result = {}

    return {
        "kiinteistotunnus": kiinteistotunnus,
        "kuntanimi":        kuntanimi,
        "kuntanimi_gen":    kuntanimi_gen,
        "kylanimi":         kylanimi,
        "muni_code":        muni_code,
        "pelastuslaitos":   pelastuslaitos_name,
        "ely_center":       ely_center_name,
        "lupapiste_url":    lupapiste_url,
        "center_lat": round(center_lat, 6),
        "center_lon": round(center_lon, 6),
        "area_m2": area_m2,
        "area_ha": round(area_m2 / 10_000, 2) if area_m2 else None,
        "nearest_grid_m": round(nearest_grid_m) if nearest_grid_m >= 0 else None,
        "powerline_buffer_ok": powerline_buffer_ok,
        "grid_status": _grid_status(nearest_grid_m),
        "grid_type": grid_type,
        "nearest_substation_m":    nearest_substation_m,
        "nearest_substation_name": nearest_substation_name,
        "groundwater_overlap": gw_overlap,
        "groundwater_unavailable": gw_unavailable,
        "groundwater_class": gw_class,
        "natura_overlap": natura_overlap,
        "heritage_overlap": heritage_overlap,
        "heritage_unavailable": heritage_unavailable,
        "heritage_source": heritage_source,
        "heritage_note": heritage_note,
        "nearest_road_m": round(nearest_road_m) if nearest_road_m >= 0 else None,
        "nearest_road_name": nearest_road_name,
        "road_protection_ok": road_protection_ok,
        "zoning_status": zoning_status,
        "zoning_unavailable": zoning_unavailable,
        "zoning_ok": zoning_ok,
        "nearest_building_m": round(nearest_bldg_m) if nearest_bldg_m >= 0 else None,
        "land_use": land_use,
        "maaperalaaji":       maaperalaaji,
        "maaperalaaji_source": soil_data.get("source", "unavailable"),
        "flood_overlap":      flood_overlap,
        "flood_unavailable":  flood_unavailable,
        "ai_strategy":        ai_result.get("strategy"),
        "ai_strategy_error":  ai_result.get("error"),
        "bess_score":           scores["total"],
        "score_grid":           scores["grid"],
        "score_groundwater":    scores["gw"],
        "score_natura":         scores["natura"],
        "score_zoning":         scores["zoning"],
        "score_settlement":     scores["settlement"],
        "score_heritage":       scores["heritage"],
        "score_road":           scores["road"],
        "score_flood":          scores["flood"],
        "score_soil":           scores["soil"],
    }


# ── Apufunktiot ───────────────────────────────────────────────────────────────

_SOIL_MAP: dict[str, tuple[str, int]] = {
    "kallio":  ("Kallio",  5),
    "moreeni": ("Moreeni", 4),
    "hiekka":  ("Hiekka",  3),
    "savi":    ("Savi",    1),
    "turve":   ("Turve",   0),
}


def _apply_manual_overrides(analysis: dict, req: "ReportRequest") -> dict:
    """
    Soveltaa manuaaliset syötteet analyysidict:iin ja laskee pisteytyksen uudelleen.
    Käytetään vain PDF-raportin generoinnissa.
    """
    a = dict(analysis)

    if req.manual_kaavoitus and req.manual_kaavoitus not in ("", "ei_tietoa"):
        a["zoning_unavailable"] = False
        if req.manual_kaavoitus == "asemakaava":
            a["zoning_ok"]     = False
            a["zoning_status"] = "Asemakaava (manuaalinen syöte)"
        elif req.manual_kaavoitus == "yleiskaava":
            a["zoning_ok"]     = True
            a["zoning_status"] = "Yleiskaava (manuaalinen syöte)"
        else:  # ei_kaavaa
            a["zoning_ok"]     = True
            a["zoning_status"] = "Ei kaavaa (manuaalinen syöte)"
        a["manual_kaavoitus"] = req.manual_kaavoitus

    if req.manual_tulvavaara and req.manual_tulvavaara not in ("", "ei_tietoa"):
        a["flood_unavailable"] = False
        a["flood_overlap"]     = (req.manual_tulvavaara == "kyllä")
        a["manual_tulvavaara"] = req.manual_tulvavaara

    soil_score_override: Optional[int] = None
    if req.manual_maapera and req.manual_maapera not in ("", "ei_tietoa") and req.manual_maapera in _SOIL_MAP:
        nimi, pts = _SOIL_MAP[req.manual_maapera]
        a["maaperalaaji"]        = nimi
        a["maaperalaaji_source"] = "manual"
        a["manual_maapera"]      = req.manual_maapera
        soil_score_override      = pts

    if req.manual_pinta_ala_ha is not None and req.manual_pinta_ala_ha > 0:
        a["area_ha"]             = req.manual_pinta_ala_ha
        a["manual_pinta_ala_ha"] = req.manual_pinta_ala_ha

    # Pisteytys uudelleen
    soil_pts = soil_score_override if soil_score_override is not None else a.get("score_soil")
    scores = _score(
        nearest_grid_m   = a.get("nearest_grid_m") if a.get("nearest_grid_m") is not None else -1,
        gw_overlap       = a.get("groundwater_overlap", False),
        gw_unavailable   = a.get("groundwater_unavailable", False),
        gw_class         = a.get("groundwater_class", ""),
        natura_overlap   = a.get("natura_overlap", False),
        zoning_ok        = a.get("zoning_ok", True),
        zoning_unavailable = a.get("zoning_unavailable", False),
        nearest_bldg_m   = a.get("nearest_building_m") if a.get("nearest_building_m") is not None else -1,
        heritage_overlap = a.get("heritage_overlap", False),
        heritage_unavailable = a.get("heritage_unavailable", False),
        road_protection_ok = a.get("road_protection_ok", True),
        flood_overlap    = a.get("flood_overlap", False),
        flood_unavailable = a.get("flood_unavailable", False),
        soil_score_pts   = soil_pts,
    )
    a.update({
        "bess_score":        scores["total"],
        "score_grid":        scores["grid"],
        "score_groundwater": scores["gw"],
        "score_natura":      scores["natura"],
        "score_zoning":      scores["zoning"],
        "score_settlement":  scores["settlement"],
        "score_heritage":    scores["heritage"],
        "score_road":        scores["road"],
        "score_flood":       scores["flood"],
        "score_soil":        scores["soil"],
    })
    return a


def _centroid(geojson: dict) -> tuple[float, float]:
    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords:
            continue
        ring = coords[0] if geom.get("type") == "Polygon" else (coords[0][0] if coords else None)
        if ring:
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            return sum(lats) / len(lats), sum(lons) / len(lons)
    return 60.6833, 22.5333   # Pöytyä default


def _area(geojson: dict) -> Optional[float]:
    for feat in geojson.get("features", []):
        v = (feat.get("properties") or {}).get("pinta_ala")
        if v:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


def _eval_zoning(zoning_data: dict) -> tuple[str, bool]:
    if not zoning_data.get("unavailable"):
        feats = zoning_data.get("features", [])
        if not feats:
            return "Ei asemakaavaa — tarkistettu MML WFS", True
        types = [(f.get("properties") or {}).get("kaavatyyppi", "") for f in feats]
        if any("asemakaava" in t.lower() for t in types):
            return "Asemakaava-alue", False
        return "Yleiskaava / maakuntakaava", True
    # MML ei saatavilla — käytetään OSM-päättelyä
    osm = zoning_data.get("osm_inference", {})
    inferred = osm.get("inferred", "unknown")
    if inferred == "asemakaava":
        return "Todennäköisesti asemakaava-alue (OSM-päättely)", False
    if inferred == "rural":
        return "Haja-asutusalue (OSM-päättely — tarkista MML WFS)", True
    return "Ei saatavilla (MML API-avain puuttuu)", True


def _nearest_road_name(lat: float, lon: float, highway_geojson: dict) -> str:
    from fingrid_api import _haversine_m, _point_to_segment_m
    best_name, best_dist = "", float("inf")
    for feat in highway_geojson.get("features", []):
        coords = feat.get("geometry", {}).get("coordinates", [])
        p = feat.get("properties") or {}
        name = p.get("name") or p.get("ref") or p.get("highway") or ""
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            d = _point_to_segment_m(lat, lon, lat1, lon1, lat2, lon2)
            if d < best_dist:
                best_dist = d
                best_name = name
    return best_name


def _parse_gw_class(luokka_text: str) -> str:
    """'Vedenhankintaa... (1E)' → '1E'.  Palauttaa '' jos ei tunnisteta."""
    m = re.search(r'\(([12E]+E?)\)\s*$', luokka_text.strip())
    return m.group(1) if m else ""


def _worst_gw_class(features: list) -> str:
    """Pahin pohjavesiluokka (1 > 1E > 2E > 2 > E) annetuista piirtein."""
    classes = {_parse_gw_class((f.get("properties") or {}).get("luokka", ""))
               for f in features} - {""}
    for prio in ("1", "1E", "2E", "2", "E"):
        if prio in classes:
            return prio
    return ""


_LANDUSE_FI_MAIN: dict[str, str] = {
    "farmland":    "Peltoalue",
    "forest":      "Metsäalue",
    "meadow":      "Niitty / laidun",
    "residential": "Asuinalue",
    "commercial":  "Kaupallinen alue",
    "industrial":  "Teollisuusalue",
    "retail":      "Vähittäiskauppa",
}


def _dominant_land_use(landuse_data: dict) -> str:
    feats = landuse_data.get("features", [])
    if not feats:
        return "Maatalousmaa / metsä (oletus)"
    p = (feats[0].get("properties") or {})
    raw = p.get("kohdeluokka") or p.get("luokka") or ""
    # kohdeluokka on jo suomeksi jos tuli OSM-fallbackista (mml_api käänsi)
    return raw or _LANDUSE_FI_MAIN.get(p.get("kohdeluokka_osm", ""), "Maatalousmaa / metsä")


def _prop_kuntanimi(prop: dict) -> str:
    for feat in (prop or {}).get("features", []):
        v = (feat.get("properties") or {}).get("kuntanimi", "")
        if v:
            return v
    return "–"


def _prop_kylanimi(prop: dict) -> str:
    for feat in (prop or {}).get("features", []):
        v = (feat.get("properties") or {}).get("kylanimi", "")
        if v and v != "–":
            return v
    return "–"


def _best_line_type(lat: float, lon: float, grid_geojson: dict) -> str:
    """Palauttaa lähimmän power-elementin tyypin (johto tai pylväs)."""
    from fingrid_api import _haversine_m, _extract_line_coords, _point_to_segment_m
    best_type, best_dist = "–", float("inf")
    for feat in grid_geojson.get("features", []):
        props = feat.get("properties") or {}
        geom  = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        if gtype == "Point":
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                d = _haversine_m(lat, lon, coords[1], coords[0])
                if d < best_dist:
                    best_dist = d
                    best_type = props.get("line_type", "–")
        else:
            for seg in _extract_line_coords(geom):
                for i in range(len(seg) - 1):
                    d = _point_to_segment_m(lat, lon, seg[i][1], seg[i][0], seg[i+1][1], seg[i+1][0])
                    if d < best_dist:
                        best_dist = d
                        best_type = props.get("line_type", "–")
    return best_type


def _grid_status(nearest_grid_m: float) -> str:
    if nearest_grid_m < 0:
        return "Ei dataa"
    if nearest_grid_m < 1_000:
        return "Erinomainen ✓"
    if nearest_grid_m < 2_000:
        return "Hyvä ✓"
    return "Tarkista — pyydä liityntätarjous Carunalta"


def _score(
    nearest_grid_m: float,
    gw_overlap: bool,
    gw_unavailable: bool,
    gw_class: str = "",
    natura_overlap: bool = False,
    zoning_ok: bool = True,
    zoning_unavailable: bool = False,
    nearest_bldg_m: float = -1,
    heritage_overlap: bool = False,
    heritage_unavailable: bool = False,
    road_protection_ok: bool = True,
    flood_overlap: bool = False,
    flood_unavailable: bool = True,
    soil_score_pts: Optional[int] = None,
) -> dict:
    """
    Pisteytys (max 110p, normalisoidaan 100:aan käytettävissä olevilla kriteereillä):
      Verkkoliityntä:   30p  (<1km=30, 1-2km=20, >2km=5)
      Pohjavesiluokka:  20p  (ei=20, luokka2/E=8, luokka1=0, N/A pois)
      Natura 2000:      15p  (ei=15, on=0)
      Ei asemakaavaa:   10p  (maaseutu=10, asemakaava=3, N/A pois)
      Asutus >300m:     10p  (>300m=10, 150-300m=5, <150m=0)
      Ei muinaismuistoja:10p (ei=10, on=0, N/A pois)
      Tiesuoja-alue OK:  5p  (ok=5, ei ok=0)
      Tulvavaara:        5p  (tulossa – ei dataa = N/A, pois indeksistä)
      Maaperä:           5p  (tulossa – ei dataa = N/A, pois indeksistä)
    """
    # Verkkoliityntä (30p)
    if nearest_grid_m < 0:
        grid = 15
    elif nearest_grid_m < 1_000:
        grid = 30
    elif nearest_grid_m < 2_000:
        grid = 20
    else:
        grid = 5

    # Pohjavesiluokka (20p) — N/A kun SYKE offline
    if gw_unavailable:
        gw = None
    elif not gw_overlap:
        gw = 20
    elif gw_class in ("1", "1E"):
        gw = 0
    else:
        gw = 8

    # Natura (15p)
    natura = 0 if natura_overlap else 15

    # Kaavoitus (10p) — N/A kun MML-avain puuttuu
    if zoning_unavailable:
        zoning = None
    else:
        zoning = 10 if zoning_ok else 3

    # Asutus (10p)
    if nearest_bldg_m < 0:
        settlement = 5
    elif nearest_bldg_m > 300:
        settlement = 10
    elif nearest_bldg_m > 150:
        settlement = 5
    else:
        settlement = 0

    # Muinaismuistot (10p) — N/A kun kumpikaan rajapinta ei vastaa
    if heritage_unavailable:
        heritage = None
    else:
        heritage = 0 if heritage_overlap else 10

    # Tiesuoja-alue (5p)
    road = 5 if road_protection_ok else 0

    # Tulvavaara (5p) — N/A kun SYKE offline
    if flood_unavailable:
        flood = None
    else:
        flood = 0 if flood_overlap else 5

    # Maaperä (5p) — N/A kun GTK ei saatavilla
    soil = soil_score_pts  # jo laskettu gtk_api:ssa (None = N/A)

    # Normalisointi: lasketaan vain niiden kriteerien yli joille on dataa
    components = [
        (grid,       30),
        (gw,         20),
        (natura,     15),
        (zoning,     10),
        (settlement, 10),
        (heritage,   10),
        (road,        5),
        (flood,       5),
        (soil,        5),
    ]
    achieved = sum(v for v, _ in components if v is not None)
    max_pts  = sum(m for v, m in components if v is not None)
    total    = min(round(achieved / max_pts * 100), 100) if max_pts else 0

    return {
        "total": total,
        "grid": grid, "gw": gw, "natura": natura,
        "zoning": zoning, "settlement": settlement,
        "heritage": heritage, "road": road,
        "flood": flood, "soil": soil,
    }


def _render_static_map(
    prop: dict,
    center_lat: float,
    center_lon: float,
    zoom: int = 16,
) -> bytes:
    """
    Synkroninen apufunktio (ajetaan to_thread:ssa).
    Käyttää staticmap-kirjastoa OSM-tiilejä vasten.
    Piirtää kiinteistörajan punaisena viivana.
    """
    from staticmap import StaticMap, Line, CircleMarker

    m = StaticMap(
        800, 500,
        headers={"User-Agent": "bess-tool/1.0 (BESS planning tool)"},
    )

    # Piirretään kaikki renkaat (Polygon / MultiPolygon)
    for feat in prop.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        rings: list = []
        if gtype == "Polygon":
            rings = geom.get("coordinates", [])
        elif gtype == "MultiPolygon":
            for polygon in geom.get("coordinates", []):
                rings.extend(polygon)
        for ring in rings:
            pts = [[c[0], c[1]] for c in ring if len(c) >= 2]
            if len(pts) >= 2:
                m.add_line(Line(pts, "#e94560", 4))

    # Centroid-piste
    m.add_marker(CircleMarker([center_lon, center_lat], "#e94560", 10))

    img = m.render(zoom=zoom, center=[center_lon, center_lat])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Phase-Lock ───────────────────────────────────────────────────────────────
PHASE_LOCK_ENABLED = os.getenv("PHASE_LOCK_ENABLED", "false").lower() == "true"
try:
    from phase_lock import (
        check_phase_allowed as _check_phase,
        get_phase_status as _get_phase_status,
        unlock_next_phase as _unlock_next_phase,
    )
    _PHASE_LOCK_OK = PHASE_LOCK_ENABLED
except Exception as _pl_err:
    _PHASE_LOCK_OK = False

# ── IFC parser imports (optional — graceful if missing) ───────────────────────
# sys.path already has bess_tool/ root (line 50 above), so permit_ai namespace works
try:
    from ifc_parser import extract_ifc_data as _extract_ifc_data
    from ifc_to_permit import map_to_permit as _map_to_permit
    _IFC_OK = True
except Exception as _ifc_err:
    _IFC_OK = False
    _ifc_err_msg = str(_ifc_err)

_IFC_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@app.post("/api/parse-ifc")
@limiter.limit("20/hour")
async def parse_ifc(
    request: Request,
    file: UploadFile = File(...),
    project_type: str = Query(default="BESS"),
    country: str = Query(default="FI"),
):
    """
    Parse an IFC file and return permit-relevant fields, missing fields,
    and compliance flags. Accepts multipart/form-data, max 50 MB.
    """
    if not _IFC_OK:
        raise HTTPException(status_code=501, detail=f"ifcopenshell ei saatavilla: {_ifc_err_msg}")

    if not file.filename or not file.filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Tiedoston tulee olla .ifc-muodossa")

    content = await file.read()
    if len(content) > _IFC_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Tiedosto liian suuri (max 50 MB)")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    allowed_project_types = {
        "BESS", "AURINKO", "TUULI", "SMR", "DATAKESKUS",
        "SCO2", "VESIVOIMA", "YVA", "VERKKO",
    }
    if project_type not in allowed_project_types:
        raise HTTPException(
            status_code=400,
            detail=f"project_type oltava: {', '.join(sorted(allowed_project_types))}",
        )
    if country not in {"FI", "SE", "DA", "NO", "PL", "DE"}:
        raise HTTPException(status_code=400, detail="country oltava: FI, SE, DA, NO, PL, DE")

    ifc_data = _extract_ifc_data(content)
    permit_map = _map_to_permit(ifc_data, project_type=project_type, country=country)

    # Add confidence score per field to response
    prefilled_with_conf = {
        field: {
            "value": info["value"],
            "confidence": info["confidence"],
        }
        for field, info in permit_map["prefilled_fields"].items()
    }

    return JSONResponse({
        "prefilled_fields":  prefilled_with_conf,
        "missing_fields":    permit_map["missing_fields"],
        "compliance_flags":  permit_map["compliance_flags"],
        "summary":           permit_map["summary"],
        "parse_errors":      ifc_data.get("parse_errors", []),
        "ifc_schema":        ifc_data.get("ifc_schema"),
        "filename":          file.filename,
    })


# ── Phase-Lock endpointit ─────────────────────────────────────────────────────

@app.get("/api/phase-status")
async def phase_status(
    session_id: str = Query(...),
    hanketyyppi: str = Query(...),
):
    """Palauttaa vaiheen tilan sessiolle ja hanketyypille."""
    if not _PHASE_LOCK_OK:
        # Phase lock disabled (demo mode) — all phases open, signal frontend to skip locks
        return JSONResponse({"completed_phase": 0, "next_phase": 1, "phase_lock_disabled": True, "phases": [
            {"name": "esiselvitys",  "phase": 1, "state": "active"},
            {"name": "lupavaihe",    "phase": 2, "state": "active"},
            {"name": "rakentaminen", "phase": 3, "state": "active"},
        ]})
    if not session_id or not hanketyyppi:
        raise HTTPException(status_code=400, detail="session_id ja hanketyyppi vaaditaan")
    return JSONResponse(_get_phase_status(session_id, hanketyyppi))


class CompletePhaseRequest(BaseModel):
    session_id:  str
    hanketyyppi: str
    phase:       int   # 1 | 2 | 3


@app.post("/api/complete-phase")
@limiter.limit("60/hour")
async def complete_phase(request: Request, req: CompletePhaseRequest):
    """Merkitsee vaiheen valmiiksi ja avaa seuraavan."""
    if not _PHASE_LOCK_OK:
        return JSONResponse({"ok": True, "next_phase": req.phase + 1})
    if req.phase not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="phase oltava 1, 2 tai 3")
    status = _unlock_next_phase(req.session_id, req.hanketyyppi, req.phase)
    return JSONResponse({"ok": True, **status})


class IFCApprovalRequest(BaseModel):
    """Insinöörin hyväksymät IFC-kentät + hakemuksen perustiedot."""
    # Hakemuksen perustiedot
    hanketyyppi:       str
    kiinteistotunnus:  str
    teho_mw:           float = 0.0
    kapasiteetti_mwh:  float = 0.0
    kunta:             str
    hakija:            str
    lang:              str = "FI"
    country:           str = "FI"
    hankkeen_vaihe:    str = ""
    kohdeviranomainen: str = ""
    # Hyväksytyt IFC-kentät (insinööri on tarkistanut)
    approved_fields:   dict = {}
    # Audit trail
    reviewer_name:     str
    review_notes:      Optional[str] = None


@app.post("/api/approve-ifc")
@limiter.limit("10/hour")
async def approve_ifc(request: Request, req: IFCApprovalRequest):
    """
    Insinööri lähettää hyväksytyt IFC-kentät → generoi final PDF + audit trail.
    Palauttaa PDF binäärinä (application/pdf).
    """
    import datetime

    approved = req.approved_fields

    # Rakenna ApplicationInput IFC-esitäyttöarvoilla
    inp = ApplicationInput(
        hanketyyppi                   = req.hanketyyppi,
        kiinteistotunnus              = req.kiinteistotunnus,
        teho_mw                       = req.teho_mw,
        kapasiteetti_mwh              = req.kapasiteetti_mwh,
        kunta                         = req.kunta,
        hakija                        = req.hakija,
        lang                          = req.lang,
        country                       = req.country,
        hankkeen_vaihe                = req.hankkeen_vaihe,
        kohdeviranomainen             = req.kohdeviranomainen,
        ifc_floor_area                = float(approved.get("floor_area_total") or 0),
        ifc_building_height           = float(approved.get("building_height") or 0),
        ifc_fire_rating               = str(approved.get("fire_rating_walls") or ""),
        ifc_materials                 = ", ".join(approved.get("materials") or []),
        ifc_storeys                   = len(approved.get("storeys") or []),
        ifc_compliance_flags          = "\n".join(approved.get("compliance_flags") or []),
    )

    # Generoi PDF taustasäikeessä (blocking — approve on harvinainen operaatio)
    loop = asyncio.get_event_loop()
    try:
        draft_bytes, sections, sources = await loop.run_in_executor(
            None, generate_application_draft, inp
        )
        pdf_bytes = await loop.run_in_executor(
            None, lambda: apply_proofread_to_pdf(inp, sections, sources)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    # Audit trail — lisätään PDF:n metatietoihin (ei sisältöön)
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    audit = {
        "timestamp":        timestamp,
        "reviewer_name":    req.reviewer_name,
        "review_notes":     req.review_notes or "",
        "approved_fields":  list(approved.keys()),
        "hanketyyppi":      req.hanketyyppi,
        "country":          req.country,
    }

    filename = (
        f"NCE_{req.hanketyyppi}_{req.kunta}_approved_"
        f"{timestamp[:10].replace('-','')}.pdf"
    )

    resp = Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-NCE-Audit-Timestamp":    audit["timestamp"],
            "X-NCE-Audit-Reviewer":     audit["reviewer_name"],
            "X-NCE-Audit-Fields":       ",".join(audit["approved_fields"]),
        },
    )
    return resp


@app.get("/api/stats")
async def get_stats():
    try:
        col = _get_chroma_col()
        chunk_count = col.count()
    except Exception:
        chunk_count = 10316  # fallback
    return {
        "chunks_total":  chunk_count,
        "countries":     6,
        "project_types": 20,
        "languages":     7,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
