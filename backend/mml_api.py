"""
MML ja SYKE WFS-rajapintahaut BESS-kaavoituskartoitukseen.

Kiinteistörajat: INSPIRE WFS (ei API-avainta)
  https://inspire-wfs.maanmittauslaitos.fi/inspire-wfs/cp/ows

Pohjavesialueet: SYKE GeoServer WFS (avoin, ei API-avainta)
  https://paikkatiedot.ymparisto.fi/geoserver/syke_vhspohjavesi/wfs
  typeNames=syke_vhspohjavesi:VHS2022_Pohjavesi
  Kentät: pvaluenimi, pvalueluokka, area_m2
  HUOM: intensiiviseen käyttöön (tuotanto/automaatio) pyydetään tunniste:
        gistuki@syke.fi — ilmainen, käsittelyaika n. 1–2 vk

Muut MML-tasot: avoin-paikkatieto WFS (vaatii ilmaisen API-avaimen)
"""

import math
import os
import re
import httpx
from typing import Optional

INSPIRE_CP_WFS   = "https://inspire-wfs.maanmittauslaitos.fi/inspire-wfs/cp/ows"
MML_MAASTO_WFS   = "https://avoin-paikkatieto.maanmittauslaitos.fi/maastotiedot/wfs/v1"
SYKE_POHJAVESI   = "https://paikkatiedot.ymparisto.fi/geoserver/syke_vhspohjavesi/wfs"
_HEADERS         = {"User-Agent": "bess-tool/1.0"}


def format_kiinteistotunnus(kt: str) -> str:
    """'636-439-4-711' → '63643900040711' (MMMKKKRRRRPPPP, 14 merkkiä)."""
    parts = kt.strip().split("-")
    if len(parts) != 4:
        raise ValueError(f"Invalid property ID: {kt!r} (expected MMM-KKK-R-PPPP)")
    muni, village, group, parcel = parts
    return f"{muni.zfill(3)}{village.zfill(3)}{group.zfill(4)}{parcel.zfill(4)}"


async def get_property_boundaries(kiinteistotunnus: str, api_key: Optional[str] = None) -> dict:
    """
    Hakee kiinteistön palstapolygonin MML INSPIRE WFS:stä.
    Ei vaadi API-avainta. Palauttaa GeoJSON EPSG:4326.
    Pinta-ala lasketaan EPSG:3067-koordinaateista (tarkin vapaasti saatavilla oleva arvo).
    """
    kt_fmt = format_kiinteistotunnus(kiinteistotunnus)
    cql = f"nationalCadastralReference='{kt_fmt}'"
    json_params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": "cp:CadastralParcel", "outputFormat": "application/json",
        "SRSNAME": "EPSG:4326",
        "CQL_FILTER": cql,
    }
    gml_params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": "cp:CadastralParcel",
        "CQL_FILTER": cql,
    }
    async with httpx.AsyncClient(timeout=30.0, headers=_HEADERS) as client:
        resp = await client.get(INSPIRE_CP_WFS, params=json_params)
        if not resp.is_success:
            hints = {401: "autentikointi vaaditaan", 403: "pääsy kielletty",
                     404: "tuntematon resurssi", 500: "palvelinvirhe"}
            raise ValueError(f"INSPIRE WFS {resp.status_code} ({hints.get(resp.status_code, 'virhe')})")
        data = resp.json()
        # Fetch GML in EPSG:3067 for accurate metric area
        gml_resp = await client.get(INSPIRE_CP_WFS, params=gml_params)
        area_m2_3067 = _parse_area_from_gml_3067(gml_resp.text) if gml_resp.is_success else None

    for feat in data.get("features", []):
        p = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        feat["properties"] = {
            "kiinteistotunnus": p.get("label") or kiinteistotunnus,
            "kuntanimi": _muni_name(kiinteistotunnus.split("-")[0]),
            "kylanimi": _village_name(kiinteistotunnus),
            "pinta_ala": area_m2_3067 if area_m2_3067 is not None else _polygon_area_m2(geom),
        }
    return data


async def get_groundwater_areas(bbox: tuple[float, float, float, float]) -> dict:
    """
    Hakee pohjavesialueet SYKE GeoServer WFS -rajapinnasta.
    bbox = (minlon, minlat, maxlon, maxlat) WGS84.
    Palauttaa GeoJSON tai unavailable-lipun virheessä.

    Intensiiviseen käyttöön (tuotanto/automaatio) pyydä tunniste:
    gistuki@syke.fi — ilmainen, käsittelyaika n. 1–2 vk
    """
    minx, miny, maxx, maxy = bbox
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": "syke_vhspohjavesi:VHS2022_Pohjavesi",
        "outputFormat": "application/json",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:4326",
    }

    async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
        try:
            resp = await client.get(SYKE_POHJAVESI, params=params)
            if resp.is_success:
                data = resp.json()
                if "features" in data:
                    for feat in data["features"]:
                        p = feat.get("properties") or {}
                        feat["properties"] = {
                            "nimi":   p.get("pvaluenimi") or p.get("nimi") or "Pohjavesialue",
                            "luokka": p.get("pvalueluokka") or p.get("luokka") or "–",
                            "area_m2": p.get("area_m2"),
                        }
                    return data
        except Exception:
            pass

    return {
        "type": "FeatureCollection",
        "features": [],
        "unavailable": True,
        "message": "SYKE pohjavesialue -rajapinta ei vastannut",
    }


async def get_zoning_info(bbox: tuple[float, float, float, float], api_key: Optional[str] = None) -> dict:
    """
    Hakee kaavoitustiedot MML Maastotiedot WFS:stä (vaatii API-avaimen).
    Palauttaa 'unavailable': True jos pyyntö epäonnistui (401/403/verkkovirhe).
    Tyhjä features-lista API-avaimen kanssa = ei asemakaavaa tällä alueella.
    """
    key = api_key or os.getenv("MML_API_KEY", "")
    minx, miny, maxx, maxy = bbox
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": "maastotiedot:KaavoitusAlue", "outputFormat": "application/json",
        "BBOX": f"{miny},{minx},{maxy},{maxx},EPSG:4326", "count": "200",
    }
    if key:
        params["api-key"] = key
    async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
        try:
            resp = await client.get(MML_MAASTO_WFS, params=params)
            if resp.status_code in (401, 403):
                return {"type": "FeatureCollection", "features": [],
                        "unavailable": True, "reason": "api_key_required"}
            resp.raise_for_status()
            data = resp.json()
            data["unavailable"] = False
            return data
        except Exception:
            return {"type": "FeatureCollection", "features": [],
                    "unavailable": True, "reason": "network_error"}


async def get_land_use(bbox: tuple[float, float, float, float], api_key: Optional[str] = None) -> dict:
    """
    Hakee maankäyttö- ja maapeitetiedot MML Maastotiedot WFS:stä (vaatii API-avaimen).
    Fallback: OSM Overpass landuse-tagit.
    """
    key = api_key or os.getenv("MML_API_KEY", "")
    minx, miny, maxx, maxy = bbox
    if key:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeName": "maastotiedot:MaankayttoJaMaapeite", "outputFormat": "application/json",
            "BBOX": f"{miny},{minx},{maxy},{maxx},EPSG:4326", "count": "200",
            "api-key": key,
        }
        async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
            try:
                resp = await client.get(MML_MAASTO_WFS, params=params)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                pass
    # OSM fallback
    return await _get_land_use_osm(bbox)


async def _get_land_use_osm(bbox: tuple[float, float, float, float]) -> dict:
    """OSM landuse-tagit maankäytön arviointiin (fallback kun MML-avain puuttuu)."""
    import httpx as _httpx
    minx, miny, maxx, maxy = bbox
    b = f"{miny},{minx},{maxy},{maxx}"
    query = (
        f"[out:json][timeout:20];"
        f"(way[\"landuse\"~\"farmland|forest|meadow|residential|commercial|industrial\"]({b});"
        f"relation[\"landuse\"~\"farmland|forest|meadow|residential|commercial|industrial\"]({b}););"
        f"out center;"
    )
    try:
        async with _httpx.AsyncClient(timeout=25.0, headers=_HEADERS) as client:
            resp = await client.post("https://overpass-api.de/api/interpreter", data={"data": query})
            resp.raise_for_status()
            raw = resp.json()
        features = []
        for elem in raw.get("elements", []):
            tags = elem.get("tags", {})
            lat = elem.get("lat") or (elem.get("center") or {}).get("lat")
            lon = elem.get("lon") or (elem.get("center") or {}).get("lon")
            if lat is None or lon is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "kohdeluokka":     _LANDUSE_FI.get(tags.get("landuse", ""), tags.get("landuse", "")),
                    "kohdeluokka_osm": tags.get("landuse", ""),
                    "source": "OpenStreetMap",
                },
            })
        return {"type": "FeatureCollection", "features": features, "source": "osm"}
    except Exception:
        return {"type": "FeatureCollection", "features": []}


async def get_natura_areas(bbox: tuple[float, float, float, float]) -> dict:
    """SYKE Natura 2000 -alueet (avoin, ei avainta)."""
    minx, miny, maxx, maxy = bbox
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": "sy:natura2000_sac_fi", "outputFormat": "application/json",
        "BBOX": f"{miny},{minx},{maxy},{maxx},EPSG:4326", "count": "100",
    }
    async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
        try:
            resp = await client.get("https://paikkatiedot.ymparisto.fi/geoserver/sy/ows", params=params)
            if resp.is_success:
                return resp.json()
        except Exception:
            pass
    return {"type": "FeatureCollection", "features": []}


# ── Apufunktiot ──────────────────────────────────────────────────────────────

def _parse_area_from_gml_3067(gml_text: str) -> Optional[float]:
    """
    Parsii posList-koordinaatit EPSG:3067 GML-vastauksesta ja laskee shoelace-kaavalla
    tarkan pinta-alan neliömetreinä. Summataan kaikki renkaat (palstat).
    """
    total = 0.0
    found = False
    for match in re.finditer(r"<gml:posList[^>]*>(.*?)</gml:posList>", gml_text, re.DOTALL):
        nums = list(map(float, match.group(1).split()))
        pts = [(nums[i], nums[i+1]) for i in range(0, len(nums)-1, 2)]
        if len(pts) < 3:
            continue
        area = 0.0
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            area += x1 * y2 - x2 * y1
        total += abs(area) / 2.0
        found = True
    return total if found else None


def _polygon_area_m2(geom: dict) -> Optional[float]:
    coords = (geom.get("coordinates") or [[]])[0]
    if len(coords) < 3:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    lon0 = sum(lons) / len(lons)
    lat0 = sum(lats) / len(lats)
    lat_rad = math.radians(lat0)
    m_lon = 111_320.0 * math.cos(lat_rad)
    m_lat = 111_320.0
    # Vähennetään sentroidi ennen shoelace-laskua → numeerinen stabiilisuus
    xs = [(lon - lon0) * m_lon for lon in lons]
    ys = [(lat - lat0) * m_lat for lat in lats]
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


from muni_names import MUNI_NAMES as _MUNI_NAMES_FULL

def _muni_name(code: str) -> str:
    return _MUNI_NAMES_FULL.get(code.zfill(3), f"Kunta {code}")


_LANDUSE_FI: dict[str, str] = {
    "farmland":    "Peltoalue",
    "forest":      "Metsäalue",
    "meadow":      "Niitty / laidun",
    "residential": "Asuinalue",
    "commercial":  "Kaupallinen alue",
    "industrial":  "Teollisuusalue",
    "retail":      "Vähittäiskauppa",
    "cemetery":    "Hautausmaa",
    "grass":       "Nurmi",
    "allotments":  "Siirtolapuutarha",
    "orchard":     "Hedelmätarha",
    "vineyard":    "Viinitarha",
    "quarry":      "Louhos / kaivos",
    "landfill":    "Kaatopaikka",
    "military":    "Sotilasalue",
    "recreation_ground": "Virkistysalue",
}


_VILLAGE_NAMES = {"636-439": "Kyrö", "636-440": "Karinainen", "636-441": "Auvainen"}

def _village_name(kt: str) -> str:
    return _VILLAGE_NAMES.get("-".join(kt.split("-")[:2]), "–")


async def infer_zoning_from_osm(center_lat: float, center_lon: float, radius_m: float = 1000.0) -> dict:
    """
    OSM-päättely asemakaavatilanteesta.
    Etsii kaupunkimaista maankäyttöä (asuinalue/teollisuus/kauppa) 1 km säteellä.
    Palauttaa {"inferred": "asemakaava" | "rural" | "unknown", "source": "osm"}.
    """
    delta = radius_m / 111_000.0
    b = f"{center_lat - delta},{center_lon - delta},{center_lat + delta},{center_lon + delta}"
    query = (
        f"[out:json][timeout:15];"
        f"(way[\"landuse\"~\"^(residential|commercial|industrial|retail)$\"]({b});"
        f"relation[\"landuse\"~\"^(residential|commercial|industrial|retail)$\"]({b}););"
        f"out center 1;"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
            resp = await client.post("https://overpass-api.de/api/interpreter", data={"data": query})
            resp.raise_for_status()
            raw = resp.json()
        if raw.get("elements"):
            return {"inferred": "asemakaava", "source": "osm"}
        return {"inferred": "rural", "source": "osm"}
    except Exception:
        return {"inferred": "unknown", "source": "osm_error"}


async def get_flood_risk(bbox: tuple[float, float, float, float]) -> dict:
    """
    SYKE tulvavaara-alueet (avoin, ei avainta).
    Palauttaa {"flood_overlap": bool, "unavailable": bool}.
    """
    SYKE_TULVA = "https://paikkatiedot.ymparisto.fi/geoserver/syke_tulvavaara/wfs"
    minx, miny, maxx, maxy = bbox
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": "syke_tulvavaara:tulvavaara_yleinen",
        "outputFormat": "application/json",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:4326",
        "count": "10",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=_HEADERS) as client:
            resp = await client.get(SYKE_TULVA, params=params)
            if resp.is_success:
                data = resp.json()
                return {
                    "flood_overlap": len(data.get("features", [])) > 0,
                    "unavailable": False,
                    "features": data.get("features", []),
                }
    except Exception:
        pass
    return {"flood_overlap": False, "unavailable": True, "features": []}
