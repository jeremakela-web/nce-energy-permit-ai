"""
Sähköverkon geodata OSM Overpass-rajapinnasta (ilmainen, ei avain).
Sisältää myös rakennushaun ja kaikki etäisyyslaskennat.

Overpass API: https://overpass-api.de/api/interpreter
  User-Agent: bess-tool/1.0  ← vaaditaan 200-vastaukseen
"""

import math
import httpx
from typing import Optional

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_HEADERS = {"User-Agent": "bess-tool/1.0 (BESS planning tool)"}


# ── Overpass-haut ─────────────────────────────────────────────────────────────

async def get_transmission_lines(
    bbox: tuple[float, float, float, float],
    mml_api_key: Optional[str] = None,  # ei käytössä OSM-haussa
) -> dict:
    """
    Hakee kaikki power-elementit OSM Overpass -rajapinnasta ilman voltage-rajoitusta.
    bbox = (minlon, minlat, maxlon, maxlat) WGS84.

    Overpass-kysely (ei voltage-filteriä — OSM-data on usein puutteellisesti taggattu):
      way["power"="line"]        – kantaverkko / alueverkko / tagittomaton jakeluverkko
      way["power"="minor_line"]  – pienjännitteen ilmajohto (yleisin 20 kV jakeluverkko)
      way["power"="cable"]       – maakaapeli (20 kV tai matalampi)
      node["power"="pole"]       – pylväs (indikoi johdon sijaintia)
      node["power"="tower"]      – pylvästorni
    """
    minx, miny, maxx, maxy = bbox
    b = f"{miny},{minx},{maxy},{maxx}"
    query = (
        f"[out:json][timeout:30];"
        f"("
        f"way[\"power\"=\"line\"]({b});"
        f"way[\"power\"=\"minor_line\"]({b});"
        f"way[\"power\"=\"cable\"]({b});"
        f"node[\"power\"=\"pole\"]({b});"
        f"node[\"power\"=\"tower\"]({b});"
        f");"
        f"out geom;"
    )
    raw = await _overpass(query)
    if raw is None:
        return {"type": "FeatureCollection", "features": [], "error": "Overpass ei vastannut"}

    features = []
    for elem in raw.get("elements", []):
        etype = elem.get("type")
        tags = elem.get("tags", {})
        power_type = tags.get("power", "line")
        voltage_kv = _parse_voltage_kv(tags.get("voltage", ""))

        if etype == "way":
            if "geometry" not in elem:
                continue
            coords = [[pt["lon"], pt["lat"]] for pt in elem["geometry"]]
            if len(coords) < 2:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "voltage_kv": voltage_kv,
                    "power_type": power_type,
                    "line_type": _classify_line(voltage_kv, power_type),
                    "name": tags.get("name", ""),
                    "operator": tags.get("operator", ""),
                    "ref": tags.get("ref", ""),
                    "source": "OpenStreetMap",
                },
            })
        elif etype == "node":
            lat = elem.get("lat")
            lon = elem.get("lon")
            if lat is None or lon is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "voltage_kv": voltage_kv,
                    "power_type": power_type,
                    "line_type": _classify_line(voltage_kv, power_type),
                    "name": tags.get("name", ""),
                    "operator": tags.get("operator", ""),
                    "ref": tags.get("ref", ""),
                    "source": "OpenStreetMap",
                },
            })

    return {"type": "FeatureCollection", "features": features}


async def get_highways(
    bbox: tuple[float, float, float, float],
) -> dict:
    """
    Hakee valtatiet ja kantatiet (OSM highway=motorway|trunk|primary) bboxin alueelta.
    Käytetään tiesuoja-alueen (20 m) tarkistukseen.
    bbox = (minlon, minlat, maxlon, maxlat) WGS84.
    """
    minx, miny, maxx, maxy = bbox
    b = f"{miny},{minx},{maxy},{maxx}"
    query = (
        f"[out:json][timeout:20];"
        f"(way[\"highway\"~\"motorway|trunk|primary\"]({b}););"
        f"out geom;"
    )
    raw = await _overpass(query)
    if raw is None:
        return {"type": "FeatureCollection", "features": [], "error": "Overpass ei vastannut"}

    features = []
    for elem in raw.get("elements", []):
        if elem.get("type") != "way" or "geometry" not in elem:
            continue
        coords = [[pt["lon"], pt["lat"]] for pt in elem["geometry"]]
        if len(coords) < 2:
            continue
        tags = elem.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "highway": tags.get("highway", ""),
                "name":    tags.get("name", ""),
                "ref":     tags.get("ref", ""),
                "source":  "OpenStreetMap",
            },
        })

    return {"type": "FeatureCollection", "features": features}


async def get_substations(
    bbox: tuple[float, float, float, float],
) -> dict:
    """
    Hakee sähköasemat OSM:sta (power=substation).
    Palauttaa GeoJSON pisteinä + lähimmän aseman nimi ja etäisyys.
    bbox = (minlon, minlat, maxlon, maxlat) WGS84.
    """
    minx, miny, maxx, maxy = bbox
    b = f"{miny},{minx},{maxy},{maxx}"
    query = (
        f"[out:json][timeout:20];"
        f"("
        f"node[\"power\"=\"substation\"]({b});"
        f"way[\"power\"=\"substation\"]({b});"
        f");"
        f"out center;"
    )
    raw = await _overpass(query)
    if raw is None:
        return {"type": "FeatureCollection", "features": [], "error": "Overpass ei vastannut"}

    features = []
    for elem in raw.get("elements", []):
        lat = elem.get("lat") or (elem.get("center") or {}).get("lat")
        lon = elem.get("lon") or (elem.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        tags = elem.get("tags", {})
        voltage_kv = _parse_voltage_kv(tags.get("voltage", ""))
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "name":       tags.get("name", "Sähköasema"),
                "voltage_kv": voltage_kv,
                "operator":   tags.get("operator", ""),
                "ref":        tags.get("ref", ""),
                "source":     "OpenStreetMap",
            },
        })

    return {"type": "FeatureCollection", "features": features}


async def get_buildings(
    bbox: tuple[float, float, float, float],
) -> dict:
    """
    Hakee rakennukset (OSM building-tagi) pienen bboxin alueelta.
    Palauttaa GeoJSON FeatureCollection pisteinä (out center).
    bbox = (minlon, minlat, maxlon, maxlat) WGS84.
    """
    minx, miny, maxx, maxy = bbox
    query = (
        f"[out:json][timeout:20];"
        f"("
        f"way[\"building\"]({miny},{minx},{maxy},{maxx});"
        f"node[\"building\"]({miny},{minx},{maxy},{maxx});"
        f");"
        f"out center;"
    )
    raw = await _overpass(query)
    if raw is None:
        return {"type": "FeatureCollection", "features": [], "error": "Overpass ei vastannut"}

    features = []
    for elem in raw.get("elements", []):
        # way → center, node → lat/lon
        lat = elem.get("lat") or (elem.get("center") or {}).get("lat")
        lon = elem.get("lon") or (elem.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        tags = elem.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "building": tags.get("building", "yes"),
                "name": tags.get("name", ""),
                "source": "OpenStreetMap",
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ── Etäisyyslaskenta ──────────────────────────────────────────────────────────

def nearest_substation_info(
    point_lat: float,
    point_lon: float,
    substations_geojson: dict,
) -> dict:
    """Palauttaa lähimmän sähköaseman nimen ja etäisyyden metreinä."""
    best_name, best_dist = "–", float("inf")
    for feat in substations_geojson.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates", [])
        if len(coords) >= 2:
            d = _haversine_m(point_lat, point_lon, coords[1], coords[0])
            if d < best_dist:
                best_dist = d
                p = feat.get("properties") or {}
                best_name = p.get("name", "Sähköasema")
    return {
        "distance_m": round(best_dist) if best_dist < float("inf") else None,
        "name": best_name if best_dist < float("inf") else None,
    }


def nearest_line_distance_m(
    point_lat: float,
    point_lon: float,
    lines_geojson: dict,
    min_voltage_kv: int = 0,
    max_voltage_kv: int = 0,  # 0 = ei ylärajaa
) -> float:
    """
    Lähimmän power-elementin etäisyys metreinä (LineString tai Point).
    min_voltage_kv=0, max_voltage_kv=0 → kaikki elementit.
    max_voltage_kv=25 → vain jakeluverkko (≤20 kV tai tagittomaton, voltage_kv=0).
    Palauttaa -1.0 jos elementtejä ei löydy.
    """
    min_dist = float("inf")
    for feat in lines_geojson.get("features", []):
        props = feat.get("properties") or {}
        v = props.get("voltage_kv", 0)
        if min_voltage_kv and v < min_voltage_kv:
            continue
        if max_voltage_kv and v > max_voltage_kv:
            continue
        geom  = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        if gtype == "Point":
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                d = _haversine_m(point_lat, point_lon, coords[1], coords[0])
                if d < min_dist:
                    min_dist = d
        else:
            for seg in _extract_line_coords(geom):
                for i in range(len(seg) - 1):
                    lon1, lat1 = seg[i]
                    lon2, lat2 = seg[i + 1]
                    d = _point_to_segment_m(point_lat, point_lon, lat1, lon1, lat2, lon2)
                    if d < min_dist:
                        min_dist = d

    return min_dist if min_dist < float("inf") else -1.0


def nearest_point_distance_m(
    point_lat: float,
    point_lon: float,
    points_geojson: dict,
) -> float:
    """
    Lähimmän pisteen (rakennus tms.) etäisyys metreinä.
    Palauttaa -1.0 jos pisteitä ei löydy.
    """
    min_dist = float("inf")
    for feat in points_geojson.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        pt = None
        if gtype == "Point" and len(coords) >= 2:
            pt = coords
        elif gtype == "Polygon" and coords:
            pt = coords[0][0] if coords[0] else None
        elif gtype == "MultiPolygon" and coords:
            pt = coords[0][0][0] if coords[0] and coords[0][0] else None
        if pt:
            d = _haversine_m(point_lat, point_lon, pt[1], pt[0])
            min_dist = min(min_dist, d)

    return min_dist if min_dist < float("inf") else -1.0


# ── Apufunktiot ───────────────────────────────────────────────────────────────

async def _overpass(query: str) -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=35.0, headers=_HEADERS) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


def _parse_voltage_kv(voltage_str: str) -> int:
    try:
        parts = [int(v) for v in str(voltage_str).split(";") if v.strip().isdigit()]
        return max(parts) // 1000 if parts else 0
    except Exception:
        return 0


def _classify_line(voltage_kv: int, power_type: str = "line") -> str:
    if power_type == "minor_line":
        return "Pienjännitejohto (jakeluverkko)"
    if power_type == "cable":
        return "Maakaapeli (jakeluverkko)"
    if power_type in ("pole", "tower"):
        return "Pylväs (jakeluverkko)"
    if voltage_kv >= 400:
        return "400 kV (Kantaverkko)"
    if voltage_kv >= 220:
        return "220 kV (Kantaverkko)"
    if voltage_kv >= 110:
        return "110 kV (Alueverkko)"
    if voltage_kv >= 20:
        return "20 kV (Jakeluverkko)"
    return "Jakeluverkko (voltage-tagi puuttuu)"


def _extract_line_coords(geom: dict) -> list:
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if gtype == "LineString":
        return [coords]
    if gtype == "MultiLineString":
        return coords
    return []


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _point_to_segment_m(
    plat: float, plon: float,
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    d_ab = _haversine_m(lat1, lon1, lat2, lon2)
    if d_ab < 1:
        return _haversine_m(plat, plon, lat1, lon1)
    dx = lon2 - lon1
    dy = lat2 - lat1
    t = max(0.0, min(1.0, ((plon - lon1) * dx + (plat - lat1) * dy) / (dx * dx + dy * dy)))
    return _haversine_m(plat, plon, lat1 + t * dy, lon1 + t * dx)
