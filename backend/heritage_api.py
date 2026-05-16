"""
Muinaismuistot ja RKY-kohteet — Museovirasto INSPIRE WFS + OSM fallback.

Päärajapinta (INSPIRE WFS):
  https://inspire.museovirasto.fi/wfs
  typeNames: ms:Muinaisjaannos  (kiinteät muinaisjäännökset)
             ms:RKY_alue        (rakennettu kulttuuriympäristö)

Fallback (OSM Overpass):
  historic=archaeological_site|ruins|castle|manor
  Ei virallinen Museovirastorekisteri, mutta kattaa yleisimmät kohteet.
  Virallinen tarkistus: https://www.kyppi.fi/palveluikkuna/mjreki/
"""

import httpx
from typing import Optional

MUSEO_WFS  = "https://inspire.museovirasto.fi/wfs"
OVERPASS   = "https://overpass-api.de/api/interpreter"
_HEADERS   = {"User-Agent": "bess-tool/1.0"}
_OSM_TAGS  = "archaeological_site|ruins|castle|manor|heritage"


async def get_heritage_sites(bbox: tuple[float, float, float, float]) -> dict:
    """
    Hakee muinaismuistot ja RKY-kohteet bboxin alueelta.
    Yrittää ensin Museovirasto INSPIRE WFS:ää, fallback OSM Overpassiin.
    bbox = (minlon, minlat, maxlon, maxlat) WGS84.
    """
    result = await _try_museovirasto(bbox)
    if result is not None:
        return result

    # Fallback: OSM Overpass
    result = await _try_osm(bbox)
    if result is not None:
        return result

    return {
        "type": "FeatureCollection",
        "features": [],
        "unavailable": True,
        "source": "none",
        "message": "Museovirasto WFS ei vastannut, OSM ei tuottanut tuloksia",
    }


async def _try_museovirasto(bbox: tuple) -> Optional[dict]:
    minx, miny, maxx, maxy = bbox
    bbox_str = f"{minx},{miny},{maxx},{maxy},EPSG:4326"
    features = []
    for layer in ("ms:Muinaisjaannos", "ms:RKY_alue"):
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": layer, "outputFormat": "application/json",
            "bbox": bbox_str, "count": "100",
        }
        try:
            async with httpx.AsyncClient(timeout=12.0, headers=_HEADERS) as client:
                resp = await client.get(MUSEO_WFS, params=params)
                if resp.is_success:
                    data = resp.json()
                    for feat in data.get("features", []):
                        p = feat.get("properties") or {}
                        feat["properties"] = {
                            "nimi":   p.get("kohdenimi") or p.get("name") or p.get("nimi") or layer,
                            "tyyppi": p.get("laji") or p.get("type") or layer.split(":")[-1],
                            "source": "Museovirasto INSPIRE WFS",
                        }
                    features.extend(data.get("features", []))
        except Exception:
            pass

    if features:
        return {"type": "FeatureCollection", "features": features, "source": "museovirasto"}
    return None


async def _try_osm(bbox: tuple) -> Optional[dict]:
    minx, miny, maxx, maxy = bbox
    b = f"{miny},{minx},{maxy},{maxx}"
    query = (
        f"[out:json][timeout:20];"
        f"("
        f"node[\"historic\"~\"{_OSM_TAGS}\"]({b});"
        f"way[\"historic\"~\"{_OSM_TAGS}\"]({b});"
        f"relation[\"historic\"~\"{_OSM_TAGS}\"]({b});"
        f");"
        f"out center;"
    )
    try:
        async with httpx.AsyncClient(timeout=25.0, headers=_HEADERS) as client:
            resp = await client.post(OVERPASS, data={"data": query})
            resp.raise_for_status()
            raw = resp.json()
    except Exception:
        return None

    features = []
    for elem in raw.get("elements", []):
        lat = elem.get("lat") or (elem.get("center") or {}).get("lat")
        lon = elem.get("lon") or (elem.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        tags = elem.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "nimi":   tags.get("name") or tags.get("historic", "Muinaismuisto"),
                "tyyppi": tags.get("historic", ""),
                "source": "OpenStreetMap (epävirallinen — tarkista Museovirasto RKI)",
            },
        })

    if features:
        return {
            "type": "FeatureCollection", "features": features,
            "source": "osm",
            "note": "OSM-data — virallinen tarkistus: kyppi.fi/palveluikkuna/mjreki/",
        }
    return {
        "type": "FeatureCollection", "features": [], "source": "osm",
        "note": "OSM-data — virallinen tarkistus: kyppi.fi/palveluikkuna/mjreki/",
    }
