"""
GTK (Geologian tutkimuskeskus) ArcGIS REST API -haku BESS-kaavoituskartoitukseen.

Maaperätiedot: GTK Maaperakartta MapServer
  https://gtkdata.gtk.fi/arcgis/rest/services/Maaperakartta/MapServer/0/query

Fallback: OSM Overpass (natural=bare_rock, natural=wetland, geological=* jne.)
"""

import httpx
from typing import Optional

GTK_MAAPERA_URL = "https://gtkdata.gtk.fi/arcgis/rest/services/Maaperakartta/MapServer/0/query"
_HEADERS = {"User-Agent": "bess-tool/1.0"}

# Pisteet maaperälajin mukaan
_SCORE_MAP: dict[str, int] = {
    "Kallio":   15,
    "Moreeni":  12,
    "Hiekka":   10,
    "Karkea":   10,
    "Savi":      5,
    "Hieta":     5,
    "Turve":     0,
}

# GTK-koodiprefiksit / nimiavainsanat → normalisoitu maaperälaji
_GTK_KOODI_MAP: dict[str, str] = {
    "Ka":  "Kallio",
    "Mr":  "Moreeni",
    "Hk":  "Hiekka",
    "Sr":  "Hiekka",   # Sora → karkea
    "Sa":  "Savi",
    "Si":  "Hieta",    # Siltti / hieta
    "Ht":  "Hieta",
    "Tu":  "Turve",
    "Lj":  "Turve",    # Lieju
}


def _normalize_gtk(koodi: str, nimi: str) -> str:
    """Muuntaa GTK-koodin tai nimen normalisoiduksi maaperälajiksi."""
    koodi = (koodi or "").strip()
    nimi = (nimi or "").strip()

    # Kokeile koodiprefiksiä (2 merkkiä)
    if len(koodi) >= 2:
        prefix = koodi[:2].capitalize()
        if prefix in _GTK_KOODI_MAP:
            return _GTK_KOODI_MAP[prefix]

    # Kokeile koko koodia (1 merkki)
    if koodi in _GTK_KOODI_MAP:
        return _GTK_KOODI_MAP[koodi]

    # Nimi-tekstiin perustuva etsintä
    nimi_lower = nimi.lower()
    if "kallio" in nimi_lower or "rock" in nimi_lower:
        return "Kallio"
    if "moreeni" in nimi_lower or "till" in nimi_lower:
        return "Moreeni"
    if "hiekka" in nimi_lower or "sand" in nimi_lower:
        return "Hiekka"
    if "sora" in nimi_lower or "gravel" in nimi_lower:
        return "Hiekka"
    if "savi" in nimi_lower or "clay" in nimi_lower:
        return "Savi"
    if "siltti" in nimi_lower or "hieta" in nimi_lower or "silt" in nimi_lower:
        return "Hieta"
    if "turve" in nimi_lower or "peat" in nimi_lower or "lieju" in nimi_lower:
        return "Turve"

    return "Ei tiedossa"


def _score(maaperalaaji: str) -> Optional[int]:
    return _SCORE_MAP.get(maaperalaaji)


async def _gtk_query(lat: float, lon: float) -> Optional[dict]:
    """Kyselee GTK ArcGIS REST -palvelusta maaperälajin. Palauttaa None virheessä."""
    params = {
        "geometry": f'{{"x":{lon},"y":{lat},"spatialReference":{{"wkid":4326}}}}',
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "json",
        "returnGeometry": "false",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=_HEADERS) as client:
            resp = await client.get(GTK_MAAPERA_URL, params=params)
            if not resp.is_success:
                return None
            data = resp.json()
            features = data.get("features") or []
            if not features:
                return None
            attrs = features[0].get("attributes") or {}
            # GTK-kentät vaihtelevat versiosta riippuen; kokeillaan yleisimmät
            koodi = (
                attrs.get("MAAPERA_KOODI")
                or attrs.get("KUVAUS_KOODI")
                or attrs.get("SYMBOL")
                or attrs.get("CODE")
                or ""
            )
            nimi = (
                attrs.get("MAAPERA_NIMI")
                or attrs.get("KUVAUS")
                or attrs.get("LABEL")
                or attrs.get("NAME")
                or ""
            )
            return {"koodi": str(koodi), "nimi": str(nimi), "attrs": attrs}
    except Exception:
        return None


async def _osm_fallback(lat: float, lon: float) -> dict:
    """
    OSM Overpass -fallback maaperälajin arvaukseen pistekoordinaatille.
    Tarkistaa natural=bare_rock/cliff (→ Kallio),
              natural=wetland / landuse=wetland (→ Turve),
              geological=* (→ käytetään arvoa sellaisenaan).
    """
    delta = 0.0005  # ~55 m
    b = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
    query = (
        f"[out:json][timeout:15];"
        f"("
        f"  way[\"natural\"~\"bare_rock|cliff\"]({b});"
        f"  relation[\"natural\"~\"bare_rock|cliff\"]({b});"
        f"  way[\"natural\"~\"wetland\"]({b});"
        f"  relation[\"natural\"~\"wetland\"]({b});"
        f"  way[\"landuse\"=\"wetland\"]({b});"
        f"  way[\"geological\"]({b});"
        f"  node[\"geological\"]({b});"
        f");"
        f"out tags 10;"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
            resp = await client.post(
                "https://overpass-api.de/api/interpreter", data={"data": query}
            )
            if not resp.is_success:
                return {"source": "unavailable", "maaperalaaji": "Ei tiedossa", "koodi": ""}
            elements = resp.json().get("elements", [])
        for elem in elements:
            tags = elem.get("tags", {})
            natural = tags.get("natural", "")
            landuse = tags.get("landuse", "")
            geological = tags.get("geological", "")
            if natural in ("bare_rock", "cliff"):
                return {"source": "osm", "maaperalaaji": "Kallio", "koodi": ""}
            if natural == "wetland" or landuse == "wetland":
                return {"source": "osm", "maaperalaaji": "Turve", "koodi": ""}
            if geological:
                return {"source": "osm", "maaperalaaji": geological, "koodi": ""}
    except Exception:
        pass
    return {"source": "unavailable", "maaperalaaji": "Ei tiedossa", "koodi": ""}


async def get_soil_type(lat: float, lon: float) -> dict:
    """
    Hakee maaperälajin GTK Maaperakartta -rajapinnasta koordinaattipisteelle.

    Palauttaa:
    {
        "maaperalaaji":       str,   # "Kallio"|"Moreeni"|"Savi"|"Turve"|"Hiekka"|"Ei tiedossa"
        "maaperalaaji_koodi": str,   # GTK:n raakakoodiarvo (tai "")
        "source":             str,   # "gtk"|"osm"|"unavailable"
        "score_pts":          int|None,
        "unavailable":        bool,
    }
    """
    gtk_result = await _gtk_query(lat, lon)

    if gtk_result is not None:
        koodi = gtk_result["koodi"]
        nimi = gtk_result["nimi"]
        maaperalaaji = _normalize_gtk(koodi, nimi)
        pts = _score(maaperalaaji)
        return {
            "maaperalaaji":       maaperalaaji,
            "maaperalaaji_koodi": koodi,
            "source":             "gtk",
            "score_pts":          pts,
            "unavailable":        False,
        }

    # GTK epäonnistui → OSM-fallback
    fallback = await _osm_fallback(lat, lon)
    maaperalaaji = fallback["maaperalaaji"]
    source = fallback["source"]
    unavailable = source == "unavailable"
    pts = _score(maaperalaaji) if not unavailable else None

    return {
        "maaperalaaji":       maaperalaaji,
        "maaperalaaji_koodi": fallback.get("koodi", ""),
        "source":             source,
        "score_pts":          pts,
        "unavailable":        unavailable,
    }
