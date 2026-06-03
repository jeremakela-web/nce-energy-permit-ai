"""
Maps parsed IFC data to permit application fields for NCE Permit AI.
Returns prefilled fields, missing required fields, and compliance flags.
"""
from __future__ import annotations

from typing import Any

# ── Compliance thresholds ─────────────────────────────────────────────────────

_YVA_AREA_THRESHOLD = 1500.0   # m² → YVA required in FI

_COUNTRY_LABELS = {
    "FI": "Finland",
    "SE": "Sweden",
    "DA": "Denmark",
    "NO": "Norway",
    "PL": "Poland",
}

# Permit-relevant fields that must be present in the final application
_REQUIRED_FIELDS = [
    "project_name",
    "building_type",
    "address",
    "floor_area_total",
    "building_height",
    "fire_rating_walls",
]

_OPTIONAL_FIELDS = [
    "materials",
    "storeys",
    "spaces",
    "structural_system",
]


def _confidence(value: Any) -> float:
    """Simple confidence: 1.0 if value present and non-empty, else 0.0."""
    if value is None:
        return 0.0
    if isinstance(value, (list, dict)) and not value:
        return 0.0
    return 1.0


def _compliance_flags(ifc: dict, project_type: str, country: str) -> list[str]:
    flags = []

    area = ifc.get("floor_area_total")
    height = ifc.get("building_height")
    fire_rating = ifc.get("fire_rating_walls")
    spaces = ifc.get("spaces") or []
    storeys = ifc.get("storeys") or []

    # YVA threshold (FI)
    if country == "FI" and area and area > _YVA_AREA_THRESHOLD:
        flags.append(
            f"floor_area {area:.0f} m² > {_YVA_AREA_THRESHOLD:.0f} m² → "
            "YVA-arviointi saattaa olla pakollinen (YVA-laki 252/2017)"
        )

    # BESS-specific
    if project_type == "BESS":
        if fire_rating is None:
            flags.append(
                "fire_rating_walls puuttuu → Tukes-tarkistus tarvitaan "
                "(Kemikaaliturvallisuuslaki 390/2005, IEC 62619)"
            )
        if area and area > 500.0:
            flags.append(
                f"floor_area {area:.0f} m² > 500 m² → "
                "Pelastussuunnitelma pakollinen (Pelastuslaki 379/2011, 15 §)"
            )
        if country == "FI" and area and area > 200.0:
            flags.append(
                "Akkuvarasto > 200 m² → ympäristölupa Luova:lta tarvitaan (YSL 527/2014)"
            )

    # Fire compartment check
    missing_fc = [s["name"] for s in spaces if not s.get("fire_compartment")]
    if missing_fc:
        sample = ", ".join(missing_fc[:3])
        flags.append(
            f"fire_compartment puuttuu tiloista: {sample}"
            + (" ym." if len(missing_fc) > 3 else "")
            + " → Tukes-tarkistus tarvitaan"
        )

    # Height flag
    if height and height > 28.0:
        flags.append(
            f"building_height {height:.1f} m > 28 m → "
            "korkea rakennus, erityissuunnitelmat pakollisia"
        )

    # Multi-storey
    if len(storeys) > 4:
        flags.append(
            f"{len(storeys)} kerrosta → palotekniset erityisvaatimukset (EN 1991)"
        )

    # Structural warning for BESS
    sc = ifc.get("structural_system", {})
    cols = sc.get("column_count") if sc else None
    if project_type == "BESS" and cols is not None and cols == 0:
        flags.append(
            "column_count = 0 → rakennejärjestelmä epäselvä, rakennesuunnitelmat tarkistettava"
        )

    return flags


def map_to_permit(
    ifc_data: dict,
    project_type: str = "BESS",
    country: str = "FI",
) -> dict:
    """
    Map parsed IFC data to permit application fields.

    Returns:
        {
            "prefilled_fields":  {field: {value, confidence}},
            "missing_fields":    [str],          # required but absent
            "compliance_flags":  [str],
            "summary":           str,
        }
    """
    prefilled: dict[str, dict] = {}
    missing: list[str] = []

    # ── Map IFC fields to permit fields ──────────────────────────────────────
    field_map = {
        "project_name":     ifc_data.get("project_name"),
        "building_type":    ifc_data.get("building_type"),
        "address":          ifc_data.get("address"),
        "floor_area_total": ifc_data.get("floor_area_total"),
        "building_height":  ifc_data.get("building_height"),
        "fire_rating_walls": ifc_data.get("fire_rating_walls"),
        "materials":        ifc_data.get("materials") or [],
        "storeys":          ifc_data.get("storeys") or [],
        "spaces":           ifc_data.get("spaces") or [],
        "ifc_schema":       ifc_data.get("ifc_schema"),
        "structural_system": ifc_data.get("structural_system") or {},
    }

    for field, value in field_map.items():
        conf = _confidence(value)
        if conf > 0.0:
            prefilled[field] = {"value": value, "confidence": conf}
        elif field in _REQUIRED_FIELDS:
            missing.append(field)

    # ── Compliance flags ──────────────────────────────────────────────────────
    flags = _compliance_flags(ifc_data, project_type, country)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pre = len(prefilled)
    n_mis = len(missing)
    area_str = (
        f"{ifc_data['floor_area_total']:.0f} m²"
        if ifc_data.get("floor_area_total") else "tuntematon"
    )
    summary = (
        f"{project_type} · {_COUNTRY_LABELS.get(country, country)} · "
        f"pinta-ala {area_str} · "
        f"{n_pre} kenttää esitäytetty · "
        f"{n_mis} pakollista kenttää puuttuu · "
        f"{len(flags)} vaatimushavaintoa"
    )

    return {
        "prefilled_fields": prefilled,
        "missing_fields": missing,
        "compliance_flags": flags,
        "summary": summary,
    }
