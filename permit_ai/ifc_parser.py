"""
IFC file parser for NCE Permit AI.
Extracts building/project metadata from IFC 2x3 and IFC 4 files.
"""
from __future__ import annotations

import io
import math
import tempfile
import os
from typing import Any


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _attr(entity, *attrs, default=None):
    """Walk a chain of attributes safely."""
    obj = entity
    for a in attrs:
        if obj is None:
            return default
        obj = _safe(lambda o=obj, k=a: getattr(o, k), default)
    return obj if obj is not None else default


def _quantity_value(q) -> float | None:
    """Return numeric value from IfcPhysicalQuantity variants."""
    for attr in ("LengthValue", "AreaValue", "VolumeValue", "CountValue",
                 "WeightValue", "TimeValue", "NominalValue"):
        v = _safe(lambda a=attr: getattr(q, a))
        if v is not None:
            if hasattr(v, "wrappedValue"):
                return float(v.wrappedValue)
            return float(v)
    return None


def _pset_value(entity, pset_name: str, prop_name: str):
    """Return a property value from a named property set."""
    try:
        for rel in entity.IsDefinedBy:
            if rel.is_a("IfcRelDefinesByProperties"):
                pset = rel.RelatingPropertyDefinition
                if not hasattr(pset, "Name") or pset.Name != pset_name:
                    continue
                if hasattr(pset, "HasProperties"):
                    for prop in pset.HasProperties:
                        if prop.Name == prop_name:
                            v = _safe(lambda p=prop: p.NominalValue)
                            if v is not None:
                                return v.wrappedValue if hasattr(v, "wrappedValue") else v
                if hasattr(pset, "Quantities"):
                    for q in pset.Quantities:
                        if q.Name == prop_name:
                            return _quantity_value(q)
    except Exception:
        pass
    return None


def _collect_quantities(ifc, entity_type: str, qty_name: str) -> list[float]:
    """Collect all area/length quantities for a given entity type."""
    values = []
    for ent in _safe(lambda: ifc.by_type(entity_type), []):
        v = None
        for rel in _safe(lambda e=ent: e.IsDefinedBy, []):
            if not _safe(lambda r=rel: r.is_a("IfcRelDefinesByProperties")):
                continue
            pset = _safe(lambda r=rel: r.RelatingPropertyDefinition)
            if pset and hasattr(pset, "Quantities"):
                for q in _safe(lambda p=pset: p.Quantities, []):
                    if q.Name == qty_name:
                        v = _quantity_value(q)
                        break
            if v is not None:
                break
        if v is not None:
            values.append(v)
    return values


def _building_address(building) -> str | None:
    addr = _attr(building, "BuildingAddress")
    if addr is None:
        return None
    parts = []
    for field in ("AddressLines", "Town", "PostalCode", "Country"):
        val = _attr(addr, field)
        if val:
            if isinstance(val, (list, tuple)):
                parts.extend(str(v) for v in val if v)
            else:
                parts.append(str(val))
    return ", ".join(parts) if parts else None


def extract_ifc_data(ifc_file_bytes: bytes) -> dict[str, Any]:
    """
    Parse IFC bytes and return a flat dict of permit-relevant fields.
    Missing or unparseable fields are None — never raises.
    """
    result: dict[str, Any] = {
        "project_name": None,
        "building_type": None,
        "address": None,
        "floor_area_total": None,
        "building_height": None,
        "storeys": [],
        "spaces": [],
        "materials": [],
        "fire_rating_walls": None,
        "structural_system": {
            "beam_count": None,
            "column_count": None,
        },
        "ifc_schema": None,
        "parse_errors": [],
    }

    try:
        import ifcopenshell
    except ImportError:
        result["parse_errors"].append("ifcopenshell not installed")
        return result

    # Write to temp file — ifcopenshell requires a path
    try:
        with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
            tmp.write(ifc_file_bytes)
            tmp_path = tmp.name
        ifc = ifcopenshell.open(tmp_path)
    except Exception as e:
        result["parse_errors"].append(f"open failed: {e}")
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    result["ifc_schema"] = _safe(lambda: ifc.schema)

    # ── Project name ────────────────────────────────────────────────────────
    projects = _safe(lambda: ifc.by_type("IfcProject"), [])
    if projects:
        result["project_name"] = _attr(projects[0], "Name") or _attr(projects[0], "LongName")

    # ── Building ────────────────────────────────────────────────────────────
    buildings = _safe(lambda: ifc.by_type("IfcBuilding"), [])
    if buildings:
        b = buildings[0]
        result["building_type"] = _attr(b, "ObjectType") or _attr(b, "Name")
        result["address"] = _building_address(b)
        h = _pset_value(b, "Pset_BuildingCommon", "TotalHeight")
        if h is None:
            h = _attr(b, "ElevationOfRefHeight")
        if h is not None:
            result["building_height"] = round(float(h), 2)

    # ── Storeys ─────────────────────────────────────────────────────────────
    storeys_raw = _safe(lambda: ifc.by_type("IfcBuildingStorey"), [])
    storey_list = []
    for s in storeys_raw:
        name = _attr(s, "Name") or _attr(s, "LongName") or "–"
        elevation = _safe(lambda x=s: float(x.Elevation)) if _attr(s, "Elevation") is not None else None
        area = _pset_value(s, "Qto_BuildingStoreyBaseQuantities", "GrossFloorArea")
        if area is None:
            area = _pset_value(s, "Pset_BuildingStoreyCommon", "GrossFloorArea")
        storey_list.append({
            "name": name,
            "elevation": round(elevation, 2) if elevation is not None else None,
            "area": round(float(area), 2) if area is not None else None,
        })
    storey_list.sort(key=lambda x: (x["elevation"] or 0))
    result["storeys"] = storey_list

    # ── Total floor area ────────────────────────────────────────────────────
    # Try Qto first, then sum storeys
    total_area = None
    if buildings:
        total_area = _pset_value(buildings[0], "Qto_BuildingBaseQuantities", "GrossFloorArea")
    if total_area is None:
        storey_areas = [s["area"] for s in storey_list if s["area"]]
        if storey_areas:
            total_area = sum(storey_areas)
    if total_area is None:
        # Fall back to summing IfcSpace areas
        space_areas = _collect_quantities(ifc, "IfcSpace", "GrossFloorArea")
        if space_areas:
            total_area = sum(space_areas)
    if total_area is not None:
        result["floor_area_total"] = round(float(total_area), 2)

    # ── Spaces ──────────────────────────────────────────────────────────────
    spaces_raw = _safe(lambda: ifc.by_type("IfcSpace"), [])
    spaces_list = []
    for sp in spaces_raw:
        name = _attr(sp, "Name") or _attr(sp, "LongName") or "–"
        area = _pset_value(sp, "Qto_SpaceBaseQuantities", "GrossFloorArea")
        if area is None:
            area = _pset_value(sp, "Pset_SpaceCommon", "GrossFloorArea")
        fire_comp = _pset_value(sp, "Pset_SpaceFireSafetyRequirements", "FireRiskFactor")
        if fire_comp is None:
            fire_comp = _pset_value(sp, "Pset_SpaceCommon", "FireCompartment")
        spaces_list.append({
            "name": name,
            "area": round(float(area), 2) if area is not None else None,
            "fire_compartment": fire_comp,
        })
    result["spaces"] = spaces_list

    # ── Materials ───────────────────────────────────────────────────────────
    mats = set()
    for mat_ent in _safe(lambda: ifc.by_type("IfcMaterial"), []):
        name = _attr(mat_ent, "Name")
        if name:
            mats.add(str(name))
    result["materials"] = sorted(mats)

    # ── Fire rating of walls ────────────────────────────────────────────────
    fire_ratings = set()
    for wall in _safe(lambda: ifc.by_type("IfcWall"), []):
        fr = _pset_value(wall, "Pset_WallCommon", "FireRating")
        if fr:
            fire_ratings.add(str(fr))
    if fire_ratings:
        result["fire_rating_walls"] = ", ".join(sorted(fire_ratings))

    # ── Structural system ───────────────────────────────────────────────────
    beams = _safe(lambda: ifc.by_type("IfcBeam"), [])
    columns = _safe(lambda: ifc.by_type("IfcColumn"), [])
    result["structural_system"]["beam_count"] = len(beams) if beams else None
    result["structural_system"]["column_count"] = len(columns) if columns else None

    return result
