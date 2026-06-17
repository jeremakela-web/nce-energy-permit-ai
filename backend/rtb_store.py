"""
RTB (Ready-to-Build) project tracking — PoC tason JSON-varasto.

Avain: hanke_id = normalize(y_tunnus) + "__" + normalize(kiinteistotunnus)
Tiedosto: permit_ai/embeddings/rtb_projects.json  (persistent disk — survives deploys)
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone

# Store on the persistent disk (same mount as ChromaDB embeddings) so data
# survives container restarts and Render deploys. Falls back to backend/ dir
# in local dev environments where the embeddings dir may not exist.
_PERSISTENT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "permit_ai", "embeddings"
)
_FILE = (
    os.path.join(_PERSISTENT_DIR, "rtb_projects.json")
    if os.path.isdir(_PERSISTENT_DIR)
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtb_projects.json")
)
_lock = threading.Lock()


def _load() -> dict:
    try:
        with open(_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    with open(_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", (s or "")).strip("_").lower()


def make_hanke_id(y_tunnus: str, kiinteistotunnus: str) -> str:
    """Derive stable hanke_id from y_tunnus + kiinteistotunnus."""
    parts = [p for p in (normalize(y_tunnus), normalize(kiinteistotunnus)) if p]
    return "__".join(parts)


def _blank_project(hanke_id: str, y_tunnus="", kiinteistotunnus="",
                   hanketyyppi="", maa="FI") -> dict:
    return {
        "hanke_id": hanke_id,
        "y_tunnus": y_tunnus,
        "kiinteistotunnus": kiinteistotunnus,
        "hanketyyppi": hanketyyppi,
        "maa": maa,
        "created_at": _now(),
        "permit_doc": {"status": "pending", "job_id": None, "phase": "", "updated_at": None},
        "land_use":   {"status": "pending", "updated_at": None},
    }


def get_project(hanke_id: str) -> dict | None:
    with _lock:
        return _load().get(hanke_id)


def update_permit_doc(
    hanke_id: str, *,
    job_id: str,
    phase: str,
    y_tunnus: str = "",
    kiinteistotunnus: str = "",
    hanketyyppi: str = "",
    maa: str = "FI",
) -> dict:
    """Mark permit document as done for this hanke."""
    with _lock:
        data = _load()
        if hanke_id not in data:
            data[hanke_id] = _blank_project(hanke_id, y_tunnus, kiinteistotunnus, hanketyyppi, maa)
        proj = data[hanke_id]
        proj["permit_doc"] = {"status": "done", "job_id": job_id, "phase": phase, "updated_at": _now()}
        if y_tunnus:          proj["y_tunnus"] = y_tunnus
        if kiinteistotunnus:  proj["kiinteistotunnus"] = kiinteistotunnus
        if hanketyyppi:       proj["hanketyyppi"] = hanketyyppi
        _save(data)
        return proj


def update_land_use(
    hanke_id: str, *,
    kiinteistotunnus: str = "",
    hanketyyppi: str = "",
    maa: str = "FI",
) -> dict:
    """Mark land use report as done for this hanke."""
    with _lock:
        data = _load()
        if hanke_id not in data:
            data[hanke_id] = _blank_project(hanke_id, kiinteistotunnus=kiinteistotunnus,
                                            hanketyyppi=hanketyyppi, maa=maa)
        proj = data[hanke_id]
        proj["land_use"] = {"status": "done", "updated_at": _now()}
        if kiinteistotunnus: proj["kiinteistotunnus"] = kiinteistotunnus
        if hanketyyppi:      proj["hanketyyppi"] = hanketyyppi
        _save(data)
        return proj


def rtb_summary(hanke_id: str) -> dict:
    """Return cockpit summary: both statuses + RTB readiness flag."""
    proj = get_project(hanke_id)
    if not proj:
        return {
            "found": False,
            "hanke_id": hanke_id,
            "permit_doc": {"status": "pending"},
            "land_use":   {"status": "pending"},
            "rtb_ready":  False,
        }
    permit_done   = proj["permit_doc"]["status"] == "done"
    land_use_done = proj["land_use"]["status"]   == "done"
    return {
        "found": True,
        "hanke_id":        hanke_id,
        "y_tunnus":        proj.get("y_tunnus", ""),
        "kiinteistotunnus": proj.get("kiinteistotunnus", ""),
        "hanketyyppi":     proj.get("hanketyyppi", ""),
        "maa":             proj.get("maa", "FI"),
        "created_at":      proj.get("created_at"),
        "permit_doc":      proj["permit_doc"],
        "land_use":        proj["land_use"],
        "rtb_ready":       permit_done and land_use_done,
    }
