"""
Phase-Lock logiikka NCE Permit AI:lle.

Järjestys: Esiselvitys (1) → Lupavaihe (2) → Rakentamisvaihe (3)
Tallennus: ~/bess_tool/backend/phase_sessions.json
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional

# Tallennetaan backend/-hakemistoon (main.py:n viereen)
_SESSIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "backend", "phase_sessions.json",
)
_SESSIONS_FILE = os.path.normpath(_SESSIONS_FILE)

_lock = threading.Lock()

PHASE_ORDER = {
    "esiselvitys":   1,
    "lupavaihe":     2,
    "rakentaminen":  3,
    "rakentamisvaihe": 3,  # alias
}

PHASE_NAMES = {1: "esiselvitys", 2: "lupavaihe", 3: "rakentaminen"}


def _load() -> dict:
    try:
        with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    with open(_SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_phase_status(session_id: str, hanketyyppi: str) -> dict:
    """
    Palauttaa vaiheen tilan sessiolle.

    Palautus:
        {
            "completed_phase": int,   # 0 = ei mitään, 1 = esiselvitys, 2 = lupavaihe, 3 = rakentaminen
            "completed_name":  str,   # "esiselvitys" | "lupavaihe" | "rakentaminen" | ""
            "next_phase":      int,   # seuraava avautuva vaihe (0 jos kaikki tehty)
            "phases": [
                {"name": "esiselvitys", "phase": 1, "state": "done"|"active"|"locked"},
                {"name": "lupavaihe",   "phase": 2, "state": ...},
                {"name": "rakentaminen","phase": 3, "state": ...},
            ]
        }
    """
    with _lock:
        data = _load()
    sessions = data.get(session_id, {})
    completed = sessions.get(hanketyyppi, {}).get("completed_phase", 0)
    next_phase = completed + 1 if completed < 3 else 0

    phases = []
    for n in (1, 2, 3):
        if n <= completed:
            state = "done"
        elif n == completed + 1:
            state = "active"
        else:
            state = "locked"
        phases.append({"name": PHASE_NAMES[n], "phase": n, "state": state})

    return {
        "completed_phase": completed,
        "completed_name":  PHASE_NAMES.get(completed, ""),
        "next_phase":      next_phase,
        "phases":          phases,
    }


def unlock_next_phase(session_id: str, hanketyyppi: str, completed_phase: int) -> dict:
    """
    Merkitsee vaiheen valmiiksi. Päivittää vain jos uusi vaihe on suurempi.
    Palauttaa päivitetyn phase_status-dictin.
    """
    with _lock:
        data = _load()
        if session_id not in data:
            data[session_id] = {}
        current = data[session_id].get(hanketyyppi, {}).get("completed_phase", 0)
        if completed_phase > current:
            data[session_id][hanketyyppi] = {
                "completed_phase": completed_phase,
                "updated_at": _now(),
            }
        _save(data)
    return get_phase_status(session_id, hanketyyppi)


def check_phase_allowed(session_id: str, hanketyyppi: str, requested_vaihe: str) -> tuple[bool, str]:
    """
    Tarkistaa onko pyydetty vaihe sallittu.
    Palauttaa (ok: bool, error_msg: str).
    """
    requested_n = PHASE_ORDER.get(requested_vaihe.lower().strip(), 0)
    if requested_n == 0:
        # Tuntematon vaihe — sallitaan (ei pakoteta)
        return True, ""
    if requested_n == 1:
        # Esiselvitys — aina sallittu
        return True, ""

    status = get_phase_status(session_id, hanketyyppi)
    completed = status["completed_phase"]

    if requested_n == 2 and completed < 1:
        return False, "Suorita esiselvitys ensin ennen lupavaihetta."
    if requested_n == 3 and completed < 2:
        return False, "Suorita lupavaihe ensin ennen rakentamisvaihetta."
    return True, ""
