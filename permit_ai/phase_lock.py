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
            "completed_name":  str,
            "next_phase":      int,
            "phases": [
                {"name": "esiselvitys", "phase": 1, "state": "done"|"active"|"locked",
                 "completion_type": "generated"|"skipped"|""},
                ...
            ]
        }
    """
    with _lock:
        data = _load()
    sessions = data.get(session_id, {})
    hanke_data = sessions.get(hanketyyppi, {})
    completed = hanke_data.get("completed_phase", 0)
    phase_details = hanke_data.get("phases", {})
    next_phase = completed + 1 if completed < 3 else 0

    phases = []
    for n in (1, 2, 3):
        if n <= completed:
            state = "done"
        elif n == completed + 1:
            state = "active"
        else:
            state = "locked"
        ct = phase_details.get(str(n), {}).get("completion_type", "generated") if n <= completed else ""
        phases.append({"name": PHASE_NAMES[n], "phase": n, "state": state, "completion_type": ct})

    return {
        "completed_phase": completed,
        "completed_name":  PHASE_NAMES.get(completed, ""),
        "next_phase":      next_phase,
        "phases":          phases,
    }


def unlock_next_phase(
    session_id: str,
    hanketyyppi: str,
    completed_phase: int,
    completion_type: str = "generated",
) -> dict:
    """
    Merkitsee vaiheen valmiiksi. Päivittää vain jos uusi vaihe on suurempi.
    completion_type: "generated" | "skipped"
    Palauttaa päivitetyn phase_status-dictin.
    """
    with _lock:
        data = _load()
        if session_id not in data:
            data[session_id] = {}
        hanke = data[session_id].get(hanketyyppi, {})
        current = hanke.get("completed_phase", 0)
        if completed_phase > current:
            phase_details = hanke.get("phases", {})
            phase_details[str(completed_phase)] = {"completion_type": completion_type}
            data[session_id][hanketyyppi] = {
                "completed_phase": completed_phase,
                "phases": phase_details,
                "updated_at": _now(),
            }
        _save(data)
    return get_phase_status(session_id, hanketyyppi)


def skip_phases(session_id: str, hanketyyppi: str, skip_through_phase: int) -> dict:
    """
    Merkitsee vaiheet 1..skip_through_phase ohitetuiksi ('skipped').
    Käytetään kun asiakas liittyy kesken matkan (jo suorittanut vaiheet muualla).
    Ei ylikirjoita jo 'generated'-tilassa olevia vaiheita.
    Palauttaa päivitetyn phase_status-dictin.
    """
    if skip_through_phase not in (1, 2, 3):
        return get_phase_status(session_id, hanketyyppi)

    with _lock:
        data = _load()
        if session_id not in data:
            data[session_id] = {}
        hanke = data[session_id].get(hanketyyppi, {})
        current = hanke.get("completed_phase", 0)
        phase_details = hanke.get("phases", {})

        for n in range(1, skip_through_phase + 1):
            # Don't overwrite a phase already completed via generation
            existing_ct = phase_details.get(str(n), {}).get("completion_type", "")
            if existing_ct != "generated":
                phase_details[str(n)] = {"completion_type": "skipped"}

        new_completed = max(current, skip_through_phase)
        data[session_id][hanketyyppi] = {
            "completed_phase": new_completed,
            "phases": phase_details,
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
