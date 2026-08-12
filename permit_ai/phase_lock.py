"""
Phase-Lock logiikka NCE Permit AI:lle.

Historiallinen (ja edelleen oletusarvoinen) järjestys jokaiselle hanketyypille:
Esiselvitys (1) -> Lupavaihe (2) -> Rakentamisvaihe (3)

Vaihemäärä on nyt hanketyyppikohtainen ominaisuus (2026-08-12, Priority 3
-arkkitehtuurityö, Path B hyväksytty) sen sijaan että se olisi kovakoodattu
kaikille yhteinen luku 3 -- katso HANKETYYPPI_PHASE_COUNT alla. Sama
periaate/muoto kuin source_policy.py:n SOURCE_HANKETYYPPI_TAG:illa: yksi
dict, oletusarvo (3) pätee kaikille joita ei ole erikseen listattu, joten
tämä muutos on puhtaasti additiivinen -- yhdenkään olemassa olevan
hanketyypin käyttäytyminen ei muutu millään tavalla. Vain SMR saa toistaiseksi
laajennetun 5-vaiheisen mallin (vaihe 4 = käyttölupa, vaihe 5 = purku); sisältö
näille uusille vaiheille (generate_application.py:n _PHASE_INSTRUCTIONS) on
oma, erillinen, sisällöllisesti katselmoitava jatko-PR (P3-3b), ei tässä.

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

# Hanketyyppikohtainen vaihemäärä. Oletusarvo (DEFAULT_PHASE_COUNT = 3) pätee
# jokaiselle hanketyypille jota ei ole erikseen listattu tässä -- identtinen
# käyttäytyminen kuin ennen tätä muutosta. Lisää tähän vain kun kyseiselle
# hanketyypille on oikeasti olemassa (tai suunnitteilla, hyväksytysti) reaalista
# sisältöä myöhemmille vaiheille -- ei speksautetusti "koska voisi joskus".
HANKETYYPPI_PHASE_COUNT: dict[str, int] = {
    "SMR": 5,
}
DEFAULT_PHASE_COUNT = 3


def get_max_phase(hanketyyppi: str) -> int:
    """Palauttaa hanketyypin käytössä olevan vaihemäärän. Oletus 3."""
    return HANKETYYPPI_PHASE_COUNT.get(hanketyyppi, DEFAULT_PHASE_COUNT)


PHASE_ORDER = {
    "esiselvitys":     1,
    "lupavaihe":       2,
    "rakentaminen":    3,
    "rakentamisvaihe": 3,  # alias
    "kayttolupa":      4,  # käyttölupa -- vain SMR toistaiseksi (ks. HANKETYYPPI_PHASE_COUNT)
    "purku":           5,  # purku/käytöstäpoisto+jätehuolto -- vain SMR toistaiseksi
}

PHASE_NAMES = {
    1: "esiselvitys",
    2: "lupavaihe",
    3: "rakentaminen",
    4: "kayttolupa",
    5: "purku",
}

# Peräkkäisyyden lukitusviestit, avaimena pyydetty vaihenumero. Säilyttää
# TÄSMÄLLEEN alkuperäisen suomenkielisen sanamuodon vaiheille 2 ja 3 (nämä
# ovat käsin kirjoitettuja, eivät mekaanisesti PHASE_NAMES:sta johdettavissa
# olevia partitiivimuotoja) -- vaiheille 4 ja 5 uusi, johdonmukainen
# sanamuoto, koska näille ei ole aiempaa merkkijonoa säilytettävänä.
_PHASE_UNLOCK_ERROR: dict[int, str] = {
    2: "Suorita esiselvitys ensin ennen lupavaihetta.",
    3: "Suorita lupavaihe ensin ennen rakentamisvaihetta.",
    4: "Suorita rakentamisvaihe ensin ennen käyttölupavaihetta.",
    5: "Suorita käyttölupavaihe ensin ennen purkuvaihetta.",
}


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
            "completed_phase": int,   # 0 = ei mitään, 1 = esiselvitys, 2 = lupavaihe,
                                       # 3 = rakentaminen, (SMR: 4 = kayttolupa, 5 = purku)
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
    max_phase = get_max_phase(hanketyyppi)
    next_phase = completed + 1 if completed < max_phase else 0

    phases = []
    for n in range(1, max_phase + 1):
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

    completed_phase must be within [1, get_max_phase(hanketyyppi)] to have
    any effect -- out-of-range values are silently ignored (same no-op
    convention skip_phases() already uses for its own out-of-range case),
    not an error. Added 2026-08-12 (P3-2) -- this function previously had
    no bound at all, flagged during P3-1's self-testing as a pre-existing
    gap (confirmed identical old-vs-new behaviour at the time, so not fixed
    there) and closed here per explicit instruction. check_phase_allowed()
    already blocks *requesting* phase 4/5 for non-SMR hanketyyppi, so this
    is defense in depth at the data-write layer, not the only thing
    preventing it.
    """
    max_phase = get_max_phase(hanketyyppi)
    if not (1 <= completed_phase <= max_phase):
        return get_phase_status(session_id, hanketyyppi)

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
    max_phase = get_max_phase(hanketyyppi)
    if skip_through_phase not in range(1, max_phase + 1):
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

    max_phase = get_max_phase(hanketyyppi)
    if requested_n > max_phase:
        # Vaihe on olemassa PHASE_ORDER:issa (esim. "kayttolupa" jollekin
        # muulle kuin SMR:lle) mutta ei käytössä tälle hanketyypille --
        # eri tilanne kuin tuntematon vaihenimi yllä, eri viesti.
        return False, f"Vaihe '{PHASE_NAMES.get(requested_n, requested_vaihe)}' ei ole käytössä tälle hanketyypille."

    status = get_phase_status(session_id, hanketyyppi)
    completed = status["completed_phase"]

    if completed < requested_n - 1:
        return False, _PHASE_UNLOCK_ERROR.get(requested_n, "Suorita edellinen vaihe ensin.")
    return True, ""
