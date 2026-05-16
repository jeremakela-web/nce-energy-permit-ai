"""
Claude AI -lupaprosessistrategia BESS-sijaintianalyysille.
Kutsuu Anthropic API:a konkreettisen lupaprosessistrategian generoimiseksi.
"""

import os
from typing import Optional

_MODEL = "claude-sonnet-4-5"  # Vaihda tarvittaessa uudempaan

_SYSTEM_PROMPT = (
    "Olet suomalainen BESS-hankkeiden lupaprosessiasiantuntija. "
    "Analysoi sijaintianalyysin tulokset ja anna konkreettinen strategia "
    "lupaprosessin optimoimiseksi. Vastaa aina suomeksi, max 150 sanaa, "
    "bullet-pointteina."
)


def _build_user_prompt(analysis: dict) -> str:
    kunta   = analysis.get("kuntanimi", "–")
    gw      = analysis.get("groundwater_overlap", False)
    gw_cls  = analysis.get("groundwater_class", "")
    pohjavesi = (f"Luokka {gw_cls}" if gw_cls else "Kyllä") if gw else "Ei"
    bldg    = analysis.get("nearest_building_m")
    verkko  = analysis.get("nearest_grid_m")
    natura  = "Kyllä" if analysis.get("natura_overlap") else "Ei"
    heritage= "Kyllä" if analysis.get("heritage_overlap") else "Ei"
    road    = "OK" if analysis.get("road_protection_ok", True) else "Suoja-alueella (<20 m)"
    flood   = analysis.get("flood_overlap")
    flood_s = "Kyllä" if flood else ("Ei" if flood is False else "Ei tiedossa")
    soil    = analysis.get("maaperalaaji", "Ei tiedossa")
    score   = analysis.get("bess_score", 0)
    zoning  = analysis.get("zoning_status", "–")

    return (
        f"Kiinteistö: {kunta}\n"
        f"Kaavatilanne: {zoning}\n"
        f"Pohjavesi: {pohjavesi}\n"
        f"Asutus: {f'{bldg} m' if bldg is not None else 'Ei tiedossa'}\n"
        f"Verkkoetäisyys: {f'{verkko} m' if verkko is not None else 'Ei tiedossa'}\n"
        f"Natura: {natura}\n"
        f"Muinaismuistot: {heritage}\n"
        f"Tiesuoja: {road}\n"
        f"Tulvavaara: {flood_s}\n"
        f"Maaperä: {soil}\n"
        f"Soveltuvuusindeksi: {score}/100\n"
        "\n"
        "Anna optimaalinen lupaprosessistrategia:\n"
        "- Mikä lupa haetaan ensin ja miksi\n"
        "- Mitä voidaan hakea rinnakkain\n"
        "- Mitkä ovat kriittiset pullonkaulat\n"
        "- Arvioitu kokonaisaikataulu\n"
        "- Erityisvaroitukset tämän kohteen osalta"
    )


async def get_lupaprosessi_strategy(analysis: dict) -> dict:
    """
    Kutsuu Claude API:a ja palauttaa lupaprosessistrategian.
    Palauttaa {"strategy": str, "error": str|None}.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {
            "strategy": None,
            "error": "ANTHROPIC_API_KEY puuttuu — aseta ympäristömuuttuja",
        }
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model=_MODEL,
            max_tokens=500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(analysis)}],
        )
        text = msg.content[0].text if msg.content else ""
        return {"strategy": text.strip(), "error": None}
    except Exception as exc:
        return {"strategy": None, "error": str(exc)}
