"""
Energy Permit AI — hakemustengeneraattori.

Generoi lupahakemusluonnoksen PDF-muodossa RAG + Claude -pohjaisesti.
Tukee hanketyypit: BESS | tuulivoima | SMR

Käyttö:
    python3 generate_application.py  (interaktiivinen testiajo)
"""

import io
import os
import re
import sys
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache

# ── ReportLab ────────────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable, KeepTogether, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.pdfgen.canvas import Canvas as _CanvasBase

# ── RAG / AI ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Vakiot
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
_DB_DIR      = os.path.join(_HERE, "embeddings")
_OUTPUT_DIR  = os.path.join(_HERE, "output")
_LOGO_PATH   = os.path.join(_HERE, "..", "backend", "nce_energy_logo.png")
_MODEL_ID    = "claude-sonnet-4-6"
_EMBED_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_embed_model() -> SentenceTransformer:
    return SentenceTransformer(_EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_chroma_col():
    client = chromadb.PersistentClient(path=_DB_DIR)
    return client.get_or_create_collection("permit_docs")

C_NAVY   = colors.HexColor("#16213e")
C_RED    = colors.HexColor("#e94560")
C_BLUE   = colors.HexColor("#3a7bd5")
C_GRAY   = colors.HexColor("#8899aa")
C_LGRAY  = colors.HexColor("#f4f6f9")
C_DGRAY  = colors.HexColor("#cccccc")
C_WARN   = colors.HexColor("#ff9800")
C_GREEN  = colors.HexColor("#4caf50")
C_WHITE  = colors.white

# ─────────────────────────────────────────────────────────────────────────────
# TASO 1 — Automaattinen tekstikorjaus
# ─────────────────────────────────────────────────────────────────────────────

# Rules that only apply to Finnish output (authority renames, FI law corrections)
_POSTPROCESS_RULES_FI: list[tuple[str, str]] = [
    # AVI — suomen taivutusmuodot (pisin ensin)
    (r'\bAVI:sta\b',   'Lupa- ja valvontavirastosta'),
    (r'\bAVI:ssa\b',   'Lupa- ja valvontavirastossa'),
    (r'\bAVI:lta\b',   'Lupa- ja valvontavirastolta'),
    (r'\bAVI:lle\b',   'Lupa- ja valvontavirastolle'),
    (r'\bAVI:ksi\b',   'Lupa- ja valvontavirastoksi'),
    (r'\bAVI:n\b',     'Lupa- ja valvontaviraston'),
    (r'\bAVI\b',       'Lupa- ja valvontavirasto'),
    # aluehallintovirasto — kaikki muodot
    (r'\b[Aa]luehallintovirastosta\b',  'Lupa- ja valvontavirastosta'),
    (r'\b[Aa]luehallintovirastossa\b',  'Lupa- ja valvontavirastossa'),
    (r'\b[Aa]luehallintovirastolta\b',  'Lupa- ja valvontavirastolta'),
    (r'\b[Aa]luehallintovirastolle\b',  'Lupa- ja valvontavirastolle'),
    (r'\b[Aa]luehallintoviraston\b',    'Lupa- ja valvontaviraston'),
    (r'\b[Aa]luehallintovirasto\b',     'Lupa- ja valvontavirasto'),
    # ELY yksinään (ei ELY-keskus jo ennestään)
    (r'\bELY\b(?!-)',  'ELY-keskus'),
    # MRL 132/1999 → Rakentamislaki (ei korvata jos jo korvattu)
    (r'(?<!/ )MRL\s+132/1999',  'Rakentamislaki (751/2023) / MRL 132/1999'),
    # Pelastuslaki virheellinen §-viite
    (r'Pelastuslai[tn]\s*\(?379/2011\)?\s*,?\s*379\s*§[:\s]',
     'Pelastuslaki 379/2011, 15 §: '),
    (r'pelastuslai[tn]\s*\(?379/2011\)?\s*,?\s*379\s*§[:\s]',
     'pelastuslaki 379/2011, 15 §: '),
    # Pelastusopiston ohjeistus → Tukesin ohje
    (r'Pelastusopiston\s+BESS-turvallisuusohjeistus(?:ta)?',
     'Tukesin ohje akkuenergiavarastoille'),
    (r'Pelastusopiston\s+(?:turvallisuus)?ohjeistus(?:ta)?',
     'Tukesin ohje'),
    (r'Pelastusopiston\s+ohje(?:istus)?(?:ta)?',
     'Tukesin ohje'),
    # BESS — C2-tyyppi → selkeä tekninen kuvaus
    (r'\bC2-tyyppi(?:ä|ssä|llä|lta|lle|ksi|stä)?\b',
     '2 tunnin purkautumisaika (C/2)'),
    (r'\bC/2-tyyppi(?:ä|ssä|llä|lta|lle|ksi|stä)?\b',
     '2 tunnin purkautumisaika (C/2)'),
]

# Rules that apply to ALL languages (symbol/emoji cleanup)
_POSTPROCESS_RULES_ALL: list[tuple[str, str]] = [
    (r'■■\s*', ''),
    (r'■\s*',  ''),
]

# Language-specific label for ⚠️ replacement
_HUOM_LABEL: dict[str, str] = {
    "FI": "[Huom] ",
    "EN": "[Note] ",
    "SE": "[Obs] ",
    "DA": "[Bem.] ",
    "NO": "[Merk] ",
    "PL": "[Uwaga] ",
}


def _postprocess_text(text: str, lang: str = "FI") -> str:
    """Fix authority names, law refs, and symbols. Finnish-specific rules only run for FI."""
    huom = _HUOM_LABEL.get(lang, "[Note] ")
    text = re.sub(r'⚠️\s*', huom, text)
    for pattern, replacement in _POSTPROCESS_RULES_ALL:
        text = re.sub(pattern, replacement, text)
    if lang == "FI":
        for pattern, replacement in _POSTPROCESS_RULES_FI:
            text = re.sub(pattern, replacement, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# TASO 2 — AI-oikoluku
# ─────────────────────────────────────────────────────────────────────────────

def _proofread_sections(sections: dict) -> dict:
    """Tarkistuta osiot Claudella ennen PDF-rakennusta."""
    client = anthropic.Anthropic()
    combined = ""
    for key, text in sections.items():
        if text and isinstance(text, str) and text.strip():
            combined += f"\n===OSIO:{key}===\n{text}\n"
    if not combined.strip():
        return sections

    prompt = (
        "Olet asiantunteva tekninen toimittaja. Tarkista ja korjaa seuraava suomalainen "
        "lupahakemusluonnos.\n\n"
        f"{combined}\n\n"
        "TEHTÄVÄ:\n"
        "1. Korjaa kirjoitusvirheet ja kielioppivirheet.\n"
        "2. Varmista viranomaisten nimet vuodelle 2026: "
        "käytä 'Lupa- ja valvontavirasto' (ei AVI), 'ELY-keskus'.\n"
        "3. Tarkista lakiviitteet: Rakentamislaki (751/2023), ei pelkkä MRL 132/1999.\n"
        "4. Varmista kappaleiden selkeä järjestys ja ammattimainen yleiskieli.\n"
        "5. ÄLÄ lisää kommentteja tai selityksiä tekemistäsi muutoksista.\n\n"
        "Palauta teksti TÄSMÄLLEEN samassa muodossa (===OSIO:key=== -jakajat mukaan lukien), "
        "vain korjattuna."
    )
    try:
        resp = anthropic.Anthropic().messages.create(
            model=_MODEL_ID,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        corrected = resp.content[0].text
        result = dict(sections)
        for block in corrected.split("===OSIO:"):
            block = block.strip()
            if not block:
                continue
            eq = block.find("===")
            if eq == -1:
                continue
            key = block[:eq].strip()
            txt = block[eq + 3:].strip()
            if key in result:
                result[key] = txt
        return result
    except Exception as exc:
        print(f"[oikoluku] Varoitus: {exc} — käytetään alkuperäistä tekstiä")
        return sections


# ─────────────────────────────────────────────────────────────────────────────
# Tietomalli
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApplicationInput:
    hanketyyppi:                  str
    kiinteistotunnus:             str
    teho_mw:                      float
    kunta:                        str
    hakija:                       str
    sijainti_ymparistovaikutukset: str = ""
    hankkeen_vaihe:               str = ""
    kohdeviranomainen:            str = ""
    lang:                         str = "FI"  # FI | EN | SE
    country:                      str = "FI"  # FI | SE | DA | NO | PL
    kapasiteetti_mwh:             float = 0.0
    y_tunnus:                     str = ""
    osoite:                       str = ""

# ─────────────────────────────────────────────────────────────────────────────
# Hanketyyppikohtaiset asetukset
# ─────────────────────────────────────────────────────────────────────────────

_HANKE_CFG = {
    "BESS": {
        "nimi_fi":    "Akkuenergiavarastohanke",
        "lyhenne":    "BESS",
        "rag_queries": [
            "BESS akkuvarasto ympäristölupa paloturvallisuusvaatimukset sijoittaminen",
            "litiumioniakku sammutusvedet pohjavesialue ympäristölupa",
            "akkuvarasto verkkoliityntä Fingrid SJV VJV vaatimukset",
        ],
        "luvat": [
            ("Ympäristölupa",                   "Lupa- ja valvontavirasto (Luova)",  "YSL 527/2014"),
            ("Rakennuslupa",                     "Kunta / rakennusvalvonta",          "Rakentamislaki 751/2023 / MRL 132/1999"),
            ("Naapurikuuleminen",                "Kunta / hakija",                    "Rakentamislaki 751/2023, 44 §"),
            ("Pelastussuunnitelma / lausunto",   "Paikallinen pelastuslaitos",        "Pelastuslaki 379/2011, 15 §"),
            ("Verkkoliityntäsopimus",            "Jakeluverkkoyhtiö / Fingrid Oyj",   "Sähkömarkkinalaki 588/2013"),
            ("Maa-aineslupa (tarvitt.)",         "Kunta",                             "Maa-aineslaki 555/1981"),
        ],
        "laki_extra": [
            "YVA-laki 252/2017 (kynnykset ylittyessä)",
            "Kemikaaliturvallisuuslaki 390/2005",
            "Luonnonsuojelulaki 9/2023",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE Energy)",
            "Asemapiirustus ja pohjakartta (M 1:500)",
            "Rakennesuunnitelma (akkukontti + perustukset)",
            "Paloturvallisuusselvitys (NFPA 855 / EN-standardit)",
            "Sammutusvesien keräyssuunnitelma",
            "Ympäristöriskiarvio (pohjavesi, maaperä)",
            "Sähköliityntäsuunnitelma (verkkoyhtiön hyväksymä)",
            "Meluselvitys (jos lähellä asutusta)",
            "Liikenneyhteydet ja huoltotie",
            "Hakijan oikeushenkilön rekisteriote",
            "Valtakirja (jos asiamies edustaa)",
        ],
    },
    "tuulivoima_maa": {
        "nimi_fi":    "Maalle sijoitettava tuulivoimahanke",
        "lyhenne":    "WPP-maa",
        "rag_queries": [
            "tuulivoima YVA ympäristövaikutusten arviointi maa lupa",
            "tuulivoimala kaava suunnittelutarveratkaisu meluselvitys linnusto",
            "tuulivoima Fingrid verkkoliityntä kantaverkko vaatimukset",
        ],
        "luvat": [
            ("YVA-menettely (≥10 MW / ≥5 voimalaa)", "ELY-keskus / Luova",       "YVA-laki 252/2017"),
            ("Osayleiskaava tai asemakaava",          "Kunta",                    "MRL 132/1999 § 77a"),
            ("Rakennuslupa",                          "Kunta / rakennusvalvonta", "Rakentamislaki 751/2023 / MRL 132/1999"),
            ("Naapurikuuleminen",                     "Kunta / hakija",           "Rakentamislaki 751/2023, 44 §"),
            ("Ympäristölupa (tarvitt.)",              "Luova",                    "YSL 527/2014"),
            ("Verkkoliityntäsopimus",                 "Fingrid Oyj / jakelu",     "Sähkömarkkinalaki 588/2013"),
            ("Lentoestevalolupa",                     "Traficom",                 "Ilmailulaki 864/2014"),
            ("Maanvuokrasopimukset",                  "Maanomistajat",            "Maakaari 540/1995"),
        ],
        "laki_extra": [
            "Luonnonsuojelulaki 9/2023",
            "Maantielaki 503/2005 (tiealueet)",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE Energy)",
            "YVA-ohjelma ja YVA-selostus (ELY:n hyväksymä)",
            "Meluselvitys (ETSU-R-97 tai IEC 61400-11)",
            "Varjostusmallinnusraportti",
            "Linnustoselvitys (pesimä- ja muuttolinnut)",
            "Lepakoiden lentoaktiviteettiselvitys",
            "Maisema- ja näkyvyysanalyysi (valokuvasovitteet)",
            "Rakennussuunnitelmat (perustukset, tiet, kaapelointi)",
            "Verkkoliityntälaskelma (tehonlaatuanalyysi)",
            "Maanomistaja- ja sopimustiedot",
            "Lentoestekartoitus (Traficom/Finavia)",
        ],
    },
    "tuulivoima_meri": {
        "nimi_fi":    "Merelle sijoitettava tuulivoimahanke (offshore)",
        "lyhenne":    "WPP-meri",
        "rag_queries": [
            "tuulivoima meri offshore lupa ympäristölupa",
            "tuulivoima YVA vesialue vesiliikenne Traficom",
            "tuulivoima Fingrid verkkoliityntä merikaapeli",
        ],
        "luvat": [
            ("YVA-menettely",                    "ELY-keskus / Luova",       "YVA-laki 252/2017"),
            ("Vesilupa",                         "Luova",                    "Vesilaki 587/2011"),
            ("Ympäristölupa",                    "Luova",                    "YSL 527/2014"),
            ("Rakennuslupa",                     "Kunta / rakennusvalvonta", "Rakentamislaki 751/2023 / MRL 132/1999"),
            ("Naapurikuuleminen",                "Kunta / hakija",           "Rakentamislaki 751/2023, 44 §"),
            ("Alusliikenteen turvallisuuslupa",  "Traficom",                 "Merilaki 674/1994"),
            ("Puolustusvoimien lausunto",        "Puolustusvoimat / PLM",    "Laki alueiden käytöstä"),
            ("Verkkoliityntäsopimus",            "Fingrid Oyj",              "Sähkömarkkinalaki 588/2013"),
            ("Maanvuokra / merialueen käyttöoik.", "Valtio / Metsähallitus", "Vesilaki 587/2011"),
        ],
        "laki_extra": [
            "Luonnonsuojelulaki 9/2023",
            "Merenkulkulaki 1672/2009",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE Energy)",
            "YVA-ohjelma ja YVA-selostus",
            "Meriekologinen vaikutusarviointi (Natura tarvittaessa)",
            "Meluselvitys (ilma- ja vedenalainen melu)",
            "Varjostus- ja näkyvyysanalyysi",
            "Merikaapelireittiselvitys",
            "Pohjasedimenttitutkimus (geotekninen)",
            "Meriliikenteen turvallisuusarviointi",
            "Linnusto- ja lepakkoselvitys merialueella",
            "Puolustusvoimien tutkavaikutusarviointi",
            "Verkkoliityntälaskelma ja muuntajamitoitus",
        ],
    },
    "aurinkovoima": {
        "nimi_fi":    "Aurinkovoimalahanke",
        "lyhenne":    "PV",
        "rag_queries": [
            "aurinkovoima aurinkopaneeli rakennuslupa ympäristölupa",
            "aurinkovoimala verkkoliityntä jakeluverkko vaatimukset",
            "aurinkovoimala maankäyttö kaava maisema",
        ],
        "luvat": [
            ("Rakennuslupa tai toimenpidelupa", "Kunta / rakennusvalvonta",  "Rakentamislaki 751/2023 / MRL 132/1999 § 125–126"),
            ("Naapurikuuleminen",               "Kunta / hakija",            "Rakentamislaki 751/2023, 44 §"),
            ("Suunnittelutarveratkaisu (tarvitt.)", "Kunta",                 "MRL 132/1999 § 137"),
            ("Ympäristölupa (tarvitt. ≥1 ha)",  "Luova / kunta",            "YSL 527/2014"),
            ("Verkkoliityntäsopimus",           "Jakeluverkkoyhtiö",         "Sähkömarkkinalaki 588/2013"),
            ("Maisema- tai kulttuuriympäristölausunto", "ELY-keskus",        "MRL 197 §"),
        ],
        "laki_extra": [
            "YVA-laki 252/2017 (≥50 ha hankkeet)",
            "Luonnonsuojelulaki 9/2023",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Asemapiirustus ja pohjakartta (M 1:500 tai 1:1000)",
            "Paneelijärjestely- ja rakennesuunnitelma",
            "Varjostus- ja häikäisyanalyysi (naapurikiinteistöt)",
            "Maisema- ja näkyvyysanalyysi",
            "Verkkoliityntäsuunnitelma (invertteri, muuntaja)",
            "Maaperä- ja hulevesiselvitys (suuri-alainen asennus)",
            "Luontoselvitys (ekologiset yhteydet, mahdollinen Natura)",
            "Asukasosallistumisen asiakirjat (suunnittelutarveratkaisussa)",
            "Hakijan rekisteriote",
        ],
    },
    "SMR": {
        "nimi_fi":    "Pienydinreaktori (SMR) — esilupalupahakemus",
        "lyhenne":    "SMR",
        "rag_queries": [
            "ydinvoima lupa STUK ydinturvallisuus YVL-ohje regulatory oversight",
            "ydinlaitos ympäristövaikutusten arviointi periaatepäätös safety design",
            "pienydinvoimala SMR rakentamislupa käyttölupa structural radiation safety",
        ],
        "luvat": [
            ("Periaatepäätös (VN)",               "Valtioneuvosto",             "Ydinenergialaki 990/1987 § 11"),
            ("YVA-menettely",                     "TEM / ELY-keskus",           "YVA-laki 252/2017"),
            ("Rakentamislupa",                    "STUK",                       "YEL 990/1987 § 18"),
            ("Käyttölupa",                        "STUK",                       "YEL 990/1987 § 20"),
            ("Ympäristölupa",                     "Luova",                      "YSL 527/2014"),
            ("Vesilupa (jäähdytysvesi)",           "Luova",                     "Vesilaki 587/2011"),
            ("Rakennuslupa",                      "Kunta",                      "Rakentamislaki 751/2023 / MRL 132/1999"),
            ("Naapurikuuleminen",                 "Kunta / hakija",              "Rakentamislaki 751/2023, 44 §"),
            ("Maankäyttösopimus / kaavoitus",     "Kunta",                      "MRL 132/1999 § 9"),
        ],
        "laki_extra": [
            "Säteilylaki 859/2018",
            "Luonnonsuojelulaki 9/2023",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE Energy)",
            "Alustava turvallisuusseloste (STUK YVL A.1 mukainen)",
            "YVA-ohjelma ja -selostus",
            "Ydinmateriaalivalvontasuunnitelma (IAEA SQ-protokolla)",
            "Säteilyturvallisuusanalyysi (YVL C.1)",
            "Turvallisuussuunnittelun periaatteet (YVL B.1)",
            "Hätäjärjestelmien ja -menettelyjen kuvaus",
            "Polttoainekierto- ja ydinjätehuoltosuunnitelma",
            "Geotekninen perusselvitys (seismisyys, hydrogeologia)",
            "Jäähdytysveden saatavuus- ja ympäristöarviointi",
            "Sosioekonominen vaikutusarviointi",
            "Kansainväliset referenssilaitosvertailut (IAEA)",
        ],
    },
    "vesivoima": {
        "nimi_fi":    "Vesivoimalahanke",
        "lyhenne":    "HPP",
        "rag_queries": [
            "vesivoima vesivoimala vesilupa rakentaminen",
            "vesistö pato padotus vesirakentaminen ympäristölupa",
            "vesivoima kalakannat ekologinen virtaama vesistö",
        ],
        "luvat": [
            ("Vesilupa (padotus, rakentaminen)", "Luova",                      "Vesilaki 587/2011 § 3:2"),
            ("Ympäristölupa",                    "Luova",                      "YSL 527/2014"),
            ("YVA-menettely (tarvitt.)",          "ELY-keskus / Luova",        "YVA-laki 252/2017"),
            ("Rakennuslupa",                     "Kunta / rakennusvalvonta",   "Rakentamislaki 751/2023 / MRL 132/1999"),
            ("Naapurikuuleminen",                "Kunta / hakija",              "Rakentamislaki 751/2023, 44 §"),
            ("Verkkoliityntäsopimus",            "Jakeluverkkoyhtiö / Fingrid", "Sähkömarkkinalaki 588/2013"),
            ("Kalastuslaki-ilmoitus",            "ELY-keskus",                 "Kalastuslaki 379/2015"),
            ("Maankäyttösopimus",                "Kunta / maanomistajat",      "MRL 132/1999 § 91a"),
        ],
        "laki_extra": [
            "Luonnonsuojelulaki 9/2023",
            "Patoturvallisuuslaki 494/2009",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE Energy)",
            "Hydraulinen mitoitusraportti (virtaama, putouskorkeus)",
            "Geotekninen pato- ja pohjarakenneselvitys",
            "Vesistövaikutusten arviointi (tulva, kuivuus, vedenlaatu)",
            "Ekologinen virtaamaselvitys (kalat, pohjaeläimet)",
            "Kalaston vaellusesteiden ja kalateiden suunnitelma",
            "Padon turvallisuussuunnitelma (PATL 494/2009)",
            "Rakennussuunnitelmat (pato, voimalaitosrakennus)",
            "Verkkoliityntälaskelma",
            "Maanomistaja- ja vesioikeusasiakirjat",
            "Hätätilannesuunnitelma (padotusriskit)",
        ],
    },
    "hybridi": {
        "nimi_fi":    "Hybridivoimalahanke (BESS + tuuli/aurinko)",
        "lyhenne":    "HYB",
        "rag_queries": [
            "BESS akkuvarasto ympäristölupa paloturvallisuus litiumioniakku",
            "tuulivoima aurinkovoima YVA lupa kaava meluselvitys",
            "hybridivoimala verkkoliityntä Fingrid SJV VJV",
        ],
        "luvat": [
            ("YVA-menettely (kynnyksen ylittyessä)", "ELY-keskus / Luova",      "YVA-laki 252/2017"),
            ("Osayleiskaava / asemakaava",           "Kunta",                   "MRL 132/1999"),
            ("Rakennuslupa (tuulivoimala)",          "Kunta / rakennusvalvonta","Rakentamislaki 751/2023 / MRL 132/1999"),
            ("Naapurikuuleminen",                   "Kunta / hakija",          "Rakentamislaki 751/2023, 44 §"),
            ("Rakennus-/toimenpidelupa (PV + BESS)", "Kunta",                   "Rakentamislaki 751/2023 / MRL 132/1999 § 126"),
            ("Ympäristölupa (BESS-komponentti)",    "Luova",                    "YSL 527/2014"),
            ("Pelastussuunnitelma / lausunto (BESS)","Pelastuslaitos",           "Pelastuslaki 379/2011, 15 §"),
            ("Verkkoliityntäsopimus",               "Fingrid Oyj / jakelu",     "Sähkömarkkinalaki 588/2013"),
            ("Lentoestevalolupa (tuulivoimala)",    "Traficom",                 "Ilmailulaki 864/2014"),
        ],
        "laki_extra": [
            "Kemikaaliturvallisuuslaki 390/2005 (BESS)",
            "Luonnonsuojelulaki 9/2023",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE Energy)",
            "YVA-ohjelma ja -selostus (tuulivoiman osalta)",
            "BESS-paloturvallisuusselvitys (NFPA 855)",
            "Sammutusvesien keräyssuunnitelma (BESS)",
            "Meluselvitys (tuulivoimalakomponentti)",
            "Varjostus- ja näkyvyysanalyysi",
            "Linnusto- ja lepakoiden aktiviteettiselvitys",
            "Integroitu verkkoliityntäsuunnitelma (tuuli + PV + BESS)",
            "Energiavarastomitoitusraportti (kapasiteetti, teho, kesto)",
            "Maisema- ja näkyvyysanalyysi",
            "Lentoestekartoitus (Traficom/Finavia)",
        ],
    },
    "business_finland": {
        "nimi_fi":    "Business Finland Sprint — T&K-rahoitushakemus",
        "lyhenne":    "BF-Sprint",
        "rag_queries": [
            "energia-alan tutkimus kehitys innovaatio rahoitus T&K",
            "akkuenergia aurinkovoima tuulivoima teknologia kehitys innovaatio",
            "energiavarasto tehoelektroniikka ohjausjärjestelmä tutkimus",
        ],
        "luvat": [],
        "liitteet": [
            "Sijaintikartta / projektikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE Energy)",
            "Hakijan taloudellinen tilanne (tilinpäätös, 2 viimeisintä vuotta)",
            "Projektisuunnitelma (T&K-kuvaus, tavoitteet, metodologia)",
            "Budjettilaskelmat ja rahoitussuunnitelma",
            "Tiimikuvaus (ansioluettelot, osaamisprofiilit)",
            "Riskiarviointi ja mitigaatiosuunnitelma",
            "Referenssit ja aiempi T&K-toiminta",
            "IPR-suunnitelma (immateriaalioikeuksien hallinta)",
        ],
    },
    "smr_bess": {
        "nimi_fi":    "SMR + BESS hybridienergijärjestelmä",
        "lyhenne":    "SMR+BESS",
        "rag_queries": [
            "ydinvoima SMR lupa STUK pre-licensing YVL turvallisuusseloste",
            "BESS akkuvarasto ympäristölupa paloturvallisuus litiumioniakku sammutusvedet",
            "pienydinvoimala energiavarasto hybridijärjestelmä verkkoliityntä Fingrid",
        ],
        "luvat": [
            ("Periaatepäätös (VN)",                "Valtioneuvosto",             "Ydinenergialaki 990/1987 § 11"),
            ("YVA-menettely",                      "TEM / ELY-keskus",           "YVA-laki 252/2017"),
            ("Rakentamislupa (ydinlaitos)",         "STUK",                       "YEL 990/1987 § 18"),
            ("Käyttölupa (ydinlaitos)",             "STUK",                       "YEL 990/1987 § 20"),
            ("Ympäristölupa (BESS-komponentti)",    "Luova",                      "YSL 527/2014"),
            ("Pelastussuunnitelma / lausunto (BESS)","Pelastuslaitos",             "Pelastuslaki 379/2011, 15 §"),
            ("Rakennuslupa",                        "Kunta",                      "Rakentamislaki 751/2023 / MRL 132/1999"),
            ("Naapurikuuleminen",                   "Kunta / hakija",             "Rakentamislaki 751/2023, 44 §"),
            ("Vesilupa (jäähdytysvesi, tarvitt.)",  "Luova",                      "Vesilaki 587/2011"),
            ("Verkkoliityntäsopimus",               "Fingrid Oyj",                "Sähkömarkkinalaki 588/2013"),
        ],
        "laki_extra": [
            "Säteilylaki 859/2018",
            "Kemikaaliturvallisuuslaki 390/2005 (BESS)",
            "Luonnonsuojelulaki 9/2023",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE Energy)",
            "Alustava turvallisuusseloste (STUK YVL A.1 mukainen)",
            "BESS-paloturvallisuusselvitys (NFPA 855 / EN-standardit)",
            "Integroitu energiavarastosuunnitelma (SMR + BESS-mitoitus)",
            "YVA-ohjelma ja -selostus",
            "Säteilyturvallisuusanalyysi (YVL C.1)",
            "Turvallisuussuunnittelun periaatteet (YVL B.1)",
            "Ydinmateriaalivalvontasuunnitelma (IAEA SQ-protokolla)",
            "Sammutusvesien keräyssuunnitelma (BESS-komponentti)",
            "Geotekninen perusselvitys (seismisyys, hydrogeologia)",
            "Jäähdytysvesitarve- ja ympäristöarviointi",
            "Verkkoliityntälaskelma (SMR + BESS yhdistetty)",
            "Hätäjärjestelmien ja -menettelyjen kuvaus",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# RAG-haku
# ─────────────────────────────────────────────────────────────────────────────

def _rag_context(
    hanketyyppi: str,
    country: str = "FI",
    n_per_query: int = 4,
) -> tuple[str, list[str]]:
    """Hae relevantit dokumenttichunkit.

    Jos country != 'FI', haetaan ensin maakohtaiset dokumentit ja täydennetään
    FI-dokumenteilla (suomalainen lainsäädäntö on aina relevanttia kontekstia).
    Graceful fallback: jos metadata-suodatus epäonnistuu, haetaan ilman suodatinta.
    """
    cfg = _HANKE_CFG[hanketyyppi]
    try:
        embed_model = _get_embed_model()
        col         = _get_chroma_col()

        seen_ids:    set[str]  = set()
        all_docs:    list[str] = []
        all_sources: set[str]  = set()

        def _collect(results: dict) -> None:
            for doc, id_ in zip(results["documents"][0], results["ids"][0]):
                if id_ not in seen_ids:
                    seen_ids.add(id_)
                    all_docs.append(doc)
                    # source = ID ilman viimeistä "__N" tai "_N" osaa
                    all_sources.add(re.sub(r"[_-]\d+$", "", id_))

        for q in cfg["rag_queries"]:
            emb = embed_model.encode([q]).tolist()

            # 1. Maakohtainen haku (vain kun country != FI ja metadata on olemassa)
            if country != "FI":
                try:
                    _collect(col.query(
                        query_embeddings=emb,
                        n_results=n_per_query,
                        where={"country": country},
                    ))
                except Exception:
                    pass  # maakohtaisia dokumentteja ei vielä indeksoitu

            # 2. FI-haku (tai koko indeksi jos metadata puuttuu)
            try:
                _collect(col.query(
                    query_embeddings=emb,
                    n_results=n_per_query,
                    where={"country": "FI"},
                ))
            except Exception:
                # Vanha indeksi ilman metadataa — hae ilman suodatinta
                try:
                    _collect(col.query(query_embeddings=emb, n_results=n_per_query))
                except Exception:
                    pass

        context = "\n\n---\n\n".join(all_docs)
        return context, sorted(all_sources)
    except Exception as exc:
        print(f"[RAG] Haku epäonnistui ({exc}) — jatketaan ilman kontekstia")
        return "", []


# ─────────────────────────────────────────────────────────────────────────────
# Claude AI — hakemustekstin generointi
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Hanketyyppien nimet muilla kielillä (meta-taulukko PDF:ssä)
# ─────────────────────────────────────────────────────────────────────────────
_HANKE_NIMI_TRANS: dict[str, dict[str, str]] = {
    "BESS":           {"EN": "Battery Energy Storage System (BESS)",     "SE": "Batterienergilagersystem (BESS)"},
    "tuulivoima_maa": {"EN": "Onshore Wind Power Project",               "SE": "Landbaserat vindkraftsprojekt"},
    "tuulivoima_meri":{"EN": "Offshore Wind Power Project",              "SE": "Offshorevindkraftsprojekt"},
    "aurinkovoima":   {"EN": "Solar Power Plant Project",                "SE": "Solkraftsprojekt"},
    "SMR":            {"EN": "Small Modular Reactor (SMR) — pre-licensing","SE": "Liten modulär reaktor (SMR) — förlicensiering"},
    "vesivoima":      {"EN": "Hydroelectric Power Project",              "SE": "Vattenkraftsprojekt"},
    "smr_bess":       {"EN": "SMR + BESS Hybrid Energy System",          "SE": "SMR + BESS hybridsystem"},
    "business_finland":{"EN": "Business Finland R&D Grant Application",  "SE": "Business Finland FoU-bidragsansökan"},
    "asuinrakennus":  {"EN": "Residential Building Permit Application",  "SE": "Bygglovsansökan för bostadsbyggnad"},
    "teollisuus":     {"EN": "Industrial Building Permit Application",   "SE": "Bygglovsansökan för industribyggnad"},
    "maatalous":      {"EN": "Agricultural Building Permit Application", "SE": "Bygglovsansökan för lantbruksbyggnad"},
    "liikerakennus":  {"EN": "Commercial Building Permit Application",   "SE": "Bygglovsansökan för affärsbyggnad"},
    "muu":            {"EN": "Other Project Permit Application",         "SE": "Tillståndsansökan för annat projekt"},
}

def _nimi(lang: str, hanketyyppi: str, nimi_fi: str) -> str:
    if lang == "FI":
        return nimi_fi
    d = _HANKE_NIMI_TRANS.get(hanketyyppi, {})
    return d.get(lang, nimi_fi)

# ─────────────────────────────────────────────────────────────────────────────
# Maakohtainen sääntelytieto
# ─────────────────────────────────────────────────────────────────────────────
_COUNTRY_CONFIG: dict[str, dict] = {
    "FI": {
        "name": "Finland",
        "prompt_prefix": "",
    },
    "SE": {
        "name": "Sweden / Sverige",
        "authorities": ["Länsstyrelsen", "Energimyndigheten", "Boverket", "Mark- och miljödomstolen", "Naturvårdsverket"],
        "key_laws": ["Plan- och bygglagen (PBL 2010:900)", "Miljöbalken (MB 1998:808)", "Ellagen (1997:857)", "Miljöprövningsförordningen (2013:251)"],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in SWEDEN. Apply Swedish regulatory framework:\n"
            "Key authorities: Länsstyrelsen (county board), Energimyndigheten (energy agency), "
            "Boverket (building standards), Mark- och miljödomstolen (environmental court), "
            "Naturvårdsverket (environmental protection).\n"
            "Key laws: Plan- och bygglagen PBL 2010:900 (building permits = Bygglov), "
            "Miljöbalken MB 1998:808 (environmental permits = Miljötillstånd), "
            "Ellagen 1997:857 (grid connection), Miljöprövningsförordningen 2013:251 (EIA = MKB).\n"
            "Replace all Finnish law references (MRL, YSL, YVA-laki) with Swedish equivalents. "
            "If a Swedish equivalent is uncertain, mark it: [Requires verification against Swedish regulations].\n\n"
        ),
    },
    "DA": {
        "name": "Denmark / Danmark",
        "authorities": ["Energistyrelsen", "Miljøstyrelsen", "kommunalbestyrelse", "Planklagenævnet", "Kystdirektoratet"],
        "key_laws": ["Planloven (LBK nr 1157/2022)", "Miljøvurderingsloven (LOV nr 973/2023)", "Elforsyningsloven (LBK nr 1255/2021)", "Naturbeskyttelsesloven"],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in DENMARK. Apply Danish regulatory framework:\n"
            "Key authorities: Energistyrelsen (Danish Energy Agency), Miljøstyrelsen (EPA), "
            "kommunalbestyrelse (municipal council), Planklagenævnet (planning appeals board), "
            "Kystdirektoratet (coastal authority for offshore).\n"
            "Key laws: Planloven for land use planning (building permit = Byggetilladelse), "
            "Miljøvurderingsloven for EIA (= Miljøkonsekvensvurdering / MKV), "
            "Elforsyningsloven for electricity supply, Naturbeskyttelsesloven for nature protection.\n"
            "Replace Finnish law references with Danish equivalents. "
            "Mark uncertain items: [Requires verification against Danish regulations].\n\n"
        ),
    },
    "NO": {
        "name": "Norway / Norge",
        "authorities": ["NVE (Norges vassdrags- og energidirektorat)", "Statsforvalteren", "DSB", "Kommunen", "Miljødirektoratet"],
        "key_laws": ["Plan- og bygningsloven (PBL 2008)", "Energiloven (1990)", "Forurensningsloven (1981)", "Naturmangfoldloven (2009)"],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in NORWAY. Apply Norwegian regulatory framework:\n"
            "Key authorities: NVE (Norwegian Water Resources and Energy Directorate), "
            "Statsforvalteren (county governor), DSB (civil protection), Kommunen (municipality), "
            "Miljødirektoratet (Environment Agency).\n"
            "Key laws: Plan- og bygningsloven PBL 2008 (building permit = Byggetillatelse), "
            "Energiloven 1990 (energy facilities), Forurensningsloven 1981 (pollution/environmental), "
            "Naturmangfoldloven 2009 (biodiversity). EIA = Konsekvensutredning (KU).\n"
            "Replace Finnish law references with Norwegian equivalents. "
            "Mark uncertain items: [Requires verification against Norwegian regulations].\n\n"
        ),
    },
    "PL": {
        "name": "Poland / Polska",
        "authorities": ["PAA (Państwowa Agencja Atomistyki)", "URE (Urząd Regulacji Energetyki)", "RDOŚ", "Starosta (building authority)", "GDOŚ"],
        "key_laws": ["Prawo atomowe (Ustawa z 29.11.2000)", "Prawo budowlane (Ustawa z 7.07.1994)", "Ustawa o OZE (20.02.2015)", "Ustawa o udostępnianiu informacji o środowisku"],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in POLAND. Apply Polish regulatory framework:\n"
            "Key authorities: PAA (State Nuclear Agency, for nuclear projects), "
            "URE (Energy Regulatory Office), RDOŚ (Regional Environmental Directorate), "
            "Starosta (poviat/district authority for building permits = Pozwolenie na budowę), "
            "GDOŚ (General Directorate for Environmental Protection).\n"
            "Key laws: Prawo budowlane 1994 (building permits), Ustawa o OZE 2015 (renewables), "
            "Prawo atomowe 2000 (nuclear), Ustawa o udostępnianiu informacji o środowisku (EIA = OOŚ).\n"
            "Replace Finnish law references with Polish equivalents. "
            "Mark uncertain items: [Requires verification against Polish regulations].\n\n"
        ),
    },
}

_SYSTEM = (
    "Olet NCE Energy Permit AI -asiantuntija, joka avustaa energia-alan lupahakemusten "
    "laadinnassa Suomessa. Kirjoitat selkeää, virallista kieltä konsulttiraporttityyliin. "
    "Viittaat aina voimassa olevaan lainsäädäntöön. Et koskaan anna harhaanjohtavaa tietoa — "
    "jos jokin asia on epävarma, merkitset sen selvästi. "
    "Kaikki tuottamasi teksti on AI-luonnos joka vaatii asiantuntijatarkistuksen."
)

_LANG_INSTRUCTIONS: dict[str, str] = {
    "FI": "",
    "EN": (
        "CRITICAL LANGUAGE REQUIREMENT: You MUST write EVERY word of this permit application "
        "in English. ALL headings, paragraphs, bullet points, footnotes, and notes must be in "
        "English. Do NOT include any Finnish words or sentences in the output. "
        "Finnish statute numbers (e.g. YSL 527/2014, MRL 132/1999) may appear as legal identifiers "
        "only — always add the English act name next to them. Finnish proper nouns such as city names, "
        "company names and authority acronyms (ELY, STUK, Luova, Fingrid, Traficom) are acceptable "
        "as proper names only.\n\n"
    ),
    "SE": (
        "KRITISKT SPRÅKKRAV: Du MÅSTE skriva VARJE ord i denna tillståndsansökan på svenska. "
        "ALLA rubriker, stycken, punktlistor, fotnoter och anmärkningar ska vara på svenska. "
        "Inkludera INTE finska ord eller meningar i utdata. "
        "Finska lagrumsnummer (t.ex. YSL 527/2014, MRL 132/1999) får förekomma som juridiska "
        "identifierare — lägg alltid till det svenska lagnamnet bredvid dem. Finska egennamn "
        "som stadsnamn, företagsnamn och myndighetsförkortningar (ELY, STUK, Luova, Fingrid, Traficom) "
        "är godtagbara enbart som egennamn.\n\n"
    ),
    "DA": (
        "KRITISK SPROGKRAV: Du SKAL skrive HVERT ord i denne tilladelsesansøgning på dansk. "
        "ALLE overskrifter, afsnit, punktlister, fodnoter og noter skal være på dansk. "
        "Medtag IKKE finske ord eller sætninger i outputtet. "
        "Finske lovnumre (f.eks. YSL 527/2014, MRL 132/1999) må forekomme som juridiske "
        "identifikatorer — tilføj altid det danske lovnavn ved siden af dem. Finske egennavne "
        "som bynavne, virksomhedsnavne og myndighedsforkortelser (ELY, STUK, Luova, Fingrid, Traficom) "
        "er acceptable udelukkende som egennavne.\n\n"
    ),
    "NO": (
        "KRITISK SPRÅKKRAV: Du MÅ skrive HVERT ord i denne tillatelsessøknaden på norsk (bokmål). "
        "ALLE overskrifter, avsnitt, punktlister, fotnoter og merknader skal være på norsk. "
        "IKKE inkluder finske ord eller setninger i utdataene. "
        "Finske lovnumre (f.eks. YSL 527/2014, MRL 132/1999) kan forekomme som juridiske "
        "identifikatorer — legg alltid til det norske lovnavnet ved siden av dem. Finske egennavn "
        "som bynavn, firmanavn og myndighetsforkortelser (ELY, STUK, Luova, Fingrid, Traficom) "
        "er akseptable utelukkende som egennavn.\n\n"
    ),
    "PL": (
        "KRYTYCZNY WYMÓG JĘZYKOWY: MUSISZ napisać KAŻDE słowo tego wniosku o zezwolenie po polsku. "
        "WSZYSTKIE nagłówki, akapity, punkty, przypisy i uwagi muszą być po polsku. "
        "NIE włączaj fińskich słów ani zdań do danych wyjściowych. "
        "Fińskie numery aktów prawnych (np. YSL 527/2014, MRL 132/1999) mogą pojawiać się jako "
        "identyfikatory prawne — zawsze dodawaj obok nich polską nazwę ustawy. Fińskie nazwy własne "
        "takie jak nazwy miast, firm i skróty nazw organów (ELY, STUK, Luova, Fingrid, Traficom) "
        "są dopuszczalne wyłącznie jako nazwy własne.\n\n"
    ),
}

_WRITE_INSTRUCTION: dict[str, str] = {
    "FI": "Kirjoita suomeksi seuraavat neljä osiota selkeästi eroteltuna otsikoilla:",
    "EN": "Write the following four sections in English, clearly separated by headings:",
    "SE": "Skriv följande fyra avsnitt på svenska, tydligt åtskilda med rubriker:",
    "DA": "Skriv følgende fire afsnit på dansk, tydeligt adskilt med overskrifter:",
    "NO": "Skriv følgende fire seksjoner på norsk, tydelig atskilt med overskrifter:",
    "PL": "Napisz następujące cztery sekcje po polsku, wyraźnie oddzielone nagłówkami:",
}

_PROMPT_HEADERS: dict[str, dict[str, str]] = {
    "FI": {
        "intro":        "Laadi lupahakemusluonnos seuraavalle hankkeelle:",
        "rag_intro":    "Alla on relevanttia dokumentaatiota (Fingrid, Tukes, Ympäristöministeriö):",
        "kuvaus":       "HANKKEEN KUVAUS",
        "perustelut":   "PERUSTELUT JA TARVE",
        "luvat":        "LUPAMENETTELYJEN KUVAUS",
        "toimenpiteet": "SEURAAVAT TOIMENPITEET",
        "kuvaus_inst":  ("Kirjoita 3–5 kappaleen kuvaus hankkeesta: tarkoitus, tekniset tiedot, "
                         "sijainti, liityntä verkkoon ja ympäristövaikutukset. Mainitse hanketyypille "
                         "tyypilliset tekniset parametrit."),
        "kuvaus_extra": " Ota huomioon annettu sijainti- ja ympäristövaikutustieto.",
        "perustelut_inst": ("Kirjoita 2–3 kappaleen perustelu miksi hanke on tarpeellinen "
                            "(energiajärjestelmän näkökulma, Suomen ilmastotavoitteet, "
                            "aluetaloudelliset vaikutukset)."),
        "luvat_inst":   ("Selitä lyhyesti (1–2 lausetta per lupa) mitä kukin tarvittava lupa "
                         "koskee ja miksi se vaaditaan tälle hankkeelle."),
        "luvat_extra":  " Viittaa erityisesti kohdeviranomaisen {auth} prosesseihin ja vaatimuksiin.",
        "toimenpiteet_first": ("Kunnan rakennusvalvonnan ennakkoneuvottelu + kaavatarkastus — "
                               "Hakija / {kunta}n rakennusvalvonta — 1–2 viikon sisällä"),
        "toimenpiteet_inst": ("Ensimmäinen toimenpide on AINA: \"{first}\".\n"
                              "Listaa sen jälkeen 5 muuta konkreettista askelta aikatauluineen "
                              "(kk tarkkuudella)."),
        "toimenpiteet_vaihe": " Ota huomioon hankkeen nykyinen vaihe: {vaihe}.",
        "viranomainen_ohje":  ("TÄRKEÄÄ: Hakemus osoitetaan viranomaiselle '{auth}'. "
                               "Mukauta hakemuksen sisältö, rakenne ja kieli sen vaatimuksiin sopivaksi. "
                               "Viittaa kyseisen viranomaisen ohjeisiin, lomakkeisiin ja vaatimuksiin."),
    },
    "EN": {
        "intro":        "Write a permit application draft for the following project:",
        "rag_intro":    "Below is relevant documentation (Fingrid, Tukes, Ministry of the Environment):",
        "kuvaus":       "PROJECT DESCRIPTION",
        "perustelut":   "JUSTIFICATION AND NEED",
        "luvat":        "PERMIT PROCEDURE DESCRIPTION",
        "toimenpiteet": "NEXT STEPS",
        "kuvaus_inst":  ("Write a 3–5 paragraph description of the project: purpose, technical details, "
                         "location, grid connection and environmental impacts. Include typical technical "
                         "parameters for this project type."),
        "kuvaus_extra": " Take into account the provided location and environmental impact information.",
        "perustelut_inst": ("Write a 2–3 paragraph justification for why the project is necessary "
                            "(energy system perspective, Finland's climate targets, "
                            "regional economic impacts)."),
        "luvat_inst":   ("Briefly explain (1–2 sentences per permit) what each required permit covers "
                         "and why it is required for this project."),
        "luvat_extra":  " Refer especially to the target authority {auth}'s processes and requirements.",
        "toimenpiteet_first": ("Pre-consultation with municipality building control + zoning review — "
                               "Applicant / {kunta} Building Control — within 1–2 weeks"),
        "toimenpiteet_inst": ("The first step is ALWAYS: \"{first}\".\n"
                              "Then list 5 more concrete steps with timelines (in months)."),
        "toimenpiteet_vaihe": " Take into account the current project phase: {vaihe}.",
        "viranomainen_ohje":  ("IMPORTANT: The application is addressed to authority '{auth}'. "
                               "Adapt the content, structure and language to meet its requirements. "
                               "Refer to that authority's guidelines, forms and requirements."),
    },
    "SE": {
        "intro":        "Skriv ett tillståndsansökningsutkast för följande projekt:",
        "rag_intro":    "Nedan finns relevant dokumentation (Fingrid, Tukes, Miljöministeriet):",
        "kuvaus":       "PROJEKTBESKRIVNING",
        "perustelut":   "MOTIVERING OCH BEHOV",
        "luvat":        "TILLSTÅNDSFÖRFARANDEN BESKRIVNING",
        "toimenpiteet": "NÄSTA STEG",
        "kuvaus_inst":  ("Skriv en beskrivning på 3–5 stycken av projektet: syfte, tekniska detaljer, "
                         "plats, nätanslutning och miljöpåverkan. Inkludera typiska tekniska parametrar "
                         "för denna projekttyp."),
        "kuvaus_extra": " Beakta den angivna plats- och miljöpåverkansinformationen.",
        "perustelut_inst": ("Skriv en 2–3 stycken motivering till varför projektet är nödvändigt "
                            "(energisystemets perspektiv, Finlands klimatmål, "
                            "regionala ekonomiska effekter)."),
        "luvat_inst":   ("Förklara kortfattat (1–2 meningar per tillstånd) vad varje nödvändigt "
                         "tillstånd gäller och varför det krävs för detta projekt."),
        "luvat_extra":  " Hänvisa särskilt till målmyndighetens {auth} processer och krav.",
        "toimenpiteet_first": ("Förkonsultation med kommunens byggnadstillsyn + planläggningsöversyn — "
                               "Sökande / {kunta}s byggnadstillsyn — inom 1–2 veckor"),
        "toimenpiteet_inst": ("Det första steget är ALLTID: \"{first}\".\n"
                              "Lista sedan 5 fler konkreta steg med tidslinjer (i månader)."),
        "toimenpiteet_vaihe": " Beakta projektets nuvarande fas: {vaihe}.",
        "viranomainen_ohje":  ("VIKTIGT: Ansökan riktas till myndigheten '{auth}'. "
                               "Anpassa innehåll, struktur och språk för att uppfylla dess krav. "
                               "Hänvisa till myndighetens riktlinjer, formulär och krav."),
    },
    "DA": {
        "intro":        "Udarbejd et udkast til tilladelsesansøgning for følgende projekt:",
        "rag_intro":    "Nedenfor er relevant dokumentation (Fingrid, Tukes, Miljøministeriet):",
        "kuvaus":       "PROJEKTBESKRIVELSE",
        "perustelut":   "BEGRUNDELSE OG BEHOV",
        "luvat":        "BESKRIVELSE AF TILLADELSES­PROCEDURER",
        "toimenpiteet": "NÆSTE SKRIDT",
        "kuvaus_inst":  ("Skriv en beskrivelse på 3–5 afsnit af projektet: formål, tekniske detaljer, "
                         "placering, nettilslutning og miljøpåvirkninger. Medtag typiske tekniske "
                         "parametre for denne projekttype."),
        "kuvaus_extra": " Tag hensyn til de angivne oplysninger om placering og miljøpåvirkning.",
        "perustelut_inst": ("Skriv en begrundelse på 2–3 afsnit for, hvorfor projektet er nødvendigt "
                            "(energisystemperspektiv, Finlands klimamål, regionale økonomiske virkninger)."),
        "luvat_inst":   ("Forklar kort (1–2 sætninger pr. tilladelse), hvad hver nødvendig tilladelse "
                         "dækker, og hvorfor den kræves for dette projekt."),
        "luvat_extra":  " Henvis især til målmyndighedens {auth} processer og krav.",
        "toimenpiteet_first": ("Forhåndskonsultation med kommunens byggesagsafdeling + planrevision — "
                               "Ansøger / {kunta} byggesagsafdeling — inden for 1–2 uger"),
        "toimenpiteet_inst": ("Det første trin er ALTID: \"{first}\".\n"
                              "Angiv derefter 5 yderligere konkrete trin med tidslinjer (i måneder)."),
        "toimenpiteet_vaihe": " Tag hensyn til projektets nuværende fase: {vaihe}.",
        "viranomainen_ohje":  ("VIGTIGT: Ansøgningen er rettet til myndighed '{auth}'. "
                               "Tilpas indhold, struktur og sprog til myndighedens krav. "
                               "Henvis til myndighedens retningslinjer, formularer og krav."),
    },
    "NO": {
        "intro":        "Utarbeid et utkast til tillatelsessøknad for følgende prosjekt:",
        "rag_intro":    "Nedenfor er relevant dokumentasjon (Fingrid, Tukes, Miljøverndepartementet):",
        "kuvaus":       "PROSJEKTBESKRIVELSE",
        "perustelut":   "BEGRUNNELSE OG BEHOV",
        "luvat":        "BESKRIVELSE AV TILLATELSESPROSEDYRER",
        "toimenpiteet": "NESTE STEG",
        "kuvaus_inst":  ("Skriv en beskrivelse på 3–5 avsnitt av prosjektet: formål, tekniske detaljer, "
                         "plassering, nettilknytning og miljøpåvirkning. Inkluder typiske tekniske "
                         "parametere for denne prosjekttypen."),
        "kuvaus_extra": " Ta hensyn til oppgitt informasjon om plassering og miljøpåvirkning.",
        "perustelut_inst": ("Skriv en begrunnelse på 2–3 avsnitt for hvorfor prosjektet er nødvendig "
                            "(energisystemperspektiv, Finlands klimamål, regionale økonomiske virkninger)."),
        "luvat_inst":   ("Forklar kortfattet (1–2 setninger per tillatelse) hva hver nødvendig tillatelse "
                         "dekker og hvorfor den kreves for dette prosjektet."),
        "luvat_extra":  " Henvis spesielt til målmyndighetens {auth} prosesser og krav.",
        "toimenpiteet_first": ("Forhåndskonsultasjon med kommunens byggesaksavdeling + reguleringsgjennomgang — "
                               "Søker / {kunta} byggesaksavdeling — innen 1–2 uker"),
        "toimenpiteet_inst": ("Det første trinnet er ALLTID: \"{first}\".\n"
                              "List deretter 5 andre konkrete trinn med tidslinjer (i måneder)."),
        "toimenpiteet_vaihe": " Ta hensyn til prosjektets nåværende fase: {vaihe}.",
        "viranomainen_ohje":  ("VIKTIG: Søknaden er adressert til myndighet '{auth}'. "
                               "Tilpass innhold, struktur og språk til myndighetens krav. "
                               "Henvis til myndighetens retningslinjer, skjemaer og krav."),
    },
    "PL": {
        "intro":        "Sporządź projekt wniosku o zezwolenie dla następującego projektu:",
        "rag_intro":    "Poniżej znajduje się odpowiednia dokumentacja (Fingrid, Tukes, Ministerstwo Środoąwiska):",
        "kuvaus":       "OPIS PROJEKTU",
        "perustelut":   "UZASADNIENIE I POTRZEBA",
        "luvat":        "OPIS PROCEDUR ZEZWOLEŃ",
        "toimenpiteet": "NASTĘPNE KROKI",
        "kuvaus_inst":  ("Napisz opis projektu w 3–5 akapitach: cel, dane techniczne, "
                         "lokalizacja, przyłączenie do sieci i wpływ na środowisko. Uwzględnij typowe "
                         "parametry techniczne dla tego typu projektu."),
        "kuvaus_extra": " Uwzględnij podane informacje o lokalizacji i oddziaływaniu na środowisko.",
        "perustelut_inst": ("Napisz uzasadnienie w 2–3 akapitach, dlaczego projekt jest konieczny "
                            "(perspektywa systemu energetycznego, fińskie cele klimatyczne, "
                            "regionalne skutki gospodarcze)."),
        "luvat_inst":   ("Krótko wyjaśnij (1–2 zdania na zezwolenie), czego dotyczy każde wymagane "
                         "zezwolenie i dlaczego jest wymagane dla tego projektu."),
        "luvat_extra":  " Odwołaj się szczególnie do procesów i wymagań organu docelowego {auth}.",
        "toimenpiteet_first": ("Wstępna konsultacja z gminnym wydziałem budowlanym + przegląd planistyczny — "
                               "Wnioskodawca / wydział budowlany {kunta} — w ciągu 1–2 tygodni"),
        "toimenpiteet_inst": ("Pierwszym krokiem jest ZAWSZE: \"{first}\".\n"
                              "Następnie wymiń 5 kolejnych konkretnych kroków z harmonogramem (w miesiącach)."),
        "toimenpiteet_vaihe": " Uwzględnij aktualną fazę projektu: {vaihe}.",
        "viranomainen_ohje":  ("WAŻNE: Wniosek jest skierowany do organu '{auth}'. "
                               "Dostosuj treść, strukturę i język do jego wymagań. "
                               "Odwołaj się do wytycznych, formularzy i wymagań tego organu."),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Käännöstaulukot viranomaisille, luvannimille, lakiviitteille ja liitteille
# ─────────────────────────────────────────────────────────────────────────────

_AUTHORITY_TRANS: dict[str, dict[str, str]] = {
    "Lupa- ja valvontavirasto (Luova)":  {"EN": "Licensing and Supervisory Authority (Luova)", "SE": "Tillstånds- och tillsynsverket (Luova)"},
    "Luova":                              {"EN": "Luova (Licensing Authority)",                  "SE": "Luova (tillståndsmyndighet)"},
    "Kunta / rakennusvalvonta":           {"EN": "Municipality / Building Control",              "SE": "Kommun / byggnadstillsyn"},
    "Kunta / hakija":                     {"EN": "Municipality / Applicant",                     "SE": "Kommun / sökande"},
    "Paikallinen pelastuslaitos":         {"EN": "Local Fire and Rescue Service",                "SE": "Lokal räddningstjänst"},
    "Jakeluverkkoyhtiö / Fingrid Oyj":    {"EN": "Distribution network operator / Fingrid Oyj", "SE": "Distributionsnätbolag / Fingrid Oyj"},
    "Jakeluverkkoyhtiö / Fingrid":        {"EN": "Distribution network operator / Fingrid",      "SE": "Distributionsnätbolag / Fingrid"},
    "Jakeluverkkoyhtiö":                  {"EN": "Distribution network operator",                "SE": "Distributionsnätbolag"},
    "Kunta":                              {"EN": "Municipality",                                  "SE": "Kommun"},
    "ELY-keskus / Luova":                 {"EN": "ELY Centre / Luova",                           "SE": "NTM-centralen / Luova"},
    "ELY-keskus":                         {"EN": "ELY Centre",                                   "SE": "NTM-centralen"},
    "Fingrid Oyj / jakelu":               {"EN": "Fingrid Oyj / distribution",                   "SE": "Fingrid Oyj / distribution"},
    "Fingrid Oyj":                        {"EN": "Fingrid Oyj",                                   "SE": "Fingrid Oyj"},
    "Traficom":                           {"EN": "Traficom (Transport and Communications Agency)", "SE": "Traficom"},
    "Maanomistajat":                      {"EN": "Landowners",                                    "SE": "Markägare"},
    "Valtioneuvosto":                     {"EN": "Council of State",                              "SE": "Statsrådet"},
    "TEM / ELY-keskus":                   {"EN": "Ministry of Economic Affairs / ELY Centre",    "SE": "ANM / NTM-centralen"},
    "STUK":                               {"EN": "STUK (Radiation and Nuclear Safety Authority)", "SE": "STUK (strålnings- och kärnsäkerhetsmyndigheten)"},
    "Puolustusvoimat / PLM":              {"EN": "Finnish Defence Forces / Ministry of Defence",  "SE": "Försvarsmakten / försvarsministeriet"},
    "Valtio / Metsähallitus":             {"EN": "State / Metsähallitus (Forests and Parks Service)", "SE": "Staten / Forststyrelsen"},
    "Luova / kunta":                      {"EN": "Luova / Municipality",                         "SE": "Luova / Kommun"},
    "Kunta / maanomistajat":              {"EN": "Municipality / Landowners",                    "SE": "Kommun / markägare"},
    "Pelastuslaitos":                     {"EN": "Rescue Services / Fire Department",            "SE": "Räddningstjänsten"},
    "AVI / Luova":                        {"EN": "AVI / Luova (Regional State Administrative Agency)", "SE": "RFV / Luova"},
}

_LUPA_TRANS: dict[str, dict[str, str]] = {
    "Ympäristölupa":                                {"EN": "Environmental permit",                              "SE": "Miljötillstånd"},
    "Ympäristölupa (tarvitt.)":                     {"EN": "Environmental permit (if required)",               "SE": "Miljötillstånd (vid behov)"},
    "Ympäristölupa (tarvitt. ≥1 ha)":              {"EN": "Environmental permit (if required, ≥1 ha)",        "SE": "Miljötillstånd (vid behov, ≥1 ha)"},
    "Ympäristölupa (BESS-komponentti)":             {"EN": "Environmental permit (BESS component)",            "SE": "Miljötillstånd (BESS-komponent)"},
    "Rakennuslupa":                                  {"EN": "Building permit",                                  "SE": "Bygglov"},
    "Rakennuslupa tai toimenpidelupa":               {"EN": "Building permit or action permit",                 "SE": "Bygglov eller åtgärdstillstånd"},
    "Rakennuslupa (tuulivoimala)":                   {"EN": "Building permit (wind turbine)",                   "SE": "Bygglov (vindkraftverk)"},
    "Rakennus-/toimenpidelupa (PV + BESS)":          {"EN": "Building/action permit (PV + BESS)",              "SE": "Bygglov/åtgärdstillstånd (PV + BESS)"},
    "Naapurikuuleminen":                             {"EN": "Neighbour consultation",                           "SE": "Grannehörande"},
    "Pelastussuunnitelma / lausunto":                {"EN": "Emergency plan / statement",                       "SE": "Räddningsplan / utlåtande"},
    "Pelastussuunnitelma / lausunto (BESS)":         {"EN": "Emergency plan / statement (BESS)",                "SE": "Räddningsplan / utlåtande (BESS)"},
    "Verkkoliityntäsopimus":                         {"EN": "Grid connection agreement",                        "SE": "Nätanslutningsavtal"},
    "Maa-aineslupa (tarvitt.)":                      {"EN": "Soil extraction permit (if required)",            "SE": "Marktäktstillstånd (vid behov)"},
    "YVA-menettely (≥10 MW / ≥5 voimalaa)":         {"EN": "EIA procedure (≥10 MW / ≥5 turbines)",            "SE": "MKB-förfarande (≥10 MW / ≥5 verk)"},
    "YVA-menettely (kynnyksen ylittyessä)":          {"EN": "EIA procedure (when threshold exceeded)",         "SE": "MKB-förfarande (vid tröskelöverskridning)"},
    "YVA-menettely (tarvitt.)":                      {"EN": "EIA procedure (if required)",                     "SE": "MKB-förfarande (vid behov)"},
    "YVA-menettely":                                 {"EN": "EIA procedure",                                    "SE": "MKB-förfarande"},
    "Osayleiskaava tai asemakaava":                  {"EN": "Local master plan or detailed plan",               "SE": "Delgeneralplan eller detaljplan"},
    "Osayleiskaava / asemakaava":                    {"EN": "Local master plan / detailed plan",                "SE": "Delgeneralplan / detaljplan"},
    "Lentoestevalolupa":                             {"EN": "Aviation obstacle lighting permit",                "SE": "Luftfartshinderlystillstånd"},
    "Lentoestevalolupa (tuulivoimala)":              {"EN": "Aviation obstacle lighting permit (wind turbine)", "SE": "Luftfartshinderlystillstånd (vindkraftverk)"},
    "Maanvuokrasopimukset":                          {"EN": "Land lease agreements",                            "SE": "Arrendeavtal"},
    "Maanvuokra / merialueen käyttöoik.":            {"EN": "Land lease / sea area usage right",               "SE": "Arrendeavtal / havsområdesanvändningsrätt"},
    "Vesilupa":                                      {"EN": "Water permit",                                     "SE": "Vattentillstånd"},
    "Vesilupa (jäähdytysvesi)":                      {"EN": "Water permit (cooling water)",                    "SE": "Vattentillstånd (kylvatten)"},
    "Vesilupa (jäähdytysvesi, tarvitt.)":            {"EN": "Water permit (cooling water, if required)",       "SE": "Vattentillstånd (kylvatten, vid behov)"},
    "Vesilupa (padotus, rakentaminen)":              {"EN": "Water permit (damming, construction)",            "SE": "Vattentillstånd (dämning, byggande)"},
    "Alusliikenteen turvallisuuslupa":               {"EN": "Vessel traffic safety permit",                    "SE": "Fartygsfartstillstånd"},
    "Puolustusvoimien lausunto":                     {"EN": "Defence Forces statement",                        "SE": "Försvarsmaktens utlåtande"},
    "Suunnittelutarveratkaisu (tarvitt.)":           {"EN": "Planning permit (if required)",                   "SE": "Planeringsbehovsbeslut (vid behov)"},
    "Maisema- tai kulttuuriympäristölausunto":        {"EN": "Landscape or cultural environment statement",    "SE": "Landskap- eller kulturmiljöutlåtande"},
    "Periaatepäätös (VN)":                           {"EN": "Decision-in-principle (Council of State)",       "SE": "Principbeslut (statsrådet)"},
    "Rakentamislupa":                                {"EN": "Construction licence",                             "SE": "Byggnadstillstånd"},
    "Rakentamislupa (ydinlaitos)":                   {"EN": "Construction licence (nuclear facility)",         "SE": "Byggnadstillstånd (kärnkraftverk)"},
    "Käyttölupa":                                    {"EN": "Operating licence",                               "SE": "Drifttillstånd"},
    "Käyttölupa (ydinlaitos)":                       {"EN": "Operating licence (nuclear facility)",            "SE": "Drifttillstånd (kärnkraftverk)"},
    "Maankäyttösopimus / kaavoitus":                 {"EN": "Land use agreement / zoning",                    "SE": "Markanvändningsavtal / planläggning"},
    "Maankäyttösopimus":                             {"EN": "Land use agreement",                              "SE": "Markanvändningsavtal"},
    "Kalastuslaki-ilmoitus":                         {"EN": "Fisheries Act notification",                      "SE": "Fiskelagsanmälan"},
}

_LAW_TRANS: dict[str, dict[str, str]] = {
    "YSL 527/2014":                                         {"EN": "Environmental Protection Act (YSL 527/2014)",                          "SE": "Miljöskyddslagen (YSL 527/2014)"},
    "Rakentamislaki 751/2023 / MRL 132/1999":               {"EN": "Building Act / Land Use and Building Act (751/2023 / 132/1999)",       "SE": "Bygglag / Plan- och bygglag (751/2023 / 132/1999)"},
    "Rakentamislaki 751/2023, 44 §":                        {"EN": "Building Act 751/2023, § 44",                                          "SE": "Bygglagen 751/2023, § 44"},
    "Pelastuslaki 379/2011, 15 §":                          {"EN": "Rescue Services Act 379/2011, § 15",                                   "SE": "Räddningslagen 379/2011, § 15"},
    "Sähkömarkkinalaki 588/2013":                           {"EN": "Electricity Market Act (588/2013)",                                    "SE": "Elmarknadslagen (588/2013)"},
    "Maa-aineslaki 555/1981":                               {"EN": "Extractable Land Resources Act (555/1981)",                            "SE": "Marktäktslagen (555/1981)"},
    "YVA-laki 252/2017":                                    {"EN": "EIA Act (252/2017)",                                                   "SE": "MKB-lagen (252/2017)"},
    "YVA-laki 252/2017 (kynnykset ylittyessä)":            {"EN": "EIA Act 252/2017 (when thresholds exceeded)",                          "SE": "MKB-lagen 252/2017 (vid tröskelöverskridning)"},
    "YVA-laki 252/2017 (≥50 ha hankkeet)":                 {"EN": "EIA Act 252/2017 (≥50 ha projects)",                                   "SE": "MKB-lagen 252/2017 (≥50 ha projekt)"},
    "MRL 132/1999 § 77a":                                   {"EN": "Land Use and Building Act 132/1999, § 77a",                            "SE": "Plan- och bygglagen 132/1999, § 77a"},
    "MRL 132/1999 § 137":                                   {"EN": "Land Use and Building Act 132/1999, § 137",                            "SE": "Plan- och bygglagen 132/1999, § 137"},
    "MRL 197 §":                                            {"EN": "Land Use and Building Act, § 197",                                     "SE": "Plan- och bygglagen, § 197"},
    "MRL 132/1999 § 91a":                                   {"EN": "Land Use and Building Act 132/1999, § 91a",                            "SE": "Plan- och bygglagen 132/1999, § 91a"},
    "MRL 132/1999 § 9":                                     {"EN": "Land Use and Building Act 132/1999, § 9",                              "SE": "Plan- och bygglagen 132/1999, § 9"},
    "MRL 132/1999":                                         {"EN": "Land Use and Building Act (132/1999)",                                 "SE": "Plan- och bygglagen (132/1999)"},
    "Ilmailulaki 864/2014":                                 {"EN": "Aviation Act (864/2014)",                                              "SE": "Luftfartslagen (864/2014)"},
    "Maakaari 540/1995":                                    {"EN": "Code of Real Estate (540/1995)",                                       "SE": "Jordabalken (540/1995)"},
    "Vesilaki 587/2011":                                    {"EN": "Water Act (587/2011)",                                                 "SE": "Vattenlagen (587/2011)"},
    "Vesilaki 587/2011 § 3:2":                              {"EN": "Water Act 587/2011, § 3:2",                                            "SE": "Vattenlagen 587/2011, § 3:2"},
    "Merilaki 674/1994":                                    {"EN": "Maritime Act (674/1994)",                                              "SE": "Sjölagen (674/1994)"},
    "Merenkulkulaki 1672/2009":                             {"EN": "Maritime Navigation Act (1672/2009)",                                  "SE": "Sjöfartslagen (1672/2009)"},
    "Laki alueiden käytöstä":                               {"EN": "Act on Land Use",                                                      "SE": "Lagen om områdesanvändning"},
    "Rakentamislaki 751/2023 / MRL 132/1999 § 125–126":    {"EN": "Building Act / Land Use and Building Act (751/2023 / 132/1999 §§ 125–126)", "SE": "Bygglag / Plan- och bygglag (751/2023 / 132/1999 §§ 125–126)"},
    "Rakentamislaki 751/2023 / MRL 132/1999 § 126":         {"EN": "Building Act / Land Use and Building Act (751/2023 / 132/1999, § 126)", "SE": "Bygglag / Plan- och bygglag (751/2023 / 132/1999, § 126)"},
    "Ydinenergialaki 990/1987 § 11":                        {"EN": "Nuclear Energy Act 990/1987, § 11",                                   "SE": "Kärnenergilagen 990/1987, § 11"},
    "YEL 990/1987 § 18":                                    {"EN": "Nuclear Energy Act 990/1987, § 18",                                   "SE": "Kärnenergilagen 990/1987, § 18"},
    "YEL 990/1987 § 20":                                    {"EN": "Nuclear Energy Act 990/1987, § 20",                                   "SE": "Kärnenergilagen 990/1987, § 20"},
    "Kalastuslaki 379/2015":                                {"EN": "Fisheries Act (379/2015)",                                             "SE": "Fiskelagen (379/2015)"},
    "Säteilylaki 859/2018":                                 {"EN": "Radiation Act (859/2018)",                                             "SE": "Strålningslagen (859/2018)"},
    "Kemikaaliturvallisuuslaki 390/2005":                   {"EN": "Chemicals Safety Act (390/2005)",                                      "SE": "Kemikaliesäkerhetslagen (390/2005)"},
    "Kemikaaliturvallisuuslaki 390/2005 (BESS)":           {"EN": "Chemicals Safety Act 390/2005 (BESS)",                                  "SE": "Kemikaliesäkerhetslagen 390/2005 (BESS)"},
    "Luonnonsuojelulaki 9/2023":                            {"EN": "Nature Conservation Act (9/2023)",                                     "SE": "Naturvårdslagen (9/2023)"},
    "Maantielaki 503/2005 (tiealueet)":                     {"EN": "Highways Act 503/2005 (road areas)",                                   "SE": "Väglagen 503/2005 (vägområden)"},
    "Patoturvallisuuslaki 494/2009":                        {"EN": "Dam Safety Act (494/2009)",                                            "SE": "Damsäkerhetslagen (494/2009)"},
}

_LIITE_TRANS: dict[str, dict[str, str]] = {
    "Sijaintikartta (M 1:20 000 tai laajempi)":             {"EN": "Location map (scale 1:20,000 or wider)",                        "SE": "Lägeskartta (skala 1:20 000 eller vidare)"},
    "Sijaintikartta / projektikartta (M 1:20 000 tai laajempi)": {"EN": "Location map / project map (scale 1:20,000 or wider)",     "SE": "Lägeskartta / projektkarta (skala 1:20 000 eller vidare)"},
    "Maankäyttöselvitys PDF (NCE Energy)":                  {"EN": "Land Use Survey PDF (NCE Energy)",                             "SE": "Markanvändningsutredning PDF (NCE Energy)"},
    "Asemapiirustus ja pohjakartta (M 1:500)":              {"EN": "Site plan and base map (1:500)",                               "SE": "Situationsplan och baskarta (1:500)"},
    "Asemapiirustus ja pohjakartta (M 1:500 tai 1:1000)":   {"EN": "Site plan and base map (1:500 or 1:1000)",                    "SE": "Situationsplan och baskarta (1:500 eller 1:1000)"},
    "Rakennesuunnitelma (akkukontti + perustukset)":         {"EN": "Structural plan (battery container + foundations)",            "SE": "Konstruktionsplan (battericontainer + fundament)"},
    "Paloturvallisuusselvitys (NFPA 855 / EN-standardit)":  {"EN": "Fire safety report (NFPA 855 / EN standards)",                 "SE": "Brandsäkerhetsutredning (NFPA 855 / EN-standarder)"},
    "Sammutusvesien keräyssuunnitelma":                      {"EN": "Fire suppression water collection plan",                       "SE": "Plan för uppsamling av brandsläckningsvatten"},
    "Sammutusvesien keräyssuunnitelma (BESS)":               {"EN": "Fire suppression water collection plan (BESS)",                "SE": "Plan för uppsamling av brandsläckningsvatten (BESS)"},
    "Sammutusvesien keräyssuunnitelma (BESS-komponentti)":   {"EN": "Fire suppression water collection plan (BESS component)",     "SE": "Plan för uppsamling av brandsläckningsvatten (BESS-komponent)"},
    "Ympäristöriskiarvio (pohjavesi, maaperä)":              {"EN": "Environmental risk assessment (groundwater, soil)",            "SE": "Miljöriskbedömning (grundvatten, mark)"},
    "Sähköliityntäsuunnitelma (verkkoyhtiön hyväksymä)":     {"EN": "Electrical connection plan (approved by grid operator)",      "SE": "Elanslutningsplan (godkänd av nätbolaget)"},
    "Meluselvitys (jos lähellä asutusta)":                   {"EN": "Noise study (if near residential areas)",                     "SE": "Bullerutredning (om nära bebyggelse)"},
    "Liikenneyhteydet ja huoltotie":                         {"EN": "Traffic connections and maintenance road",                     "SE": "Trafikförbindelser och underhållsväg"},
    "Hakijan oikeushenkilön rekisteriote":                   {"EN": "Applicant's legal entity registration extract",               "SE": "Sökandens juridiska enhets registerutdrag"},
    "Hakijan rekisteriote":                                  {"EN": "Applicant's registration extract",                             "SE": "Sökandens registerutdrag"},
    "Valtakirja (jos asiamies edustaa)":                     {"EN": "Power of attorney (if agent represents)",                     "SE": "Fullmakt (om ombud företräder)"},
    "YVA-ohjelma ja YVA-selostus (ELY:n hyväksymä)":        {"EN": "EIA programme and EIA report (ELY Centre approved)",          "SE": "MKB-program och MKB-rapport (NTM-centralen godkänd)"},
    "YVA-ohjelma ja YVA-selostus":                           {"EN": "EIA programme and EIA report",                                "SE": "MKB-program och MKB-rapport"},
    "YVA-ohjelma ja -selostus":                              {"EN": "EIA programme and report",                                    "SE": "MKB-program och rapport"},
    "YVA-ohjelma ja -selostus (tuulivoiman osalta)":         {"EN": "EIA programme and report (wind power component)",             "SE": "MKB-program och rapport (vindkraftsdelen)"},
    "Meluselvitys (ETSU-R-97 tai IEC 61400-11)":             {"EN": "Noise study (ETSU-R-97 or IEC 61400-11)",                    "SE": "Bullerutredning (ETSU-R-97 eller IEC 61400-11)"},
    "Meluselvitys (tuulivoimalakomponentti)":                {"EN": "Noise study (wind turbine component)",                        "SE": "Bullerutredning (vindkraftverkskomponent)"},
    "Meluselvitys (ilma- ja vedenalainen melu)":             {"EN": "Noise study (airborne and underwater noise)",                 "SE": "Bullerutredning (luftburet och undervattensbuller)"},
    "Varjostusmallinnusraportti":                            {"EN": "Shadow flicker modelling report",                             "SE": "Skuggningsmodelleringsrapport"},
    "Varjostus- ja näkyvyysanalyysi":                        {"EN": "Shadow flicker and visibility analysis",                      "SE": "Skuggnings- och synlighetsanalys"},
    "Varjostus- ja häikäisyanalyysi (naapurikiinteistöt)":   {"EN": "Shadow and glare analysis (neighbouring properties)",        "SE": "Skuggnings- och bländningsanalys (grannfastigheter)"},
    "Linnustoselvitys (pesimä- ja muuttolinnut)":            {"EN": "Bird survey (breeding and migratory birds)",                  "SE": "Fågelinventering (häcknings- och sträckfåglar)"},
    "Lepakoiden lentoaktiviteettiselvitys":                  {"EN": "Bat flight activity survey",                                  "SE": "Fladdermössens flygaktivitetsutredning"},
    "Linnusto- ja lepakoiden aktiviteettiselvitys":          {"EN": "Bird and bat activity survey",                               "SE": "Fågel- och fladdermösaktivitetsinventering"},
    "Linnusto- ja lepakkoselvitys merialueella":             {"EN": "Bird and bat survey in sea area",                            "SE": "Fågel- och fladdermusinventering i havsområdet"},
    "Maisema- ja näkyvyysanalyysi (valokuvasovitteet)":      {"EN": "Landscape and visibility analysis (photomontages)",          "SE": "Landskap- och synlighetsanalys (fotomontage)"},
    "Maisema- ja näkyvyysanalyysi":                          {"EN": "Landscape and visibility analysis",                          "SE": "Landskap- och synlighetsanalys"},
    "Rakennussuunnitelmat (perustukset, tiet, kaapelointi)": {"EN": "Construction plans (foundations, roads, cabling)",           "SE": "Byggplaner (fundament, vägar, kablering)"},
    "Rakennussuunnitelmat (pato, voimalaitosrakennus)":      {"EN": "Construction plans (dam, power plant building)",             "SE": "Byggplaner (damm, kraftverksbyggnad)"},
    "Verkkoliityntälaskelma (tehonlaatuanalyysi)":           {"EN": "Grid connection calculation (power quality analysis)",       "SE": "Nätanslutningsberäkning (elkvalitetsanalys)"},
    "Verkkoliityntälaskelma ja muuntajamitoitus":            {"EN": "Grid connection calculation and transformer sizing",         "SE": "Nätanslutningsberäkning och transformatordimensionering"},
    "Verkkoliityntälaskelma (SMR + BESS yhdistetty)":        {"EN": "Grid connection calculation (SMR + BESS combined)",         "SE": "Nätanslutningsberäkning (SMR + BESS kombinerat)"},
    "Verkkoliityntälaskelma":                                {"EN": "Grid connection calculation",                               "SE": "Nätanslutningsberäkning"},
    "Maanomistaja- ja sopimustiedot":                        {"EN": "Landowner and agreement information",                       "SE": "Markägare- och avtalsuppgifter"},
    "Maanomistaja- ja vesioikeusasiakirjat":                 {"EN": "Landowner and water rights documents",                      "SE": "Markägare- och vattendokument"},
    "Lentoestekartoitus (Traficom/Finavia)":                 {"EN": "Aviation obstacle survey (Traficom/Finavia)",               "SE": "Luftfartshinderkartläggning (Traficom/Finavia)"},
    "Meriekologinen vaikutusarviointi (Natura tarvittaessa)":{"EN": "Marine ecological impact assessment (Natura if required)",  "SE": "Marinekologisk konsekvensutredning (Natura vid behov)"},
    "Merikaapelireittiselvitys":                             {"EN": "Submarine cable route survey",                              "SE": "Havskabelruttutredning"},
    "Pohjasedimenttitutkimus (geotekninen)":                 {"EN": "Seabed sediment study (geotechnical)",                      "SE": "Bottensedimentundersökning (geoteknisk)"},
    "Meriliikenteen turvallisuusarviointi":                  {"EN": "Maritime traffic safety assessment",                        "SE": "Säkerhetsbedömning av sjötrafik"},
    "Puolustusvoimien tutkavaikutusarviointi":               {"EN": "Defence Forces radar impact assessment",                    "SE": "Försvarsmaktens radarpåverkansutredning"},
    "Paneelijärjestely- ja rakennesuunnitelma":              {"EN": "Panel layout and structural plan",                          "SE": "Panellayout och konstruktionsplan"},
    "Verkkoliityntäsuunnitelma (invertteri, muuntaja)":      {"EN": "Grid connection plan (inverter, transformer)",              "SE": "Nätanslutningsplan (växelriktare, transformator)"},
    "Maaperä- ja hulevesiselvitys (suuri-alainen asennus)":  {"EN": "Soil and stormwater study (large-scale installation)",     "SE": "Mark- och dagvattenutredning (storskalig installation)"},
    "Luontoselvitys (ekologiset yhteydet, mahdollinen Natura)":{"EN": "Nature survey (ecological corridors, possible Natura)",  "SE": "Naturinventering (ekologiska förbindelser, möjlig Natura)"},
    "Asukasosallistumisen asiakirjat (suunnittelutarveratkaisussa)":{"EN": "Public participation documents (planning permit procedure)", "SE": "Medborgardeltagandedokument (planeringsbehovsbeslut)"},
    "Alustava turvallisuusseloste (STUK YVL A.1 mukainen)": {"EN": "Preliminary safety report (per STUK YVL A.1)",             "SE": "Preliminär säkerhetsredogörelse (enl. STUK YVL A.1)"},
    "Ydinmateriaalivalvontasuunnitelma (IAEA SQ-protokolla)":{"EN": "Nuclear materials safeguards plan (IAEA SQ protocol)",     "SE": "Kärnmaterialövervakningsplan (IAEA SQ-protokoll)"},
    "Säteilyturvallisuusanalyysi (YVL C.1)":                {"EN": "Radiation safety analysis (YVL C.1)",                      "SE": "Strålsäkerhetsanalys (YVL C.1)"},
    "Turvallisuussuunnittelun periaatteet (YVL B.1)":        {"EN": "Safety design principles (YVL B.1)",                       "SE": "Säkerhetsdesignprinciper (YVL B.1)"},
    "Hätäjärjestelmien ja -menettelyjen kuvaus":             {"EN": "Description of emergency systems and procedures",          "SE": "Beskrivning av nödsystem och -förfaranden"},
    "Polttoainekierto- ja ydinjätehuoltosuunnitelma":        {"EN": "Fuel cycle and nuclear waste management plan",             "SE": "Bränslecykel- och kärnavfallshanteringsplan"},
    "Geotekninen perusselvitys (seismisyys, hydrogeologia)": {"EN": "Geotechnical baseline study (seismicity, hydrogeology)",   "SE": "Geoteknisk grundutredning (seismicitet, hydrogeologi)"},
    "Jäähdytysveden saatavuus- ja ympäristöarviointi":       {"EN": "Cooling water availability and environmental assessment",  "SE": "Kylvattentillgång och miljöbedömning"},
    "Jäähdytysvesitarve- ja ympäristöarviointi":             {"EN": "Cooling water demand and environmental assessment",        "SE": "Kylvattenbehov och miljöbedömning"},
    "Sosioekonominen vaikutusarviointi":                     {"EN": "Socioeconomic impact assessment",                          "SE": "Socioekonomisk konsekvensutredning"},
    "Kansainväliset referensssilaitosvertailut (IAEA)":      {"EN": "International reference plant comparisons (IAEA)",         "SE": "Internationella referensanläggningsjämförelser (IAEA)"},
    "Hydraulinen mitoitusraportti (virtaama, putouskorkeus)":{"EN": "Hydraulic design report (flow rate, head)",               "SE": "Hydraulisk dimensioneringsrapport (flöde, fallhöjd)"},
    "Geotekninen pato- ja pohjarakenneselvitys":             {"EN": "Geotechnical dam and foundation study",                    "SE": "Geoteknisk dam- och grundläggningsutredning"},
    "Vesistövaikutusten arviointi (tulva, kuivuus, vedenlaatu)":{"EN": "Watercourse impact assessment (flooding, drought, water quality)", "SE": "Vattendragspåverkansutredning (översvämning, torka, vattenkvalitet)"},
    "Ekologinen virtaamaselvitys (kalat, pohjaeläimet)":     {"EN": "Ecological flow study (fish, benthic fauna)",              "SE": "Ekologisk flödesutredning (fisk, bottendjur)"},
    "Kalaston vaellusesteiden ja kalateiden suunnitelma":    {"EN": "Fish migration barrier and fish pass plan",                "SE": "Plan för fiskvandringsbarriärer och fiskvägar"},
    "Padon turvallisuussuunnitelma (PATL 494/2009)":         {"EN": "Dam safety plan (Dam Safety Act 494/2009)",                "SE": "Damsäkerhetsplan (Damsäkerhetslagen 494/2009)"},
    "Hätätilannesuunnitelma (padotusriskit)":                {"EN": "Emergency plan (dam failure risks)",                       "SE": "Nödlägesplan (dammrasterrisker)"},
    "BESS-paloturvallisuusselvitys (NFPA 855)":              {"EN": "BESS fire safety report (NFPA 855)",                       "SE": "BESS brandsäkerhetsrapport (NFPA 855)"},
    "BESS-paloturvallisuusselvitys (NFPA 855 / EN-standardit)":{"EN": "BESS fire safety report (NFPA 855 / EN standards)",     "SE": "BESS brandsäkerhetsrapport (NFPA 855 / EN-standarder)"},
    "Integroitu verkkoliityntäsuunnitelma (tuuli + PV + BESS)":{"EN": "Integrated grid connection plan (wind + PV + BESS)",   "SE": "Integrerad nätanslutningsplan (vind + PV + BESS)"},
    "Integroitu energiavarastosuunnitelma (SMR + BESS-mitoitus)":{"EN": "Integrated energy storage plan (SMR + BESS sizing)",  "SE": "Integrerad energilagringsplan (SMR + BESS-dimensionering)"},
    "Energiavarastomitoitusraportti (kapasiteetti, teho, kesto)":{"EN": "Energy storage sizing report (capacity, power, duration)", "SE": "Energilagringsrapport (kapacitet, effekt, varaktighet)"},
    "Hakijan taloudellinen tilanne (tilinpäätös, 2 viimeisintä vuotta)":{"EN": "Applicant's financial status (financial statements, last 2 years)", "SE": "Sökandens ekonomiska ställning (bokslut, 2 senaste år)"},
    "Projektisuunnitelma (T&K-kuvaus, tavoitteet, metodologia)":{"EN": "Project plan (R&D description, objectives, methodology)", "SE": "Projektplan (FoU-beskrivning, mål, metodologi)"},
    "Budjettilaskelmat ja rahoitussuunnitelma":               {"EN": "Budget calculations and financing plan",                  "SE": "Budgetkalkyl och finansieringsplan"},
    "Tiimikuvaus (ansioluettelot, osaamisprofiilit)":         {"EN": "Team description (CVs, competency profiles)",             "SE": "Teambeskrivning (meritförteckningar, kompetensprofiler)"},
    "Riskiarviointi ja mitigaatiosuunnitelma":                {"EN": "Risk assessment and mitigation plan",                     "SE": "Riskbedömning och mitigeringsplan"},
    "Referenssit ja aiempi T&K-toiminta":                    {"EN": "References and previous R&D activities",                  "SE": "Referenser och tidigare FoU-verksamhet"},
    "IPR-suunnitelma (immateriaalioikeuksien hallinta)":      {"EN": "IPR plan (intellectual property rights management)",      "SE": "IPR-plan (immaterialrättshantering)"},
}


def _t_str(lang: str, fi: str, trans_dict: dict) -> str:
    """Käännä merkkijono käyttäen annettua käännöstaulukkoa. Fallback → FI."""
    if lang == "FI":
        return fi
    d = trans_dict.get(fi, {})
    return d.get(lang, fi)

def _t_auth(lang: str, fi: str) -> str:
    return _t_str(lang, fi, _AUTHORITY_TRANS)

def _t_lupa(lang: str, fi: str) -> str:
    return _t_str(lang, fi, _LUPA_TRANS)

def _t_law(lang: str, fi: str) -> str:
    return _t_str(lang, fi, _LAW_TRANS)

def _t_liite(lang: str, fi: str) -> str:
    return _t_str(lang, fi, _LIITE_TRANS)


# ─────────────────────────────────────────────────────────────────────────────
# PDF-käännöstaulukko (UI-tekstit, ei AI-sisältö)
# ─────────────────────────────────────────────────────────────────────────────

_PDF_STRINGS: dict[str, dict[str, str]] = {
    "FI": {
        "sub_title":       "Rakennuslupahakemusluonnos",
        "esiselvitys_sub": ("Esiselvitys- ja ennakkoneuvottelumateriaali — "
                            "Valmisteltu rakennusvalvonnan ennakkoneuvottelua varten"),
        "disclaimer_h":    "AI-LUONNOS — VAATII ASIANTUNTIJATARKISTUKSEN",
        "disclaimer_b":    ("Tämä asiakirja on tekoälyavusteisesti laadittu luonnos. Se ei ole juridisesti "
                            "sitova eikä korvaa pätevän lupa-asiantuntijan tai lakimiehen neuvoja. Ennen "
                            "hakemuksen jättämistä asiakirja on tarkistutettava alan ammattilaisella."),
        "m_hakija":        "Hakija",       "m_ytunnus":    "Y-tunnus",
        "m_hanketyyppi":   "Hanketyyppi",  "m_teho":       "Teho / kapasiteetti",
        "m_kunta":         "Sijaintikunta","m_kt":         "Kiinteistötunnus",
        "m_maa":           "Maa",
        "m_laadittu":      "Laadittu",     "m_laatinut_lbl": "Laatinut",
        "m_laatinut":      "NCE Energy Permit AI (tekoälyavusteinen)",
        "sec1": "1. Hankkeen kuvaus",             "sec2": "2. Perustelut ja tarve",
        "sec3": "3. Tarvittavat luvat ja viranomaiset", "sec4": "4. Lakiviitteet",
        "sec5": "5. Liiteluettelo",               "sec6": "6. Seuraavat toimenpiteet",
        "liitteet_note":   ("Seuraavat liitteet on toimitettava hakemuksen yhteydessä. "
                            "Merkitse ☐-ruutuun kun liite on valmis."),
        "lahteet_h":       "Lähteet ja tietolähteet",
        "lahteet_b":       "Tämä luonnos on laadittu hyödyntäen seuraavia viranomaisdokumentteja:",
        "yhteystiedot_h":  "Hakijan yhteystiedot",
        "yht_hakija":      "Hakija",     "yht_ytunnus":   "Y-tunnus",
        "yht_osoite":      "Osoite",     "yht_lisatietoja": "Lisätietoja",
        "footer":          ("NCE Energy Permit AI  ·  ncenergy.fi  ·  info@ncenergy.fi  "
                            "·  AI-luonnos — vaatii asiantuntijatarkistuksen"),
        "th_lupa":  "Lupa / ilmoitus", "th_viran": "Viranomainen", "th_laki": "Lakiperuste",
        "th_nro":   "Nro",  "th_liite": "Liite",  "th_tila": "Tila",
        "liite_toimitettu": "[ ] Toimitettu",
        "toim_nro": "Nro", "toim_toimenpide": "Toimenpide",
        "toim_vastuutaho": "Vastuutaho", "toim_aikataulu": "Aikataulu",
        "hdr_draft": "lupahakemusluonnos", "hdr_right": "ncenergy.fi  |  AI-luonnos",
        "ftr_ai":    "AI-luonnos — vaatii tarkistuksen", "ftr_sivu": "Sivu",
        "bf_title": "T&K-rahoitushakemus — luonnos",
        "bf_kotipaikka": "Kotipaikka", "bf_vaihe": "Vaihe", "bf_tk_kuvaus": "T&K-kuvaus",
        "esiselvitys_p":   ("Hanke on esiselvitysvaiheessa. Lopulliset tekniset mitoitukset, "
                            "sijaintisuunnitelmat ja ympäristövaikutusten arvioinnit tarkentuvat "
                            "jatkosuunnittelun myötä."),
        "bess_pintaala":   "Laitosalueen arvioitu pinta-ala on 0,4–0,6 ha.",
        "mks_viittaus":    ("Hankealueen maankäyttö on selvitetty NCE Energyn maankäyttöselvityksessä "
                            "(ks. Liite 0b: Maankäyttöselvitys PDF). Selvitys sisältää kiinteistötiedot, "
                            "kaavatilanteen, suojelualueet sekä pohjavesialuetiedot."),
        "kaava_BESS":      ("<b>Kaavatilanne (kriittisin esiselvityskohta):</b> BESS-hankkeen sijoituspaikan "
                            "kaavatilanne on selvitettävä ensimmäisenä. Useimmissa kunnissa akkuenergiavaraston "
                            "sijoittaminen edellyttää asemakaavaa tai suunnittelutarveratkaisua. Kaavatilanne "
                            "vaikuttaa eniten lupaprosessin kokonaiskestoon — rakennusvalvonnan "
                            "ennakkoneuvottelu ensitoimenpiteenä."),
        "kaava_tuuli":     ("<b>Kaavatilanne ja YVA-tarve:</b> Tuulivoimahanke edellyttää lähes aina "
                            "osayleiskaavaa tai asemakaavaa (MRL 132/1999, 77a §). YVA-menettely "
                            "(YVA-laki 252/2017) on pakollinen ≥10 MW tai ≥5 voimalan hankkeille — "
                            "kaava- ja YVA-prosessit kulkevat usein rinnakkain ja kestävät yhteensä "
                            "3–6 vuotta. Kaavatilanne selvitetään ensimmäisenä ennen muita lupia."),
        "kaava_SMR":       ("<b>STUK pre-licensing (kriittisin ensimmäinen vaihe):</b> Ydinlaitoshankkeessa "
                            "valtioneuvoston periaatepäätös (ydinenergialaki 990/1987, 11 §) ja STUK:n "
                            "ennakkolupamenettely ovat pakollisia ennen kaikkia muita lupia. STUK:n "
                            "YVL-ohjeiden mukainen turvallisuusseloste käynnistää prosessin. Kaavatilanne "
                            "selvitetään rinnalla, mutta ydinturvallisuusmenettely on hallitseva tekijä."),
        "kaava_aurinkovoima": ("<b>Toimenpidelupa vai rakennuslupa — ja kaavatilanne:</b> Pienimuotoiselle "
                            "aurinkopuistolle (alle noin 1 ha) riittää usein toimenpidelupa rakennusluvan "
                            "sijaan (Rakentamislaki 751/2023 / MRL 132/1999, 126 §). YVA-menettely ei koske "
                            "alle 50 ha hankkeita. Kaavatilanne on silti tarkistettava — asemakaava-alueen "
                            "ulkopuolella voidaan tarvita suunnittelutarveratkaisu."),
        "kaava_generic":   ("<b>Kaavatilanne:</b> Hankkeen sijoituspaikan voimassa oleva kaavatilanne on "
                            "tarkistettava rakennusvalvonnan ennakkoneuvottelussa ennen lupahakemuksen "
                            "jättämistä. Kaavatilanne vaikuttaa suoraan lupaprosessin kestoon ja "
                            "vaatimuksiin — rakentaminen edellyttää usein asemakaavaa tai sen muutosta "
                            "taikka suunnittelutarveratkaisua."),
    },
    "EN": {
        "sub_title":       "Building Permit Application Draft",
        "esiselvitys_sub": ("Pre-study and Pre-consultation Material — "
                            "Prepared for building permit pre-consultation"),
        "disclaimer_h":    "AI DRAFT — REQUIRES EXPERT REVIEW",
        "disclaimer_b":    ("This document is an AI-assisted draft. It is not legally binding and does not "
                            "replace the advice of a qualified permit expert or lawyer. Before submitting "
                            "the application, this document must be reviewed by a professional."),
        "m_hakija":        "Applicant",      "m_ytunnus":    "Business ID",
        "m_hanketyyppi":   "Project Type",   "m_teho":       "Capacity / Power",
        "m_kunta":         "Municipality",   "m_kt":         "Property ID",
        "m_maa":           "Country",
        "m_laadittu":      "Prepared",       "m_laatinut_lbl": "Prepared by",
        "m_laatinut":      "NCE Energy Permit AI (AI-assisted)",
        "sec1": "1. Project Description",          "sec2": "2. Justification and Need",
        "sec3": "3. Required Permits and Authorities", "sec4": "4. Legal References",
        "sec5": "5. Appendix List",                "sec6": "6. Next Steps",
        "liitteet_note":   ("The following appendices must be submitted with the application. "
                            "Mark the checkbox when the appendix is ready."),
        "lahteet_h":       "Sources and References",
        "lahteet_b":       "This draft was prepared using the following official documents:",
        "yhteystiedot_h":  "Applicant Contact Details",
        "yht_hakija":      "Applicant",  "yht_ytunnus":    "Business ID",
        "yht_osoite":      "Address",    "yht_lisatietoja": "Further information",
        "footer":          ("NCE Energy Permit AI  ·  ncenergy.fi  ·  info@ncenergy.fi  "
                            "·  AI draft — requires expert review"),
        "th_lupa":  "Permit / Notification", "th_viran": "Authority", "th_laki": "Legal Basis",
        "th_nro":   "No.", "th_liite": "Appendix", "th_tila": "Status",
        "liite_toimitettu": "[ ] Submitted",
        "toim_nro": "No.", "toim_toimenpide": "Action",
        "toim_vastuutaho": "Responsible", "toim_aikataulu": "Timeline",
        "hdr_draft": "permit application draft", "hdr_right": "ncenergy.fi  |  AI draft",
        "ftr_ai":    "AI draft — requires review", "ftr_sivu": "Page",
        "bf_title": "R&D Funding Application Draft",
        "bf_kotipaikka": "Location", "bf_vaihe": "Phase", "bf_tk_kuvaus": "R&D Description",
        "esiselvitys_p":   ("The project is in the pre-study phase. Final technical specifications, "
                            "site plans and environmental impact assessments will be refined "
                            "during further planning."),
        "bess_pintaala":   "The estimated site area is 0.4–0.6 ha.",
        "mks_viittaus":    ("The land use of the project area has been investigated in NCE Energy's "
                            "land use report (see Appendix 0b: Land Use Report PDF). The report includes "
                            "property information, zoning status, protected areas and groundwater area data."),
        "kaava_BESS":      ("<b>Zoning status (most critical pre-study item):</b> The zoning status of the "
                            "BESS project site must be determined first. In most municipalities, siting a "
                            "battery energy storage system requires a detailed plan or a planning permit. "
                            "Zoning status has the greatest impact on the total duration of the permit "
                            "process — pre-consultation with the building authority is the first step."),
        "kaava_tuuli":     ("<b>Zoning status and EIA requirement:</b> A wind power project almost always "
                            "requires a local master plan or detailed plan (MRL 132/1999, 77a §). The EIA "
                            "procedure (YVA-laki 252/2017) is mandatory for projects ≥10 MW or ≥5 turbines "
                            "— zoning and EIA processes often run in parallel, taking 3–6 years combined. "
                            "Zoning is resolved first before other permits."),
        "kaava_SMR":       ("<b>STUK pre-licensing (most critical first step):</b> For a nuclear facility "
                            "project, the Council of State's decision-in-principle (Nuclear Energy Act "
                            "990/1987, § 11) and STUK's pre-licensing procedure are mandatory before all "
                            "other permits. STUK's YVL-guideline safety report initiates the process. "
                            "Zoning is addressed in parallel but the nuclear safety procedure is dominant."),
        "kaava_aurinkovoima": ("<b>Action permit vs. building permit — and zoning:</b> For a small-scale "
                            "solar park (below approx. 1 ha), an action permit often suffices instead of a "
                            "full building permit (Construction Act 751/2023 / MRL 132/1999, § 126). EIA is "
                            "not required for projects under 50 ha. Zoning must still be checked — a "
                            "planning permit may be needed outside detailed plan areas."),
        "kaava_generic":   ("<b>Zoning status:</b> The current zoning status of the project site must be "
                            "verified in a pre-consultation meeting with the building authority before the "
                            "permit application is submitted. Zoning directly affects the duration and "
                            "requirements of the permit process — construction often requires a detailed "
                            "plan, an amendment to one, or a planning permit."),
    },
    "SE": {
        "sub_title":       "Bygglovsansökan — utkast",
        "esiselvitys_sub": ("Förundersökning och förkonsultationsmaterial — "
                            "Utarbetat för förkonsultation med byggnadstillsyn"),
        "disclaimer_h":    "AI-UTKAST — KRÄVER EXPERTGRANSKNING",
        "disclaimer_b":    ("Detta dokument är ett AI-assisterat utkast. Det är inte juridiskt bindande och "
                            "ersätter inte råd från en kvalificerad tillståndsexpert eller jurist. Innan "
                            "ansökan lämnas in måste dokumentet granskas av en fackman."),
        "m_hakija":        "Sökande",          "m_ytunnus":    "FO-nummer",
        "m_hanketyyppi":   "Projekttyp",       "m_teho":       "Kapacitet / Effekt",
        "m_kunta":         "Kommun",           "m_kt":         "Fastighetsbeteckning",
        "m_maa":           "Land",
        "m_laadittu":      "Upprättat",        "m_laatinut_lbl": "Upprättat av",
        "m_laatinut":      "NCE Energy Permit AI (AI-assisterat)",
        "sec1": "1. Projektbeskrivning",             "sec2": "2. Motivering och behov",
        "sec3": "3. Nödvändiga tillstånd och myndigheter", "sec4": "4. Laghänvisningar",
        "sec5": "5. Bilagor",                        "sec6": "6. Nästa steg",
        "liitteet_note":   ("Följande bilagor ska lämnas in tillsammans med ansökan. "
                            "Markera rutan när bilagan är klar."),
        "lahteet_h":       "Källor och referenser",
        "lahteet_b":       "Detta utkast har upprättats med hjälp av följande officiella dokument:",
        "yhteystiedot_h":  "Sökandens kontaktuppgifter",
        "yht_hakija":      "Sökande",   "yht_ytunnus":    "FO-nummer",
        "yht_osoite":      "Adress",    "yht_lisatietoja": "Mer information",
        "footer":          ("NCE Energy Permit AI  ·  ncenergy.fi  ·  info@ncenergy.fi  "
                            "·  AI-utkast — kräver expertgranskning"),
        "th_lupa":  "Tillstånd / anmälan", "th_viran": "Myndighet", "th_laki": "Rättslig grund",
        "th_nro":   "Nr", "th_liite": "Bilaga", "th_tila": "Status",
        "liite_toimitettu": "[ ] Inlämnad",
        "toim_nro": "Nr", "toim_toimenpide": "Åtgärd",
        "toim_vastuutaho": "Ansvarig", "toim_aikataulu": "Tidplan",
        "hdr_draft": "bygglovsansökan — utkast", "hdr_right": "ncenergy.fi  |  AI-utkast",
        "ftr_ai":    "AI-utkast — kräver granskning", "ftr_sivu": "Sida",
        "bf_title": "FoU-finansieringsansökan — utkast",
        "bf_kotipaikka": "Hemort", "bf_vaihe": "Fas", "bf_tk_kuvaus": "FoU-beskrivning",
        "esiselvitys_p":   ("Projektet befinner sig i förundersökningsfasen. Slutliga tekniska "
                            "specifikationer, platsplaner och miljökonsekvensbedömningar preciseras "
                            "under den fortsatta planeringen."),
        "bess_pintaala":   "Den uppskattade anläggningsytan är 0,4–0,6 ha.",
        "mks_viittaus":    ("Markanvändningen i projektområdet har utretts i NCE Energys "
                            "markanvändningsutredning (se Bilaga 0b: Markanvändningsutredning PDF). "
                            "Utredningen innehåller fastighetsuppgifter, planläggningsstatus, "
                            "skyddsområden och grundvattenuppgifter."),
        "kaava_BESS":      ("<b>Planläggningsstatus (viktigaste förundersökningspunkten):</b> "
                            "Planläggningsstatusen för BESS-projektplatsen måste utredas först. I de flesta "
                            "kommuner kräver placering av ett batterienergilager en detaljplan eller "
                            "planeringstillstånd. Planläggningsstatus påverkar mest den totala längden på "
                            "tillståndsprocessen — förkonsultation med byggnadstillsynen är det första steget."),
        "kaava_tuuli":     ("<b>Planläggningsstatus och MKB-krav:</b> Ett vindkraftsprojekt kräver nästan "
                            "alltid en lokal översiktsplan eller detaljplan (MRL 132/1999, 77a §). "
                            "MKB-förfarandet (YVA-laki 252/2017) är obligatoriskt för projekt ≥10 MW eller "
                            "≥5 verk — plan- och MKB-processerna löper ofta parallellt och tar sammanlagt "
                            "3–6 år. Planläggningsstatus klarläggs först."),
        "kaava_SMR":       ("<b>STUK förlicensiering (viktigaste första steget):</b> För ett kärnkraftverk "
                            "krävs statsrådets principbeslut (kärnenergilag 990/1987, 11 §) och STUK:s "
                            "förlicensieringsförfarande innan alla andra tillstånd. STUK:s "
                            "säkerhetsredogörelse enligt YVL-riktlinjerna inleder processen. "
                            "Planläggning hanteras parallellt men kärnkraftssäkerhetsförfarandet är "
                            "den dominerande faktorn."),
        "kaava_aurinkovoima": ("<b>Åtgärdstillstånd eller bygglov — och planläggning:</b> För en liten "
                            "solkraftspark (under ca 1 ha) räcker det ofta med åtgärdstillstånd istället "
                            "för fullt bygglov (Bygglag 751/2023 / MRL 132/1999, 126 §). MKB krävs inte "
                            "för projekt under 50 ha. Planläggningsstatus måste ändå kontrolleras — "
                            "planeringstillstånd kan behövas utanför detaljplaneområden."),
        "kaava_generic":   ("<b>Planläggningsstatus:</b> Gällande planläggningsstatus för projektplatsen "
                            "måste verifieras i ett förkonsultationsmöte med byggnadstillsynen innan "
                            "tillståndsansökan lämnas in. Planläggning påverkar direkt varaktigheten och "
                            "kraven i tillståndsprocessen — byggande kräver ofta en detaljplan, en ändring "
                            "av en sådan eller ett planeringstillstånd."),
    },
    "DA": {
        "sub_title":       "Byggetilladelsesansøgning — udkast",
        "esiselvitys_sub": ("Forundersøgelses- og forhåndskonsultationsmateriale — "
                            "Udarbejdet til forhåndskonsultation med byggesagsafdelingen"),
        "disclaimer_h":    "AI-UDKAST — KRÆVER EKSPERTGENNEMGANG",
        "disclaimer_b":    ("Dette dokument er et AI-assisteret udkast. Det er ikke juridisk bindende og "
                            "erstatter ikke rådgivning fra en kvalificeret tilladelsesekspert eller advokat. "
                            "Inden ansøgningen indsendes, skal dokumentet gennemgås af en fagmand."),
        "m_hakija":        "Ansøger",          "m_ytunnus":    "CVR-nummer",
        "m_hanketyyppi":   "Projekttype",      "m_teho":       "Kapacitet / Effekt",
        "m_kunta":         "Kommune",          "m_kt":         "Ejendomsnummer",
        "m_maa":           "Land",
        "m_laadittu":      "Udarbejdet",       "m_laatinut_lbl": "Udarbejdet af",
        "m_laatinut":      "NCE Energy Permit AI (AI-assisteret)",
        "sec1": "1. Projektbeskrivelse",             "sec2": "2. Begrundelse og behov",
        "sec3": "3. Nødvendige tilladelser og myndigheder", "sec4": "4. Lovhenvisninger",
        "sec5": "5. Bilagsliste",                    "sec6": "6. Næste skridt",
        "liitteet_note":   ("Følgende bilag skal indsendes sammen med ansøgningen. "
                            "Sæt kryds i afkrydsningsfeltet, når bilaget er klar."),
        "lahteet_h":       "Kilder og referencer",
        "lahteet_b":       "Dette udkast er udarbejdet ved hjælp af følgende officielle dokumenter:",
        "yhteystiedot_h":  "Ansøgerens kontaktoplysninger",
        "yht_hakija":      "Ansøger",    "yht_ytunnus":    "CVR-nummer",
        "yht_osoite":      "Adresse",    "yht_lisatietoja": "Yderligere oplysninger",
        "footer":          ("NCE Energy Permit AI  ·  ncenergy.fi  ·  info@ncenergy.fi  "
                            "·  AI-udkast — kræver ekspertgennemgang"),
        "th_lupa":  "Tilladelse / anmeldelse", "th_viran": "Myndighed", "th_laki": "Retsgrundlag",
        "th_nro":   "Nr.", "th_liite": "Bilag", "th_tila": "Status",
        "liite_toimitettu": "[ ] Indsendt",
        "toim_nro": "Nr.", "toim_toimenpide": "Handling",
        "toim_vastuutaho": "Ansvarlig", "toim_aikataulu": "Tidsplan",
        "hdr_draft": "tilladelsesansøgning — udkast", "hdr_right": "ncenergy.fi  |  AI-udkast",
        "ftr_ai":    "AI-udkast — kræver gennemgang", "ftr_sivu": "Side",
        "bf_title": "F&U-finansieringsansøgning — udkast",
        "bf_kotipaikka": "Hjemsted", "bf_vaihe": "Fase", "bf_tk_kuvaus": "F&U-beskrivelse",
        "esiselvitys_p":   ("Projektet befinder sig i forundersøgelsesfasen. Endelige tekniske "
                            "specifikationer, lokalplaner og miljøkonsekvensvurderinger vil blive "
                            "præciseret under den videre planlægning."),
        "bess_pintaala":   "Det anslåede anlægsareal er 0,4–0,6 ha.",
        "mks_viittaus":    ("Arealanvendelsen i projektområdet er undersøgt i NCE Energys "
                            "arealanvendelsesrapport (se Bilag 0b: Arealanvendelsesrapport PDF). "
                            "Rapporten indeholder ejendomsoplysninger, planlægningsstatus, "
                            "beskyttede områder og grundvandsdata."),
        "kaava_BESS":      ("<b>Planlægningsstatus (det vigtigste forundersøgelseselement):</b> "
                            "Planlægningsstatusen for BESS-projektstedet skal fastlægges først. I de fleste "
                            "kommuner kræver placering af et batterienergilagringssystem en lokalplan eller "
                            "en planlægningstilladelse. Planlægningsstatus har størst indflydelse på den "
                            "samlede varighed af tilladelsesprocessen — forhåndskonsultation med "
                            "byggesagsafdelingen er det første trin."),
        "kaava_tuuli":     ("<b>Planlægningsstatus og VVM-krav:</b> Et vindkraftprojekt kræver næsten altid "
                            "en lokaloversigsplan eller lokalplan (MRL 132/1999, 77a §). VVM-proceduren "
                            "(YVA-laki 252/2017) er obligatorisk for projekter ≥10 MW eller ≥5 møller — "
                            "plan- og VVM-processerne forløber ofte parallelt og tager tilsammen 3–6 år. "
                            "Planlægningsstatus afklares først."),
        "kaava_SMR":       ("<b>STUK forlicensiering (det vigtigste første trin):</b> For et kernekraftanlæg "
                            "er statsrådets principbeslutning (kernenergiloven 990/1987, § 11) og STUKs "
                            "forlicensieringsprocedure obligatoriske inden alle andre tilladelser. STUKs "
                            "sikkerhedsredegørelse i henhold til YVL-retningslinjerne indleder processen. "
                            "Planlægning behandles parallelt, men kernesikkerhedsproceduren er den "
                            "dominerende faktor."),
        "kaava_aurinkovoima": ("<b>Handlingstilladelse eller byggetilladelse — og planlægning:</b> For en lille "
                            "solcellepark (under ca. 1 ha) er en handlingstilladelse ofte tilstrækkelig i "
                            "stedet for en fuld byggetilladelse (Bygglov 751/2023 / MRL 132/1999, § 126). "
                            "VVM kræves ikke for projekter under 50 ha. Planlægningsstatus skal dog stadig "
                            "kontrolleres — en planlægningstilladelse kan være nødvendig uden for "
                            "lokalplanområder."),
        "kaava_generic":   ("<b>Planlægningsstatus:</b> Den gældende planlægningsstatus for projektstedet "
                            "skal verificeres på et forhåndskonsultationsmøde med byggesagsafdelingen, "
                            "inden tilladelsesansøgningen indsendes. Planlægning påvirker direkte "
                            "varigheden og kravene i tilladelsesprocessen — byggeri kræver ofte en "
                            "lokalplan, en ændring heraf eller en planlægningstilladelse."),
    },
    "NO": {
        "sub_title":       "Søknad om byggetillatelse — utkast",
        "esiselvitys_sub": ("Forstudie- og forhåndskonsultasjonsmateriale — "
                            "Utarbeidet til forhåndskonsultasjon med byggesaksavdelingen"),
        "disclaimer_h":    "AI-UTKAST — KREVER EKSPERTGJENNOMGANG",
        "disclaimer_b":    ("Dette dokumentet er et AI-assistert utkast. Det er ikke juridisk bindende og "
                            "erstatter ikke råd fra en kvalifisert tillatelsesekspert eller advokat. "
                            "Før søknaden sendes inn, må dokumentet gjennomgås av en fagperson."),
        "m_hakija":        "Søker",             "m_ytunnus":    "Org.nummer",
        "m_hanketyyppi":   "Prosjekttype",      "m_teho":       "Kapasitet / Effekt",
        "m_kunta":         "Kommune",           "m_kt":         "Eiendomsnummer",
        "m_maa":           "Land",
        "m_laadittu":      "Utarbeidet",        "m_laatinut_lbl": "Utarbeidet av",
        "m_laatinut":      "NCE Energy Permit AI (AI-assistert)",
        "sec1": "1. Prosjektbeskrivelse",            "sec2": "2. Begrunnelse og behov",
        "sec3": "3. Nødvendige tillatelser og myndigheter", "sec4": "4. Lovhenvisninger",
        "sec5": "5. Vedleggsliste",                  "sec6": "6. Neste steg",
        "liitteet_note":   ("Følgende vedlegg må leveres sammen med søknaden. "
                            "Kryss av i boksen når vedlegget er klart."),
        "lahteet_h":       "Kilder og referanser",
        "lahteet_b":       "Dette utkastet er utarbeidet ved hjelp av følgende offisielle dokumenter:",
        "yhteystiedot_h":  "Søkerens kontaktopplysninger",
        "yht_hakija":      "Søker",      "yht_ytunnus":    "Org.nummer",
        "yht_osoite":      "Adresse",    "yht_lisatietoja": "Ytterligere informasjon",
        "footer":          ("NCE Energy Permit AI  ·  ncenergy.fi  ·  info@ncenergy.fi  "
                            "·  AI-utkast — krever ekspertgjennomgang"),
        "th_lupa":  "Tillatelse / melding", "th_viran": "Myndighet", "th_laki": "Rettsgrunnlag",
        "th_nro":   "Nr.", "th_liite": "Vedlegg", "th_tila": "Status",
        "liite_toimitettu": "[ ] Innlevert",
        "toim_nro": "Nr.", "toim_toimenpide": "Tiltak",
        "toim_vastuutaho": "Ansvarlig", "toim_aikataulu": "Tidsplan",
        "hdr_draft": "tillatelsessøknad — utkast", "hdr_right": "ncenergy.fi  |  AI-utkast",
        "ftr_ai":    "AI-utkast — krever gjennomgang", "ftr_sivu": "Side",
        "bf_title": "FoU-finansieringssøknad — utkast",
        "bf_kotipaikka": "Hjemsted", "bf_vaihe": "Fase", "bf_tk_kuvaus": "FoU-beskrivelse",
        "esiselvitys_p":   ("Prosjektet er i forstudiefasen. Endelige tekniske spesifikasjoner, "
                            "stedplaner og miljøkonsekvensutredninger vil bli presisert "
                            "under videre planlegging."),
        "bess_pintaala":   "Det anslåtte anleggsarealet er 0,4–0,6 ha.",
        "mks_viittaus":    ("Arealbruken i prosjektområdet er undersøkt i NCE Energys "
                            "arealbruksrapport (se Vedlegg 0b: Arealbruksrapport PDF). "
                            "Rapporten inneholder eiendomsopplysninger, reguleringstatus, "
                            "verneområder og grunnvannsdata."),
        "kaava_BESS":      ("<b>Reguleringstatus (viktigste forstudieelement):</b> "
                            "Reguleringsstatusen for BESS-prosjektstedet må fastlegges først. I de fleste "
                            "kommuner krever plassering av et batterienergilagringssystem en reguleringsplan "
                            "eller dispensasjon. Reguleringstatus har størst innvirkning på den totale "
                            "varigheten av tillatelsprosessen — forhåndskonsultasjon med byggesaksavdelingen "
                            "er det første trinnet."),
        "kaava_tuuli":     ("<b>Reguleringstatus og KU-krav:</b> Et vindkraftprosjekt krever nesten alltid "
                            "en kommunedelplan eller reguleringsplan (MRL 132/1999, 77a §). KU-prosedyren "
                            "(YVA-laki 252/2017) er obligatorisk for prosjekter ≥10 MW eller ≥5 turbiner — "
                            "plan- og KU-prosessene løper ofte parallelt og tar til sammen 3–6 år. "
                            "Reguleringstatus avklares først."),
        "kaava_SMR":       ("<b>STUK forhåndslisensering (viktigste første trinn):</b> For et kjernekraftanlegg "
                            "er statsrådets prinsippbeslutning (atomenergisloven 990/1987, § 11) og STUKs "
                            "forhåndslisensieringsprosedyre obligatoriske før alle andre tillatelser. STUKs "
                            "sikkerhetsrapport i henhold til YVL-retningslinjene starter prosessen. "
                            "Regulering håndteres parallelt, men kjernekraftsikkerhetsprosedyren er den "
                            "dominerende faktoren."),
        "kaava_aurinkovoima": ("<b>Tiltak- eller byggetillatelse — og regulering:</b> For en liten "
                            "solkraftpark (under ca. 1 ha) er en tiltakstillatelse ofte tilstrekkelig i "
                            "stedet for full byggetillatelse (Bygglov 751/2023 / MRL 132/1999, § 126). "
                            "KU kreves ikke for prosjekter under 50 ha. Reguleringstatus må likevel "
                            "sjekkes — dispensasjon kan være nødvendig utenfor reguleringsplanområder."),
        "kaava_generic":   ("<b>Reguleringstatus:</b> Gjeldende reguleringstatus for prosjektstedet "
                            "må verifiseres på et forhåndskonsultasjonsmøte med byggesaksavdelingen "
                            "før tillatelssøknaden sendes inn. Regulering påvirker direkte varigheten "
                            "og kravene i tillatelsprosessen — bygging krever ofte en reguleringsplan, "
                            "en endring av denne eller dispensasjon."),
    },
    "PL": {
        "sub_title":       "Wniosek o pozwolenie na budowę — szkic",
        "esiselvitys_sub": ("Materiał z analizy wstępnej i konsultacji wstępnych — "
                            "Przygotowany do wstępnej konsultacji z wydziałem budowlanym"),
        "disclaimer_h":    "SZKIC AI — WYMAGA PRZEGLĄDU EKSPERTA",
        "disclaimer_b":    ("Niniejszy dokument jest szkicem przygotowanym z pomocą AI. Nie jest prawnie "
                            "wiążący i nie zastępuje porady wykwalifikowanego eksperta ds. zezwoleń ani "
                            "prawnika. Przed złożeniem wniosku dokument musi zostać sprawdzony przez "
                            "specjalistę."),
        "m_hakija":        "Wnioskodawca",      "m_ytunnus":    "NIP/KRS",
        "m_hanketyyppi":   "Typ projektu",      "m_teho":       "Moc / pojemność",
        "m_kunta":         "Gmina",             "m_kt":         "Numer nieruchomości",
        "m_maa":           "Kraj",
        "m_laadittu":      "Sporządzono",       "m_laatinut_lbl": "Sporządzone przez",
        "m_laatinut":      "NCE Energy Permit AI (wspomagane przez AI)",
        "sec1": "1. Opis projektu",                  "sec2": "2. Uzasadnienie i potrzeba",
        "sec3": "3. Wymagane zezwolenia i organy",   "sec4": "4. Podstawy prawne",
        "sec5": "5. Lista załączników",              "sec6": "6. Następne kroki",
        "liitteet_note":   ("Następujące załączniki muszą zostać złożone wraz z wnioskiem. "
                            "Zaznacz pole wyboru, gdy załącznik jest gotowy."),
        "lahteet_h":       "Źródła i odniesienia",
        "lahteet_b":       "Niniejszy szkic został przygotowany przy użyciu następujących oficjalnych dokumentów:",
        "yhteystiedot_h":  "Dane kontaktowe wnioskodawcy",
        "yht_hakija":      "Wnioskodawca", "yht_ytunnus":    "NIP/KRS",
        "yht_osoite":      "Adres",        "yht_lisatietoja": "Dodatkowe informacje",
        "footer":          ("NCE Energy Permit AI  ·  ncenergy.fi  ·  info@ncenergy.fi  "
                            "·  Szkic AI — wymaga przeglądu eksperta"),
        "th_lupa":  "Zezwolenie / zgłoszenie", "th_viran": "Organ", "th_laki": "Podstawa prawna",
        "th_nro":   "Nr", "th_liite": "Załącznik", "th_tila": "Status",
        "liite_toimitettu": "[ ] Złożony",
        "toim_nro": "Nr", "toim_toimenpide": "Działanie",
        "toim_vastuutaho": "Odpowiedzialny", "toim_aikataulu": "Harmonogram",
        "hdr_draft": "wniosek o zezwolenie — szkic", "hdr_right": "ncenergy.fi  |  Szkic AI",
        "ftr_ai":    "Szkic AI — wymaga przeglądu", "ftr_sivu": "Strona",
        "bf_title": "Wniosek o dofinansowanie B+R — szkic",
        "bf_kotipaikka": "Siedziba", "bf_vaihe": "Faza", "bf_tk_kuvaus": "Opis B+R",
        "esiselvitys_p":   ("Projekt jest w fazie analizy wstępnej. Ostateczne specyfikacje techniczne, "
                            "plany lokalizacyjne i oceny oddziaływania na środowisko zostaną doprecyzowane "
                            "w trakcie dalszego planowania."),
        "bess_pintaala":   "Szacunkowa powierzchnia instalacji wynosi 0,4–0,6 ha.",
        "mks_viittaus":    ("Zagospodarowanie terenu obszaru projektu zostało zbadane w raporcie NCE Energy "
                            "dotyczącym zagospodarowania terenu (zob. Załącznik 0b: Raport PDF). Raport "
                            "zawiera informacje o nieruchomości, status planistyczny, obszary chronione "
                            "i dane o wodach gruntowych."),
        "kaava_BESS":      ("<b>Status planistyczny (najważniejszy element analizy wstępnej):</b> "
                            "Status planistyczny terenu projektu BESS musi zostać ustalony w pierwszej "
                            "kolejności. W większości gmin umiejscowienie systemu magazynowania energii "
                            "w akumulatorach wymaga miejscowego planu zagospodarowania lub decyzji "
                            "o warunkach zabudowy. Status planistyczny ma największy wpływ na całkowity "
                            "czas trwania procesu uzyskiwania zezwoleń — wstępna konsultacja z wydziałem "
                            "budowlanym jest pierwszym krokiem."),
        "kaava_tuuli":     ("<b>Status planistyczny i wymóg OOŚ:</b> Projekt farmy wiatrowej niemal zawsze "
                            "wymaga planu miejscowego (MRL 132/1999, 77a §). Procedura OOŚ "
                            "(YVA-laki 252/2017) jest obowiązkowa dla projektów ≥10 MW lub ≥5 turbin — "
                            "procesy planistyczne i OOŚ przebiegają często równolegle i trwają łącznie "
                            "3–6 lat. Status planistyczny ustala się w pierwszej kolejności."),
        "kaava_SMR":       ("<b>Wstępne licencjonowanie STUK (najważniejszy pierwszy krok):</b> W przypadku "
                            "obiektu jądrowego decyzja zasadnicza Rady Stanu (ustawa o energii jądrowej "
                            "990/1987, § 11) i procedura wstępnego licencjonowania STUK są obowiązkowe "
                            "przed wszystkimi innymi zezwoleniami. Raport bezpieczeństwa STUK zgodny "
                            "z wytycznymi YVL inicjuje proces. Planowanie odbywa się równolegle, "
                            "ale procedura bezpieczeństwa jądrowego jest czynnikiem dominującym."),
        "kaava_aurinkovoima": ("<b>Pozwolenie na roboty budowlane lub pozwolenie na budowę — i planowanie:</b> "
                            "Dla małej elektrowni słonecznej (poniżej ok. 1 ha) często wystarczy zgłoszenie "
                            "robót budowlanych zamiast pełnego pozwolenia na budowę (Ustawa budowlana "
                            "751/2023 / MRL 132/1999, § 126). OOŚ nie jest wymagana dla projektów poniżej "
                            "50 ha. Status planistyczny musi jednak zostać sprawdzony — decyzja o warunkach "
                            "zabudowy może być konieczna poza obszarami objętymi miejscowym planem."),
        "kaava_generic":   ("<b>Status planistyczny:</b> Obowiązujący status planistyczny terenu projektu "
                            "musi zostać zweryfikowany na spotkaniu konsultacyjnym z wydziałem budowlanym "
                            "przed złożeniem wniosku o zezwolenie. Planowanie bezpośrednio wpływa na czas "
                            "trwania i wymagania procesu uzyskiwania zezwoleń — budowa często wymaga "
                            "miejscowego planu zagospodarowania, jego zmiany lub decyzji o warunkach "
                            "zabudowy."),
    },
}

_KAAVA_KEY: dict[str, str] = {
    "BESS":          "kaava_BESS",
    "tuulivoima_maa": "kaava_tuuli",
    "tuulivoima_meri": "kaava_tuuli",
    "SMR":           "kaava_SMR",
    "smr_bess":      "kaava_SMR",
    "aurinkovoima":  "kaava_aurinkovoima",
}


def _s(lang: str, key: str) -> str:
    """Hae käännetty merkkijono PDF-layoutille. Fallback → FI."""
    d = _PDF_STRINGS.get(lang) or _PDF_STRINGS["FI"]
    return d.get(key) or _PDF_STRINGS["FI"].get(key, key)


def _generate_bf_sections(inp: ApplicationInput, rag_context: str) -> dict[str, str]:
    """Business Finland Sprint -hakemusosioiden generointi."""
    now = datetime.now().strftime("%d.%m.%Y")
    vaihe = inp.hankkeen_vaihe or "esiselvitys"
    tk_kuvaus = inp.sijainti_ymparistovaikutukset or ""
    viranomainen_bf = inp.kohdeviranomainen or "Business Finland (avustushakemus)"

    lang_prefix = _LANG_INSTRUCTIONS.get(getattr(inp, "lang", "FI"), "")
    prompt = f"""{lang_prefix}Laadi Business Finland Sprint -rahoitushakemuksen luonnos:

Hakija / yritys: {inp.hakija}
Sijaintikunta: {inp.kunta}
Hankkeen vaihe: {vaihe}
Kohdeviranomainen / rahoittaja: {viranomainen_bf}
T&K-haasteet / innovaatiokuvaus: {tk_kuvaus if tk_kuvaus else '(ei täydennetty)'}
Päivämäärä: {now}

Alla on relevanttia energia-alan T&K-dokumentaatiota:
{rag_context}

Kirjoita suomeksi seuraavat neljä osiota selkeästi eroteltuna otsikoilla:

## T&K-KUVAUS
Kirjoita 3–5 kappaleen kuvaus tutkimus- ja kehitystyöstä: tutkimusongelma, innovaatio, teknologinen lähestymistapa, odotetut tulokset ja tieteellinen/teknologinen uutuusarvo. Ota huomioon hakijan toimiala ja T&K-haasteiden kuvaus.

## BUDJETTI JA RAHOITUSRAKENNE
Kirjoita 2–3 kappaletta budjettirakenteesta ja rahoitussuunnitelmasta: kokonaisbudjetti jakautuminen (henkilöstökulut, alihankinnat, laitteet, muut), oma rahoitusosuus ja haettava BF-tuki, kustannustehokkuus.

## TIIMIKUVAUS
Kirjoita 2–3 kappaletta tiimin osaamistaustasta: keskeiset henkilöt ja roolit, relevantit aiemmat projektit ja referenssit, yhteistyökumppanit ja alihankkijat.

## PROJEKTIAIKATAULU
Listaa projektin vaiheet ja keskeisimmät välitavoitteet (milestones) kvartaali- tai kuukausitarkkuudella. Aloita hankkeen käynnistämisestä ja pääty loppuraporttiin."""

    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp   = claude.messages.create(
        model=_MODEL_ID,
        max_tokens=4000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text

    def _extract(text: str, header: str, next_headers: list[str]) -> str:
        start = text.find(f"## {header}")
        if start == -1:
            return ""
        start = text.find("\n", start) + 1
        end   = len(text)
        for nh in next_headers:
            pos = text.find(f"## {nh}", start)
            if pos != -1:
                end = min(end, pos)
        return text[start:end].strip()

    headers = ["T&K-KUVAUS", "BUDJETTI JA RAHOITUSRAKENNE", "TIIMIKUVAUS", "PROJEKTIAIKATAULU"]
    return {
        "tk_kuvaus":    _extract(raw, "T&K-KUVAUS",               headers[1:]),
        "budjetti":     _extract(raw, "BUDJETTI JA RAHOITUSRAKENNE", headers[2:]),
        "tiimi":        _extract(raw, "TIIMIKUVAUS",               headers[3:]),
        "aikataulu":    _extract(raw, "PROJEKTIAIKATAULU",          []),
    }


def _generate_sections(inp: ApplicationInput, rag_context: str) -> dict[str, str]:
    """
    Kutsu Claude-API ja generoi kaikki hakemuksen osiot yhdellä kutsulla.
    Palauttaa dict: { "kuvaus": ..., "luvat_teksti": ..., "laki": ..., "toimenpiteet": ... }
    """
    cfg  = _HANKE_CFG[inp.hanketyyppi]
    now  = datetime.now().strftime("%d.%m.%Y")
    lang = getattr(inp, "lang", "FI")
    country = getattr(inp, "country", "FI") or "FI"
    ph   = _PROMPT_HEADERS.get(lang, _PROMPT_HEADERS["FI"])
    lang_prefix  = _LANG_INSTRUCTIONS.get(lang, "")
    write_instr  = _WRITE_INSTRUCTION.get(lang, _WRITE_INSTRUCTION["FI"])
    country_prefix = _COUNTRY_CONFIG.get(country, {}).get("prompt_prefix", "")

    sijainti_lisatieto = ""
    if inp.sijainti_ymparistovaikutukset:
        sijainti_lisatieto = f"\nSijainti / ympäristövaikutukset: {inp.sijainti_ymparistovaikutukset}"
    vaihe_lisatieto = ""
    if inp.hankkeen_vaihe:
        vaihe_lisatieto = f"\nHankkeen vaihe: {inp.hankkeen_vaihe}"
    viranomainen_lisatieto = ""
    if inp.kohdeviranomainen:
        viranomainen_lisatieto = f"\nKohdeviranomainen: {inp.kohdeviranomainen}"

    viranomainen_ohje = ""
    if inp.kohdeviranomainen:
        viranomainen_ohje = "\n\n" + ph["viranomainen_ohje"].format(auth=inp.kohdeviranomainen)

    kap_lisatieto = ""
    if inp.kapasiteetti_mwh and inp.kapasiteetti_mwh > 0:
        kap_lisatieto = f"\nKapasiteetti: {inp.kapasiteetti_mwh} MWh"

    first_action  = ph["toimenpiteet_first"].format(kunta=inp.kunta)
    kuvaus_inst   = ph["kuvaus_inst"] + (ph["kuvaus_extra"] if inp.sijainti_ymparistovaikutukset else "")
    luvat_inst    = ph["luvat_inst"] + (ph["luvat_extra"].format(auth=inp.kohdeviranomainen) if inp.kohdeviranomainen else "")
    toim_inst     = (ph["toimenpiteet_inst"].format(first=first_action)
                     + (ph["toimenpiteet_vaihe"].format(vaihe=inp.hankkeen_vaihe) if inp.hankkeen_vaihe else ""))

    prompt = f"""{lang_prefix}{country_prefix}{ph["intro"]}

Hanketyyppi: {inp.hanketyyppi} ({cfg['nimi_fi']})
Kiinteistötunnus: {inp.kiinteistotunnus}
Teho: {inp.teho_mw} MW{kap_lisatieto}
Kunta: {inp.kunta}
Hakija: {inp.hakija}{sijainti_lisatieto}{vaihe_lisatieto}{viranomainen_lisatieto}
Päivämäärä: {now}{viranomainen_ohje}

{ph["rag_intro"]}
{rag_context}

{write_instr}

## {ph["kuvaus"]}
{kuvaus_inst}

## {ph["perustelut"]}
{ph["perustelut_inst"]}

## {ph["luvat"]}
{luvat_inst}

## {ph["toimenpiteet"]}
{toim_inst}"""

    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp   = claude.messages.create(
        model=_MODEL_ID,
        max_tokens=8000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text

    # Parsitaan osiot käyttämällä kielen mukaisia otsikoita
    h = [ph["kuvaus"], ph["perustelut"], ph["luvat"], ph["toimenpiteet"]]

    def _extract(text: str, header: str, next_headers: list[str]) -> str:
        start = text.find(f"## {header}")
        if start == -1:
            return ""
        start = text.find("\n", start) + 1
        end   = len(text)
        for nh in next_headers:
            pos = text.find(f"## {nh}", start)
            if pos != -1:
                end = min(end, pos)
        return text[start:end].strip()

    return {
        "kuvaus":        _extract(raw, h[0], h[1:]),
        "perustelut":    _extract(raw, h[1], h[2:]),
        "luvat_teksti":  _extract(raw, h[2], h[3:]),
        "toimenpiteet":  _extract(raw, h[3], []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF-generointi
# ─────────────────────────────────────────────────────────────────────────────

def _st() -> dict:
    """Paragraphityylit."""
    return {
        "title":    ParagraphStyle("at", fontSize=20, textColor=C_NAVY,
                                   fontName="Helvetica-Bold", spaceAfter=3, leading=24),
        "sub":      ParagraphStyle("as", fontSize=10, textColor=C_RED,
                                   fontName="Helvetica-Bold", spaceAfter=4),
        "meta":     ParagraphStyle("am", fontSize=8.5, textColor=C_GRAY,
                                   leading=13, spaceAfter=2),
        "h2":       ParagraphStyle("ah2", fontSize=11, textColor=C_NAVY,
                                   fontName="Helvetica-Bold", spaceBefore=14,
                                   spaceAfter=5, leading=15),
        "body":     ParagraphStyle("ab", fontSize=9, leading=14, spaceAfter=5),
        "small":    ParagraphStyle("asm", fontSize=7.5, textColor=C_GRAY,
                                   leading=11, spaceAfter=2),
        "warn":     ParagraphStyle("aw", fontSize=8, textColor=C_WARN,
                                   fontName="Helvetica-Bold", alignment=TA_CENTER,
                                   spaceBefore=4, spaceAfter=4),
        "bullet":   ParagraphStyle("abul", fontSize=9, leading=14,
                                   leftIndent=14, spaceAfter=3),
    }


def _hr(color=C_DGRAY, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=2)


def _disclaimer_box(st: dict, lang: str = "FI") -> Table:
    """Oranssi AI-varoituslaatikko."""
    C_WARN_BG = colors.HexColor("#fff3e0")
    C_WARN_BD = colors.HexColor("#ff9800")
    row = [[Paragraph(
        f"{_s(lang, 'disclaimer_h')}\n{_s(lang, 'disclaimer_b')}",
        ParagraphStyle("disc", fontSize=8, textColor=colors.HexColor("#7a4400"),
                       fontName="Helvetica-Bold", alignment=TA_CENTER, leading=12)
    )]]
    tbl = Table(row, colWidths=[16.5 * cm])
    tbl.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 1.5, C_WARN_BD),
        ("BACKGROUND", (0, 0), (-1, -1), C_WARN_BG),
        ("PADDING",    (0, 0), (-1, -1), 10),
    ]))
    return tbl


def _luvat_table(hanketyyppi: str, st: dict, lang: str = "FI") -> Table:
    """Lupa-taulukko hanketyyppikohtaisesti."""
    cfg  = _HANKE_CFG[hanketyyppi]
    _th  = ParagraphStyle("th", fontSize=8.5, fontName="Helvetica-Bold")
    rows = [[
        Paragraph(_s(lang, "th_lupa"),  _th),
        Paragraph(_s(lang, "th_viran"), _th),
        Paragraph(_s(lang, "th_laki"),  _th),
    ]]
    for lupa, viranomainen, laki in cfg["luvat"]:
        rows.append([
            Paragraph(_t_lupa(lang, lupa),         ParagraphStyle("td", fontSize=8.5, leading=12)),
            Paragraph(_t_auth(lang, viranomainen), ParagraphStyle("td", fontSize=8.5, leading=12, textColor=C_BLUE)),
            Paragraph(_t_law(lang, laki),          ParagraphStyle("td", fontSize=7.5, leading=11, textColor=C_GRAY)),
        ])

    col_w = [6.5*cm, 5.5*cm, 4.5*cm]
    tbl   = Table(rows, colWidths=col_w, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), C_WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",       (0, 0), (-1, -1), 0.4, C_DGRAY),
        ("PADDING",    (0, 0), (-1, -1), 7),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl


def _liitteet_table(hanketyyppi: str, lang: str = "FI") -> Table:
    """Liiteluettelo checkboxeilla."""
    cfg  = _HANKE_CFG[hanketyyppi]
    _th2 = ParagraphStyle("th2", fontSize=8.5, fontName="Helvetica-Bold")
    rows = [[
        Paragraph(_s(lang, "th_nro"),   _th2),
        Paragraph(_s(lang, "th_liite"), _th2),
        Paragraph(_s(lang, "th_tila"),  ParagraphStyle("th2c", fontSize=8.5, fontName="Helvetica-Bold",
                                                        alignment=TA_CENTER)),
    ]]
    for i, liite in enumerate(cfg["liitteet"]):
        if i == 0:
            nro = "0"
        elif i == 1:
            nro = "0b"
        else:
            nro = str(i - 1)
        rows.append([
            Paragraph(nro,   ParagraphStyle("tn", fontSize=8.5)),
            Paragraph(_t_liite(lang, liite), ParagraphStyle("tl", fontSize=8.5, leading=12)),
            Paragraph(_s(lang, "liite_toimitettu"),
                      ParagraphStyle("tc", fontSize=7.5, textColor=C_GRAY, alignment=TA_CENTER)),
        ])

    tbl = Table(rows, colWidths=[1.0*cm, 12.5*cm, 3.0*cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_DGRAY),
        ("PADDING",       (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ALIGN",         (0, 0), (0, -1), "CENTER"),
        ("ALIGN",         (2, 0), (2, -1), "CENTER"),
    ]))
    return tbl


def _md_table_to_rl(lines: list, st: dict):
    """Muunna markdown-taulukon rivit ReportLab Table -objektiksi."""
    rows = []
    for line in lines:
        if re.match(r'^\|[-:| ]+\|$', line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        return None
    col_count = max(len(r) for r in rows)
    rows = [r + [''] * (col_count - len(r)) for r in rows]
    page_w, _ = A4
    avail_w = page_w - 2 * 2.2 * cm
    col_w = avail_w / col_count
    th_style = ParagraphStyle("md_th", fontSize=8, fontName="Helvetica-Bold",
                               textColor=C_WHITE)
    td_style = ParagraphStyle("md_td", fontSize=8, fontName="Helvetica", leading=11)
    tbl_data = []
    for i, row in enumerate(rows):
        tbl_data.append([Paragraph(cell, th_style if i == 0 else td_style)
                         for cell in row])
    tbl = Table(tbl_data, colWidths=[col_w] * col_count)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), C_NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DGRAY),
        ("PADDING",        (0, 0), (-1, -1), 5),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


def _para_text(text: str, st: dict) -> list:
    """Muunna AI:n tuottama teksti Paragraph-listaksi (kappalejaot \\n\\n)."""
    items = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        # Markdown-taulukko
        lines = para.splitlines()
        if len(lines) >= 2 and any(re.match(r'^\|[-:| ]+\|$', l.strip()) for l in lines):
            tbl = _md_table_to_rl(lines, st)
            if tbl is not None:
                items.append(tbl)
                continue
        # Alaotsikko (##)
        if para.startswith("## "):
            items.append(Paragraph(para[3:], ParagraphStyle(
                "ai_h3", fontSize=9.5, fontName="Helvetica-Bold",
                textColor=C_BLUE, spaceBefore=6, spaceAfter=3)))
        # Listakohta (- tai *)
        elif para.startswith(("- ", "* ", "• ")):
            for line in para.splitlines():
                line = line.lstrip("-*• ").strip()
                if line:
                    items.append(Paragraph(f"• {line}", st["bullet"]))
        # Numeroitu lista
        elif len(para) > 1 and para[0].isdigit() and para[1] in ".):":
            for line in para.splitlines():
                line = line.strip()
                if line:
                    items.append(Paragraph(line, st["bullet"]))
        else:
            clean = para.replace("**", "")
            items.append(Paragraph(clean, st["body"]))
    return items


def _toimenpiteet_elements(text: str, st: dict, lang: str = "FI") -> list:
    """Muunna toimenpide-teksti 4-sarakkeiseksi PDF-taulukoksi jos mahdollista."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    rows = []
    for line in lines:
        m = re.match(r'^(\d+)[.)]\s+(.+)', line)
        if not m:
            rows = []
            break
        nro = m.group(1)
        rest = m.group(2)
        parts = re.split(r'\s*[–—|]\s*', rest, maxsplit=2)
        if len(parts) == 3:
            rows.append([nro, parts[0], parts[1], parts[2]])
        elif len(parts) == 2:
            rows.append([nro, parts[0], parts[1], ''])
        else:
            rows.append([nro, rest, '', ''])
    if len(rows) < 2:
        return _para_text(text, st)
    header = [_s(lang, "toim_nro"), _s(lang, "toim_toimenpide"),
              _s(lang, "toim_vastuutaho"), _s(lang, "toim_aikataulu")]
    th_s = ParagraphStyle("tp_th", fontSize=8, fontName="Helvetica-Bold", textColor=C_WHITE)
    td_s = ParagraphStyle("tp_td", fontSize=8, fontName="Helvetica", leading=11)
    tbl_data = [[Paragraph(h, th_s) for h in header]]
    for row in rows:
        tbl_data.append([Paragraph(str(c), td_s) for c in row])
    tbl = Table(tbl_data, colWidths=[1.0*cm, 7.0*cm, 4.5*cm, 3.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), C_NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DGRAY),
        ("PADDING",        (0, 0), (-1, -1), 5),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))
    return [tbl]


def _make_canvas_cls(inp: ApplicationInput, now: str):
    """Palauta NumberedCanvas-aliluokka ylä- ja alatunnisteella (Sivu X / Y)."""

    class _NumberedCanvas(_CanvasBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states: list[dict] = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_decorations(self._pageNumber, total)
                _CanvasBase.showPage(self)
            _CanvasBase.save(self)

        def _draw_decorations(self, page_num: int, total: int):
            page_w, page_h = A4
            m = 2 * cm
            self.saveState()
            self.setStrokeColor(C_DGRAY)
            self.setLineWidth(0.3)
            self.setFont("Helvetica", 6.5)
            self.setFillColor(C_GRAY)
            _lang = getattr(inp, "lang", "FI")
            _draft = _s(_lang, "hdr_draft")
            # Ylätunniste
            self.line(m, page_h - 1.55*cm, page_w - m, page_h - 1.55*cm)
            self.drawString(m, page_h - 1.2*cm,
                f"{inp.hanketyyppi} {_draft}  |  {inp.kunta}  |  {now}")
            self.drawRightString(page_w - m, page_h - 1.2*cm, _s(_lang, "hdr_right"))
            # Alatunniste
            self.line(m, 1.45*cm, page_w - m, 1.45*cm)
            self.drawString(m, 0.9*cm,
                f"{inp.hanketyyppi} {_draft}  |  {inp.kiinteistotunnus}  |  {inp.kunta}")
            self.drawRightString(page_w - m, 0.9*cm,
                f"{now}  |  {_s(_lang, 'ftr_ai')}  |  {_s(_lang, 'ftr_sivu')} {page_num} / {total}")
            self.restoreState()

    return _NumberedCanvas


def _generate_bf_pdf(inp: ApplicationInput, sections: dict, sources: list[str]) -> bytes:
    """PDF-rakenne Business Finland Sprint -hakemukselle."""
    buf    = io.BytesIO()
    now    = datetime.now().strftime("%d.%m.%Y")
    cfg    = _HANKE_CFG["business_finland"]
    st     = _st()
    margin = 2.2 * cm

    canvas_cls = _make_canvas_cls(inp, now)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
    )
    story = []

    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("Business Finland Sprint", st["sub"]))
    story.append(Paragraph(_s(_bf_lang, "bf_title"), st["title"]))
    story.append(Paragraph(f"{inp.hakija}  ·  {inp.kunta}  ·  {now}", st["meta"]))
    story.append(Spacer(1, 4*mm))
    story.append(_hr(C_NAVY, 1.5))
    story.append(Spacer(1, 3*mm))

    meta_rows = [
        [_s(_bf_lang, "m_hakija"),       inp.hakija],
        [_s(_bf_lang, "bf_kotipaikka"),  inp.kunta],
        [_s(_bf_lang, "bf_vaihe"),       inp.hankkeen_vaihe or "–"],
        [_s(_bf_lang, "bf_tk_kuvaus"),   (inp.sijainti_ymparistovaikutukset or "–")[:120]],
        [_s(_bf_lang, "m_laadittu"),     now],
        [_s(_bf_lang, "m_laatinut_lbl"), _s(_bf_lang, "m_laatinut")],
    ]
    meta_tbl = Table(
        [[Paragraph(k, ParagraphStyle("mk", fontSize=8.5, textColor=C_GRAY,
                                      fontName="Helvetica-Bold")),
          Paragraph(v, ParagraphStyle("mv", fontSize=8.5, leading=12))]
         for k, v in meta_rows],
        colWidths=[4.5*cm, 12.0*cm],
    )
    meta_tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LGRAY, C_WHITE]),
        ("PADDING",        (0, 0), (-1, -1), 6),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DGRAY),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))
    _bf_lang = inp.lang or "FI"
    story.append(meta_tbl)
    story.append(Spacer(1, 6*mm))
    story.append(_disclaimer_box(st, _bf_lang))
    story.append(Spacer(1, 8*mm))

    story.append(KeepTogether([Paragraph("1. T&K-kuvaus", st["h2"]), _hr()]))
    story.extend(_para_text(sections.get("tk_kuvaus", "–"), st))
    story.append(Spacer(1, 4*mm))

    story.append(KeepTogether([Paragraph("2. Budjetti ja rahoitusrakenne", st["h2"]), _hr()]))
    story.extend(_para_text(sections.get("budjetti", "–"), st))
    story.append(Spacer(1, 4*mm))

    story.append(KeepTogether([Paragraph("3. Tiimikuvaus", st["h2"]), _hr()]))
    story.extend(_para_text(sections.get("tiimi", "–"), st))
    story.append(Spacer(1, 4*mm))

    story.append(KeepTogether([Paragraph("4. Projektiaikataulu", st["h2"]), _hr()]))
    story.extend(_para_text(sections.get("aikataulu", "–"), st))
    story.append(Spacer(1, 4*mm))

    story.append(KeepTogether([Paragraph(_s(_bf_lang, "sec5"), st["h2"]), _hr()]))
    story.append(_liitteet_table("business_finland", _bf_lang))
    story.append(Spacer(1, 4*mm))

    if sources:
        story.append(KeepTogether([
            Paragraph(_s(_bf_lang, "lahteet_h"), st["h2"]), _hr(),
            Paragraph(_s(_bf_lang, "lahteet_b"), st["body"]),
        ]))
        for s in sources:
            story.append(Paragraph(f"• {s}", st["bullet"]))
        story.append(Spacer(1, 3*mm))

    story.append(_hr(C_NAVY, 1.0))
    story.append(Paragraph(
        _s(_bf_lang, "footer"),
        ParagraphStyle("end", fontSize=7.5, textColor=C_GRAY, alignment=TA_CENTER, leading=11),
    ))
    doc.build(story, canvasmaker=canvas_cls)
    return buf.getvalue()


def generate_pdf(inp: ApplicationInput, sections: dict, sources: list[str]) -> bytes:
    """Rakenna PDF ja palauta bytes."""
    buf    = io.BytesIO()
    now    = datetime.now().strftime("%d.%m.%Y")
    cfg    = _HANKE_CFG[inp.hanketyyppi]
    st     = _st()
    margin = 2.2 * cm

    canvas_cls = _make_canvas_cls(inp, now)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
    )

    lang    = inp.lang or "FI"
    country = getattr(inp, "country", "FI") or "FI"
    story   = []

    # ── Kansilehti ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(_s(lang, "sub_title"), st["sub"]))
    story.append(Paragraph(_nimi(lang, inp.hanketyyppi, cfg['nimi_fi']), st["title"]))
    story.append(Paragraph(
        _s(lang, "esiselvitys_sub"),
        ParagraphStyle("kan_sub2", fontSize=9, textColor=C_GRAY,
                       fontName="Helvetica", spaceAfter=4, leading=13),
    ))
    story.append(Paragraph(f"{inp.teho_mw} MW  ·  {inp.kunta}  ·  {inp.kiinteistotunnus}", st["meta"]))
    story.append(Spacer(1, 4*mm))
    story.append(_hr(C_NAVY, 1.5))
    story.append(Spacer(1, 3*mm))

    # Metataulukko
    teho_val = f"{inp.teho_mw} MW"
    if inp.kapasiteetti_mwh and inp.kapasiteetti_mwh > 0:
        teho_val += f"  /  {inp.kapasiteetti_mwh} MWh"
    meta_rows = [
        [_s(lang, "m_hakija"),      inp.hakija],
        [_s(lang, "m_ytunnus"),     inp.y_tunnus if inp.y_tunnus else ""],
        [_s(lang, "m_hanketyyppi"), f"{inp.hanketyyppi} — {_nimi(lang, inp.hanketyyppi, cfg['nimi_fi'])}"],
        [_s(lang, "m_teho"),        teho_val],
        [_s(lang, "m_kunta"),       inp.kunta],
        [_s(lang, "m_kt"),          inp.kiinteistotunnus],
        *([[_s(lang, "m_maa"),       _COUNTRY_CONFIG.get(country, {}).get("name", country)]]
          if country != "FI" else []),
        [_s(lang, "m_laadittu"),        now],
        [_s(lang, "m_laatinut_lbl"),    _s(lang, "m_laatinut")],
    ]
    meta_tbl = Table(
        [[Paragraph(k, ParagraphStyle("mk", fontSize=8.5, textColor=C_GRAY,
                                      fontName="Helvetica-Bold")),
          Paragraph(v, ParagraphStyle("mv", fontSize=8.5, leading=12))]
         for k, v in meta_rows],
        colWidths=[4.5*cm, 12.0*cm],
    )
    meta_tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LGRAY, C_WHITE]),
        ("PADDING",        (0, 0), (-1, -1), 6),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DGRAY),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 6*mm))

    # AI-varoituslaatikko
    story.append(_disclaimer_box(st, lang))
    story.append(Spacer(1, 8*mm))

    # ── 1. Hankkeen kuvaus ────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec1"), st["h2"]),
        _hr(),
    ]))
    story.extend(_para_text(sections.get("kuvaus", "–"), st))
    story.append(Paragraph(_s(lang, "esiselvitys_p"), st["body"]))
    if inp.hanketyyppi == "BESS":
        story.append(Paragraph(_s(lang, "bess_pintaala"), st["body"]))
    story.append(Paragraph(_s(lang, "mks_viittaus"), st["body"]))
    story.append(Spacer(1, 4*mm))

    # ── 2. Perustelut ja tarve ────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec2"), st["h2"]),
        _hr(),
    ]))
    story.extend(_para_text(sections.get("perustelut", "–"), st))
    story.append(Spacer(1, 4*mm))

    # ── 3. Tarvittavat luvat ja viranomaiset ─────────────────────────────────
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec3"), st["h2"]),
        _hr(),
    ]))
    story.append(_luvat_table(inp.hanketyyppi, st, lang))
    story.append(Spacer(1, 5*mm))
    _kaava_key = _KAAVA_KEY.get(inp.hanketyyppi, "kaava_generic")
    story.append(Paragraph(_s(lang, _kaava_key), st["body"]))

    # AI:n lupakuvaukset
    luvat_txt = sections.get("luvat_teksti", "")
    if luvat_txt:
        story.extend(_para_text(luvat_txt, st))
    story.append(Spacer(1, 4*mm))

    # ── 4. Lakiviitteet ───────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec4"), st["h2"]),
        _hr(),
    ]))
    laki_rows_fi = {laki for _, _, laki in cfg["luvat"]}
    laki_rows_fi.update(cfg.get("laki_extra", []))
    for ref in sorted(laki_rows_fi):
        story.append(Paragraph(f"• {_t_law(lang, ref)}", st["bullet"]))
    story.append(Spacer(1, 4*mm))

    # ── 5. Liiteluettelo ──────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec5"), st["h2"]),
        _hr(),
    ]))
    story.append(Paragraph(_s(lang, "liitteet_note"), st["body"]))
    story.append(Spacer(1, 3*mm))
    story.append(_liitteet_table(inp.hanketyyppi, lang))
    story.append(Spacer(1, 4*mm))

    # ── 6. Seuraavat toimenpiteet ─────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec6"), st["h2"]),
        _hr(),
    ]))
    story.extend(_toimenpiteet_elements(sections.get("toimenpiteet", "–"), st, lang))
    story.append(Spacer(1, 4*mm))

    # ── Lähteet ───────────────────────────────────────────────────────────────
    if sources:
        story.append(KeepTogether([
            Paragraph(_s(lang, "lahteet_h"), st["h2"]),
            _hr(),
            Paragraph(_s(lang, "lahteet_b"), st["body"]),
        ]))
        for s in sources:
            story.append(Paragraph(f"• {s}", st["bullet"]))
        story.append(Spacer(1, 3*mm))

    # ── Hakijan yhteystiedot ──────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph(_s(lang, "yhteystiedot_h"), st["h2"]),
        _hr(),
    ]))
    yhteystiedot_data = [
        [_s(lang, "yht_hakija"),      inp.hakija],
        [_s(lang, "yht_ytunnus"),     inp.y_tunnus if inp.y_tunnus else "–"],
        [_s(lang, "yht_osoite"),      inp.osoite if inp.osoite else "–"],
        [_s(lang, "yht_lisatietoja"), "NCE Energy Permit AI  ·  ncenergy.fi  ·  info@ncenergy.fi"],
    ]
    yht_tbl = Table(
        [[Paragraph(k, ParagraphStyle("yk", fontSize=8.5, textColor=C_GRAY, fontName="Helvetica-Bold")),
          Paragraph(v, ParagraphStyle("yv", fontSize=8.5, leading=12))]
         for k, v in yhteystiedot_data],
        colWidths=[4.5*cm, 12.0*cm],
    )
    yht_tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LGRAY, C_WHITE]),
        ("PADDING",        (0, 0), (-1, -1), 6),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DGRAY),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(yht_tbl)
    story.append(Spacer(1, 4*mm))

    # ── Loppumerkintä ─────────────────────────────────────────────────────────
    story.append(_hr(C_NAVY, 1.0))
    story.append(Paragraph(
        _s(lang, "footer"),
        ParagraphStyle("end", fontSize=7.5, textColor=C_GRAY, alignment=TA_CENTER, leading=11),
    ))

    doc.build(story, canvasmaker=canvas_cls)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Pääfunktio
# ─────────────────────────────────────────────────────────────────────────────

def generate_application_draft(inp: ApplicationInput) -> tuple:
    """Generoi luonnos-PDF ilman oikolukua. Palauttaa (pdf_bytes, sections, sources)."""
    is_bf = inp.hanketyyppi == "business_finland"
    rag_ctx, sources = _rag_context(inp.hanketyyppi, inp.country or "FI")
    if is_bf:
        sections = _generate_bf_sections(inp, rag_ctx)
    else:
        sections = _generate_sections(inp, rag_ctx)
    _lang = inp.lang or "FI"
    sections = {k: _postprocess_text(v, _lang) if isinstance(v, str) else v
                for k, v in sections.items()}
    if is_bf:
        pdf_bytes = _generate_bf_pdf(inp, sections, sources)
    else:
        pdf_bytes = generate_pdf(inp, sections, sources)
    return pdf_bytes, sections, sources


def apply_proofread_to_pdf(inp: ApplicationInput, sections: dict, sources: list) -> bytes:
    """Oikolue sections Claudella ja rakenna lopullinen PDF."""
    _lang = inp.lang or "FI"
    sections = _proofread_sections(sections)
    sections = {k: _postprocess_text(v, _lang) if isinstance(v, str) else v
                for k, v in sections.items()}
    is_bf = inp.hanketyyppi == "business_finland"
    if is_bf:
        return _generate_bf_pdf(inp, sections, sources)
    return generate_pdf(inp, sections, sources)


def generate_application(inp: ApplicationInput) -> str:
    """
    Generoi lupahakemus-PDF (tai BF-hakemus) ja palauta tallennuspolku.
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    is_bf = inp.hanketyyppi == "business_finland"

    print(f"[1/3] Haetaan RAG-konteksti ({inp.hanketyyppi}, maa={inp.country or 'FI'})…")
    rag_ctx, sources = _rag_context(inp.hanketyyppi, inp.country or "FI")
    print(f"      {len(rag_ctx.split())} sanaa, lähteet: {sources}")

    print("[2/4] Generoidaan hakemusteksti (Claude)…")
    if is_bf:
        sections = _generate_bf_sections(inp, rag_ctx)
    else:
        sections = _generate_sections(inp, rag_ctx)
    print(f"      Osiot: {list(sections.keys())}")

    print("[3/4] Oikoluku ja tekstikorjaus (Claude + säännöt)…")
    sections = _proofread_sections(sections)
    _lang = inp.lang or "FI"
    sections = {k: _postprocess_text(v, _lang) if isinstance(v, str) else v
                for k, v in sections.items()}

    print("[4/4] Rakennetaan PDF…")
    if is_bf:
        pdf_bytes = _generate_bf_pdf(inp, sections, sources)
    else:
        pdf_bytes = generate_pdf(inp, sections, sources)

    kt_safe  = inp.kiinteistotunnus.replace("/", "-")
    out_path = os.path.join(_OUTPUT_DIR, f"hakemus_{kt_safe}.pdf")
    with open(out_path, "wb") as f:
        f.write(pdf_bytes)

    print(f"✅ PDF tallennettu: {out_path} ({len(pdf_bytes)//1024} KB)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Testiajo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_inp = ApplicationInput(
        hanketyyppi      = "BESS",
        kiinteistotunnus = "636-439-4-711",
        teho_mw          = 1.0,
        kunta            = "Pöytyä",
        hakija           = "Carbon Zero Finland Oy",
    )
    path = generate_application(test_inp)
    os.system(f"open '{path}'")
