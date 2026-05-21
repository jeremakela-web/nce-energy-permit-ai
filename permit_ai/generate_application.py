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

_POSTPROCESS_RULES: list[tuple[str, str]] = [
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
    # ■■ / ■ -merkit pois (PDF-fontit eivät tue) — poistetaan tai korvataan tekstillä
    (r'■■\s*', ''),
    (r'■\s*',  ''),
    # ⚠️-emoji pois (ei toimi PDF-fonteissa) — säilytetään ⚠ (U+26A0) yksinään jos ok
    (r'⚠️\s*', '[Huom] '),
    # Pelastuslaki virheellinen §-viite: lain numero ≠ pykälänumero
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


def _postprocess_text(text: str) -> str:
    """Korjaa vanhat viranomaisnimet ja lakiviitteet automaattisesti."""
    for pattern, replacement in _POSTPROCESS_RULES:
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

def _rag_context(hanketyyppi: str, n_per_query: int = 4) -> tuple[str, list[str]]:
    """Hae relevantit dokumenttichunkit kaikilla hanketyyppikohtaisilla kyselyillä."""
    cfg = _HANKE_CFG[hanketyyppi]
    try:
        embed_model = _get_embed_model()
        col         = _get_chroma_col()

        seen_ids:    set[str]  = set()
        all_docs:    list[str] = []
        all_sources: set[str]  = set()

        for q in cfg["rag_queries"]:
            emb     = embed_model.encode([q]).tolist()
            results = col.query(query_embeddings=emb, n_results=n_per_query)
            for doc, id_ in zip(results["documents"][0], results["ids"][0]):
                if id_ not in seen_ids:
                    seen_ids.add(id_)
                    all_docs.append(doc)
                    all_sources.add("_".join(id_.split("_")[:-1]))

        context = "\n\n---\n\n".join(all_docs)
        return context, sorted(all_sources)
    except Exception as exc:
        print(f"[RAG] Haku epäonnistui ({exc}) — jatketaan ilman kontekstia")
        return "", []


# ─────────────────────────────────────────────────────────────────────────────
# Claude AI — hakemustekstin generointi
# ─────────────────────────────────────────────────────────────────────────────

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
        "IMPORTANT: Write this entire permit application draft in English. "
        "All section titles, body text, and explanations must be in English. "
        "Legal references may keep Finnish statute numbers but add an English explanation.\n\n"
    ),
    "SE": (
        "VIKTIGT: Skriv hela detta tillståndsansökningsutkast på svenska. "
        "Alla rubrikerna, brödtexten och förklaringarna ska vara på svenska. "
        "Lagstiftningshänvisningar kan behålla de finska lagrummen men lägg till en svensk förklaring.\n\n"
    ),
}

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
    cfg = _HANKE_CFG[inp.hanketyyppi]
    now = datetime.now().strftime("%d.%m.%Y")

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
        viranomainen_ohje = (
            f"\n\nTÄRKEÄÄ: Hakemus osoitetaan viranomaiselle '{inp.kohdeviranomainen}'. "
            "Mukauta hakemuksen sisältö, rakenne ja kieli sen vaatimuksiin sopivaksi. "
            "Viittaa kyseisen viranomaisen ohjeisiin, lomakkeisiin ja vaatimuksiin."
        )

    kap_lisatieto = ""
    if inp.kapasiteetti_mwh and inp.kapasiteetti_mwh > 0:
        kap_lisatieto = f"\nKapasiteetti: {inp.kapasiteetti_mwh} MWh"

    lang_prefix = _LANG_INSTRUCTIONS.get(getattr(inp, "lang", "FI"), "")
    prompt = f"""{lang_prefix}Laadi lupahakemusluonnos seuraavalle hankkeelle:

Hanketyyppi: {inp.hanketyyppi} ({cfg['nimi_fi']})
Kiinteistötunnus: {inp.kiinteistotunnus}
Teho: {inp.teho_mw} MW{kap_lisatieto}
Kunta: {inp.kunta}
Hakija: {inp.hakija}{sijainti_lisatieto}{vaihe_lisatieto}{viranomainen_lisatieto}
Päivämäärä: {now}{viranomainen_ohje}

Alla on relevanttia dokumentaatiota (Fingrid, Tukes, Ympäristöministeriö):
{rag_context}

Kirjoita suomeksi seuraavat neljä osiota selkeästi eroteltuna otsikoilla:

## HANKKEEN KUVAUS
Kirjoita 3–5 kappaleen kuvaus hankkeesta: tarkoitus, tekniset tiedot, sijainti, liityntä verkkoon ja ympäristövaikutukset. Mainitse hanketyypille tyypilliset tekniset parametrit.{' Ota huomioon annettu sijainti- ja ympäristövaikutustieto.' if inp.sijainti_ymparistovaikutukset else ''}

## PERUSTELUT JA TARVE
Kirjoita 2–3 kappaleen perustelu miksi hanke on tarpeellinen (energiajärjestelmän näkökulma, Suomen ilmastotavoitteet, aluetaloudelliset vaikutukset).

## LUPAMENETTELYJEN KUVAUS
Selitä lyhyesti (1–2 lausetta per lupa) mitä kukin tarvittava lupa koskee ja miksi se vaaditaan tälle hankkeelle.{' Viittaa erityisesti kohdeviranomaisen ' + inp.kohdeviranomainen + ' prosesseihin ja vaatimuksiin.' if inp.kohdeviranomainen else ''}

## SEURAAVAT TOIMENPITEET
Ensimmäinen toimenpide on AINA: "Kunnan rakennusvalvonnan ennakkoneuvottelu + kaavatarkastus — Hakija / {inp.kunta}n rakennusvalvonta — 1–2 viikon sisällä".
Listaa sen jälkeen 5 muuta konkreettista askelta aikatauluineen (kk tarkkuudella).{' Ota huomioon hankkeen nykyinen vaihe: ' + inp.hankkeen_vaihe + '.' if inp.hankkeen_vaihe else ''}"""

    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp   = claude.messages.create(
        model=_MODEL_ID,
        max_tokens=8000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text

    # Parsitaan osiot
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

    headers = ["HANKKEEN KUVAUS", "PERUSTELUT JA TARVE", "LUPAMENETTELYJEN KUVAUS", "SEURAAVAT TOIMENPITEET"]
    return {
        "kuvaus":        _extract(raw, "HANKKEEN KUVAUS",       headers[1:]),
        "perustelut":    _extract(raw, "PERUSTELUT JA TARVE",   headers[2:]),
        "luvat_teksti":  _extract(raw, "LUPAMENETTELYJEN KUVAUS", headers[3:]),
        "toimenpiteet":  _extract(raw, "SEURAAVAT TOIMENPITEET",  []),
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
            Paragraph(lupa,         ParagraphStyle("td", fontSize=8.5, leading=12)),
            Paragraph(viranomainen, ParagraphStyle("td", fontSize=8.5, leading=12, textColor=C_BLUE)),
            Paragraph(laki,         ParagraphStyle("td", fontSize=7.5, leading=11, textColor=C_GRAY)),
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
            Paragraph(liite, ParagraphStyle("tl", fontSize=8.5, leading=12)),
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

    lang   = inp.lang or "FI"
    story  = []

    # ── Kansilehti ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(_s(lang, "sub_title"), st["sub"]))
    story.append(Paragraph(f"{cfg['nimi_fi']}", st["title"]))
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
        [_s(lang, "m_hanketyyppi"), f"{inp.hanketyyppi} — {cfg['nimi_fi']}"],
        [_s(lang, "m_teho"),        teho_val],
        [_s(lang, "m_kunta"),       inp.kunta],
        [_s(lang, "m_kt"),          inp.kiinteistotunnus],
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
    laki_rows = {laki for _, _, laki in cfg["luvat"]}
    laki_rows.update(cfg.get("laki_extra", []))
    for ref in sorted(laki_rows):
        story.append(Paragraph(f"• {ref}", st["bullet"]))
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
    rag_ctx, sources = _rag_context(inp.hanketyyppi)
    if is_bf:
        sections = _generate_bf_sections(inp, rag_ctx)
    else:
        sections = _generate_sections(inp, rag_ctx)
    sections = {k: _postprocess_text(v) if isinstance(v, str) else v
                for k, v in sections.items()}
    if is_bf:
        pdf_bytes = _generate_bf_pdf(inp, sections, sources)
    else:
        pdf_bytes = generate_pdf(inp, sections, sources)
    return pdf_bytes, sections, sources


def apply_proofread_to_pdf(inp: ApplicationInput, sections: dict, sources: list) -> bytes:
    """Oikolue sections Claudella ja rakenna lopullinen PDF."""
    sections = _proofread_sections(sections)
    sections = {k: _postprocess_text(v) if isinstance(v, str) else v
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

    print(f"[1/3] Haetaan RAG-konteksti ({inp.hanketyyppi})…")
    rag_ctx, sources = _rag_context(inp.hanketyyppi)
    print(f"      {len(rag_ctx.split())} sanaa, lähteet: {sources}")

    print("[2/4] Generoidaan hakemusteksti (Claude)…")
    if is_bf:
        sections = _generate_bf_sections(inp, rag_ctx)
    else:
        sections = _generate_sections(inp, rag_ctx)
    print(f"      Osiot: {list(sections.keys())}")

    print("[3/4] Oikoluku ja tekstikorjaus (Claude + säännöt)…")
    sections = _proofread_sections(sections)
    sections = {k: _postprocess_text(v) if isinstance(v, str) else v
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
