"""
Energy Permit AI — hakemustengeneraattori.

Generoi lupahakemusluonnoksen PDF-muodossa RAG + Claude -pohjaisesti.
Tukee hanketyypit: BESS | tuulivoima | SMR

Käyttö:
    python3 generate_application.py  (interaktiivinen testiajo)
"""

import io
import os
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
# Tietomalli
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApplicationInput:
    hanketyyppi:      str    # ks. _HANKE_CFG avaimet
    kiinteistotunnus: str
    teho_mw:          float
    kunta:            str
    hakija:           str

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
            ("Ympäristölupa",               "Lupa- ja valvontavirasto (Luova)",  "YSL 527/2014"),
            ("Rakennuslupa",                "Kunta / rakennusvalvonta",          "MRL 132/1999"),
            ("Toimenpideilmoitus pelast.",  "Paikallinen pelastuslaitos",        "Pelastuslaki 379/2011"),
            ("Verkkoliityntäsopimus",       "Jakeluverkkoyhtiö / Fingrid Oyj",   "Sähkömarkkinalaki 588/2013"),
            ("Maa-aineslupa (tarvitt.)",    "Kunta",                             "Maa-aineslaki 555/1981"),
        ],
        "liitteet": [
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
            ("Rakennuslupa",                          "Kunta / rakennusvalvonta", "MRL 132/1999"),
            ("Ympäristölupa (tarvitt.)",              "Luova",                    "YSL 527/2014"),
            ("Verkkoliityntäsopimus",                 "Fingrid Oyj / jakelu",     "Sähkömarkkinalaki 588/2013"),
            ("Lentoestevalolupa",                     "Traficom",                 "Ilmailulaki 864/2014"),
            ("Maanvuokrasopimukset",                  "Maanomistajat",            "Maakaari 540/1995"),
        ],
        "liitteet": [
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
            ("Rakennuslupa",                     "Kunta / rakennusvalvonta", "MRL 132/1999"),
            ("Alusliikenteen turvallisuuslupa",  "Traficom",                 "Merilaki 674/1994"),
            ("Puolustusvoimien lausunto",        "Puolustusvoimat / PLM",    "Laki alueiden käytöstä"),
            ("Verkkoliityntäsopimus",            "Fingrid Oyj",              "Sähkömarkkinalaki 588/2013"),
            ("Maanvuokra / merialueen käyttöoik.", "Valtio / Metsähallitus", "Vesilaki 587/2011"),
        ],
        "liitteet": [
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
            ("Rakennuslupa tai toimenpidelupa", "Kunta / rakennusvalvonta",  "MRL 132/1999 § 125–126"),
            ("Suunnittelutarveratkaisu (tarvitt.)", "Kunta",                 "MRL 132/1999 § 137"),
            ("Ympäristölupa (tarvitt. ≥1 ha)",  "Luova / kunta",            "YSL 527/2014"),
            ("Verkkoliityntäsopimus",           "Jakeluverkkoyhtiö",         "Sähkömarkkinalaki 588/2013"),
            ("Maisema- tai kulttuuriympäristölausunto", "ELY-keskus",        "MRL 197 §"),
        ],
        "liitteet": [
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
            ("Rakennuslupa",                      "Kunta",                      "MRL 132/1999"),
            ("Maankäyttösopimus / kaavoitus",     "Kunta",                      "MRL 132/1999 § 9"),
        ],
        "liitteet": [
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
            ("Rakennuslupa",                     "Kunta / rakennusvalvonta",   "MRL 132/1999"),
            ("Verkkoliityntäsopimus",            "Jakeluverkkoyhtiö / Fingrid", "Sähkömarkkinalaki 588/2013"),
            ("Kalastuslaki-ilmoitus",            "ELY-keskus",                 "Kalastuslaki 379/2015"),
            ("Maankäyttösopimus",                "Kunta / maanomistajat",      "MRL 132/1999"),
        ],
        "liitteet": [
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
            ("Rakennuslupa (tuulivoimala)",          "Kunta / rakennusvalvonta","MRL 132/1999"),
            ("Rakennus-/toimenpidelupa (PV + BESS)", "Kunta",                   "MRL 132/1999 § 126"),
            ("Ympäristölupa (BESS-komponentti)",    "Luova",                    "YSL 527/2014"),
            ("Toimenpideilmoitus pelast. (BESS)",   "Pelastuslaitos",           "Pelastuslaki 379/2011"),
            ("Verkkoliityntäsopimus",               "Fingrid Oyj / jakelu",     "Sähkömarkkinalaki 588/2013"),
            ("Lentoestevalolupa (tuulivoimala)",    "Traficom",                 "Ilmailulaki 864/2014"),
        ],
        "liitteet": [
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
            ("Toimenpideilmoitus pelast. (BESS)",   "Pelastuslaitos",             "Pelastuslaki 379/2011"),
            ("Rakennuslupa",                        "Kunta",                      "MRL 132/1999"),
            ("Vesilupa (jäähdytysvesi, tarvitt.)",  "Luova",                      "Vesilaki 587/2011"),
            ("Verkkoliityntäsopimus",               "Fingrid Oyj",                "Sähkömarkkinalaki 588/2013"),
        ],
        "liitteet": [
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
    "laadinnassa Suomessa. Kirjoitat selkeää, virallista suomen kieltä konsulttiraporttityyliin. "
    "Viittaat aina voimassa olevaan lainsäädäntöön. Et koskaan anna harhaanjohtavaa tietoa — "
    "jos jokin asia on epävarma, merkitset sen selvästi. "
    "Kaikki tuottamasi teksti on AI-luonnos joka vaatii asiantuntijatarkistuksen."
)

def _generate_sections(inp: ApplicationInput, rag_context: str) -> dict[str, str]:
    """
    Kutsu Claude-API ja generoi kaikki hakemuksen osiot yhdellä kutsulla.
    Palauttaa dict: { "kuvaus": ..., "luvat_teksti": ..., "laki": ..., "toimenpiteet": ... }
    """
    cfg = _HANKE_CFG[inp.hanketyyppi]
    now = datetime.now().strftime("%d.%m.%Y")

    prompt = f"""Laadi lupahakemusluonnos seuraavalle hankkeelle:

Hanketyyppi: {inp.hanketyyppi} ({cfg['nimi_fi']})
Kiinteistötunnus: {inp.kiinteistotunnus}
Teho: {inp.teho_mw} MW
Kunta: {inp.kunta}
Hakija: {inp.hakija}
Päivämäärä: {now}

Alla on relevanttia dokumentaatiota (Fingrid, Pelastusopisto, Tukes):
{rag_context}

Kirjoita suomeksi seuraavat neljä osiota selkeästi eroteltuna otsikoilla:

## HANKKEEN KUVAUS
Kirjoita 3–5 kappaleen kuvaus hankkeesta: tarkoitus, tekniset tiedot, sijainti, liityntä verkkoon ja ympäristövaikutukset. Mainitse hanketyypille tyypilliset tekniset parametrit.

## PERUSTELUT JA TARVE
Kirjoita 2–3 kappaleen perustelu miksi hanke on tarpeellinen (energiajärjestelmän näkökulma, Suomen ilmastotavoitteet, aluetaloudelliset vaikutukset).

## LUPAMENETTELYJEN KUVAUS
Selitä lyhyesti (1–2 lausetta per lupa) mitä kukin tarvittava lupa koskee ja miksi se vaaditaan tälle hankkeelle.

## SEURAAVAT TOIMENPITEET
Listaa 5–7 konkreettista seuraavaa askelta aikatauluineen (kk tarkkuudella). Aloita kiireellisimmästä."""

    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp   = claude.messages.create(
        model=_MODEL_ID,
        max_tokens=4000,
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


def _disclaimer_box(st: dict) -> Table:
    """Oranssi AI-varoituslaatikko."""
    C_WARN_BG = colors.HexColor("#fff3e0")
    C_WARN_BD = colors.HexColor("#ff9800")
    row = [[Paragraph(
        "⚠️  AI-LUONNOS — VAATII ASIANTUNTIJATARKISTUKSEN  ⚠️\n"
        "Tämä asiakirja on tekoälyavusteisesti laadittu luonnos. Se ei ole juridisesti sitova eikä korvaa "
        "pätevän lupa-asiantuntijan tai lakimiehen neuvoja. Ennen hakemuksen jättämistä asiakirja on "
        "tarkistutettava alan ammattilaisella.",
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


def _luvat_table(hanketyyppi: str, st: dict) -> Table:
    """Lupa-taulukko hanketyyppikohtaisesti."""
    cfg  = _HANKE_CFG[hanketyyppi]
    rows = [[
        Paragraph("Lupa / ilmoitus", ParagraphStyle("th", fontSize=8.5, fontName="Helvetica-Bold")),
        Paragraph("Viranomainen",    ParagraphStyle("th", fontSize=8.5, fontName="Helvetica-Bold")),
        Paragraph("Lakiperuste",     ParagraphStyle("th", fontSize=8.5, fontName="Helvetica-Bold")),
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


def _liitteet_table(hanketyyppi: str) -> Table:
    """Liiteluettelo checkboxeilla."""
    cfg  = _HANKE_CFG[hanketyyppi]
    rows = [[
        Paragraph("Nro", ParagraphStyle("th2", fontSize=8.5, fontName="Helvetica-Bold")),
        Paragraph("Liite", ParagraphStyle("th2", fontSize=8.5, fontName="Helvetica-Bold")),
        Paragraph("Tila", ParagraphStyle("th2", fontSize=8.5, fontName="Helvetica-Bold",
                                         alignment=TA_CENTER)),
    ]]
    for i, liite in enumerate(cfg["liitteet"], 1):
        rows.append([
            Paragraph(str(i), ParagraphStyle("tn", fontSize=8.5)),
            Paragraph(liite,  ParagraphStyle("tl", fontSize=8.5, leading=12)),
            Paragraph("[ ] Toimitettu",
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


def _para_text(text: str, st: dict) -> list:
    """Muunna AI:n tuottama teksti Paragraph-listaksi (kappalejaot \\n\\n)."""
    items = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
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
            # Poista ** markdown
            clean = para.replace("**", "")
            items.append(Paragraph(clean, st["body"]))
    return items


def _page_footer(canv, doc, inp: ApplicationInput, now: str):
    canv.saveState()
    page_w, _ = A4
    m = 2 * cm
    canv.setStrokeColor(C_DGRAY)
    canv.setLineWidth(0.3)
    canv.line(m, 1.45*cm, page_w - m, 1.45*cm)
    canv.setFont("Helvetica", 6.5)
    canv.setFillColor(C_GRAY)
    canv.drawString(m, 0.9*cm,
        f"{inp.hanketyyppi} lupahakemusluonnos  |  {inp.kiinteistotunnus}  |  {inp.kunta}")
    canv.drawRightString(page_w - m, 0.9*cm,
        f"{now}  |  AI-luonnos — vaatii tarkistuksen  |  Sivu {doc.page}")
    canv.restoreState()


def generate_pdf(inp: ApplicationInput, sections: dict, sources: list[str]) -> bytes:
    """Rakenna PDF ja palauta bytes."""
    buf    = io.BytesIO()
    now    = datetime.now().strftime("%d.%m.%Y")
    cfg    = _HANKE_CFG[inp.hanketyyppi]
    st     = _st()
    margin = 2.2 * cm

    footer_cb = lambda canv, doc: _page_footer(canv, doc, inp, now)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=2.0*cm, bottomMargin=2.2*cm,
    )

    story = []

    # ── Kansilehti ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("Lupahakemusluonnos", st["sub"]))
    story.append(Paragraph(cfg["nimi_fi"], st["title"]))
    story.append(Paragraph(f"{inp.teho_mw} MW  ·  {inp.kunta}  ·  {inp.kiinteistotunnus}", st["meta"]))
    story.append(Spacer(1, 4*mm))
    story.append(_hr(C_NAVY, 1.5))
    story.append(Spacer(1, 3*mm))

    # Metataulukko
    meta_rows = [
        ["Hakija",           inp.hakija],
        ["Hanketyyppi",      f"{inp.hanketyyppi} — {cfg['nimi_fi']}"],
        ["Teho",             f"{inp.teho_mw} MW"],
        ["Sijaintikunta",    inp.kunta],
        ["Kiinteistötunnus", inp.kiinteistotunnus],
        ["Laadittu",         now],
        ["Laatinut",         "NCE Energy Permit AI (tekoälyavusteinen)"],
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
    story.append(_disclaimer_box(st))
    story.append(Spacer(1, 8*mm))

    # ── 1. Hankkeen kuvaus ────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("1. Hankkeen kuvaus", st["h2"]),
        _hr(),
    ]))
    story.extend(_para_text(sections.get("kuvaus", "–"), st))
    story.append(Spacer(1, 4*mm))

    # ── 2. Perustelut ja tarve ────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("2. Perustelut ja tarve", st["h2"]),
        _hr(),
    ]))
    story.extend(_para_text(sections.get("perustelut", "–"), st))
    story.append(Spacer(1, 4*mm))

    # ── 3. Tarvittavat luvat ja viranomaiset ─────────────────────────────────
    story.append(KeepTogether([
        Paragraph("3. Tarvittavat luvat ja viranomaiset", st["h2"]),
        _hr(),
    ]))
    story.append(_luvat_table(inp.hanketyyppi, st))
    story.append(Spacer(1, 5*mm))

    # AI:n lupakuvaukset
    luvat_txt = sections.get("luvat_teksti", "")
    if luvat_txt:
        story.extend(_para_text(luvat_txt, st))
    story.append(Spacer(1, 4*mm))

    # ── 4. Lakiviitteet ───────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("4. Lakiviitteet", st["h2"]),
        _hr(),
    ]))
    laki_rows = list({laki for _, _, laki in cfg["luvat"]})
    for ref in sorted(laki_rows):
        story.append(Paragraph(f"• {ref}", st["bullet"]))
    story.append(Spacer(1, 4*mm))

    # ── 5. Liiteluettelo ──────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("5. Liiteluettelo", st["h2"]),
        _hr(),
    ]))
    story.append(Paragraph(
        "Seuraavat liitteet on toimitettava hakemuksen yhteydessä. "
        "Merkitse ☐-ruutuun kun liite on valmis.",
        st["body"],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(_liitteet_table(inp.hanketyyppi))
    story.append(Spacer(1, 4*mm))

    # ── 6. Seuraavat toimenpiteet ─────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("6. Seuraavat toimenpiteet", st["h2"]),
        _hr(),
    ]))
    story.extend(_para_text(sections.get("toimenpiteet", "–"), st))
    story.append(Spacer(1, 4*mm))

    # ── Lähteet ───────────────────────────────────────────────────────────────
    if sources:
        story.append(KeepTogether([
            Paragraph("Lähteet ja tietolähteet", st["h2"]),
            _hr(),
            Paragraph(
                "Tämä luonnos on laadittu hyödyntäen seuraavia viranomaisdokumentteja:",
                st["body"],
            ),
        ]))
        for s in sources:
            story.append(Paragraph(f"• {s}", st["bullet"]))
        story.append(Spacer(1, 3*mm))

    # ── Loppumerkintä ─────────────────────────────────────────────────────────
    story.append(_hr(C_NAVY, 1.0))
    story.append(Paragraph(
        "NCE Energy Permit AI  ·  ncenergy.fi  ·  info@ncenergy.fi  "
        "·  AI-luonnos — vaatii asiantuntijatarkistuksen",
        ParagraphStyle("end", fontSize=7.5, textColor=C_GRAY, alignment=TA_CENTER, leading=11),
    ))

    doc.build(story, onFirstPage=footer_cb, onLaterPages=footer_cb)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Pääfunktio
# ─────────────────────────────────────────────────────────────────────────────

def generate_application(inp: ApplicationInput) -> str:
    """
    Generoi lupahakemus-PDF ja palauta tallennuspolku.
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    print(f"[1/3] Haetaan RAG-konteksti ({inp.hanketyyppi})…")
    rag_ctx, sources = _rag_context(inp.hanketyyppi)
    print(f"      {len(rag_ctx.split())} sanaa, lähteet: {sources}")

    print("[2/3] Generoidaan hakemusteksti (Claude)…")
    sections = _generate_sections(inp, rag_ctx)
    print(f"      Osiot: {list(sections.keys())}")

    print("[3/3] Rakennetaan PDF…")
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
