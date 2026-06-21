"""
Energy Permit AI — hakemustengeneraattori.

Generoi lupahakemusluonnoksen PDF-muodossa RAG + Claude -pohjaisesti.
Tukee hanketyypit: BESS | tuulivoima | aurinkovoima | SMR | vesivoima | hybridit

Käyttö:
    python3 generate_application.py  (interaktiivinen testiajo)
"""

import io
import logging
import os
import re
import sys
import unicodedata

logger = logging.getLogger(__name__)
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# ── ReportLab ────────────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak, HRFlowable, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.pdfgen.canvas import Canvas as _CanvasBase

# ── TrueType font registration (UTF-8 safe, replaces Latin-1 Helvetica) ──────
_DEJAVU_PATH      = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if os.path.exists(_DEJAVU_PATH):
    pdfmetrics.registerFont(TTFont("DejaVu", _DEJAVU_PATH))
    PDF_FONT = "DejaVu"
else:
    PDF_FONT = "Helvetica"
if os.path.exists(_DEJAVU_BOLD_PATH):
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", _DEJAVU_BOLD_PATH))
    PDF_FONT_BOLD = "DejaVu-Bold"
else:
    PDF_FONT_BOLD = "Helvetica-Bold"

# ── RAG / AI ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Vakiot
# ─────────────────────────────────────────────────────────────────────────────


class InsufficientSourcesError(Exception):
    """Raised when RAG retrieval returns too few or too-low-relevance chunks."""
    def __init__(self, chunks_found: int, avg_relevance: float):
        self.chunks_found   = chunks_found
        self.avg_relevance  = avg_relevance
        super().__init__(
            f"RAG_FAIL: chunks={chunks_found} avg_score={avg_relevance:.2f} — "
            "insufficient sources for a reliable permit draft"
        )


# TODO: domain muutos ncepermit.ai kun NCE Global perustettu
_HERE        = os.path.dirname(os.path.abspath(__file__))
_DB_DIR      = os.path.join(_HERE, "embeddings")
_OUTPUT_DIR  = os.path.join(_HERE, "output")
_LOGO_PATH   = os.path.join(_HERE, "..", "backend", "nce_energy_logo.png")
_MODEL_ID      = "claude-sonnet-4-5"
_MODEL_ID_FAST = "claude-haiku-4-5-20251001"   # oikoluku ja nopeat kutsut
_EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"   # multilingual; switched to v2 at runtime
_EMBED_MODEL_V2 = "paraphrase-multilingual-mpnet-base-v2"
_COLLECTION_V2  = "permit_docs_v2"
_CHROMA_COLLECTION = "permit_docs"              # v1 fallback; switched to v2 at runtime

# Sentinel values sent by the frontend when a field is not applicable
_SENTINEL_VALS = frozenset({
    "EI-SOVELLU", "N/A", "EJ TILLÄMPLIGT", "IKKE RELEVANT", "NIE DOTYCZY",
})


def _clean_kt(kt: str) -> str:
    """Replace frontend sentinel 'EI-SOVELLU' / 'N/A' etc. with dash.
    Normalizes to full zero-padded form: KKK-VVV-NNNN-NNNN (e.g. 108-403-1-1 → 108-403-0001-0001)."""
    if not kt or kt.upper() in {v.upper() for v in _SENTINEL_VALS}:
        return "–"
    parts = kt.split("-")
    if len(parts) == 4:
        return f"{parts[0]}-{parts[1]}-{parts[2].zfill(4)}-{parts[3].zfill(4)}"
    return kt


_LATIN1_CHARMAP: dict[str, str] = {
    "—": "-", "–": "-",   # em-dash, en-dash
    "‘": "'", "’": "'",   # left/right single quote
    "“": '"', "”": '"',   # left/right double quote
    "…": "...", "•": "-", # ellipsis, bullet
    "×": "x",  "−": "-", # multiplication sign, minus sign
    "·": "-",                  # middle dot
}


def _latin1_safe(text: str) -> str:
    """NFC-normalise and return text unchanged.

    When PDF_FONT is a TrueType font (DejaVu), ReportLab handles the full
    Unicode range natively — no Latin-1 transliteration needed.  We keep
    the NFC step so combining diacritics (a+U+0308) are collapsed to
    precomposed ä before they reach the PDF renderer.

    For Helvetica (Latin-1 font, local macOS), encode character-by-character
    so that ä/ö (which ARE valid Latin-1) are always preserved. Only truly
    non-Latin-1 characters (em-dashes, smart quotes, …) are mapped or dropped.
    """
    text = unicodedata.normalize("NFC", text)
    if PDF_FONT != "Helvetica":
        return text  # TrueType font — pass through unchanged
    try:
        text.encode("latin-1")
        return text
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Character-by-character: keep anything encodable as Latin-1 (incl. ä/ö),
        # map known typographic chars, drop the rest.
        out = []
        for ch in text:
            try:
                ch.encode("latin-1")
                out.append(ch)
            except (UnicodeEncodeError, UnicodeDecodeError):
                out.append(_LATIN1_CHARMAP.get(ch, ""))
        return "".join(out)


# Deterministic repair for Finnish words commonly generated without diacritics.
# Pattern tuples: (regex_without_diacritics, correct_form). Applied before proofread.
_FI_DIAK = [
    # jäähdytys-
    (r"jaahdytysjarjestelm([aä])", r"jäähdytysjärjestelmä"),
    (r"jaahdytysteho([na]?)", r"jäähdytysteho\1"),
    (r"jaahdytykse([nlt]|lle|ltä|stä|llä)?", r"jäähdytyks\1" if False else r"jäähdytykse\1"),
    (r"jaahdytyksen", "jäähdytyksen"),
    (r"jaahdytykset", "jäähdytykset"),
    (r"jaahdytyksia", "jäähdytyksiä"),
    (r"jaahdytys", "jäähdytys"),
    # lämpö-
    (r"hukkalamm([oöön])", r"hukkalämmö\1"),
    (r"hukkalammost[aä]", "hukkalämmöstä"),
    (r"hukkalampo([an]?)", r"hukkalämpö\1"),
    (r"kaukolamm([oöön])", r"kaukolämmö\1"),
    (r"kaukolammost[aä]", "kaukolämmöstä"),
    (r"kaukolampo([an]?)", r"kaukolämpö\1"),
    (r"maalampo([an]?)", r"maalämpö\1"),
    (r"lampotila([nsa]?)", r"lämpötila\1"),
    (r"lampo([na]?)\b", r"lämpö\1"),
    # käyttö-
    (r"kaytettavyyden", "käytettävyyden"),
    (r"kaytettavyys", "käytettävyys"),
    (r"kayttoonotoss[aä]", "käyttöönotossa"),
    (r"kayttoonoton", "käyttöönoton"),
    (r"kayttoonotto", "käyttöönotto"),
    (r"kayttoa\b", "käyttöä"),
    (r"kayton\b", "käytön"),
    (r"kaytt[oö]\b", "käyttö"),
    (r"kayttaa\b", "käyttää"),
    # ympäristö-
    (r"ymparistovaikutusten", "ympäristövaikutusten"),
    (r"ymparistovaikutukset", "ympäristövaikutukset"),
    (r"ymparistovaikutuksia", "ympäristövaikutuksia"),
    (r"ymparistoluvasta", "ympäristöluvasta"),
    (r"ymparistoluvan", "ympäristöluvan"),
    (r"ymparistolupa", "ympäristölupa"),
    (r"ympariston\b", "ympäristön"),
    (r"ymparisto\b", "ympäristö"),
    # järjestelmä-
    (r"jarjestelmaan\b", "järjestelmään"),
    (r"jarjestelmassa\b", "järjestelmässä"),
    (r"jarjestelmaa\b", "järjestelmää"),
    (r"jarjestelman\b", "järjestelmän"),
    (r"jarjestelma\b", "järjestelmä"),
    # sähkö-
    (r"sahkoliittyma", "sähköliittymä"),
    (r"sahkoverkko", "sähköverkko"),
    (r"sahkoasema", "sähköasema"),
    (r"sahkoa\b", "sähköä"),
    (r"sahkon\b", "sähkön"),
    (r"sahko\b", "sähkö"),
    # häiriö-
    (r"hairiotilannetta", "häiriötilannetta"),
    (r"hairiotilanne", "häiriötilanne"),
    (r"hairion\b", "häiriön"),
    (r"hairio\b", "häiriö"),
    # päätös-
    (r"paatoksessa", "päätöksessä"),
    (r"paatokset\b", "päätökset"),
    (r"paatosten", "päätösten"),
    (r"paatosta\b", "päätöstä"),
    (r"paatos\b", "päätös"),
    # näkökulma-
    (r"nakokulmat\b", "näkökulmat"),
    (r"nakokulman\b", "näkökulman"),
    (r"nakokulma\b", "näkökulma"),
    # maaperä-
    (r"maaperaan\b", "maaperään"),
    (r"maaperassa\b", "maaperässä"),
    (r"maaperaa\b", "maaperää"),
    (r"maapera\b", "maaperä"),
    # käsittely-
    (r"kasittelyaika", "käsittelyaika"),
    (r"kasitellaan", "käsitellään"),
    (r"kasiteltava", "käsiteltävä"),
    (r"kasittely", "käsittely"),
    # misc high-frequency
    (r"tarkea\b", "tärkeä"),
    (r"tarkeaa\b", "tärkeää"),
    (r"tarkeimmat\b", "tärkeimmät"),
    (r"tarkeinta\b", "tärkeintä"),
    (r"tarkeys\b", "tärkeys"),
    (r"patevyysvaatimus", "pätevyysvaatimus"),
    (r"patevyys\b", "pätevyys"),
    (r"loytaa\b", "löytää"),
    (r"loytyi\b", "löytyi"),
    # työ-
    (r"tyomaalla\b", "työmaalla"),
    (r"tyomaan\b", "työmaan"),
    (r"tyomaa\b", "työmaa"),
    (r"tyontekijat\b", "työntekijät"),
    (r"tyontekija\b", "työntekijä"),
    (r"tyossä\b", "työssä"),
    (r"tyossa\b", "työssä"),
    (r"tyohon\b", "työhön"),
    (r"tyota\b", "työtä"),
    (r"tyon\b", "työn"),
    (r"tyo\b", "työ"),
    # hyödyntää / hyöty
    (r"hyodyntamiseksi\b", "hyödyntämiseksi"),
    (r"hyodyntamista\b", "hyödyntämistä"),
    (r"hyodyntaminen\b", "hyödyntäminen"),
    (r"hyodyntaa\b", "hyödyntää"),
    (r"hyotya\b", "hyötyä"),
    (r"hyodyt\b", "hyödyt"),
    (r"hyoty\b", "hyöty"),
    # määrä / säädös
    (r"maaraysten\b", "määräysten"),
    (r"maaraykset\b", "määräykset"),
    (r"maarays\b", "määräys"),
    (r"maaraan\b", "määrään"),
    (r"maara\b", "määrä"),
    (r"saadosten\b", "säädösten"),
    (r"saadokset\b", "säädökset"),
    (r"saados\b", "säädös"),
    (r"saantely\b", "säätely"),
    # sisältä- / sisältö
    (r"sisaltavat\b", "sisältävät"),
    (r"sisaltaa\b", "sisältää"),
    (r"sisaltoa\b", "sisältöä"),
    (r"sisalto\b", "sisältö"),
    # täyttää / täyttö
    (r"taytettava\b", "täytettävä"),
    (r"tayttamiseksi\b", "täyttämiseksi"),
    (r"tayttaminen\b", "täyttäminen"),
    (r"tayttaa\b", "täyttää"),
    (r"tayttyy\b", "täyttyy"),
    (r"taytto\b", "täyttö"),
    # selvittää / selvitys
    (r"selvittamiseksi\b", "selvittämiseksi"),
    (r"selvittaminen\b", "selvittäminen"),
    (r"selvittaa\b", "selvittää"),
    # liittää / liittymä
    (r"liittamiseksi\b", "liittämiseksi"),
    (r"liittamisesta\b", "liittämisestä"),
    (r"liittaminen\b", "liittäminen"),
    (r"liittaa\b", "liittää"),
    # hyväksyä
    (r"hyvaksyttava\b", "hyväksyttävä"),
    (r"hyvaksytaan\b", "hyväksytään"),
    (r"hyvaksytty\b", "hyväksytty"),
    (r"hyvaksymista\b", "hyväksymistä"),
    (r"hyvaksyminen\b", "hyväksyminen"),
    (r"hyvaksyy\b", "hyväksyy"),
    # ylläpito
    (r"yllapitosuunnitelma\b", "ylläpitosuunnitelma"),
    (r"yllapidon\b", "ylläpidon"),
    (r"yllapito\b", "ylläpito"),
    # jäte-
    (r"jatehuolto\b", "jätehuolto"),
    (r"jateveden\b", "jäteveden"),
    (r"jatevesi\b", "jätevesi"),
    (r"jatteiden\b", "jätteiden"),
    (r"jatteet\b", "jätteet"),
    # järjestelmä — puuttuvat taivutusmuodot
    (r"jarjestelmalla\b",  "järjestelmällä"),
    (r"jarjestelmalle\b",  "järjestelmälle"),
    (r"jarjestelmalta\b",  "järjestelmältä"),
    (r"jarjestelmia\b",    "järjestelmiä"),
    (r"jarjestelmissa\b",  "järjestelmissä"),
    (r"jarjestelmista\b",  "järjestelmistä"),
    (r"jarjestelmat\b",    "järjestelmät"),
    (r"jarjestelmia\b",    "järjestelmiä"),
    # akkujarjestelma / paloilmoitinjarjestelma compounds
    (r"jarjestelm([aäoö]\w*)\b", r"järjestelm\1"),
    # järjestää / järjestely
    (r"jarjestelyt\b", "järjestelyt"),
    (r"jarjestelyn\b", "järjestelyn"),
    (r"jarjestely\b", "järjestely"),
    (r"jarjestetaan\b", "järjestetään"),
    (r"jarjestaa\b", "järjestää"),
    # käynnistää
    (r"kaynnistaminen\b", "käynnistäminen"),
    (r"kaynnistaa\b", "käynnistää"),
    # näyttää / näköala
    (r"nayttaminen\b", "näyttäminen"),
    (r"naytteet\b", "näytteet"),
    (r"nayttaa\b", "näyttää"),
    # pääsy / päästö / pääoma
    (r"paastoja\b", "päästöjä"),
    (r"paastojen\b", "päästöjen"),
    (r"paastot\b", "päästöt"),
    (r"paasylle\b", "pääsylle"),
    (r"paasy\b", "pääsy"),
    # sähkö compounds not yet covered
    (r"sahkonsyoton\b", "sähkönsyötön"),
    (r"sahkonsyotto\b", "sähkönsyöttö"),
    (r"sahkoteho\b", "sähköteho"),
    # lämpö compounds not yet covered
    (r"lampojarjestelma\b", "lämpöjärjestelmä"),
    (r"lampoverkko\b", "lämpöverkko"),
    (r"lampopumppu\b", "lämpöpumppu"),
    (r"lampoenergia\b", "lämpöenergia"),
    # häiriönhallinta
    (r"hairionhallintaa\b", "häiriönhallintaa"),
    (r"hairionhallinta\b", "häiriönhallinta"),
    # yhteydessä
    (r"yhteytta\b", "yhteyttä"),
    (r"yhteydessa\b", "yhteydessä"),
    # BESS-spesifiset: paloturvallisuus-yhdyssanat
    (r"paloturvallisuusselvityksessa\b", "paloturvallisuusselvityksessä"),
    (r"paloturvallisuusselvityksen\b",   "paloturvallisuusselvityksen"),
    (r"paloturvallisuusselvitysta\b",    "paloturvallisuusselvitystä"),
    (r"paloturvallisuusselvitys\b",      "paloturvallisuusselvitys"),
    (r"paloturvallisuusvaatimuksista\b", "paloturvallisuusvaatimuksista"),
    (r"paloturvallisuusvaatimukset\b",   "paloturvallisuusvaatimukset"),
    (r"paloturvallisuusriskeista\b",     "paloturvallisuusriskeistä"),
    (r"paloturvallisuusriskit\b",        "paloturvallisuusriskit"),
    (r"paloturvallisuuteen\b",           "paloturvallisuuteen"),
    (r"paloturvallisuutta\b",            "paloturvallisuutta"),
    # sammutusjarjestelma
    (r"sammutusjarjestelman\b",  "sammutusjarjestelmän"),
    (r"sammutusjarjestelmaa\b",  "sammutusjarjestelmää"),
    (r"sammutusjarjestelma\b",   "sammutusjärjestelmä"),
    # akku- / energia- yhdyssanat
    (r"akkujarjestelmaan\b",     "akkujärjestelmään"),
    (r"akkujarjestelman\b",      "akkujärjestelmän"),
    (r"akkujarjestelma\b",       "akkujärjestelmä"),
    (r"energiavarastoja\b",      "energiavarastoja"),
    (r"energiavarastoon\b",      "energiavarastoon"),
    (r"energiavaraston\b",       "energiavaraston"),
    (r"energiavarastosta\b",     "energiavarastosta"),
    # hallinta
    (r"hallintajarjestelma\b",   "hallintajärjestelmä"),
    (r"hallintajarjestelman\b",  "hallintajärjestelmän"),
    # käyttöönotto-taivutus
    (r"kayttoonottoon\b",        "käyttöönottoon"),
    (r"kayttoonotolta\b",        "käyttöönotolta"),
    (r"kayttoonoton\b",          "käyttöönoton"),
    (r"kayttoonotosta\b",        "käyttöönotosta"),
    # suunnittelu
    (r"suunnittelijan\b",        "suunnittelijan"),
    (r"suunnittelusta\b",        "suunnittelusta"),
    (r"suunnittelussa\b",        "suunnittelussa"),
    # tarkastusasiakirja
    (r"tarkastusasiakirjassa\b", "tarkastusasiakirjassa"),
    (r"tarkastusasiakirjan\b",   "tarkastusasiakirjan"),
    (r"tarkastusasiakirjaa\b",   "tarkastusasiakirjaa"),
]
_FI_DIAK_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _FI_DIAK]


def _fix_fi_diacritics(text: str) -> str:
    """NFC-normalise then apply deterministic diacritics repair.

    Step 1: unicodedata.normalize('NFC') converts combining sequences like
            a + U+0308 → ä (U+00E4).  This alone fixes most AI-generated
            broken diacritics before any regex is needed.
    Step 2: regex patterns catch the remaining cases where the AI simply
            omitted the diacritic entirely (kaytettavyys → käytettävyys).
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace('■', '').replace('■', '')
    for rx, repl in _FI_DIAK_RE:
        text = rx.sub(repl, text)
    return text


# Fallback location labels per hanketyyppi when sijainti field is empty / a long sentence
_HANKETYYPPI_DEFAULT_ALUE: dict[str, str] = {
    "datakeskus":      "Teollisuusalue",
    "BESS":            "Teollisuusalue",
    "aurinkovoima":    "Aurinkoalue",
    "tuulivoima_maa":  "Hankerajojen alue",
    "tuulivoima_meri": "Merialue",
}


def _cover_location(kunta: str, sijainti: str, hanketyyppi: str) -> str:
    """Return 'Kunta[, Alue]' for the cover title meta line.

    Rules:
    - If sijainti contains a short location label (≤20 chars, first comma-segment),
      append it: 'Turku, Teollisuusalue'.
    - Otherwise fall back to the per-hanketyyppi default if one is defined.
    - Never expose raw long description text in the title.
    """
    hint = ""
    if sijainti:
        seg = sijainti.split("\n")[0].split(",")[0].strip()
        if seg and len(seg) <= 20:
            hint = seg
    if not hint:
        hint = _HANKETYYPPI_DEFAULT_ALUE.get(hanketyyppi, "")
    return f"{kunta}, {hint}" if hint else kunta


# Short display names for the PDF cover title line
_HANKE_SHORT: dict[str, str] = {
    "BESS":            "BESS",
    "tuulivoima_maa":  "Tuulivoima (maa)",
    "tuulivoima_meri": "Tuulivoima (meri)",
    "aurinkovoima":    "Aurinkovoima",
    "SMR":             "SMR",
    "vesivoima":       "Vesivoima",
    "hybridi":         "Hybridivoimala",
    "smr_bess":        "SMR+BESS",
    "ymparistolupa":   "Ympäristölupa",
    "datakeskus":      "Datakeskus",
    "asuinrakennus":   "Asuinrakennus",
    "toimitila":       "Toimitila",
    "teollisuus":      "Teollisuus",
    "maatalous":       "Maatalous",
}


@lru_cache(maxsize=1)
def _get_embed_model() -> SentenceTransformer:
    return SentenceTransformer(_EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_chroma_col():
    client = chromadb.PersistentClient(path=_DB_DIR)
    return client.get_or_create_collection(_CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


def activate_v2() -> None:
    """Switch permit application RAG to permit_docs_v2 + mpnet."""
    global _EMBED_MODEL, _CHROMA_COLLECTION
    _EMBED_MODEL = _EMBED_MODEL_V2
    _CHROMA_COLLECTION = _COLLECTION_V2
    _get_embed_model.cache_clear()
    _get_chroma_col.cache_clear()


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
    # Rakennuslupa → Rakentamislupa (Rakentamislaki 751/2023, voimaan 1.1.2025)
    (r'\bRakennus(lu[pv]\w*)\b', r'Rakentamis\1'),
    (r'\brakennus(lu[pv]\w*)\b', r'rakentamis\1'),
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
    text = text.replace('&amp;', '&')
    text = re.sub(r'⚠️\s*', huom, text)
    for pattern, replacement in _POSTPROCESS_RULES_ALL:
        text = re.sub(pattern, replacement, text)
    if lang == "FI":
        for pattern, replacement in _POSTPROCESS_RULES_FI:
            text = re.sub(pattern, replacement, text)
    return text


# Vain sisältöosioiden tekstit lasketaan — kansilehti/disclaimer/footer eivät koskaan
_CONTENT_SECTION_KEYS: frozenset[str] = frozenset({
    "kuvaus", "perustelut", "luvat_teksti", "toimenpiteet",
})


def _limit_expert_reviews(text: str, max_count: int = 3) -> str:
    """Rajoita 'Asiantuntijatarkistus suositellaan' täsmälleen max_count kertaan."""
    phrase = "Asiantuntijatarkistus suositellaan"
    before = text.count(phrase)
    print(f"Expert reviews count BEFORE: {before}")
    parts = text.split(phrase)
    if len(parts) <= max_count + 1:
        print(f"Expert reviews count AFTER:  {before} (ei muutosta)")
        return text
    result = phrase.join(parts[:max_count + 1])
    for part in parts[max_count + 1:]:
        result += part
    after = result.count(phrase)
    print(f"Expert reviews count AFTER:  {after}")
    return result


def _final_polish(sections: dict, lang: str) -> dict:
    """Loppu-oikoluku — suoritetaan AINA viimeisenä ennen PDF-rakennusta.

    1. Deterministinen diakriittikorjaus (ä/ö) kaikille suomenkielisille kentille.
    2. Viranomaistermien ja lakiviitteiden korjaus (_postprocess_text).
    3. Asiantuntijatarkistus-merkintöjen karsinta enintään 3 kappaleeseen.
    """
    result = {}
    for k, v in sections.items():
        if not isinstance(v, str):
            result[k] = v
            continue
        v = _fix_fi_diacritics(v)
        v = _postprocess_text(v, lang)
        result[k] = v

    # Globaali rajoitin: VAIN sisältöosiot (ei kansilehti/disclaimer/footer)
    _SEP = "\x00||SEC||\x00"
    str_keys = [k for k, v in result.items()
                if isinstance(v, str) and k in _CONTENT_SECTION_KEYS]
    if str_keys:
        combined = _SEP.join(result[k] for k in str_keys)
        combined = _limit_expert_reviews(combined, max_count=3)
        for k, part in zip(str_keys, combined.split(_SEP)):
            result[k] = part

    return _limit_huom_markers(result, lang, max_count=3)


_HUOM_PRIORITY_KW = [
    "verkkoliittym", "verkkoliitynt", "kaavoitus", "asemakaava", "yleiskaava",
    "ympäristölupa", "ymparistolupa", "ympäristövaikutus", "ymparistovaikutus",
    "yva", "meluselvitys", "pohjavesi", "natura", "suojelualue",
]


def _limit_huom_markers(sections: dict, lang: str, max_count: int = 4) -> dict:
    """Rajoita epävarmuusmerkintöjen määrä max_count kappaleeseen.

    Priorisoi tärkeät aihepiirit (verkkoliittymä, kaavoitus, ympäristölupa jne.)
    ennen tekstijärjestyksessä ensimmäisiä esiintymiä."""
    huom = _HUOM_LABEL.get(lang, "[Note] ")
    all_markers = []  # (section_key, marker_index_in_section, priority_score)
    for key, val in sections.items():
        if not isinstance(val, str):
            continue
        start = 0
        idx = 0
        while True:
            pos = val.find(huom, start)
            if pos == -1:
                break
            ctx = val[max(0, pos - 300): pos + 100].lower()
            score = sum(1 for kw in _HUOM_PRIORITY_KW if kw in ctx)
            all_markers.append((key, idx, score))
            idx += 1
            start = pos + len(huom)

    if len(all_markers) <= max_count:
        return sections

    # Keep the max_count highest-scoring markers (ties broken by earlier position)
    ranked = sorted(range(len(all_markers)),
                    key=lambda i: (-all_markers[i][2], i))
    keep = {(all_markers[i][0], all_markers[i][1]) for i in ranked[:max_count]}

    result = {}
    for key, val in sections.items():
        if not isinstance(val, str):
            result[key] = val
            continue
        parts = val.split(huom)
        out = [parts[0]]
        for idx, part in enumerate(parts[1:]):
            if (key, idx) in keep:
                out.append(huom + part)
            else:
                out.append(part)
        result[key] = "".join(out)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TASO 2 — AI-oikoluku
# ─────────────────────────────────────────────────────────────────────────────

def _proofread_sections(sections: dict) -> dict:
    """Tarkistuta osiot Claudella ennen PDF-rakennusta."""
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
        "1. DIAKRIITTIMERKIT (korkein prioriteetti): Etsi ja korjaa kaikki puuttuvat tai "
        "väärin kirjoitetut suomalaiset diakriittimerkit. Esimerkkejä: "
        "kaytettavyys→käytettävyys, jaahdytyksen→jäähdytyksen, hairiotilanne→häiriötilanne, "
        "lampotila→lämpötila, ymparistovaikutukset→ympäristövaikutukset, "
        "jarjestelma→järjestelmä, paatos→päätös, tarkea→tärkeä, nakokulmasta→näkökulmasta, "
        "loytyy→löytyy, yhteydenotto→yhteydenotto. Korvaa AINA a→ä ja o→ö silloin kun "
        "suomen kieli niin vaatii.\n"
        "2. Korjaa muut kirjoitusvirheet ja kielioppivirheet.\n"
        "3. Varmista viranomaisten nimet vuodelle 2026: "
        "käytä 'Lupa- ja valvontavirasto' (ei AVI), 'ELY-keskus'.\n"
        "4. Tarkista lakiviitteet: Rakentamislaki (751/2023), ei pelkkä MRL 132/1999.\n"
        "5. Varmista kappaleiden selkeä järjestys ja ammattimainen yleiskieli.\n"
        "6. ÄLÄ lisää kommentteja tai selityksiä tekemistäsi muutoksista.\n\n"
        "Palauta teksti TÄSMÄLLEEN samassa muodossa (===OSIO:key=== -jakajat mukaan lukien), "
        "vain korjattuna."
    )
    try:
        resp = anthropic.Anthropic().messages.create(
            model=_MODEL_ID,
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}],
        )
        corrected = unicodedata.normalize("NFC", resp.content[0].text)
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
    # IFC-esitäyttö (valinnainen) — täyttää generate_pdf:n sections-diktiin
    ifc_floor_area:               float = 0.0   # m²
    ifc_building_height:          float = 0.0   # m
    ifc_fire_rating:              str = ""
    ifc_materials:                str = ""      # pilkulla erotettu lista
    ifc_storeys:                  int = 0
    ifc_compliance_flags:         str = ""      # rivinvaihdolla erotettu lista

# ─────────────────────────────────────────────────────────────────────────────
# Hanketyyppikohtaiset asetukset
# ─────────────────────────────────────────────────────────────────────────────

_HANKE_CFG = {
    "BESS": {
        "nimi_fi":    "Akkuenergiavarastohanke",
        "lyhenne":    "BESS",
        "kasittelyaika": {"FI": "6–18 kk", "EN": "6–18 months"},
        "rag_queries": [
            "BESS akkuvarasto ympäristölupa paloturvallisuusvaatimukset sijoittaminen",
            "litiumioniakku sammutusvedet pohjavesialue ympäristölupa",
            "akkuvarasto verkkoliityntä Fingrid SJV VJV vaatimukset",
        ],
        "luvat": [
            ("Ympäristölupa",                   "Lupa- ja valvontavirasto (Luova)",  "YSL 527/2014"),
            ("Rakentamislupa",                    "Kunta / rakennusvalvonta",          "Rakentamislaki 751/2023"),
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
            "Maankäyttöselvitys PDF (NCE)",
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
        "context_extra": (
            "BESS (AKKUENERGIAVARASTOHANKE) — PAKOLLISET SISÄLTÖVAATIMUKSET:\n\n"
            "OSIO 1 — Hankkeen kuvaus: sisällytä KAIKKI seuraavat tiedot:\n"
            "- Akkukonttien lukumäärä: arvioi 2–4 konttia riippuen toimittajasta "
            "(esim. CATL, BYD, Wärtsilä); yksittäisen kontin kapasiteetti 2,5–5 MWh\n"
            "- Palokuormitusluokka: P3 (Suomen rakentamismääräyskokoelma E2) — "
            "litiumioniakkukontit kuuluvat palokuormitusluokkaan P3\n"
            "- Paloturvallisuus: kirjoita YKSI selkeä kappale jossa mainitaan P3-luokka, "
            "automaattinen sammutusjärjestelmä ja pelastussuunnitelma — "
            "ÄLÄ toista samaa paloturvallisuusasiaa useaan kertaan eri kappaleissa\n"
            "- Viittaa paloturvallisuusselvitykseen täsmälleen näin: "
            "'ks. Liite 5: Paloturvallisuusselvitys'\n"
            "- Viittaa sähköliityntäsuunnitelmaan täsmälleen näin: "
            "'ks. Liite 8: Sähköliitynnän suunnitelma'\n"
            "- Naapurikuuleminen: mainitse sen status (tehty / kesken / tulossa) "
            "ja viittaa [Rakentamislaki 751/2023, 44 §]\n\n"
            "OSIO 3 — Luvat: mainitse erikseen omana kohtanaan:\n"
            "- Kemikaali-ilmoitusvelvollisuus Tukesille: kynnysarvo on 333 kg litiumia "
            "(vastaa noin 1,5–2 MWh LFP-teknologialla) — arvioi onko 10 MWh-hanke "
            "selvästi yli kynnyksen ja edellyttää kemikaaliturvallisuuslain "
            "(390/2005) mukaista lupaa\n"
            "- Pelastussuunnitelman hyväksyttäminen pelastuslaitoksella "
            "[Pelastuslaki 379/2011, 15 §] ennen käyttöönottoa"
        ),
        "context_extra_phases": {
            "rakentaminen": (
                "BESS RAKENTAMISVAIHE — LISÄVAATIMUKSET:\n\n"
                "OSIO 1 — Hankkeen kuvaus: kuvaile rakentamisvaiheen toteutus:\n"
                "- Aloitusilmoitus rakennusvalvonnalle on jätettävä ennen töiden aloittamista "
                "[Rakentamislaki 751/2023]\n"
                "- Vastaava työnjohtaja on nimettävä ja hyväksytettävä rakennusvalvonnassa "
                "ennen aloitusilmoituksen jättämistä\n"
                "- Tarkastusasiakirja: pitäjä = vastaava työnjohtaja; asiakirja on oltava "
                "käytössä koko rakentamisen ajan ja toimitetaan loppukatselmuksessa\n"
                "- Viittaa myönnettyyn rakentamislupaan ja sen ehtoihin\n\n"
                "OSIO 6 — Seuraavat toimenpiteet: järjestä katselmusaikataulu:\n"
                "- Pohjakatselmus ennen perustustöiden aloittamista — rakennusvalvonta\n"
                "- Rakennekatselmus runkovaiheen jälkeen — rakennusvalvonta\n"
                "- Loppukatselmus: rakennusvalvonta + sähköturvallisuustarkastus "
                "(Tukes-valtuutettu laitos) + pelastuslaitoksen hyväksyntä paloturvallisuusjärjestelmille\n"
                "- Kaupallinen käyttöönotto vasta kaikkien katselmuksien ja loppukatselmuksen "
                "hyväksynnän jälkeen [Rakentamislaki 751/2023, 87 §]"
            ),
        },
    },
    "tuulivoima_maa": {
        "nimi_fi":    "Maalle sijoitettava tuulivoimahanke",
        "lyhenne":    "WPP-maa",
        "kasittelyaika": {"FI": "4–7 vuotta", "EN": "4–7 years"},
        "rag_queries": [
            "tuulivoima YVA ympäristövaikutusten arviointi maa lupa",
            "tuulivoimala kaava suunnittelutarveratkaisu meluselvitys linnusto",
            "tuulivoima Fingrid verkkoliityntä kantaverkko vaatimukset",
        ],
        "luvat": [
            ("YVA-menettely (≥10 MW / ≥5 voimalaa)", "ELY-keskus / Luova",       "YVA-laki 252/2017"),
            ("Osayleiskaava tai asemakaava",          "Kunta",                    "MRL 132/1999 § 77a"),
            ("Rakentamislupa",                         "Kunta / rakennusvalvonta", "Rakentamislaki 751/2023"),
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
            "Maankäyttöselvitys PDF (NCE)",
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
        "context_extra": (
            "TUULIVOIMA (MAA) — PAKOLLISET SISÄLTÖVAATIMUKSET:\n\n"
            "OSIO 1 — Hankkeen kuvaus: sisällytä KAIKKI seuraavat tiedot:\n"
            "- Turbiinien napakorkeusarvio n. 150–180 m, kokonaiskorkeus (roottorin kärki) n. 220–250 m\n"
            "- Varjostus (shadow flicker): kuvaa varjostusvaikutukset lähikiinteistöille ja mainitse "
            "automaattinen varjostuksenhallintajärjestelmä (STF-ohjaus)\n"
            "- Standardit erikseen: IEC 61400-1 (rakennesuunnittelun kuormat) ja "
            "IEC 61400-11 (äänitehotason mittaukset) — molemmat nimeltä\n"
            "- Luontoselvitykset: pesimälinnusto, muuttolinnusto ja lepakoiden "
            "lentoaktiviteetti — selvitykset tehdään ennen YVA-ohjelman jättämistä\n\n"
            "OSIO 3 — Luvat: kuvaa YVA-prosessin kulku vaiheistettuna:\n"
            "Vaihe 1: YVA-ohjelma → ELY-keskus antaa lausunnon → julkinen kuuleminen (45 pv)\n"
            "Vaihe 2: YVA-selostus → ELY-keskus antaa perustellun päätelmän\n"
            "Vaihe 3: Osayleiskaava (MRL 132/1999 § 77a) — kulkee yleensä rinnakkain YVA:n kanssa; "
            "pakollinen edellytys tuulivoimarakentamiselle\n"
            "Vaihe 4: Rakentamislupa vasta lainvoimaisen kaavan ja YVA-päätelmän jälkeen\n"
            "Mainitse myös Traficomin lentoestevalolupa (Ilmailulaki 864/2014) omana kohtanaan."
        ),
    },
    "tuulivoima_meri": {
        "nimi_fi":    "Merelle sijoitettava tuulivoimahanke (offshore)",
        "lyhenne":    "WPP-meri",
        "kasittelyaika": {"FI": "7–12 vuotta", "EN": "7–12 years"},
        "rag_queries": [
            "tuulivoima meri offshore lupa ympäristölupa",
            "tuulivoima YVA vesialue vesiliikenne Traficom",
            "tuulivoima Fingrid verkkoliityntä merikaapeli",
        ],
        "luvat": [
            ("YVA-menettely",                    "ELY-keskus / Luova",       "YVA-laki 252/2017"),
            ("Vesilupa",                         "Luova",                    "Vesilaki 587/2011"),
            ("Ympäristölupa",                    "Luova",                    "YSL 527/2014"),
            ("Rakentamislupa",                    "Kunta / rakennusvalvonta", "Rakentamislaki 751/2023"),
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
            "Maankäyttöselvitys PDF (NCE)",
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
        "kasittelyaika": {"FI": "1–3 vuotta", "EN": "1–3 years"},
        "rag_queries": [
            "aurinkovoima aurinkopaneeli rakentamislupa ympäristölupa",
            "aurinkovoimala verkkoliityntä jakeluverkko vaatimukset",
            "aurinkovoimala maankäyttö kaava maisema",
        ],
        "luvat": [
            ("Rakentamislupa",                  "Kunta / rakennusvalvonta",  "Rakentamislaki 751/2023"),
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
        "kuvaus_extra_inst": (
            "Kirjoita vähintään 5 kappaletta. Sisällytä KAIKKI: "
            "(1) Paneeliteknologia — monikidepii (mc-Si) ~20–22 % hyötysuhde vs. ohutkalvo ~12–15 %; "
            "(2) Suuntaus ja kallistuskulma — etelään, 30–35 astetta optimaalinen Suomessa; "
            "(3) Invertterit — string-invertteri vs. mikroinvertterit, DC/AC-muunto, MPPT-säätö; "
            "(4) Maankäyttö — noin 1–1,5 ha/MW eli 10–15 ha yhteensä; "
            "(5) Häikäisyanalyysi — naapurikiinteistöt ja liikenneväylät, selvitys rakennusluvan liite; "
            "(6) Seurantajärjestelmä (tracker) — yksiakselinen +15–25 %, kaksiakselinen +25–35 % tuotantolisä."
        ),
        "perustelut_extra_inst": (
            " Kirjoita 4–5 kappaletta. Käsittele erikseen: Suomen aurinkoenergiapotentiaali ja "
            "vuotuinen säteilymäärä Etelä-Suomessa (~1 000 kWh/m²/a), hankkeen vuosituotantoarvio "
            "(noin 9–11 GWh/a 10 MW laitokselle), CO₂-päästövähennysvaikutus, "
            "aluetaloudelliset hyödyt rakennusvaiheen aikana sekä sähkömarkkinanäkymät ja PPA-sopimukset."
        ),
        "luvat_extra_inst": (
            " Selitä jokainen lupa 2–3 lauseella sisältäen: hakemuksen sisältövaatimukset, "
            "käsittelyaika-arvio ja vastuuviranomaiset. Korosta suunnittelutarveratkaisun "
            "ja häikäisyselvityksen merkitystä aurinkopuistohankkeelle."
        ),
    },
    "SMR": {
        "nimi_fi":    "Pienydinreaktori (SMR) — ennakkolupahakemus",
        "lyhenne":    "SMR",
        "kasittelyaika": {"FI": "10–15 vuotta", "EN": "10–15 years"},
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
            ("Rakentamislupa",                     "Kunta",                      "Rakentamislaki 751/2023"),
            ("Naapurikuuleminen",                 "Kunta / hakija",              "Rakentamislaki 751/2023, 44 §"),
            ("Maankäyttösopimus / kaavoitus",     "Kunta",                      "MRL 132/1999 § 9"),
        ],
        "laki_extra": [
            "Säteilylaki 859/2018",
            "Luonnonsuojelulaki 9/2023",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE)",
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
        "kasittelyaika": {"FI": "5–10 vuotta", "EN": "5–10 years"},
        "rag_queries": [
            "vesivoima vesivoimala vesilupa rakentaminen",
            "vesistö pato padotus vesirakentaminen ympäristölupa",
            "vesivoima kalakannat ekologinen virtaama vesistö",
        ],
        "luvat": [
            ("Vesilupa (padotus, rakentaminen)", "Luova",                      "Vesilaki 587/2011 § 3:2"),
            ("Ympäristölupa",                    "Luova",                      "YSL 527/2014"),
            ("YVA-menettely (tarvitt.)",          "ELY-keskus / Luova",        "YVA-laki 252/2017"),
            ("Rakentamislupa",                    "Kunta / rakennusvalvonta",   "Rakentamislaki 751/2023"),
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
            "Maankäyttöselvitys PDF (NCE)",
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
        "kasittelyaika": {"FI": "4–8 vuotta", "EN": "4–8 years"},
        "rag_queries": [
            "BESS akkuvarasto ympäristölupa paloturvallisuus litiumioniakku",
            "tuulivoima aurinkovoima YVA lupa kaava meluselvitys",
            "hybridivoimala verkkoliityntä Fingrid SJV VJV",
        ],
        "luvat": [
            ("YVA-menettely (kynnyksen ylittyessä)", "ELY-keskus / Luova",      "YVA-laki 252/2017"),
            ("Osayleiskaava / asemakaava",           "Kunta",                   "MRL 132/1999"),
            ("Rakentamislupa (tuulivoimala)",         "Kunta / rakennusvalvonta","Rakentamislaki 751/2023"),
            ("Naapurikuuleminen",                   "Kunta / hakija",          "Rakentamislaki 751/2023, 44 §"),
            ("Rakentamislupa (PV + BESS)",           "Kunta",                   "Rakentamislaki 751/2023"),
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
            "Maankäyttöselvitys PDF (NCE)",
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
        "nimi_fi":    "SMR + BESS -hybridienergiajärjestelmä",
        "lyhenne":    "SMR+BESS",
        "kasittelyaika": {"FI": "10–15 vuotta", "EN": "10–15 years"},
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
            ("Rakentamislupa",                       "Kunta",                      "Rakentamislaki 751/2023"),
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
            "Maankäyttöselvitys PDF (NCE)",
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
    "ymparistolupa": {
        "nimi_fi":    "Ympäristölupahakemus",
        "lyhenne":    "YL",
        "kasittelyaika": {"FI": "3–12 kk", "EN": "3–12 months"},
        "rag_queries": [
            "ympäristölupa lupahakemus YSL 527/2014 luvantarve Luova toiminta",
            "ympäristölupa hakemuksen sisältö selvitykset liitteet ympäristövaikutukset",
            "ympäristönsuojelulaki maaperän pilaantuminen pohjavesi päästöt ilmanlaatu melu",
        ],
        "luvat": [
            ("Ympäristölupa",                     "Lupa- ja valvontavirasto (Luova)", "YSL 527/2014"),
            ("Ympäristövaikutusten arviointi (YVA)", "ELY-keskus / Luova",            "YVA-laki 252/2017"),
            ("Rekisteröinti-ilmoitus (tarvitt.)", "Kunta",                            "YSL 527/2014, 10 §"),
            ("Vesilupa (tarvitt.)",               "Lupa- ja valvontavirasto (Luova)", "Vesilaki 587/2011"),
            ("Meluilmoitus (tarvitt.)",           "Kunta",                            "YSL 527/2014, 118 §"),
        ],
        "laki_extra": [
            "Luonnonsuojelulaki 9/2023",
            "Jätelaki 646/2011",
            "Kemikaalilaki 599/2013",
            "Terveydensuojelulaki 763/1994",
            "Maankäyttö- ja rakennuslaki 132/1999 / Rakentamislaki 751/2023",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE)",
            "Luvan lomake — Luova (tai Luovan sähköinen hakemuspohja)",
            "Toimintakuvaus ja prosessikaavio",
            "Ympäristövaikutusten selvitys (maaperä, pohjavesi, ilma, melu, tärinä)",
            "Päästöluettelo ja mittaustulokset (ilma, vesi, maaperä)",
            "Jätteen käsittely- ja varastointisuunnitelma",
            "Pohjavesialueen kartta (SYKE:n tietojärjestelmä)",
            "Naapuritilojen omistajatiedot (kiinteistörekisteri)",
            "Poikkeustilanteiden toimintaohje",
            "YVA-selostus tai perustelut YVA:n soveltumattomuudesta",
            "Hakijan oikeushenkilön rekisteriote (kaupparekisteri)",
            "Valtakirja (jos asiamies edustaa)",
        ],
    },
    "datakeskus": {
        "nimi_fi":    "Datakeskushanke",
        "lyhenne":    "DC",
        "kasittelyaika": {"FI": "2–5 vuotta", "EN": "2–5 years"},
        "rag_queries": [
            "datakeskus rakentamislupa ympäristölupa jäähdytys meluhaitat",
            "datakeskus sähköliityntä Fingrid kantaverkko kapasiteetti",
            "datakeskus kaavoitus asemakaavanmuutos YVA ympäristövaikutukset",
        ],
        "context_extra": (
            "DATAKESKUS — OSIOKOHTAISET SISÄLTÖVAATIMUKSET:\n\n"
            "HANKKEEN KUVAUS -osiossa TÄYTYY olla kaikki seuraavat:\n"
            "1) Hankkeen tarkoitus ja liiketoiminnallinen perustelu\n"
            "2) Tekniset parametrit: IT-kuorma (käytä annettua teho-arvoa MW), "
            "arvioitu kokonaiskulutus (IT-kuorma × 1,3 = kokonaisteho MW), PUE-tavoite 1,3\n"
            "3) Jäähdytysratkaisu: Free Cooling (ulkoilmajäähdytys) toimii Turun kylmässä "
            "ilmastossa suurimman osan vuodesta — ei mekaanista jäähdytystä tarvita talvella\n"
            "4) Hukkalämmön hyödyntäminen: paluulämpö ~25–35 °C palautetaan Turun "
            "kaukolämpöverkkoon (Turku Energia / Fortum) — merkittävä ympäristöetu\n"
            "5) Sijaintiedut: teollisuusalue, liikenneyhteydet, olemassa oleva infrastruktuuri\n\n"
            "PERUSTELUT-osiossa TÄYTYY olla:\n"
            "1) Digitalisaation kasvava kapasiteettitarve Suomessa\n"
            "2) Energiatehokkuus: PUE 1,3 on hyvä taso, Free Cooling säästää energiaa\n"
            "3) Hukkalämmön hyödyntäminen kaukolämpöön = konkreettinen hiilineutraaliushyöty\n"
            "4) Turku on ihanteellinen sijainti: kylmä ilmasto, teollisuusinfra, satamalogistiikka\n\n"
            "SEURAAVAT TOIMENPITEET -osiossa TÄYTYY olla täsmälleen 6 vaihetta:\n"
            "1. Ennakkoneuvottelu rakennusvalvonta – Lupakonsultti / NCE – 1–2 vk\n"
            "2. Asemakaavanmuutos-selvitys – Projektipäällikkö / NCE – 1–3 kk\n"
            "3. YVA-harkinta ELY-keskuksen kanssa – Lupakonsultti / NCE – 2–4 kk\n"
            "4. Verkkoliittymäneuvottelu Turku Energia + Fingrid – IT-arkkitehti / Hakija – 3–6 kk\n"
            "5. Rakentamislupahakemus – Lupakonsultti / NCE – 6–12 kk\n"
            "6. Ympäristölupahakemus (jäähdytys ja melu) – Lupakonsultti / NCE – 6–12 kk\n\n"
            "VERKKOLIITYNTÄ — käytä tätä tarkkaa muotoilua: "
            "'Jakeluverkkoliittymä (alle 110 kV) solmitaan Turku Energian kanssa. "
            "Mahdollinen kantaverkkoliittymä (110 kV) Fingrid Oyj:n kanssa.'\n\n"
            "YVA — käytä tätä tarkkaa muotoilua: "
            "'[teho] MW datakeskus ei automaattisesti ylitä YVA-kynnysarvoa, mutta "
            "tapauskohtainen harkinta tehdään ELY-keskuksessa.' (korvaa [teho] hankkeen teholla)"
        ),
        "luvat": [
            ("Rakentamislupa",                     "Kunta / rakennusvalvonta",    "Rakentamislaki 751/2023"),
            ("Asemakaavanmuutos (tarvitt.)",        "Kunta + ELY-keskus",          "MRL 132/1999"),
            ("YVA-harkinta (tapauskohtainen)",       "ELY-keskus / Luova",          "YVA-laki 252/2017"),
            ("Ympäristölupa (jäähdytys, melu)",    "Lupa- ja valvontavirasto",    "YSL 527/2014"),
            ("Naapurikuuleminen",                   "Kunta / hakija",              "Rakentamislaki 751/2023, 44 §"),
            ("Verkkoliityntäsopimus (jakelu)",      "Turku Energia (DSO)",         "Sähkömarkkinalaki 588/2013"),
            ("Verkkoliityntäsopimus (kantaverkko)", "Fingrid Oyj (TSO)",           "Sähkömarkkinalaki 588/2013"),
        ],
        "laki_extra": [
            "Luonnonsuojelulaki 9/2023",
            "Meluselvitysasetus 993/1992",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE)",
            "Asemapiirustus ja pohjakartta (M 1:500)",
            "Rakennussuunnitelmat (tekninen tila, jäähdytysjärjestelmät)",
            "Meluselvitys (jäähdytys- ja aggregaattimelu)",
            "Jäähdytyksen lämpökuormaselvitys",
            "Sähköjärjestelmäsuunnitelma (UPS, varavoima, liityntä)",
            "Tulipalonsammutus- ja paloturvallisuussuunnitelma",
            "Verkkoliityntälaskelma (Fingrid kapasiteettiselvitys)",
            "Ympäristövaikutusten arviointi (tarvittaessa)",
            "PUE- ja energiatehokkuusselvitys",
            "Hakijan rekisteriote",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# RAG-haku
# ─────────────────────────────────────────────────────────────────────────────

def _rag_context(
    hanketyyppi: str,
    country: str = "FI",
    n_per_query: int = 2,
) -> tuple[str, list[dict], bool, list[str], list[str]]:
    """Hae relevantit dokumenttichunkit.

    Jos country != 'FI', haetaan ensin maakohtaiset dokumentit ja täydennetään
    FI-dokumenteilla (suomalainen lainsäädäntö on aina relevanttia kontekstia).
    Graceful fallback: jos metadata-suodatus epäonnistuu, haetaan ilman suodatinta.

    Palauttaa (context_text, sources, warning_flag, precedent_chunks, precedent_sources).
    Nostaa InsufficientSourcesError jos RAG-laatu on riittämätön (hard stop).
    """
    cfg = _HANKE_CFG[hanketyyppi]
    try:
        embed_model = _get_embed_model()
        col         = _get_chroma_col()

        seen_ids:        set[str]        = set()
        all_docs:        list[str]       = []
        all_distances:   list[float]     = []
        all_source_meta: dict[str, dict] = {}  # src_id → {display, url}

        def _collect(results: dict) -> None:
            docs      = results["documents"][0]
            ids       = results["ids"][0]
            metas     = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            if not metas:
                metas = [{}] * len(ids)
            if not distances:
                distances = [0.5] * len(ids)
            for doc, id_, meta, dist in zip(docs, ids, metas, distances):
                if id_ not in seen_ids:
                    seen_ids.add(id_)
                    all_docs.append(doc)
                    all_distances.append(dist)
                    src_id = re.sub(r"[_-]\d+$", "", id_)
                    if src_id not in all_source_meta:
                        meta = meta or {}
                        all_source_meta[src_id] = {
                            "display": meta.get("source", src_id),
                            "url":     meta.get("url"),
                        }

        # Country-specific queries for step 1 — avoids Finnish cross-lingual distance penalty.
        # Falls back to Finnish cfg queries if no override defined for this country+type.
        _country_queries = (
            _COUNTRY_RAG_QUERIES.get(country, {}).get(hanketyyppi)
            or cfg["rag_queries"]
        )

        # Step 1: country-specific retrieval using native-language queries (once, before FI loop)
        if country != "FI":
            try:
                for cq in _country_queries:
                    cemb = embed_model.encode([cq]).tolist()
                    _collect(col.query(
                        query_embeddings=cemb,
                        n_results=n_per_query,
                        where={"country": {"$in": [country, "EU"]}},
                    ))
            except Exception:
                pass  # maakohtaisia dokumentteja ei vielä indeksoitu

        # Step 2: FI retrieval using Finnish cfg queries (always; provides base context)
        first_emb = None
        for q in cfg["rag_queries"]:
            emb = embed_model.encode([q]).tolist()
            if first_emb is None:
                first_emb = emb

            try:
                _collect(col.query(
                    query_embeddings=emb,
                    n_results=n_per_query,
                    where={"country": {"$in": ["FI", "EU"]}},
                ))
            except Exception:
                # Vanha indeksi ilman metadataa — hae ilman suodatinta
                try:
                    _collect(col.query(query_embeddings=emb, n_results=n_per_query))
                except Exception:
                    pass

        # ── Task 2: RAG confidence check ─────────────────────────────────────
        chunks_returned = len(all_docs)
        if all_distances:
            relevance_scores = [max(0.0, 1.0 - d) for d in all_distances]
            avg_score = sum(relevance_scores) / len(relevance_scores)
        else:
            avg_score = 0.0

        if chunks_returned < 5 or avg_score < 0.65:
            logger.warning(
                "RAG_FAIL: %s %s chunks=%d avg_score=%.2f",
                hanketyyppi, country, chunks_returned, avg_score,
            )
            raise InsufficientSourcesError(chunks_returned, avg_score)
        elif chunks_returned < 12 or avg_score < 0.75:
            warning_flag = True
            logger.warning(
                "RAG_WARN: %s %s chunks=%d avg_score=%.2f",
                hanketyyppi, country, chunks_returned, avg_score,
            )
        else:
            warning_flag = False

        # ── Task 3: Precedent retrieval ───────────────────────────────────────
        precedent_chunks:  list[str] = []
        precedent_sources: list[str] = []
        if first_emb:
            try:
                prec_results = col.query(
                    query_embeddings=first_emb,
                    n_results=5,
                    where={"$and": [
                        {"country": {"$in": [country, "EU"]}},
                        {"doc_type": "case_law"},
                    ]},
                )
                prec_docs   = prec_results["documents"][0] if prec_results["documents"] else []
                prec_metas  = (prec_results.get("metadatas") or [[]])[0]
                for doc, meta in zip(prec_docs, prec_metas):
                    meta = meta or {}
                    precedent_chunks.append(doc)
                    precedent_sources.append(meta.get("source", "Viranomainen"))
            except Exception:
                pass  # case_law-metadata ei indeksoitu tai yhteensopivuusvirhe

        context = "\n\n---\n\n".join(all_docs)
        sources = [{"id": sid, **info} for sid, info in sorted(all_source_meta.items())]
        return context, sources, warning_flag, precedent_chunks, precedent_sources

    except InsufficientSourcesError:
        raise
    except Exception as exc:
        print(f"[RAG] Haku epäonnistui ({exc}) — jatketaan ilman kontekstia")
        return "", [], False, [], []


def _statutory_sources(hanketyyppi: str, country: str = "FI") -> list[str]:
    """Palauta lakiviitteet hankkeelle (luvat + laki_extra), maakohtainen override ensin."""
    country_override = _COUNTRY_LUVAT.get(country, {}).get(hanketyyppi)
    if country_override:
        refs = {laki for _, _, laki in country_override}
    else:
        cfg  = _HANKE_CFG[hanketyyppi]
        refs = {laki for _, _, laki in cfg["luvat"]}
        refs.update(cfg.get("laki_extra", []))
    return sorted(refs)


# ─────────────────────────────────────────────────────────────────────────────
# Claude AI — hakemustekstin generointi
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Hanketyyppien nimet muilla kielillä (meta-taulukko PDF:ssä)
# ─────────────────────────────────────────────────────────────────────────────
_HANKE_NIMI_TRANS: dict[str, dict[str, str]] = {
    "BESS":           {"EN": "Battery Energy Storage System (BESS)",         "SE": "Batterienergilagringssystem (BESS)",
                       "DA": "Batterienergilagringssystem (BESS)",            "NO": "Batterienergilagringssystem (BESS)",
                       "PL": "System magazynowania energii w akumulatorach (BESS)"},
    "tuulivoima_maa": {"EN": "Onshore Wind Power Project",                   "SE": "Landbaserat vindkraftsprojekt",
                       "DA": "Landbaseret vindkraftsprojekt",                  "NO": "Landbasert vindkraftprosjekt",
                       "PL": "Lądowy projekt farmy wiatrowej"},
    "tuulivoima_meri":{"EN": "Offshore Wind Power Project",                  "SE": "Offshorevindkraftsprojekt",
                       "DA": "Offshore-vindkraftsprojekt",                    "NO": "Offshore-vindkraftprosjekt",
                       "PL": "Morski projekt farmy wiatrowej"},
    "aurinkovoima":   {"EN": "Solar Power Plant Project",                    "SE": "Solkraftsprojekt",
                       "DA": "Solkraftværksprojekt",                          "NO": "Solkraftverksprosjekt",
                       "PL": "Projekt elektrowni słonecznej"},
    "SMR":            {"EN": "Small Modular Reactor (SMR) — pre-licensing",  "SE": "Liten modulär reaktor (SMR) — förlicensiering",
                       "DA": "Lille modulær reaktor (SMR) — forhåndslicensiering", "NO": "Liten modulær reaktor (SMR) — forhåndslisensering",
                       "PL": "Mały reaktor modułowy (SMR) — wstępne licencjonowanie"},
    "vesivoima":      {"EN": "Hydroelectric Power Project",                  "SE": "Vattenkraftsprojekt",
                       "DA": "Vandkraftsprojekt",                             "NO": "Vannkraftprosjekt",
                       "PL": "Projekt elektrowni wodnej"},
    "smr_bess":       {"EN": "SMR + BESS Hybrid Energy System",              "SE": "SMR + BESS hybridsystem",
                       "DA": "SMR + BESS hybridsystem",                       "NO": "SMR + BESS hybridsystem",
                       "PL": "System hybrydowy SMR + BESS"},
    "asuinrakennus":  {"EN": "Residential Construction Permit Application",   "SE": "Bygglovsansökan för bostadsbyggnad",
                       "DA": "Byggetilladelsesansøgning for beboelsesbygning", "NO": "Byggetillatelsessøknad for boligbygg",
                       "PL": "Wniosek o pozwolenie na budowę budynku mieszkalnego"},
    "teollisuus":     {"EN": "Industrial Construction Permit Application",    "SE": "Bygglovsansökan för industribyggnad",
                       "DA": "Byggetilladelsesansøgning for industribygning",  "NO": "Byggetillatelsessøknad för industribygg",
                       "PL": "Wniosek o pozwolenie na budowę budynku przemysłowego"},
    "maatalous":      {"EN": "Agricultural Construction Permit Application",  "SE": "Bygglovsansökan för lantbruksbyggnad",
                       "DA": "Byggetilladelsesansøgning for landbrugsbygning", "NO": "Byggetillatelsessøknad for landbruksbygg",
                       "PL": "Wniosek o pozwolenie na budowę budynku rolniczego"},
    "liikerakennus":  {"EN": "Commercial Construction Permit Application",    "SE": "Bygglovsansökan för affärsbyggnad",
                       "DA": "Byggetilladelsesansøgning for erhvervsbygning",  "NO": "Byggetillatelsessøknad for næringsbygg",
                       "PL": "Wniosek o pozwolenie na budowę budynku handlowego"},
    "muu":            {"EN": "Other Project Permit Application",             "SE": "Tillståndsansökan för annat projekt",
                       "DA": "Tilladelsesansøgning for andet projekt",         "NO": "Tillatelsessøknad for annet prosjekt",
                       "PL": "Wniosek o zezwolenie na inny projekt"},
    "ymparistolupa":  {"EN": "Environmental Permit Application (YSL 527/2014)", "SE": "Miljötillståndsansökan",
                       "DA": "Miljøtilladelsesansøgning",                      "NO": "Søknad om miljøtillatelse",
                       "PL": "Wniosek o pozwolenie środowiskowe"},
    "datakeskus":     {"EN": "Data Centre Permit Application",                  "SE": "Tillståndsansökan för datacenter",
                       "DA": "Tilladelsesansøgning for datacenter",              "NO": "Tillatelsessøknad for datasenter",
                       "PL": "Wniosek o zezwolenie na centrum danych"},
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
    "EE": {
        "name": "Estonia / Eesti",
        "authorities": [
            "Keskkonnaamet (Environmental Board — EIA, environmental permits)",
            "Kohaliku omavalitsuse (Municipality — building permit, detailed plan)",
            "Konkurentsiamet (Competition Authority — electricity production license)",
            "Elering AS (TSO — transmission grid connection)",
            "Elektrilevi OÜ (DSO — distribution grid connection)",
            "Kaitseministeerium (Ministry of Defence — wind radar clearance)",
            "Terviseamet (Health Board — radiation, nuclear)",
            "Transpordiamet (Transport Administration — aviation, maritime)",
        ],
        "key_laws": [
            "Elektrituruseadus (ETS, RT I 2012) — electricity market, production license",
            "Keskkonnaseadustiku üldosa seadus (KeÜS) — general environmental framework",
            "KMH-KSH seadus (RT I 2013) — EIA (Keskkonnamõju hindamine)",
            "Ehitusseadustik (EhS, RT I 2015) — building code, construction permit",
            "Planeerimisseadus (PlanS, RT I 2015) — spatial planning, detailed plan",
            "Energiamajanduse korralduse seadus (ESOS) — energy sector organisation",
        ],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in ESTONIA (Eesti). "
            "Apply Estonian regulatory framework:\n"
            "Key authorities: Keskkonnaamet (Environmental Board — EIA / KMH decisions, environmental permits), "
            "Kohaliku omavalitsuse volikogu (Municipality — building permit = ehitusluba, "
            "use permit = kasutusluba, detailed plan = detailplaneering), "
            "Konkurentsiamet (Competition Authority — electricity production license = "
            "elektrienergia tootmise luba), "
            "Elering AS (TSO, equivalent to Fingrid — transmission grid connection = liitumisleping), "
            "Elektrilevi OÜ (DSO — distribution grid connection for smaller projects), "
            "Kaitseministeerium (Ministry of Defence — mandatory military radar clearance for wind).\n"
            "Key laws: Elektrituruseadus ETS (electricity market + production license), "
            "KMH-KSH seadus (EIA = Keskkonnamõju hindamine / KMH), "
            "Ehitusseadustik EhS (building permit = ehitusluba), "
            "Planeerimisseadus PlanS (spatial plan = detailplaneering), "
            "Keskkonnaseadustiku üldosa seadus KeÜS (environmental code general part), "
            "Energiamajanduse korralduse seadus ESOS (energy sector organisation + NECP targets).\n"
            "Key market facts: National wind target ~1,200 MW by 2030 (baseline ~310 MW 2020); "
            "solar target ~415 MW by 2030 (baseline ~100 MW 2020); "
            "offshore wind combined permit (ühisluba) replaces former 3-step process; "
            "EU-approved offshore wind state aid €2.6 billion (2024); "
            "small rooftop solar < 15 kW household / < 50 kW commercial: simplified path "
            "(ehitusteatis + teatis to Konkurentsiamet, no full license). "
            "BESS: no dedicated BESS regulation — general ETS framework applies. "
            "SMR: no nuclear power law yet (draft Tuumaenergia seadus under development) — "
            "regWarning status, fundamental legislative gap.\n"
            "Replace Finnish law references (MRL, YSL, YVA-laki) with Estonian equivalents. "
            "Mark uncertain items: [Requires verification against Estonian regulations].\n\n"
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Maakohtaiset lupa-/viranomainen-/laki-rivit (ylikirjoittavat FI-oletustan)
# Avain (lupa) = suomenkielinen vakioavain → _t_lupa() kääntää sen
# Viranomainen / laki = natiivikielinen nimi (ei käännetä, pysyy ao. kielisenä)
# ─────────────────────────────────────────────────────────────────────────────
_COUNTRY_LUVAT: dict[str, dict[str, list[tuple[str, str, str]]]] = {
    "PL": {
        "SMR": [
            ("Periaatepäätös (VN)",           "PAA (Państwowa Agencja Atomistyki)",  "Prawo atomowe (Ustawa z 29.11.2000)"),
            ("YVA-menettely",                  "RDOŚ / GDOŚ",                         "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Rakentamislupa (ydinlaitos)",     "PAA (Państwowa Agencja Atomistyki)",  "Prawo atomowe (Ustawa z 29.11.2000)"),
            ("Käyttölupa (ydinlaitos)",         "PAA (Państwowa Agencja Atomistyki)",  "Prawo atomowe (Ustawa z 29.11.2000)"),
            ("Vesilupa (jäähdytysvesi)",        "Wody Polskie (PGWWP)",                "Prawo wodne (Ustawa z 20.07.2017)"),
            ("Pozwolenie na budowę",                    "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Maankäyttösopimus / kaavoitus",   "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "BESS": [
            ("Pozwolenie na budowę",                    "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Ympäristölupa",                   "Starosta / RDOŚ",                     "Prawo ochrony środowiska (Ustawa z 27.04.2001)"),
            ("Verkkoliityntäsopimus",           "URE (Urząd Regulacji Energetyki)",    "Ustawa Prawo energetyczne (1997)"),
            ("YVA-menettely (tarvitt.)",        "RDOŚ",                                "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Maankäyttösopimus / kaavoitus",   "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "tuulivoima_maa": [
            ("Pozwolenie na budowę",                    "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("YVA-menettely",                   "RDOŚ",                                "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Verkkoliityntäsopimus",           "URE (Urząd Regulacji Energetyki)",    "Ustawa Prawo energetyczne (1997)"),
            ("Etäisyysvaatimus (tuulivoima)",   "Gmina / Starostwo Powiatowe",         "Ustawa o inwestycjach w zakresie elektrowni wiatrowych (2016)"),
            ("Maankäyttösopimus / kaavoitus",   "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "aurinkovoima": [
            ("Pozwolenie na budowę",                    "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Verkkoliityntäsopimus",           "URE (Urząd Regulacji Energetyki)",    "Ustawa Prawo energetyczne (1997)"),
            ("YVA-menettely (tarvitt.)",        "RDOŚ",                                "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Maankäyttösopimus / kaavoitus",   "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "smr_bess": [
            ("Periaatepäätös (VN)",             "PAA (Państwowa Agencja Atomistyki)",  "Prawo atomowe (Ustawa z 29.11.2000)"),
            ("YVA-menettely",                   "RDOŚ / GDOŚ",                         "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Rakentamislupa (ydinlaitos)",      "PAA (Państwowa Agencja Atomistyki)",  "Prawo atomowe (Ustawa z 29.11.2000)"),
            ("Pozwolenie na budowę",                    "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Vesilupa (jäähdytysvesi)",        "Wody Polskie (PGWWP)",                "Prawo wodne (Ustawa z 20.07.2017)"),
            ("Verkkoliityntäsopimus",           "URE (Urząd Regulacji Energetyki)",    "Ustawa Prawo energetyczne (1997)"),
            ("Maankäyttösopimus / kaavoitus",   "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "vesivoima": [
            ("Vesilupa (padotus, rakentaminen)", "Wody Polskie (PGWWP)",               "Prawo wodne (Ustawa z 20.07.2017)"),
            ("Ympäristölupa",                   "RDOŚ",                                "Prawo ochrony środowiska (Ustawa z 27.04.2001)"),
            ("YVA-menettely (tarvitt.)",        "RDOŚ / GDOŚ",                         "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Pozwolenie na budowę",            "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Verkkoliityntäsopimus",           "URE (Urząd Regulacji Energetyki)",    "Ustawa Prawo energetyczne (1997)"),
            ("Maankäyttösopimus",               "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "datakeskus": [
            ("Pozwolenie na budowę",            "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Ympäristölupa (tarvitt.)",        "Starosta / RDOŚ",                     "Prawo ochrony środowiska (Ustawa z 27.04.2001)"),
            ("Verkkoliityntäsopimus",           "URE (Urząd Regulacji Energetyki)",    "Ustawa Prawo energetyczne (1997)"),
            ("Maankäyttösopimus / kaavoitus",   "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "teollisuus": [
            ("Pozwolenie na budowę",            "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Pozwolenie na emisję",            "Starosta / Marszałek Województwa",    "Prawo ochrony środowiska (Ustawa z 27.04.2001)"),
            ("YVA-menettely (tarvitt.)",        "RDOŚ",                                "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Maankäyttösopimus / kaavoitus",   "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "asuinrakennus": [
            ("Pozwolenie na budowę",            "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Warunki zabudowy",                "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
            ("Naapurikuuleminen",               "Gmina / hakija",                      "Prawo budowlane (Ustawa z 7.07.1994)"),
        ],
        "maatalous": [
            ("Pozwolenie na budowę",            "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Zgłoszenie robót budowlanych",    "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994) art. 29"),
            ("Maankäyttösopimus",               "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "liikerakennus": [
            ("Pozwolenie na budowę",            "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Warunki zabudowy",                "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
            ("Maankäyttösopimus",               "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "tuulivoima_meri": [
            ("YVA-menettely",                   "RDOŚ / GDOŚ",                         "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Pozwolenie na obszary morskie",   "Urząd Morski",                        "Ustawa o obszarach morskich RP (Ustawa z 21.03.1991)"),
            ("Decyzja o środowiskowych uwarunkowaniach", "RDOŚ",                       "Ustawa o udostępnianiu informacji o środowisku"),
            ("Pozwolenie na budowę",            "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Verkkoliityntäsopimus",           "URE (Urząd Regulacji Energetyki)",    "Ustawa Prawo energetyczne (1997)"),
        ],
        "offshore_wind": [
            ("YVA-menettely",                   "RDOŚ / GDOŚ",                         "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Pozwolenie na obszary morskie",   "Urząd Morski",                        "Ustawa o obszarach morskich RP (Ustawa z 21.03.1991)"),
            ("Decyzja o środowiskowych uwarunkowaniach", "RDOŚ",                       "Ustawa o udostępnianiu informacji o środowisku"),
            ("Pozwolenie na budowę",            "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Verkkoliityntäsopimus",           "URE (Urząd Regulacji Energetyki)",    "Ustawa Prawo energetyczne (1997)"),
        ],
        "egs": [
            ("Koncesja na poszukiwanie kopalin (geotermia)", "Minister Klimatu i Środowiska", "Prawo geologiczne i górnicze (Ustawa z 9.06.2011)"),
            ("Pozwolenie na budowę",            "Starostwo Powiatowe",                 "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("YVA-menettely (tarvitt.)",        "RDOŚ",                                "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Maankäyttösopimus",               "Gmina (urząd gminy / miasta)",        "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
        ],
        "hybridi": [
            ("Pozwolenie na budowę (instalacja hybrydowa)", "Starostwo Powiatowe",     "Prawo budowlane (Ustawa z 7.07.1994)"),
            ("Decyzja o środowiskowych uwarunkowaniach",  "RDOŚ",                      "Ustawa o udostępnianiu informacji o środowisku (Dz.U. 2023 poz. 1029)"),
            ("Koncesja na wytwarzanie energii",           "URE (Urząd Regulacji Energetyki)", "Ustawa Prawo energetyczne (1997)"),
            ("Decyzja o warunkach zabudowy (tarvitt.)",  "Wójt / Burmistrz / Prezydent miasta", "Ustawa o planowaniu i zagospodarowaniu przestrzennym (2003)"),
            ("Umowa przyłączeniowa (sieć)",               "URE / operator systemu dystrybucyjnego", "Ustawa Prawo energetyczne (1997)"),
        ],
    },
    # ── Sverige ──────────────────────────────────────────────────────────────
    "SE": {
        "SMR": [
            ("Periaatepäätös (VN)",              "Nærings- och beredskapsdept. / Regeringen", "Kärntekniklag (SFS 1984:3)"),
            ("YVA-menettely",                    "Länsstyrelsen / Mark- och miljödomstolen",  "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Rakentamislupa (ydinlaitos)",       "Strålsäkerhetsmyndigheten (SSM)",           "Kärntekniklag (SFS 1984:3)"),
            ("Käyttölupa (ydinlaitos)",           "Strålsäkerhetsmyndigheten (SSM)",           "Kärntekniklag (SFS 1984:3)"),
            ("Vesilupa (jäähdytysvesi)",          "Mark- och miljödomstolen",                  "Miljöbalken (SFS 1998:808) kap. 11"),
            ("Bygglov",                      "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "BESS": [
            ("Bygglov",                      "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Ympäristölupa",                     "Länsstyrelsen",                             "Miljöbalken (SFS 1998:808)"),
            ("Verkkoliityntäsopimus",             "Svenska kraftnät / lokalt elnätsbolag",     "Ellagen (SFS 1997:857)"),
            ("YVA-menettely (tarvitt.)",          "Länsstyrelsen",                             "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "tuulivoima_maa": [
            ("YVA-menettely (≥10 MW / ≥5 voimalaa)", "Länsstyrelsen",                         "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Tillstånd / koncession",               "Energimyndigheten",                      "Ellagen (SFS 1997:857) / Miljöbalken"),
            ("Osayleiskaava tai asemakaava",          "Kommunen",                              "Plan- och bygglagen (SFS 2010:900)"),
            ("Bygglov",                          "Kommunen (byggnadsnämnd)",              "Plan- och bygglagen (SFS 2010:900)"),
            ("Ympäristölupa (tarvitt.)",              "Mark- och miljödomstolen",              "Miljöbalken (SFS 1998:808)"),
            ("Verkkoliityntäsopimus",                 "Svenska kraftnät",                      "Ellagen (SFS 1997:857)"),
        ],
        "aurinkovoima": [
            ("Bygglov",                      "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Verkkoliityntäsopimus",             "Svenska kraftnät / lokalt elnätsbolag",     "Ellagen (SFS 1997:857)"),
            ("YVA-menettely (tarvitt.)",          "Länsstyrelsen",                             "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "smr_bess": [
            ("Periaatepäätös (VN)",              "Nærings- och beredskapsdept. / Regeringen", "Kärntekniklag (SFS 1984:3)"),
            ("YVA-menettely",                    "Länsstyrelsen / Mark- och miljödomstolen",  "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Rakentamislupa (ydinlaitos)",       "Strålsäkerhetsmyndigheten (SSM)",           "Kärntekniklag (SFS 1984:3)"),
            ("Bygglov",                      "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Vesilupa (jäähdytysvesi)",          "Mark- och miljödomstolen",                  "Miljöbalken (SFS 1998:808) kap. 11"),
            ("Verkkoliityntäsopimus",             "Svenska kraftnät",                          "Ellagen (SFS 1997:857)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "vesivoima": [
            ("Vesilupa (padotus, rakentaminen)",  "Mark- och miljödomstolen",                  "Miljöbalken (SFS 1998:808) kap. 11"),
            ("Ympäristölupa",                     "Länsstyrelsen / Mark- och miljödomstolen",  "Miljöbalken (SFS 1998:808)"),
            ("YVA-menettely (tarvitt.)",          "Länsstyrelsen",                             "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Bygglov",                      "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Verkkoliityntäsopimus",             "Svenska kraftnät",                          "Ellagen (SFS 1997:857)"),
        ],
        "tuulivoima_meri": [
            ("YVA-menettely",                    "Länsstyrelsen / Mark- och miljödomstolen",  "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Tillstånd / koncession (offshore)", "Energimyndigheten",                        "Ellagen (SFS 1997:857) / Kontinentalsockellagen"),
            ("Ympäristölupa",                     "Mark- och miljödomstolen",                  "Miljöbalken (SFS 1998:808)"),
            ("Vesilupa (merialue)",               "Mark- och miljödomstolen",                  "Miljöbalken (SFS 1998:808) kap. 11"),
            ("Verkkoliityntäsopimus",             "Svenska kraftnät",                          "Ellagen (SFS 1997:857)"),
        ],
        "datakeskus": [
            ("Bygglov",                           "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Miljöprövning (tarvitt.)",          "Länsstyrelsen",                             "Miljöbalken (SFS 1998:808) kap. 9"),
            ("Verkkoliityntäsopimus",             "Svenska kraftnät / lokalt elnätsbolag",     "Ellagen (SFS 1997:857)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "teollisuus": [
            ("Bygglov",                           "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Miljötillstånd (tarvitt.)",         "Länsstyrelsen / Mark- och miljödomstolen",  "Miljöbalken (SFS 1998:808) kap. 9"),
            ("YVA-menettely (tarvitt.)",          "Länsstyrelsen",                             "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "asuinrakennus": [
            ("Bygglov",                           "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Detaljplan / planbesked",           "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Naapurikuuleminen / grannemedgivande", "Kommunen / hakija",                     "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "maatalous": [
            ("Bygglov (tarvitt.)",                "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Miljöprövning (karjatalous)",       "Länsstyrelsen",                             "Miljöbalken (SFS 1998:808) kap. 9"),
            ("Maankäyttösopimus",                 "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "liikerakennus": [
            ("Bygglov",                           "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Detaljplan / planbesked",           "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Maankäyttösopimus",                 "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "smr_se": [
            ("Periaatepäätös (VN)",              "Nærings- och beredskapsdept. / Regeringen", "Kärntekniklag (SFS 1984:3)"),
            ("YVA-menettely",                    "Länsstyrelsen / Mark- och miljödomstolen",  "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Rakentamislupa (ydinlaitos)",       "Strålsäkerhetsmyndigheten (SSM)",           "Kärntekniklag (SFS 1984:3)"),
            ("Käyttölupa (ydinlaitos)",           "Strålsäkerhetsmyndigheten (SSM)",           "Kärntekniklag (SFS 1984:3)"),
            ("Vesilupa (jäähdytysvesi)",          "Mark- och miljödomstolen",                  "Miljöbalken (SFS 1998:808) kap. 11"),
            ("Bygglov",                           "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Plan- och bygglagen (SFS 2010:900)"),
        ],
        "offshore_wind": [
            ("YVA-menettely",                    "Länsstyrelsen / Mark- och miljödomstolen",  "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Tillstånd (flytande offshore)",     "Energimyndigheten",                         "Ellagen (SFS 1997:857) / Kontinentalsockellagen"),
            ("Ympäristölupa",                     "Mark- och miljödomstolen",                  "Miljöbalken (SFS 1998:808)"),
            ("Vesilupa (merialue)",               "Mark- och miljödomstolen",                  "Miljöbalken (SFS 1998:808) kap. 11"),
            ("Verkkoliityntäsopimus",             "Svenska kraftnät",                          "Ellagen (SFS 1997:857)"),
        ],
        "egs": [
            ("Markkoncessionsansökan (djupgeotermisk)", "Bergsstaten",                         "Minerallagen (SFS 1991:45)"),
            ("Bygglov",                           "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Miljötillstånd (tarvitt.)",         "Länsstyrelsen",                             "Miljöbalken (SFS 1998:808) kap. 9"),
            ("Vattendomstol (borrning)",          "Mark- och miljödomstolen",                  "Miljöbalken (SFS 1998:808) kap. 11"),
        ],
        "hybridi": [
            ("Bygglov (hybridanläggning)",        "Kommunen (byggnadsnämnd)",                  "Plan- och bygglagen (SFS 2010:900)"),
            ("Miljötillstånd (IED/IPPC)",         "Länsstyrelsen / Mark- och miljödomstolen",  "Miljöbalken (SFS 1998:808) kap. 9"),
            ("Nätanslutningsavtal",               "Affärsverket svenska kraftnät / VNB",       "Ellagen (SFS 1997:857)"),
            ("MKB (tarvitt.)",                   "Länsstyrelsen",                             "Miljöbalken (SFS 1998:808) kap. 6"),
            ("Detaljplan / ändring av detaljplan", "Kommunen",                                "Plan- och bygglagen (SFS 2010:900) kap. 5"),
        ],
    },
    # ── Danmark ──────────────────────────────────────────────────────────────
    "DA": {
        "SMR": [
            ("Periaatepäätös (VN)",              "Energistyrelsen / Klima-, Energi- og Forsyningsministeriet", "Lov om anvendelse af Danmarks undergrund (nr. 181/1990)"),
            ("YVA-menettely",                    "Miljøministeriet / Miljøstyrelsen",          "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Rakentamislupa (ydinlaitos)",       "Sundhedsstyrelsen / Statens Institut for Strålebeskyttelse (SIS)", "Lov om brug af radioaktive stoffer (nr. 94/2003)"),
            ("Käyttölupa (ydinlaitos)",           "Sundhedsstyrelsen / SIS",                   "Lov om brug af radioaktive stoffer (nr. 94/2003)"),
            ("Vesilupa (jäähdytysvesi)",          "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 705/2022)"),
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen (planafdelingen)",                 "Planloven (LBK nr. 1157/2021)"),
        ],
        "BESS": [
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Ympäristölupa",                     "Kommunen / Miljøstyrelsen",                 "Miljøbeskyttelsesloven (LBK nr. 1218/2019)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 119/2020)"),
            ("YVA-menettely (tarvitt.)",          "Miljøstyrelsen",                            "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
        ],
        "tuulivoima_maa": [
            ("YVA-menettely (≥10 MW / ≥5 voimalaa)", "Miljøstyrelsen",                        "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Vindmølletilladelse",                   "Energistyrelsen",                       "Lov om vedvarende energi (LBK nr. 388/2022)"),
            ("Osayleiskaava tai asemakaava",           "Kommunen",                             "Planloven (LBK nr. 1157/2021)"),
            ("Byggetilladelse",                          "Kommunen (teknik og miljø)",            "Byggeloven (LBK nr. 1178/2023)"),
            ("Ympäristölupa (tarvitt.)",              "Miljøstyrelsen",                        "Miljøbeskyttelsesloven (LBK nr. 1218/2019)"),
            ("Verkkoliityntäsopimus",                 "Energinet",                             "Elforsyningsloven (LBK nr. 119/2020)"),
        ],
        "aurinkovoima": [
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 119/2020)"),
            ("YVA-menettely (tarvitt.)",          "Miljøstyrelsen",                            "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
        ],
        "smr_bess": [
            ("Periaatepäätös (VN)",              "Energistyrelsen / Klima-, Energi- og Forsyningsministeriet", "Lov om anvendelse af Danmarks undergrund (nr. 181/1990)"),
            ("YVA-menettely",                    "Miljøministeriet / Miljøstyrelsen",          "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Rakentamislupa (ydinlaitos)",       "Sundhedsstyrelsen / SIS",                   "Lov om brug af radioaktive stoffer (nr. 94/2003)"),
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Vesilupa (jäähdytysvesi)",          "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 705/2022)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 119/2020)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
        ],
        "vesivoima": [
            ("Vesilupa (padotus, rakentaminen)",  "Kystdirektoratet / Miljøstyrelsen",         "Vandforsyningsloven (LBK nr. 118/2020)"),
            ("Ympäristölupa",                     "Miljøstyrelsen",                            "Miljøbeskyttelsesloven (LBK nr. 1218/2019)"),
            ("YVA-menettely (tarvitt.)",          "Miljøstyrelsen",                            "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 119/2020)"),
        ],
        "tuulivoima_meri": [
            ("YVA-menettely",                    "Miljøstyrelsen",                             "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Havvindtilladelse",                 "Energistyrelsen",                           "Lov om fremme af vedvarende energi (LBK nr. 388/2022)"),
            ("Ympäristölupa",                     "Miljøstyrelsen",                            "Miljøbeskyttelsesloven (LBK nr. 1218/2019)"),
            ("Vesilupa (merialue)",               "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 705/2022)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 119/2020)"),
        ],
        "datakeskus": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Miljøgodkendelse (tarvitt.)",       "Kommunen / Miljøstyrelsen",                 "Miljøbeskyttelsesloven (LBK nr. 1218/2019)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 119/2020)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
        ],
        "teollisuus": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Miljøgodkendelse",                  "Kommunen / Miljøstyrelsen",                 "Miljøbeskyttelsesloven (LBK nr. 1218/2019)"),
            ("YVA-menettely (tarvitt.)",          "Miljøstyrelsen",                            "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
        ],
        "asuinrakennus": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Lokalplan",                         "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
            ("Naapurikuuleminen / nabohøring",    "Kommunen / hakija",                        "Byggeloven (LBK nr. 1178/2023)"),
        ],
        "maatalous": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Miljøgodkendelse (husdyr)",         "Kommunen",                                  "Lov om miljøgodkendelse m.v. af husdyrbrug (LBK nr. 442/2022)"),
            ("Lokalplan",                         "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
        ],
        "liikerakennus": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Lokalplan",                         "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
            ("Maankäyttösopimus",                 "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
        ],
        "smr_da": [
            ("Periaatepäätös (VN)",              "Energistyrelsen / Klima-, Energi- og Forsyningsministeriet", "Lov om anvendelse af Danmarks undergrund (nr. 181/1990)"),
            ("YVA-menettely",                    "Miljøministeriet / Miljøstyrelsen",          "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Rakentamislupa (ydinlaitos)",       "Sundhedsstyrelsen / Statens Institut for Strålebeskyttelse (SIS)", "Lov om brug af radioaktive stoffer (nr. 94/2003)"),
            ("Käyttölupa (ydinlaitos)",           "Sundhedsstyrelsen / SIS",                   "Lov om brug af radioaktive stoffer (nr. 94/2003)"),
            ("Vesilupa (jäähdytysvesi)",          "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 705/2022)"),
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen (planafdelingen)",                 "Planloven (LBK nr. 1157/2021)"),
        ],
        "offshore_wind": [
            ("YVA-menettely",                    "Miljøstyrelsen",                             "Miljøvurderingsloven (LBK nr. 1976/2021)"),
            ("Havvindtilladelse (flydende)",      "Energistyrelsen",                           "Lov om fremme af vedvarende energi (LBK nr. 388/2022)"),
            ("Ympäristölupa",                     "Miljøstyrelsen",                            "Miljøbeskyttelsesloven (LBK nr. 1218/2019)"),
            ("Vesilupa (merialue)",               "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 705/2022)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 119/2020)"),
        ],
        "egs": [
            ("Forundersøgelsestilladelse (geotermisk)", "Energistyrelsen",                     "Undergrundsloven (LBK nr. 1505/2019)"),
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2023)"),
            ("Miljøgodkendelse (tarvitt.)",       "Miljøstyrelsen",                            "Miljøbeskyttelsesloven (LBK nr. 1218/2019)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 1157/2021)"),
        ],
    },
    # ── Norge ─────────────────────────────────────────────────────────────────
    "NO": {
        "SMR": [
            ("Periaatepäätös (VN)",              "Nærings- og fiskeridepartementet (NFD)",     "Atomenergiloven (LOV-1972-05-12-28)"),
            ("YVA-menettely",                    "Statsforvalteren / Miljødirektoratet",        "Plan- og bygningsloven (LOV-2008-06-27-71) kap. 14"),
            ("Rakentamislupa (ydinlaitos)",       "Direktoratet for strålevern og atomsikkerhet (DSA)", "Strålevernloven (LOV-2000-05-12-36)"),
            ("Käyttölupa (ydinlaitos)",           "Direktoratet for strålevern og atomsikkerhet (DSA)", "Strålevernloven (LOV-2000-05-12-36)"),
            ("Vesilupa (jäähdytysvesi)",          "NVE (Norges vassdrags- og energidirektorat)","Vassdragsreguleringsloven (LOV-1917-12-14-17)"),
            ("Byggetillatelse",                      "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "BESS": [
            ("Byggetillatelse",                      "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Ympäristölupa",                     "Statsforvalteren",                           "Forurensningsloven (LOV-1981-03-13-6)"),
            ("Verkkoliityntäsopimus",             "Statnett / lokalt nettselskap",              "Energiloven (LOV-1990-06-29-50)"),
            ("YVA-menettely (tarvitt.)",          "Statsforvalteren / Miljødirektoratet",        "Plan- og bygningsloven kap. 14"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "tuulivoima_maa": [
            ("YVA-menettely (≥10 MW / ≥5 voimalaa)", "NVE / Miljødirektoratet",               "Plan- og bygningsloven kap. 14"),
            ("Konsesjon (anleggskonsesjon)",          "NVE",                                   "Energiloven (LOV-1990-06-29-50) § 3-1"),
            ("Osayleiskaava tai asemakaava",          "Kommunen",                              "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Byggetillatelse",                          "Kommunen (plan og bygning)",            "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Ympäristölupa (tarvitt.)",              "Statsforvalteren",                      "Forurensningsloven (LOV-1981-03-13-6)"),
            ("Verkkoliityntäsopimus",                 "Statnett",                              "Energiloven (LOV-1990-06-29-50)"),
        ],
        "aurinkovoima": [
            ("Byggetillatelse",                      "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Verkkoliityntäsopimus",             "Statnett / lokalt nettselskap",              "Energiloven (LOV-1990-06-29-50)"),
            ("YVA-menettely (tarvitt.)",          "NVE / Miljødirektoratet",                   "Plan- og bygningsloven kap. 14"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "smr_bess": [
            ("Periaatepäätös (VN)",              "Nærings- og fiskeridepartementet (NFD)",     "Atomenergiloven (LOV-1972-05-12-28)"),
            ("YVA-menettely",                    "Statsforvalteren / Miljødirektoratet",        "Plan- og bygningsloven kap. 14"),
            ("Rakentamislupa (ydinlaitos)",       "Direktoratet for strålevern og atomsikkerhet (DSA)", "Strålevernloven (LOV-2000-05-12-36)"),
            ("Byggetillatelse",                      "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Vesilupa (jäähdytysvesi)",          "NVE",                                        "Vassdragsreguleringsloven (LOV-1917-12-14-17)"),
            ("Verkkoliityntäsopimus",             "Statnett",                                   "Energiloven (LOV-1990-06-29-50)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "vesivoima": [
            ("Vesilupa (padotus, rakentaminen)",  "NVE",                                        "Vassdragsreguleringsloven (LOV-1917-12-14-17)"),
            ("Ympäristölupa",                     "Statsforvalteren",                           "Forurensningsloven (LOV-1981-03-13-6)"),
            ("YVA-menettely (tarvitt.)",          "NVE / Miljødirektoratet",                   "Plan- og bygningsloven kap. 14"),
            ("Byggetillatelse",                      "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Verkkoliityntäsopimus",             "Statnett",                                   "Energiloven (LOV-1990-06-29-50)"),
        ],
        "tuulivoima_meri": [
            ("YVA-menettely",                    "NVE / Miljødirektoratet",                    "Plan- og bygningsloven kap. 14"),
            ("Konsesjon (offshore-konsesjon)",    "NVE / Olje- og energidepartementet",         "Havenergilova (LOV-2010-06-04-21)"),
            ("Ympäristölupa",                     "Statsforvalteren",                           "Forurensningsloven (LOV-1981-03-13-6)"),
            ("Vesilupa (merialue)",               "Kystverket",                                 "Havne- og farvannsloven (LOV-2019-06-21-70)"),
            ("Verkkoliityntäsopimus",             "Statnett",                                   "Energiloven (LOV-1990-06-29-50)"),
        ],
        "datakeskus": [
            ("Byggetillatelse",                   "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Utslippstillatelse (tarvitt.)",     "Statsforvalteren",                           "Forurensningsloven (LOV-1981-03-13-6)"),
            ("Verkkoliityntäsopimus",             "Statnett / lokalt nettselskap",              "Energiloven (LOV-1990-06-29-50)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "teollisuus": [
            ("Byggetillatelse",                   "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Utslippstillatelse",                "Statsforvalteren",                           "Forurensningsloven (LOV-1981-03-13-6)"),
            ("YVA-menettely (tarvitt.)",          "Statsforvalteren / Miljødirektoratet",        "Plan- og bygningsloven kap. 14"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "asuinrakennus": [
            ("Byggetillatelse",                   "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Reguleringsplan",                   "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Naapurikuuleminen / nabovarsel",    "Kommunen / hakija",                         "Plan- og bygningsloven (LOV-2008-06-27-71) § 21-3"),
        ],
        "maatalous": [
            ("Byggetillatelse",                   "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Utslippstillatelse (husdyr)",       "Statsforvalteren",                           "Forurensningsloven (LOV-1981-03-13-6)"),
            ("Maankäyttösopimus",                 "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "liikerakennus": [
            ("Byggetillatelse",                   "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Reguleringsplan",                   "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Maankäyttösopimus",                 "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "smr_no": [
            ("Periaatepäätös (VN)",              "Nærings- og fiskeridepartementet (NFD)",     "Atomenergiloven (LOV-1972-05-12-28)"),
            ("YVA-menettely",                    "Statsforvalteren / Miljødirektoratet",        "Plan- og bygningsloven (LOV-2008-06-27-71) kap. 14"),
            ("Rakentamislupa (ydinlaitos)",       "Direktoratet for strålevern og atomsikkerhet (DSA)", "Strålevernloven (LOV-2000-05-12-36)"),
            ("Käyttölupa (ydinlaitos)",           "Direktoratet for strålevern og atomsikkerhet (DSA)", "Strålevernloven (LOV-2000-05-12-36)"),
            ("Vesilupa (jäähdytysvesi)",          "NVE (Norges vassdrags- og energidirektorat)","Vassdragsreguleringsloven (LOV-1917-12-14-17)"),
            ("Byggetillatelse",                   "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "offshore_wind": [
            ("YVA-menettely",                    "NVE / Miljødirektoratet",                    "Plan- og bygningsloven kap. 14"),
            ("Konsesjon (flytende offshore)",     "NVE / Olje- og energidepartementet",         "Havenergilova (LOV-2010-06-04-21)"),
            ("Ympäristölupa",                     "Statsforvalteren",                           "Forurensningsloven (LOV-1981-03-13-6)"),
            ("Vesilupa (merialue)",               "Kystverket",                                 "Havne- og farvannsloven (LOV-2019-06-21-70)"),
            ("Verkkoliityntäsopimus",             "Statnett",                                   "Energiloven (LOV-1990-06-29-50)"),
        ],
        "egs": [
            ("Undersøkelsestillatelse (geotermisk)", "Direktoratet for mineralforvaltning (DMF)", "Mineralloven (LOV-2009-06-19-101)"),
            ("Byggetillatelse",                   "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Utslippstillatelse (tarvitt.)",     "Statsforvalteren",                           "Forurensningsloven (LOV-1981-03-13-6)"),
            ("Maankäyttösopimus",                 "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
        "hybridi": [
            ("Byggetillatelse (hybridanlegg)",    "Kommunen (plan og bygning)",                 "Plan- og bygningsloven (LOV-2008-06-27-71)"),
            ("Konsesjon (energianlegg)",          "NVE / Olje- og energidepartementet",         "Energiloven (LOV-1990-06-29-50)"),
            ("Nettilkoblingsavtale",              "Statnett / lokal netteier",                  "Energiloven (LOV-1990-06-29-50)"),
            ("Utslippstillatelse (tarvitt.)",     "Statsforvalteren",                           "Forurensningsloven (LOV-1981-03-13-6)"),
            ("Reguleringsplan",                   "Kommunen",                                   "Plan- og bygningsloven (LOV-2008-06-27-71)"),
        ],
    },
    # ── Deutschland ──────────────────────────────────────────────────────────────
    "DE": {
        "SMR": [
            ("Standortgenehmigung / AtG-Genehmigung", "BMUV / Länderaufsichtsbehörde",         "Atomgesetz (AtG, BGBl. I S. 1553/1959)"),
            ("UVP-Prüfung",                       "Genehmigungsbehörde (Länder)",               "UVPG (Gesetz über die Umweltverträglichkeitsprüfung)"),
            ("Baugenehmigung (ydinlaitos)",        "Landesbaubehörde",                          "Landesbauordnung (LBauO, je nach Bundesland)"),
            ("Wasserrechtliche Erlaubnis",         "Untere Wasserbehörde (Land)",               "Wasserhaushaltsgesetz (WHG, § 8)"),
            ("Bauleitplanung",                     "Gemeinde",                                  "Baugesetzbuch (BauGB, § 1)"),
        ],
        "BESS": [
            ("Baugenehmigung",                     "Untere Baubehörde (Landkreis/Stadt)",       "Landesbauordnung (LBauO)"),
            ("BImSchG-Genehmigung (tarvitt.)",     "Immissionsschutzbehörde (Land)",            "Bundes-Immissionsschutzgesetz (BImSchG, § 4)"),
            ("Verkkoliityntäsopimus",              "Übertragungsnetzbetreiber (ÜNB) / VNB",    "Energiewirtschaftsgesetz (EnWG, § 17)"),
            ("UVP (tarvitt.)",                     "Genehmigungsbehörde",                       "UVPG"),
            ("Bauleitplanung",                     "Gemeinde",                                  "Baugesetzbuch (BauGB)"),
        ],
        "tuulivoima_maa": [
            ("BImSchG-Genehmigung",               "Immissionsschutzbehörde (Land)",             "Bundes-Immissionsschutzgesetz (BImSchG, § 4)"),
            ("UVP-Prüfung",                        "Genehmigungsbehörde",                       "UVPG"),
            ("Bauleitplanung",                     "Gemeinde",                                  "Baugesetzbuch (BauGB)"),
            ("Verkkoliityntäsopimus",              "Übertragungsnetzbetreiber (ÜNB)",           "Energiewirtschaftsgesetz (EnWG)"),
            ("Artenschutzrechtliche Prüfung",      "Untere Naturschutzbehörde",                 "Bundesnaturschutzgesetz (BNatSchG, § 44)"),
        ],
        "aurinkovoima": [
            ("Baugenehmigung (tarvitt.)",          "Untere Baubehörde",                         "Landesbauordnung (LBauO)"),
            ("Verkkoliityntäsopimus",              "Verteilnetzbetreiber (VNB)",                "Energiewirtschaftsgesetz (EnWG, § 17)"),
            ("Bauleitplanung (tarvitt.)",          "Gemeinde",                                  "Baugesetzbuch (BauGB)"),
        ],
        "vesivoima": [
            ("Wasserrechtliche Erlaubnis / Bewilligung", "Untere Wasserbehörde (Land)",         "Wasserhaushaltsgesetz (WHG, § 8 ff.)"),
            ("BImSchG-Genehmigung (tarvitt.)",     "Immissionsschutzbehörde",                   "Bundes-Immissionsschutzgesetz (BImSchG)"),
            ("UVP-Prüfung",                        "Genehmigungsbehörde",                       "UVPG"),
            ("Baugenehmigung",                     "Untere Baubehörde",                         "Landesbauordnung (LBauO)"),
            ("Verkkoliityntäsopimus",              "Übertragungsnetzbetreiber (ÜNB)",           "Energiewirtschaftsgesetz (EnWG)"),
        ],
        "datakeskus": [
            ("Baugenehmigung",                     "Untere Baubehörde (Landkreis/Stadt)",       "Landesbauordnung (LBauO)"),
            ("BImSchG-Genehmigung (tarvitt.)",     "Immissionsschutzbehörde",                   "Bundes-Immissionsschutzgesetz (BImSchG)"),
            ("Verkkoliityntäsopimus",              "Verteilnetzbetreiber (VNB) / ÜNB",         "Energiewirtschaftsgesetz (EnWG, § 17)"),
            ("Bauleitplanung",                     "Gemeinde",                                  "Baugesetzbuch (BauGB)"),
        ],
        "teollisuus": [
            ("Baugenehmigung",                     "Untere Baubehörde",                         "Landesbauordnung (LBauO)"),
            ("BImSchG-Genehmigung",               "Immissionsschutzbehörde (Land)",             "Bundes-Immissionsschutzgesetz (BImSchG, § 4)"),
            ("UVP (tarvitt.)",                     "Genehmigungsbehörde",                       "UVPG"),
            ("Bauleitplanung",                     "Gemeinde",                                  "Baugesetzbuch (BauGB)"),
        ],
        "asuinrakennus": [
            ("Baugenehmigung",                     "Untere Baubehörde (Landkreis/Stadt)",       "Landesbauordnung (LBauO)"),
            ("Bebauungsplan",                      "Gemeinde",                                  "Baugesetzbuch (BauGB, § 30)"),
            ("Nachbaranhörung",                    "Gemeinde / Bauherr",                        "Landesbauordnung (LBauO)"),
        ],
        "maatalous": [
            ("Baugenehmigung / Bauprivileg",       "Untere Baubehörde",                         "Landesbauordnung (LBauO) / BauGB § 35"),
            ("BImSchG-Genehmigung (Tierhaltung)", "Immissionsschutzbehörde",                    "Bundes-Immissionsschutzgesetz (BImSchG, § 4)"),
            ("UVP (tarvitt.)",                     "Genehmigungsbehörde",                       "UVPG"),
        ],
        "liikerakennus": [
            ("Baugenehmigung",                     "Untere Baubehörde",                         "Landesbauordnung (LBauO)"),
            ("Bebauungsplan",                      "Gemeinde",                                  "Baugesetzbuch (BauGB)"),
            ("Bauleitplanung",                     "Gemeinde",                                  "Baugesetzbuch (BauGB, § 1)"),
        ],
        "smr_bess": [
            ("Standortgenehmigung / AtG-Genehmigung", "BMUV / Länderaufsichtsbehörde",         "Atomgesetz (AtG)"),
            ("BImSchG-Genehmigung (BESS-osuus)",   "Immissionsschutzbehörde",                   "Bundes-Immissionsschutzgesetz (BImSchG, § 4)"),
            ("Baugenehmigung",                     "Untere Baubehörde",                         "Landesbauordnung (LBauO)"),
            ("Wasserrechtliche Erlaubnis",         "Untere Wasserbehörde",                      "Wasserhaushaltsgesetz (WHG, § 8)"),
            ("Verkkoliityntäsopimus",              "Übertragungsnetzbetreiber (ÜNB)",           "Energiewirtschaftsgesetz (EnWG)"),
            ("Bauleitplanung",                     "Gemeinde",                                  "Baugesetzbuch (BauGB)"),
        ],
        "tuulivoima_meri": [
            ("BImSchG-Genehmigung (offshore)",     "BSH (Bundesamt für Seeschifffahrt)",        "Windenergie-auf-See-Gesetz (WindSeeG)"),
            ("UVP-Prüfung",                        "BSH",                                       "UVPG"),
            ("Verkkoliityntäsopimus",              "Übertragungsnetzbetreiber (ÜNB)",           "Energiewirtschaftsgesetz (EnWG)"),
        ],
        "smr_de": [
            ("Standortgenehmigung / AtG-Genehmigung", "BMUV / Länderaufsichtsbehörde",         "Atomgesetz (AtG, BGBl. I S. 1553/1959)"),
            ("UVP-Prüfung",                        "Genehmigungsbehörde (Länder)",               "UVPG (Gesetz über die Umweltverträglichkeitsprüfung)"),
            ("Baugenehmigung (ydinlaitos)",         "Landesbaubehörde",                          "Landesbauordnung (LBauO, je nach Bundesland)"),
            ("Wasserrechtliche Erlaubnis",          "Untere Wasserbehörde (Land)",               "Wasserhaushaltsgesetz (WHG, § 8)"),
            ("Bauleitplanung",                      "Gemeinde",                                  "Baugesetzbuch (BauGB, § 1)"),
        ],
        "offshore_wind": [
            ("BImSchG-Genehmigung (flytende offshore)", "BSH (Bundesamt für Seeschifffahrt)",   "Windenergie-auf-See-Gesetz (WindSeeG)"),
            ("UVP-Prüfung",                        "BSH",                                       "UVPG"),
            ("Verkkoliityntäsopimus",              "Übertragungsnetzbetreiber (ÜNB)",           "Energiewirtschaftsgesetz (EnWG)"),
        ],
        "egs": [
            ("Bergrechtliche Betriebsplanzulassung", "Bergamt (Landesbergbehörde)",             "Bundesberggesetz (BBergG, § 55)"),
            ("Wasserrechtliche Erlaubnis (Tiefengeothermie)", "Untere Wasserbehörde",           "Wasserhaushaltsgesetz (WHG, § 8)"),
            ("Baugenehmigung",                     "Untere Baubehörde",                         "Landesbauordnung (LBauO)"),
            ("UVP (tarvitt.)",                     "Genehmigungsbehörde",                       "UVPG"),
        ],
        "hybridi": [
            ("BImSchG-Genehmigung (Hauptanlage)",  "Immissionsschutzbehörde (Land)",            "Bundes-Immissionsschutzgesetz (BImSchG, § 4)"),
            ("Baugenehmigung",                     "Untere Baubehörde (Landkreis/Stadt)",       "Landesbauordnung (LBauO)"),
            ("Verkkoliityntäsopimus",              "Übertragungsnetzbetreiber (ÜNB) / VNB",    "Energiewirtschaftsgesetz (EnWG, § 17)"),
            ("UVP-Prüfung (tarvitt.)",             "Genehmigungsbehörde",                       "UVPG"),
            ("Bauleitplanung",                     "Gemeinde",                                  "Baugesetzbuch (BauGB, § 1)"),
        ],
    },
    # ── Eesti (Estonia) ──────────────────────────────────────────────────────────
    "EE": {
        # ── Tuulivoima (onshore wind) ─────────────────────────────────────────
        "tuulivoima_maa": [
            ("Detailplaneering",                   "Kohaliku omavalitsuse volikogu",              "Planeerimisseadus (PlanS, RT I 2015)"),
            ("YVA-menettely (KMH)",                "Keskkonnaamet (Environmental Board)",         "KMH-KSH seadus (RT I, 04.01.2013, 10)"),
            ("Kaitseministeerium radar clearance", "Kaitseministeerium (Ministry of Defence)",    "Kiirgusseadus / national security assessment"),
            ("Rakentamislupa (ehitusluba)",         "Kohaliku omavalitsuse (Municipality)",       "Ehitusseadustik (EhS, RT I 2015)"),
            ("Elektrienergia tootmise luba",        "Konkurentsiamet (Competition Authority)",    "Elektrituruseadus (ETS, RT I 2012) §14"),
            ("Verkkoliityntäsopimus (liitumisleping)", "Elering AS (TSO)",                        "Elektrituruseadus (ETS) §§73-83; RfG 2016/631"),
            ("Kasutusluba (use permit)",            "Kohaliku omavalitsuse (Municipality)",       "Ehitusseadustik (EhS) §§61-75"),
        ],
        # ── Aurinkovoima — ground-mounted utility scale ───────────────────────
        "aurinkovoima": [
            ("Detailplaneering (tarvitt.)",         "Kohaliku omavalitsuse volikogu",             "Planeerimisseadus (PlanS, RT I 2015)"),
            ("YVA-menettely (KMH) (tarvitt.)",      "Keskkonnaamet (Environmental Board)",        "KMH-KSH seadus (RT I, 04.01.2013, 10) §7"),
            ("Rakentamislupa (ehitusluba)",          "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS, RT I 2015) §19"),
            ("Elektrienergia tootmise luba",         "Konkurentsiamet (Competition Authority)",   "Elektrituruseadus (ETS, RT I 2012) §14"),
            ("Verkkoliityntäsopimus (liitumisleping)", "Elektrilevi OÜ / Elering AS",             "Elektrituruseadus (ETS) §§73-83; RfG 2016/631"),
            ("Kasutusluba (use permit)",             "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS) §§61-75"),
        ],
        # ── Offshore wind ─────────────────────────────────────────────────────
        "tuulivoima_meri": [
            ("Ühisluba (combined offshore permit)", "Majandus- ja Kommunikatsiooniministeerium",  "Elektrituruseadus + Majandusvööndi seadus (offshore reform 2023)"),
            ("YVA-menettely (KMH, integrated)",    "Keskkonnaamet (within ühisluba process)",     "KMH-KSH seadus (RT I, 04.01.2013, 10) §6"),
            ("Kaitseministeerium radar clearance", "Kaitseministeerium (Ministry of Defence)",    "National security assessment (mandatory)"),
            ("Navigation/aviation safety",         "Transpordiamet (Transport Administration)",   "Lennundusseadus / Meresõiduohutuse seadus"),
            ("Elektrienergia tootmise luba",        "Konkurentsiamet (Competition Authority)",    "Elektrituruseadus (ETS, RT I 2012) §14"),
            ("Verkkoliityntäsopimus (liitumisleping)", "Elering AS (TSO)",                        "Elektrituruseadus (ETS) §§73-83; RfG 2016/631 / HVDC 2016/1447"),
        ],
        # ── Offshore wind alias ───────────────────────────────────────────────
        "offshore_wind": [
            ("Ühisluba (combined offshore permit)", "Majandus- ja Kommunikatsiooniministeerium",  "Elektrituruseadus + Majandusvööndi seadus (offshore reform 2023)"),
            ("YVA-menettely (KMH, integrated)",    "Keskkonnaamet (within ühisluba process)",     "KMH-KSH seadus (RT I, 04.01.2013, 10) §6"),
            ("Kaitseministeerium radar clearance", "Kaitseministeerium (Ministry of Defence)",    "National security assessment (mandatory)"),
            ("Navigation/aviation safety",         "Transpordiamet (Transport Administration)",   "Lennundusseadus / Meresõiduohutuse seadus"),
            ("Elektrienergia tootmise luba",        "Konkurentsiamet (Competition Authority)",    "Elektrituruseadus (ETS, RT I 2012) §14"),
            ("Verkkoliityntäsopimus (liitumisleping)", "Elering AS (TSO)",                        "Elektrituruseadus (ETS) §§73-83; RfG 2016/631 / HVDC 2016/1447"),
        ],
        # ── BESS ──────────────────────────────────────────────────────────────
        "BESS": [
            ("Rakentamislupa (ehitusluba)",         "Kohaliku omavalitsuse (Municipality)",       "Ehitusseadustik (EhS, RT I 2015) §19"),
            ("Paloturvallisuuslupa (Päästeamet)",  "Päästeamet (Estonian Rescue Board)",          "Tuleohutusseadus (Fire Safety Act)"),
            ("Elektrienergia tootmise luba (tarvitt.)", "Konkurentsiamet (Competition Authority)", "Elektrituruseadus (ETS, RT I 2012) §14 (jos >1 MW)"),
            ("YVA-menettely (KMH) (tarvitt.)",     "Keskkonnaamet",                               "KMH-KSH seadus §7 — ei erillistä BESS-lakia"),
            ("Verkkoliityntäsopimus (liitumisleping)", "Elektrilevi OÜ / Elering AS",             "Elektrituruseadus (ETS) §§73-83; RfG 2016/631 + DCC 2016/1388"),
            ("Kasutusluba (use permit)",            "Kohaliku omavalitsuse (Municipality)",       "Ehitusseadustik (EhS) §§61-75"),
        ],
        # ── SMR — regWarning: no nuclear law in Estonia ───────────────────────
        "SMR": [
            ("⚠️ Tuumaenergia seadus puuttuu — draft laki",  "Riigikogu (Parliament) — lakia ei hyväksytty",  "Kiirgusseadus (RT I 2019) — kattaa vain säteilykäytön; ei reaktorilupaa"),
            ("Periaatepäätös (Riigikogu otsus, tarvittaessa)", "Riigikogu (Parliament of Estonia)", "Vaatii parlamentin päätöksen — ei vakiintunutta menettelyä"),
            ("YVA-menettely (KMH)",                "Keskkonnaamet / Riigikogu",                   "KMH-KSH seadus (RT I, 04.01.2013, 10) §6 — pakollinen ydinlaitoksille"),
            ("Riigi eriplaneering",                "Rahandusministeerium / Riigikogu",             "Planeerimisseadus (PlanS, RT I 2015) §9"),
            ("Rakentamislupa (ehitusluba, tarvitt.)", "Kohaliku omavalitsuse / kansallinen viranomainen", "Ehitusseadustik (EhS) — ei ydinlaitoskohtaisia säännöksiä"),
            ("Elektrienergia tootmise luba",        "Konkurentsiamet (Competition Authority)",    "Elektrituruseadus (ETS, RT I 2012) §14"),
            ("Verkkoliityntäsopimus (liitumisleping)", "Elering AS (TSO)",                        "Elektrituruseadus (ETS) §§73-83"),
        ],
        # ── SMR variant alias ─────────────────────────────────────────────────
        "smr_ee": [
            ("⚠️ Tuumaenergia seadus puuttuu — draft laki",  "Riigikogu (Parliament) — lakia ei hyväksytty",  "Kiirgusseadus (RT I 2019) — kattaa vain säteilykäytön; ei reaktorilupaa"),
            ("Periaatepäätös (Riigikogu otsus, tarvittaessa)", "Riigikogu (Parliament of Estonia)", "Vaatii parlamentin päätöksen — ei vakiintunutta menettelyä"),
            ("YVA-menettely (KMH)",                "Keskkonnaamet / Riigikogu",                   "KMH-KSH seadus (RT I, 04.01.2013, 10) §6 — pakollinen ydinlaitoksille"),
            ("Riigi eriplaneering",                "Rahandusministeerium / Riigikogu",             "Planeerimisseadus (PlanS, RT I 2015) §9"),
            ("Rakentamislupa (ehitusluba, tarvitt.)", "Kohaliku omavalitsuse / kansallinen viranomainen", "Ehitusseadustik (EhS) — ei ydinlaitoskohtaisia säännöksiä"),
            ("Elektrienergia tootmise luba",        "Konkurentsiamet (Competition Authority)",    "Elektrituruseadus (ETS, RT I 2012) §14"),
            ("Verkkoliityntäsopimus (liitumisleping)", "Elering AS (TSO)",                        "Elektrituruseadus (ETS) §§73-83"),
        ],
        # ── Datakeskus ────────────────────────────────────────────────────────
        "datakeskus": [
            ("Detailplaneering (tarvitt.)",         "Kohaliku omavalitsuse volikogu",             "Planeerimisseadus (PlanS, RT I 2015)"),
            ("Rakentamislupa (ehitusluba)",          "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS, RT I 2015) §19"),
            ("YVA-menettely (KMH) (tarvitt.)",      "Keskkonnaamet",                              "KMH-KSH seadus §7"),
            ("Verkkoliityntäsopimus (liitumisleping)", "Elektrilevi OÜ / Elering AS",             "Elektrituruseadus (ETS) §§73-83; DCC 2016/1388"),
            ("Kasutusluba (use permit)",             "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS) §§61-75"),
        ],
        # ── Teollisuus ────────────────────────────────────────────────────────
        "teollisuus": [
            ("Detailplaneering (tarvitt.)",         "Kohaliku omavalitsuse volikogu",             "Planeerimisseadus (PlanS, RT I 2015)"),
            ("Keskkonnaluba (environmental permit)", "Keskkonnaamet (Environmental Board)",        "Keskkonnaseadustiku üldosa seadus (KeÜS, RT I 2011)"),
            ("YVA-menettely (KMH) (tarvitt.)",      "Keskkonnaamet",                              "KMH-KSH seadus (RT I, 04.01.2013, 10)"),
            ("Rakentamislupa (ehitusluba)",          "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS, RT I 2015) §19"),
            ("Kasutusluba (use permit)",             "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS) §§61-75"),
        ],
        # ── Asuinrakennus ─────────────────────────────────────────────────────
        "asuinrakennus": [
            ("Detailplaneering (tarvitt.)",         "Kohaliku omavalitsuse volikogu",             "Planeerimisseadus (PlanS, RT I 2015)"),
            ("Rakentamislupa (ehitusluba)",          "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS, RT I 2015) §19"),
            ("Naapurikuuleminen",                   "Kohaliku omavalitsuse / hakija",             "Ehitusseadustik (EhS) §§27-45"),
            ("Kasutusluba (use permit)",             "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS) §§61-75"),
        ],
        # ── Liikerakennus ─────────────────────────────────────────────────────
        "liikerakennus": [
            ("Detailplaneering (tarvitt.)",         "Kohaliku omavalitsuse volikogu",             "Planeerimisseadus (PlanS, RT I 2015)"),
            ("Rakentamislupa (ehitusluba)",          "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS, RT I 2015) §19"),
            ("Kasutusluba (use permit)",             "Kohaliku omavalitsuse (Municipality)",      "Ehitusseadustik (EhS) §§61-75"),
        ],
        # ── Maatalous ─────────────────────────────────────────────────────────
        "maatalous": [
            ("Rakentamislupa tai ehitusteatis",     "Kohaliku omavalitsuse (Municipality)",       "Ehitusseadustik (EhS, RT I 2015) §§19-26"),
            ("Keskkonnaluba (tarvitt., suuret tilat)", "Keskkonnaamet",                           "Keskkonnaseadustiku üldosa seadus (KeÜS) — IPPC-kynnys"),
            ("Põllumajandusamet maakasutusluba",   "Põllumajandusamet (Agricultural Board)",     "Maaparandusseadus — jos maankäyttö muuttuu"),
        ],
        # ── Hybridi (BESS + wind/solar) ───────────────────────────────────────
        "hybridi": [
            ("Detailplaneering",                   "Kohaliku omavalitsuse volikogu",              "Planeerimisseadus (PlanS, RT I 2015)"),
            ("YVA-menettely (KMH)",                "Keskkonnaamet (Environmental Board)",         "KMH-KSH seadus (RT I, 04.01.2013, 10)"),
            ("Rakentamislupa (ehitusluba)",         "Kohaliku omavalitsuse (Municipality)",       "Ehitusseadustik (EhS, RT I 2015) §19"),
            ("Paloturvallisuuslupa (BESS-osuus)",  "Päästeamet (Estonian Rescue Board)",          "Tuleohutusseadus (Fire Safety Act)"),
            ("Elektrienergia tootmise luba",        "Konkurentsiamet (Competition Authority)",    "Elektrituruseadus (ETS, RT I 2012) §14"),
            ("Verkkoliityntäsopimus (liitumisleping)", "Elering AS / Elektrilevi OÜ",             "Elektrituruseadus (ETS) §§73-83; RfG 2016/631"),
            ("Kasutusluba (use permit)",            "Kohaliku omavalitsuse (Municipality)",       "Ehitusseadustik (EhS) §§61-75"),
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Maakohtaiset liiteluettelot (ylikirjoittavat FI-oletuksen)
# Erityisesti ydinhankkeet: kansallisen turvallisuusviranomaisen dokumenttityypit
# ─────────────────────────────────────────────────────────────────────────────
_COUNTRY_LIITTEET: dict[str, dict[str, list[str]]] = {
    "SE": {
        "SMR": [
            "Sijaintikartta / Lägesbeskrivning (skala 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "SSM preliminär säkerhetsredovisning (PSR) — Kärntekniklag (SFS 1984:3)",
            "Miljökonsekvensbeskrivning (MKB) — Miljöbalken kap. 6",
            "Hydrogeologisk utredning (kylvattenresurs)",
            "Nätanslutningsplan (Svenska kraftnät)",
            "Detaljplan / kommunal markanvändningsplan",
            "Bolagsregistreringsutdrag (Bolagsverket)",
            "Fullmakt (om ombud företräder sökanden)",
        ],
        "smr_bess": [
            "Sijaintikartta / Lägesbeskrivning (skala 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "SSM preliminär säkerhetsredovisning (PSR) — Kärntekniklag (SFS 1984:3)",
            "Miljökonsekvensbeskrivning (MKB) — Miljöbalken kap. 6",
            "Brandsäkerhetsrapport BESS (NFPA 855 / EN 50604-1)",
            "Hydrogeologisk utredning (kylvattenresurs)",
            "Nätanslutningsplan (Svenska kraftnät)",
            "Detaljplan / kommunal markanvändningsplan",
            "Bolagsregistreringsutdrag (Bolagsverket)",
        ],
    },
    "DA": {
        "SMR": [
            "Kortbilag / Beliggenhedskort (1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Nuklær sikkerhedsredegørelse (Sundhedsstyrelsen / SIS) — Lov nr. 94/2003",
            "VVM-redegørelse (Vurdering af Virkninger på Miljøet) — Miljøvurderingsloven",
            "Hydrogeologisk undersøgelse (kølevandsbehov)",
            "Nettilslutningsplan (Energinet)",
            "Lokalplan / kommuneplanramme",
            "Virksomhedsregistreringsudskrift (CVR)",
            "Fuldmagt (hvis repræsentant handler på vegne af ansøger)",
        ],
        "smr_bess": [
            "Kortbilag / Beliggenhedskort (1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Nuklær sikkerhedsredegørelse (Sundhedsstyrelsen / SIS) — Lov nr. 94/2003",
            "VVM-redegørelse — Miljøvurderingsloven",
            "Brandsikkerhedsrapport BESS (NFPA 855 / EN 50604-1)",
            "Nettilslutningsplan (Energinet)",
            "Lokalplan / kommuneplanramme",
            "Virksomhedsregistreringsudskrift (CVR)",
        ],
    },
    "NO": {
        "SMR": [
            "Kart / Stedsbeskrivelse (1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Sikkerhetsanalyse (DSA — Direktoratet for strålevern og atomsikkerhet) — Strålevernloven",
            "Konsekvensutredning (KU) — Plan- og bygningsloven kap. 14",
            "Hydrogeologisk utredning (kjølevannsressurs)",
            "Nettilknytningsplan (Statnett)",
            "Reguleringsplan / kommuneplan",
            "Foretaksregistreringsutskrift (Brønnøysundregistrene)",
            "Fullmakt (dersom representant opptrer på vegne av søker)",
        ],
        "smr_bess": [
            "Kart / Stedsbeskrivelse (1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Sikkerhetsanalyse (DSA) — Strålevernloven (LOV-2000-05-12-36)",
            "Konsekvensutredning (KU) — Plan- og bygningsloven kap. 14",
            "Brannsikkerhetsrapport BESS (NFPA 855 / EN 50604-1)",
            "Nettilknytningsplan (Statnett)",
            "Reguleringsplan / kommuneplan",
            "Foretaksregistreringsutskrift (Brønnøysundregistrene)",
        ],
    },
    "PL": {
        "SMR": [
            "Mapa lokalizacyjna (skala 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Raport bezpieczeństwa (PAA — Państwowa Agencja Atomistyki) — Prawo atomowe",
            "Raport o oddziaływaniu na środowisko (OOŚ) — Ustawa środowiskowa",
            "Badanie hydrogeologiczne (zasoby wód chłodniczych)",
            "Plan przyłączenia do sieci (PSE S.A.)",
            "Miejscowy plan zagospodarowania przestrzennego (MPZP)",
            "Odpis z KRS / CEIDG",
            "Pełnomocnictwo (jeżeli reprezentant działa w imieniu wnioskodawcy)",
        ],
        "smr_bess": [
            "Mapa lokalizacyjna (skala 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Raport bezpieczeństwa (PAA — Państwowa Agencja Atomistyki) — Prawo atomowe",
            "Raport OOŚ — Ustawa środowiskowa",
            "Raport bezpieczeństwa pożarowego BESS (NFPA 855 / EN 50604-1)",
            "Plan przyłączenia do sieci (PSE S.A.)",
            "Miejscowy plan zagospodarowania przestrzennego (MPZP)",
            "Odpis z KRS / CEIDG",
        ],
    },
    "EE": {
        "SMR": [
            "Asukohakaart / Sijaintikartta (mõõtkava 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "⚠️ Kiirgusseaduse alusel kiirguspraktika litsents (Terviseamet) — ydinlaitokselle ei ole voimassa olevaa lupakategoriaa",
            "KMH aruanne (Keskkonnamõju hindamise aruanne) — KMH-KSH seadus §6",
            "Riigi eriplaneering (national spatial plan) — Planeerimisseadus §9",
            "Hüdrogeoloogiline uuring (jahutusvee ressurss / cooling water resource)",
            "Liitumisleping Elering AS-iga (grid connection agreement)",
            "Äriregistri väljavõte (Company Registry extract)",
            "Volikiri (if representative acts on behalf of applicant)",
            "⚠️ Tuumaenergia seadus (nuclear law) ei ole vastu võetud — tarvitaan erillinen oikeudellinen selvitys",
        ],
        "smr_ee": [
            "Asukohakaart / Sijaintikartta (mõõtkava 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "⚠️ Kiirgusseaduse alusel kiirguspraktika litsents (Terviseamet) — ydinlaitokselle ei ole voimassa olevaa lupakategoriaa",
            "KMH aruanne (Keskkonnamõju hindamise aruanne) — KMH-KSH seadus §6",
            "Riigi eriplaneering (national spatial plan) — Planeerimisseadus §9",
            "Hüdrogeoloogiline uuring (jahutusvee ressurss / cooling water resource)",
            "Liitumisleping Elering AS-iga (grid connection agreement)",
            "Äriregistri väljavõte (Company Registry extract)",
            "⚠️ Tuumaenergia seadus (nuclear law) ei ole vastu võetud — tarvitaan erillinen oikeudellinen selvitys",
        ],
        "tuulivoima_maa": [
            "Asukohakaart / Sijaintikartta (mõõtkava 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Detailplaneering (kohaliku omavalitsuse poolt kinnitatud)",
            "KMH aruanne (Keskkonnamõju hindamise aruanne) — KMH-KSH seadus",
            "Kaitseministeeriumi radarikooskõlastus (Ministry of Defence radar clearance)",
            "Mürauuringu aruanne (noise assessment — 45/40 dB(A) limit compliance)",
            "Ornitoloogiline uuring (bird impact assessment — Baltic flyway)",
            "Liitumisleping Elering AS-iga (grid connection agreement)",
            "Ehitusprojekt (construction project, licensed architect/engineer)",
            "Äriregistri väljavõte (Company Registry extract)",
        ],
        "tuulivoima_meri": [
            "Ühisluba taotlus (combined offshore permit application — MKM)",
            "KMH programm (scoping document, Keskkonnaamet approved)",
            "KMH aruanne (EIA report — marine ecology, birds, visual, noise)",
            "Kaitseministeeriumi radarikooskõlastus (radar clearance — mandatory)",
            "Merealade planeering kooskõla (Maritime Spatial Plan zone confirmation)",
            "Liitumisleping Elering AS-iga (grid connection agreement)",
            "Kalandusuuringute mõjuhinnang (fisheries impact assessment)",
            "Meresõiduohutuse hinnang (maritime navigation safety — Transpordiamet)",
            "Äriregistri väljavõte (Company Registry extract)",
        ],
    },
}

_SYSTEM = (
    "Käytä aina oikeita suomenkielisiä merkkejä: ä, ö, å. "
    "ÄLÄ KOSKAAN kirjoita 'a' tai 'o' silloin kun oikea merkki on 'ä' tai 'ö'. "
    "Tämä on kriittinen vaatimus. "
    "Olet NCE Permit AI -asiantuntija, joka avustaa energia-alan lupahakemusten "
    "laadinnassa. Kirjoitat selkeää, virallista kieltä konsulttiraporttityyliin. "
    "Viittaat aina voimassa olevaan lainsäädäntöön. "
    "AJATTELUKETJU — ENNEN KIRJOITTAMISTA käy jokainen hakemus läpi tässä järjestyksessä:\n"
    "1. ANALYSOI: Tunnista hankkeen ominaispiirteet ja riskitekijät "
    "(hanketyyppi, sijainti, koko, maa, relevantit viranomaiset).\n"
    "2. HAE: Paikanna relevanteimmat säädösvaatimukset JA ennakkotapaukset "
    "tälle hankeprofiilille annetusta RAG-kontekstista.\n"
    "3. VERTAA: Vertaa hanketta ennakkotapauksiin — mitkä riskit olivat läsnä, "
    "miten ne ratkaistiin, mikä teki hankkeista onnistuneita tai epäonnistuneita.\n"
    "4. ARVIOI: Määritä mitkä riskit ovat kriittisimpiä hyväksyntätodennäköisyyden kannalta.\n"
    "5. SUOSITTELE: Ehdota konkreettisia toimia hyväksyntätodennäköisyyden parantamiseksi — "
    "tietyt asiakirjat, selvitykset tai suunnittelumuutokset.\n"
    "6. ELINKAARI: Laajenna suositukset hankkeen seuraavaan elinkaarivaiheeseen VAIN JOS "
    "haetusta kontekstista löytyy riittävästi tietoa kyseisestä vaiheesta. Jos haettu data "
    "ei kata myöhempiä vaiheita (esim. rakentaminen tai käyttö), kirjoita eksplisiittisesti: "
    "'Haettu lähdeaineisto ei riitä myöhemmän vaiheen suosituksiin.' Älä generoi spekulatiivisia "
    "suosituksia vaiheille, joista ei ole haettua säädös- tai ennakkotapausaineistoa.\n"
    "Jokainen vaihe näkyy tulosraportissa omana alaotsikkonaan (### ANALYSOI jne.). "
    "KRIITTINEN SÄÄNTÖ — EPÄVARMA TIETO: Jos jokin yksittäinen fakta, vaatimus tai "
    "lakiviite on epävarma, puuttuu annetusta kontekstista tai vaatii erikoisasiantuntemusta, "
    "lisää välittömästi kyseisen lauseen tai kappaleen jälkeen merkintä "
    "'⚠️ Asiantuntijatarkistus suositellaan'. Älä koskaan täytä tietopuutteita arvauksilla "
    "tai spekulaatiolla — mieluummin merkitse asia epävarmaksi kuin generoi väärää tietoa. "
    "Kaikki tuottamasi teksti on AI-luonnos joka vaatii asiantuntijatarkistuksen. "
    "PAKOLLINEN MINIMIVAATIMUS: Koko vastauksessa TÄYTYY esiintyä merkintä "
    "'⚠️ Asiantuntijatarkistus suositellaan' VÄHINTÄÄN 2 kertaa ja ENINTÄÄN 3 kertaa. "
    "Jos olet kirjoittanut koko tekstin etkä ole vielä lisännyt kahta merkintää, "
    "lisää ne sopiviin kohtiin ennen vastauksen päättämistä. "
    "Tämä on ehdoton vaatimus — vastaus on virheellinen jos merkintöjä on alle 2. "
    "HUOM-LAUSEOHJE: Jokainen ⚠️-merkintä on kirjoitettava täytenä lauseena joka alkaa "
    "merkinnällä. ÄLÄ KOSKAAN kirjoita irtonaista lausetta joka alkaa pienellä kirjaimella "
    "tai kesken ajatuksen — merkintä on aina oma itsenäinen virkkeensä. "
    "YHTEYSTIETOSÄÄNTÖ: Älä koskaan generoi hakijan osoitetta, puhelinnumeroa, "
    "sähköpostia tai Y-tunnusta tekstiosioihin — käytä vain luvan sisältöön liittyviä tietoja. "
    "LAUSERAKENNE: Kirjoita lyhyitä, selkeitä virkkeitä (enintään 2 lausetta per kappale). "
    "Vältä pitkiä luettelomaisesti yhdisteltyjä juridisia lauseita. "
    "KIRJOITUSOHJE: Kirjoita kaikki suomenkieliset sanat oikein diakriittimerkein — "
    "ä (ei a), ö (ei o), å (ei a). Esimerkkejä: käytettävyys, jäähdytys, häiriötilanne, "
    "yhteydenotto, järjestelmä, ympäristö, lämpö, päätös, näkökulma, tärkeä, löytää. "
    "TÄRKEÄÄ: Perusta JOKAINEN suositus, riskiarvio ja elinkaarisuositus yksinomaan haettuihin "
    "säädöslähteisiin ja ennakkotapauksiin tästä kontekstista. Jos jonkin ajatteluketjun vaiheen "
    "tieto puuttuu, ilmoita eksplisiittisesti mikä puuttuu oletuksien tekemisen sijaan. "
    "Älä koskaan generoi sisältöä joka ei perustu haettuihin lähteisiin."
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
    "ET": (
        "KRIITILINE KEELUNÕUE: PEAD kirjutama IGA sõna selles loataotluses eesti keeles. "
        "KÕIK pealkirjad, lõigud, loendipunktid, allmärkused ja märkused peavad olema eesti keeles. "
        "ÄRA lisa soome keelseid sõnu ega lauseid väljundisse. "
        "Soome seadusenumbrid (nt YSL 527/2014, MRL 132/1999) võivad esineda juriidiliste "
        "identifikaatoritena — lisa alati kõrval eestikeelne seaduse nimetus. Soome pärisnimed "
        "nagu linnanimed, firmade nimed ja ametite lühendid (ELY, STUK, Luova, Fingrid, Traficom) "
        "on aktsepteeritavad ainult pärisnimede kujul.\n\n"
    ),
}

_PHASE_INSTRUCTIONS: dict[str, str] = {
    "esiselvitys": (
        "VAIHEEN KIRJOITUSOHJE — ESISELVITYS:\n"
        "- Kyse on alustavasta selvityksestä: sijainti, kaava, ympäristövaikutukset\n"
        "- Ei sitovia hakemuksia vielä — kaikki on selvityksen tasolla\n"
        "- Viranomaisyhteydet tarkoittavat ennakkoneuvotteluja, ei hakemuksia\n"
        "- Käytä sävyä: 'selvitetään', 'arvioidaan', 'alustavasti', 'on tarkoitus'\n"
        "- Tekniset parametrit ovat arvioita, ei lopullisia\n"
        "- Älä viittaa liitteisiin joita ei ole vielä tehty"
    ),
    "lupavaihe": (
        "VAIHEEN KIRJOITUSOHJE — LUPAVAIHE:\n"
        "- Käytä konkreettisia teknisiä parametreja (määrät, mitat, luokat) — ei yleistyksiä\n"
        "- Viittaa liitteisiin nimeltä: 'ks. Liite 5: Paloturvallisuusselvitys', "
        "'ks. Liite 8: Sähköliitynnän suunnitelma'\n"
        "- Mainitse naapurikuulemisen status (tehty / kesken / tulossa)\n"
        "- Mainitse kemikaali-ilmoitusvelvollisuudet ja kynnysarvot numeroin\n"
        "- Käytä sävyä: 'haetaan', 'toimitetaan', 'vaaditaan', 'edellyttää'\n"
        "- Kirjoita kattavasti — osiossa 1 vähintään 5 kappaletta, osiossa 2 vähintään 4 kappaletta\n"
        "- Jokainen lupa selitetään lyhyesti: sisältö, vastuuviranomainen, käsittelyaika"
    ),
    "rakentaminen": (
        "VAIHEEN KIRJOITUSOHJE — RAKENTAMISVAIHE:\n"
        "- Mainitse aloitusilmoitus rakennusvalvonnalle ennen töiden aloittamista\n"
        "- Vastaava työnjohtaja on nimitettävä ja hyväksytettävä rakennusvalvonnassa\n"
        "- Tarkastusasiakirja on oltava käytössä koko rakentamisen ajan\n"
        "- Katselmusaikataulu: pohjakatselmus, rakennekatselmus, loppukatselmus\n"
        "- Käyttöönottotarkastus ennen toiminnan aloittamista (sähkö, pelastuslaitos)\n"
        "- Käytä sävyä: 'toteutetaan', 'tarkastetaan', 'otetaan käyttöön', 'varmistetaan'\n"
        "- Viittaa myönnettyyn rakentamislupaan ja sen ehtoihin"
    ),
}
# Alias so both "Rakentaminen" and "Rakentamisvaihe" (frontend values) resolve correctly
_PHASE_INSTRUCTIONS["rakentamisvaihe"] = _PHASE_INSTRUCTIONS["rakentaminen"]
_HANKE_CFG["BESS"]["context_extra_phases"]["rakentamisvaihe"] = (
    _HANKE_CFG["BESS"]["context_extra_phases"]["rakentaminen"]
)
# Country-specific RAG queries — override Finnish default queries for country chunk retrieval.
# Used in _rag_context step 1 so cross-lingual embedding similarity stays above threshold.
_COUNTRY_RAG_QUERIES: dict[str, dict[str, list[str]]] = {
    "DE": {
        "tuulivoima_maa": [
            "Windenergie Onshore BImSchG Genehmigung Deutschland",
            "Windpark Artenschutz BNatSchG Rotmilan TAK Abstand",
            "WindBG Flächenziel Raumordnung Bebauungsplan",
        ],
        "tuulivoima_meri": [
            "Offshore Windenergie WindSeeG BSH Genehmigung",
            "Offshore Windpark HVDC TenneT Netzanschluss",
        ],
        "BESS": [
            "Batteriespeicher BESS BImSchG Genehmigung Deutschland",
            "Stromspeicher Lithium EnWG Netzentgelt FCR Regelenergie",
        ],
        "SMR": [
            "Kernkraft SMR Atomgesetz AtG Deutschland Verbot",
            "nuclear reactor permit Germany prohibition AtG",
        ],
        "aurinkovoima": [
            "Photovoltaik Solar Genehmigung Deutschland EEG Freiflächenanlage",
            "Agri-PV Floating-PV Solaranlage Genehmigung",
        ],
    },
    "EE": {
        "tuulivoima_maa": [
            "tuuleenergia tuulpark luba Eesti ehitusluba KMH",
            "wind energy permit Estonia environmental impact assessment",
            "tuulienergia võrguga ühendamine Elering liitumisleping",
        ],
        "BESS": [
            "akuenergia salvestus Eesti luba elektriturg",
            "battery storage permit Estonia electricity market act",
        ],
        "SMR": [
            "tuumaenergia SMR seadus Eesti Riigikogu kiirgus",
            "nuclear energy law Estonia parliament permit missing",
        ],
    },
}

# Country-variant and new-type aliases — map to nearest base type so _HANKE_CFG
# lookups never raise KeyError. _COUNTRY_LUVAT overrides the permit list per country.
_HANKE_CFG["smr_se"]       = _HANKE_CFG["SMR"]
_HANKE_CFG["smr_no"]       = _HANKE_CFG["SMR"]
_HANKE_CFG["smr_da"]       = _HANKE_CFG["SMR"]
_HANKE_CFG["smr_de"]       = _HANKE_CFG["SMR"]
_HANKE_CFG["smr_ee"]       = _HANKE_CFG["SMR"]
_HANKE_CFG["offshore_wind"] = _HANKE_CFG["tuulivoima_meri"]
_HANKE_CFG["egs"]           = _HANKE_CFG["aurinkovoima"]


_WRITE_INSTRUCTION: dict[str, str] = {
    "FI": ("Kirjoita suomeksi seuraavat neljä osiota selkeästi eroteltuna otsikoilla. "
           "Jokainen osio sisältää ajatteluketjun vaiheet näkyvinä alaotsikkoina (### ANALYSOI, ### HAE jne.). "
           "Viittaa lakeihin lyhentein hakasulkeissa, esim. [YSL §27] tai [Rakentamislaki 751/2023]. "
           "Kirjoita lyhyitä virkkeitä — enintään kaksi lausetta per kappale, ei pitkiä juridisia luettelolauseita. "
           "PAKOLLINEN VAATIMUS: Lisää merkintä '⚠️ Asiantuntijatarkistus suositellaan' TÄSMÄLLEEN 2–3 kertaa koko vastauksessa — ei vähemmän, ei enemmän. "
           "Sijoita merkinnät Hankkeen kuvaus- ja Perustelut-osioihin epävarmojen tai asiantuntemusta vaativien kohtien jälkeen. "
           "Älä spekuloi äläkä täytä tietopuutteita oletuksilla. "
           "TÄRKEÄÄ: Perusta jokainen suositus, riskiarvio ja elinkaarisuositus yksinomaan haettuihin säädöslähteisiin ja ennakkotapauksiin. "
           "Jos tietoa puuttuu, ilmoita eksplisiittisesti mikä puuttuu — älä koskaan generoi sisältöä joka ei perustu haettuihin lähteisiin:"),
    "EN": ("Write the following four sections in English, clearly separated by headings. "
           "Each section contains the reasoning-chain steps as visible sub-headings (### ANALYZE, ### RETRIEVE, etc.). "
           "Include inline law citations in brackets, e.g. [EIA Act] or [Building Act 751/2023]. "
           "If any information is uncertain, missing or requires specialist expertise, "
           "add the marker '⚠️ Expert review recommended' immediately after that point — "
           "do not speculate or fill gaps with assumptions. "
           "IMPORTANT: Base every recommendation, risk assessment and lifecycle extension strictly on "
           "retrieved regulatory sources and precedent cases found in this context. "
           "If information is missing for any reasoning step, explicitly state what is missing. "
           "Never generate content that is not grounded in the retrieved sources:"),
    "SE": ("Skriv följande fyra avsnitt på svenska, tydligt åtskilda med rubriker. "
           "Varje avsnitt innehåller resonemangsstegen som synliga underrubriker (### ANALYSERA, ### HÄMTA osv.). "
           "Inkludera lagcitat i hakparentes, t.ex. [PBL 2010:900] eller [MB 1998:808]. "
           "Om någon uppgift är osäker, saknas eller kräver specialistkunskap, "
           "lägg till märkningen '⚠️ Expertgranskning rekommenderas' direkt efter det berörda stycket — "
           "spekulera inte och fyll inte i kunskapsluckor med antaganden. "
           "VIKTIGT: Basera varje rekommendation, riskbedömning och livscykelavsnitt strikt på "
           "hämtade regulatoriska källor och prejudikatfall i denna kontext. "
           "Om information saknas för något resonemangssteg, ange explicit vad som saknas. "
           "Generera aldrig innehåll som inte är grundat i de hämtade källorna:"),
    "DA": ("Skriv følgende fire afsnit på dansk, tydeligt adskilt med overskrifter. "
           "Hvert afsnit indeholder ræsonnementstrinnene som synlige underoverskrifter (### ANALYSER, ### HENT osv.). "
           "Inkluder lovcitater i kantede parenteser, f.eks. [PBL §12] eller [MBL]. "
           "Hvis en oplysning er usikker, mangler eller kræver specialistviden, "
           "tilføj mærket '⚠️ Ekspertgennemgang anbefales' umiddelbart efter det pågældende afsnit — "
           "spekuler ikke og udfyld ikke videnshuller med antagelser. "
           "VIGTIGT: Basér enhver anbefaling, risikovurdering og livscyklusudvidelse strengt på "
           "hentede regulatoriske kilder og præcedenssager i denne kontekst. "
           "Hvis oplysninger mangler for et ræsonnementstrin, anfør eksplicit hvad der mangler. "
           "Generer aldrig indhold, der ikke er baseret på de hentede kilder:"),
    "NO": ("Skriv følgende fire seksjoner på norsk, tydelig atskilt med overskrifter. "
           "Hver seksjon inneholder resonnementstrinnene som synlige underoverskrifter (### ANALYSER, ### HENT osv.). "
           "Inkluder lovhenvisninger i hakeparenteser, f.eks. [PBL §12-1] eller [NVE-forskrift]. "
           "Hvis en opplysning er usikker, mangler eller krever spesialistkompetanse, "
           "legg til merket '⚠️ Ekspertgjennomgang anbefales' umiddelbart etter det aktuelle avsnittet — "
           "ikke spekuler og ikke fyll kunnskapshull med antakelser. "
           "VIKTIG: Baser enhver anbefaling, risikovurdering og livssyklusutvidelse strengt på "
           "hentede regulatoriske kilder og presedenssaker i denne konteksten. "
           "Hvis informasjon mangler for et resonnementstrinn, angi eksplisitt hva som mangler. "
           "Generer aldri innhold som ikke er grunnlagt i de hentede kildene:"),
    "PL": ("Napisz następujące cztery sekcje po polsku, wyraźnie oddzielone nagłówkami. "
           "Każda sekcja zawiera kroki łańcucha rozumowania jako widoczne podtytuły (### ANALIZA, ### POBIERZ itp.). "
           "Umieść odniesienia do przepisów w nawiasach kwadratowych, np. [Ustawa OOŚ] lub [Prawo budowlane Art. 28]. "
           "Jeśli jakakolwiek informacja jest niepewna, brakuje jej lub wymaga wiedzy specjalistycznej, "
           "dodaj oznaczenie '⚠️ Zalecana weryfikacja przez eksperta' bezpośrednio po danym fragmencie — "
           "nie spekuluj i nie uzupełniaj luk w wiedzy założeniami. "
           "WAŻNE: Każdą rekomendację, ocenę ryzyka i rozszerzenie cyklu życia opieraj wyłącznie na "
           "pobranych źródłach regulacyjnych i sprawach precedensowych w tym kontekście. "
           "Jeśli informacje dla jakiegokolwiek kroku rozumowania są niedostępne, explicite podaj co brakuje. "
           "Nigdy nie generuj treści, która nie jest oparta na pobranych źródłach:"),
}

# Hanketyypit joissa epävarmuusmerkintä on erityisen kriittinen
_CRITICAL_HANKE_TYPES: set[str] = {"SMR", "smr_bess", "ymparistolupa"}

_CRITICAL_EXTRA: dict[str, str] = {
    "FI": ("⚠️ ERITYISOHJE TÄLLE HANKETYYPILLE: {hanketyyppi}-hankkeissa viranomaisvaatimukset, "
           "turvallisuusmääräykset ja lakiperusta ovat erityisen tarkkoja ja muuttuvia. "
           "Käytä merkintää '⚠️ Asiantuntijatarkistus suositellaan' AINA kun: "
           "(a) viranomaisvaatimus tai lupamenettely on epäselkä tai mahdollisesti muuttunut, "
           "(b) tekninen raja-arvo tai parametri ei perustu annettuun dokumentaatioon, "
           "(c) lainsäädäntötieto on puutteellinen tai tulkinnanvarainen. "
           "Älä koskaan generoi lukuja, aikatauluja tai vaatimuksia ilman dokumentoitua perustetta."),
    "EN": ("⚠️ SPECIAL INSTRUCTION FOR THIS PROJECT TYPE: For {hanketyyppi} projects, regulatory requirements, "
           "safety regulations and statutory basis are particularly precise and subject to change. "
           "Use the marker '⚠️ Expert review recommended' WHENEVER: "
           "(a) a regulatory requirement or permit procedure is unclear or potentially changed, "
           "(b) a technical limit or parameter is not grounded in the provided documentation, "
           "(c) legal information is incomplete or open to interpretation. "
           "Never generate figures, timelines or requirements without a documented basis."),
    "SE": ("⚠️ SÄRSKILD INSTRUKTION FÖR DENNA PROJEKTTYP: För {hanketyyppi}-projekt är myndighetskrav, "
           "säkerhetsföreskrifter och rättslig grund särskilt precisa och föränderliga. "
           "Använd märkningen '⚠️ Expertgranskning rekommenderas' ALLTID när: "
           "(a) ett myndighetskrav eller tillståndsförfarande är oklart eller möjligen förändrat, "
           "(b) ett tekniskt gränsvärde eller en parameter inte grundar sig på given dokumentation, "
           "(c) lagstiftningsinformation är ofullständig eller tolkningsbar. "
           "Generera aldrig siffror, tidsplaner eller krav utan dokumenterat underlag."),
    "DA": ("⚠️ SÆRLIG INSTRUKTION FOR DENNE PROJEKTTYPE: For {hanketyyppi}-projekter er myndighedskrav, "
           "sikkerhedsforskrifter og retsgrundlag særligt præcise og foranderlige. "
           "Brug mærket '⚠️ Ekspertgennemgang anbefales' ALTID når: "
           "(a) et myndighedskrav eller tilladelsesprocedure er uklart eller muligvis ændret, "
           "(b) en teknisk grænseværdi eller parameter ikke er baseret på den givne dokumentation, "
           "(c) lovgivningsoplysninger er ufuldstændige eller åbne for fortolkning. "
           "Generer aldrig tal, tidsplaner eller krav uden dokumenteret grundlag."),
    "NO": ("⚠️ SPESIELL INSTRUKS FOR DENNE PROSJEKTTYPEN: For {hanketyyppi}-prosjekter er myndighetskrav, "
           "sikkerhetsforskrifter og rettsgrunnlag særlig presise og i endring. "
           "Bruk merket '⚠️ Ekspertgjennomgang anbefales' ALLTID når: "
           "(a) et myndighetskrav eller tillatelsesprosedyre er uklart eller muligens endret, "
           "(b) en teknisk grenseverdi eller parameter ikke er basert på gitt dokumentasjon, "
           "(c) lovgivningsinformasjon er ufullstendig eller tolkbar. "
           "Generer aldri tall, tidsplaner eller krav uten dokumentert grunnlag."),
    "PL": ("⚠️ SPECJALNA INSTRUKCJA DLA TEGO TYPU PROJEKTU: W projektach {hanketyyppi} wymogi regulacyjne, "
           "przepisy bezpieczeństwa i podstawa prawna są szczególnie precyzyjne i zmienne. "
           "Używaj oznaczenia '⚠️ Zalecana weryfikacja przez eksperta' ZAWSZE gdy: "
           "(a) wymóg regulacyjny lub procedura zezwolenia jest niejasna lub mogła ulec zmianie, "
           "(b) wartość graniczna techniczna lub parametr nie wynika z dostarczonej dokumentacji, "
           "(c) informacje prawne są niekompletne lub otwarte na interpretację. "
           "Nigdy nie generuj liczb, harmonogramów ani wymogów bez udokumentowanej podstawy."),
}

_PROMPT_HEADERS: dict[str, dict[str, str]] = {
    "FI": {
        "intro":        "Laadi lupahakemusluonnos seuraavalle hankkeelle:",
        "rag_intro":    "Alla on relevanttia dokumentaatiota (Fingrid, Tukes, Ympäristöministeriö):",
        "kuvaus":       "HANKKEEN KUVAUS",
        "perustelut":   "PERUSTELUT JA TARVE",
        "luvat":        "LUPAMENETTELYJEN KUVAUS",
        "toimenpiteet": "SEURAAVAT TOIMENPITEET",
        "kuvaus_inst":  ("Kirjoita tämä osio kahdessa näkyvässä vaiheessa:\n\n"
                         "### ANALYSOI\n"
                         "Tunnista hankkeen ominaispiirteet ja riskitekijät: hanketyyppi, sijainti, koko, "
                         "maa ja relevantit viranomaiset. Mainitse hanketyypille tyypilliset tekniset parametrit.\n\n"
                         "### HAE\n"
                         "Paikanna relevanteimmat säädösvaatimukset ja ennakkotapaukset tälle hankeprofiilille. "
                         "Kirjoita 3–4 kappaleen kuvaus: tarkoitus, tekniset tiedot, sijainti, "
                         "verkkoon liittyminen ja ympäristövaikutukset. Osion on oltava riittävän "
                         "kattava ennakkoneuvottelua varten."),
        "kuvaus_extra": " Ota huomioon annettu sijainti- ja ympäristövaikutustieto.",
        "perustelut_inst": ("Kirjoita tämä osio kahdessa näkyvässä vaiheessa:\n\n"
                            "### VERTAA\n"
                            "Vertaa tätä hanketta RAG-kontekstin ennakkotapauksiin: mitkä riskit olivat läsnä, "
                            "miten ne ratkaistiin, mikä teki hankkeista onnistuneita tai epäonnistuneita.\n\n"
                            "### ARVIOI\n"
                            "Määritä kriittisimmät riskit hyväksyntätodennäköisyyden kannalta. "
                            "Kirjoita 2–3 kappaleen perustelu miksi hanke on tarpeellinen "
                            "(energiajärjestelmän näkökulma, Suomen ilmastotavoitteet, "
                            "aluetaloudelliset vaikutukset, teknologiset edut) ja nimeä "
                            "suurin yksittäinen hyväksyntäriski."),
        "luvat_inst":   ("### HAE — LUPAMENETTELYT\n"
                         "Selitä lyhyesti (1–2 lausetta per lupa) mitä kukin tarvittava lupa "
                         "koskee, miksi se vaaditaan tälle hankkeelle ja mikä viranomainen käsittelee sen. "
                         "Viittaa relevantteihin ennakkotapauksiin tai erityisvaatimuksiin tarvittaessa."),
        "luvat_extra":  " Viittaa erityisesti kohdeviranomaisen {auth} prosesseihin ja vaatimuksiin.",
        "toimenpiteet_first": ("Kunnan rakennusvalvonnan ennakkoneuvottelu + kaavatarkastus — "
                               "Hakija / {kunta}n rakennusvalvonta — 1–2 viikon sisällä"),
        "toimenpiteet_inst": ("Kirjoita tämä osio kahdessa näkyvässä vaiheessa:\n\n"
                              "### SUOSITTELE\n"
                              "Ensimmäinen toimenpide on AINA: \"{first}\".\n"
                              "Listaa sen jälkeen täsmälleen 4 konkreettista toimenpidettä, jotka "
                              "parantavat hyväksyntätodennäköisyyttä (selvitykset, lausunnot, "
                              "suunnittelumuutokset, asiakirjat). "
                              "Muoto: numero. Toimenpide – Vastuutaho – Aikataulu\n\n"
                              "### ELINKAARI\n"
                              "Vaihe 6: Laajenna suositukset hankkeen seuraavaan elinkaarivaiheeseen "
                              "VAIN JOS haetusta kontekstista löytyy riittävästi aineistoa kyseisestä "
                              "vaiheesta. Jos aineisto ei kata myöhempiä vaiheita, kirjoita: "
                              "'Haettu lähdeaineisto ei riitä myöhemmän vaiheen suosituksiin.' "
                              "Muoto: 6. Toimenpide – Vastuutaho – Aikataulu"),
        "toimenpiteet_vaihe": " Ota huomioon hankkeen nykyinen vaihe: {vaihe}.",
        "phase_label":        "Hankkeen vaihe",
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
        "kuvaus_inst":  ("Write this section in two visible steps:\n\n"
                         "### ANALYZE\n"
                         "Identify the project's key characteristics and risk factors: type, location, size, "
                         "country, relevant authorities. Include typical technical parameters for this project type.\n\n"
                         "### RETRIEVE\n"
                         "Identify the most relevant regulatory requirements and precedent cases for this "
                         "project profile from the RAG context. Write a 3–4 paragraph description: purpose, "
                         "technical details, location, grid connection and environmental impacts. "
                         "The section must be comprehensive enough for pre-consultation."),
        "kuvaus_extra": " Take into account the provided location and environmental impact information.",
        "perustelut_inst": ("Write this section in two visible steps:\n\n"
                            "### COMPARE\n"
                            "Compare this project against precedent cases from the RAG context: what risks "
                            "were present, how were they resolved, what made projects succeed or fail.\n\n"
                            "### ASSESS\n"
                            "Determine which risks are most critical for approval likelihood in this case. "
                            "Write a 2–3 paragraph justification for why the project is necessary "
                            "(energy system perspective, Finland's climate targets, "
                            "regional economic impacts) and name the single greatest approval risk."),
        "luvat_inst":   ("### RETRIEVE — PERMITS\n"
                         "Briefly explain (1–2 sentences per permit) what each required permit covers, "
                         "why it is required for this project and which authority handles it. "
                         "Reference relevant precedents or special requirements where applicable."),
        "luvat_extra":  " Refer especially to the target authority {auth}'s processes and requirements.",
        "toimenpiteet_first": ("Pre-consultation with municipality building control + zoning review — "
                               "Applicant / {kunta} Building Control — within 1–2 weeks"),
        "toimenpiteet_inst": ("Write this section in two visible steps:\n\n"
                              "### RECOMMEND\n"
                              "The first step is ALWAYS: \"{first}\".\n"
                              "Then list 4 concrete actions that improve approval probability "
                              "(studies, statements, design changes, documents). "
                              "Format: number. Action – Responsible party – Timeline\n\n"
                              "### LIFECYCLE\n"
                              "Step 6: Extend the recommendations to the next project lifecycle phase "
                              "ONLY IF the retrieved context contains sufficient data about that phase. "
                              "If retrieved sources do not cover later phases, explicitly state: "
                              "'Insufficient source data for later phase recommendations.' "
                              "Format: 6. Action – Responsible party – Timeline"),
        "toimenpiteet_vaihe": " Take into account the current project phase: {vaihe}.",
        "phase_label":        "Project phase",
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
        "kuvaus_inst":  ("Skriv detta avsnitt i två synliga steg:\n\n"
                         "### ANALYSERA\n"
                         "Identifiera projektets nyckelkarakteristika och riskfaktorer: typ, plats, storlek, "
                         "land och relevanta myndigheter. Inkludera typiska tekniska parametrar.\n\n"
                         "### HÄMTA\n"
                         "Identifiera de mest relevanta lagkraven och prejudikatfallen för denna profil "
                         "från RAG-kontexten. Skriv en 3–4 stycken beskrivning: syfte, tekniska detaljer, "
                         "plats, nätanslutning och miljöpåverkan. "
                         "Avsnittet måste vara tillräckligt utförligt för förkonsultation."),
        "kuvaus_extra": " Beakta den angivna plats- och miljöpåverkansinformationen.",
        "perustelut_inst": ("Skriv detta avsnitt i två synliga steg:\n\n"
                            "### JÄMFÖR\n"
                            "Jämför detta projekt med prejudikatfall från RAG-kontexten: vilka risker förekom, "
                            "hur löstes de, vad gjorde projekten framgångsrika eller misslyckade.\n\n"
                            "### BEDÖM\n"
                            "Fastställ vilka risker är mest kritiska för godkännandesannolikheten. "
                            "Skriv en 2–3 stycken motivering till varför projektet är nödvändigt "
                            "(energisystemets perspektiv, Finlands klimatmål, "
                            "regionala ekonomiska effekter) och namnge den enskilt största risken."),
        "luvat_inst":   ("### HÄMTA — TILLSTÅND\n"
                         "Förklara kortfattat (1–2 meningar per tillstånd) vad varje nödvändigt tillstånd "
                         "gäller, varför det krävs och vilken myndighet handlägger det. "
                         "Hänvisa till relevanta prejudikat eller särskilda krav vid behov."),
        "luvat_extra":  " Hänvisa särskilt till målmyndighetens {auth} processer och krav.",
        "toimenpiteet_first": ("Förkonsultation med kommunens byggnadstillsyn + planläggningsöversyn — "
                               "Sökande / {kunta}s byggnadstillsyn — inom 1–2 veckor"),
        "toimenpiteet_inst": ("Skriv detta avsnitt i två synliga steg:\n\n"
                              "### REKOMMENDERA\n"
                              "Det första steget är ALLTID: \"{first}\".\n"
                              "Lista sedan 4 konkreta åtgärder som förbättrar godkännandesannolikheten "
                              "(utredningar, yttranden, designändringar, dokument). "
                              "Format: nummer. Åtgärd – Ansvarig part – Tidslinje\n\n"
                              "### LIVSCYKEL\n"
                              "Steg 6: Utöka rekommendationerna till nästa projektlivscykelfas "
                              "ENDAST OM hämtad kontext innehåller tillräckliga data om den fasen. "
                              "Om källmaterialet inte täcker senare faser, ange explicit: "
                              "'Otillräckliga källdata för rekommendationer för senare faser.' "
                              "Format: 6. Åtgärd – Ansvarig part – Tidslinje"),
        "toimenpiteet_vaihe": " Beakta projektets nuvarande fas: {vaihe}.",
        "phase_label":        "Projektfas",
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
        "kuvaus_inst":  ("Skriv dette afsnit i to synlige trin:\n\n"
                         "### ANALYSER\n"
                         "Identificer projektets nøglekarakteristika og risikofaktorer: type, placering, størrelse, "
                         "land og relevante myndigheder. Medtag typiske tekniske parametre.\n\n"
                         "### HENT\n"
                         "Identificer de mest relevante lovkrav og præcedenssager for denne profil "
                         "fra RAG-konteksten. Skriv en beskrivelse på 3–4 afsnit: formål, tekniske detaljer, "
                         "placering, nettilslutning og miljøpåvirkninger. "
                         "Afsnittet skal være tilstrækkeligt fyldestgørende til forhåndskonsultation."),
        "kuvaus_extra": " Tag hensyn til de angivne oplysninger om placering og miljøpåvirkning.",
        "perustelut_inst": ("Skriv dette afsnit i to synlige trin:\n\n"
                            "### SAMMENLIGN\n"
                            "Sammenlign dette projekt med præcedenssager fra RAG-konteksten: hvilke risici "
                            "var til stede, hvordan blev de løst, hvad gjorde projekterne succesrige eller mislykkede.\n\n"
                            "### VURDER\n"
                            "Fastslå hvilke risici er mest kritiske for godkendelsessandsynligheden. "
                            "Skriv en begrundelse på 2–3 afsnit for, hvorfor projektet er nødvendigt "
                            "(energisystemperspektiv, Finlands klimamål, regionale økonomiske virkninger) "
                            "og nævn den største enkeltrisiko."),
        "luvat_inst":   ("### HENT — TILLADELSER\n"
                         "Forklar kort (1–2 sætninger pr. tilladelse) hvad hver nødvendig tilladelse dækker, "
                         "hvorfor den kræves og hvilken myndighed behandler den. "
                         "Henvis til relevante præcedenser eller særlige krav efter behov."),
        "luvat_extra":  " Henvis især til målmyndighedens {auth} processer og krav.",
        "toimenpiteet_first": ("Forhåndskonsultation med kommunens byggesagsafdeling + planrevision — "
                               "Ansøger / {kunta} byggesagsafdeling — inden for 1–2 uger"),
        "toimenpiteet_inst": ("Skriv dette afsnit i to synlige trin:\n\n"
                              "### ANBEFAL\n"
                              "Det første trin er ALTID: \"{first}\".\n"
                              "Angiv derefter 4 konkrete handlinger, der forbedrer godkendelsessandsynligheden "
                              "(undersøgelser, udtalelser, designændringer, dokumenter). "
                              "Format: nummer. Handling – Ansvarlig part – Tidslinje\n\n"
                              "### LIVSCYKLUS\n"
                              "Trin 6: Udvid anbefalingerne til næste projektlivscyklusfase "
                              "KUN HVIS den hentede kontekst indeholder tilstrækkelige data om den fase. "
                              "Hvis kilderne ikke dækker senere faser, anfør eksplicit: "
                              "'Utilstrækkelige kildedata til anbefalinger for senere faser.' "
                              "Format: 6. Handling – Ansvarlig part – Tidslinje"),
        "toimenpiteet_vaihe": " Tag hensyn til projektets nuværende fase: {vaihe}.",
        "phase_label":        "Projektfase",
        "viranomainen_ohje":  ("VIGTIGT: Ansøgningen er rettet til myndighed '{auth}'. "
                               "Tilpas indhold, struktur og sprog til myndighedens krav. "
                               "Henvis til myndighedens retningslinjer, formularer og krav."),
    },
    "NO": {
        "intro":        "Utarbeid et utkast til tillatelsessøknad for følgende prosjekt:",
        "rag_intro":    "Nedenfor er relevant dokumentasjon (Fingrid, Tukes, Klima- og miljødepartementet):",
        "kuvaus":       "PROSJEKTBESKRIVELSE",
        "perustelut":   "BEGRUNNELSE OG BEHOV",
        "luvat":        "BESKRIVELSE AV TILLATELSESPROSEDYRER",
        "toimenpiteet": "NESTE STEG",
        "kuvaus_inst":  ("Skriv denne seksjonen i to synlige trinn:\n\n"
                         "### ANALYSER\n"
                         "Identifiser prosjektets nøkkelkarakteristikker og risikofaktorer: type, plassering, størrelse, "
                         "land og relevante myndigheter. Inkluder typiske tekniske parametere.\n\n"
                         "### HENT\n"
                         "Identifiser de mest relevante lovkravene og presedenssaker for denne profilen "
                         "fra RAG-konteksten. Skriv en beskrivelse på 3–4 avsnitt: formål, tekniske detaljer, "
                         "plassering, nettilknytning og miljøpåvirkning. "
                         "Seksjonen må være tilstrekkelig utfyllende for forhåndskonsultasjon."),
        "kuvaus_extra": " Ta hensyn til oppgitt informasjon om plassering og miljøpåvirkning.",
        "perustelut_inst": ("Skriv denne seksjonen i to synlige trinn:\n\n"
                            "### SAMMENLIGN\n"
                            "Sammenlign dette prosjektet med presedenssaker fra RAG-konteksten: hvilke risikoer "
                            "var til stede, hvordan ble de løst, hva gjorde prosjektene vellykkede eller mislykkede.\n\n"
                            "### VURDER\n"
                            "Fastslå hvilke risikoer er mest kritiske for godkjenningssannsynligheten. "
                            "Skriv en begrunnelse på 2–3 avsnitt for hvorfor prosjektet er nødvendig "
                            "(energisystemperspektiv, Finlands klimamål, regionale økonomiske virkninger) "
                            "og navngi den største enkeltrisikoen."),
        "luvat_inst":   ("### HENT — TILLATELSER\n"
                         "Forklar kortfattet (1–2 setninger per tillatelse) hva hver nødvendig tillatelse "
                         "dekker, hvorfor den kreves og hvilken myndighet behandler den. "
                         "Henvis til relevante presedenser eller spesielle krav ved behov."),
        "luvat_extra":  " Henvis spesielt til målmyndighetens {auth} prosesser og krav.",
        "toimenpiteet_first": ("Forhåndskonsultasjon med kommunens byggesaksavdeling + reguleringsgjennomgang — "
                               "Søker / {kunta} byggesaksavdeling — innen 1–2 uker"),
        "toimenpiteet_inst": ("Skriv denne seksjonen i to synlige trinn:\n\n"
                              "### ANBEFAL\n"
                              "Det første trinnet er ALLTID: \"{first}\".\n"
                              "List deretter 4 konkrete tiltak som forbedrer godkjenningssannsynligheten "
                              "(utredninger, uttalelser, designendringer, dokumenter). "
                              "Format: nummer. Tiltak – Ansvarlig part – Tidslinje\n\n"
                              "### LIVSSYKLUS\n"
                              "Trinn 6: Utvid anbefalingene til neste prosjektlivssyklusfase "
                              "KUN HVIS hentet kontekst inneholder tilstrekkelige data om den fasen. "
                              "Hvis kildene ikke dekker senere faser, angi eksplisitt: "
                              "'Utilstrekkelige kildedata for anbefalinger for senere faser.' "
                              "Format: 6. Tiltak – Ansvarlig part – Tidslinje"),
        "toimenpiteet_vaihe": " Ta hensyn til prosjektets nåværende fase: {vaihe}.",
        "phase_label":        "Prosjektfase",
        "viranomainen_ohje":  ("VIKTIG: Søknaden er adressert til myndighet '{auth}'. "
                               "Tilpass innhold, struktur og språk til myndighetens krav. "
                               "Henvis til myndighetens retningslinjer, skjemaer og krav."),
    },
    "PL": {
        "intro":        "Sporządź projekt wniosku o zezwolenie dla następującego projektu:",
        "rag_intro":    "Poniżej znajduje się odpowiednia dokumentacja (Fingrid, Tukes, Ministerstwo Środowiska):",
        "kuvaus":       "OPIS PROJEKTU",
        "perustelut":   "UZASADNIENIE I POTRZEBA",
        "luvat":        "OPIS PROCEDUR ZEZWOLEŃ",
        "toimenpiteet": "NASTĘPNE KROKI",
        "kuvaus_inst":  ("Napisz tę sekcję w dwóch widocznych krokach:\n\n"
                         "### ANALIZA\n"
                         "Zidentyfikuj kluczowe cechy i czynniki ryzyka projektu: typ, lokalizacja, rozmiar, "
                         "kraj i właściwe organy. Uwzględnij typowe parametry techniczne.\n\n"
                         "### POBIERZ\n"
                         "Zidentyfikuj najbardziej istotne wymogi prawne i sprawy precedensowe dla tego profilu "
                         "z kontekstu RAG. Napisz opis w 3–4 akapitach: cel, dane techniczne, "
                         "lokalizacja, przyłączenie do sieci i wpływ na środowisko. "
                         "Sekcja musi być wystarczająco wyczerpująca do wstępnych konsultacji."),
        "kuvaus_extra": " Uwzględnij podane informacje o lokalizacji i oddziaływaniu na środowisko.",
        "perustelut_inst": ("Napisz tę sekcję w dwóch widocznych krokach:\n\n"
                            "### PORÓWNAJ\n"
                            "Porównaj ten projekt ze sprawami precedensowymi z kontekstu RAG: jakie ryzyka "
                            "wystąpiły, jak zostały rozwiązane, co sprawiło, że projekty zakończyły się "
                            "sukcesem lub niepowodzeniem.\n\n"
                            "### OCEŃ\n"
                            "Określ, które ryzyka są najbardziej krytyczne dla prawdopodobieństwa uzyskania zgody. "
                            "Napisz uzasadnienie w 2–3 akapitach, dlaczego projekt jest konieczny "
                            "(perspektywa systemu energetycznego, fińskie cele klimatyczne, "
                            "regionalne skutki gospodarcze) i wskaż największe pojedyncze ryzyko."),
        "luvat_inst":   ("### POBIERZ — ZEZWOLENIA\n"
                         "Krótko wyjaśnij (1–2 zdania na zezwolenie) czego dotyczy każde wymagane zezwolenie, "
                         "dlaczego jest wymagane i który organ je rozpatruje. "
                         "W razie potrzeby odwołaj się do precedensów lub szczególnych wymagań."),
        "luvat_extra":  " Odwołaj się szczególnie do procesów i wymagań organu docelowego {auth}.",
        "toimenpiteet_first": ("Wstępna konsultacja z gminnym wydziałem budowlanym + przegląd planistyczny — "
                               "Wnioskodawca / wydział budowlany {kunta} — w ciągu 1–2 tygodni"),
        "toimenpiteet_inst": ("Napisz tę sekcję w dwóch widocznych krokach:\n\n"
                              "### REKOMENDUJ\n"
                              "Pierwszym krokiem jest ZAWSZE: \"{first}\".\n"
                              "Następnie wymiń 4 konkretne działania poprawiające prawdopodobieństwo uzyskania zgody "
                              "(badania, opinie, zmiany projektowe, dokumenty). "
                              "Format: numer. Działanie – Strona odpowiedzialna – Harmonogram\n\n"
                              "### CYKL ŻYCIA\n"
                              "Krok 6: Rozszerz rekomendacje na następną fazę cyklu życia projektu "
                              "TYLKO JEŚLI pobrany kontekst zawiera wystarczające dane dotyczące tej fazy. "
                              "Jeśli źródła nie obejmują późniejszych faz, napisz explicite: "
                              "'Niewystarczające dane źródłowe dla rekomendacji późniejszych faz.' "
                              "Format: 6. Działanie – Strona odpowiedzialna – Harmonogram"),
        "toimenpiteet_vaihe": " Uwzględnij aktualną fazę projektu: {vaihe}.",
        "phase_label":        "Faza projektu",
        "viranomainen_ohje":  ("WAŻNE: Wniosek jest skierowany do organu '{auth}'. "
                               "Dostosuj treść, strukturę i język do jego wymagań. "
                               "Odwołaj się do wytycznych, formularzy i wymagań tego organu."),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Käännöstaulukot viranomaisille, luvannimille, lakiviitteille ja liitteille
# ─────────────────────────────────────────────────────────────────────────────

_AUTHORITY_TRANS: dict[str, dict[str, str]] = {
    "Lupa- ja valvontavirasto (Luova)":  {"EN": "Licensing and Supervisory Authority (Luova)", "SE": "Tillstånds- och tillsynsverket (Luova)",   "DA": "Licenserings- og tilsynsmyndighed (Luova)",         "NO": "Lisensierings- og tilsynsmyndighet (Luova)",         "PL": "Organ licencyjny i nadzorczy (Luova)"},
    "Luova":                              {"EN": "Luova (Licensing Authority)",                  "SE": "Luova (tillståndsmyndighet)",              "DA": "Luova (licensmyndighed)",                           "NO": "Luova (lisensieringsmyndighet)",                     "PL": "Luova (organ licencyjny)"},
    "Kunta / rakennusvalvonta":           {"EN": "Municipality / Building Control",              "SE": "Kommun / byggnadstillsyn",                 "DA": "Kommune / byggesagsafdeling",                       "NO": "Kommune / byggesaksavdeling",                        "PL": "Gmina / wydział budowlany"},
    "Kunta / hakija":                     {"EN": "Municipality / Applicant",                     "SE": "Kommun / sökande",                         "DA": "Kommune / ansøger",                                 "NO": "Kommune / søker",                                    "PL": "Gmina / wnioskodawca"},
    "Paikallinen pelastuslaitos":         {"EN": "Local Fire and Rescue Service",                "SE": "Lokal räddningstjänst",                    "DA": "Lokal brandvæsen",                                  "NO": "Lokalt brannvesen",                                  "PL": "Lokalna straż pożarna"},
    "Jakeluverkkoyhtiö / Fingrid Oyj":    {"EN": "Distribution network operator / Fingrid Oyj", "SE": "Distributionsnätbolag / Fingrid Oyj",      "DA": "Distributionsnetoperatør / Fingrid Oyj",            "NO": "Distribusjonsnettoperatør / Fingrid Oyj",            "PL": "Operator sieci dystrybucyjnej / Fingrid Oyj"},
    "Jakeluverkkoyhtiö / Fingrid":        {"EN": "Distribution network operator / Fingrid",      "SE": "Distributionsnätbolag / Fingrid",          "DA": "Distributionsnetoperatør / Fingrid",                "NO": "Distribusjonsnettoperatør / Fingrid",                "PL": "Operator sieci dystrybucyjnej / Fingrid"},
    "Jakeluverkkoyhtiö":                  {"EN": "Distribution network operator",                "SE": "Distributionsnätbolag",                    "DA": "Distributionsnetoperatør",                          "NO": "Distribusjonsnettoperatør",                          "PL": "Operator sieci dystrybucyjnej"},
    "Kunta":                              {"EN": "Municipality",                                  "SE": "Kommun",                                   "DA": "Kommune",                                           "NO": "Kommune",                                            "PL": "Gmina"},
    "ELY-keskus / Luova":                 {"EN": "ELY Centre / Luova",                           "SE": "NTM-centralen / Luova",                    "DA": "ELY-center / Luova",                                "NO": "ELY-senter / Luova",                                 "PL": "Centrum ELY / Luova"},
    "ELY-keskus":                         {"EN": "ELY Centre",                                   "SE": "NTM-centralen",                            "DA": "ELY-center",                                        "NO": "ELY-senter",                                         "PL": "Centrum ELY"},
    "Fingrid Oyj / jakelu":               {"EN": "Fingrid Oyj / distribution",                   "SE": "Fingrid Oyj / distribution",               "DA": "Fingrid Oyj / distribution",                        "NO": "Fingrid Oyj / distribusjon",                         "PL": "Fingrid Oyj / dystrybucja"},
    "Fingrid Oyj":                        {"EN": "Fingrid Oyj",                                   "SE": "Fingrid Oyj",                              "DA": "Fingrid Oyj",                                       "NO": "Fingrid Oyj",                                        "PL": "Fingrid Oyj"},
    "Traficom":                           {"EN": "Traficom (Transport and Communications Agency)", "SE": "Traficom",                                "DA": "Traficom",                                          "NO": "Traficom",                                           "PL": "Traficom"},
    "Maanomistajat":                      {"EN": "Landowners",                                    "SE": "Markägare",                               "DA": "Jordejere",                                         "NO": "Grunneiere",                                         "PL": "Właściciele gruntów"},
    "Valtioneuvosto":                     {"EN": "Council of State",                              "SE": "Statsrådet",                               "DA": "Statsrådet",                                        "NO": "Statsrådet",                                         "PL": "Rada Ministrów"},
    "TEM / ELY-keskus":                   {"EN": "Ministry of Economic Affairs / ELY Centre",    "SE": "ANM / NTM-centralen",                      "DA": "Erhvervsministeriet / ELY-center",                  "NO": "Nærings- og fiskeridepartementet / ELY-senter",      "PL": "Ministerstwo Gospodarki / Centrum ELY"},
    "STUK":                               {"EN": "STUK (Radiation and Nuclear Safety Authority)", "SE": "STUK (strålnings- och kärnsäkerhetsmyndigheten)", "DA": "STUK (stralings- og nuklearsikkerhedsmyndighed)", "NO": "STUK (strålings- og kjernesikkerhetsmyndighet)", "PL": "STUK (organ bezpieczeństwa jądrowego i radiacyjnego)"},
    "Puolustusvoimat / PLM":              {"EN": "Finnish Defence Forces / Ministry of Defence",  "SE": "Försvarsmakten / försvarsministeriet",     "DA": "Det finske forsvar / forsvarsministeriet",          "NO": "Det finske forsvaret / forsvarsdepartementet",       "PL": "Fińskie Siły Zbrojne / Ministerstwo Obrony"},
    "Valtio / Metsähallitus":             {"EN": "State / Metsähallitus (Forests and Parks Service)", "SE": "Staten / Forststyrelsen",              "DA": "Staten / Metsähallitus (skov- og parktjeneste)",    "NO": "Staten / Metsähallitus (skog- og parktjeneste)",    "PL": "Państwo / Metsähallitus (służba leśna)"},
    "Luova / kunta":                      {"EN": "Luova / Municipality",                         "SE": "Luova / Kommun",                           "DA": "Luova / Kommune",                                   "NO": "Luova / Kommune",                                    "PL": "Luova / Gmina"},
    "Kunta / maanomistajat":              {"EN": "Municipality / Landowners",                    "SE": "Kommun / markägare",                       "DA": "Kommune / jordejere",                               "NO": "Kommune / grunneiere",                               "PL": "Gmina / właściciele gruntów"},
    "Pelastuslaitos":                     {"EN": "Rescue Services / Fire Department",            "SE": "Räddningstjänsten",                         "DA": "Brandvæsenet",                                      "NO": "Brannvesenet",                                       "PL": "Straż pożarna"},
    "AVI / Luova":                        {"EN": "AVI / Luova (Regional State Administrative Agency)", "SE": "RFV / Luova",                         "DA": "AVI / Luova (regional statsforvaltning)",           "NO": "AVI / Luova (regional statsforvaltning)",            "PL": "AVI / Luova (regionalny urząd administracji)"},
}

_LUPA_TRANS: dict[str, dict[str, str]] = {
    "Ympäristölupa":                                {"EN": "Environmental permit",                              "SE": "Miljötillstånd",                              "DA": "Miljøgodkendelse",                              "NO": "Miljøtillatelse",                              "PL": "Pozwolenie środowiskowe"},
    "Ympäristölupa (tarvitt.)":                     {"EN": "Environmental permit (if required)",               "SE": "Miljötillstånd (vid behov)",                  "DA": "Miljøgodkendelse (om nødvendigt)",               "NO": "Miljøtillatelse (om nødvendig)",                "PL": "Pozwolenie środowiskowe (jeśli wymagane)"},
    "Ympäristölupa (tarvitt. ≥1 ha)":              {"EN": "Environmental permit (if required, ≥1 ha)",        "SE": "Miljötillstånd (vid behov, ≥1 ha)",           "DA": "Miljøgodkendelse (om nødvendigt, ≥1 ha)",       "NO": "Miljøtillatelse (om nødvendig, ≥1 ha)",        "PL": "Pozwolenie środowiskowe (jeśli wymagane, ≥1 ha)"},
    "Ympäristölupa (BESS-komponentti)":             {"EN": "Environmental permit (BESS component)",            "SE": "Miljötillstånd (BESS-komponent)",              "DA": "Miljøgodkendelse (BESS-komponent)",              "NO": "Miljøtillatelse (BESS-komponent)",              "PL": "Pozwolenie środowiskowe (komponent BESS)"},
    "Naapurikuuleminen":                             {"EN": "Neighbour consultation",                           "SE": "Grannehörande",                               "DA": "Nabohøring",                                    "NO": "Nabohøring",                                    "PL": "Konsultacje sąsiedzkie"},
    "Pelastussuunnitelma / lausunto":                {"EN": "Emergency plan / statement",                       "SE": "Räddningsplan / utlåtande",                   "DA": "Redningsplan / udtalelse",                      "NO": "Redningsplan / uttalelse",                      "PL": "Plan ratunkowy / opinia"},
    "Pelastussuunnitelma / lausunto (BESS)":         {"EN": "Emergency plan / statement (BESS)",                "SE": "Räddningsplan / utlåtande (BESS)",            "DA": "Redningsplan / udtalelse (BESS)",                "NO": "Redningsplan / uttalelse (BESS)",                "PL": "Plan ratunkowy / opinia (BESS)"},
    "Verkkoliityntäsopimus":                         {"EN": "Grid connection agreement",                        "SE": "Nätanslutningsavtal",                         "DA": "Nettilslutningsaftale",                         "NO": "Nettilknytningsavtale",                         "PL": "Umowa przyłączeniowa do sieci"},
    "Maa-aineslupa (tarvitt.)":                      {"EN": "Soil extraction permit (if required)",            "SE": "Marktäktstillstånd (vid behov)",               "DA": "Jordudgravningstilladelse (om nødvendigt)",     "NO": "Masseuttakstillatelse (om nødvendig)",          "PL": "Pozwolenie na wydobycie gruntu (jeśli wymagane)"},
    "YVA-menettely (≥10 MW / ≥5 voimalaa)":         {"EN": "EIA procedure (≥10 MW / ≥5 turbines)",            "SE": "MKB-förfarande (≥10 MW / ≥5 verk)",          "DA": "VVM-procedure (≥10 MW / ≥5 møller)",           "NO": "KU-prosess (≥10 MW / ≥5 turbiner)",            "PL": "Procedura OOŚ (≥10 MW / ≥5 turbin)"},
    "YVA-menettely (kynnyksen ylittyessä)":          {"EN": "EIA procedure (when threshold exceeded)",         "SE": "MKB-förfarande (vid tröskelöverskridning)",   "DA": "VVM-procedure (når tærskel overskrides)",       "NO": "KU-prosess (når terskel overskrides)",          "PL": "Procedura OOŚ (gdy próg zostanie przekroczony)"},
    "YVA-menettely (tarvitt.)":                      {"EN": "EIA procedure (if required)",                     "SE": "MKB-förfarande (vid behov)",                  "DA": "VVM-procedure (om nødvendigt)",                 "NO": "KU-prosess (om nødvendig)",                     "PL": "Procedura OOŚ (jeśli wymagana)"},
    "YVA-menettely":                                 {"EN": "EIA procedure",                                    "SE": "MKB-förfarande",                              "DA": "VVM-procedure",                                 "NO": "KU-prosess",                                    "PL": "Procedura OOŚ"},
    "Osayleiskaava tai asemakaava":                  {"EN": "Local master plan or detailed plan",               "SE": "Delgeneralplan eller detaljplan",             "DA": "Lokalplan eller rammeplan",                     "NO": "Kommuneplan eller reguleringsplan",             "PL": "Plan miejscowy lub plan szczegółowy"},
    "Osayleiskaava / asemakaava":                    {"EN": "Local master plan / detailed plan",                "SE": "Delgeneralplan / detaljplan",                 "DA": "Lokalplan / rammeplan",                         "NO": "Kommuneplan / reguleringsplan",                 "PL": "Plan miejscowy / plan szczegółowy"},
    "Lentoestevalolupa":                             {"EN": "Aviation obstacle lighting permit",                "SE": "Luftfartshinderlystillstånd",                 "DA": "Luftfartshindringstillladelse",                 "NO": "Luftfartshindertillatelse",                     "PL": "Pozwolenie na oznakowanie przeszkód lotniczych"},
    "Lentoestevalolupa (tuulivoimala)":              {"EN": "Aviation obstacle lighting permit (wind turbine)", "SE": "Luftfartshinderlystillstånd (vindkraftverk)", "DA": "Luftfartshindringstillladelse (vindmølle)",     "NO": "Luftfartshindertillatelse (vindturbin)",        "PL": "Pozwolenie na oznakowanie (turbina wiatrowa)"},
    "Maanvuokrasopimukset":                          {"EN": "Land lease agreements",                            "SE": "Arrendeavtal",                                "DA": "Jordlejeaftaler",                               "NO": "Jordleieavtaler",                               "PL": "Umowy dzierżawy gruntów"},
    "Maanvuokra / merialueen käyttöoik.":            {"EN": "Land lease / sea area usage right",               "SE": "Arrendeavtal / havsområdesanvändningsrätt",   "DA": "Jordleje / brugsret til havområde",             "NO": "Jordleie / bruksrett til havområde",            "PL": "Dzierżawa gruntu / prawo użytkowania obszaru morskiego"},
    "Vesilupa":                                      {"EN": "Water permit",                                     "SE": "Vattentillstånd",                             "DA": "Vandtilladelse",                                "NO": "Vanntillatelse",                                "PL": "Pozwolenie wodnoprawne"},
    "Vesilupa (jäähdytysvesi)":                      {"EN": "Water permit (cooling water)",                    "SE": "Vattentillstånd (kylvatten)",                 "DA": "Vandtilladelse (kølevand)",                     "NO": "Vanntillatelse (kjølevann)",                    "PL": "Pozwolenie wodnoprawne (woda chłodząca)"},
    "Vesilupa (jäähdytysvesi, tarvitt.)":            {"EN": "Water permit (cooling water, if required)",       "SE": "Vattentillstånd (kylvatten, vid behov)",      "DA": "Vandtilladelse (kølevand, om nødvendigt)",      "NO": "Vanntillatelse (kjølevann, om nødvendig)",     "PL": "Pozwolenie wodnoprawne (woda chłodząca, jeśli wymagane)"},
    "Vesilupa (padotus, rakentaminen)":              {"EN": "Water permit (damming, construction)",            "SE": "Vattentillstånd (dämning, byggande)",          "DA": "Vandtilladelse (opstemning, byggeri)",           "NO": "Vanntillatelse (demning, bygging)",             "PL": "Pozwolenie wodnoprawne (piętrzenie, budowa)"},
    "Alusliikenteen turvallisuuslupa":               {"EN": "Vessel traffic safety permit",                    "SE": "Fartygsfartstillstånd",                        "DA": "Skibsfartstilladelse",                          "NO": "Skipsfartssikkerhetstillatelse",                "PL": "Zezwolenie na bezpieczeństwo ruchu statków"},
    "Puolustusvoimien lausunto":                     {"EN": "Defence Forces statement",                        "SE": "Försvarsmaktens utlåtande",                   "DA": "Forsvarets udtalelse",                          "NO": "Forsvarets uttalelse",                          "PL": "Opinia Sił Zbrojnych"},
    "Suunnittelutarveratkaisu (tarvitt.)":           {"EN": "Planning permit (if required)",                   "SE": "Planeringsbehovsbeslut (vid behov)",           "DA": "Planlægningsbehov (om nødvendigt)",             "NO": "Planbehovsvurdering (om nødvendig)",            "PL": "Decyzja o warunkach zabudowy (jeśli wymagana)"},
    "Maisema- tai kulttuuriympäristölausunto":        {"EN": "Landscape or cultural environment statement",    "SE": "Landskap- eller kulturmiljöutlåtande",         "DA": "Landskabs- eller kulturmiljøudtalelse",         "NO": "Landskap- eller kulturmiljøuttalelse",          "PL": "Opinia krajobrazowa lub środowiska kulturowego"},
    "Periaatepäätös (VN)":                           {"EN": "Decision-in-principle (Council of State)",       "SE": "Principbeslut (statsrådet)",                  "DA": "Principbeslutning (statsrådet)",                "NO": "Prinsippvedtak (statsrådet)",                   "PL": "Decyzja zasadnicza (Rada Ministrów)"},
    "Rakentamislupa":                                {"EN": "Construction permit",                              "SE": "Bygglov",                                     "DA": "Byggetilladelse",                               "NO": "Byggetillatelse",                               "PL": "Pozwolenie na budowę"},
    "Rakentamislupa (tuulivoimala)":                 {"EN": "Construction permit (wind turbine)",               "SE": "Bygglov (vindkraftverk)",                     "DA": "Byggetilladelse (vindmølle)",                   "NO": "Byggetillatelse (vindturbin)",                  "PL": "Pozwolenie na budowę (turbina wiatrowa)"},
    "Rakentamislupa (PV + BESS)":                    {"EN": "Construction permit (PV + BESS)",                  "SE": "Bygglov (PV + BESS)",                         "DA": "Byggetilladelse (PV + BESS)",                   "NO": "Byggetillatelse (PV + BESS)",                   "PL": "Pozwolenie na budowę (PV + BESS)"},
    "Rakentamislupa (ydinlaitos)":                   {"EN": "Construction licence (nuclear facility)",         "SE": "Byggnadstillstånd (kärnkraftverk)",            "DA": "Byggetilladelse (kerneanlæg)",                  "NO": "Byggetillatelse (kjernekraftanlegg)",           "PL": "Pozwolenie na budowę (obiekt jądrowy)"},
    "Käyttölupa":                                    {"EN": "Operating licence",                               "SE": "Drifttillstånd",                               "DA": "Driftstilladelse",                              "NO": "Driftstillatelse",                              "PL": "Zezwolenie na eksploatację"},
    "Käyttölupa (ydinlaitos)":                       {"EN": "Operating licence (nuclear facility)",            "SE": "Drifttillstånd (kärnkraftverk)",               "DA": "Driftstilladelse (kerneanlæg)",                 "NO": "Driftstillatelse (kjernekraftanlegg)",          "PL": "Zezwolenie na eksploatację (obiekt jądrowy)"},
    "Maankäyttösopimus / kaavoitus":                 {"EN": "Land use agreement / zoning",                    "SE": "Markanvändningsavtal / planläggning",          "DA": "Arealforbrugaftale / zoneinddeling",            "NO": "Arealbruksavtale / soneinndeling",              "PL": "Umowa o użytkowaniu gruntu / podział na strefy"},
    "Maankäyttösopimus":                             {"EN": "Land use agreement",                              "SE": "Markanvändningsavtal",                         "DA": "Arealforbrugaftale",                            "NO": "Arealbruksavtale",                              "PL": "Umowa o użytkowaniu gruntu"},
    "Kalastuslaki-ilmoitus":                         {"EN": "Fisheries Act notification",                      "SE": "Fiskelagsanmälan",                             "DA": "Fiskerilovsanmeldelse",                         "NO": "Fiskelovsmelding",                              "PL": "Zgłoszenie na mocy ustawy rybackiej"},
}

_LAW_TRANS: dict[str, dict[str, str]] = {
    "YSL 527/2014":                                         {"EN": "Environmental Protection Act (YSL 527/2014)",                          "SE": "Miljöskyddslagen (YSL 527/2014)",                      "DA": "Miljøbeskyttelsesloven (YSL 527/2014)",                   "NO": "Miljøvernloven (YSL 527/2014)",                    "PL": "Ustawa o ochronie środowiska (YSL 527/2014)"},
    "Rakentamislaki 751/2023 / MRL 132/1999":               {"EN": "Building Act / Land Use and Building Act (751/2023 / 132/1999)",       "SE": "Bygglag / Plan- och bygglag (751/2023 / 132/1999)",    "DA": "Byggelov / Planlægningslov (751/2023 / 132/1999)",        "NO": "Byggelov / Plan- og bygningsloven (751/2023 / 132/1999)", "PL": "Prawo budowlane / Ustawa o zagospodarowaniu przestrzennym (751/2023 / 132/1999)"},
    "Rakentamislaki 751/2023, 44 §":                        {"EN": "Building Act 751/2023, § 44",                                          "SE": "Bygglagen 751/2023, § 44",                             "DA": "Byggeloven 751/2023, § 44",                               "NO": "Byggeloven 751/2023, § 44",                        "PL": "Prawo budowlane 751/2023, § 44"},
    "Pelastuslaki 379/2011, 15 §":                          {"EN": "Rescue Services Act 379/2011, § 15",                                   "SE": "Räddningslagen 379/2011, § 15",                        "DA": "Redningstjenesteloven 379/2011, § 15",                    "NO": "Brannvernloven 379/2011, § 15",                    "PL": "Ustawa o ochronie przeciwpożarowej 379/2011, § 15"},
    "Sähkömarkkinalaki 588/2013":                           {"EN": "Electricity Market Act (588/2013)",                                    "SE": "Elmarknadslagen (588/2013)",                           "DA": "Elmarkedsloven (588/2013)",                               "NO": "Energiloven (588/2013)",                           "PL": "Ustawa o rynku energii elektrycznej (588/2013)"},
    "Maa-aineslaki 555/1981":                               {"EN": "Extractable Land Resources Act (555/1981)",                            "SE": "Marktäktslagen (555/1981)",                            "DA": "Råstofloven (555/1981)",                                  "NO": "Mineralressursloven (555/1981)",                   "PL": "Ustawa o kopalinach pospolitych (555/1981)"},
    "YVA-laki 252/2017":                                    {"EN": "EIA Act (252/2017)",                                                   "SE": "MKB-lagen (252/2017)",                                 "DA": "VVM-loven (252/2017)",                                    "NO": "KU-loven (252/2017)",                              "PL": "Ustawa OOŚ (252/2017)"},
    "YVA-laki 252/2017 (kynnykset ylittyessä)":            {"EN": "EIA Act 252/2017 (when thresholds exceeded)",                          "SE": "MKB-lagen 252/2017 (vid tröskelöverskridning)",        "DA": "VVM-loven 252/2017 (når grænseværdier overskrides)",      "NO": "KU-loven 252/2017 (når terskler overskrides)",    "PL": "Ustawa OOŚ 252/2017 (gdy progi są przekroczone)"},
    "YVA-laki 252/2017 (≥50 ha hankkeet)":                 {"EN": "EIA Act 252/2017 (≥50 ha projects)",                                   "SE": "MKB-lagen 252/2017 (≥50 ha projekt)",                 "DA": "VVM-loven 252/2017 (≥50 ha projekter)",                   "NO": "KU-loven 252/2017 (≥50 ha prosjekter)",           "PL": "Ustawa OOŚ 252/2017 (≥50 ha projekty)"},
    "MRL 132/1999 § 77a":                                   {"EN": "Land Use and Building Act 132/1999, § 77a",                            "SE": "Plan- och bygglagen 132/1999, § 77a",                  "DA": "Planlægningsloven 132/1999, § 77a",                       "NO": "Plan- og bygningsloven 132/1999, § 77a",          "PL": "Ustawa o zagospodarowaniu przestrzennym 132/1999, § 77a"},
    "MRL 132/1999 § 137":                                   {"EN": "Land Use and Building Act 132/1999, § 137",                            "SE": "Plan- och bygglagen 132/1999, § 137",                  "DA": "Planlægningsloven 132/1999, § 137",                       "NO": "Plan- og bygningsloven 132/1999, § 137",          "PL": "Ustawa o zagospodarowaniu przestrzennym 132/1999, § 137"},
    "MRL 197 §":                                            {"EN": "Land Use and Building Act, § 197",                                     "SE": "Plan- och bygglagen, § 197",                           "DA": "Planlægningsloven, § 197",                                "NO": "Plan- og bygningsloven, § 197",                    "PL": "Ustawa o zagospodarowaniu przestrzennym, § 197"},
    "MRL 132/1999 § 91a":                                   {"EN": "Land Use and Building Act 132/1999, § 91a",                            "SE": "Plan- och bygglagen 132/1999, § 91a",                  "DA": "Planlægningsloven 132/1999, § 91a",                       "NO": "Plan- og bygningsloven 132/1999, § 91a",          "PL": "Ustawa o zagospodarowaniu przestrzennym 132/1999, § 91a"},
    "MRL 132/1999 § 9":                                     {"EN": "Land Use and Building Act 132/1999, § 9",                              "SE": "Plan- och bygglagen 132/1999, § 9",                    "DA": "Planlægningsloven 132/1999, § 9",                         "NO": "Plan- og bygningsloven 132/1999, § 9",            "PL": "Ustawa o zagospodarowaniu przestrzennym 132/1999, § 9"},
    "MRL 132/1999":                                         {"EN": "Land Use and Building Act (132/1999)",                                 "SE": "Plan- och bygglagen (132/1999)",                       "DA": "Planlægningsloven (132/1999)",                            "NO": "Plan- og bygningsloven (132/1999)",                "PL": "Ustawa o zagospodarowaniu przestrzennym (132/1999)"},
    "Ilmailulaki 864/2014":                                 {"EN": "Aviation Act (864/2014)",                                              "SE": "Luftfartslagen (864/2014)",                            "DA": "Luftfartsloven (864/2014)",                               "NO": "Luftfartsloven (864/2014)",                        "PL": "Ustawa lotnicza (864/2014)"},
    "Maakaari 540/1995":                                    {"EN": "Code of Real Estate (540/1995)",                                       "SE": "Jordabalken (540/1995)",                               "DA": "Tinglysningsloven (540/1995)",                            "NO": "Eiendomsloven (540/1995)",                         "PL": "Ustawa o nieruchomościach (540/1995)"},
    "Vesilaki 587/2011":                                    {"EN": "Water Act (587/2011)",                                                 "SE": "Vattenlagen (587/2011)",                               "DA": "Vandloven (587/2011)",                                    "NO": "Vannressursloven (587/2011)",                      "PL": "Prawo wodne (587/2011)"},
    "Vesilaki 587/2011 § 3:2":                              {"EN": "Water Act 587/2011, § 3:2",                                            "SE": "Vattenlagen 587/2011, § 3:2",                          "DA": "Vandloven 587/2011, § 3:2",                               "NO": "Vannressursloven 587/2011, § 3:2",                 "PL": "Prawo wodne 587/2011, § 3:2"},
    "Merilaki 674/1994":                                    {"EN": "Maritime Act (674/1994)",                                              "SE": "Sjölagen (674/1994)",                                  "DA": "Søloven (674/1994)",                                      "NO": "Sjøloven (674/1994)",                              "PL": "Kodeks morski (674/1994)"},
    "Merenkulkulaki 1672/2009":                             {"EN": "Maritime Navigation Act (1672/2009)",                                  "SE": "Sjöfartslagen (1672/2009)",                            "DA": "Søfartsloven (1672/2009)",                                "NO": "Navigasjonsloven (1672/2009)",                     "PL": "Ustawa o żegludze morskiej (1672/2009)"},
    "Laki alueiden käytöstä":                               {"EN": "Act on Land Use",                                                      "SE": "Lagen om områdesanvändning",                           "DA": "Lov om arealanvendelse",                                  "NO": "Lov om arealbruk",                                 "PL": "Ustawa o użytkowaniu gruntów"},
    "Rakentamislaki 751/2023 / MRL 132/1999 § 125–126":    {"EN": "Building Act / Land Use and Building Act (751/2023 / 132/1999 §§ 125–126)", "SE": "Bygglag / Plan- och bygglag (751/2023 / 132/1999 §§ 125–126)", "DA": "Byggelov / Planlægningslov (751/2023 / 132/1999 §§ 125–126)", "NO": "Byggelov / Plan- og bygningsloven (751/2023 / 132/1999 §§ 125–126)", "PL": "Prawo budowlane (751/2023 / 132/1999 §§ 125–126)"},
    "Rakentamislaki 751/2023 / MRL 132/1999 § 126":         {"EN": "Building Act / Land Use and Building Act (751/2023 / 132/1999, § 126)", "SE": "Bygglag / Plan- och bygglag (751/2023 / 132/1999, § 126)", "DA": "Byggelov / Planlægningslov (751/2023 / 132/1999, § 126)", "NO": "Byggelov / Plan- og bygningsloven (751/2023 / 132/1999, § 126)", "PL": "Prawo budowlane (751/2023 / 132/1999, § 126)"},
    "Ydinenergialaki 990/1987 § 11":                        {"EN": "Nuclear Energy Act 990/1987, § 11",                                   "SE": "Kärnenergilagen 990/1987, § 11",                        "DA": "Kerneenergieloven 990/1987, § 11",                        "NO": "Atomenergiloven 990/1987, § 11",                   "PL": "Ustawa o energii jądrowej 990/1987, § 11"},
    "YEL 990/1987 § 18":                                    {"EN": "Nuclear Energy Act 990/1987, § 18",                                   "SE": "Kärnenergilagen 990/1987, § 18",                        "DA": "Kerneenergieloven 990/1987, § 18",                        "NO": "Atomenergiloven 990/1987, § 18",                   "PL": "Ustawa o energii jądrowej 990/1987, § 18"},
    "YEL 990/1987 § 20":                                    {"EN": "Nuclear Energy Act 990/1987, § 20",                                   "SE": "Kärnenergilagen 990/1987, § 20",                        "DA": "Kerneenergieloven 990/1987, § 20",                        "NO": "Atomenergiloven 990/1987, § 20",                   "PL": "Ustawa o energii jądrowej 990/1987, § 20"},
    "Kalastuslaki 379/2015":                                {"EN": "Fisheries Act (379/2015)",                                             "SE": "Fiskelagen (379/2015)",                                "DA": "Fiskeriloven (379/2015)",                                 "NO": "Fiskeloven (379/2015)",                            "PL": "Ustawa o rybołówstwie (379/2015)"},
    "Säteilylaki 859/2018":                                 {"EN": "Radiation Act (859/2018)",                                             "SE": "Strålningslagen (859/2018)",                           "DA": "Strålingsloven (859/2018)",                               "NO": "Strålevernloven (859/2018)",                       "PL": "Ustawa prawo atomowe (859/2018)"},
    "Kemikaaliturvallisuuslaki 390/2005":                   {"EN": "Chemicals Safety Act (390/2005)",                                      "SE": "Kemikaliesäkerhetslagen (390/2005)",                   "DA": "Kemikaliesikkerhedsloven (390/2005)",                     "NO": "Kjemikaliesikkerhetsloven (390/2005)",             "PL": "Ustawa o bezpieczeństwie chemicznym (390/2005)"},
    "Kemikaaliturvallisuuslaki 390/2005 (BESS)":           {"EN": "Chemicals Safety Act 390/2005 (BESS)",                                  "SE": "Kemikaliesäkerhetslagen 390/2005 (BESS)",              "DA": "Kemikaliesikkerhedsloven 390/2005 (BESS)",                "NO": "Kjemikaliesikkerhetsloven 390/2005 (BESS)",        "PL": "Ustawa o bezpieczeństwie chemicznym 390/2005 (BESS)"},
    "Luonnonsuojelulaki 9/2023":                            {"EN": "Nature Conservation Act (9/2023)",                                     "SE": "Naturvårdslagen (9/2023)",                             "DA": "Naturbeskyttelsesloven (9/2023)",                         "NO": "Naturmangfoldloven (9/2023)",                      "PL": "Ustawa o ochronie przyrody (9/2023)"},
    "Maantielaki 503/2005 (tiealueet)":                     {"EN": "Highways Act 503/2005 (road areas)",                                   "SE": "Väglagen 503/2005 (vägområden)",                       "DA": "Vejloven 503/2005 (vejarealer)",                          "NO": "Vegloven 503/2005 (vegarealer)",                   "PL": "Ustawa o drogach publicznych 503/2005 (obszary drogowe)"},
    "Patoturvallisuuslaki 494/2009":                        {"EN": "Dam Safety Act (494/2009)",                                            "SE": "Damsäkerhetslagen (494/2009)",                         "DA": "Dæmningssikkerhedsloven (494/2009)",                      "NO": "Damsikkerhetsloven (494/2009)",                    "PL": "Ustawa o bezpieczeństwie budowli piętrzących (494/2009)"},
}

_LIITE_TRANS: dict[str, dict[str, str]] = {
    "Sijaintikartta (M 1:20 000 tai laajempi)":             {"EN": "Location map (scale 1:20,000 or wider)",                        "SE": "Lägeskartta (skala 1:20 000 eller vidare)",              "DA": "Oversigtskort (målestok 1:20.000 eller bredere)",      "NO": "Oversiktskart (målestokk 1:20 000 eller bredere)",    "PL": "Mapa lokalizacyjna (skala 1:20 000 lub szersza)"},
    "Sijaintikartta / projektikartta (M 1:20 000 tai laajempi)": {"EN": "Location map / project map (scale 1:20,000 or wider)",     "SE": "Lägeskartta / projektkarta (skala 1:20 000 eller vidare)", "DA": "Oversigtskort / projektkort (1:20.000 eller bredere)", "NO": "Oversiktskart / prosjektkart (1:20 000 eller bredere)", "PL": "Mapa lokalizacyjna / projektu (1:20 000 lub szersza)"},
    "Maankäyttöselvitys PDF (NCE)":                  {"EN": "Land Use Survey PDF (NCE)",                                    "SE": "Markanvändningsutredning PDF (NCE)",                     "DA": "Arealanvendelsesrapport PDF (NCE)",                    "NO": "Arealbruksutredning PDF (NCE)",                       "PL": "Raport zagospodarowania terenu PDF (NCE)"},
    "Asemapiirustus ja pohjakartta (M 1:500)":              {"EN": "Site plan and base map (1:500)",                               "SE": "Situationsplan och baskarta (1:500)",                    "DA": "Situationsplan og basiskortet (1:500)",                "NO": "Situasjonsplan og basiskart (1:500)",                 "PL": "Plan zagospodarowania i mapa podkładowa (1:500)"},
    "Asemapiirustus ja pohjakartta (M 1:500 tai 1:1000)":   {"EN": "Site plan and base map (1:500 or 1:1000)",                    "SE": "Situationsplan och baskarta (1:500 eller 1:1000)",       "DA": "Situationsplan og basiskortet (1:500 eller 1:1000)",   "NO": "Situasjonsplan og basiskart (1:500 eller 1:1000)",    "PL": "Plan zagospodarowania i mapa (1:500 lub 1:1000)"},
    "Rakennesuunnitelma (akkukontti + perustukset)":         {"EN": "Structural plan (battery container + foundations)",            "SE": "Konstruktionsplan (battericontainer + fundament)",       "DA": "Konstruktionsplan (battericontainer + fundamenter)",   "NO": "Konstruksjonsplan (battericontainer + fundamenter)",  "PL": "Plan konstrukcyjny (kontener bateryjny + fundamenty)"},
    "Paloturvallisuusselvitys (NFPA 855 / EN-standardit)":  {"EN": "Fire safety report (NFPA 855 / EN standards)",                 "SE": "Brandsäkerhetsutredning (NFPA 855 / EN-standarder)",    "DA": "Brandsikkerhedsrapport (NFPA 855 / EN-standarder)",   "NO": "Brannsikkerhetsrapport (NFPA 855 / EN-standarder)",  "PL": "Raport bezpieczeństwa pożarowego (NFPA 855 / EN)"},
    "Sammutusvesien keräyssuunnitelma":                      {"EN": "Fire suppression water collection plan",                       "SE": "Plan för uppsamling av brandsläckningsvatten",          "DA": "Plan for opsamling af brandslukningsmiddel",          "NO": "Plan for oppsamling av brannslukkingsvann",           "PL": "Plan zbierania wody gaśniczej"},
    "Sammutusvesien keräyssuunnitelma (BESS)":               {"EN": "Fire suppression water collection plan (BESS)",                "SE": "Plan för uppsamling av brandsläckningsvatten (BESS)",   "DA": "Opsamlingsplan for brandslukningsmiddel (BESS)",      "NO": "Oppsamlingsplan for brannslukkingsvann (BESS)",       "PL": "Plan zbierania wody gaśniczej (BESS)"},
    "Sammutusvesien keräyssuunnitelma (BESS-komponentti)":   {"EN": "Fire suppression water collection plan (BESS component)",     "SE": "Plan för uppsamling av brandsläckningsvatten (BESS-komponent)", "DA": "Opsamlingsplan (BESS-komponent)",           "NO": "Oppsamlingsplan (BESS-komponent)",                    "PL": "Plan zbierania wody gaśniczej (komponent BESS)"},
    "Ympäristöriskiarvio (pohjavesi, maaperä)":              {"EN": "Environmental risk assessment (groundwater, soil)",            "SE": "Miljöriskbedömning (grundvatten, mark)",                 "DA": "Miljørisikovurdering (grundvand, jordbund)",           "NO": "Miljørisikovurdering (grunnvann, jordsmonn)",         "PL": "Ocena ryzyka środowiskowego (wody gruntowe, gleba)"},
    "Sähköliityntäsuunnitelma (verkkoyhtiön hyväksymä)":     {"EN": "Electrical connection plan (approved by grid operator)",      "SE": "Elanslutningsplan (godkänd av nätbolaget)",             "DA": "Eltilslutningsplan (godkendt af netoperatør)",         "NO": "Strømtilkoblingsplan (godkjent av nettoperatør)",     "PL": "Plan przyłączenia elektrycznego (zatwierdzony przez operatora sieci)"},
    "Meluselvitys (jos lähellä asutusta)":                   {"EN": "Noise study (if near residential areas)",                     "SE": "Bullerutredning (om nära bebyggelse)",                  "DA": "Støjundersøgelse (hvis nær bebyggelse)",               "NO": "Støyutredning (hvis nær boligområde)",                "PL": "Badanie hałasu (jeśli blisko zabudowy)"},
    "Liikenneyhteydet ja huoltotie":                         {"EN": "Traffic connections and maintenance road",                     "SE": "Trafikförbindelser och underhållsväg",                  "DA": "Trafikforbindelser og vedligeholdsvej",                "NO": "Trafikkforbindelser og servicevei",                   "PL": "Połączenia komunikacyjne i droga serwisowa"},
    "Hakijan oikeushenkilön rekisteriote":                   {"EN": "Applicant's legal entity registration extract",               "SE": "Sökandens juridiska enhets registerutdrag",             "DA": "Ansøgerens juridiske enheds registerudskrift",         "NO": "Søkerens juridiske enhets registerutskrift",          "PL": "Odpis z rejestru osoby prawnej wnioskodawcy"},
    "Hakijan rekisteriote":                                  {"EN": "Applicant's registration extract",                             "SE": "Sökandens registerutdrag",                              "DA": "Ansøgerens registerudskrift",                          "NO": "Søkerens registerutskrift",                           "PL": "Odpis z rejestru wnioskodawcy"},
    "Valtakirja (jos asiamies edustaa)":                     {"EN": "Power of attorney (if agent represents)",                     "SE": "Fullmakt (om ombud företräder)",                        "DA": "Fuldmagt (hvis agent repræsenterer)",                  "NO": "Fullmakt (hvis agent representerer)",                 "PL": "Pełnomocnictwo (jeśli przedstawiciel reprezentuje)"},
    "YVA-ohjelma ja YVA-selostus (ELY:n hyväksymä)":        {"EN": "EIA programme and EIA report (ELY Centre approved)",          "SE": "MKB-program och MKB-rapport (NTM-centralen godkänd)",   "DA": "VVM-program og VVM-rapport (godkendt af ELY-center)",  "NO": "KU-program og KU-rapport (godkjent av ELY-senter)",  "PL": "Program OOŚ i raport OOŚ (zatwierdzony przez Centrum ELY)"},
    "YVA-ohjelma ja YVA-selostus":                           {"EN": "EIA programme and EIA report",                                "SE": "MKB-program och MKB-rapport",                           "DA": "VVM-program og VVM-rapport",                           "NO": "KU-program og KU-rapport",                           "PL": "Program OOŚ i raport OOŚ"},
    "YVA-ohjelma ja -selostus":                              {"EN": "EIA programme and report",                                    "SE": "MKB-program och rapport",                               "DA": "VVM-program og rapport",                               "NO": "KU-program og rapport",                              "PL": "Program OOŚ i raport"},
    "YVA-ohjelma ja -selostus (tuulivoiman osalta)":         {"EN": "EIA programme and report (wind power component)",             "SE": "MKB-program och rapport (vindkraftsdelen)",             "DA": "VVM-program og rapport (vindkraftsdelen)",             "NO": "KU-program og rapport (vindkraftdelen)",              "PL": "Program OOŚ i raport (część wiatrowa)"},
    "Meluselvitys (ETSU-R-97 tai IEC 61400-11)":             {"EN": "Noise study (ETSU-R-97 or IEC 61400-11)",                    "SE": "Bullerutredning (ETSU-R-97 eller IEC 61400-11)",        "DA": "Støjundersøgelse (ETSU-R-97 eller IEC 61400-11)",      "NO": "Støyutredning (ETSU-R-97 eller IEC 61400-11)",        "PL": "Badanie hałasu (ETSU-R-97 lub IEC 61400-11)"},
    "Meluselvitys (tuulivoimalakomponentti)":                {"EN": "Noise study (wind turbine component)",                        "SE": "Bullerutredning (vindkraftverkskomponent)",             "DA": "Støjundersøgelse (vindmøllekomponent)",                "NO": "Støyutredning (vindturbinkomponent)",                 "PL": "Badanie hałasu (komponent turbiny wiatrowej)"},
    "Meluselvitys (ilma- ja vedenalainen melu)":             {"EN": "Noise study (airborne and underwater noise)",                 "SE": "Bullerutredning (luftburet och undervattensbuller)",    "DA": "Støjundersøgelse (luftbåren og undervandsstøj)",       "NO": "Støyutredning (luftbåren og undervanns støy)",        "PL": "Badanie hałasu (powietrzny i podwodny)"},
    "Varjostusmallinnusraportti":                            {"EN": "Shadow flicker modelling report",                             "SE": "Skuggningsmodelleringsrapport",                         "DA": "Skyggeblinksmodelleringsrapport",                      "NO": "Skyggeblinksmodelleringsrapport",                     "PL": "Raport z modelowania cieni"},
    "Varjostus- ja näkyvyysanalyysi":                        {"EN": "Shadow flicker and visibility analysis",                      "SE": "Skuggnings- och synlighetsanalys",                      "DA": "Skygge- og synlighedsanalyse",                         "NO": "Skygge- og synlighetsanalyse",                        "PL": "Analiza cieni i widoczności"},
    "Varjostus- ja häikäisyanalyysi (naapurikiinteistöt)":   {"EN": "Shadow and glare analysis (neighbouring properties)",        "SE": "Skuggnings- och bländningsanalys (grannfastigheter)",  "DA": "Skygge- og blændanalyse (naboejendomme)",              "NO": "Skygge- og blendanalyse (naboeiendommer)",            "PL": "Analiza cieni i oślepiania (nieruchomości sąsiednie)"},
    "Linnustoselvitys (pesimä- ja muuttolinnut)":            {"EN": "Bird survey (breeding and migratory birds)",                  "SE": "Fågelinventering (häcknings- och sträckfåglar)",        "DA": "Fuglekortlægning (yngle- og trækfugle)",               "NO": "Fuglekartlegging (hekkende og trekkende fugler)",     "PL": "Inwentaryzacja ptaków (lęgowe i migrujące)"},
    "Lepakoiden lentoaktiviteettiselvitys":                  {"EN": "Bat flight activity survey",                                  "SE": "Fladdermössens flygaktivitetsutredning",                "DA": "Flagermusenes flyveaktivitetsundersøgelse",            "NO": "Flaggermusenes flygeaktivitetskartlegging",            "PL": "Badanie aktywności lotnej nietoperzy"},
    "Linnusto- ja lepakoiden aktiviteettiselvitys":          {"EN": "Bird and bat activity survey",                               "SE": "Fågel- och fladdermösaktivitetsinventering",            "DA": "Fugle- og flagermusaktivitetsundersøgelse",            "NO": "Fugle- og flaggermusaktivitetskartlegging",           "PL": "Inwentaryzacja aktywności ptaków i nietoperzy"},
    "Linnusto- ja lepakkoselvitys merialueella":             {"EN": "Bird and bat survey in sea area",                            "SE": "Fågel- och fladdermusinventering i havsområdet",        "DA": "Fugle- og flagermusundersøgelse i havområdet",         "NO": "Fugle- og flaggermuskartlegging i havområdet",        "PL": "Inwentaryzacja ptaków i nietoperzy na obszarze morskim"},
    "Maisema- ja näkyvyysanalyysi (valokuvasovitteet)":      {"EN": "Landscape and visibility analysis (photomontages)",          "SE": "Landskap- och synlighetsanalys (fotomontage)",          "DA": "Landskabs- og synlighedsanalyse (fotomontager)",       "NO": "Landskaps- og synlighetsanalyse (fotomontasjer)",     "PL": "Analiza krajobrazowa i widoczności (fotomontaże)"},
    "Maisema- ja näkyvyysanalyysi":                          {"EN": "Landscape and visibility analysis",                          "SE": "Landskap- och synlighetsanalys",                        "DA": "Landskabs- og synlighedsanalyse",                      "NO": "Landskaps- og synlighetsanalyse",                     "PL": "Analiza krajobrazowa i widoczności"},
    "Rakennussuunnitelmat (perustukset, tiet, kaapelointi)": {"EN": "Construction plans (foundations, roads, cabling)",           "SE": "Byggplaner (fundament, vägar, kablering)",              "DA": "Bygningsplaner (fundamenter, veje, kabling)",          "NO": "Bygningsplaner (fundamenter, veier, kabling)",        "PL": "Plany budowlane (fundamenty, drogi, okablowanie)"},
    "Rakennussuunnitelmat (pato, voimalaitosrakennus)":      {"EN": "Construction plans (dam, power plant building)",             "SE": "Byggplaner (damm, kraftverksbyggnad)",                  "DA": "Bygningsplaner (dæmning, kraftværksbygning)",          "NO": "Bygningsplaner (dam, kraftverksbygg)",                "PL": "Plany budowlane (zapora, budynek elektrowni)"},
    "Verkkoliityntälaskelma (tehonlaatuanalyysi)":           {"EN": "Grid connection calculation (power quality analysis)",       "SE": "Nätanslutningsberäkning (elkvalitetsanalys)",           "DA": "Nettilslutningsberegning (elkvalitetsanalyse)",        "NO": "Nettilkoblingsberegning (strømkvalitetsanalyse)",     "PL": "Obliczenia przyłączeniowe do sieci (analiza jakości energii)"},
    "Verkkoliityntälaskelma ja muuntajamitoitus":            {"EN": "Grid connection calculation and transformer sizing",         "SE": "Nätanslutningsberäkning och transformatordimensionering", "DA": "Nettilslutningsberegning og transformatordimensionering", "NO": "Nettilkoblingsberegning og transformatordimensjonering", "PL": "Obliczenia przyłączeniowe i dobór transformatora"},
    "Verkkoliityntälaskelma (SMR + BESS yhdistetty)":        {"EN": "Grid connection calculation (SMR + BESS combined)",         "SE": "Nätanslutningsberäkning (SMR + BESS kombinerat)",       "DA": "Nettilslutningsberegning (SMR + BESS kombineret)",     "NO": "Nettilkoblingsberegning (SMR + BESS kombinert)",      "PL": "Obliczenia przyłączeniowe (SMR + BESS łącznie)"},
    "Verkkoliityntälaskelma":                                {"EN": "Grid connection calculation",                               "SE": "Nätanslutningsberäkning",                               "DA": "Nettilslutningsberegning",                             "NO": "Nettilkoblingsberegning",                             "PL": "Obliczenia przyłączeniowe do sieci"},
    "Maanomistaja- ja sopimustiedot":                        {"EN": "Landowner and agreement information",                       "SE": "Markägare- och avtalsuppgifter",                        "DA": "Jordejere og aftaleoplysninger",                       "NO": "Grunneier- og avtaleopplysninger",                    "PL": "Informacje o właścicielach gruntów i umowach"},
    "Maanomistaja- ja vesioikeusasiakirjat":                 {"EN": "Landowner and water rights documents",                      "SE": "Markägare- och vattendokument",                         "DA": "Jordejere og vandrettsdokumenter",                     "NO": "Grunneier- og vannrettsdokumenter",                   "PL": "Dokumenty właścicieli gruntów i praw do wód"},
    "Lentoestekartoitus (Traficom/Finavia)":                 {"EN": "Aviation obstacle survey (Traficom/Finavia)",               "SE": "Luftfartshinderkartläggning (Traficom/Finavia)",        "DA": "Luftfartsforhindringskortlægning (Traficom/Finavia)",  "NO": "Luftfartshinderkartlegging (Traficom/Finavia)",       "PL": "Inwentaryzacja przeszkód lotniczych (Traficom/Finavia)"},
    "Meriekologinen vaikutusarviointi (Natura tarvittaessa)":{"EN": "Marine ecological impact assessment (Natura if required)",  "SE": "Marinekologisk konsekvensutredning (Natura vid behov)", "DA": "Marinøkologisk konsekvensvurdering (Natura om nødvendigt)", "NO": "Marinøkologisk konsekvensutredning (Natura om nødvendig)", "PL": "Morska ocena oddziaływania na środowisko (Natura jeśli wymagana)"},
    "Merikaapelireittiselvitys":                             {"EN": "Submarine cable route survey",                              "SE": "Havskabelruttutredning",                                "DA": "Undersøisk kabelruteundersøgelse",                     "NO": "Undervannkabelruteundersøkelse",                      "PL": "Badanie trasy kabla podmorskiego"},
    "Pohjasedimenttitutkimus (geotekninen)":                 {"EN": "Seabed sediment study (geotechnical)",                      "SE": "Bottensedimentundersökning (geoteknisk)",               "DA": "Havbundssedimentundersøgelse (geoteknisk)",            "NO": "Havbunnsedimentundersøkelse (geoteknisk)",            "PL": "Badanie osadów dna morskiego (geotechniczne)"},
    "Meriliikenteen turvallisuusarviointi":                  {"EN": "Maritime traffic safety assessment",                        "SE": "Säkerhetsbedömning av sjötrafik",                       "DA": "Sikkerhedsvurdering af skibstrafik",                   "NO": "Sikkerhetsvurdering av skipstrafikk",                 "PL": "Ocena bezpieczeństwa ruchu morskiego"},
    "Puolustusvoimien tutkavaikutusarviointi":               {"EN": "Defence Forces radar impact assessment",                    "SE": "Försvarsmaktens radarpåverkansutredning",               "DA": "Forsvarets radarkonsekvensundersøgelse",               "NO": "Forsvarets radarpåvirkningsutredning",                "PL": "Ocena wpływu na radar Sił Zbrojnych"},
    "Paneelijärjestely- ja rakennesuunnitelma":              {"EN": "Panel layout and structural plan",                          "SE": "Panellayout och konstruktionsplan",                     "DA": "Panelplacerings- og konstruktionsplan",                "NO": "Panelplasserings- og konstruksjonsplan",              "PL": "Plan rozmieszczenia paneli i konstrukcji"},
    "Verkkoliityntäsuunnitelma (invertteri, muuntaja)":      {"EN": "Grid connection plan (inverter, transformer)",              "SE": "Nätanslutningsplan (växelriktare, transformator)",       "DA": "Nettilslutningsplan (inverter, transformator)",        "NO": "Nettilkoblingsplan (inverter, transformator)",        "PL": "Plan przyłączenia do sieci (falownik, transformator)"},
    "Maaperä- ja hulevesiselvitys (suuri-alainen asennus)":  {"EN": "Soil and stormwater study (large-scale installation)",     "SE": "Mark- och dagvattenutredning (storskalig installation)", "DA": "Jordbunds- og regnvandsundersøgelse (storskalig)",    "NO": "Grunn- og overvannsstudie (storskala installasjon)",  "PL": "Badanie gleby i wód opadowych (instalacja wielkoskalowa)"},
    "Luontoselvitys (ekologiset yhteydet, mahdollinen Natura)":{"EN": "Nature survey (ecological corridors, possible Natura)",  "SE": "Naturinventering (ekologiska förbindelser, möjlig Natura)", "DA": "Naturundersøgelse (økologiske forbindelser, mulig Natura)", "NO": "Naturkartlegging (økologiske korridorer, mulig Natura)", "PL": "Inwentaryzacja przyrodnicza (korytarze ekologiczne, możliwa Natura)"},
    "Asukasosallistumisen asiakirjat (suunnittelutarveratkaisussa)":{"EN": "Public participation documents (planning permit procedure)", "SE": "Medborgardeltagandedokument (planeringsbehovsbeslut)", "DA": "Borgerdeltakelsesdokumenter (planlægningsbehovsvurdering)", "NO": "Innbyggermedvirkningsdokumenter (planbehovsvurdering)", "PL": "Dokumenty uczestnictwa społecznego (decyzja o warunkach zabudowy)"},
    "Alustava turvallisuusseloste (STUK YVL A.1 mukainen)": {"EN": "Preliminary safety report (per STUK YVL A.1)",             "SE": "Preliminär säkerhetsredogörelse (enl. STUK YVL A.1)",  "DA": "Foreløbig sikkerhedsrapport (STUK YVL A.1)",          "NO": "Foreløpig sikkerhetsrapport (STUK YVL A.1)",          "PL": "Wstępny raport bezpieczeństwa (STUK YVL A.1)"},
    "Ydinmateriaalivalvontasuunnitelma (IAEA SQ-protokolla)":{"EN": "Nuclear materials safeguards plan (IAEA SQ protocol)",     "SE": "Kärnmaterialövervakningsplan (IAEA SQ-protokoll)",      "DA": "Kernematerialekontrolplan (IAEA SQ-protokol)",         "NO": "Kjernematerialkontrollplan (IAEA SQ-protokoll)",      "PL": "Plan kontroli materiałów jądrowych (protokół IAEA SQ)"},
    "Säteilyturvallisuusanalyysi (YVL C.1)":                {"EN": "Radiation safety analysis (YVL C.1)",                      "SE": "Strålsäkerhetsanalys (YVL C.1)",                        "DA": "Strålingsikkerhedsanalyse (YVL C.1)",                  "NO": "Strålesikkerhetsanalyse (YVL C.1)",                   "PL": "Analiza bezpieczeństwa radiacyjnego (YVL C.1)"},
    "Turvallisuussuunnittelun periaatteet (YVL B.1)":        {"EN": "Safety design principles (YVL B.1)",                       "SE": "Säkerhetsdesignprinciper (YVL B.1)",                    "DA": "Sikkerhedsdesignprincipper (YVL B.1)",                 "NO": "Sikkerhetsdesignprinsipper (YVL B.1)",                "PL": "Zasady projektowania bezpieczeństwa (YVL B.1)"},
    "Hätäjärjestelmien ja -menettelyjen kuvaus":             {"EN": "Description of emergency systems and procedures",          "SE": "Beskrivning av nödsystem och -förfaranden",             "DA": "Beskrivelse af nødsystemer og -procedurer",            "NO": "Beskrivelse av nødsystemer og -prosedyrer",           "PL": "Opis systemów i procedur awaryjnych"},
    "Polttoainekierto- ja ydinjätehuoltosuunnitelma":        {"EN": "Fuel cycle and nuclear waste management plan",             "SE": "Bränslecykel- och kärnavfallshanteringsplan",           "DA": "Brændselscyklus- og kernekraftaffaldshåndteringsplan", "NO": "Brenselssyklus- og kjernekraftavfallshåndteringsplan", "PL": "Plan cyklu paliwowego i gospodarki odpadami jądrowymi"},
    "Geotekninen perusselvitys (seismisyys, hydrogeologia)": {"EN": "Geotechnical baseline study (seismicity, hydrogeology)",   "SE": "Geoteknisk grundutredning (seismicitet, hydrogeologi)", "DA": "Geoteknisk basisundersøgelse (seismicitet, hydrogeologi)", "NO": "Geoteknisk basisundersøkelse (seismisitet, hydrogeologi)", "PL": "Geotechniczne badanie podstawowe (sejsmiczność, hydrogeologia)"},
    "Jäähdytysveden saatavuus- ja ympäristöarviointi":       {"EN": "Cooling water availability and environmental assessment",  "SE": "Kylvattentillgång och miljöbedömning",                 "DA": "Kølevandsadgang og miljøvurdering",                    "NO": "Kjølevanntilgang og miljøvurdering",                  "PL": "Dostępność wody chłodzącej i ocena środowiskowa"},
    "Jäähdytysvesitarve- ja ympäristöarviointi":             {"EN": "Cooling water demand and environmental assessment",        "SE": "Kylvattenbehov och miljöbedömning",                    "DA": "Kølevandsbehov og miljøvurdering",                     "NO": "Kjølevannbehov og miljøvurdering",                    "PL": "Zapotrzebowanie na wodę chłodzącą i ocena środowiskowa"},
    "Sosioekonominen vaikutusarviointi":                     {"EN": "Socioeconomic impact assessment",                          "SE": "Socioekonomisk konsekvensutredning",                    "DA": "Socioøkonomisk konsekvensvurdering",                   "NO": "Sosioøkonomisk konsekvensutredning",                  "PL": "Ocena oddziaływania społeczno-ekonomicznego"},
    "Kansainväliset referenssilaitosvertailut (IAEA)":       {"EN": "International reference plant comparisons (IAEA)",         "SE": "Internationella referensanläggningsjämförelser (IAEA)", "DA": "Internationale referenceanslægssammenligninger (IAEA)", "NO": "Internasjonale referanseanleggssammenligninger (IAEA)", "PL": "Porównania z międzynarodowymi instalacjami referencyjnymi (IAEA)"},
    "Hydraulinen mitoitusraportti (virtaama, putouskorkeus)":{"EN": "Hydraulic design report (flow rate, head)",               "SE": "Hydraulisk dimensioneringsrapport (flöde, fallhöjd)",  "DA": "Hydraulisk designrapport (gennemstrømning, faldhøjde)", "NO": "Hydraulisk dimensjoneringsrapport (gjennomstrømning, fallhøyde)", "PL": "Raport projektowania hydraulicznego (przepływ, spadek)"},
    "Geotekninen pato- ja pohjarakenneselvitys":             {"EN": "Geotechnical dam and foundation study",                    "SE": "Geoteknisk dam- och grundläggningsutredning",           "DA": "Geoteknisk dæmnings- og fundamentundersøgelse",        "NO": "Geoteknisk dam- og fundamentundersøkelse",            "PL": "Geotechniczne badanie zapory i fundamentów"},
    "Vesistövaikutusten arviointi (tulva, kuivuus, vedenlaatu)":{"EN": "Watercourse impact assessment (flooding, drought, water quality)", "SE": "Vattendragspåverkansutredning (översvämning, torka, vattenkvalitet)", "DA": "Vandløbspåvirkningsvurdering (oversvømmelse, tørke, vandkvalitet)", "NO": "Vassdragspåvirkningsvurdering (flom, tørke, vannkvalitet)", "PL": "Ocena oddziaływania na cieki wodne (powodzie, susza, jakość wody)"},
    "Ekologinen virtaamaselvitys (kalat, pohjaeläimet)":     {"EN": "Ecological flow study (fish, benthic fauna)",              "SE": "Ekologisk flödesutredning (fisk, bottendjur)",          "DA": "Økologisk strømningsundersøgelse (fisk, bunddyr)",     "NO": "Økologisk gjennomstrømningsstudie (fisk, bunnfauna)",  "PL": "Badanie przepływu ekologicznego (ryby, fauna denna)"},
    "Kalaston vaellusesteiden ja kalateiden suunnitelma":    {"EN": "Fish migration barrier and fish pass plan",                "SE": "Plan för fiskvandringsbarriärer och fiskvägar",         "DA": "Plan for fiskevandringshindringer og fiskepassager",   "NO": "Plan for fiskevandringshindringer og fiskepassasjer",  "PL": "Plan barier migracji ryb i przepławek"},
    "Padon turvallisuussuunnitelma (PATL 494/2009)":         {"EN": "Dam safety plan (Dam Safety Act 494/2009)",                "SE": "Damsäkerhetsplan (Damsäkerhetslagen 494/2009)",         "DA": "Dæmningssikkerhedsplan (Dæmningssikkerhedsloven 494/2009)", "NO": "Damsikkerhetsplan (Damsikkerhetsloven 494/2009)",    "PL": "Plan bezpieczeństwa zapory (ustawa 494/2009)"},
    "Hätätilannesuunnitelma (padotusriskit)":                {"EN": "Emergency plan (dam failure risks)",                       "SE": "Nödlägesplan (dammbrotsrisker)",                        "DA": "Beredskabsplan (dæmningsbrudsrisici)",                 "NO": "Beredskapsplan (dambruddsrisici)",                    "PL": "Plan awaryjny (ryzyko przerwania zapory)"},
    "BESS-paloturvallisuusselvitys (NFPA 855)":              {"EN": "BESS fire safety report (NFPA 855)",                       "SE": "BESS brandsäkerhetsrapport (NFPA 855)",                 "DA": "BESS brandsikkerhedsrapport (NFPA 855)",               "NO": "BESS brannsikkerhetsrapport (NFPA 855)",              "PL": "Raport bezpieczeństwa pożarowego BESS (NFPA 855)"},
    "BESS-paloturvallisuusselvitys (NFPA 855 / EN-standardit)":{"EN": "BESS fire safety report (NFPA 855 / EN standards)",     "SE": "BESS brandsäkerhetsrapport (NFPA 855 / EN-standarder)", "DA": "BESS brandsikkerhedsrapport (NFPA 855 / EN-standarder)", "NO": "BESS brannsikkerhetsrapport (NFPA 855 / EN-standarder)", "PL": "Raport bezpieczeństwa pożarowego BESS (NFPA 855 / EN)"},
    "Integroitu verkkoliityntäsuunnitelma (tuuli + PV + BESS)":{"EN": "Integrated grid connection plan (wind + PV + BESS)",   "SE": "Integrerad nätanslutningsplan (vind + PV + BESS)",      "DA": "Integreret nettilslutningsplan (vind + PV + BESS)",    "NO": "Integrert nettilkoblingsplan (vind + PV + BESS)",     "PL": "Zintegrowany plan przyłączenia do sieci (wiatr + PV + BESS)"},
    "Integroitu energiavarastosuunnitelma (SMR + BESS-mitoitus)":{"EN": "Integrated energy storage plan (SMR + BESS sizing)",  "SE": "Integrerad energilagringsplan (SMR + BESS-dimensionering)", "DA": "Integreret energilagringsplan (SMR + BESS-dimensionering)", "NO": "Integrert energilagringsplan (SMR + BESS-dimensjonering)", "PL": "Zintegrowany plan magazynowania energii (SMR + BESS)"},
    "Energiavarastomitoitusraportti (kapasiteetti, teho, kesto)":{"EN": "Energy storage sizing report (capacity, power, duration)", "SE": "Energilagringsrapport (kapacitet, effekt, varaktighet)", "DA": "Energilagringsrapport (kapacitet, effekt, varighed)",  "NO": "Energilagringsrapport (kapasitet, effekt, varighet)",  "PL": "Raport doboru magazynowania energii (pojemność, moc, czas)"},
    "Hakijan taloudellinen tilanne (tilinpäätös, 2 viimeisintä vuotta)":{"EN": "Applicant's financial status (financial statements, last 2 years)", "SE": "Sökandens ekonomiska ställning (bokslut, 2 senaste år)", "DA": "Ansøgerens finansielle stilling (regnskaber, seneste 2 år)", "NO": "Søkerens finansielle stilling (regnskap, siste 2 år)", "PL": "Sytuacja finansowa wnioskodawcy (sprawozdania, ostatnie 2 lata)"},
    "Projektisuunnitelma (T&K-kuvaus, tavoitteet, metodologia)":{"EN": "Project plan (R&D description, objectives, methodology)", "SE": "Projektplan (FoU-beskrivning, mål, metodologi)",        "DA": "Projektplan (F&U-beskrivning, mål, metodologi)",       "NO": "Prosjektplan (FoU-beskrivning, mål, metodologi)",     "PL": "Plan projektu (opis B+R, cele, metodologia)"},
    "Budjettilaskelmat ja rahoitussuunnitelma":               {"EN": "Budget calculations and financing plan",                  "SE": "Budgetkalkyl och finansieringsplan",                    "DA": "Budgetberegninger og finansieringsplan",               "NO": "Budsjettberegninger og finansieringsplan",            "PL": "Obliczenia budżetowe i plan finansowania"},
    "Tiimikuvaus (ansioluettelot, osaamisprofiilit)":         {"EN": "Team description (CVs, competency profiles)",             "SE": "Teambeskrivning (meritförteckningar, kompetensprofiler)", "DA": "Teambeskrivelse (CV'er, kompetenceprofiler)",         "NO": "Teambeskrivelse (CVer, kompetanseprofiler)",          "PL": "Opis zespołu (CV, profile kompetencji)"},
    "Riskiarviointi ja mitigaatiosuunnitelma":                {"EN": "Risk assessment and mitigation plan",                     "SE": "Riskbedömning och mitigeringsplan",                     "DA": "Risikovurdering og mitigeringsplan",                   "NO": "Risikovurdering og mitigeringsplan",                  "PL": "Ocena ryzyka i plan mitygacji"},
    "Referenssit ja aiempi T&K-toiminta":                    {"EN": "References and previous R&D activities",                  "SE": "Referenser och tidigare FoU-verksamhet",               "DA": "Referencer og tidligere F&U-aktiviteter",             "NO": "Referanser og tidligere FoU-aktiviteter",             "PL": "Referencje i wcześniejsza działalność B+R"},
    "IPR-suunnitelma (immateriaalioikeuksien hallinta)":      {"EN": "IPR plan (intellectual property rights management)",      "SE": "IPR-plan (immaterialrättshantering)",                   "DA": "IPR-plan (intellektuel ejendomsretshåndtering)",       "NO": "IPR-plan (forvaltning av immaterielle rettigheter)",  "PL": "Plan IPR (zarządzanie prawami własności intelektualnej)"},
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


# Hankkeen vaiheen käännökset (FI-avain → muut kielet)
_VAIHE_TRANS: dict[str, dict[str, str]] = {
    "Esiselvitys":  {"EN": "Pre-study",       "SE": "Förundersökning", "DA": "Forundersøgelse", "NO": "Forstudie",      "PL": "Analiza wstępna"},
    "esiselvitys":  {"EN": "Pre-study",       "SE": "Förundersökning", "DA": "Forundersøgelse", "NO": "Forstudie",      "PL": "Analiza wstępna"},
    "Lupavaihe":    {"EN": "Permit phase",     "SE": "Tillståndsfas",   "DA": "Tilladelsesfase", "NO": "Tillatelsefase", "PL": "Faza zezwoleń"},
    "lupavaihe":    {"EN": "Permit phase",     "SE": "Tillståndsfas",   "DA": "Tilladelsesfase", "NO": "Tillatelsefase", "PL": "Faza zezwoleń"},
    "Rakentaminen": {"EN": "Construction",     "SE": "Byggfas",         "DA": "Anlægsfase",      "NO": "Byggefase",      "PL": "Budowa"},
    "rakentaminen": {"EN": "Construction",     "SE": "Byggfas",         "DA": "Anlægsfase",      "NO": "Byggefase",      "PL": "Budowa"},
}

def _t_vaihe(lang: str, vaihe: str) -> str:
    """Käännä hankkeen vaihe -arvo kohdekielelle. Fallback → alkuperäinen arvo."""
    if lang == "FI" or not vaihe:
        return vaihe
    return _VAIHE_TRANS.get(vaihe, {}).get(lang, vaihe)


# ─────────────────────────────────────────────────────────────────────────────
# PDF-käännöstaulukko (UI-tekstit, ei AI-sisältö)
# ─────────────────────────────────────────────────────────────────────────────

_PDF_STRINGS: dict[str, dict[str, str]] = {
    "FI": {
        "sub_title":       "Rakentamislupahakemusluonnos",
        "esiselvitys_sub": ("Esiselvitys- ja ennakkoneuvottelumateriaali — "
                            "Valmisteltu rakennusvalvonnan ennakkoneuvottelua varten"),
        "lupavaihe_sub":   "Lupavaihe — Rakentamislupahakemusluonnos",
        "rakentaminen_sub":"Rakentamisvaihe — Toteutus ja valvonta",
        "disclaimer_h":    "AI-LUONNOS — VAATII ASIANTUNTIJATARKISTUKSEN",
        "disclaimer_b":    ("Tämä raportti on NCE Permit AI:n generoima luonnos. Se vaatii "
                            "pätevyysvaatimukset täyttävän asiantuntijan tarkistuksen ennen "
                            "viranomaiskäyttöä."),
        "nce_speed_note":  ("NCE Permit AI generoi hakemuspohjan muutamassa minuutissa. "
                            "Viranomaisen arviointiviive on erillinen prosessi ja vaihtelee "
                            "hanketyypeittäin (ks. alta)."),
        "arviointiviive_lbl": "Viranomaisen arviointiviive",
        "m_hakija":        "Hakija",       "m_ytunnus":    "Y-tunnus",
        "m_hanketyyppi":   "Hanketyyppi",  "m_teho":       "Teho / kapasiteetti",
        "m_kunta":         "Sijaintikunta","m_kt":         "Kiinteistötunnus",
        "m_maa":           "Maa",
        "m_laadittu":      "Laadittu",     "m_laatinut_lbl": "Laatinut",
        "m_laatinut":      "NCE Permit AI (tekoälyavusteinen)",
        "sec1": "1. Hankkeen kuvaus",             "sec2": "2. Perustelut ja tarve",
        "sec3": "3. Tarvittavat luvat ja viranomaiset", "sec4": "4. Lakiviitteet",
        "sec5": "5. Liiteluettelo",               "sec6": "6. Seuraavat toimenpiteet",
        "sec_standards": "Sovellettavat standardit (EU/kansainväliset)",
        "th_std_code": "Standardi", "th_std_scope": "Soveltamisala",
        "th_std_supervisor": "Valvova viranomainen",
        "liite_standards": "Standardien vaatimustenmukaisuusselvitys",
        "liitteet_note":   ("Seuraavat liitteet on toimitettava hakemuksen yhteydessä. "
                            "Merkitse ☐-ruutuun kun liite on valmis."),
        "lahteet_h":       "Lähteet ja tietolähteet",
        "lahteet_laki_h":  "Säädösperusta",
        "lahteet_rag_h":   "Viranomaislähteet",
        "lahteet_b":       "Seuraavia viranomaisdokumentteja on käytetty luonnoksen valmistelussa:",
        "yhteystiedot_h":  "Hakijan yhteystiedot",
        "yht_hakija":      "Hakija",     "yht_ytunnus":   "Y-tunnus",
        "yht_osoite":      "Osoite",     "yht_lisatietoja": "Lisätietoja",
        "footer":          ("NCE Permit AI (Native Clean Energy)  ·  ncenergy.fi  ·  "
                            "info@ncenergy.fi"
                            "·  AI-luonnos — vaatii asiantuntijatarkistuksen"),
        "th_lupa":  "Lupa / ilmoitus", "th_viran": "Viranomainen", "th_laki": "Lakiperuste",
        "th_nro":   "Nro",  "th_liite": "Liite",  "th_tila": "Tila",
        "liite_toimitettu": "[ ] Toimitettu",
        "toim_nro": "Nro", "toim_toimenpide": "Toimenpide",
        "toim_vastuutaho": "Vastuutaho", "toim_aikataulu": "Aikataulu",
        "hdr_draft": "lupahakemusluonnos", "hdr_right": "ncenergy.fi  |  AI-luonnos",
        "ftr_ai":    "AI-luonnos — vaatii tarkistuksen", "ftr_sivu": "Sivu",
        "esiselvitys_p":   ("Hanke on esiselvitysvaiheessa. Lopulliset tekniset mitoitukset, "
                            "sijaintisuunnitelmat ja ympäristövaikutusten arvioinnit tarkentuvat "
                            "jatkosuunnittelun myötä."),
        "lupavaihe_p":     ("Hanke on lupavaiheessa. Rakentamislupahakemus ja liitteet "
                            "valmistellaan viranomaiselle jätettäväksi."),
        "rakentaminen_p":  ("Hanke on rakentamisvaiheessa. Toteutus etenee hyväksyttyjen lupien "
                            "ja suunnitelmien mukaisesti viranomaisvalvonnassa."),
        "bess_pintaala":   "Laitosalueen arvioitu pinta-ala on 0,4–0,6 ha.",
        "mks_viittaus":    ("Hankealueen maankäyttö on selvitetty NCE:n maankäyttöselvityksessä "
                            "(ks. Liite 0b: Maankäyttöselvitys PDF). Selvitys sisältää kiinteistötiedot, "
                            "kaavatilanteen, suojelualueet sekä pohjavesialuetiedot ja vastaa "
                            "rakentamislain 61 §:n (751/2023) mukaista selvitystä rakennuspaikan "
                            "ominaisuuksista."),
        "kaava_BESS":      ("<b>Kaavatilanne (kriittisin esiselvityskohta):</b> BESS-hankkeen sijoituspaikan "
                            "kaavatilanne on selvitettävä ensimmäisenä. Useimmissa kunnissa akkuenergiavaraston "
                            "sijoittaminen edellyttää asemakaavaa tai suunnittelutarveratkaisua. Kaavatilanne "
                            "vaikuttaa eniten lupaprosessin kokonaiskestoon — rakennusvalvonnan "
                            "ennakkoneuvottelu ensitoimenpiteenä."),
        "kaava_tuuli":     ("<b>Kaavatilanne ja YVA-tarve:</b> Tuulivoimahanke edellyttää osayleiskaavaa "
                            "(MRL 132/1999, 77a §) — se on pakollinen edellytys rakentamisluvalle. "
                            "YVA-menettely (YVA-laki 252/2017) on pakollinen ≥10 MW tai ≥5 voimalan "
                            "hankkeille. YVA-prosessin vaiheet: (1) YVA-ohjelma → ELY-keskuksen lausunto "
                            "→ julkinen kuuleminen; (2) YVA-selostus → ELY-keskuksen perusteltu päätelmä; "
                            "(3) osayleiskaava kulkee yleensä rinnakkain YVA:n kanssa; "
                            "(4) rakentamislupa vasta lainvoimaisen kaavan ja YVA-päätelmän jälkeen. "
                            "Kaava- ja YVA-prosessit kestävät yhteensä 3–6 vuotta."),
        "kaava_SMR":       ("<b>STUK pre-licensing (kriittisin ensimmäinen vaihe):</b> Ydinlaitoshankkeessa "
                            "valtioneuvoston periaatepäätös (ydinenergialaki 990/1987, 11 §) ja STUK:n "
                            "ennakkolupamenettely ovat pakollisia ennen kaikkia muita lupia. STUK:n "
                            "YVL-ohjeiden mukainen turvallisuusseloste käynnistää prosessin. Kaavatilanne "
                            "selvitetään rinnalla, mutta ydinturvallisuusmenettely on hallitseva tekijä."),
        "kaava_aurinkovoima": ("<b>Rakentamislupa ja kaavatilanne:</b> Pienimuotoiselle aurinkopuistolle "
                            "(alle noin 1 ha) voidaan soveltaa rakentamislain kevennettyä menettelyä "
                            "(Rakentamislaki 751/2023, 49 §). YVA-menettely ei koske alle 50 ha hankkeita. "
                            "Kaavatilanne on silti tarkistettava — asemakaava-alueen ulkopuolella voidaan "
                            "tarvita suunnittelutarveratkaisu."),
        "kaava_generic":   ("<b>Kaavatilanne:</b> Hankkeen sijoituspaikan voimassa oleva kaavatilanne on "
                            "tarkistettava rakennusvalvonnan ennakkoneuvottelussa ennen lupahakemuksen "
                            "jättämistä. Kaavatilanne vaikuttaa suoraan lupaprosessin kestoon ja "
                            "vaatimuksiin — rakentaminen edellyttää usein asemakaavaa tai sen muutosta "
                            "taikka suunnittelutarveratkaisua."),
        "nce_info_desc":   ("NCE Permit AI on tekoälypohjainen työkalu energiahankkeiden "
                            "lupaprosessien automatisointiin."),
    },
    "EN": {
        "sub_title":       "Construction Permit Application Draft",
        "esiselvitys_sub": ("Pre-study and Pre-consultation Material — "
                            "Prepared for construction permit pre-consultation"),
        "lupavaihe_sub":   "Permit Phase — Construction Permit Application Draft",
        "rakentaminen_sub":"Construction Phase — Execution and Supervision",
        "disclaimer_h":    "AI DRAFT — REQUIRES EXPERT REVIEW",
        "disclaimer_b":    ("This report is an AI-generated draft requiring review by a qualified "
                            "expert before use with authorities."),
        "nce_speed_note":  ("NCE Permit AI generates the application draft in minutes. "
                            "Authority processing time is a separate process and varies by project type "
                            "(see below)."),
        "arviointiviive_lbl": "Authority processing time",
        "m_hakija":        "Applicant",      "m_ytunnus":    "Business ID",
        "m_hanketyyppi":   "Project Type",   "m_teho":       "Capacity / Power",
        "m_kunta":         "Municipality",   "m_kt":         "Property ID",
        "m_maa":           "Country",
        "m_laadittu":      "Prepared",       "m_laatinut_lbl": "Prepared by",
        "m_laatinut":      "NCE Permit AI (AI-assisted)",
        "sec1": "1. Project Description",          "sec2": "2. Justification and Need",
        "sec3": "3. Required Permits and Authorities", "sec4": "4. Legal References",
        "sec5": "5. Appendix List",                "sec6": "6. Next Steps",
        "sec_standards": "Applicable Standards (EU/International)",
        "th_std_code": "Standard", "th_std_scope": "Scope",
        "th_std_supervisor": "Supervisory Authority",
        "liite_standards": "Standards Compliance Declaration",
        "liitteet_note":   ("The following appendices must be submitted with the application. "
                            "Mark the checkbox when the appendix is ready."),
        "lahteet_h":       "Sources and References",
        "lahteet_laki_h":  "Statutory Basis",
        "lahteet_rag_h":   "Authority Sources",
        "lahteet_b":       "The following official documents were used in preparing this draft:",
        "yhteystiedot_h":  "Applicant Contact Details",
        "yht_hakija":      "Applicant",  "yht_ytunnus":    "Business ID",
        "yht_osoite":      "Address",    "yht_lisatietoja": "Further information",
        "footer":          ("NCE Permit AI (Native Clean Energy)  ·  ncenergy.fi  ·  "
                            "info@ncenergy.fi"
                            "·  AI draft — requires expert review"),
        "th_lupa":  "Permit / Notification", "th_viran": "Authority", "th_laki": "Legal Basis",
        "th_nro":   "No.", "th_liite": "Appendix", "th_tila": "Status",
        "liite_toimitettu": "[ ] Submitted",
        "toim_nro": "No.", "toim_toimenpide": "Action",
        "toim_vastuutaho": "Responsible", "toim_aikataulu": "Timeline",
        "hdr_draft": "permit application draft", "hdr_right": "ncenergy.fi  |  AI draft",
        "ftr_ai":    "AI draft — requires review", "ftr_sivu": "Page",
        "esiselvitys_p":   ("The project is in the pre-study phase. Final technical specifications, "
                            "site plans and environmental impact assessments will be refined "
                            "during further planning."),
        "lupavaihe_p":     ("The project is in the permit phase. The construction permit application "
                            "and attachments are being prepared for submission to the authority."),
        "rakentaminen_p":  ("The project is in the construction phase. Implementation proceeds in "
                            "accordance with approved permits and plans under regulatory supervision."),
        "bess_pintaala":   "The estimated site area is 0.4–0.6 ha.",
        "mks_viittaus":    ("The land use of the project area has been investigated in NCE's "
                            "land use report (see Appendix 0b: Land Use Report PDF). The report includes "
                            "property information, zoning status, protected areas and groundwater area data, "
                            "in accordance with the requirements for site surveys under applicable "
                            "national planning legislation."),
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
        "kaava_aurinkovoima": ("<b>Construction permit and zoning:</b> For a small-scale solar park "
                            "(below approx. 1 ha), the simplified procedure under the Construction Act "
                            "751/2023 (§ 49) may apply. EIA is not required for projects under 50 ha. "
                            "Zoning must still be checked — a planning permit may be needed outside "
                            "detailed plan areas."),
        "kaava_generic":   ("<b>Zoning status:</b> The current zoning status of the project site must be "
                            "verified in a pre-consultation meeting with the building authority before the "
                            "permit application is submitted. Zoning directly affects the duration and "
                            "requirements of the permit process — construction often requires a detailed "
                            "plan, an amendment to one, or a planning permit."),
        "nce_info_desc":   ("NCE Permit AI is an AI-powered tool for automating permit processes "
                            "in energy projects."),
    },
    "SE": {
        "sub_title":       "Bygglovsansökan — utkast",
        "esiselvitys_sub": ("Förundersökning och förkonsultationsmaterial — "
                            "Utarbetat för förkonsultation med byggnadstillsyn"),
        "lupavaihe_sub":   "Tillståndsfas — Utkast till bygglovsansökan",
        "rakentaminen_sub":"Byggfas — Genomförande och tillsyn",
        "disclaimer_h":    "AI-UTKAST — KRÄVER EXPERTGRANSKNING",
        "disclaimer_b":    ("Denna rapport är ett AI-genererat utkast som kräver granskning av en "
                            "kvalificerad expert före användning hos myndigheter."),
        "m_hakija":        "Sökande",          "m_ytunnus":    "Organisationsnummer",
        "m_hanketyyppi":   "Projekttyp",       "m_teho":       "Kapacitet / Effekt",
        "m_kunta":         "Kommun",           "m_kt":         "Fastighetsbeteckning",
        "m_maa":           "Land",
        "m_laadittu":      "Upprättat",        "m_laatinut_lbl": "Upprättat av",
        "m_laatinut":      "NCE Permit AI (AI-assisterat)",
        "sec1": "1. Projektbeskrivning",             "sec2": "2. Motivering och behov",
        "sec3": "3. Nödvändiga tillstånd och myndigheter", "sec4": "4. Laghänvisningar",
        "sec5": "5. Bilagor",                        "sec6": "6. Nästa steg",
        "sec_standards": "Tillämpliga standarder (EU/internationella)",
        "th_std_code": "Standard", "th_std_scope": "Tillämpningsområde",
        "th_std_supervisor": "Tillsynsmyndighet",
        "liite_standards": "Standarder efterlevnadsdeklaration",
        "liitteet_note":   ("Följande bilagor ska lämnas in tillsammans med ansökan. "
                            "Markera rutan när bilagan är klar."),
        "lahteet_h":       "Källor och referenser",
        "lahteet_laki_h":  "Rättslig grund",
        "lahteet_rag_h":   "Myndighetskällor",
        "lahteet_b":       "Följande officiella dokument har använts vid upprättandet av detta utkast:",
        "yhteystiedot_h":  "Sökandens kontaktuppgifter",
        "yht_hakija":      "Sökande",   "yht_ytunnus":    "Organisationsnummer",
        "yht_osoite":      "Adress",    "yht_lisatietoja": "Mer information",
        "footer":          ("NCE Permit AI (Native Clean Energy)  ·  ncenergy.fi  ·  "
                            "info@ncenergy.fi"
                            "·  AI-utkast — kräver expertgranskning"),
        "th_lupa":  "Tillstånd / anmälan", "th_viran": "Myndighet", "th_laki": "Rättslig grund",
        "th_nro":   "Nr", "th_liite": "Bilaga", "th_tila": "Status",
        "liite_toimitettu": "[ ] Inlämnad",
        "toim_nro": "Nr", "toim_toimenpide": "Åtgärd",
        "toim_vastuutaho": "Ansvarig", "toim_aikataulu": "Tidplan",
        "hdr_draft": "bygglovsansökan — utkast", "hdr_right": "ncenergy.fi  |  AI-utkast",
        "ftr_ai":    "AI-utkast — kräver granskning", "ftr_sivu": "Sida",
        "esiselvitys_p":   ("Projektet befinner sig i förundersökningsfasen. Slutliga tekniska "
                            "specifikationer, platsplaner och miljökonsekvensbedömningar preciseras "
                            "under den fortsatta planeringen."),
        "lupavaihe_p":     ("Projektet befinner sig i tillståndsfasen. Bygglovsansökan och bilagor "
                            "förbereds för inlämning till myndigheten."),
        "rakentaminen_p":  ("Projektet befinner sig i byggfasen. Genomförandet sker i enlighet med "
                            "godkända tillstånd och planer under myndighetstillsyn."),
        "bess_pintaala":   "Den uppskattade anläggningsytan är 0,4–0,6 ha.",
        "mks_viittaus":    ("Markanvändningen i projektområdet har utretts i NCE:s "
                            "markanvändningsutredning (se Bilaga 0b: Markanvändningsutredning PDF). "
                            "Utredningen innehåller fastighetsuppgifter, planläggningsstatus, "
                            "skyddsområden och grundvattenuppgifter, i enlighet med krav på "
                            "platsutredning enligt plan- och bygglagen (PBL 2010:900), 10 kap."),
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
        "kaava_aurinkovoima": ("<b>Bygglov och planläggning:</b> Från och med 1.1.2025 ersätter rakentamislupa "
                            "(bygglov) de tidigare typerna bygglov och åtgärdstillstånd (Bygglag 751/2023). "
                            "MKB krävs inte för projekt under 50 ha. Planläggningsstatus måste ändå kontrolleras — "
                            "planeringstillstånd kan behövas utanför detaljplaneområden."),
        "kaava_generic":   ("<b>Planläggningsstatus:</b> Gällande planläggningsstatus för projektplatsen "
                            "måste verifieras i ett förkonsultationsmöte med byggnadstillsynen innan "
                            "tillståndsansökan lämnas in. Planläggning påverkar direkt varaktigheten och "
                            "kraven i tillståndsprocessen — byggande kräver ofta en detaljplan, en ändring "
                            "av en sådan eller ett planeringstillstånd."),
        "nce_info_desc":   ("NCE Permit AI är ett AI-drivet verktyg för att automatisera "
                            "tillståndsprocesser inom energiprojekt."),
    },
    "DA": {
        "sub_title":       "Byggetilladelsesansøgning — udkast",
        "esiselvitys_sub": ("Forundersøgelses- og forhåndskonsultationsmateriale — "
                            "Udarbejdet til forhåndskonsultation med byggesagsafdelingen"),
        "lupavaihe_sub":   "Tilladelsefase — Udkast til byggetilladelsesansøgning",
        "rakentaminen_sub":"Anlægsfase — Udførelse og tilsyn",
        "disclaimer_h":    "AI-UDKAST — KRÆVER EKSPERTGENNEMGANG",
        "disclaimer_b":    ("Denne rapport er et AI-genereret udkast, der kræver gennemgang af en "
                            "kvalificeret ekspert, før det anvendes over for myndigheder."),
        "m_hakija":        "Ansøger",          "m_ytunnus":    "CVR-nummer",
        "m_hanketyyppi":   "Projekttype",      "m_teho":       "Kapacitet / Effekt",
        "m_kunta":         "Kommune",          "m_kt":         "Ejendomsnummer",
        "m_maa":           "Land",
        "m_laadittu":      "Udarbejdet",       "m_laatinut_lbl": "Udarbejdet af",
        "m_laatinut":      "NCE Permit AI (AI-assisteret)",
        "sec1": "1. Projektbeskrivelse",             "sec2": "2. Begrundelse og behov",
        "sec3": "3. Nødvendige tilladelser og myndigheder", "sec4": "4. Lovhenvisninger",
        "sec5": "5. Bilagsliste",                    "sec6": "6. Næste skridt",
        "sec_standards": "Gældende standarder (EU/internationale)",
        "th_std_code": "Standard", "th_std_scope": "Anvendelsesområde",
        "th_std_supervisor": "Tilsynsmyndighed",
        "liite_standards": "Erklæring om standardoverholdelse",
        "liitteet_note":   ("Følgende bilag skal indsendes sammen med ansøgningen. "
                            "Sæt kryds i afkrydsningsfeltet, når bilaget er klar."),
        "lahteet_h":       "Kilder og referencer",
        "lahteet_laki_h":  "Retsgrundlag",
        "lahteet_rag_h":   "Myndighedskilder",
        "lahteet_b":       "Følgende officielle dokumenter er anvendt ved udarbejdelsen af dette udkast:",
        "yhteystiedot_h":  "Ansøgerens kontaktoplysninger",
        "yht_hakija":      "Ansøger",    "yht_ytunnus":    "CVR-nummer",
        "yht_osoite":      "Adresse",    "yht_lisatietoja": "Yderligere oplysninger",
        "footer":          ("NCE Permit AI (Native Clean Energy)  ·  ncenergy.fi  ·  "
                            "info@ncenergy.fi"
                            "·  AI-udkast — kræver ekspertgennemgang"),
        "th_lupa":  "Tilladelse / anmeldelse", "th_viran": "Myndighed", "th_laki": "Retsgrundlag",
        "th_nro":   "Nr.", "th_liite": "Bilag", "th_tila": "Status",
        "liite_toimitettu": "[ ] Indsendt",
        "toim_nro": "Nr.", "toim_toimenpide": "Handling",
        "toim_vastuutaho": "Ansvarlig", "toim_aikataulu": "Tidsplan",
        "hdr_draft": "tilladelsesansøgning — udkast", "hdr_right": "ncenergy.fi  |  AI-udkast",
        "ftr_ai":    "AI-udkast — kræver gennemgang", "ftr_sivu": "Side",
        "esiselvitys_p":   ("Projektet befinder sig i forundersøgelsesfasen. Endelige tekniske "
                            "specifikationer, lokalplaner og miljøkonsekvensvurderinger vil blive "
                            "præciseret under den videre planlægning."),
        "lupavaihe_p":     ("Projektet befinder sig i tilladelsefasen. Byggetilladelsesansøgning og "
                            "bilag forberedes til indgivelse til myndigheden."),
        "rakentaminen_p":  ("Projektet befinder sig i anlægsfasen. Gennemførelsen sker i "
                            "overensstemmelse med godkendte tilladelser og planer under myndighedstilsyn."),
        "bess_pintaala":   "Det anslåede anlægsareal er 0,4–0,6 ha.",
        "mks_viittaus":    ("Arealanvendelsen i projektområdet er undersøgt i NCE's "
                            "arealanvendelsesrapport (se Bilag 0b: Arealanvendelsesrapport PDF). "
                            "Rapporten indeholder ejendomsoplysninger, planlægningsstatus, "
                            "beskyttede områder og grundvandsdata, i overensstemmelse med krav til "
                            "stedundersøgelse i henhold til planlovens bestemmelser."),
        "kaava_BESS":      ("<b>Planlægningsstatus (det vigtigste forundersøgelseselement):</b> "
                            "Planlægningsstatusen for BESS-projektstedet skal fastlægges først. I de fleste "
                            "kommuner kræver placering af et batterienergilagringssystem en lokalplan eller "
                            "en planlægningstilladelse. Planlægningsstatus har størst indflydelse på den "
                            "samlede varighed af tilladelsesprocessen — forhåndskonsultation med "
                            "byggesagsafdelingen er det første trin."),
        "kaava_tuuli":     ("<b>Planlægningsstatus og VVM-krav:</b> Et vindkraftprojekt kræver næsten altid "
                            "en lokaloversigtsplan eller lokalplan (MRL 132/1999, 77a §). VVM-proceduren "
                            "(YVA-laki 252/2017) er obligatorisk for projekter ≥10 MW eller ≥5 møller — "
                            "plan- og VVM-processerne forløber ofte parallelt og tager tilsammen 3–6 år. "
                            "Planlægningsstatus afklares først."),
        "kaava_SMR":       ("<b>STUK forlicensiering (det vigtigste første trin):</b> For et kernekraftanlæg "
                            "er statsrådets principbeslutning (kernenergiloven 990/1987, § 11) og STUKs "
                            "forlicensieringsprocedure obligatoriske inden alle andre tilladelser. STUKs "
                            "sikkerhedsredegørelse i henhold til YVL-retningslinjerne indleder processen. "
                            "Planlægning behandles parallelt, men kernesikkerhedsproceduren er den "
                            "dominerende faktor."),
        "kaava_aurinkovoima": ("<b>Byggetilladelse og planlægning:</b> Fra og med 1.1.2025 erstatter rakentamislupa "
                            "(byggetilladelse) de tidligere typer byggetilladelse og handlingstilladelse (Bygglov 751/2023). "
                            "VVM kræves ikke for projekter under 50 ha. Planlægningsstatus skal dog stadig "
                            "kontrolleres — en planlægningstilladelse kan være nødvendig uden for "
                            "lokalplanområder."),
        "kaava_generic":   ("<b>Planlægningsstatus:</b> Den gældende planlægningsstatus for projektstedet "
                            "skal verificeres på et forhåndskonsultationsmøde med byggesagsafdelingen, "
                            "inden tilladelsesansøgningen indsendes. Planlægning påvirker direkte "
                            "varigheden og kravene i tilladelsesprocessen — byggeri kræver ofte en "
                            "lokalplan, en ændring heraf eller en planlægningstilladelse."),
        "nce_info_desc":   ("NCE Permit AI er et AI-drevet værktøj til automatisering af "
                            "tilladelsesprocesser i energiprojekter."),
    },
    "NO": {
        "sub_title":       "Søknad om byggetillatelse — utkast",
        "esiselvitys_sub": ("Forstudie- og forhåndskonsultasjonsmateriale — "
                            "Utarbeidet til forhåndskonsultasjon med byggesaksavdelingen"),
        "lupavaihe_sub":   "Tillatelsefase — Utkast til byggetillatelsessøknad",
        "rakentaminen_sub":"Byggefase — Gjennomføring og tilsyn",
        "disclaimer_h":    "AI-UTKAST — KREVER EKSPERTGJENNOMGANG",
        "disclaimer_b":    ("Denne rapporten er et AI-generert utkast som krever gjennomgang av en "
                            "kvalifisert ekspert før bruk overfor myndigheter."),
        "m_hakija":        "Søker",             "m_ytunnus":    "Org.nummer",
        "m_hanketyyppi":   "Prosjekttype",      "m_teho":       "Kapasitet / Effekt",
        "m_kunta":         "Kommune",           "m_kt":         "Eiendomsnummer",
        "m_maa":           "Land",
        "m_laadittu":      "Utarbeidet",        "m_laatinut_lbl": "Utarbeidet av",
        "m_laatinut":      "NCE Permit AI (AI-assistert)",
        "sec1": "1. Prosjektbeskrivelse",            "sec2": "2. Begrunnelse og behov",
        "sec3": "3. Nødvendige tillatelser og myndigheter", "sec4": "4. Lovhenvisninger",
        "sec5": "5. Vedleggsliste",                  "sec6": "6. Neste steg",
        "sec_standards": "Gjeldende standarder (EU/internasjonale)",
        "th_std_code": "Standard", "th_std_scope": "Anvendelsesområde",
        "th_std_supervisor": "Tilsynsmyndighet",
        "liite_standards": "Erklæring om standardoverholdelse",
        "liitteet_note":   ("Følgende vedlegg må leveres sammen med søknaden. "
                            "Kryss av i boksen når vedlegget er klart."),
        "lahteet_h":       "Kilder og referanser",
        "lahteet_laki_h":  "Rettsgrunnlag",
        "lahteet_rag_h":   "Myndighetskilder",
        "lahteet_b":       "Følgende offisielle dokumenter er benyttet ved utarbeidelsen av dette utkastet:",
        "yhteystiedot_h":  "Søkerens kontaktopplysninger",
        "yht_hakija":      "Søker",      "yht_ytunnus":    "Org.nummer",
        "yht_osoite":      "Adresse",    "yht_lisatietoja": "Ytterligere informasjon",
        "footer":          ("NCE Permit AI (Native Clean Energy)  ·  ncenergy.fi  ·  "
                            "info@ncenergy.fi"
                            "·  AI-utkast — krever ekspertgjennomgang"),
        "th_lupa":  "Tillatelse / melding", "th_viran": "Myndighet", "th_laki": "Rettsgrunnlag",
        "th_nro":   "Nr.", "th_liite": "Vedlegg", "th_tila": "Status",
        "liite_toimitettu": "[ ] Innlevert",
        "toim_nro": "Nr.", "toim_toimenpide": "Tiltak",
        "toim_vastuutaho": "Ansvarlig", "toim_aikataulu": "Tidsplan",
        "hdr_draft": "tillatelsessøknad — utkast", "hdr_right": "ncenergy.fi  |  AI-utkast",
        "ftr_ai":    "AI-utkast — krever gjennomgang", "ftr_sivu": "Side",
        "esiselvitys_p":   ("Prosjektet er i forstudiefasen. Endelige tekniske spesifikasjoner, "
                            "stedplaner og miljøkonsekvensutredninger vil bli presisert "
                            "under videre planlegging."),
        "lupavaihe_p":     ("Prosjektet er i tillatelsefasen. Byggetillatelsessøknad og vedlegg "
                            "forberedes for innsending til myndighetene."),
        "rakentaminen_p":  ("Prosjektet er i byggefasen. Gjennomføringen skjer i henhold til "
                            "godkjente tillatelser og planer under myndighetstilsyn."),
        "bess_pintaala":   "Det anslåtte anleggsarealet er 0,4–0,6 ha.",
        "mks_viittaus":    ("Arealbruken i prosjektområdet er undersøkt i NCE's "
                            "arealbruksrapport (se Vedlegg 0b: Arealbruksrapport PDF). "
                            "Rapporten inneholder eiendomsopplysninger, reguleringstatus, "
                            "verneområder og grunnvannsdata, i samsvar med krav til stedsanalyse "
                            "i henhold til plan- og bygningsloven (PBL), § 28."),
        "kaava_BESS":      ("<b>Reguleringstatus (viktigste forstudieelement):</b> "
                            "Reguleringsstatusen for BESS-prosjektstedet må fastlegges først. I de fleste "
                            "kommuner krever plassering av et batterienergilagringssystem en reguleringsplan "
                            "eller dispensasjon. Reguleringstatus har størst innvirkning på den totale "
                            "varigheten av tillatelsesprosessen — forhåndskonsultasjon med byggesaksavdelingen "
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
        "kaava_aurinkovoima": ("<b>Byggetillatelse og regulering:</b> Fra og med 1.1.2025 erstatter rakentamislupa "
                            "(byggetillatelse) de tidligere typene byggetillatelse og tiltakstillatelse (Bygglov 751/2023). "
                            "KU kreves ikke for prosjekter under 50 ha. Reguleringstatus må likevel "
                            "sjekkes — dispensasjon kan være nødvendig utenfor reguleringsplanområder."),
        "kaava_generic":   ("<b>Reguleringstatus:</b> Gjeldende reguleringstatus for prosjektstedet "
                            "må verifiseres på et forhåndskonsultasjonsmøte med byggesaksavdelingen "
                            "før tillatelsessøknaden sendes inn. Regulering påvirker direkte varigheten "
                            "og kravene i tillatelsesprosessen — bygging krever ofte en reguleringsplan, "
                            "en endring av denne eller dispensasjon."),
        "nce_info_desc":   ("NCE Permit AI er et AI-drevet verktøy for automatisering av "
                            "tillatelsesprosesser i energiprosjekter."),
    },
    "PL": {
        "sub_title":       "Wniosek o pozwolenie na budowę — szkic",
        "esiselvitys_sub": ("Materiał z analizy wstępnej i konsultacji wstępnych — "
                            "Przygotowany do wstępnej konsultacji z wydziałem budowlanym"),
        "lupavaihe_sub":   "Faza zezwoleń — Szkic wniosku o pozwolenie na budowę",
        "rakentaminen_sub":"Faza budowy — Realizacja i nadzór",
        "disclaimer_h":    "SZKIC AI — WYMAGA PRZEGLĄDU EKSPERTA",
        "disclaimer_b":    ("Ten raport jest wersją roboczą wygenerowaną przez AI i wymaga przeglądu "
                            "przez wykwalifikowanego eksperta przed użyciem w organach."),
        "m_hakija":        "Wnioskodawca",      "m_ytunnus":    "NIP/KRS",
        "m_hanketyyppi":   "Typ projektu",      "m_teho":       "Moc / pojemność",
        "m_kunta":         "Gmina",             "m_kt":         "Numer nieruchomości",
        "m_maa":           "Kraj",
        "m_laadittu":      "Sporządzono",       "m_laatinut_lbl": "Sporządzone przez",
        "m_laatinut":      "NCE Permit AI (wspomagane przez AI)",
        "sec1": "1. Opis projektu",                  "sec2": "2. Uzasadnienie i potrzeba",
        "sec3": "3. Wymagane zezwolenia i organy",   "sec4": "4. Podstawy prawne",
        "sec5": "5. Lista załączników",              "sec6": "6. Następne kroki",
        "sec_standards": "Obowiązujące normy (UE/międzynarodowe)",
        "th_std_code": "Norma", "th_std_scope": "Zakres stosowania",
        "th_std_supervisor": "Organ nadzorczy",
        "liite_standards": "Deklaracja zgodności z normami",
        "liitteet_note":   ("Następujące załączniki muszą zostać złożone wraz z wnioskiem. "
                            "Zaznacz pole wyboru, gdy załącznik jest gotowy."),
        "lahteet_h":       "Źródła i odniesienia",
        "lahteet_laki_h":  "Podstawa prawna",
        "lahteet_rag_h":   "Źródła urzędowe",
        "lahteet_b":       "Przy sporządzaniu niniejszego szkicu wykorzystano następujące oficjalne dokumenty:",
        "yhteystiedot_h":  "Dane kontaktowe wnioskodawcy",
        "yht_hakija":      "Wnioskodawca", "yht_ytunnus":    "NIP/KRS",
        "yht_osoite":      "Adres",        "yht_lisatietoja": "Dodatkowe informacje",
        "footer":          ("NCE Permit AI (Native Clean Energy)  ·  ncenergy.fi  ·  "
                            "info@ncenergy.fi"
                            "·  Szkic AI — wymaga przeglądu eksperta"),
        "th_lupa":  "Zezwolenie / zgłoszenie", "th_viran": "Organ", "th_laki": "Podstawa prawna",
        "th_nro":   "Nr", "th_liite": "Załącznik", "th_tila": "Status",
        "liite_toimitettu": "[ ] Złożony",
        "toim_nro": "Nr", "toim_toimenpide": "Działanie",
        "toim_vastuutaho": "Odpowiedzialny", "toim_aikataulu": "Harmonogram",
        "hdr_draft": "wniosek o zezwolenie — szkic", "hdr_right": "ncenergy.fi  |  Szkic AI",
        "ftr_ai":    "Szkic AI — wymaga przeglądu", "ftr_sivu": "Strona",
        "esiselvitys_p":   ("Projekt jest w fazie analizy wstępnej. Ostateczne specyfikacje techniczne, "
                            "plany lokalizacyjne i oceny oddziaływania na środowisko zostaną doprecyzowane "
                            "w trakcie dalszego planowania."),
        "lupavaihe_p":     ("Projekt jest w fazie zezwoleń. Wniosek o pozwolenie na budowę i załączniki "
                            "są przygotowywane do złożenia organowi."),
        "rakentaminen_p":  ("Projekt jest w fazie budowy. Realizacja przebiega zgodnie z zatwierdzonymi "
                            "zezwoleniami i planami pod nadzorem organów."),
        "bess_pintaala":   "Szacunkowa powierzchnia instalacji wynosi 0,4–0,6 ha.",
        "mks_viittaus":    ("Zagospodarowanie terenu obszaru projektu zostało zbadane w raporcie NCE "
                            "dotyczącym zagospodarowania terenu (zob. Załącznik 0b: Raport PDF). Raport "
                            "zawiera informacje o nieruchomości, status planistyczny, obszary chronione "
                            "i dane o wodach gruntowych, zgodnie z wymogami dotyczącymi analizy "
                            "lokalizacji na podstawie ustawy o planowaniu i zagospodarowaniu "
                            "przestrzennym (Dz.U. 2023 poz. 977)."),
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
        "nce_info_desc":   ("NCE Permit AI to narzędzie oparte na sztucznej inteligencji do automatyzacji "
                            "procesów uzyskiwania zezwoleń dla projektów energetycznych."),
    },
    "DE": {
        "sub_title":       "Genehmigungsantragsentwurf",
        "esiselvitys_sub": ("Voruntersuchungs- und Vorbesprechungsmaterial — "
                            "Vorbereitet für die Vorbesprechung mit der Baugenehmigungsbehörde"),
        "lupavaihe_sub":   "Genehmigungsphase — Genehmigungsantragsentwurf",
        "rakentaminen_sub":"Bauphase — Durchführung und Überwachung",
        "disclaimer_h":    "KI-ENTWURF — EXPERTENPRÜFUNG ERFORDERLICH",
        "disclaimer_b":    ("Dieser Bericht ist ein KI-generierter Entwurf, der vor der Verwendung bei "
                            "Behörden von einem qualifizierten Experten geprüft werden muss."),
        "nce_speed_note":  ("NCE Permit AI erstellt eine Antragsvorlage in wenigen Minuten. "
                            "Die Bearbeitungszeit der Behörde ist ein separater Prozess und variiert "
                            "je nach Projekttyp (siehe unten)."),
        "arviointiviive_lbl": "Bearbeitungszeit der Behörde",
        "m_hakija":        "Antragsteller",   "m_ytunnus":    "Handelsreg.-Nr.",
        "m_hanketyyppi":   "Projekttyp",      "m_teho":       "Leistung / Kapazität",
        "m_kunta":         "Standortgemeinde","m_kt":         "Grundstücksnummer",
        "m_maa":           "Land",
        "m_laadittu":      "Erstellt",        "m_laatinut_lbl": "Erstellt von",
        "m_laatinut":      "NCE Permit AI (KI-gestützt)",
        "sec1": "1. Projektbeschreibung",            "sec2": "2. Begründung und Bedarf",
        "sec3": "3. Erforderliche Genehmigungen",    "sec4": "4. Rechtsgrundlagen",
        "sec5": "5. Anlagenverzeichnis",             "sec6": "6. Nächste Schritte",
        "sec_standards": "Geltende Normen (EU/international)",
        "th_std_code": "Norm", "th_std_scope": "Anwendungsbereich",
        "th_std_supervisor": "Aufsichtsbehörde",
        "liite_standards": "Normkonformitätserklärung",
        "liitteet_note":   ("Die folgenden Anlagen müssen zusammen mit dem Antrag eingereicht werden. "
                            "Setzen Sie ein Häkchen, wenn die Anlage fertig ist."),
        "lahteet_h":       "Quellen und Referenzen",
        "lahteet_laki_h":  "Rechtsgrundlage",
        "lahteet_rag_h":   "Behördliche Quellen",
        "lahteet_b":       "Bei der Erstellung dieses Entwurfs wurden folgende offizielle Dokumente verwendet:",
        "yhteystiedot_h":  "Kontaktdaten des Antragstellers",
        "yht_hakija":      "Antragsteller",  "yht_ytunnus":    "Handelsreg.-Nr.",
        "yht_osoite":      "Adresse",        "yht_lisatietoja": "Weitere Informationen",
        "footer":          ("NCE Permit AI (Native Clean Energy)  ·  ncenergy.fi  ·  "
                            "info@ncenergy.fi  ·  KI-Entwurf — Expertenprüfung erforderlich"),
        "th_lupa":  "Genehmigung / Meldung", "th_viran": "Behörde", "th_laki": "Rechtsgrundlage",
        "th_nro":   "Nr", "th_liite": "Anlage", "th_tila": "Status",
        "liite_toimitettu": "[ ] Eingereicht",
        "toim_nro": "Nr", "toim_toimenpide": "Maßnahme",
        "toim_vastuutaho": "Verantwortlich", "toim_aikataulu": "Zeitplan",
        "hdr_draft": "Genehmigungsantrag — Entwurf", "hdr_right": "ncenergy.fi  |  KI-Entwurf",
        "ftr_ai":    "KI-Entwurf — Prüfung erforderlich", "ftr_sivu": "Seite",
        "esiselvitys_p":   ("Das Projekt befindet sich in der Voruntersuchungsphase. Endgültige technische "
                            "Spezifikationen, Standortpläne und Umweltverträglichkeitsprüfungen werden im "
                            "Laufe der weiteren Planung konkretisiert."),
        "lupavaihe_p":     ("Das Projekt befindet sich in der Genehmigungsphase. Der Bauantrag und die "
                            "Anlagen werden zur Einreichung bei der Behörde vorbereitet."),
        "rakentaminen_p":  ("Das Projekt befindet sich in der Bauphase. Die Ausführung erfolgt gemäß den "
                            "genehmigten Genehmigungen und Plänen unter behördlicher Aufsicht."),
        "bess_pintaala":   "Die geschätzte Installationsfläche beträgt 0,4–0,6 ha.",
        "kaava_BESS":      ("<b>Bebauungsplanstatus (wichtigster Aspekt der Voruntersuchung):</b> "
                            "Der Bebauungsplanstatus des BESS-Projektgeländes muss zuerst geklärt werden. "
                            "In den meisten Gemeinden erfordert die Aufstellung eines Batteriespeichers "
                            "einen Bebauungsplan oder eine Baugenehmigung. Der Planungsstatus hat den "
                            "größten Einfluss auf die Gesamtdauer des Genehmigungsverfahrens."),
        "kaava_tuuli":     ("<b>Bebauungsplanstatus und UVP-Pflicht:</b> Windparkprojekte erfordern fast "
                            "immer einen Bebauungsplan. Das UVP-Verfahren ist für Projekte ab 50 MW "
                            "oder ab 50 m Nabenhöhe obligatorisch. Plan- und UVP-Verfahren laufen "
                            "oft parallel und dauern zusammen 3–6 Jahre."),
        "kaava_SMR":       ("<b>Vorläufige Genehmigung (wichtigster erster Schritt):</b> Für eine "
                            "Kernkraftanlage ist ein Grundsatzbeschluss der Regierung und das "
                            "vorläufige Genehmigungsverfahren der Atomsicherheitsbehörde vor allen "
                            "anderen Genehmigungen obligatorisch."),
        "kaava_aurinkovoima": ("<b>Baugenehmigung oder Bauanzeige — und Planung:</b> Für kleine "
                            "Solaranlagen (unter ca. 1 ha) ist oft eine Bauanzeige statt einer "
                            "vollständigen Baugenehmigung ausreichend. UVP ist für Projekte unter "
                            "50 ha nicht erforderlich."),
        "kaava_generic":   ("<b>Bebauungsplanstatus:</b> Der geltende Bebauungsplanstatus des "
                            "Projektgeländes muss in einer Beratung mit der Baubehörde vor der "
                            "Einreichung des Genehmigungsantrags überprüft werden."),
        "nce_info_desc":   ("NCE Permit AI ist ein KI-gestütztes Werkzeug zur Automatisierung von "
                            "Genehmigungsverfahren für Energieprojekte."),
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


_ISO_STANDARDS: dict[str, list[tuple[str, str]]] = {
    "BESS": [
        ("IEC 62933-1/2/5", "Battery energy storage systems — general, unit parameters, safety"),
        ("IEC 62619",       "Secondary cells and batteries — safety requirements for use in industrial applications"),
        ("IEC 61508",       "Functional safety of E/E/PE safety-related systems"),
        ("NFPA 855",        "Standard for the Installation of Stationary Energy Storage Systems"),
        ("ISO 14001",       "Environmental management systems — Requirements with guidance"),
    ],
    "tuulivoima_maa": [
        ("IEC 61400-1",   "Wind turbines — Part 1: Design load requirements"),
        ("IEC 61400-2",   "Wind turbines — Part 2: Small wind turbines"),
        ("IEC 61400-21-1","Wind turbines — Part 21-1: Power quality measurements"),
        ("ISO 9001",      "Quality management systems — Requirements"),
        ("ISO 14001",     "Environmental management systems — Requirements with guidance"),
    ],
    "tuulivoima_meri": [
        ("IEC 61400-3-1", "Wind turbines — Design requirements for fixed offshore wind turbines"),
        ("IEC 61400-3-2", "Wind turbines — Design requirements for floating offshore wind turbines"),
        ("IEC 61892-1",   "Mobile and fixed offshore units — Electrical installations"),
        ("DNV-ST-0126",   "Support structures for wind turbines"),
        ("ISO 9001",      "Quality management systems — Requirements"),
        ("ISO 14001",     "Environmental management systems — Requirements with guidance"),
    ],
    "aurinkovoima": [
        ("IEC 61215-1/2", "Terrestrial PV modules — Design qualification and type approval"),
        ("IEC 61730-1/2", "Photovoltaic (PV) module safety qualification"),
        ("IEC 62548",     "Photovoltaic (PV) arrays — Design requirements"),
        ("IEC 62109-1/2", "Safety for power converters for use in PV power systems"),
        ("ISO 14001",     "Environmental management systems — Requirements with guidance"),
    ],
    "SMR": [
        ("IAEA SSR-2/1",  "Safety of Nuclear Power Plants: Design (Specific Safety Requirements)"),
        ("IAEA SSG-52",   "Design of the Reactor Core for Nuclear Power Plants (SMR applicable)"),
        ("ISO 19443",     "Quality management systems — Specific requirements for nuclear sector (ITNS)"),
        ("IEC 61513",     "Nuclear power plants — I&C systems important to safety — general requirements"),
        ("IEC 60880",     "Nuclear power plants — Software for computers performing safety functions"),
    ],
    "smr_bess": [
        ("IAEA SSR-2/1",  "Safety of Nuclear Power Plants: Design"),
        ("IEC 62933-1/2/5","Battery energy storage systems — general, parameters, safety"),
        ("IEC 62619",     "Secondary cells and batteries — safety requirements"),
        ("ISO 19443",     "Quality management systems — nuclear sector (ITNS)"),
        ("IEC 61513",     "I&C systems important to safety — general requirements"),
        ("NFPA 855",      "Standard for Installation of Stationary Energy Storage Systems"),
    ],
    "vesivoima": [
        ("IEC 60041",  "Field acceptance tests to determine the hydraulic performance of turbines/pumps"),
        ("IEC 61116",  "Electromechanical equipment guide for small hydroelectric installations"),
        ("ISO 9001",   "Quality management systems — Requirements"),
        ("ISO 14001",  "Environmental management systems — Requirements with guidance"),
    ],
    "_generic": [
        ("ISO 9001",   "Quality management systems — Requirements"),
        ("ISO 14001",  "Environmental management systems — Requirements with guidance"),
        ("IEC 60364",  "Low-voltage electrical installations"),
    ],
}

_HANKE_STD_KEY: dict[str, str] = {
    "BESS":           "BESS",
    "tuulivoima_maa": "tuulivoima_maa",
    "tuulivoima_meri":"tuulivoima_meri",
    "aurinkovoima":   "aurinkovoima",
    "SMR":            "SMR",
    "smr_bess":       "smr_bess",
    "vesivoima":      "vesivoima",
}

_NATIONAL_SUPERVISORS: dict[str, dict[str, str]] = {
    "FI": {
        "BESS":           "Tukes (kemikaalit/sähkö), Pelastuslaitos (paloturvallisuus)",
        "tuulivoima_maa": "ELY-keskus / Luova (YVA), Tukes (sähköturvallisuus)",
        "tuulivoima_meri":"ELY-keskus / Luova (YVA), Traficom, Tukes",
        "aurinkovoima":   "Tukes (sähköturvallisuus), Kunta (rakennusvalvonta)",
        "SMR":            "STUK (ydinturvallisuus, YVL-ohjeet), TEM (periaatepäätös)",
        "smr_bess":       "STUK (ydinturvallisuus), Tukes (BESS-komponentti)",
        "vesivoima":      "AVI (vesilupa), ELY-keskus, Tukes",
        "_generic":       "Tukes (turvallisuus), Kunta (rakennusvalvonta)",
    },
    "SE": {
        "BESS":           "MSB (Myndigheten för samhällsskydd och beredskap), Ei",
        "tuulivoima_maa": "Länsstyrelsen, Energimyndigheten, Naturvårdsverket",
        "tuulivoima_meri":"Länsstyrelsen, Energimyndigheten, Transportstyrelsen",
        "aurinkovoima":   "Ei (Energimarknadsinspektionen), Boverket",
        "SMR":            "Strålsäkerhetsmyndigheten (SSM)",
        "smr_bess":       "Strålsäkerhetsmyndigheten (SSM), MSB",
        "vesivoima":      "Kammarkollegiet, Länsstyrelsen, Naturvårdsverket",
        "_generic":       "Boverket, Länsstyrelsen",
    },
    "DA": {
        "BESS":           "Sikkerhedsstyrelsen (Sik), Energistyrelsen",
        "tuulivoima_maa": "Energistyrelsen, Miljøstyrelsen, Erhvervsstyrelsen",
        "tuulivoima_meri":"Energistyrelsen, Søfartsstyrelsen, Miljøstyrelsen",
        "aurinkovoima":   "Sikkerhedsstyrelsen, Energistyrelsen",
        "SMR":            "Sundhedsstyrelsen (Statens Institut for Strålebeskyttelse)",
        "smr_bess":       "Sundhedsstyrelsen, Sikkerhedsstyrelsen",
        "vesivoima":      "Miljøstyrelsen, Energistyrelsen",
        "_generic":       "Sikkerhedsstyrelsen, Erhvervsstyrelsen",
    },
    "NO": {
        "BESS":           "DSB (Direktoratet for samfunnssikkerhet og beredskap), NVE",
        "tuulivoima_maa": "NVE, Statsforvalteren, Miljødirektoratet",
        "tuulivoima_meri":"NVE, Sjøfartsdirektoratet, Miljødirektoratet",
        "aurinkovoima":   "NVE, DSB",
        "SMR":            "DSA (Direktoratet for strålevern og atomsikkerhet), NVE",
        "smr_bess":       "DSA (strålevern), DSB (BESS-komponentti), NVE",
        "vesivoima":      "NVE (konsesjon), Miljødirektoratet",
        "_generic":       "DSB, NVE, Statsforvalteren",
    },
    "PL": {
        "BESS":           "UDT (Urząd Dozoru Technicznego), URE",
        "tuulivoima_maa": "URE (koncesja), RDOŚ (OOŚ), GDOŚ",
        "tuulivoima_meri":"URE, GDOŚ, Urząd Morski",
        "aurinkovoima":   "URE, UDT, GUNB",
        "SMR":            "PAA (Państwowa Agencja Atomistyki)",
        "smr_bess":       "PAA (bezpieczeństwo jądrowe), UDT (komponent BESS)",
        "vesivoima":      "PGW Wody Polskie, URE",
        "_generic":       "UDT, URE, GUNB",
    },
    "EE": {
        "BESS":           "Päästeamet (fire safety), Konkurentsiamet (license), Elering/Elektrilevi (grid)",
        "tuulivoima_maa": "Keskkonnaamet (KMH/EIA), Kaitseministeerium (radar), Konkurentsiamet",
        "tuulivoima_meri":"Majandus- ja Kommunikatsiooniministeerium (ühisluba), Keskkonnaamet, Transpordiamet",
        "offshore_wind":  "Majandus- ja Kommunikatsiooniministeerium (ühisluba), Keskkonnaamet, Elering AS",
        "aurinkovoima":   "Konkurentsiamet (license), Elektrilevi (grid), Keskkonnaamet",
        "SMR":            "Terviseamet (radiation, no nuclear law yet), Riigikogu (parliamentary decision)",
        "smr_ee":         "Terviseamet (radiation, no nuclear law yet), Riigikogu (parliamentary decision)",
        "datakeskus":     "Päästeamet (fire), Konkurentsiamet, Elering/Elektrilevi",
        "teollisuus":     "Keskkonnaamet (environmental permit), Päästeamet, Kohaliku omavalitsuse",
        "asuinrakennus":  "Kohaliku omavalitsuse (municipality), Terviseamet (if radiation sources)",
        "liikerakennus":  "Kohaliku omavalitsuse (municipality), Päästeamet",
        "maatalous":      "Põllumajandusamet (Agricultural Board), Keskkonnaamet",
        "hybridi":        "Keskkonnaamet (KMH), Päästeamet (BESS), Konkurentsiamet, Elering AS",
        "_generic":       "Keskkonnaamet, Kohaliku omavalitsuse, Konkurentsiamet",
    },
}


_BESS_MARKET_DATA: dict[str, dict] = {
    "FI": {"index": 110, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "SE": {"index": 145, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "DA": {"index": 160, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "NO": {"index": 130, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "PL": {"index": 775, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "EE": {"index": 120, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index (estimate)", "date": "Q1/2026"},
}


def _s(lang: str, key: str) -> str:
    """Hae käännetty merkkijono PDF-layoutille. Fallback → FI."""
    d = _PDF_STRINGS.get(lang) or _PDF_STRINGS["FI"]
    return d.get(key) or _PDF_STRINGS["FI"].get(key, key)


def _generate_sections(
    inp: ApplicationInput,
    rag_context: str,
    prec_chunks: Optional[list] = None,
    prec_sources: Optional[list] = None,
) -> dict[str, str]:
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
        _vaihe_translated = _t_vaihe(lang, inp.hankkeen_vaihe)
        _phase_label = ph.get("phase_label", "Hankkeen vaihe")
        vaihe_lisatieto = f"\n{_phase_label}: {_vaihe_translated}"
    viranomainen_lisatieto = ""
    if inp.kohdeviranomainen:
        viranomainen_lisatieto = f"\nKohdeviranomainen: {inp.kohdeviranomainen}"

    viranomainen_ohje = ""
    if inp.kohdeviranomainen:
        viranomainen_ohje = "\n\n" + ph["viranomainen_ohje"].format(auth=inp.kohdeviranomainen)

    kap_lisatieto = ""
    if inp.kapasiteetti_mwh and inp.kapasiteetti_mwh > 0:
        kap_lisatieto = f"\nKapasiteetti: {inp.kapasiteetti_mwh} MWh"

    ifc_block = ""
    if inp.ifc_floor_area or inp.ifc_building_height or inp.ifc_materials or inp.ifc_compliance_flags:
        parts = ["\n[IFC-malli — esitäytetyt rakennustiedot]"]
        if inp.ifc_floor_area:
            parts.append(f"  Kerrosala: {inp.ifc_floor_area:.0f} m²")
        if inp.ifc_building_height:
            parts.append(f"  Rakennuksen korkeus: {inp.ifc_building_height:.1f} m")
        if inp.ifc_storeys:
            parts.append(f"  Kerroksia: {inp.ifc_storeys}")
        if inp.ifc_fire_rating:
            parts.append(f"  Palosuojausluokka (seinät): {inp.ifc_fire_rating}")
        if inp.ifc_materials:
            parts.append(f"  Materiaalit: {inp.ifc_materials}")
        if inp.ifc_compliance_flags:
            parts.append("  Vaatimushavainnot:")
            for flag in inp.ifc_compliance_flags.splitlines():
                if flag.strip():
                    parts.append(f"    - {flag.strip()}")
        parts.append("  Käytä yllä olevia tietoja kuvaus- ja perustelut-osioissa.")
        ifc_block = "\n".join(parts)

    first_action  = ph["toimenpiteet_first"].format(kunta=inp.kunta)
    kuvaus_inst   = ph["kuvaus_inst"] + (ph["kuvaus_extra"] if inp.sijainti_ymparistovaikutukset else "")
    if cfg.get("kuvaus_extra_inst"):
        kuvaus_inst += " " + cfg["kuvaus_extra_inst"]
    luvat_inst    = ph["luvat_inst"] + (ph["luvat_extra"].format(auth=inp.kohdeviranomainen) if inp.kohdeviranomainen else "")
    if cfg.get("luvat_extra_inst"):
        luvat_inst += " " + cfg["luvat_extra_inst"]
    perustelut_inst = ph["perustelut_inst"] + (cfg.get("perustelut_extra_inst", ""))
    toim_inst     = (ph["toimenpiteet_inst"].format(first=first_action)
                     + (ph["toimenpiteet_vaihe"].format(vaihe=_t_vaihe(lang, inp.hankkeen_vaihe)) if inp.hankkeen_vaihe else ""))

    _std_key   = _HANKE_STD_KEY.get(inp.hanketyyppi, "_generic")
    _standards = _ISO_STANDARDS.get(_std_key, _ISO_STANDARDS["_generic"])
    _sup_dict  = _NATIONAL_SUPERVISORS.get(country, _NATIONAL_SUPERVISORS["FI"])
    _supervisor = _sup_dict.get(inp.hanketyyppi) or _sup_dict.get("_generic", "–")
    _std_lines = "\n".join(f"  - {code}: {scope}" for code, scope in _standards)
    standards_block = (
        f"\nSovellettavat EU/kansainväliset standardit ({inp.hanketyyppi}):\n{_std_lines}"
        f"\nKansallinen valvontaviranomainen ({country}): {_supervisor}"
    )

    bess_market_block = ""
    if inp.hanketyyppi == "BESS":
        _md = _BESS_MARKET_DATA.get(country, _BESS_MARKET_DATA["FI"])
        bess_market_block = (
            f"\nEuroopan BESS-reservimarkkinat {country}-indeksi: "
            f"{_md['index']} {_md['unit']} 2h-akustolle "
            f"({_md['date']}, lähde: {_md['source']}). "
            f"Mainitse tämä perustelut-osiossa."
        )

    critical_block = ""
    if inp.hanketyyppi in _CRITICAL_HANKE_TYPES:
        _crit_tmpl = _CRITICAL_EXTRA.get(lang, _CRITICAL_EXTRA["FI"])
        critical_block = "\n\n" + _crit_tmpl.format(hanketyyppi=inp.hanketyyppi)

    context_extra_block = ""
    if cfg.get("context_extra"):
        context_extra_block = "\n\n" + cfg["context_extra"]

    _vaihe_key = (inp.hankkeen_vaihe or "esiselvitys").lower()
    # Phase-specific context_extra (additive on top of base context_extra)
    _phase_extra = cfg.get("context_extra_phases", {}).get(_vaihe_key, "")
    if _phase_extra:
        context_extra_block += "\n\n" + _phase_extra
    _phase_instr = _PHASE_INSTRUCTIONS.get(
        _vaihe_key,
        _PHASE_INSTRUCTIONS.get("esiselvitys", "")
    )
    phase_block = f"\n\n{_phase_instr}" if _phase_instr else ""
    if inp.hanketyyppi == "datakeskus" and inp.teho_mw:
        _kokon = round(float(inp.teho_mw) * 1.3, 1)
        context_extra_block += (
            f"\n\nIT-TEHOTIEDOT TÄHÄN RAPORTTIIN: IT-kuorma {inp.teho_mw} MW, "
            f"arvioitu kokonaiskulutus ~{_kokon} MW (PUE 1.3). "
            f"Käytä AINA näitä lukuja raportissa — älä käytä muita lukuja."
        )

    _prec_block = ""
    if prec_chunks:
        _prec_lines = "\n\n---\n\n".join(
            f"[{src}]\n{chunk}"
            for chunk, src in zip(prec_chunks, prec_sources or ["Viranomainen"] * len(prec_chunks))
        )
        _prec_block = f"\n\n---ENNAKKOTAPAUKSET---\n{_prec_lines}\n---ENNAKKOTAPAUKSET_LOPPU---"

    prompt = f"""{lang_prefix}{country_prefix}{ph["intro"]}

Hanketyyppi: {inp.hanketyyppi} ({cfg['nimi_fi']})
Kiinteistötunnus: {_clean_kt(inp.kiinteistotunnus)}
Teho: {inp.teho_mw} MW{kap_lisatieto}
Kunta: {inp.kunta}
Hakija: {inp.hakija}{sijainti_lisatieto}{vaihe_lisatieto}{viranomainen_lisatieto}{ifc_block}
Päivämäärä: {now}{viranomainen_ohje}{standards_block}{bess_market_block}{critical_block}{context_extra_block}{phase_block}

{ph["rag_intro"]}
{rag_context}{_prec_block}

{write_instr}

## {ph["kuvaus"]}
{kuvaus_inst}

## {ph["perustelut"]}
{perustelut_inst}

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
    raw = unicodedata.normalize("NFC", resp.content[0].text)
    logger.warning("[DEBUG sections] stop_reason=%s tokens=%s raw_len=%d raw_start=%r",
                   resp.stop_reason, resp.usage, len(raw), raw[:120])
    # Write raw to /tmp for debug endpoint
    try:
        with open("/tmp/debug_raw_claude.txt", "w", encoding="utf-8") as _f:
            _f.write(f"stop_reason={resp.stop_reason}\ntokens={resp.usage}\n\n{raw}")
    except Exception:
        pass

    # Parsitaan osiot käyttämällä kielen mukaisia otsikoita
    h = [ph["kuvaus"], ph["perustelut"], ph["luvat"], ph["toimenpiteet"]]

    _HEADER_RE_CACHE: dict[str, re.Pattern] = {}

    def _header_pattern(hl: str) -> re.Pattern:
        if hl not in _HEADER_RE_CACHE:
            # Matches: ## [N. ]HEADER, # [N. ]HEADER, **HEADER**, HEADER:
            esc = re.escape(hl)
            _HEADER_RE_CACHE[hl] = re.compile(
                r'(?:#{1,3}\s*(?:\d+[\.\)]\s*)?|\*\*\s*|^)' + esc + r'\s*(?:\*\*)?[:\s]',
                re.IGNORECASE | re.MULTILINE
            )
        return _HEADER_RE_CACHE[hl]

    def _extract(text: str, header: str, next_headers: list[str]) -> str:
        hl = header.lower()
        m = _header_pattern(hl).search(text.lower())
        if not m:
            logger.warning("[DEBUG _extract] header NOT found: %r", hl)
            return ""
        start = text.find("\n", m.start()) + 1
        end   = len(text)
        for nh in next_headers:
            m2 = _header_pattern(nh.lower()).search(text.lower(), start)
            if m2 and m2.start() < end:
                end = m2.start()
        return text[start:end].strip()

    result = {
        "kuvaus":        _extract(raw, h[0], h[1:]),
        "perustelut":    _extract(raw, h[1], h[2:]),
        "luvat_teksti":  _extract(raw, h[2], h[3:]),
        "toimenpiteet":  _extract(raw, h[3], []),
    }
    logger.warning("[DEBUG sections] lengths: %s", {k: len(v) for k, v in result.items()})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PDF-generointi
# ─────────────────────────────────────────────────────────────────────────────

def _st() -> dict:
    """Paragraphityylit."""
    return {
        "title":    ParagraphStyle("at", fontSize=20, textColor=C_NAVY,
                                   fontName=PDF_FONT_BOLD, spaceAfter=3, leading=24),
        "sub":      ParagraphStyle("as", fontSize=10, textColor=C_RED,
                                   fontName=PDF_FONT_BOLD, spaceAfter=4),
        "meta":     ParagraphStyle("am", fontSize=8.5, textColor=C_GRAY,
                                   leading=13, spaceAfter=2),
        "h2":       ParagraphStyle("ah2", fontSize=11, textColor=C_NAVY,
                                   fontName=PDF_FONT_BOLD, spaceBefore=14,
                                   spaceAfter=0, leading=15, keepWithNext=1),
        "h3":       ParagraphStyle("ah3", fontSize=9.5, textColor=C_NAVY,
                                   fontName=PDF_FONT_BOLD, spaceBefore=8,
                                   spaceAfter=0, leading=13, keepWithNext=1),
        "body":     ParagraphStyle("ab", fontSize=9, leading=14, spaceAfter=5),
        "small":    ParagraphStyle("asm", fontSize=7.5, textColor=C_GRAY,
                                   leading=11, spaceAfter=2),
        "warn":     ParagraphStyle("aw", fontSize=8, textColor=C_WARN,
                                   fontName=PDF_FONT_BOLD, alignment=TA_CENTER,
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
                       fontName=PDF_FONT_BOLD, alignment=TA_CENTER, leading=12)
    )]]
    tbl = Table(row, colWidths=[16.5 * cm])
    tbl.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 1.5, C_WARN_BD),
        ("BACKGROUND", (0, 0), (-1, -1), C_WARN_BG),
        ("PADDING",    (0, 0), (-1, -1), 10),
    ]))
    return tbl


def _luvat_table(hanketyyppi: str, st: dict, lang: str = "FI", country: str = "FI") -> Table:
    """Lupa-taulukko hanketyyppikohtaisesti, maakohtaisilla ylikirjoituksilla."""
    luvat_rows = (
        _COUNTRY_LUVAT.get(country, {}).get(hanketyyppi)
        or _HANKE_CFG[hanketyyppi]["luvat"]
    )
    _th  = ParagraphStyle("th", fontSize=8.5, fontName=PDF_FONT_BOLD)
    rows = [[
        Paragraph(_s(lang, "th_lupa"),  _th),
        Paragraph(_s(lang, "th_viran"), _th),
        Paragraph(_s(lang, "th_laki"),  _th),
    ]]
    for lupa, viranomainen, laki in luvat_rows:
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


def _liitteet_table(hanketyyppi: str, lang: str = "FI", country: str = "FI") -> Table:
    """Liiteluettelo checkboxeilla, maakohtaisella ylikirjoituksella."""
    country_liitteet = _COUNTRY_LIITTEET.get(country, {}).get(hanketyyppi)
    liitteet = country_liitteet if country_liitteet else _HANKE_CFG[hanketyyppi]["liitteet"]
    _th2 = ParagraphStyle("th2", fontSize=8.5, fontName=PDF_FONT_BOLD)
    rows = [[
        Paragraph(_s(lang, "th_nro"),   _th2),
        Paragraph(_s(lang, "th_liite"), _th2),
        Paragraph(_s(lang, "th_tila"),  ParagraphStyle("th2c", fontSize=8.5, fontName=PDF_FONT_BOLD,
                                                        alignment=TA_CENTER)),
    ]]
    for i, liite in enumerate(liitteet, start=1):
        rows.append([
            Paragraph(str(i), ParagraphStyle("tn", fontSize=8.5, alignment=TA_CENTER)),
            Paragraph(_t_liite(lang, liite), ParagraphStyle("tl", fontSize=8.5, leading=12)),
            Paragraph(_s(lang, "liite_toimitettu"),
                      ParagraphStyle("tc", fontSize=7.5, textColor=C_GRAY, alignment=TA_CENTER)),
        ])
    # Standardien vaatimustenmukaisuusselvitys (aina viimeisenä liitteenä)
    rows.append([
        Paragraph(str(len(liitteet) + 1), ParagraphStyle("tn", fontSize=8.5, alignment=TA_CENTER)),
        Paragraph(_s(lang, "liite_standards"), ParagraphStyle("tl", fontSize=8.5, leading=12)),
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


def _standards_table(hanketyyppi: str, country: str, lang: str, st: dict) -> Table:
    """ISO/IEC-standarditaulukko: Standardi | Soveltamisala | Valvova viranomainen."""
    std_key    = _HANKE_STD_KEY.get(hanketyyppi, "_generic")
    standards  = _ISO_STANDARDS.get(std_key, _ISO_STANDARDS["_generic"])
    sup_country = _NATIONAL_SUPERVISORS.get(country, _NATIONAL_SUPERVISORS["FI"])
    supervisor = sup_country.get(hanketyyppi) or sup_country.get("_generic", "–")

    _th  = ParagraphStyle("stth",  fontSize=8.5, fontName=PDF_FONT_BOLD)
    _td  = ParagraphStyle("sttd",  fontSize=8.0, leading=11)
    _tds = ParagraphStyle("sttds", fontSize=7.5, leading=11, textColor=C_GRAY)
    rows = [[
        Paragraph(_s(lang, "th_std_code"),       _th),
        Paragraph(_s(lang, "th_std_scope"),      _th),
        Paragraph(_s(lang, "th_std_supervisor"), _th),
    ]]
    for i, (code, scope) in enumerate(standards):
        rows.append([
            Paragraph(f"<b>{code}</b>", _td),
            Paragraph(scope,            _tds),
            Paragraph(supervisor if i == 0 else "", _tds),
        ])

    col_w = [3.5*cm, 10.5*cm, 5.0*cm]
    tbl   = Table(rows, colWidths=col_w, repeatRows=1)
    _nosplit_pairs = [("NOSPLIT", (0, i), (-1, i + 1)) for i in range(len(rows) - 1)]
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_DGRAY),
        ("PADDING",       (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("SPAN",          (2, 1), (2, len(standards))),
        ("VALIGN",        (2, 1), (2, len(standards)), "MIDDLE"),
        *_nosplit_pairs,
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
    th_style = ParagraphStyle("md_th", fontSize=8, fontName=PDF_FONT_BOLD,
                               textColor=C_WHITE)
    td_style = ParagraphStyle("md_td", fontSize=8, fontName=PDF_FONT, leading=11)
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


_HUOM_PREFIXES = tuple(lbl.strip() for lbl in _HUOM_LABEL.values())


def _para_text(text: str, st: dict) -> list:
    """Muunna AI:n tuottama teksti Paragraph-listaksi (kappalejaot \\n\\n)."""
    text = _fix_fi_diacritics(text)
    text = _latin1_safe(text)
    _ai_h2 = ParagraphStyle("ai_h2", fontSize=10.5, fontName=PDF_FONT_BOLD,
                             textColor=C_NAVY, spaceBefore=10, spaceAfter=3,
                             leading=14, keepWithNext=1)
    _ai_h3 = ParagraphStyle("ai_h3", fontSize=9.5, fontName=PDF_FONT_BOLD,
                             textColor=C_BLUE, spaceBefore=6, spaceAfter=3,
                             leading=13, keepWithNext=1)
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
        # Otsikot: ### ja ##
        if para.startswith("### "):
            items.append(Paragraph(para[4:], _ai_h3))
        elif para.startswith("## "):
            items.append(Paragraph(para[3:], _ai_h2))
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
            clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', para)
            is_huom = any(para.startswith(pfx) for pfx in _HUOM_PREFIXES)
            if is_huom and items:
                # [Huom]-kappale pysyy edellisen elementin kanssa samalla sivulla
                p = Paragraph(clean, st["body"])
                prev = items.pop()
                items.append(KeepTogether([prev, p]))
            else:
                # spaceBefore=0 ensimmäisessä kappaleessa välttää otsikon ja tekstin välisen aukon
                if not items:
                    _st = ParagraphStyle("body_first", parent=st["body"], spaceBefore=0)
                    items.append(Paragraph(clean, _st))
                else:
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
    th_s = ParagraphStyle("tp_th", fontSize=8, fontName=PDF_FONT_BOLD,
                          textColor=C_WHITE, leading=11)
    th_c = ParagraphStyle("tp_thc", fontSize=8, fontName=PDF_FONT_BOLD,
                          textColor=C_WHITE, leading=11, alignment=1)
    td_s = ParagraphStyle("tp_td",  fontSize=8, fontName=PDF_FONT,      leading=12)
    td_c = ParagraphStyle("tp_tdc", fontSize=8, fontName=PDF_FONT_BOLD, leading=12, alignment=1)
    # colWidths sum to 16.6 cm = A4 content width (21 - 2×2.2)
    tbl_data = [[Paragraph(header[0], th_c),
                 Paragraph(header[1], th_s),
                 Paragraph(header[2], th_s),
                 Paragraph(header[3], th_s)]]
    for row in rows:
        tbl_data.append([
            Paragraph(str(row[0]), td_c),
            Paragraph(str(row[1]), td_s),
            Paragraph(str(row[2]), td_s),
            Paragraph(str(row[3]), td_s),
        ])
    tbl = Table(tbl_data, colWidths=[0.9*cm, 7.2*cm, 4.5*cm, 4.0*cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), C_NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DGRAY),
        ("TOPPADDING",     (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 7),
        ("LEFTPADDING",    (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("NOSPLIT",        (0, 0), (-1, 1)),   # header stays with first data row
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
            self.setFont(PDF_FONT, 6.5)
            self.setFillColor(C_GRAY)
            _lang  = getattr(inp, "lang", "FI")
            _draft = _s(_lang, "hdr_draft")
            _kunta = _latin1_safe(inp.kunta or "")
            _kt    = _latin1_safe(_clean_kt(inp.kiinteistotunnus or ""))
            _ht    = _latin1_safe(inp.hanketyyppi or "")
            # Ylätunniste
            self.line(m, page_h - 1.55*cm, page_w - m, page_h - 1.55*cm)
            self.drawString(m, page_h - 1.2*cm,
                f"{_ht} {_draft}  |  {_kunta}  |  {now}")
            self.drawRightString(page_w - m, page_h - 1.2*cm, _s(_lang, "hdr_right"))
            # Alatunniste
            self.line(m, 1.45*cm, page_w - m, 1.45*cm)
            self.drawString(m, 0.9*cm,
                f"{_ht} {_draft}  |  {_kt}  |  {_kunta}")
            self.drawRightString(page_w - m, 0.9*cm,
                f"{now}  |  {_s(_lang, 'ftr_ai')}  |  {_s(_lang, 'ftr_sivu')} {page_num} / {total}")
            self.restoreState()

    return _NumberedCanvas


def generate_pdf(
    inp: ApplicationInput,
    sections: dict,
    sources: list[dict],
    warning_flag: bool = False,
    prec_chunks: Optional[list] = None,
    prec_sources: Optional[list] = None,
) -> bytes:
    """Rakenna PDF ja palauta bytes."""
    prec_chunks  = prec_chunks  or []
    prec_sources = prec_sources or []
    # Hard cap: enintään 3 "Asiantuntijatarkistus suositellaan" VAIN sisältöosioissa
    _SEC_SEP = "\x00||SEC||\x00"
    _str_keys = [k for k, v in sections.items()
                 if isinstance(v, str) and k in _CONTENT_SECTION_KEYS]
    if _str_keys:
        _combined = _SEC_SEP.join(sections[k] for k in _str_keys)
        _combined = _limit_expert_reviews(_combined, max_count=3)
        sections = dict(sections)
        for _k, _part in zip(_str_keys, _combined.split(_SEC_SEP)):
            sections[_k] = _part

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
    # Phase-aware subtitle
    _vaihe_raw = (inp.hankkeen_vaihe or "Esiselvitys").lower()
    if _vaihe_raw in ("esiselvitys", ""):
        _phase_sub_txt = _s(lang, "esiselvitys_sub")
    elif _vaihe_raw == "lupavaihe":
        _phase_sub_txt = _s(lang, "lupavaihe_sub")
    elif _vaihe_raw in ("rakentaminen", "rakentamisvaihe"):
        _phase_sub_txt = _s(lang, "rakentaminen_sub")
    else:
        _phase_sub_txt = f"{_t_vaihe(lang, inp.hankkeen_vaihe)} — {_s(lang, 'sub_title')}"
    story.append(Paragraph(
        _phase_sub_txt,
        ParagraphStyle("kan_sub2", fontSize=9, textColor=C_GRAY,
                       fontName=PDF_FONT, spaceAfter=4, leading=13),
    ))
    _meta_kt = _clean_kt(inp.kiinteistotunnus)
    _hanke_short = _HANKE_SHORT.get(inp.hanketyyppi, inp.hanketyyppi.replace("_", " ").title())
    _location = _cover_location(inp.kunta, inp.sijainti_ymparistovaikutukset or "", inp.hanketyyppi)
    _meta_parts = [f"{inp.teho_mw} MW {_hanke_short}", _location]
    if _meta_kt != "–":
        _meta_parts.append(_meta_kt)
    story.append(Paragraph("  ·  ".join(_meta_parts), st["meta"]))
    story.append(Spacer(1, 4*mm))
    story.append(_hr(C_NAVY, 1.5))
    story.append(Spacer(1, 3*mm))

    # Metataulukko
    teho_val = f"{inp.teho_mw} MW"
    if inp.kapasiteetti_mwh and inp.kapasiteetti_mwh > 0:
        teho_val += f"  /  {inp.kapasiteetti_mwh} MWh"
    _ph_labels = _PROMPT_HEADERS.get(lang, _PROMPT_HEADERS["FI"])
    meta_rows = [
        [_s(lang, "m_hakija"),      inp.hakija],
        [_s(lang, "m_ytunnus"),     inp.y_tunnus if inp.y_tunnus else ""],
        [_s(lang, "m_hanketyyppi"), f"{inp.hanketyyppi} — {_nimi(lang, inp.hanketyyppi, cfg['nimi_fi'])}"],
        [_s(lang, "m_teho"),        teho_val],
        [_s(lang, "m_kunta"),       inp.kunta],
        [_s(lang, "m_kt"),          _clean_kt(inp.kiinteistotunnus)],
        *([[_ph_labels.get("phase_label", "Hankkeen vaihe"),
            _t_vaihe(lang, inp.hankkeen_vaihe)]]
          if inp.hankkeen_vaihe else []),
        *([[_s(lang, "m_maa"),       _COUNTRY_CONFIG.get(country, {}).get("name", country)]]
          if country != "FI" else []),
        [_s(lang, "m_laadittu"),        now],
        [_s(lang, "m_laatinut_lbl"),    _s(lang, "m_laatinut")],
        *([[_s(lang, "arviointiviive_lbl"),
            cfg.get("kasittelyaika", {}).get(lang, cfg.get("kasittelyaika", {}).get("EN", ""))]]
          if cfg.get("kasittelyaika") else []),
    ]
    meta_tbl = Table(
        [[Paragraph(k, ParagraphStyle("mk", fontSize=8.5, textColor=C_GRAY,
                                      fontName=PDF_FONT_BOLD)),
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
    if _s(lang, "nce_speed_note"):
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            _s(lang, "nce_speed_note"),
            ParagraphStyle("nce_note", fontSize=8, textColor=C_NAVY,
                           fontName="Helvetica-Oblique", leading=12, alignment=TA_CENTER),
        ))
    story.append(Spacer(1, 8*mm))

    # ── 1. Hankkeen kuvaus ────────────────────────────────────────────────────
    # PageBreak takaa puhtaan aloituksen. KeepTogether sisältää vain otsikon + HR +
    # ensimmäisen kappaleen — [:2] ylitti sivun korkeuden pitkällä AI-sisällöllä ja
    # aiheutti otsikon jäämisen yksin sivulle.
    story.append(PageBreak())
    _kuvaus_elems = _para_text(sections.get("kuvaus", "–"), st)
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec1"), st["h2"]),
        _hr(),
        Spacer(1, 2*mm),
        *_kuvaus_elems[:1],
    ]))
    story.extend(_kuvaus_elems[1:])
    _vaihe_norm = (inp.hankkeen_vaihe or "esiselvitys").lower()
    if _vaihe_norm == "lupavaihe":
        story.append(Paragraph(_s(lang, "lupavaihe_p"), st["body"]))
    elif _vaihe_norm in ("rakentaminen", "rakentamisvaihe"):
        story.append(Paragraph(_s(lang, "rakentaminen_p"), st["body"]))
    else:
        story.append(Paragraph(_s(lang, "esiselvitys_p"), st["body"]))
    if inp.hanketyyppi == "BESS":
        story.append(Paragraph(_s(lang, "bess_pintaala"), st["body"]))
    story.append(Paragraph(_s(lang, "mks_viittaus"), st["body"]))
    story.append(Spacer(1, 4*mm))

    # ── 2. Perustelut ja tarve ────────────────────────────────────────────────
    story.append(PageBreak())
    _perust_elems = _para_text(sections.get("perustelut", "–"), st)
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec2"), st["h2"]),
        _hr(),
        Spacer(1, 2*mm),
        *_perust_elems[:1],
    ]))
    story.extend(_perust_elems[1:])
    story.append(Spacer(1, 4*mm))

    # ── 3. Tarvittavat luvat ja viranomaiset ─────────────────────────────────
    story.append(PageBreak())
    _luvat_tbl = _luvat_table(inp.hanketyyppi, st, lang, country)
    _country_luvat_data = _COUNTRY_LUVAT.get(country, {}).get(inp.hanketyyppi)
    _luvat_row_count = len(_country_luvat_data or _HANKE_CFG.get(inp.hanketyyppi, {}).get("luvat", []))
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec3"), st["h2"]),
        _hr(),
        _luvat_tbl,
    ]))
    story.append(Spacer(1, 5*mm))
    _kaava_key = _KAAVA_KEY.get(inp.hanketyyppi, "kaava_generic")
    story.append(Paragraph(_s(lang, _kaava_key), st["body"]))
    luvat_txt = sections.get("luvat_teksti", "")
    if luvat_txt:
        story.extend(_para_text(luvat_txt, st))
    story.append(Spacer(1, 4*mm))

    # ── ISO/IEC-standardit ───────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec_standards"), st["h2"]),
        _hr(),
        _standards_table(inp.hanketyyppi, country, lang, st),
    ]))
    story.append(Spacer(1, 4*mm))

    # ── 4. Lakiviitteet ───────────────────────────────────────────────────────
    story.append(PageBreak())
    country_luvat_override = _COUNTRY_LUVAT.get(country, {}).get(inp.hanketyyppi)
    if country_luvat_override:
        laki_set = {laki for _, _, laki in country_luvat_override}
    else:
        laki_set = {laki for _, _, laki in cfg["luvat"]}
        laki_set.update(cfg.get("laki_extra", []))
    laki_bullets = [Paragraph(f"• {_t_law(lang, ref)}", st["bullet"])
                    for ref in sorted(laki_set)]
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec4"), st["h2"]),
        _hr(),
        *laki_bullets[:2],
    ]))
    for b in laki_bullets[2:]:
        story.append(b)
    story.append(Spacer(1, 4*mm))

    # ── 5. Liiteluettelo ──────────────────────────────────────────────────────
    story.append(PageBreak())
    _liite_tbl = _liitteet_table(inp.hanketyyppi, lang, country)
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec5"), st["h2"]),
        _hr(),
        Paragraph(_s(lang, "liitteet_note"), st["body"]),
        Spacer(1, 3*mm),
        _liite_tbl,
    ]))
    story.append(Spacer(1, 4*mm))

    # ── 6. Seuraavat toimenpiteet ─────────────────────────────────────────────
    story.append(PageBreak())
    _toim_elems = _toimenpiteet_elements(sections.get("toimenpiteet", "–"), st, lang)
    story.append(KeepTogether([
        Paragraph(_s(lang, "sec6"), st["h2"]),
        _hr(),
        *_toim_elems,
    ]))
    story.append(Spacer(1, 4*mm))

    # ── Varoitusbanneri (Task 4: warning_flag) ─────────────────────────────────
    if warning_flag:
        _warn_text = {
            "FI": "⚠️ Lähdeaineisto rajallinen — tarkista kaikki kohdat huolellisesti",
            "EN": "⚠️ Limited source material — verify all sections carefully",
            "SE": "⚠️ Begränsat källmaterial — kontrollera alla avsnitt noggrant",
            "DA": "⚠️ Begrænset kildemateriale — kontroller alle afsnit omhyggeligt",
            "NO": "⚠️ Begrenset kildemateriale — kontroller alle seksjoner nøye",
            "PL": "⚠️ Ograniczony materiał źródłowy — sprawdź dokładnie wszystkie sekcje",
            "DE": "⚠️ Begrenztes Quellmaterial — alle Abschnitte sorgfältig prüfen",
        }
        story.append(Spacer(1, 4*mm))
        _w_row = [[Paragraph(
            _warn_text.get(lang, _warn_text["FI"]),
            ParagraphStyle("warn_b", fontSize=8.5, textColor=colors.HexColor("#7a4400"),
                           fontName=PDF_FONT_BOLD, alignment=TA_CENTER, leading=13),
        )]]
        _w_tbl = Table(_w_row, colWidths=[16.5*cm])
        _w_tbl.setStyle(TableStyle([
            ("BOX",        (0, 0), (-1, -1), 1.2, colors.HexColor("#ff9800")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff3e0")),
            ("PADDING",    (0, 0), (-1, -1), 8),
        ]))
        story.append(_w_tbl)
        story.append(Spacer(1, 4*mm))

    # ── Lähteet (vain RAG-viranomaislähteet; säädösperusta on jo osiossa 4) ───
    if sources:
        story.append(KeepTogether([
            Paragraph(_s(lang, "lahteet_h"), st["h2"]),
            _hr(),
        ]))
        story.append(Paragraph(_s(lang, "lahteet_b"), st["body"]))
        # Task 4: per-source citation lines
        _src_label = {"FI": "Lähde", "EN": "Source", "SE": "Källa", "DA": "Kilde",
                       "NO": "Kilde", "PL": "Źródło", "DE": "Quelle"}.get(lang, "Lähde")
        _reg_label = {"FI": "viranomaisrekisteri", "EN": "authority registry",
                       "SE": "myndighetsregister", "DA": "myndighedsregister",
                       "NO": "myndighetsregister", "PL": "rejestr organów",
                       "DE": "Behördenregister"}.get(lang, "viranomaisrekisteri")
        for src in sources[:8]:
            src_display = src.get("display") or src.get("id", "–")
            src_country = src.get("country", country)
            cite_line   = f"{_src_label}: {src_display} ({src_country} {_reg_label})"
            story.append(Paragraph(f"• {cite_line}", st["bullet"]))
        story.append(Spacer(1, 3*mm))

    # ── Ennakkotapaukset (Task 3) ──────────────────────────────────────────────
    if prec_chunks:
        _prec_h = {"FI": "Ennakkotapaukset ja viranomaisratkaisut",
                    "EN": "Precedents and Regulatory Decisions",
                    "SE": "Prejudikat och myndighetsbeslut",
                    "DA": "Præjudikater og myndighedsafgørelser",
                    "NO": "Prejudikater og myndighetsbeslutninger",
                    "PL": "Precedensy i decyzje organów regulacyjnych",
                    "DE": "Präzedenzfälle und Behördenentscheidungen"}.get(lang, "Ennakkotapaukset")
        story.append(PageBreak())
        story.append(KeepTogether([
            Paragraph(_prec_h, st["h2"]),
            _hr(),
        ]))
        _prec_note = {
            "FI": ("Seuraavat viranomaisratkaisut ja ennakkotapaukset on haettu NCE RAG-tietokannasta. "
                   "Ne antavat viitteitä lupakäytännöistä — tarkista ajantasaisuus ennen käyttöä."),
            "EN": ("The following regulatory decisions and precedents were retrieved from the NCE RAG database. "
                   "They provide guidance on permit practices — verify currency before relying on them."),
        }.get(lang, (
            "Seuraavat viranomaisratkaisut ja ennakkotapaukset on haettu NCE RAG-tietokannasta. "
            "Ne antavat viitteitä lupakäytännöistä — tarkista ajantasaisuus ennen käyttöä."
        ))
        story.append(Paragraph(_prec_note, st["body"]))
        story.append(Spacer(1, 3*mm))
        for chunk, src in zip(prec_chunks[:3], prec_sources[:3]):
            story.append(Paragraph(
                f"<b>{src}</b>",
                ParagraphStyle("prec_src", fontSize=8.5, fontName=PDF_FONT_BOLD, textColor=C_NAVY,
                               spaceBefore=4, leading=12),
            ))
            story.extend(_para_text(chunk[:600] + ("…" if len(chunk) > 600 else ""), st))
            story.append(Spacer(1, 2*mm))
        story.append(Spacer(1, 4*mm))

    # ── NCE Permit AI -infolaatikko ───────────────────────────────────────────
    _nce_desc = _s(lang, "nce_info_desc") or (
        "NCE Permit AI on tekoälypohjainen työkalu energia-alan lupahakemusten "
        "valmisteluun. Palvelu hyödyntää RAG-teknologiaa (Retrieval-Augmented Generation) "
        "ja hakee tietoa Fingridin, Tukesin, SYKE:n ja STUK YVL -ohjeistojen "
        "ajantasaisesta dokumentaatiosta. Luonnos vaatii aina asiantuntijatarkistuksen "
        "ennen viranomaiskäsittelyä."
    )
    _nce_info_rows = [
        [Paragraph("NCE Permit AI — Tietoja raportista", ParagraphStyle(
            "nce_ih", fontSize=9, fontName=PDF_FONT_BOLD, textColor=C_NAVY)),
         Paragraph("ncenergy.fi  ·  info@ncenergy.fi", ParagraphStyle(
            "nce_ir", fontSize=8.5, textColor=C_GRAY, alignment=TA_RIGHT))],
        [Paragraph(_nce_desc, ParagraphStyle(
            "nce_ib", fontSize=8.5, leading=13, textColor=colors.HexColor("#2d3748"))),
         ""],
    ]
    _nce_info_tbl = Table(_nce_info_rows, colWidths=[11.5*cm, 5.0*cm])
    _nce_info_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#EBF4F7")),
        ("BACKGROUND",   (0, 1), (-1, 1), colors.HexColor("#F7FBFD")),
        ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#B0D4E0")),
        ("LINEBELOW",    (0, 0), (-1, 0), 0.5, colors.HexColor("#B0D4E0")),
        ("SPAN",         (0, 1), (-1, 1)),
        ("PADDING",      (0, 0), (-1, -1), 7),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(KeepTogether([
        _nce_info_tbl,
        Spacer(1, 4*mm),
    ]))

    # ── Hakijan yhteystiedot ──────────────────────────────────────────────────
    yhteystiedot_data = [
        [_s(lang, "yht_hakija"), inp.hakija],
        *([[_s(lang, "yht_ytunnus"), inp.y_tunnus]] if inp.y_tunnus else []),
        *([[_s(lang, "yht_osoite"),  inp.osoite]]  if inp.osoite  else []),
        [_s(lang, "yht_lisatietoja"), "NCE Permit AI  ·  ncenergy.fi  ·  info@ncenergy.fi"],
    ]
    yht_tbl = Table(
        [[Paragraph(k, ParagraphStyle("yk", fontSize=8.5, textColor=C_GRAY, fontName=PDF_FONT_BOLD)),
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
    story.append(KeepTogether([
        Paragraph(_s(lang, "yhteystiedot_h"), st["h2"]),
        _hr(),
        yht_tbl,
    ]))
    story.append(Spacer(1, 4*mm))

    # ── Task 5: Client briefing page ─────────────────────────────────────────
    story.append(PageBreak())
    _brief_title = {
        "FI": "Ohjeistus asiantuntijatarkistukseen",
        "EN": "Expert Review Guide",
        "SE": "Guide för expertgranskning",
        "DA": "Vejledning til ekspertgennemgang",
        "NO": "Veiledning for ekspertgjennomgang",
        "PL": "Przewodnik po przeglądzie eksperckim",
        "DE": "Leitfaden zur Expertenprüfung",
    }.get(lang, "Expert Review Guide")
    story.append(Paragraph(_brief_title, st["h2"]))
    story.append(_hr(C_NAVY, 1.5))
    story.append(Spacer(1, 4*mm))

    _chunk_count = len(sources)
    _brief_rows = []

    # What the report contains
    _cont_h = {"FI": "Raportti sisältää", "EN": "This report contains", "SE": "Rapporten innehåller",
                "DA": "Rapporten indeholder", "NO": "Rapporten inneholder",
                "PL": "Raport zawiera", "DE": "Der Bericht enthält"}.get(lang, "This report contains")
    _cont_v = {
        "FI": (f"AI-luonnos — {_chunk_count} RAG-lähteestä haettu konteksti  •  "
               f"Hanketyyppi: {inp.hanketyyppi}  •  Maa: {country}  •  "
               f"Laadittu: {now}  •  NCE Permit AI MVP"),
        "EN": (f"AI draft — context from {_chunk_count} RAG sources  •  "
               f"Project type: {inp.hanketyyppi}  •  Country: {country}  •  "
               f"Generated: {now}  •  NCE Permit AI MVP"),
    }.get(lang, (
        f"AI draft — context from {_chunk_count} RAG sources  •  "
        f"Project type: {inp.hanketyyppi}  •  Country: {country}  •  "
        f"Generated: {now}  •  NCE Permit AI MVP"
    ))
    _brief_rows.append([_cont_h, _cont_v])

    # What it does NOT contain
    _not_h = {"FI": "Ei sisällä", "EN": "Does NOT contain", "SE": "Innehåller INTE",
               "DA": "Indeholder IKKE", "NO": "Inneholder IKKE",
               "PL": "NIE zawiera", "DE": "Enthält NICHT"}.get(lang, "Does NOT contain")
    _not_v = {
        "FI": ("Juridisia neuvoja  •  Sitovia viranomaistulkintoja  •  "
               "Paikkansapitäviä kiinteistö- tai kaava-tietoja  •  "
               "Lopullisia kustannusarvioita"),
        "EN": ("Legal advice  •  Binding regulatory interpretations  •  "
               "Verified property or zoning data  •  Final cost estimates"),
    }.get(lang, (
        "Legal advice  •  Binding regulatory interpretations  •  "
        "Verified property or zoning data  •  Final cost estimates"
    ))
    _brief_rows.append([_not_h, _not_v])

    # What to check
    _chk_h = {"FI": "Tarkista ennen jättämistä", "EN": "Verify before submission",
               "SE": "Verifiera före inlämning", "DA": "Kontrollér inden indsendelse",
               "NO": "Verifiser før innsending", "PL": "Sprawdź przed złożeniem",
               "DE": "Vor der Einreichung prüfen"}.get(lang, "Verify before submission")
    _chk_items = []
    if warning_flag:
        _chk_items.append({"FI": "⚠️ Merkityt osiot — erityistä huomiota vaativat kohdat",
                            "EN": "⚠️ Flagged sections — require special attention"}.get(lang,
                           "⚠️ Flagged sections — require special attention"))
    _chk_items += [
        {"FI": "Lakiviitteiden ajantasaisuus", "EN": "Currency of statutory references",
         "SE": "Rättsreferensernas aktualitet", "DA": "Lovhenvisningernes aktualitet",
         "NO": "Lovhenvisningenes aktualitet", "PL": "Aktualność odniesień prawnych",
         "DE": "Aktualität der gesetzlichen Referenzen"}.get(lang, "Currency of statutory references"),
        {"FI": "Kaavatilanne ja kiinteistötiedot", "EN": "Zoning status and property data",
         "SE": "Planstatus och fastighetsuppgifter", "DA": "Planstatus og ejendomsdata",
         "NO": "Planstatus og eiendomsdata", "PL": "Status planu i dane nieruchomości",
         "DE": "Bebauungsplan und Grundstücksdaten"}.get(lang, "Zoning status and property data"),
        {"FI": "Paikalliset kaavoitusmääräykset", "EN": "Local zoning requirements",
         "SE": "Lokala planbestämmelser", "DA": "Lokale planbestemmelser",
         "NO": "Lokale planbestemmelser", "PL": "Lokalne wymogi planistyczne",
         "DE": "Lokale Bebauungsvorschriften"}.get(lang, "Local zoning requirements"),
    ]
    _brief_rows.append([_chk_h, "  •  ".join(_chk_items)])

    # Contact
    _brief_rows.append(["NCE Permit AI", "info@ncenergy.fi  ·  ncenergy.fi"])

    _brief_tbl = Table(
        [[Paragraph(k, ParagraphStyle("bk", fontSize=8.5, textColor=C_GRAY, fontName=PDF_FONT_BOLD,
                                       leading=13)),
          Paragraph(v, ParagraphStyle("bv", fontSize=8.5, leading=13))]
         for k, v in _brief_rows],
        colWidths=[4.5*cm, 12.0*cm],
    )
    _brief_tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LGRAY, C_WHITE]),
        ("PADDING",        (0, 0), (-1, -1), 7),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DGRAY),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(_brief_tbl)
    story.append(Spacer(1, 6*mm))

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
    rag_ctx, sources, warning_flag, prec_chunks, prec_sources = \
        _rag_context(inp.hanketyyppi, inp.country or "FI")
    sections = _generate_sections(inp, rag_ctx, prec_chunks, prec_sources)
    _lang = inp.lang or "FI"
    sections = _final_polish(sections, _lang)
    pdf_bytes = generate_pdf(inp, sections, sources, warning_flag, prec_chunks, prec_sources)
    return pdf_bytes, sections, sources


def apply_proofread_to_pdf(
    inp: ApplicationInput,
    sections: dict,
    sources: list,
    warning_flag: bool = False,
    prec_chunks: Optional[list] = None,
    prec_sources: Optional[list] = None,
) -> bytes:
    """Oikolue sections Claudella ja rakenna lopullinen PDF."""
    _lang = inp.lang or "FI"
    sections = _proofread_sections(sections)
    sections = _final_polish(sections, _lang)
    return generate_pdf(
        inp, sections, sources,
        warning_flag, prec_chunks or [], prec_sources or [],
    )


def generate_application(inp: ApplicationInput) -> str:
    """Generoi lupahakemus-PDF ja palauta tallennuspolku."""
    logger.warning("DEBUG TEST: äö toimii - raw=%s", repr("testäö"))
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    print(f"[1/3] Haetaan RAG-konteksti ({inp.hanketyyppi}, maa={inp.country or 'FI'})…")
    rag_ctx, sources, warning_flag, prec_chunks, prec_sources = \
        _rag_context(inp.hanketyyppi, inp.country or "FI")
    print(f"      {len(rag_ctx.split())} sanaa, lähteet: {[s['display'] for s in sources]}")
    if warning_flag:
        print("      ⚠️  RAG_WARN: rajallinen lähdeaineisto")

    print("[2/4] Generoidaan hakemusteksti (Claude)…")
    sections = _generate_sections(inp, rag_ctx, prec_chunks, prec_sources)
    print(f"      Osiot: {list(sections.keys())}")

    print("[3/4] Oikoluku ja tekstikorjaus (Claude + säännöt)…")
    _lang = inp.lang or "FI"
    sections = _proofread_sections(sections)
    sections = _final_polish(sections, _lang)

    print("[4/4] Rakennetaan PDF…")
    pdf_bytes = generate_pdf(inp, sections, sources, warning_flag, prec_chunks, prec_sources)

    _FILE_PREFIX = {"FI": "hakemus", "EN": "application", "SE": "ansökan",
                     "DA": "ansøgning", "NO": "søknad", "PL": "wniosek"}
    _prefix    = _FILE_PREFIX.get(inp.lang or "FI", "hakemus")
    _kt        = re.sub(r"[^a-zA-Z0-9À-ɏ]", "_", inp.hanketyyppi or "doc")
    _kunta     = re.sub(r"[^a-zA-Z0-9À-ɏ]", "_", inp.kunta or "hanke")
    out_path   = os.path.join(_OUTPUT_DIR, f"{_prefix}_{_kt}_{_kunta}.pdf")
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
