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
    CondPageBreak, HRFlowable, Image, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
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


class GenerationCapError(RuntimeError):
    """Raised when a single generation exceeds its per-generation_id ceiling on
    RAG retrieval calls or Claude API calls (see retrieval_trace.py's
    count_retrieval_calls()/count_claude_calls()). TASO 1 cost & resource
    guardrail — a hard, visibility-only ceiling; NOT throttling or rate limiting
    (see the task's explicit out-of-scope note). Today's pipeline always makes
    exactly 1 RAG retrieval call and 4 Claude API calls per generation with zero
    variance, so this is a forward-looking safety net (e.g. for a future
    Autonomous RAG Agent's retry/iterate loop), not a response to any observed
    runaway behavior. Caps confirmed with the user 2026-08-07 before being
    hardcoded, per the task's explicit instruction.
    """
    def __init__(self, kind: str, generation_id: str, count: int, cap: int):
        self.kind          = kind           # "retrieval_iteration" | "claude_call"
        self.generation_id = generation_id
        self.count         = count
        self.cap           = cap
        super().__init__(
            f"GENERATION_CAP_HIT: {kind} cap exceeded for generation_id={generation_id} "
            f"({count}/{cap}) — stopping to avoid an uncontrolled cost/resource loop"
        )


class PhaseContentNotAvailableError(Exception):
    """Raised when a hanketyyppi has a real, declared later phase (per
    _PHASE_RAG_QUERIES[hanketyyppi] — e.g. SMR's kayttolupa/purku) but no
    _PHASE_INSTRUCTIONS writing-guidance entry for it yet (P3-3b, the
    content-gated follow-up, is what adds those entries). Hanketyyppi-
    scoped, not just vaihe-name-scoped — a vaihe name absent from THIS
    hanketyyppi's own _PHASE_RAG_QUERIES entry (e.g. "kayttolupa" for
    BESS, which was never declared) does NOT raise this.

    Deliberately distinct from an unrecognized/typo'd hankkeen_vaihe string,
    which still falls back to esiselvitys's tone unchanged, same as always —
    this is specifically for phases the platform KNOWS about but has no
    content for yet, so generation stops cleanly here instead of silently
    producing an esiselvitys-toned report for what should be an operating-
    license or decommissioning document. Same reasoning as
    GenerationCapError's retrieval-iteration check above: never silently
    swallow this into a wrong-but-plausible-looking fallback.
    """
    def __init__(self, hanketyyppi: str, vaihe: str):
        self.hanketyyppi = hanketyyppi
        self.vaihe       = vaihe
        super().__init__(
            f"PHASE_CONTENT_NOT_AVAILABLE: {hanketyyppi}/{vaihe} — phase "
            "recognized but writing guidance not yet published (P3-3b pending)"
        )


# TODO: domain muutos ncepermit.ai kun NCE Global perustettu
_HERE        = os.path.dirname(os.path.abspath(__file__))
_DB_DIR      = os.path.join(_HERE, "embeddings")
_OUTPUT_DIR  = os.path.join(_HERE, "output")
_RAQS_DB     = os.path.join(_DB_DIR, "raqs_reviews.db")
_CITATION_GAP_DB = os.path.join(_DB_DIR, "citation_gap_flags.db")
# TASO 1 cost & resource guardrail (confirmed with user 2026-08-07). Current fixed
# pipeline usage was 1 RAG retrieval call + 4 Claude API calls per generation
# (draft + proofread + RAQS×2); the RAQS-removal task (2026-08-09) cut the
# dead draft-PDF RAQS call, so it's now 3 (draft + proofread + RAQS×1). Caps
# left as-is (now ~2x / 3x headroom instead of 1.5x / 3x) rather than
# retuned down — still reacting to an observed runaway, not the fixed
# baseline itself, so tighter caps aren't a goal of this task.
_RAG_ITERATION_CAP = 3
_CLAUDE_CALL_CAP    = 6
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


_UNICODE_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def _fix_unicode_subscripts(text: str) -> str:
    """Convert Unicode subscript digits to ReportLab <sub> markup.

    DejaVu Sans on Render lacks glyphs for U+2080-U+2089, rendering them as
    black squares. Chemical formulas like H₂ and CO₂ must be converted to
    H<sub>2</sub> / CO<sub>2</sub> before the text reaches the PDF renderer.
    """
    return re.sub(
        r"[₀₁₂₃₄₅₆₇₈₉]+",
        lambda m: f"<sub>{m.group().translate(_UNICODE_SUBSCRIPTS)}</sub>",
        text,
    )


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


# SentenceTransformer.encode() is not thread-safe: concurrent calls can segfault the ONNX
# runtime when two background threads share the same model instance. Serialise all RAG
# retrieval so only one _rag_context runs at a time (Claude API calls run concurrently —
# they are a remote service and don't share any in-process state).
import threading as _threading
_RAG_LOCK = _threading.Lock()

# TEMPORARY diagnostic instrumentation (2026-07-25) — investigating the recurring
# ~350-380s "Claude API -aikakatkaisu (>120s)" timeout. Records wall-clock checkpoints
# for the single most recent generate_application_draft() call. Not job_id-keyed —
# only meaningful when tested one generation at a time. Remove once root-caused.
import time as _time
_LAST_TIMING: dict = {}


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

# Deterministic backstop for the tuulivoima_maa Traficom hardcoding (PR #36/#37):
# prompt-level instructions alone aren't reliable here, because RAG-retrieved
# Finnish source chunks can surface the literal word "Traficom" regardless of
# what the prompt says — confirmed live (PR #37 deployed, DA/NO/PL all still
# showed the leak in the same generated text as the correct new authority).
# This guarantees the wrong Finnish entity never reaches the final PDF for
# non-FI countries, independent of what the LLM actually wrote.
#
# 2026-08-23 (manual QA item 4): broadened to also catch "Finavia" (Finland's
# airport operator, the second half of the "Lentoestekartoitus (Traficom/
# Finavia)" liite name in _LIITE_TRANS) — same class of leak, same target
# replacement per country, since both are FI-specific aviation entities that
# the generic national aviation authority below correctly substitutes for.
_TRAFICOM_REPLACEMENT: dict[str, str] = {
    "SE": "Transportstyrelsen",
    "DA": "Trafikstyrelsen",
    "NO": "Luftfartstilsynet",
    "PL": "ULC (Urząd Lotnictwa Cywilnego)",
    "EE": "Transpordiamet",
    "DE": "Luftfahrtbundesamt",
    "LV": "LGS (Latvijas gaisa satiksme)",
    "LT": "ANCO / Civilinės aviacijos administracija",
}
_TRAFICOM_RE = re.compile(r'\bTraficom\w*|\bFinavia\w*')


def _fix_hardcoded_traficom(text: str, country: str) -> str:
    """Replace any literal 'Traficom' or 'Finavia' mention with the correct
    national aviation authority for non-FI countries. See
    _TRAFICOM_REPLACEMENT comment above.

    Traficom and Finavia both map to the SAME replacement authority (both are
    FI-specific aviation entities; the target country only has one relevant
    equivalent), so a combined mention like "Traficom/Finavia" would collapse
    to a visible duplicate ("Transportstyrelsen/Transportstyrelsen") without
    this dedup step.
    """
    auth = _TRAFICOM_REPLACEMENT.get(country)
    if not auth:
        return text
    text = _TRAFICOM_RE.sub(auth, text)
    dup = re.escape(auth)
    # (?<!\w)/(?!\w) rather than \b: several replacement values themselves end
    # in ")" (e.g. PL "ULC (Urząd Lotnictwa Cywilnego)"), and \b requires a
    # genuine word/non-word transition — it silently fails to match when the
    # duplicate is immediately followed by another non-word character (e.g.
    # the liite name's own closing paren), which \w-based lookarounds handle.
    return re.sub(rf'(?<!\w){dup}\s*/\s*{dup}(?!\w)', auth, text)


# Deterministic backstop for the ELY-keskus/NTM-centralen leak (manual QA item
# 4, 2026-08-23): ELY-keskus is a Finnish REGIONAL authority. _AUTHORITY_TRANS
# only holds a literal per-language CALQUE of the Finnish name for EN/SE/DA/
# NO/PL (e.g. Swedish "NTM-centralen") — not a real competent authority in the
# target country — and has no entry at all for DE/ET/LV/LT, so those fall
# through to the raw untranslated Finnish "ELY-keskus". Confirmed live: a SE
# generation showed "NTM-centralen" in its bilaga list, reading as if Sweden
# had an authority literally called that. Same class of bug as Traficom
# above, same backstop shape (country dict + regex + function), applied at
# the same _t_liite() choke point so it's independent of _AUTHORITY_TRANS's
# own (currently incomplete) per-language coverage.
_ELY_REPLACEMENT: dict[str, str] = {
    "SE": "Länsstyrelsen",
    "DA": "Miljøstyrelsen",
    "NO": "Statsforvalteren",
    "PL": "RDOŚ (Regionalna Dyrekcja Ochrony Środowiska)",
    "EE": "Keskkonnaamet",
    "DE": "zuständige Umweltbehörde (Land)",
    "LV": "Valsts vides dienests",
    "LT": "Aplinkos apsaugos agentūra",
}
_ELY_RE = re.compile(
    r'\bELY[-\s]keskus\w*|\bELY:n\b|\bELY\s*Centre\w*|\bELY-center\w*|\bELY-senter\w*|'
    r'\bCentrum\s+ELY\w*|\bNTM-central\w*'
)


def _fix_hardcoded_ely(text: str, country: str) -> str:
    """Replace any Finnish 'ELY-keskus' mention (raw, or its literal
    per-language calque such as Swedish 'NTM-centralen') with the real
    competent regional/environmental authority for non-FI countries. See
    _ELY_REPLACEMENT comment above."""
    auth = _ELY_REPLACEMENT.get(country)
    if not auth:
        return text
    return _ELY_RE.sub(auth, text)


# Item 5(B) law-citation backstop (2026-08-23, SE research — see
# SE_LAW_CITATION_RESEARCH.md for full sourcing). Deterministic backstop for
# bare/untranslated Finnish statute citations (e.g. "MRL 132/1999", "YVA-laki
# 252/2017") leaking directly into freeform generated prose — same RAG-leak
# failure mode the Traficom/STUK/ELY-keskus backstops above guard against
# (RAG-retrieved Finnish source chunks can surface the literal citation
# regardless of what the prompt says). This is a SEPARATE, additional layer
# to the _LAW_TRANS table fix (also shipped in this change) which covers the
# STRUCTURED law-reference call sites (_t_law(), the Lakiviitteet section,
# the liite/lupa tables) — this backstop covers the un-structured case where
# the model writes the citation directly into kuvaus/perustelut/luvat_teksti/
# toimenpiteet prose, bypassing _t_law() entirely.
#
# Only SE is populated so far (this is the first country researched in the
# item 5(B)/item 4 combined queue — see SE_LAW_CITATION_RESEARCH.md). Adding
# a country here requires real per-country statute research first, same
# discipline as _TRAFICOM_REPLACEMENT/_ELY_REPLACEMENT above — do NOT add a
# plausible-looking entry without that research.
#
# Deliberately drops any trailing Finnish paragraph/section reference (e.g.
# ", 77a §") rather than trying to preserve or translate it — there is no
# verified section-level correspondence between the Finnish law's paragraph
# numbering and the target country's own law, and keeping the Finnish
# paragraph number next to the (now correct) target-country act would read
# as a claim about THAT act's own section numbering, which would be a new,
# separate fabrication risk.
_LAW_CITATION_REPLACEMENT: dict[str, dict[str, str]] = {
    "MRL":                  {"SE": "Plan- och bygglagen (2010:900)"},
    "Rakentamislaki":       {"SE": "Plan- och bygglagen (2010:900)"},
    "YVA-laki":             {"SE": "Miljöbalken kap. 6 (miljöbedömningar)"},
    "YSL":                  {"SE": "Miljöbalken kap. 9 (miljöfarlig verksamhet)"},
    "Vesilaki":             {"SE": "Miljöbalken kap. 11 (vattenverksamhet)"},
    "Pelastuslaki":         {"SE": "Lag (2003:778) om skydd mot olyckor"},
    "Ydinenergialaki":      {"SE": "Lag (1984:3) om kärnteknisk verksamhet"},
    "Patoturvallisuuslaki": {"SE": "Förordning (2014:214) om dammsäkerhet"},
}
_LAW_CITATION_NUMS: dict[str, str] = {
    "MRL": "132/1999", "Rakentamislaki": "751/2023", "YVA-laki": "252/2017",
    "YSL": "527/2014", "Vesilaki": "587/2011", "Pelastuslaki": "379/2011",
    "Ydinenergialaki": "990/1987", "Patoturvallisuuslaki": "494/2009",
}
_LAW_CITATION_RE: dict[str, re.Pattern] = {
    name: re.compile(
        rf'\b{re.escape(name)}\s+{re.escape(num)}'
        rf'(?:\s*,?\s*§\s*[\w:–\-]+|\s*,?\s*\d+[a-z]?\s*§)?'
    )
    for name, num in _LAW_CITATION_NUMS.items()
}


def _fix_hardcoded_law_citations(text: str, country: str) -> str:
    """Replace any bare/untranslated Finnish statute citation (name + number,
    with or without a trailing paragraph reference) with the real, researched
    statutory equivalent for non-FI countries. See _LAW_CITATION_REPLACEMENT
    comment above. No-ops entirely for countries without a researched entry."""
    for name, repl_by_country in _LAW_CITATION_REPLACEMENT.items():
        repl = repl_by_country.get(country)
        if not repl:
            continue
        text = _LAW_CITATION_RE[name].sub(repl, text)
    return text


# Same class of bug as Traficom above, found via the 2026-07-28 translation-table
# audit: SMR/smr_bess's base liitteet list includes "Alustava turvallisuusseloste
# (STUK YVL A.1 mukainen)" (STUK = Finnish nuclear safety regulator). SE/DA/NO/PL/
# EE/DE all have full _COUNTRY_LIITTEET overrides for SMR+smr_bess with zero STUK
# mentions (verified) — this is a no-op backstop for them. LV and LT have no
# override at all, so this item falls through to the base Finnish list unchanged.
# LT has a real, well-documented nuclear safety regulator (VATESI). Latvia has no
# nuclear power plants and no nuclear regulatory framework at all (established
# elsewhere in this file's LV country config) — a real authority name would be
# fabricated, so LV gets an explicit verification hedge instead, matching the
# "[Vaatii tarkistuksen...]" pattern already used throughout this file for cases
# with no identified equivalent.
_STUK_REPLACEMENT: dict[str, str] = {
    "LT": "VATESI (Valstybinė atominės energetikos saugos inspekcija)",
    "LV": "[Vaatii tarkistuksen — Latvialla ei ole omaa ydinturvallisuusviranomaista]",
}
_STUK_RE = re.compile(r'\bSTUK\w*')


def _fix_hardcoded_stuk(text: str, country: str) -> str:
    """Replace any literal 'STUK' mention with the correct national nuclear safety
    authority (or a verification hedge) for non-FI countries. See
    _STUK_REPLACEMENT comment above."""
    auth = _STUK_REPLACEMENT.get(country)
    if not auth:
        return text
    return _STUK_RE.sub(auth, text)


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
    "LT": "[Pastaba] ",
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


def _final_polish(sections: dict, lang: str, country: str = "FI", hanketyyppi: str = "") -> dict:
    """Loppu-oikoluku — suoritetaan AINA viimeisenä ennen PDF-rakennusta.

    1. Deterministinen diakriittikorjaus (ä/ö) kaikille suomenkielisille kentille.
    2. Viranomaistermien ja lakiviitteiden korjaus (_postprocess_text).
    3. tuulivoima_maa: pakollinen Traficom-korvaus muille kuin FI-hankkeille.
    4. Bare FI-lakiviitteiden korvaus (_fix_hardcoded_law_citations) — ei
       hanketyyppi-rajattu, koska lakiviitteet esiintyvät useissa hanketyypeissä.
    5. Asiantuntijatarkistus-merkintöjen karsinta enintään 3 kappaleeseen.
    """
    result = {}
    for k, v in sections.items():
        if not isinstance(v, str):
            result[k] = v
            continue
        v = _fix_fi_diacritics(v)
        v = _postprocess_text(v, lang)
        if hanketyyppi == "tuulivoima_maa":
            v = _fix_hardcoded_traficom(v, country)
        v = _fix_hardcoded_law_citations(v, country)
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

def _proofread_sections(sections: dict, generation_id: str = "") -> dict:
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
    _PROOF_SYSTEM = (
        "Olet asiantunteva tekninen toimittaja. Korjaat suomalaisia lupahakemusluonnoksia "
        "diakriittimerkki- ja kielioppivirheiden osalta. Palauta teksti täsmälleen samassa "
        "rakenteessa kuin sait sen."
    )
    # Streaming, same reasoning and pattern as the main generation call in
    # _generate_sections(): a blocking call with timeout=120.0 can trigger the
    # SDK's default retry on a call that's merely slow, cascading into a
    # multi-minute wait before finally hitting this function's own fallback.
    # timeout=600.0 is safe here specifically because the fallback-on-genuine-
    # timeout behavior below is unchanged — a real stall still degrades
    # gracefully to the original text, it just won't be mistaken for a stall
    # after 120s of ordinary (if slow) progress.
    _LAST_TIMING["t7_proofread_start"] = _time.monotonic()
    # Cost & resource guardrail (TASO 1): checked outside the try/except below —
    # this function's broad `except Exception: return sections` fallback exists
    # for genuine proofread hiccups (degrade to unproofread text, never block the
    # PDF); a cap hit is different and must fail cleanly, not be swallowed into
    # that same silent-degradation path. See GenerationCapError.
    if generation_id:
        _n_calls = _retrieval_trace.count_claude_calls(generation_id)
        if _n_calls >= _CLAUDE_CALL_CAP:
            _retrieval_trace.log_guardrail_hit(
                generation_id=generation_id,
                guard_type="claude_call_cap",
                count_at_trip=_n_calls,
                cap=_CLAUDE_CALL_CAP,
                detail="call_type=proofread",
            )
            raise GenerationCapError("claude_call", generation_id, _n_calls, _CLAUDE_CALL_CAP)
    try:
        with anthropic.Anthropic(timeout=600.0).messages.stream(
            model=_MODEL_ID,
            max_tokens=6000,
            system=[{"type": "text", "text": _PROOF_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            raw_parts = [text for text in stream.text_stream]
            resp = stream.get_final_message()
        _LAST_TIMING["t7c_proofread_succeeded"] = _time.monotonic()
        _pu = resp.usage
        logger.warning(
            "[cache/proof] creation=%s read=%s input=%s output=%s",
            getattr(_pu, "cache_creation_input_tokens", 0),
            getattr(_pu, "cache_read_input_tokens", 0),
            _pu.input_tokens, _pu.output_tokens,
        )
        if generation_id:
            _retrieval_trace.log_api_call(
                generation_id=generation_id,
                call_type="proofread",
                model=_MODEL_ID,
                input_tokens=_pu.input_tokens,
                cache_creation_input_tokens=getattr(_pu, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(_pu, "cache_read_input_tokens", 0) or 0,
                output_tokens=_pu.output_tokens,
            )
        corrected = unicodedata.normalize("NFC", "".join(raw_parts))
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
    except anthropic.APIStatusError as _ae:
        _msg = (
            f"Anthropic-krediitit lopussa (402) — lisää krediittejä Consolessa"
            if _ae.status_code == 402
            else f"Anthropic API virhe {_ae.status_code}"
        )
        _LAST_TIMING["t7b_proofread_FALLBACK"] = _time.monotonic()
        _LAST_TIMING["proofread_fallback_reason"] = _msg
        print(f"[oikoluku] Varoitus: {_msg} — käytetään alkuperäistä tekstiä")
        return sections
    except anthropic.APITimeoutError:
        _LAST_TIMING["t7b_proofread_FALLBACK"] = _time.monotonic()
        _LAST_TIMING["proofread_fallback_reason"] = "aikakatkaisu (>600s)"
        print("[oikoluku] Varoitus: aikakatkaisu (>600s) — käytetään alkuperäistä tekstiä")
        return sections
    except Exception as exc:
        _LAST_TIMING["t7b_proofread_FALLBACK"] = _time.monotonic()
        _LAST_TIMING["proofread_fallback_reason"] = str(exc)
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
    # White-label branding (B2B — None means use NCE defaults)
    logo_path:                    Optional[str] = None
    footer_name:                  Optional[str] = None
    # Retrieval-tracing key (see retrieval_trace.py) — links this generation's
    # RAG retrievals + RAQS outcome for later lookup. Caller sets this to the
    # same job_id used for status polling; empty string disables tracing.
    generation_id:                str = ""

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
            "BESS (AKKUENERGIAVARASTOHANKE) — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — YLEINEN RAKENNUSOSIO "
            "[MRL 132/1999, Rakentamislaki 751/2023, MRA 895/1999]:\n"
            "Käytä tässä osiossa '1. Hankkeen kuvaus' -luvun alaotsikoiden alla.\n"
            "- Kaavoitusvaatimukset: tarkista onko hankealueella voimassa oleva "
            "asema- tai yleiskaava [MRL 132/1999, § 55–71]. Mikäli hanke sijoittuu "
            "suunnittelutarveratkaisu-alueelle, tarvitaan erillinen STR [MRL 137 §]. "
            "BESS-kontit ovat rakennusluvanvaraisia [Rakentamislaki 751/2023, 20 §]; "
            "toimita hakemus sähköisesti (Lupapiste.fi) [751/2023, 50 §].\n"
            "- Akkukonttien lukumäärä: arvioi 2–4 konttia toimittajasta riippuen "
            "(esim. CATL, BYD, Wärtsilä); yksittäisen kontin kapasiteetti 2,5–5 MWh.\n"
            "- Palokuormitusluokka: P3 (Suomen rakentamismääräyskokoelma E2) — "
            "litiumioniakkukontit kuuluvat luokkaan P3. Kirjoita YKSI selkeä kappale "
            "P3-luokasta, automaattisesta sammutusjärjestelmästä ja "
            "pelastussuunnitelmasta — ÄLÄ toista samaa asiaa useaan kertaan. "
            "Viittaa täsmälleen: 'ks. Liite 5: Paloturvallisuusselvitys'.\n"
            "- Viittaa sähköliityntäsuunnitelmaan: 'ks. Liite 8: Sähköliitynnän suunnitelma'.\n"
            "- Naapurikuuleminen: mainitse status (tehty / kesken / tulossa) "
            "[Rakentamislaki 751/2023, 44 §].\n\n"

            "KERROS 2 — YMPÄRISTÖ + BAT "
            "[YSL 527/2014, YVA-laki 252/2017, EU 684/2014]:\n"
            "Käytä osiossa '2. Perustelut' ja '3. Luvat'.\n"
            "- YVA-kynnys: BESS-hankkeelle ei ole suoraa YVA-kynnystä; arvioi "
            "kuitenkin yhdistelmähankkeen (BESS + tuuli/aurinko) YVA-velvollisuus "
            "[YVA-laki 252/2017, liite 1 — energiantuotantolaitos ≥ 50 MW tai "
            "hankkeiden yhteisvaikutus].\n"
            "- Ympäristölupa [YSL 527/2014]: vaaditaan jos hanke sijoittuu "
            "pohjavesialueelle (luokka I tai II) tai IED-kynnys (Industrial Emissions "
            "Directive 2010/75/EU) ylittyy. Muutoin ELY-keskukselle ympäristöilmoitus "
            "[YSL 63 §] vähintään 30 vrk ennen toiminnan aloittamista.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'BAT-päätelmät' (Best Available Techniques): viittaa IED:n BAT-asiakirjoihin; "
            "BESS-spesifinen BAT kattaa solujen termisen hallinnan, "
            "elektrolyytin vuoto-onnettomuussuojauksen ja EMC-suojauksen.\n"
            "- EU 684/2014 / verkkoliityntä: akkuvarasto on verkkoliityntäkoodin "
            "piirissä — noudatettava Fingrid SJV-vaatimuksia (≥ 5 MW) tai "
            "VJV-vaatimuksia (< 5 MW) [Sähkömarkkinalaki 588/2013].\n"
            "- Pohjavesi- ja maaperäriskiarvio: arvioi elektrolyytin vuoto-onnettomuuden "
            "vaikutus lähimpiin pohjavesialueisiin ja maaperään "
            "[YSL 527/2014, 16 §; Ympäristövahinkolaki 737/1994].\n\n"

            "KERROS 3 — ERITYISRISKIT / BESS-SPESIFISET:\n"
            "Käytä osiossa '1. Hankkeen kuvaus', '2. Perustelut' ja '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — mainitse tekstissä 'KTMp 335/2004': "
            "sähköturvallisuusmääräykset (KTMp 335/2004) ja sähköturvallisuuslaki "
            "1135/2016 — akkujärjestelmä tarkastettava Tukes-valtuutetun "
            "tarkastuslaitoksen toimesta ennen käyttöönottoa.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Tukes S10-ohje': palonsammutus akkuvarastoissa perustuu Tukes S10-ohjeeseen; "
            "automaattinen sammutusjärjestelmä pakollinen; "
            "sammutusaineen valinta (CO₂, inerttikaasu tai vesisumu) akkukemian mukaan.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti "
            "sanat 'ATEX-direktiivi 2014/34/EU': räjähdysvaarallisten tilojen luokittelu "
            "[ATEX-direktiivi 2014/34/EU; VNa 1439/2020] — akkutilojen vyöhykejako "
            "Zone 1 tai Zone 2 riippuu tuuletustavasta ja H₂/CO-kaasupitoisuuksista.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti "
            "sanat 'BMS' ja 'thermal runaway': "
            "thermal runaway -hallintakeinot: BMS (Battery Management System), "
            "solujen välinen terminen eristys, H₂/CO-kaasupitoisuusseuranta, "
            "automaattinen hälytys- ja sammutustoiminto.\n"
            "- IEC 62933 -sarja: IEC 62933-1 (terminologia), "
            "IEC 62933-2-1 (turvallisuusvaatimukset stationäärisille varastoille), "
            "IEC 62933-5-2 (ympäristövaatimukset) — mainitse mitkä osat soveltuvat.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti "
            "sanat 'UN 38.3': litiumakkujen kuljetustestaus (UN 38.3) — "
            "toimittajan esitettävä vaatimustenmukaisuustodistukset akkumoduulien toimitukselle.\n"
            "- Kemikaali-ilmoitusvelvollisuus Tukesille: kynnysarvo 333 kg litiumia "
            "(~1,5–2 MWh LFP-teknologialla) — arvioi kemikaaliturvallisuuslain "
            "(390/2005) mukaisen luvan tarve.\n\n"

            "KERROS 4 — VIRANOMAISLIITE [kontaktit, lomakkeet, määräajat]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- Fingrid liittymäsopimus: hae liittymispisteselvitys Fingridiltä / "
            "jakeluverkkoyhtiöltä ennen lupahakemusta [Sähkömarkkinalaki 588/2013] — "
            "käsittelyaika 2–4 kk; SJV-liittymäpyyntö Fingridiä, VJV:lle "
            "paikallinen jakeluverkkoyhtiö.\n"
            "- ELY-keskus ympäristöilmoitus [YSL 63 §]: toimita vähintään "
            "30 vuorokautta ennen toiminnan aloittamista, mikäli täyttä "
            "ympäristölupaa ei vaadita.\n"
            "- Kunnan rakennusvalvonta: hae rakentamislupa Lupapiste.fi -palvelussa "
            "[Rakentamislaki 751/2023, 50 §] — liitteiksi asemapiirros, "
            "pohjapiirros, rakennesuunnitelma ja paloturvallisuusselvitys.\n"
            "- Tukes kemikaali-ilmoitus: lomake tukes.fi — toimita "
            "30 vrk ennen toiminnan aloittamista [kemikaaliturvallisuuslaki 390/2005].\n"
            "- Pelastussuunnitelma: hyväksytettävä paikallisessa pelastuslaitoksessa "
            "[Pelastuslaki 379/2011, 15 §] ennen käyttöönottoa."
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
            "Jäähavainnointijärjestelmän kuvaus",
        ],
        "context_extra": (
            "TUULIVOIMA (MAA) — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [IEC 61400-sarja, VNa 993/1992]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Turbiinien napakorkeusarvio 150–180 m, kokonaiskorkeus (roottorin kärki) 220–250 m.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'melumallinnus': äänitehotasomittaukset IEC 61400-11 mukaan; melumallinnus "
            "ISO 9613-2-standardilla; raja-arvot 40 dB(A) yöllä asuinalueilla [VNa 993/1992].\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'varjostusvaikutus': shadow flicker -analyysi, automaattinen STF-ohjausjärjestelmä "
            "rajoittaa varjostuksen enintään 8 h/v tai 30 min/d per kohde.\n"
            "- Standardit: IEC 61400-1 (rakenteen kuormitukset) ja IEC 61400-11 (äänitehotaso).\n"
            "- Luontoselvitykset: pesimälinnusto, muuttolinnusto, lepakoiden lentoaktiviteetti "
            "— tehdään ennen YVA-ohjelman jättämistä.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'jäähavainnointi': kylmän ilmaston tuulivoimaloissa lapoihin voi kertyä jäätä "
            "(icing), joka aiheuttaa jään sinkoutumisriskin (ice throw) ympäristöön; "
            "hankkeeseen sisällytetään jäähavainnointi- ja tunnistusjärjestelmä (ice detection) "
            "sekä automaattinen pysäytys jään kertyessä, kansainvälisen IEA Wind Task 19 "
            "-ohjeistuksen (Wind Energy in Cold Climates) mukaisesti.\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YVA-laki 252/2017, Luonnonsuojelulaki 9/2023]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'YVA-kynnys': YVA-laki 252/2017 liite 1 kohta 7c — tuulivoiman YVA-kynnys on "
            "≥10 voimalaa tai napakorkeus ≥45 m; kynnyksen alittuessa YVA-harkinta "
            "tapauskohtaisesti ELY-keskuksessa.\n"
            "- Natura 2000: erityinen luonnonsuojelun arviointi [Luonnonsuojelulaki 9/2023, 69 §] "
            "jos hanke sijoittuu Natura-alueen läheisyyteen.\n"
            "- Linnusto: merikotkalle ≥2 km suojeluetäisyys (FI); DE-hankkeissa "
            "TAK-etäisyydet: Rotmilan ≥1500 m, Seeadler ≥3000 m [Helgoland-Papier].\n"
            "- Maisemavaikutukset: valokuvasovitteet ja näkyvyysanalyysi liitteiksi.\n"
            "- Jääriskin ympäristö- ja turvallisuusvaikutukset: jään sinkoutumisen "
            "turvaetäisyydet yleisiin teihin, kulkuväyliin ja rakennuksiin arvioidaan osana "
            "turvallisuusselvitystä; riskialueet merkitään varoituskyltein jääkaudella.\n"
            "- FI-erityinen: erillistä jäähavainnointia koskevaa lakipykälää ei ole — vaatimus "
            "asetetaan tyypillisesti rakennusluvan ehdoissa ja teknisenä turvallisuusvaatimuksena; "
            "Ilmatieteen laitoksen ja VTT:n jäätämistutkimusta voidaan käyttää tausta-aineistona.\n"
            "- SE-erityinen: erityistä isbildning-lainsäädäntöä ei ole tunnistettu; "
            "turvaetäisyydet ja jäähavainnointivaatimukset asetetaan yleensä osana bygglov- ja "
            "miljötillstånd-ehtoja, erityisesti teiden ja asutuksen läheisyydessä "
            "[Vaatii tarkistuksen].\n"
            "- LT-erityinen: ei tunnistettua erillistä jää-/icing-säädöstä; vaatimus "
            "todennäköisesti sisältyy statybos leidimas -ehtoihin "
            "[Vaatii tarkistuksen Liettuan lainsäädännön osalta].\n"
            "- DA-erityinen: erityistä jäähavainnointilainsäädäntöä ei ole tunnistettu; "
            "tekniset turvallisuusvaatimukset asetetaan tyypillisesti rakennusluvan ja "
            "sähköntuotantoluvan ehdoissa. Sikkerhedsstyrelsen (sähköturvallisuusviranomainen) "
            "voidaan käyttää tausta-auktoriteettina [Vaatii tarkistuksen].\n"
            "- NO-erityinen: DSB kattaa myös sähköturvallisuuden osana laajempaa "
            "siviiliturvallisuusmandaattiaan; jäähavainnointivaatimukset asetetaan "
            "tyypillisesti NVE:n konsesjonin lupaehdoissa.\n"
            "- PL-erityinen: UDT (Urząd Dozoru Technicznego) on lähin puolalainen vastine "
            "Tukesille teknisen/sähköturvallisuuden osalta; jäähavainnointivaatimukset "
            "asetetaan tyypillisesti rakennusluvan turvallisuusehdoissa "
            "[Vaatii tarkistuksen].\n"
            "- EE-erityinen: TTJA (Tehnilise Järelevalve Amet) on Viron tekninen "
            "valvontaviranomainen sähkö-/laiteturvallisuudessa; jäähavainnointivaatimukset "
            "asetetaan tyypillisesti ehitusluba-lupaehtoina [Vaatii tarkistuksen].\n"
            "- DE-erityinen: Saksassa ei ole yhtä keskitettyä Tukesia vastaavaa "
            "liittovaltiotason viranomaista — valvonta jakautuu osavaltiotason "
            "Gewerbeaufsichtsamt-viranomaisille ja sertifioiduille tarkastuslaitoksille "
            "(TÜV/DEKRA) [Vaatii tarkistuksen osavaltiotason Landesrechtin osalta].\n"
            "- LV-erityinen: VUGD on lähin Latvian turvallisuusviranomainen mutta painottuu "
            "paloturvallisuuteen, ei erityisesti jään sinkoutumisriskiin; erillistä "
            "jäähavainnointisäädöstä ei ole tunnistettu "
            "[Vaatii tarkistuksen Latvian lainsäädännön osalta].\n\n"

            "KERROS 3 — LUPAPROSESSI [MRL 132/1999 § 77a, YVA-laki 252/2017, Ilmailulaki 864/2014]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- YVA-vaiheistus: (1) YVA-ohjelma → ELY lausunto → julkinen kuuleminen 45 pv; "
            "(2) YVA-selostus → ELY perusteltu päätelmä; (3) kunta päättää kaavoittamisesta.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Osayleiskaava': MRL 132/1999 § 77a Osayleiskaava on lakisääteinen edellytys "
            "tuulivoimarakentamiselle; kulkee yleensä rinnakkain YVA-menettelyn kanssa.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti maininta "
            "lentoeste- tai ilmailuestemerkintäluvasta ja sen myöntävästä KANSALLISESTA "
            "ilmailuviranomaisesta. KÄYTÄ AINA hankkeen maan omaa viranomaista alla olevasta "
            "listasta — ÄLÄ KOSKAAN kirjoita 'Traficom' muiden kuin Suomen hankkeissa:\n"
            "- FI-erityinen: Traficom myöntää lentoestevaloluvan [Ilmailulaki 864/2014]; "
            "haettava ennen rakentamista — Finavia lausunto tarvittaessa.\n"
            "- SE-erityinen: Transportstyrelsen asettaa lentoestevalovaatimukset "
            "[TSFS 2020:88].\n"
            "- DA-erityinen: Trafikstyrelsen hyväksyy luftfartsafmærkning-merkinnän "
            "[BL 3-11].\n"
            "- NO-erityinen: Luftfartstilsynet hyväksyy merkinnän rakenteille ≥60 m.\n"
            "- PL-erityinen: ULC (Urząd Lotnictwa Cywilnego) hyväksyy oznakowanie "
            "przeszkód lotniczych -merkinnän [Dz.U. 2021 poz. 264].\n"
            "- EE-erityinen: Transpordiamet vastaa lentoestehyväksynnästä.\n"
            "- DE-erityinen: Luftfahrtbundesamt / Bundeswehr hyväksyvät merkinnän "
            "[LuftVG § 14–18.1].\n"
            "- LV-erityinen: LGS (Latvijas gaisa satiksme) hyväksyy merkinnän rakenteille "
            "≥60 m [Aviācijas likums].\n"
            "- LT-erityinen: ANCO / Civilinės aviacijos administracija hyväksyy merkinnän "
            "rakenteille ≥45 m.\n"
            "- Fingrid verkkoliityntä: SJV-liittymäpyyntö (≥5 MW) / VJV (<5 MW).\n"
            "- Jäähavainnointijärjestelmän kuvaus ja pysäytyslogiikka liitetään "
            "rakentamislupahakemukseen turvallisuusselvityksenä; vaatimus perustuu "
            "lupaehtoihin, ei erillisenä lakisääteisenä velvoitteena.\n"
            "- DE-erityinen: BImSchG-lupa Immissionsschutzbehördeltä; WindBG 2 % Flächenziel "
            "valtioiden maapinta-alasta tuulivoimalle.\n"
            "- NO-erityinen: NVE konsesjon [Energiloven § 3-1]; ennakkoneuvottelu NVE:n kanssa.\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- ELY-keskus YVA: 4–7 vuotta kokonaisaikataulu hakemuksesta rakentamislupaan.\n"
            "- Kunta osayleiskaava: 2–4 vuotta rinnakkain YVA:n kanssa.\n"
            "- Lentoestemerkinnän hyväksyntä hankkeen maan omalta ilmailuviranomaiselta "
            "(ks. KERROS 3 — ÄLÄ KOSKAAN Traficom muille kuin Suomen hankkeille; "
            "FI-hankkeissa Traficom): hae vähintään 60 pv ennen käyttöönottoa.\n"
            "- Fingrid liittymispisteselvitys: 2–4 kk käsittelyaika; hae ennen rakentamislupaa.\n"
            "- Maanvuokrasopimukset: solmittava ennen kaavatyön aloittamista."
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
            "Jäähavainnointijärjestelmän kuvaus",
        ],
        "context_extra": (
            "TUULIVOIMA (MERI/OFFSHORE) — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [IEC 61400-3-sarja, Vesilaki 587/2011]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Perustusrakenne: monopile (< 35 m syvyys), jacket (35–55 m), kelluvat rakenteet (> 55 m).\n"
            "- Merialueen olosuhteet: aallonkorkeus, jääkuorma, virtaukset — geotekninen "
            "merenpohjatutkimus pakollinen liite.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'meriekologinen vaikutusarviointi': lintujen, hylkeiden ja kalakantojen vaikutusselvitys "
            "on pakollinen liite [Luonnonsuojelulaki 9/2023; EU lintudirektiiivi 2009/147/EY].\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'vedenalainen melu': offshore-rakentamisen aikainen vedenalainen melu (pile driving) "
            "vaikuttaa kalakantoihin ja merinisäkkäisiin — arviointi IEC 62600-sarjan mukaan.\n"
            "- Merikaapelointi: HVAC (< 80 km) tai HVDC (pitkät etäisyydet) — kaapelireitti "
            "vaatii Traficomin ja Puolustusvoimien lausunnon.\n"
            "- Standardit: IEC 61400-3-1 (kiinteät rakenteet), IEC 61400-3-2 (kelluvat rakenteet).\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'jäähavainnointi': kylmän ilmaston merituulivoimaloissa lapoihin voi kertyä jäätä "
            "(icing), joka aiheuttaa jään sinkoutumisriskin (ice throw) merenkulkijoille ja "
            "huoltoliikenteelle; hankkeeseen sisällytetään jäähavainnointi- ja tunnistusjärjestelmä "
            "(ice detection) sekä automaattinen pysäytys jään kertyessä, kansainvälisen "
            "IEA Wind Task 19 -ohjeistuksen (Wind Energy in Cold Climates) mukaisesti — koskee "
            "sekä rakenteiden jääkuormaa (KERROS 1 yllä) että erillistä turvallisuusriskiä.\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YVA-laki 252/2017, Luonnonsuojelulaki 9/2023]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- YVA on AINA pakollinen offshore-tuulivoimahankkeelle "
            "[YVA-laki 252/2017, liite 1 kohta 7d].\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'Natura 2000': erityinen luonnonsuojelun arviointi [Luonnonsuojelulaki 9/2023, 69 §] "
            "jos hanke sijoittuu tai vaikuttaa Natura-alueeseen — merellä erityisesti "
            "SPA-linnustoalueet [EU lintudirektiiivi 2009/147/EY].\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'YVA-kynnys': YVA-laki 252/2017 liite 1 kohta 7d — offshore-tuulivoima ylittää aina "
            "YVA-kynnyksen; YVA-ohjelma jätettävä ELY-keskukselle ennen suunnittelun aloittamista.\n"
            "- Merialuesuunnitelma: tarkista onko alue osoitettu tuulivoima-alueeksi "
            "[Laki merialuesuunnittelusta 905/2016].\n"
            "- Puolustusvoimien tutkavaikutusarviointi: tehtävä ennen lupaprosessin aloittamista.\n"
            "- Jääriskin meriturvallisuusvaikutukset: jään sinkoutumisen turvaetäisyydet "
            "merikulkuväyliin ja huoltoreitteihin arvioidaan osana turvallisuusselvitystä; "
            "riskialueet merkitään merikarttaan ja varoitusjärjestelmällä jääkaudella.\n"
            "- FI-erityinen: erillistä jäähavainnointia koskevaa lakipykälää ei ole — vaatimus "
            "asetetaan tyypillisesti rakennus- ja vesiluvan ehdoissa; Ilmatieteen laitoksen "
            "merijäätietoa voidaan käyttää tausta-aineistona.\n"
            "- SE-erityinen: erityistä isbildning-lainsäädäntöä ei ole tunnistettu; "
            "turvaetäisyydet ja jäähavainnointivaatimukset asetetaan yleensä osana "
            "vattenverksamhet- ja miljötillstånd-ehtoja [Vaatii tarkistuksen].\n"
            "- LT-erityinen: ei tunnistettua erillistä jää-/icing-säädöstä merituulivoimalle; "
            "vaatimus todennäköisesti sisältyy statybos leidimas- ja PAV-ehtoihin "
            "[Vaatii tarkistuksen Liettuan lainsäädännön osalta].\n"
            "- DA-erityinen: erityistä jäähavainnointilainsäädäntöä ei ole tunnistettu "
            "merituulivoimalle; tekniset turvallisuusvaatimukset asetetaan tyypillisesti "
            "vesilupa- ja sähköntuotantoluvan ehdoissa. Sikkerhedsstyrelsen "
            "(sähköturvallisuusviranomainen) voidaan käyttää tausta-auktoriteettina "
            "[Vaatii tarkistuksen].\n"
            "- NO-erityinen: DSB kattaa myös sähköturvallisuuden osana laajempaa "
            "siviiliturvallisuusmandaattiaan; jäähavainnointivaatimukset asetetaan "
            "tyypillisesti NVE:n konsesjonin lupaehdoissa (koskee myös offshore-hankkeita).\n"
            "- PL-erityinen: UDT (Urząd Dozoru Technicznego) on lähin puolalainen vastine "
            "Tukesille teknisen/sähköturvallisuuden osalta; jäähavainnointivaatimukset "
            "asetetaan tyypillisesti rakennusluvan turvallisuusehdoissa "
            "[Vaatii tarkistuksen].\n"
            "- EE-erityinen: TTJA (Tehnilise Järelevalve Amet) on Viron tekninen "
            "valvontaviranomainen sähkö-/laiteturvallisuudessa; jäähavainnointivaatimukset "
            "asetetaan tyypillisesti ehitusluba-lupaehtoina merituulivoimallekin "
            "[Vaatii tarkistuksen].\n"
            "- DE-erityinen: Saksassa ei ole yhtä keskitettyä Tukesia vastaavaa "
            "liittovaltiotason viranomaista — valvonta jakautuu osavaltiotason "
            "Gewerbeaufsichtsamt-viranomaisille ja sertifioiduille tarkastuslaitoksille "
            "(TÜV/DEKRA) [Vaatii tarkistuksen osavaltiotason Landesrechtin osalta].\n"
            "- LV-erityinen: VUGD on lähin Latvian turvallisuusviranomainen mutta painottuu "
            "paloturvallisuuteen, ei erityisesti jään sinkoutumisriskiin; erillistä "
            "jäähavainnointisäädöstä ei ole tunnistettu "
            "[Vaatii tarkistuksen Latvian lainsäädännön osalta].\n\n"

            "KERROS 3 — LUPAPROSESSI [Vesilaki 587/2011, Merilaki 674/1994]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Vesilupa': Vesilaki 587/2011 § 3:2 — vesilupa Luovalta on pakollinen "
            "offshore-rakentamiselle vesistöön; hakemukseen sisällytettävä hydraulinen selvitys.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Metsähallitus': valtion merialueen käyttöoikeussopimus Metsähallituksen kanssa "
            "on edellytys hankkeelle talousvesialueella [Vesilaki 587/2011].\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Puolustusvoimien lausunto': PLM/Puolustusvoimat antavat lausunnon "
            "tutkavaikutuksista; ilman hyväksyttyä lausuntoa lupahakemus ei etene.\n"
            "- Traficom merenkulkuturvallisuus: alusliikenteen turvallisuuslupa [Merilaki 674/1994]; "
            "merikaapelit merkittävä merikarttaan, IALA-merimerkkistandardit.\n"
            "- Fingrid: suuret offshore-hankkeet liittyvät kantaverkkoon (SJV) — "
            "liittymispisteselvitys ennen lupaprosessia [Sähkömarkkinalaki 588/2013].\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- Kokonaisaikataulu 7–12 vuotta hakemuksesta käyttöönottoon.\n"
            "- Metsähallitus merialueen vuokrasopimus: neuvottelut aloitettava varhaisessa vaiheessa.\n"
            "- ELY-keskus YVA-ohjelma: jätä ennen muita lupahakemuksia — "
            "yhteysviranomainen ELY-keskus tai Luova.\n"
            "- PLM Puolustusvoimat: tutkavaikutusselvitys toimitettava heti suunnittelun alussa.\n"
            "- Traficom: merenkulkuturvallisuuslupa haettava ennen rakentamista.\n"
            "- Fingrid liittymispisteselvitys: 2–4 kk käsittelyaika; hae ennen rakentamislupaa."
        ),
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
            "Purkusuunnitelma ja maisemointivelvoite",
            "Hakijan rekisteriote",
        ],
        "context_extra": (
            "AURINKOVOIMA (PV) — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [Rakentamislaki 751/2023, IEC 61215]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Paneeliteknologia: monikidepii (mc-Si) ~20–22 % hyötysuhde; bifacial +10–15 %.\n"
            "- Suuntaus ja kallistus: etelään, 30–35 astetta optimaalinen Suomessa.\n"
            "- Invertterit: string-invertteri vs. mikroinvertterit, DC/AC-muunto, MPPT-säätö.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'häikäisyanalyysi': polarimetrinen häikäisyanalyysi naapurikiinteistöille ja "
            "liikenneväylille — selvitys on rakennusluvan pakollinen liite [Rakentamislaki "
            "751/2023, 44 §].\n"
            "- Maankäyttö: noin 1–1,5 ha/MW; agri-PV mahdollistaa maataloustoiminnan "
            "aurinkopaneelien alla.\n"
            "- Seurantajärjestelmä (tracker): yksiakselinen +15–25 %, kaksiakselinen +25–35 %.\n"
            "- Hankkeen elinkaari ja purku: aurinkovoimalan tekninen käyttöikä on tyypillisesti "
            "25–30 vuotta, jonka jälkeen laitteisto puretaan ja alue maisemoidaan ennalleen tai "
            "sovitun jälkikäytön mukaisesti. Purkusuunnitelma ja maisemointivelvoite kirjataan "
            "maanvuokrasopimukseen tai lupaehtoihin.\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YVA-laki 252/2017, YSL 527/2014]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'YVA-kynnys': YVA-laki 252/2017 liite 1 — aurinkopuistolle YVA-kynnys ≥50 ha; "
            "alle 50 ha hankkeilla ei automaattista YVA-velvollisuutta mutta "
            "tapauskohtainen harkinta ELY-keskuksessa.\n"
            "- Maisema- ja kulttuuriympäristövaikutukset: ELY-lausunto [MRL 197 §].\n"
            "- Luontoarvot: ekologiset yhteydet, mahdollinen Natura 2000 -arviointi.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'purkuvelvoite': hankkeen purkuvelvoite ja alueen maisemointi hankkeen päättyessä "
            "on keskeinen ympäristöperuste — purkukustannuksiin varaudutaan taloudellisella "
            "vakuudella (purkuvakuus), joka turvaa alueen ennallistamisen jos toiminnanharjoittaja "
            "ei täytä velvoitettaan.\n"
            "- FI-erityinen: Rakentamislaki 751/2023 ja maanvuokrasopimus määrittävät "
            "purkuvelvoitteen yksityiskohdat; erillistä valtakunnallista aurinkovoimalan "
            "purkulakia ei ole, joten velvoite perustuu sopimusehtoihin ja kunnan lupaehtoihin.\n"
            "- SE-erityinen: Miljöbalken (MB 1998:808) 2 kap. 8 § asettaa toiminnanharjoittajalle "
            "efterbehandlingsansvar (jälkihoitovelvollisuus) — alue on palautettava ennalleen "
            "toiminnan päätyttyä; suurempien laitosten kohdalla Länsstyrelsen voi vaatia "
            "taloudellisen vakuuden (ekonomisk säkerhet, MB 16 kap. 3 §) purkukustannusten "
            "kattamiseksi [Vaatii tarkistuksen tarkkojen pykälänumeroiden osalta]. "
            "Energimyndigheten nätkoncessionsansökan suurille laitoksille.\n"
            "- LT-erityinen: Liettuan lainsäädännössä ei ole tunnistettu erillistä "
            "aurinkovoimalan purku- tai maisemointisäädöstä — velvoite perustunee "
            "maanvuokrasopimukseen ja PAV-menettelyn (Poveikio aplinkai vertinimas) ehtoihin "
            "[Vaatii tarkistuksen Liettuan lainsäädännön osalta].\n"
            "- DA-erityinen: Miljøbeskyttelsesloven antaa Miljøstyrelselle valtuudet määrätä "
            "ennallistamisesta; rakennuslupa ja sähköntuotantolupa (Energistyrelsen) "
            "sisältävät tyypillisesti toiminnanharjoittajan kustannuksella tehtävän alueen "
            "ennallistamisvelvoitteen sekä Energistyrelsenin hyväksymän purkusuunnitelman "
            "vaatimuksen.\n"
            "- NO-erityinen: NVE:n myöntämä energialupa (konsesjon) sisältää vakiomuotoisen "
            "purkuvelvoitteen kehittäjälle; velvoite kytkeytyy maankäyttösuunnitteluun "
            "[Plan- og bygningsloven] ja Statsforvalterenin ympäristövalvontaan.\n"
            "- PL-erityinen: Purku- ja ennallistamisvelvoite on vakiokäytäntö lupaehdoissa ja "
            "maanvuokrasopimuksissa; sähköntuotantokonsession haltijan on toimitettava URE:lle "
            "tekninen, taloudellinen ja ympäristövaikutusraportti 18 kk ennen konsession "
            "päättymistä (kansainvälisesti ns. ARO, Asset Retirement Obligation).\n"
            "- EE-erityinen: Ei tunnistettua erillistä aurinkovoimalan purku- tai "
            "ennallistamissäädöstä — velvoite perustunee maanvuokrasopimukseen ja "
            "Keskkonnaameti valvomiin lupaehtoihin "
            "[Vaatii tarkistuksen Viron lainsäädännön osalta].\n"
            "- DE-erityinen: Rückbauverpflichtung (purkuvelvoite) on vakiintunut käytäntö — "
            "kunnat sääntelevät purkua, ennallistamista ja jatkokäyttöä "
            "kaupunkikehityssopimuksin (städtebaulicher Vertrag) osana Bebauungsplan-"
            "menettelyä; velvoite turvataan taloudellisella vakuudella (Rückbaubürgschaft). "
            "Yhtenäistä valtakunnallista standardia vakuuden tasolle ei toistaiseksi ole "
            "(Umweltbundesamt suosittelee yhtenäistämistä).\n"
            "- LV-erityinen: Ei tunnistettua erillistä aurinkovoimalan purku- tai "
            "ennallistamissäädöstä — velvoite perustunee maanvuokrasopimukseen ja "
            "VPVB/Valsts vides dienesta valvomiin lupaehtoihin "
            "[Vaatii tarkistuksen Latvian lainsäädännön osalta].\n"
            "- DE-erityinen: EEG Ausschreibung (EEG 2023) — tarjouskilpailu syöttötariffeista.\n\n"

            "KERROS 3 — LUPAPROSESSI [MRL 132/1999 § 137, YSL 527/2014]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Suunnittelutarveratkaisu': MRL 132/1999 § 137 — "
            "suunnittelutarveratkaisu vaaditaan jos hanke sijoittuu suunnittelutarvealueelle; "
            "käsittely kunnassa, naapurikuuleminen pakollinen.\n"
            "- Ympäristölupa: tarvitaan ≥1 ha tai pohjavesialueelle [YSL 527/2014].\n"
            "- Maisema- tai kulttuuriympäristölausunto: ELY-keskus, 30 pv.\n"
            "- Verkkoliityntäsopimus: jakeluverkkoyhtiö (DSO), käsittelyaika 3–6 kk.\n"
            "- Purkusuunnitelma ja maisemointivelvoite: liitetään hakemukseen alustava purku- "
            "ja maisemointisuunnitelma sekä selvitys taloudellisesta vakuudesta (purkuvakuus), "
            "erityisesti pitkäaikaisilla maanvuokrasopimuksilla (20–30 v).\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- Kunta rakennusvalvonta: ennakkoneuvottelu + rakentamislupa 6–12 kk.\n"
            "- ELY-keskus maisema: lausunto 30 pv; Natura-arviointi lisää 3–6 kk.\n"
            "- Jakeluverkkoyhtiö verkkoliityntä: liittymistilaus ensin, sopimus 3–6 kk.\n"
            "- Asukasosallistuminen: suunnittelutarveratkaisussa julkinen kuuleminen."
        ),
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
        "context_extra": (
            "SMR (PIENYDINREAKTORI) — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT JA TURVALLISUUS [STUK YVL-ohjeet, Säteilylaki 859/2018]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Reaktoriteknologia: SMR-tyyppi, lämpöteho (MWth), sähköteho (MWe); "
            "turvallisuussuunnittelun perusperiaatteet (defence-in-depth, passive safety).\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'STUK YVL-ohje': STUK:n YVL-ohjeisto on sitova sääntely; "
            "YVL A.1 (turvallisuusvaatimukset), YVL B.1 (rakenteelliset perusteet), "
            "YVL C.1 (säteilyturvallisuusanalyysi) — kaikki ohjeet soveltuvat SMR-hankkeeseen.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'IAEA SSR-2/1': Safety of Nuclear Power Plants: Design (IAEA SSR-2/1 Rev.1, 2016) "
            "on kansainvälinen turvallisuusstandardi joka STUK edellyttää noudatettavan.\n"
            "- Säteilysuoja: ALARA-periaate [Säteilylaki 859/2018] — annosrajat ympäristölle.\n"
            "- Polttoainekierto ja ydinjätehuoltosuunnitelma: esitettävä hakemuksessa.\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YVA-laki 252/2017, Ydinenergialaki 990/1987]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- YVA-menettely on AINA pakollinen ydinlaitokselle [YVA-laki 252/2017, liite 1 kohta 6].\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'Ydinenergialaki 990/1987': laki kattaa kaikki ydinlaitokset Suomessa; "
            "§ 11 periaatepäätösprosessi, § 18 rakentamislupa (STUK), § 20 käyttölupa (STUK).\n"
            "- DBA-analyysi (Design Basis Accident): radioaktiivisten päästöjen mallintaminen.\n"
            "- Jäähdytysvesi: ympäristövaikutukset vesistöön [Vesilaki 587/2011].\n"
            "- EE-erityinen: Eesti Tuumaenergia Seadus. DE-erityinen: Atomgesetz (AtG) — "
            "federal licensing. Muissa maissa: kansallinen ydinenergiasäädös soveltuu.\n\n"

            "KERROS 3 — LUPAPROSESSI [Ydinenergialaki 990/1987 § 11/18/20]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Periaatepäätös': Valtioneuvosto myöntää periaatepäätöksen [YEL § 11], "
            "eduskunta vahvistaa — tehtävä ennen kaikkia muita rakentamislupia.\n"
            "- STUK rakentamislupa [YEL § 18]: rakennussuunnitelmien STUK-hyväksyntä "
            "ennen rakentamista.\n"
            "- STUK käyttölupa [YEL § 20]: koekäyttövaihe ennen kaupallista tuotantoa.\n"
            "- IAEA safeguards-sopimus ja lisäpöytäkirja (AP): ilmoitusvelvollisuus.\n"
            "- Pre-licensing menettely STUKin kanssa: suositellaan ennen lupahakemusta.\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- TEM: ydinpolitiikan koordinaatio, periaatepäätösmenettely 2–4 vuotta.\n"
            "- STUK pre-licensing: aloita 3–5 vuotta ennen rakentamislupahakemusta.\n"
            "- ELY-keskus YVA: 3–5 vuotta, aloita rinnakkain periaatepäätösmenettelyn kanssa.\n"
            "- Kunta: asemakaavanmuutos ja rakentamislupa 2–3 vuotta.\n"
            "- Kokonaisaikataulu 10–15 vuotta: periaatepäätös → rakentamislupa → käyttölupa."
        ),
        # Later-lifecycle phases (2026-08-12, P3-3a): structural scaffold only,
        # matching BESS's context_extra_phases pattern above -- deliberately
        # empty. Real prose for "kayttolupa"/"purku" is P3-3b, content-gated
        # (native-Finnish/subject-matter reviewer sign-off required before
        # it ships, per 2026-08-11 planning). Requesting these phases today
        # raises PhaseContentNotAvailableError before this dict is ever
        # consulted -- see _PHASE_RAG_QUERIES["SMR"] above (the declared-
        # support signal) and that exception's docstring.
        "context_extra_phases": {},
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
        "context_extra": (
            "VESIVOIMA — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [Vesilaki 587/2011, Patoturvallisuuslaki 494/2009]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Voimalaitostyyppi: virtavesilaitos (juoksutus) vs. säätövoimalaitos (pato + allas).\n"
            "- Hydraulinen mitoitus: virtaama (m³/s), putouskorkeus (m), "
            "teho P = η × ρ × g × Q × H.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'ekologinen virtaama': ekologinen minimivirtaama on ylläpidettävä kalakannalle ja "
            "vesiekosysteemille [Vesilaki 587/2011; Kalastuslaki 379/2015] — arvo määritetään "
            "ELY-keskuksen kanssa lupaprosessissa.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'patoturvallisuus': pato luokitellaan P1 (suurin vaara) tai P2/P3 "
            "[Patoturvallisuuslaki 494/2009, 10 §]; P1-padolle pakollinen "
            "vahingonvaaraselvitys ja hätätilanneohjeet.\n"
            "- Kalaväylä/kalatiesuunnitelma: kalojen vaelluseste on poistettava tai kiertotie "
            "rakennettava [Kalastuslaki 379/2015, 51 §; EU vesipuitedirektiivi 2000/60/EY].\n"
            "- Generaattori → muuntaja → jakeluverkko; DSO-liittymissopimus ennen rakentamista.\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YVA-laki 252/2017, YSL 527/2014]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'YVA-kynnys': YVA-laki 252/2017 liite 1 kohta 8a — vesivoimala on YVA-velvollinen "
            "jos sähköteho ≥ 10 MW tai padottavan vesialtaan pinta-ala ≥ 1 km².\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'vesipuitedirektiivi': EU vesipuitedirektiivin [2000/60/EY] hyvä ekologinen tila "
            "on vesistön tilatavoite — vesivoiman ekologiset vaikutukset arvioidaan suhteessa "
            "direktiivin vaatimuksiin.\n"
            "- Ekologiset vaikutukset: kalakannat (vaelluskalatutkimus), pohjaeläimet, "
            "vesikasvillisuus, tulvariski ylä- ja alapuolisessa vesistössä.\n"
            "- Natura 2000: erityinen arviointi [Luonnonsuojelulaki 9/2023, 69 §] jos hanke "
            "vaikuttaa Natura-alueeseen.\n\n"

            "KERROS 3 — LUPAPROSESSI [Vesilaki 587/2011, Kalastuslaki 379/2015]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Vesilaki 587/2011': Luovalta haetaan vesilupa [Vesilaki 587/2011 § 3:2] "
            "padotukseen ja vesirakentamiseen; luvan käsittely 2–5 vuotta; "
            "kalastuskunta- ja osakaskuntakuulemiset pakollisia.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Patoturvallisuuslaki 494/2009': P1-pato vaatii ELY-keskuksen "
            "patoturvallisuusvalvontaa; vahingonvaaraselvitys ja hätätilanneohjeet "
            "[Patoturvallisuuslaki 494/2009, 10 §] toimitettava lupahakemuksen liitteenä.\n"
            "- Kalatalousmaksu: korvauksena kalakannoille aiheutuvasta haitasta "
            "[Kalastuslaki 379/2015].\n"
            "- Ympäristölupa: tarvitaan jos hanke vaikuttaa pohjavesiin tai aiheuttaa "
            "merkittäviä ympäristöhaittoja [YSL 527/2014].\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- Luova vesilupa: 2–5 vuotta käsittelyaika — pisin yksittäinen lupaprosessi.\n"
            "- ELY-keskus YVA: 3–5 vuotta rinnakkain vesilupamenettelyn kanssa.\n"
            "- ELY-keskus patoturvallisuus: ilmoitus P1-padolle rakentamisen alkaessa.\n"
            "- Kalastuslaki-ilmoitus ELY-keskukselle: 60 pv ennen rakentamista.\n"
            "- Kalatutkimuslaitos: vaelluskalatutkimus tilattava vähintään 2 vuotta "
            "ennen hakemusta.\n"
            "- Kokonaisaikataulu 5–10 vuotta."
        ),
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
        "context_extra": (
            "HYBRIDIVOIMALA (BESS + TUULI/AURINKO) — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [IEC 61400-sarja, IEC 62933-sarja, Rakentamislaki 751/2023]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Komponentit: tuulivoima (MW-lkm × yksikkökoko) + aurinkopaneelit (MWp) + "
            "BESS (MW/MWh) — integroitu verkkoliityntä yhdestä liittymispisteestä.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'energiavarastomitoitus': BESS-kapasiteetti (MWh) ja -teho (MW) mitoitetaan "
            "tuuli-/aurinkoprofiilin tasoittamiseen; mitoitusperusteet esitettävä [IEC 62933-2-1].\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'thermal runaway' ja 'BMS': BESS-osahankkeen paloturvallisuus — "
            "Battery Management System (BMS), terminen eristys, H₂/CO-seuranta, "
            "automaattinen sammutus [Tukes S10-ohje; NFPA 855].\n"
            "- Tuulivoimalakomponentti: napakorkeusarvio, melumalli IEC 61400-11, "
            "varjostusvaikutusanalyysi.\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YVA-laki 252/2017, YSL 527/2014]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'YVA-kynnys': hybridihankkeen yhteisvaikutus arvioidaan YVA-laki 252/2017 "
            "liite 1 mukaan — tuulivoiman YVA-kynnys ≥ 10 voimalaa tai napakorkeus ≥ 45 m; "
            "BESS- ja PV-komponenttien yhteisvaikutus voi laukaista YVA-velvollisuuden.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'BAT-päätelmät': BESS-komponentin ympäristöluvassa sovellettava BAT "
            "(Best Available Techniques) solujen termiselle hallinnalle, "
            "elektrolyytin vuoto-onnettomuussuojaukselle ja sammutusvesilinjastolle [YSL 527/2014].\n"
            "- Linnusto: sama suojeluetäisyysvaatimus kuin erilliselle tuulivoimahankkeelle.\n\n"

            "KERROS 3 — LUPAPROSESSI [MRL 132/1999 § 77a, Pelastuslaki 379/2011]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Osayleiskaava': tuulivoimakomponentti edellyttää MRL 132/1999 § 77a mukaisen "
            "osayleiskaavan; kulkee rinnakkain YVA-menettelyn kanssa.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Tukes S10-ohje': BESS-komponentin palonsammutus perustuu Tukes S10-ohjeeseen; "
            "automaattinen sammutusjärjestelmä pakollinen; pelastussuunnitelma hyväksytettävä "
            "pelastuslaitoksessa [Pelastuslaki 379/2011, 15 §].\n"
            "- Traficom lentoestevalolupa [Ilmailulaki 864/2014]: haettava ennen rakentamista.\n"
            "- Fingrid SJV-liittymäpyyntö (≥ 5 MW yhdistetty teho): 2–4 kk käsittelyaika.\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- ELY-keskus YVA: 4–7 vuotta; aloita ensin ennen muita lupaprosesseja.\n"
            "- Kunta osayleiskaava: 2–4 vuotta rinnakkain YVA:n kanssa.\n"
            "- Tukes kemikaali-ilmoitus (BESS): 30 pv ennen toiminnan aloittamista.\n"
            "- Traficom lentoestevalolupa: hae ≥ 60 pv ennen käyttöönottoa.\n"
            "- Fingrid liittymispisteselvitys: 2–4 kk; hae ennen rakentamislupahakemusta.\n"
            "- Kokonaisaikataulu 4–8 vuotta."
        ),
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
        "context_extra": (
            "SMR + BESS (HYBRIDIENERGIAJÄRJESTELMÄ) — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [STUK YVL-ohjeet, IEC 62933-sarja, Säteilylaki 859/2018]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- SMR-komponentti: reaktorityyppi, lämpöteho (MWth), sähköteho (MWe); "
            "turvallisuusperiaatteet (defence-in-depth, passive safety).\n"
            "- BESS-komponentti: kapasiteetti (MWh), teho (MW); "
            "integraatio SMR:n tehonsäätelyyn.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'STUK YVL-ohje': STUK:n YVL-ohjeisto on sitova sääntely; "
            "YVL A.1 (turvallisuusvaatimukset), YVL B.1 (rakenteelliset perusteet), "
            "YVL C.1 (säteilyturvallisuusanalyysi) — kaikki soveltuvat SMR-komponenttiin.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'thermal runaway' ja 'BMS': BESS-osahankkeen paloturvallisuus — "
            "Battery Management System (BMS), terminen eristys, H₂/CO-seuranta, "
            "automaattinen sammutus [Tukes S10-ohje; NFPA 855].\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YVA-laki 252/2017, Ydinenergialaki 990/1987]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- YVA on AINA pakollinen ydinlaitokselle [YVA-laki 252/2017, liite 1 kohta 6].\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'Ydinenergialaki 990/1987': laki kattaa kaikki ydinlaitokset Suomessa; "
            "§ 11 periaatepäätösprosessi (VN), § 18 rakentamislupa (STUK), "
            "§ 20 käyttölupa (STUK).\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'BAT-päätelmät': BESS-komponentin ympäristöluvassa sovellettava BAT "
            "(Best Available Techniques) solujen termiselle hallinnalle, "
            "elektrolyytin vuoto-onnettomuussuojaukselle ja sammutusvesilinjastolle "
            "[YSL 527/2014].\n"
            "- DBA-analyysi (Design Basis Accident): radioaktiivisten päästöjen mallintaminen.\n\n"

            "KERROS 3 — LUPAPROSESSI [Ydinenergialaki 990/1987 § 11/18/20, Pelastuslaki 379/2011]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Periaatepäätös': Valtioneuvosto myöntää periaatepäätöksen [YEL § 11], "
            "eduskunta vahvistaa — tehtävä ennen kaikkia muita rakentamislupia; "
            "2–4 vuotta prosessointiaikaa.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Tukes S10-ohje': BESS-komponentin palonsammutus perustuu Tukes S10-ohjeeseen; "
            "automaattinen sammutusjärjestelmä pakollinen; pelastussuunnitelma "
            "hyväksytettävä pelastuslaitoksessa [Pelastuslaki 379/2011, 15 §].\n"
            "- STUK rakentamislupa [YEL § 18] ja käyttölupa [YEL § 20] SMR-komponentille.\n"
            "- Fingrid SJV-liittymäpyyntö koko yhdistetyn tehon osalta.\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- TEM: ydinpolitiikan koordinaatio, periaatepäätösmenettely 2–4 vuotta.\n"
            "- STUK pre-licensing: aloita 3–5 vuotta ennen rakentamislupahakemusta.\n"
            "- ELY-keskus YVA: 3–5 vuotta rinnakkain periaatepäätösmenettelyn kanssa.\n"
            "- Tukes kemikaali-ilmoitus (BESS): 30 pv ennen toiminnan aloittamista.\n"
            "- Kokonaisaikataulu 10–15 vuotta."
        ),
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
        "context_extra": (
            "YMPÄRISTÖLUPAHAKEMUS — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TOIMINNAN KUVAUS [YSL 527/2014, YVA-laki 252/2017]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Toiminnan kuvaus: mitä tehdään, missä, milloin; "
            "prosessikaavio ainevirroista (raaka-aineet, tuotteet, päästöt, jätteet).\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'toimintakuvaus ja prosessikaavio': ympäristölupahakemus edellyttää "
            "yksityiskohtaisen toimintakuvauksen ja prosessikaavioa "
            "[YSL 527/2014, 39 §]; kuvaa tuotantokapasiteetti, "
            "käytettävät kemikaalit ja raaka-aineet.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'päästöluettelo': hakemukseen sisällytettävä päästöluettelo "
            "ilmaan (g/s, t/a), vesistöön (mg/l) ja maaperään; "
            "mittaustulokset tai laskennalliset arviot [YSL 527/2014, 39 §].\n"
            "- Naapurikuuleminen: Luova tai kunta kuulee naapurit ja sidosryhmät.\n\n"

            "KERROS 2 — YMPÄRISTÖVAIKUTUKSET + BAT [YSL 527/2014, IED 2010/75/EU]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'BAT-päätelmät': Best Available Techniques (BAT) on ympäristöluvan "
            "keskeinen vaatimus [IED-direktiivi 2010/75/EU; YSL 527/2014] — "
            "toiminnanharjoittajan on noudatettava sovellettavia BAT-päätelmiä "
            "päästöille, energiatehokkuudelle ja jätehuollolle.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'YVA-menettely': tarkista onko toiminnalle tehty YVA [YVA-laki 252/2017]; "
            "YVA-selostus tai perustelut soveltumattomuudesta on liitettävä hakemukseen.\n"
            "- Pohjavesi- ja maaperävaikutukset: erityinen selvitys pohjavesialueilla.\n"
            "- Melu, tärinä ja haju: selvitys lähikiinteistöihin kohdistuvista haitoista.\n\n"

            "KERROS 3 — LUPAPROSESSI [YSL 527/2014, Vesilaki 587/2011]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Lupa- ja valvontavirasto': Luova (Lupa- ja valvontavirasto) käsittelee "
            "suurten laitosten ympäristöluvat [YSL 527/2014] — "
            "käsittelyaika 6–18 kk; kuulemisaika 30 pv naapureille.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'rekisteröinti-ilmoitus': pienet, vähäriskiset toiminnot voidaan "
            "rekisteröidä [YSL 527/2014, 10 §] kunnalle ilman varsinaista "
            "lupamenettelyä — selvitä soveltuvuus ennen hakemusta.\n"
            "- Vesilupa [Vesilaki 587/2011]: tarvitaan jos toiminta vaikuttaa vesistöön.\n"
            "- Jätehuoltosuunnitelma: liitettävä hakemukseen [Jätelaki 646/2011].\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- Luova (isot hankkeet): 6–18 kk käsittelyaika.\n"
            "- Kunta (pienet hankkeet): 3–6 kk käsittelyaika.\n"
            "- Ennakkoneuvottelu Luovan tai kunnan ympäristönsuojeluviranomaisella "
            "suositellaan ennen hakemuksen jättämistä.\n"
            "- ELY-keskus lausunnot: 30–60 pv kuulemisaika.\n"
            "- Tarkkailuohjelma: hyväksytettävä Luovalla tai kunnalla luvan myöntämisen "
            "jälkeen ennen toiminnan aloittamista."
        ),
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
            "DATAKESKUS — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [Rakentamislaki 751/2023, EU 2023/1791 energiatehokkuusasetus]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Tekniset parametrit: IT-kuorma (käytä annettua teho-arvoa MW); "
            "arvioitu kokonaiskulutus = IT-kuorma × PUE-arvo.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'PUE': Power Usage Effectiveness (PUE) on datakeskuksen energiatehokkuuden "
            "keskeinen mittari; PUE 1,2–1,4 on hyvä taso — EU:n energiatehokkuusasetus "
            "[EU 2023/1791] edellyttää PUE-raportoinnin suurilta datakeskuksilta.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'Free Cooling': ulkoilmajäähdytys (Free Cooling) hyödyntää Suomen kylmää ilmastoa "
            "— ei mekaanista jäähdytystä tarvita suuren osan vuodesta; "
            "hukkalämpö ~25–35 °C voidaan palauttaa kaukolämpöverkkoon.\n"
            "- Sijaintiedut: teollisuusalue, liikenneyhteydet, olemassa oleva infrastruktuuri.\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YVA-laki 252/2017, YSL 527/2014]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'YVA-kynnys': datakeskus ei automaattisesti ylitä YVA-kynnysarvoa "
            "[YVA-laki 252/2017 liite 1], mutta tapauskohtainen harkinta tehdään "
            "ELY-keskuksessa suuren sähkötehon (≥ 50 MW) tai merkittävien "
            "ympäristövaikutusten perusteella.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'hukkalämmön hyödyntäminen': hukkalämpö kaukolämpöverkkoon on konkreettinen "
            "hiilineutraaliushyöty ja keskeinen ympäristöperustelu hankkeelle.\n"
            "- Digitalisaation kasvava kapasiteettitarve Suomessa ja Euroopassa.\n"
            "- Meluselvitys: jäähdytys- ja aggregaattimelu — naapurikuuleminen pakollinen.\n\n"

            "KERROS 3 — LUPAPROSESSI [MRL 132/1999, Rakentamislaki 751/2023]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'asemakaavanmuutos': datakeskus vaatii usein asemakaavanmuutoksen "
            "[MRL 132/1999] jos nykyinen kaava ei salli käyttötarkoitusta — "
            "käsittely kunnassa ja ELY-keskuksessa 1–3 vuotta.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Fingrid': verkkoliityntä — jakeluverkkoliittymä (< 110 kV) DSO:n kanssa; "
            "mahdollinen kantaverkkoliittymä (≥ 110 kV) Fingrid Oyj:n kanssa "
            "[Sähkömarkkinalaki 588/2013]; liittymispisteselvitys ensin.\n"
            "- Ympäristölupa [YSL 527/2014]: tarvitaan jäähdytysveden käytölle tai "
            "merkittävälle melulle.\n"
            "- Rakentamislupa: haettava Lupapiste.fi-palvelusta [Rakentamislaki 751/2023, 50 §].\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "Seuraavat toimenpiteet — 6 vaihetta:\n"
            "1. Ennakkoneuvottelu rakennusvalvonta – Lupakonsultti / NCE – 1–2 vk\n"
            "2. Asemakaavanmuutos-selvitys – Projektipäällikkö / NCE – 1–3 kk\n"
            "3. YVA-harkinta ELY-keskuksen kanssa – Lupakonsultti / NCE – 2–4 kk\n"
            "4. Verkkoliittymäneuvottelu DSO + Fingrid – IT-arkkitehti / Hakija – 3–6 kk\n"
            "5. Rakentamislupahakemus – Lupakonsultti / NCE – 6–12 kk\n"
            "6. Ympäristölupahakemus (jäähdytys ja melu) – Lupakonsultti / NCE – 6–12 kk"
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
    "asuinrakennus": {
        "nimi_fi":    "Asuinrakennushanke",
        "lyhenne":    "AR",
        "kasittelyaika": {"FI": "1–3 kk", "EN": "1–3 months"},
        "rag_queries": [
            "asuinrakennus rakentamislupa rakennusvalvonta kaavoitus",
            "asunto rakentaminen suunnittelutarveratkaisu naapurikuuleminen",
            "asuinrakennus paloturvallisuus vastaava työnjohtaja katselmus",
        ],
        "luvat": [
            ("Rakentamislupa",                        "Kunta / rakennusvalvonta",    "Rakentamislaki 751/2023"),
            ("Naapurikuuleminen",                     "Kunta / hakija",             "Rakentamislaki 751/2023, 44 §"),
            ("Suunnittelutarveratkaisu (tarvitt.)",   "Kunta",                       "MRL 132/1999 § 137"),
            ("Toimenpidelupa (tarvitt.)",             "Kunta / rakennusvalvonta",    "Rakentamislaki 751/2023"),
        ],
        "laki_extra": [
            "Maankäyttö- ja rakennuslaki 132/1999 (kaavoitus)",
            "Asunto-osakeyhtiölaki 1599/2009 (tarvitt.)",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Asemapiirustus ja pohjakartta (M 1:500)",
            "Rakennuspiirustukset (pohja-, julkisivu- ja leikkauspiirustukset)",
            "Energiaselvitys (RakMk D3 / Rakentamislaki 751/2023)",
            "Paloturvallisuusselvitys (paloluokka P1/P2/P3)",
            "Rakennesuunnitelma (pääpiirustukset)",
            "Vastaavan työnjohtajan nimitysasiakirja",
            "Hakijan rekisteriote tai henkilötiedot",
        ],
        "context_extra": (
            "ASUINRAKENNUS — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [Rakentamislaki 751/2023, Suomen rakentamismääräyskokoelma]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Rakennuksen tiedot: kerrosala (m²), kerrosluku, rakennusmateriaalit, "
            "lämmitysjärjestelmä, liittyminen kunnan verkostoihin.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'energiaselvitys': asuinrakennukselle on laadittava energiaselvitys "
            "[Rakentamislaki 751/2023; YM asetus 1010/2017] — E-luku (kWh/m²/a) "
            "osoittaa rakennuksen energiatehokkuusluokan.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'vastaava työnjohtaja': vastaava työnjohtaja nimettävä ja rakennusvalvonnan "
            "hyväksyttävä ennen aloitusilmoitusta [Rakentamislaki 751/2023, 68 §].\n"
            "- Paloturvallisuus: paloluokka P1, P2 tai P3 "
            "[Suomen rakentamismääräyskokoelma E1].\n\n"

            "KERROS 2 — YMPÄRISTÖ + KAAVA [MRL 132/1999, Rakentamislaki 751/2023]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'kaavamääräykset': tarkista asema- tai yleiskaavan rakennusoikeus, "
            "kerrosluku ja käyttötarkoitusmerkintä [MRL 132/1999] — rakentamislupa "
            "ei voi ylittää kaavan sallimia rajoja.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'suunnittelutarveratkaisu': jos hanke sijoittuu suunnittelutarvealueelle, "
            "vaaditaan STR [MRL 132/1999 § 137] ennen rakentamislupaa — "
            "julkinen kuuleminen ja naapurikuuleminen pakollisia.\n"
            "- Pohjavesi- ja tulvavaara-alueet: tarkista SYKE:n paikkatietopalvelusta.\n\n"

            "KERROS 3 — LUPAPROSESSI [Rakentamislaki 751/2023]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Rakentamislaki 751/2023': rakentamislupa haetaan Lupapiste.fi-palvelussa; "
            "liitteiksi asemapiirros, pohjapiirros, julkisivupiirros, leikkauspiirros, "
            "energiaselvitys ja vastaavan työnjohtajan nimitys.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'loppukatselmus': rakennusvalvonta suorittaa loppukatselmuksen ennen "
            "asunnon käyttöönottoa [Rakentamislaki 751/2023, 87 §]; "
            "pohjakatselmus, runkokatselmus ja loppukatselmus.\n"
            "- Naapurikuuleminen: hakija tai kunta kuulee naapurit "
            "[Rakentamislaki 751/2023, 44 §].\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- Kunta rakennusvalvonta: ennakkoneuvottelu + rakentamislupa 1–3 kk.\n"
            "- Aloitusilmoitus ennen töiden aloittamista; katselmusaikataulu sovittava.\n"
            "- Energiatodistus: laadittava ennen käyttöönottoa [Laki 50/2013].\n"
            "- Kokonaisaikataulu rakentamisluvasta käyttöönottoon 6–18 kk."
        ),
    },
    "liikerakennus": {
        "nimi_fi":    "Liikerakennushanke",
        "lyhenne":    "LR",
        "kasittelyaika": {"FI": "1–6 kk", "EN": "1–6 months"},
        "rag_queries": [
            "liikerakennus kauppa toimisto rakentamislupa kaavoitus Suomi",
            "liikerakennus paloturvallisuus pelastuslaitos tarkastus",
            "kaupallinen rakennus energiaselvitys esteettömyys vastaava työnjohtaja",
        ],
        "luvat": [
            ("Rakentamislupa",                       "Kunta / rakennusvalvonta",    "Rakentamislaki 751/2023"),
            ("Naapurikuuleminen",                    "Kunta / hakija",             "Rakentamislaki 751/2023, 44 §"),
            ("Suunnittelutarveratkaisu (tarvitt.)",  "Kunta",                       "MRL 132/1999 § 137"),
            ("Asemakaavanmuutos (tarvitt.)",          "Kunta + ELY-keskus",          "MRL 132/1999"),
            ("Pelastussuunnitelma / lausunto",       "Paikallinen pelastuslaitos",  "Pelastuslaki 379/2011, 15 §"),
        ],
        "laki_extra": [
            "Maankäyttö- ja rakennuslaki 132/1999 (kaavoitus)",
            "Esteettömyysvaatimukset (YM asetus 241/2017)",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Asemapiirustus ja pohjakartta (M 1:500)",
            "Rakennuspiirustukset (pohja-, julkisivu- ja leikkauspiirustukset)",
            "Paloturvallisuusselvitys ja pelastussuunnitelma",
            "Energiaselvitys (E-luku)",
            "Esteettömyysselvitys (YM asetus 241/2017)",
            "Vastaavan työnjohtajan nimitysasiakirja",
            "Hakijan rekisteriote",
        ],
        "context_extra": (
            "LIIKERAKENNUS — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [Rakentamislaki 751/2023, Suomen rakentamismääräyskokoelma]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Rakennuksen tiedot: käyttötarkoitus (kauppa, toimisto, palvelu), "
            "kerrosala (m²), kerrosluku, asiakaskapasiteetti, pysäköinti.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'paloturvallisuusselvitys': liikerakennus vaatii paloturvallisuusselvityksen "
            "[Suomen rakentamismääräyskokoelma E1]; paloluokka P1/P2/P3, "
            "sprinklerijärjestelmä (yleensä pakollinen yli 1000 m² liiketilalle), "
            "pelastussuunnitelma ja poistumistiet.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'esteettömyys': liikerakennus on oltava esteettömästi saavutettavissa "
            "[YM asetus 241/2017; Rakentamislaki 751/2023] — invapaikkamäärä, "
            "luiskat, hissit, wc-tilat.\n"
            "- Energiaselvitys: E-luku ja energialuokka.\n\n"

            "KERROS 2 — YMPÄRISTÖ + KAAVA [MRL 132/1999]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'kaavamääräykset': asemakaava määrittää käyttötarkoituksen (K = kauppa, "
            "A = asunto, T = teollisuus) — liikerakennus edellyttää "
            "oikeaa kaavamerkintää [MRL 132/1999]; asemakaavanmuutos tarvittaessa.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'liikenneselvitys': suuret liikerakennukset (yleensä yli 2000 m² tai "
            "merkittävä liikennemäärä) edellyttävät liikenneselvitystä "
            "pysäköinnistä, asiakkaiden kulkureiteistä ja liikenneturvallisuudesta.\n\n"

            "KERROS 3 — LUPAPROSESSI [Rakentamislaki 751/2023, Pelastuslaki 379/2011]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Rakentamislaki 751/2023': rakentamislupa haetaan Lupapiste.fi-palvelussa; "
            "liitteiksi piirustukset, paloturvallisuusselvitys, esteettömyysselvitys, "
            "energiaselvitys ja vastaavan työnjohtajan nimitys.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Pelastuslaitos': pelastuslaitoksen tarkastus ja pelastussuunnitelman "
            "hyväksyntä ovat pakollisia ennen käyttöönottoa [Pelastuslaki 379/2011, 15 §]; "
            "sammutuskalusto, paloilmoitinjärjestelmä ja poistumistiesuunnitelma.\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- Kunta rakennusvalvonta: ennakkoneuvottelu + rakentamislupa 1–6 kk.\n"
            "- Pelastuslaitos: ennakkolausunto paloturvallisuudesta.\n"
            "- Asemakaavanmuutos (tarvittaessa): 1–3 vuotta.\n"
            "- Kokonaisaikataulu rakentamisluvasta käyttöönottoon 6–24 kk."
        ),
    },
    "maatalous": {
        "nimi_fi":    "Maatalousrakennushanke",
        "lyhenne":    "MAT",
        "kasittelyaika": {"FI": "1–6 kk", "EN": "1–6 months"},
        "rag_queries": [
            "maatalousrakennus navetta kasvihuone rakentamislupa ympäristölupa",
            "kotieläintalous ympäristölupa lantala etäisyysvaatimukset Suomi",
            "maatilarakennus pohjavesi nitraattiasetus Luova ELY-keskus",
        ],
        "luvat": [
            ("Rakentamislupa",                            "Kunta / rakennusvalvonta",    "Rakentamislaki 751/2023"),
            ("Naapurikuuleminen",                         "Kunta / hakija",             "Rakentamislaki 751/2023, 44 §"),
            ("Ympäristölupa (suuri kotieläintalous)",     "Lupa- ja valvontavirasto",   "YSL 527/2014"),
            ("Ympäristölupailmoitus (tarvitt.)",          "Kunta",                       "YSL 527/2014, 10 §"),
            ("Maaseutuviraston tukikelpoisuustarkistus",  "Ruokavirasto / ELY-keskus",  "EU:n maataloustukiasetus"),
        ],
        "laki_extra": [
            "Nitraattiasetus 1250/2014 (lantavarastointi)",
            "Luonnonsuojelulaki 9/2023",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Asemapiirustus ja pohjakartta (M 1:500)",
            "Rakennuspiirustukset (pohja-, julkisivu- ja leikkauspiirustukset)",
            "Lantavaraston mitoituslaskelma ja sijoitussuunnitelma",
            "Ympäristölupahakemus (suuri kotieläintalous)",
            "Pohjavesialueen selvitys (SYKE)",
            "Vastaavan työnjohtajan nimitysasiakirja",
            "Hakijan rekisteriote tai henkilötiedot",
        ],
        "context_extra": (
            "MAATALOUSRAKENNUS — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [Rakentamislaki 751/2023, Nitraattiasetus 1250/2014]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Rakennustyyppi: navetta, kasvihuone, lato, kuivuri, kasvisten varastointitila.\n"
            "- Kotieläintalous: eläinmäärä (eläinyksiköt) ja lantalan mitoitus.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'lantavaraston mitoitus': lantavarasto on mitoitettava vähintään 12 kuukauden "
            "lantamäärälle [Nitraattiasetus 1250/2014, 5 §]; etäisyys vesistöistä "
            "vähintään 10–100 m riippuen varaston tyypistä.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'vastaava työnjohtaja': vastaavan työnjohtajan nimeäminen ja rakennusvalvonnan "
            "hyväksyntä pakollinen [Rakentamislaki 751/2023, 68 §].\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YSL 527/2014, Nitraattiasetus 1250/2014]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'ympäristölupa': suuri kotieläintalous vaatii ympäristöluvan Luovalta "
            "[YSL 527/2014, liite 1] — kynnysarvot: nautoja ≥ 75, sikoja ≥ 210, "
            "kanoja ≥ 10 000; alle kynnyksen: ympäristöilmoitus kunnalle.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'nitraattiasetus': nitraattiasetuksen [1250/2014] vaatimukset koskevat "
            "lannan levitysalueita, lantavaraston etäisyyksiä vesistöistä ja "
            "levityskieltoja keväällä ja syksyllä.\n"
            "- Natura 2000 ja luonnonsuojelukohteet: tarkista sijainti ennen lupahakemusta.\n\n"

            "KERROS 3 — LUPAPROSESSI [Rakentamislaki 751/2023, YSL 527/2014]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Rakentamislaki 751/2023': rakentamislupa haetaan Lupapiste.fi-palvelussa; "
            "liitteiksi piirustukset, lantavaraston mitoituslaskelma, pohjavesiselvitys "
            "ja naapurikuuleminen.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Ruokavirasto': EU:n maataloustuet (CAP-tuet) — tarkista tukikelpoisuus "
            "ELY-keskuksesta tai Ruokavirastosta ennen rakentamisen aloittamista; "
            "tuettavan rakennuksen on täytettävä tukiehdot.\n"
            "- ELY-keskus ympäristöilmoitus tai Luova ympäristölupa: "
            "hae rinnakkain rakentamislupaprosessin kanssa.\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- Kunta rakennusvalvonta: ennakkoneuvottelu + rakentamislupa 1–3 kk.\n"
            "- Luova ympäristölupa (suuri kotieläintalous): 6–18 kk.\n"
            "- Ruokavirasto / ELY tukikelpoisuustarkistus: ennen hankintasopimuksia.\n"
            "- Nitraattiasetus: toimita lantavaraston sijoitussuunnitelma ELY-keskukselle.\n"
            "- Kokonaisaikataulu 1–6 kk rakentamisluvalle, 12–18 kk ympäristöluvalle."
        ),
    },
    "teollisuus": {
        "nimi_fi":    "Teollisuusrakennushanke",
        "lyhenne":    "TEOL",
        "kasittelyaika": {"FI": "3–12 kk", "EN": "3–12 months"},
        "rag_queries": [
            "teollisuusrakennus rakentamislupa ympäristölupa kaavoitus Suomi",
            "teollisuuslaitos kemikaalilupa Tukes ympäristövaikutukset",
            "teollisuus paloturvallisuus rakentamismääräykset vastaava työnjohtaja",
        ],
        "luvat": [
            ("Rakentamislupa",                        "Kunta / rakennusvalvonta",    "Rakentamislaki 751/2023"),
            ("Naapurikuuleminen",                     "Kunta / hakija",             "Rakentamislaki 751/2023, 44 §"),
            ("Asemakaavanmuutos (tarvitt.)",           "Kunta + ELY-keskus",          "MRL 132/1999"),
            ("Ympäristölupa (tarvitt.)",              "Lupa- ja valvontavirasto",    "YSL 527/2014"),
            ("Kemikaalilupa tai -ilmoitus (tarvitt.)", "Tukes",                       "Kemikaaliturvallisuuslaki 390/2005"),
            ("Pelastussuunnitelma / lausunto",        "Paikallinen pelastuslaitos",  "Pelastuslaki 379/2011, 15 §"),
            ("Verkkoliityntäsopimus (tarvitt.)",      "Turku Energia / DSO",         "Sähkömarkkinalaki 588/2013"),
        ],
        "laki_extra": [
            "YVA-laki 252/2017 (kynnyksen ylittyessä)",
            "Luonnonsuojelulaki 9/2023",
            "Jätelaki 646/2011",
        ],
        "liitteet": [
            "Sijaintikartta (M 1:20 000 tai laajempi)",
            "Maankäyttöselvitys PDF (NCE)",
            "Asemapiirustus ja pohjakartta (M 1:500)",
            "Rakennussuunnitelmat (rakennepiirrokset, julkisivut, leikkaukset)",
            "Paloturvallisuusselvitys (rakennusluokka P1/P2/P3)",
            "Ympäristövaikutusten selvitys (tarvittaessa)",
            "Kemikaalilupahakemus / ilmoitus (Tukes, tarvittaessa)",
            "Meluselvitys (jos sijoittuu lähelle asutusta)",
            "Vastaavan työnjohtajan nimitysasiakirja",
            "Hakijan rekisteriote",
        ],
        "context_extra": (
            "TEOLLISUUSRAKENNUS — NELJÄKERROSRAKENNE:\n\n"

            "KERROS 1 — TEKNISET TIEDOT [Rakentamislaki 751/2023, Suomen rakentamismääräyskokoelma]:\n"
            "Käytä osiossa '1. Hankkeen kuvaus'.\n"
            "- Hankkeen kuvaus: käyttötarkoitus (kevyt/raskas teollisuus), kerrosala (m²), "
            "korkeus, rakentamismateriaalit ja rakenneratkaisu.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'rakennuksen paloluokka': teollisuusrakennus kuuluu paloluokkaan P1, P2 tai P3 "
            "[Suomen rakentamismääräyskokoelma E1] käyttötarkoituksen, kerrosalan ja "
            "henkilömäärän perusteella — paloluokka määrittää rakenteelliset palovaatimukset.\n"
            "- PAKOLLINEN osiossa '1. Hankkeen kuvaus' — kirjoita tekstiin pakollisesti sanat "
            "'vastaava työnjohtaja': vastaavan työnjohtajan nimeäminen ja rakennusvalvonnan "
            "hyväksyntä on pakollista ennen aloitusilmoituksen jättämistä "
            "[Rakentamislaki 751/2023, 68 §].\n"
            "- Kemikaalivarastointi: arvioi kemikaalien määrät ja luokitus — "
            "Tukesin kemikaali-ilmoituksen tai -luvan tarve.\n\n"

            "KERROS 2 — YMPÄRISTÖ + YVA [YVA-laki 252/2017, YSL 527/2014]:\n"
            "Käytä osiossa '2. Perustelut'.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'YVA-kynnys': YVA-laki 252/2017 liite 1 — teollisuuslaitos on YVA-velvollinen "
            "jos se ylittää liitteen 1 kynnysarvot (esim. metalli-, kemian- tai "
            "elintarviketeollisuuden laitoksille määritellyt kapasiteettirajat); "
            "tapauskohtainen harkinta ELY-keskuksessa.\n"
            "- PAKOLLINEN osiossa '2. Perustelut' — kirjoita tekstiin pakollisesti sanat "
            "'ympäristölupa': ympäristölupa [YSL 527/2014] tarvitaan jos laitos aiheuttaa "
            "merkittäviä päästöjä ilmaan, vesistöön tai maaperään tai sijoittuu "
            "pohjavesialueelle; IED-direktiivi 2010/75/EU soveltuu suurille laitoksille.\n"
            "- Melu- ja tärinäselvitys: teollisuuslaitos lähellä asutusta vaatii "
            "melumittaukset ja arvioinnin [VNa 993/1992].\n\n"

            "KERROS 3 — LUPAPROSESSI [Rakentamislaki 751/2023, Kemikaaliturvallisuuslaki 390/2005]:\n"
            "Käytä osiossa '3. Luvat'.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Rakentamislaki 751/2023': rakentamislupa haetaan Lupapiste.fi-palvelussa "
            "[Rakentamislaki 751/2023, 50 §]; liitteiksi asemapiirros, pohjapiirros, "
            "rakennesuunnitelma, paloturvallisuusselvitys ja vastaavan työnjohtajan nimitys.\n"
            "- PAKOLLINEN osiossa '3. Luvat' — kirjoita tekstiin pakollisesti sanat "
            "'Tukes': kemikaaliturvallisuuslain [390/2005] mukainen Tukesin lupa tai "
            "ilmoitus tarvitaan vaarallisten kemikaalien valmistukseen tai varastointiin "
            "kynnysarvojen ylittyessä; lisäksi turvallisuusselvitys suurille laitoksille.\n"
            "- Asemakaavanmuutos: tarvitaan jos alue ei ole asemakaavoitettu "
            "teollisuuskäyttöön [MRL 132/1999] — käsittelyaika kunnassa 1–3 vuotta.\n\n"

            "KERROS 4 — VIRANOMAISLIITE [aikataulut, kontaktit]:\n"
            "Käytä osiossa '3. Luvat' ja '6. Seuraavat toimenpiteet'.\n"
            "- Kunta rakennusvalvonta: ennakkoneuvottelu + rakentamislupa 3–12 kk.\n"
            "- Tukes kemikaali-ilmoitus tai -lupa: toimita 30–60 pv ennen toiminnan aloittamista.\n"
            "- ELY-keskus YVA-harkinta: 2–4 kk; aloita ennen rakentamislupahakemusta.\n"
            "- Luova ympäristölupa: 6–18 kk; hae rinnakkain rakentamislupaprosessin kanssa.\n"
            "- Pelastussuunnitelma: hyväksytettävä pelastuslaitoksessa ennen käyttöönottoa.\n"
            "- Kokonaisaikataulu 3–12 kk rakentamisluvalle, 12–24 kk ympäristöluvalle."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# RAG-haku
# ─────────────────────────────────────────────────────────────────────────────
from source_policy import is_chunk_relevant as _is_chunk_relevant
import retrieval_trace as _retrieval_trace

def _apply_source_cap(
    candidates: list[tuple],
    top_n: int,
    cap: int,
) -> list[tuple]:
    """Per-source cap with backfill (2026-08-12, coverage remediation ranking
    fix). `candidates` is a list of tuples whose first element is distance
    (ascending = more relevant) and last element is the source id; any middle
    elements are carried through unchanged. Returns at most `top_n` tuples.

    See _rag_context()'s call site for the full reasoning on cap=8. Extracted
    as its own function so it's directly unit-testable, independent of the
    embedding/ChromaDB machinery around it.

    Fixes crowding-out (one oversized source taking most of the slots), not
    sparsity (if the candidate pool itself only has a few distinct sources,
    this cannot manufacture more). A no-op whenever len(candidates) <= top_n.
    """
    ordered = sorted(candidates, key=lambda x: x[0])
    selected: list[tuple] = []
    per_source_count: dict[str, int] = {}
    deferred: list[tuple] = []
    for cand in ordered:
        if len(selected) >= top_n:
            break
        src = cand[-1]
        if per_source_count.get(src, 0) < cap:
            selected.append(cand)
            per_source_count[src] = per_source_count.get(src, 0) + 1
        else:
            deferred.append(cand)
    if len(selected) < top_n:
        for cand in deferred:
            if len(selected) >= top_n:
                break
            selected.append(cand)
    return selected


def _humanize_source_id(raw: str) -> str:
    """Turn a raw ChromaDB source stem (the ingestion-time filename, e.g.
    "lion_2025_bess") into a generic, human-readable display title (e.g.
    "Lion 2025 Bess") -- a snake_case-to-Title-Case pass, nothing source-
    specific.

    2026-08-24: found this was the direct root cause of a real, systemic
    RAQS false-positive pattern (the "epävarmuus" criterion, a pure Claude
    judgment call with no deterministic matching anywhere in the pipeline,
    was comparing the human-formatted in-text citation Claude itself wrote
    -- e.g. "[Lion 2025 BESS, s. 20]" -- against this exact raw slug shown
    unchanged in the source list it's given, and understandably concluding
    they didn't match) -- and the SAME raw, un-humanized value was also
    rendering directly into the customer-facing PDF's own "Lähteet ja
    tietolähteet" section (both read sites use the same `display` field on
    the same `sources` list; fixed once here, both close).

    Deliberately generic rather than a per-source manual title table --
    with 8,000+ chunks across hundreds of sources in 9 countries, a manual
    table doesn't scale as a first pass. This will render acronyms
    (YSL, YVA, YM, ...) in Title Case rather than their correct all-caps
    form (e.g. "Ysl 527 2014" not "YSL 527 2014") -- an accepted,
    explicitly-scoped tradeoff; a manual override table for specific
    sources that don't humanize well can be added incrementally later,
    as they're actually noticed, not speculatively now.
    """
    if not raw:
        return raw
    words = [w for w in re.split(r"[_\-]+", raw) if w]
    return " ".join(w.capitalize() for w in words) or raw


def _rag_context(
    hanketyyppi: str,
    country: str = "FI",
    n_per_query: int = 2,
    generation_id: str = "",
    vaihe: str = "",
) -> tuple[str, list[dict], bool, list[str], list[str]]:
    """Hae relevantit dokumenttichunkit.

    Jos country != 'FI', haetaan ensin maakohtaiset dokumentit ja täydennetään
    FI-dokumenteilla (suomalainen lainsäädäntö on aina relevanttia kontekstia).
    Graceful fallback: jos metadata-suodatus epäonnistuu, haetaan ilman suodatinta.

    `vaihe` (2026-08-12, P3-3a): if _PHASE_RAG_QUERIES has an entry for
    (hanketyyppi, vaihe.lower()), those queries are collected ADDITIONALLY
    to the normal Step 1/2 queries below (never a replacement) -- see
    _PHASE_RAG_QUERIES's own comment. Empty string (default) or any
    hanketyyppi/vaihe combo not in that dict is a complete no-op, byte-
    identical to calling this function without the parameter at all.

    Palauttaa (context_text, sources, warning_flag, precedent_chunks, precedent_sources).
    Nostaa InsufficientSourcesError jos RAG-laatu on riittämätön (hard stop).

    `generation_id`, if given, links this call to a `retrieval_trace` row (see
    retrieval_trace.py) — purely additive logging, does not change retrieval
    behaviour. Empty string (default) disables tracing.

    Cost & resource guardrail (TASO 1): raises GenerationCapError if this
    generation_id has already made `_RAG_ITERATION_CAP` retrieval calls — checked
    outside the try/except below so it can never be silently swallowed into an
    empty-context fallback (that would let generation continue with zero RAG
    content, exactly the "silently return wrong output" this guardrail exists
    to prevent).
    """
    if generation_id:
        _n_retr = _retrieval_trace.count_retrieval_calls(generation_id)
        if _n_retr >= _RAG_ITERATION_CAP:
            _retrieval_trace.log_guardrail_hit(
                generation_id=generation_id,
                guard_type="retrieval_iteration_cap",
                count_at_trip=_n_retr,
                cap=_RAG_ITERATION_CAP,
                detail=f"hanketyyppi={hanketyyppi} country={country}",
            )
            raise GenerationCapError("retrieval_iteration", generation_id, _n_retr, _RAG_ITERATION_CAP)

    cfg = _HANKE_CFG[hanketyyppi]
    try:
        embed_model = _get_embed_model()
        col         = _get_chroma_col()

        seen_ids:        set[str]        = set()
        all_docs:        list[str]       = []
        all_distances:   list[float]     = []
        all_chunk_ids:   list[str]       = []
        all_source_types: list[str]      = []
        all_source_ids:  list[str]       = []  # src_id per chunk, parallel to all_docs — used by the per-source cap below
        all_source_meta: dict[str, dict] = {}  # src_id → {display, url}

        def _collect(results: dict, allowed_countries=None) -> None:
            docs      = results["documents"][0]
            ids       = results["ids"][0]
            metas     = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            if not metas:
                metas = [{}] * len(ids)
            if not distances:
                distances = [0.5] * len(ids)
            for doc, id_, meta, dist in zip(docs, ids, metas, distances):
                meta = meta or {}
                if allowed_countries is not None:
                    if meta.get("country", "") not in allowed_countries:
                        continue
                # Cross-type filter via source_policy (single source of truth)
                if not _is_chunk_relevant(meta, hanketyyppi):
                    continue
                _src_name = meta.get("source", "")
                if id_ not in seen_ids:
                    seen_ids.add(id_)
                    all_docs.append(doc)
                    all_distances.append(dist)
                    all_chunk_ids.append(id_)
                    all_source_types.append(meta.get("source_type") or "unknown")
                    # Use meta["source"] as dedup key so all chunks from the same
                    # document collapse into one source entry (fixes caruna repeating)
                    src_id = _src_name or re.sub(r"[_-]\d+$", "", id_)
                    all_source_ids.append(src_id)
                    if src_id not in all_source_meta:
                        all_source_meta[src_id] = {
                            "display": _humanize_source_id(_src_name or src_id),
                            "url":     meta.get("url"),
                        }

        # Country-specific queries for step 1 — avoids Finnish cross-lingual distance penalty.
        # Falls back to Finnish cfg queries if no override defined for this country+type.
        _country_queries = (
            _COUNTRY_RAG_QUERIES.get(country, {}).get(hanketyyppi)
            or cfg["rag_queries"]
        )

        # Oversample before post-filtering: ChromaDB where= operator is unreliable in v1.5.x
        # (raises ValueError for all operators). Post-filter via allowed_countries in _collect()
        # is the real guard — fetch a large pool first, then keep only permitted countries.
        # v2 now has 7231 chunks across 8 countries. Small countries (DA=265, DE=60, EE=79)
        # need a large pool to appear in the top-k after post-filtering. 500 covers even DE
        # (60/7231 = 0.8%) reliably when combined with native-language queries in Step 1.
        _n_oversample = max(500, n_per_query * 80)
        # FI reports get FI+EU chunks; all other countries get only their own + EU chunks
        # so Finnish-jurisdiction chunks (ELY-keskus, Tukes, YVA-laki, Fingrid) cannot
        # contaminate authority tables or citations in DE/EE/SE/etc. reports.
        _allowed_countries = {"FI", "EU"} if country == "FI" else {country, "EU"}

        # Step 1: country-specific retrieval using native-language queries (once, before FI loop)
        if country != "FI":
            try:
                for cq in _country_queries:
                    cemb = embed_model.encode([cq]).tolist()
                    _collect(
                        col.query(query_embeddings=cemb, n_results=_n_oversample),
                        allowed_countries=_allowed_countries,
                    )
            except Exception:
                pass  # maakohtaisia dokumentteja ei vielä indeksoitu

        # Step 1.5: phase-aware retrieval (2026-08-12, P3-3a). ADDITIVE to
        # Step 1/2, not a replacement -- a phase-4/5 request should still
        # retrieve base pre-license context alongside phase-specific later-
        # lifecycle content, not lose one for the other. No-op for every
        # hanketyyppi/vaihe combo not explicitly listed in
        # _PHASE_RAG_QUERIES (confirmed in this PR's self-test: zero effect
        # on phases 1-3 or any non-SMR hanketyyppi).
        _phase_queries = _PHASE_RAG_QUERIES.get(hanketyyppi, {}).get((vaihe or "").lower(), [])
        for pq in _phase_queries:
            try:
                pemb = embed_model.encode([pq]).tolist()
                _collect(
                    col.query(query_embeddings=pemb, n_results=_n_oversample),
                    allowed_countries=_allowed_countries,
                )
            except Exception:
                pass

        # Step 2: Finnish cfg queries for base context; post-filter enforces country scope.
        for q in cfg["rag_queries"]:
            emb = embed_model.encode([q]).tolist()
            try:
                _collect(
                    col.query(query_embeddings=emb, n_results=_n_oversample),
                    allowed_countries=_allowed_countries,
                )
            except Exception:
                pass

        # Sort by relevance and keep only the top-N chunks for scoring + context.
        # n_oversample=500 is large for discovery (ensures small countries like DA/DE appear),
        # but averaging over 400+ marginally-relevant chunks would dilute avg_score below threshold.
        # Top-50 keeps the most relevant subset, matching what Claude can attend to effectively.
        _top_n_ctx = max(50, n_per_query * 20)
        if len(all_docs) > _top_n_ctx:
            # Per-source cap (2026-08-12, coverage remediation ranking fix): a
            # pure global sort let one oversized document (e.g. Poland's
            # 205-chunk energy-law statute, Norway's 463-chunk statnett_nvf_2025,
            # Sweden's 531-chunk national energy/climate plan) crowd out
            # smaller, correctly-tagged, genuinely on-topic sources -- up to
            # 84% of a real top-50 window from a single document in the worst
            # cases found. CAP=8 reasoning (real corpus stats, 2026-08-11):
            # median source size across the whole corpus is 6 chunks, 58.4% of
            # all 226 sources have <=8 chunks (fully represented, zero
            # truncation, under this cap) -- cap=8 targets exactly the small
            # minority of oversized outliers (39 sources >50 chunks, 23 >100)
            # without touching normally-sized sources at all. See
            # _apply_source_cap()'s docstring for what this does and doesn't fix.
            _selected = _apply_source_cap(
                list(zip(all_distances, all_docs, all_chunk_ids, all_source_types, all_source_ids)),
                top_n=_top_n_ctx,
                cap=8,
            )
            all_distances    = [p[0] for p in _selected]
            all_docs         = [p[1] for p in _selected]
            all_chunk_ids    = [p[2] for p in _selected]
            all_source_types = [p[3] for p in _selected]

        # ── Task 2: RAG confidence check ─────────────────────────────────────
        chunks_returned = len(all_docs)
        if all_distances:
            relevance_scores = [max(0.0, 1.0 - d) for d in all_distances]
            avg_score = sum(relevance_scores) / len(relevance_scores)
        else:
            avg_score = 0.0

        # Non-FI countries mix cross-lingual chunks (lower cosine sim by nature) with FI base chunks,
        # so the blended avg is inherently ~0.04-0.06 lower than a pure-FI retrieval.
        # Very small collections (EU=586, EE=79, DE=60 chunks) pull the avg further down even after
        # top-50 trimming; 0.52 covers them while still blocking truly irrelevant context (< 0.50).
        # LV exception: Latvian text scores ~0.05 lower than other non-FI languages with mpnet
        # (BESS avg=0.51, aurinkovoima avg=0.49 on 157 genuine LV chunks). Use 0.46 floor for LV.
        _min_score = 0.46 if country == "LV" else (0.52 if country != "FI" else 0.65)

        # ── Retrieval trace (TASO 1, purely additive) — log what was retrieved for
        # this call regardless of pass/fail, so a RAG_FAIL is debuggable too. Never
        # raises and never blocks generation; see retrieval_trace.py.
        try:
            _all_queries_used = list(dict.fromkeys(list(_country_queries) + list(cfg["rag_queries"])))
            _retrieval_trace.log_retrieval(
                generation_id=generation_id,
                query_text=" || ".join(_all_queries_used),
                country=country,
                hanketyyppi_tag=hanketyyppi,
                chunks=[
                    {"chunk_id": cid, "score": sc, "source_type": st}
                    for cid, sc, st in zip(all_chunk_ids, relevance_scores if all_distances else [], all_source_types)
                ],
                avg_score=avg_score,
                min_score_threshold=_min_score,
            )
        except Exception:
            pass

        if chunks_returned < 5 or avg_score < _min_score:
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
        # Oversample + Python post-filter (same pattern as _collect(), commit 3b486dab)
        # instead of where= — this block was the one place left still using ChromaDB's
        # where= operator directly after 3b486dab migrated Steps 1/2 off it.
        # Query with _country_queries[0] (native-language, country-aware — same list
        # used in Step 1) instead of a Finnish-only query, so non-FI precedent retrieval
        # doesn't pay the cross-lingual penalty against a foreign-language pool.
        # Primary: doc_type="case_law" chunks in the allowed country pool. Fallback:
        # no case_law content for this country yet — closest regulatory chunks in the
        # same pool serve as de-facto precedent.
        precedent_chunks:  list[str] = []
        precedent_sources: list[str] = []
        if _country_queries:
            try:
                prec_emb = embed_model.encode([_country_queries[0]]).tolist()
                prec_results = col.query(query_embeddings=prec_emb, n_results=_n_oversample)
                prec_docs  = prec_results["documents"][0] if prec_results["documents"] else []
                prec_metas = (prec_results.get("metadatas") or [[]])[0]
                if not prec_metas:
                    prec_metas = [{}] * len(prec_docs)

                pool = [
                    (doc, meta or {}) for doc, meta in zip(prec_docs, prec_metas)
                    if (meta or {}).get("country", "") in _allowed_countries
                ]
                picked = [p for p in pool if p[1].get("doc_type") == "case_law"][:5]
                if not picked:
                    picked = pool[:5]

                for doc, meta in picked:
                    precedent_chunks.append(doc)
                    precedent_sources.append(meta.get("source", "Viranomainen"))
            except Exception:
                pass

        context = "\n\n---\n\n".join(all_docs)
        sources = [{"id": sid, **info} for sid, info in sorted(all_source_meta.items())]

        # Rollback checkpoint (TASO 1): only on the success path — a RAG_FAIL
        # already halts the pipeline via InsufficientSourcesError, so there is
        # nothing to checkpoint/resume for a failed retrieval. Chunk IDs only,
        # same "no full text" rule as log_retrieval() — a resume would re-fetch
        # from ChromaDB by id, not from a duplicated text copy here.
        if generation_id:
            _retrieval_trace.save_checkpoint(
                generation_id=generation_id,
                step="retrieval",
                state={
                    "chunk_ids":  all_chunk_ids,
                    "source_types": all_source_types,
                    "avg_score":  round(avg_score, 4),
                    "country":    country,
                    "hanketyyppi_tag": hanketyyppi,
                },
            )

        return context, sources, warning_flag, precedent_chunks, precedent_sources, round(avg_score, 3)

    except InsufficientSourcesError:
        raise
    except Exception as exc:
        print(f"[RAG] Haku epäonnistui ({exc}) — jatketaan ilman kontekstia")
        return "", [], False, [], [], 0.0


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
                       "PL": "System magazynowania energii w akumulatorach (BESS)",
                       "DE": "Batteriespeichersystem (BESS)",                 "ET": "Akupatarei energiasalvestussüsteem (BESS)",
                       "LV": "Bateriju energijas uzkrāšanas sistēma (BESS)",  "LT": "Baterijų energijos kaupimo sistema (BESS)"},
    "tuulivoima_maa": {"EN": "Onshore Wind Power Project",                   "SE": "Landbaserat vindkraftsprojekt",
                       "DA": "Landbaseret vindkraftsprojekt",                  "NO": "Landbasert vindkraftprosjekt",
                       "PL": "Lądowy projekt farmy wiatrowej",
                       "DE": "Onshore-Windenergieprojekt",                    "ET": "Maismaa tuuleenergiaprojekt",
                       "LV": "Sauszemes vēja enerģijas projekts",             "LT": "Sausumos vėjo energijos projektas"},
    "tuulivoima_meri":{"EN": "Offshore Wind Power Project",                  "SE": "Offshorevindkraftsprojekt",
                       "DA": "Offshore-vindkraftsprojekt",                    "NO": "Offshore-vindkraftprosjekt",
                       "PL": "Morski projekt farmy wiatrowej",
                       "DE": "Offshore-Windenergieprojekt",                   "ET": "Meretuuleenergia projekt",
                       "LV": "Jūras vēja enerģijas projekts",                 "LT": "Jūrinis vėjo energijos projektas"},
    "aurinkovoima":   {"EN": "Solar Power Plant Project",                    "SE": "Solkraftsprojekt",
                       "DA": "Solkraftværksprojekt",                          "NO": "Solkraftverksprosjekt",
                       "PL": "Projekt elektrowni słonecznej",
                       "DE": "Solarkraftwerksprojekt",                        "ET": "Päikeseelektrijaama projekt",
                       "LV": "Saules elektrostacijas projekts",               "LT": "Saulės elektrinės projektas"},
    "SMR":            {"EN": "Small Modular Reactor (SMR) — pre-licensing",  "SE": "Liten modulär reaktor (SMR) — förlicensiering",
                       "DA": "Lille modulær reaktor (SMR) — forhåndslicensiering", "NO": "Liten modulær reaktor (SMR) — forhåndslisensering",
                       "PL": "Mały reaktor modułowy (SMR) — wstępne licencjonowanie",
                       "DE": "Kleiner modularer Reaktor (SMR) — Vorgenehmigungsverfahren",
                       "ET": "Väike moodulreaktor (SMR) — eelloa taotlus",
                       "LV": "Mazais moduļreaktors (SMR) — iepriekšējā licencēšana",
                       "LT": "Mažasis modulinis reaktorius (SMR) — išankstinis licencijavimas"},
    "vesivoima":      {"EN": "Hydroelectric Power Project",                  "SE": "Vattenkraftsprojekt",
                       "DA": "Vandkraftsprojekt",                             "NO": "Vannkraftprosjekt",
                       "PL": "Projekt elektrowni wodnej",
                       "DE": "Wasserkraftprojekt",                            "ET": "Hüdroelektrijaama projekt",
                       "LV": "Hidroelektrostacijas projekts",                 "LT": "Hidroelektrinės projektas"},
    "smr_bess":       {"EN": "SMR + BESS Hybrid Energy System",              "SE": "SMR + BESS hybridsystem",
                       "DA": "SMR + BESS hybridsystem",                       "NO": "SMR + BESS hybridsystem",
                       "PL": "System hybrydowy SMR + BESS",
                       "DE": "SMR + BESS Hybrid-Energiesystem",               "ET": "SMR + BESS hübriidenergiasüsteem",
                       "LV": "SMR + BESS hibrīda energosistēma",              "LT": "SMR + BESS hibridinė energetikos sistema"},
    "asuinrakennus":  {"EN": "Residential Construction Permit Application",   "SE": "Bygglovsansökan för bostadsbyggnad",
                       "DA": "Byggetilladelsesansøgning for beboelsesbygning", "NO": "Byggetillatelsessøknad for boligbygg",
                       "PL": "Wniosek o pozwolenie na budowę budynku mieszkalnego",
                       "DE": "Baugenehmigungsantrag für Wohngebäude",         "ET": "Elamu ehitusloa taotlus",
                       "LV": "Dzīvojamās ēkas būvatļaujas pieteikums",        "LT": "Gyvenamojo pastato statybos leidimo paraiška"},
    "teollisuus":     {"EN": "Industrial Construction Permit Application",    "SE": "Bygglovsansökan för industribyggnad",
                       "DA": "Byggetilladelsesansøgning for industribygning",  "NO": "Byggetillatelsessøknad för industribygg",
                       "PL": "Wniosek o pozwolenie na budowę budynku przemysłowego",
                       "DE": "Baugenehmigungsantrag für Industriegebäude",    "ET": "Tööstushoone ehitusloa taotlus",
                       "LV": "Rūpnieciskās ēkas būvatļaujas pieteikums",      "LT": "Pramoninio pastato statybos leidimo paraiška"},
    "maatalous":      {"EN": "Agricultural Construction Permit Application",  "SE": "Bygglovsansökan för lantbruksbyggnad",
                       "DA": "Byggetilladelsesansøgning for landbrugsbygning", "NO": "Byggetillatelsessøknad for landbruksbygg",
                       "PL": "Wniosek o pozwolenie na budowę budynku rolniczego",
                       "DE": "Baugenehmigungsantrag für landwirtschaftliches Gebäude",
                       "ET": "Põllumajandushoone ehitusloa taotlus",
                       "LV": "Lauksaimniecības ēkas būvatļaujas pieteikums",  "LT": "Žemės ūkio pastato statybos leidimo paraiška"},
    "liikerakennus":  {"EN": "Commercial Construction Permit Application",    "SE": "Bygglovsansökan för affärsbyggnad",
                       "DA": "Byggetilladelsesansøgning for erhvervsbygning",  "NO": "Byggetillatelsessøknad for næringsbygg",
                       "PL": "Wniosek o pozwolenie na budowę budynku handlowego",
                       "DE": "Baugenehmigungsantrag für Gewerbegebäude",      "ET": "Äriehitise ehitusloa taotlus",
                       "LV": "Komerciālās ēkas būvatļaujas pieteikums",       "LT": "Komercinio pastato statybos leidimo paraiška"},
    "muu":            {"EN": "Other Project Permit Application",             "SE": "Tillståndsansökan för annat projekt",
                       "DA": "Tilladelsesansøgning for andet projekt",         "NO": "Tillatelsessøknad for annet prosjekt",
                       "PL": "Wniosek o zezwolenie na inny projekt",
                       "DE": "Genehmigungsantrag für sonstiges Projekt",      "ET": "Muu projekti loataotlus",
                       "LV": "Cita projekta atļaujas pieteikums",             "LT": "Kito projekto leidimo paraiška"},
    "ymparistolupa":  {"EN": "Environmental Permit Application (YSL 527/2014)", "SE": "Miljötillståndsansökan",
                       "DA": "Miljøtilladelsesansøgning",                      "NO": "Søknad om miljøtillatelse",
                       "PL": "Wniosek o pozwolenie środowiskowe",
                       "DE": "Umweltgenehmigungsantrag",                      "ET": "Keskkonnaloa taotlus",
                       "LV": "Vides atļaujas pieteikums",                     "LT": "Aplinkosaugos leidimo paraiška"},
    "datakeskus":     {"EN": "Data Centre Permit Application",                  "SE": "Tillståndsansökan för datacenter",
                       "DA": "Tilladelsesansøgning for datacenter",              "NO": "Tillatelsessøknad for datasenter",
                       "PL": "Wniosek o zezwolenie na centrum danych",
                       "DE": "Genehmigungsantrag für Rechenzentrum",          "ET": "Andmekeskuse loataotlus",
                       "LV": "Datu centra atļaujas pieteikums",               "LT": "Duomenų centro leidimo paraiška"},
    "hybridi":        {"EN": "Hybrid Power Plant Project (BESS + Wind/Solar)", "SE": "Hybridkraftverksprojekt (BESS + vind/sol)",
                       "DA": "Hybridkraftværksprojekt (BESS + vind/sol)",      "NO": "Hybridkraftverksprosjekt (BESS + vind/sol)",
                       "PL": "Projekt hybrydowej elektrowni (BESS + wiatr/słońce)",
                       "DE": "Hybridkraftwerksprojekt (BESS + Wind/Solar)",    "ET": "Hübriidelektrijaama projekt (BESS + tuul/päike)",
                       "LV": "Hibrīda elektrostacijas projekts (BESS + vējš/saule)",
                       "LT": "Hibridinės elektrinės projektas (BESS + vėjas/saulė)"},
}
# offshore_wind is the same real-world project concept as tuulivoima_meri (the
# JS form config aliases HANKETYYPIT.offshore_wind = HANKETYYPIT.tuulivoima_meri
# for the same reason) — reuse its translations rather than duplicating them.
_HANKE_NIMI_TRANS["offshore_wind"] = _HANKE_NIMI_TRANS["tuulivoima_meri"]
# The six smr_XX country variants (smr_se/no/da/de/ee/lv) share the identical
# nimi_fi ("Pienydinreaktori (SMR) — ennakkolupahakemus") as the base "SMR"
# hanketyyppi — they are the same project concept, just pre-scoped to a given
# country, so they reuse SMR's translations rather than duplicating them.
for _smr_country in ("se", "no", "da", "de", "ee", "lv"):
    _HANKE_NIMI_TRANS[f"smr_{_smr_country}"] = _HANKE_NIMI_TRANS["SMR"]
del _smr_country

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
        "authorities": ["Länsstyrelsen", "Energimyndigheten", "Boverket", "Mark- och miljödomstolen", "Naturvårdsverket",
                         "Svenska kraftnät (TSO — transmission grid connection, equivalent to Finnish Fingrid)",
                         "Elsäkerhetsverket (Swedish National Electrical Safety Board — electrical installation safety, equivalent to Finnish Tukes)"],
        "key_laws": ["Plan- och bygglagen (PBL 2010:900)", "Miljöbalken (MB 1998:808)", "Ellagen (1997:857)",
                     "Miljöprövningsförordningen (2013:251)", "Lag (2003:778) om skydd mot olyckor (LSO)",
                     "Lag (1984:3) om kärnteknisk verksamhet", "Förordning (2014:214) om dammsäkerhet",
                     "Lag (2010:1011) om brandfarliga och explosiva varor (LBE)"],
        # 2026-08-23 (item 5(B), SE research — see SE_LAW_CITATION_RESEARCH.md):
        # the previous version of this prefix already correctly listed Sweden's
        # real laws (PBL, Miljöbalken) but didn't say which specific Finnish
        # statute each one replaces -- the model had the right raw material in
        # its own context but no explicit mapping, which likely explains why it
        # kept inventing plausible-sounding-but-fake Swedish law names
        # ("Miljöskyddslagen", "MKB-lagen") instead of using the real ones
        # already given to it. This version adds the explicit per-statute
        # mapping (verified against Riksdagen/Boverket/MSB/Naturvårdsverket,
        # not recalled) and an explicit instruction not to invent names.
        # Prompt-level complement to the _fix_hardcoded_law_citations()
        # backstop above -- this reduces how often that backstop needs to fire
        # at all; the backstop remains the safety net for whatever still slips
        # through (same RAG-leak risk as Traficom/STUK/ELY-keskus).
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in SWEDEN. Apply Swedish regulatory framework:\n"
            "Key authorities: Länsstyrelsen (county board), Energimyndigheten (energy agency), "
            "Boverket (building standards), Mark- och miljödomstolen (environmental court), "
            "Naturvårdsverket (environmental protection), "
            "Svenska kraftnät (TSO — transmission grid connection = nätanslutningsavtal; "
            "equivalent to Finnish Fingrid), "
            "Elsäkerhetsverket (electrical installation safety authority; equivalent to Finnish Tukes), "
            "Transportstyrelsen (Swedish Transport Agency — aviation obstacle lighting for wind "
            "turbines per TSFS 2020:88; equivalent to Finnish Traficom), "
            "Strålsäkerhetsmyndigheten SSM (nuclear/radiation safety authority; equivalent to Finnish STUK), "
            "MSB — Myndigheten för samhällsskydd och beredskap (fire/explosion safety, civil protection).\n"
            "Key laws — use EXACTLY these, mapped to the Finnish statute each one replaces:\n"
            "- Plan- och bygglagen PBL 2010:900 = building permits AND zoning (Bygglov/detaljplan) — "
            "replaces BOTH Finnish MRL 132/1999 and Rakentamislaki 751/2023, since Sweden does not "
            "split these into two separate acts the way Finland's 2023 reform did.\n"
            "- Miljöbalken MB 1998:808 kap. 6 (miljöbedömningar) = environmental impact assessment — "
            "replaces Finnish YVA-laki 252/2017.\n"
            "- Miljöbalken MB 1998:808 kap. 9 (miljöfarlig verksamhet) = environmental permit — "
            "replaces Finnish YSL 527/2014.\n"
            "- Miljöbalken MB 1998:808 kap. 11 (vattenverksamhet) = water operations permit — "
            "replaces Finnish Vesilaki 587/2011.\n"
            "- Miljöprövningsförordningen 2013:251 = classification of which activities need a permit "
            "and at what level (complements MB kap. 9, does not replace it).\n"
            "- Lag (2003:778) om skydd mot olyckor (LSO) = rescue services / fire safety — "
            "replaces Finnish Pelastuslaki 379/2011.\n"
            "- Lag (2010:1011) om brandfarliga och explosiva varor (LBE) = flammable/explosive goods "
            "handling permit, administered by MSB — relevant for BESS fire/explosion safety.\n"
            "- Lag (1984:3) om kärnteknisk verksamhet = nuclear technical activities, reviewed by SSM — "
            "replaces Finnish Ydinenergialaki 990/1987.\n"
            "- Förordning (2014:214) om dammsäkerhet = dam safety, supervised by Länsstyrelsen — "
            "replaces Finnish Patoturvallisuuslaki 494/2009.\n"
            "- Ellagen 1997:857 = grid connection. TSFS 2020:88 = aviation obstacle lighting.\n"
            "- For onshore/offshore wind specifically: Miljöbalken 16 kap. 4 § gives the host "
            "municipality an unconditional veto over a wind power permit (no justification required, "
            "not appealable) — this has no Finnish equivalent and materially affects approval "
            "likelihood; mention it where relevant.\n"
            "CRITICAL: Never cite a Finnish statute number (MRL, YSL, YVA-laki, Rakentamislaki, Vesilaki, "
            "Pelastuslaki, Ydinenergialaki, Patoturvallisuuslaki) directly in the output — always use the "
            "correct Swedish law from the mapping above instead, and never invent a plausible-sounding "
            "Swedish law name that is not in the list above.\n"
            "Replace ALL Finnish entity references (ELY-keskus, Tukes, Traficom, Fingrid, STUK, Luova) "
            "with the Swedish equivalents listed above (Länsstyrelsen, Elsäkerhetsverket, Transportstyrelsen, "
            "Svenska kraftnät, SSM). "
            "If a Swedish equivalent is uncertain, mark it: [Requires verification against Swedish regulations].\n\n"
        ),
    },
    "DA": {
        "name": "Denmark / Danmark",
        "authorities": ["Energistyrelsen", "Miljøstyrelsen", "kommunalbestyrelse", "Planklagenævnet", "Kystdirektoratet",
                         "Energinet (TSO — transmission grid connection, equivalent to Finnish Fingrid)",
                         "Sikkerhedsstyrelsen (Danish Safety Technology Authority — electrical installation safety, equivalent to Finnish Tukes)"],
        "key_laws": ["Planloven (LBK nr 1157/2022)", "Miljøvurderingsloven (LOV nr 973/2023)", "Elforsyningsloven (LBK nr 1255/2021)", "Naturbeskyttelsesloven"],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in DENMARK. Apply Danish regulatory framework:\n"
            "Key authorities: Energistyrelsen (Danish Energy Agency), Miljøstyrelsen (EPA), "
            "kommunalbestyrelse (municipal council), Planklagenævnet (planning appeals board), "
            "Kystdirektoratet (coastal authority for offshore), "
            "Energinet (TSO — transmission grid connection = nettilslutningsaftale; "
            "equivalent to Finnish Fingrid), "
            "Sikkerhedsstyrelsen (electrical installation safety authority; equivalent to Finnish Tukes), "
            "Trafikstyrelsen (Danish Civil Aviation and Railway Authority — aviation obstacle lighting "
            "for wind turbines per BL 3-11; equivalent to Finnish Traficom).\n"
            "Key laws: Planloven for land use planning (building permit = Byggetilladelse), "
            "Miljøvurderingsloven for EIA (= Miljøkonsekvensvurdering / MKV), "
            "Elforsyningsloven for electricity supply, Naturbeskyttelsesloven for nature protection, "
            "BL 3-11 (aviation obstacle lighting for wind turbines).\n"
            "Replace ALL Finnish law/entity references (MRL, YSL, YVA-laki, ELY-keskus, Tukes, Traficom, Fingrid) "
            "with the Danish equivalents listed above (Miljøstyrelsen, Sikkerhedsstyrelsen, Trafikstyrelsen, "
            "Energinet). "
            "Mark uncertain items: [Requires verification against Danish regulations].\n\n"
        ),
    },
    "NO": {
        "name": "Norway / Norge",
        "authorities": ["NVE (Norges vassdrags- og energidirektorat)", "Statsforvalteren", "DSB", "Kommunen", "Miljødirektoratet",
                         "Statnett (TSO — transmission grid connection, equivalent to Finnish Fingrid)"],
        "key_laws": ["Plan- og bygningsloven (PBL 2008)", "Energiloven (1990)", "Forurensningsloven (1981)", "Naturmangfoldloven (2009)"],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in NORWAY. Apply Norwegian regulatory framework:\n"
            "Key authorities: NVE (Norwegian Water Resources and Energy Directorate — concessions), "
            "Statsforvalteren (county governor), "
            "DSB (Direktoratet for samfunnssikkerhet og beredskap — civil protection, incl. electrical "
            "installation safety; equivalent to Finnish Tukes), "
            "Kommunen (municipality), Miljødirektoratet (Environment Agency), "
            "Statnett (TSO — transmission grid connection = nettilknytningsavtale; "
            "equivalent to Finnish Fingrid), "
            "Luftfartstilsynet (Norwegian Civil Aviation Authority — aviation obstacle lighting for "
            "structures ≥60m including wind turbines; equivalent to Finnish Traficom).\n"
            "Key laws: Plan- og bygningsloven PBL 2008 (building permit = Byggetillatelse), "
            "Energiloven 1990 (energy facilities), Forurensningsloven 1981 (pollution/environmental), "
            "Naturmangfoldloven 2009 (biodiversity). EIA = Konsekvensutredning (KU).\n"
            "Replace ALL Finnish law/entity references (MRL, YSL, YVA-laki, ELY-keskus, Tukes, Traficom, Fingrid) "
            "with the Norwegian equivalents listed above (Statsforvalteren, DSB, Luftfartstilsynet, Statnett). "
            "Mark uncertain items: [Requires verification against Norwegian regulations].\n\n"
        ),
    },
    "PL": {
        "name": "Poland / Polska",
        "authorities": ["PAA (Państwowa Agencja Atomistyki)",
                         "URE (Urząd Regulacji Energetyki — energy market regulator, licensing)",
                         "PSE S.A. (Polskie Sieci Elektroenergetyckie — TSO, transmission grid operator; "
                         "equivalent to Finnish Fingrid)",
                         "RDOŚ", "Starosta (building authority)", "GDOŚ",
                         "UDT (Urząd Dozoru Technicznego — Technical Inspection Authority; closest Polish "
                         "equivalent to Finnish Tukes for technical/electrical equipment safety)"],
        "key_laws": ["Prawo atomowe (Ustawa z 29.11.2000)", "Prawo budowlane (Ustawa z 7.07.1994)", "Ustawa o OZE (20.02.2015)", "Ustawa o udostępnianiu informacji o środowisku"],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in POLAND. Apply Polish regulatory framework:\n"
            "Key authorities: PAA (State Nuclear Agency, for nuclear projects), "
            "URE (Energy Regulatory Office — market regulator and licensing, NOT the grid operator itself), "
            "PSE S.A. (Polskie Sieci Elektroenergetyckie — the actual transmission grid operator/TSO; "
            "grid connection = umowa przyłączeniowa; equivalent to Finnish Fingrid), "
            "RDOŚ (Regional Environmental Directorate), "
            "Starosta (poviat/district authority for building permits = Pozwolenie na budowę), "
            "GDOŚ (General Directorate for Environmental Protection), "
            "UDT (Urząd Dozoru Technicznego — technical/electrical equipment safety inspection; "
            "closest Polish equivalent to Finnish Tukes), "
            "ULC (Urząd Lotnictwa Cywilnego — Civil Aviation Authority; aviation obstacle marking/lighting "
            "per Rozporządzenie Ministra Infrastruktury z 12.01.2021 r., Dz.U. 2021 poz. 264; "
            "equivalent to Finnish Traficom).\n"
            "Key laws: Prawo budowlane 1994 (building permits), Ustawa o OZE 2015 (renewables), "
            "Prawo atomowe 2000 (nuclear), Ustawa o udostępnianiu informacji o środowisku (EIA = OOŚ), "
            "Rozporządzenie w sprawie przeszkód lotniczych, Dz.U. 2021 poz. 264 (aviation obstacle marking).\n"
            "Replace ALL Finnish law/entity references (MRL, YSL, YVA-laki, ELY-keskus, Tukes, Traficom, Fingrid) "
            "with the Polish equivalents listed above (RDOŚ, UDT, ULC, PSE S.A. — not URE, which is the "
            "regulator rather than the grid operator). "
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
            "TTJA (Tehnilise Järelevalve Amet — Technical Regulatory Authority; "
            "electrical/technical equipment safety, equivalent to Finnish Tukes)",
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
            "Kaitseministeerium (Ministry of Defence — mandatory military radar clearance for wind), "
            "TTJA (Tehnilise Järelevalve Amet — technical/electrical equipment safety; "
            "equivalent to Finnish Tukes), "
            "Transpordiamet (Transport Administration — aviation obstacle clearance for wind turbines; "
            "equivalent to Finnish Traficom).\n"
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
            "Replace ALL Finnish law/entity references (MRL, YSL, YVA-laki, ELY-keskus, Tukes, Traficom, Fingrid) "
            "with the Estonian equivalents listed above (Keskkonnaamet, TTJA, Transpordiamet, Elering AS). "
            "Mark uncertain items: [Requires verification against Estonian regulations].\n\n"
        ),
    },
    "LV": {
        "name": "Latvia / Latvija",
        "authorities": [
            "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija — utilities regulator)",
            "VPVB (Vides pārraudzības valsts birojs — EIA screening authority, under VARAM)",
            "Valsts vides dienests (VVD — environmental permit authority)",
            "Pašvaldības būvvalde (Municipal building authority — building permits)",
            "AST (Augstsprieguma tīkls — TSO, transmission grid connection)",
            "Sadales tīkls AS (DSO — distribution grid connection for smaller projects)",
            "VUGD (Valsts ugunsdzēsības un glābšanas dienests — fire safety authority)",
            "LGS (Latvijas gaisa satiksme — aviation obstacle clearance)",
            "NBS (Nacionālie bruņotie spēki — military radar clearance for wind)",
        ],
        "key_laws": [
            "Elektroenerģijas tirgus likums (ETL, 2005) — electricity market, SPRK licensing",
            "Atjaunojamās enerģijas likums (AEL, 2022/2023) — renewables and BESS co-location",
            "Enerģētikas likums (1998) — energy sector framework",
            "Likums par ietekmes uz vidi novērtējumu (IVN likums) — EIA procedure",
            "Vides aizsardzības likums (1997) — environmental permits (A/B/C kategorija)",
            "Būvniecības likums (2013) — building permits (Būvatļauja) from pašvaldības būvvalde",
            "Teritorijas attīstības plānošanas likums (2011) — spatial planning, detālplānojums",
            "MK noteikumi Nr. 631 — electricity generation licensing procedure",
        ],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in LATVIA (Latvija). "
            "Apply Latvian regulatory framework throughout — do NOT use Finnish law references.\n"
            "Key authorities: "
            "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija — electricity licensing: "
            "ražošanas licence >1 MW, reģistrācija 50 kW–1 MW, below 50 kW no action), "
            "VPVB (Vides pārraudzības valsts birojs — EIA/IVN screening and decisions), "
            "Valsts vides dienests VVD (environmental permits, A/B/C kategorija), "
            "Pašvaldības būvvalde (municipal building authority — Būvatļauja; "
            "IMPORTANT: exact requirements vary by municipality, always verify locally), "
            "AST — Augstsprieguma tīkls (TSO — transmission grid connection ≥110 kV, "
            "tehniskie noteikumi → pievienošanas līgums), "
            "Sadales tīkls AS (DSO — distribution grid connection for projects <110 kV), "
            "VUGD (State Fire and Rescue Service — ugunsdrošības atzinums mandatory for BESS), "
            "LGS (Latvijas gaisa satiksme — aviation obstacle clearance for wind >60m), "
            "NBS (Nacionālie bruņotie spēki — military radar clearance for wind).\n"
            "Key laws: Elektroenerģijas tirgus likums ETL (electricity market, grid connection ETL §§73-83), "
            "Atjaunojamās enerģijas likums AEL (2023 — renewables auctions, net metering, BESS co-location), "
            "Vides aizsardzības likums (environmental permits), "
            "Likums par ietekmes uz vidi novērtējumu IVN likums (EIA = IVN; screening by VPVB; "
            "mandatory for wind >2 turbines or >5 MW), "
            "Būvniecības likums (building permit = Būvatļauja from pašvaldības būvvalde; "
            "digital submission via BIS portal bis.gov.lv; 3 levels: apliecinājums / "
            "paskaidrojuma raksts / būvatļauja), "
            "Teritorijas attīstības plānošanas likums (spatial plan = teritorijas plānojums; "
            "detailed plan = detālplānojums; municipal council dome lēmums required), "
            "MK noteikumi Nr. 631 (electricity generation licensing procedure details).\n"
            "Latvia-specific notes: "
            "Latvia has NO nuclear power plants and no nuclear regulatory framework — "
            "SMR projects would require entirely new primary legislation before any permit path exists. "
            "BESS market: FCR/mFRR via AST balancing tenders; Baltic desynchronisation from "
            "IPS/UPS planned 2025 — impacts reserve market volumes significantly. "
            "Municipal building permits (Būvatļauja) are issued by pašvaldību būvvaldes — "
            "requirements vary by municipality; this document uses general principles from "
            "Būvniecības likums but local verification is always required. "
            "Replace ALL Finnish law references (MRL, YSL, YVA-laki, ELY-keskus, Tukes, Traficom, Fingrid) "
            "with the Latvian equivalents listed above. "
            "Mark uncertain items: [Requires verification against Latvian regulations].\n\n"
        ),
    },
    "LT": {
        "name": "Lithuania / Lietuva",
        "authorities": [
            "VERT (Valstybinė energetikos reguliavimo taryba — independent energy regulator, electricity licensing)",
            "LITGRID AB (TSO — transmission grid connection 110/330 kV, balancing market)",
            "ESO (Energijos skirstymo operatorius — DSO, distribution grid connection)",
            "Savivaldybės administracija (Municipal administration — statybos leidimas / building permit)",
            "AAA (Aplinkos apsaugos agentūra — EIA/PAV screening authority, under Aplinkos ministerija)",
            "PAGD (Priešgaisrinės apsaugos ir gelbėjimo departamentas — fire safety authority, under Interior Ministry)",
            "ANCO / Civilinės aviacijos administracija (CAA Lithuania — aviation obstacle clearance for wind >45m)",
            "Lietuvos kariuomenė (Lithuanian Armed Forces — military radar clearance for wind parks)",
        ],
        "key_laws": [
            "Elektros energetikos įstatymas (EEĮ, 2000) — electricity market, VERT licensing",
            "Atsinaujinančių išteklių energetikos įstatymas (AIEI, 2011) — renewables, OZE auctions, BESS co-location",
            "Energetikos įstatymas (2002) — energy sector framework, VERT mandate",
            "Planuojamos ūkinės veiklos poveikio aplinkai vertinimo įstatymas (PAV įstatymas) — EIA procedure",
            "Statybos įstatymas (consolidated) — building permits (statybos leidimas) via INFOSTATYBA",
            "Teritorijų planavimo įstatymas (consolidated) — spatial planning, bendrasis planas, detalusis planas",
            "Aplinkos apsaugos įstatymas — environmental permits (taršos leidimas/TIPK)",
        ],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in LITHUANIA (Lietuva). "
            "Apply Lithuanian regulatory framework throughout — do NOT use Finnish law references.\n"
            "Key authorities: "
            "VERT (Valstybinė energetikos reguliavimo taryba — electricity generation licensing: "
            "gamybos licencija >100 kW, registracija 10–100 kW, below 10 kW no action required), "
            "LITGRID AB (TSO — transmission grid connection 110/330 kV for projects >5 MW; "
            "techniniai prisijungimo sąlygai → tinklų prijungimo sutartis), "
            "ESO (Energijos skirstymo operatorius — DSO, distribution grid connection for projects <5 MW), "
            "Savivaldybės administracija (municipal administration — statybos leidimas; "
            "IMPORTANT: exact requirements vary by municipality, always verify locally; "
            "digital submission via INFOSTATYBA portal infostatyba.planuojuinfo.lt), "
            "AAA (Aplinkos apsaugos agentūra — PAV/EIA screening and decisions; "
            "mandatory PAV for wind >5 turbines or >30 MW; solar >50 MW; BESS screening recommended >10 MWh), "
            "PAGD (Priešgaisrinės apsaugos ir gelbėjimo departamentas — fire safety examination, "
            "priešgaisrinės saugos ekspertizė mandatory before statybos leidimas for BESS), "
            "ANCO/CAA Lithuania (aviation obstacle clearance for wind >45m height), "
            "Lietuvos kariuomenė (mandatory military radar clearance for all wind parks).\n"
            "Key laws: Elektros energetikos įstatymas EEĮ (electricity market, VERT licensing thresholds), "
            "Atsinaujinančių išteklių energetikos įstatymas AIEI "
            "(renewables, OZE feed-in premium auctions administered by VERT, "
            "net metering for prosumers ≤30 kW, BESS co-location with renewables), "
            "PAV įstatymas (EIA = PAV; two-stage: atranka screening by AAA then full PAV ataskaita; "
            "typical duration 6–18 months; PAV ataskaita must precede statybos leidimas), "
            "Statybos įstatymas (building permit = statybos leidimas from savivaldybės administracija; "
            "INFOSTATYBA portal; standard timeline 20–40 working days for complete submission), "
            "Teritorijų planavimo įstatymas (bendrasis planas / specialusis planas / detalusis planas; "
            "detalusis planas required for wind parks >1 turbine in most municipalities; "
            "bendrasis planas must designate wind energy zones), "
            "Energetikos įstatymas (energy sector framework, VERT mandate and independence), "
            "Aplinkos apsaugos įstatymas (environmental permit: taršos leidimas or TIPK/TIPK-A).\n"
            "Lithuania-specific notes: "
            "Lithuania has NO operating nuclear power plants (Ignalina NPP shut down 2009) — "
            "SMR projects would require entirely new primary legislation before any permit path exists. "
            "BESS market: FCR/aFRR/mFRR balancing via LITGRID tenders; "
            "BESS >100 kW participating in balancing market requires balanso atsakingosios šalies (BRP) "
            "sutartis with LITGRID; SCADA connection to LITGRID required for >1 MW BESS. "
            "Baltic desynchronisation from IPS/UPS to Continental European grid significantly increases "
            "FCR/FRR market volumes — major BESS business case driver. "
            "VERT OZE auctions (aukcionai) allocate competitive support (fiksuota priemoka/FIP) for "
            "10–20 years to winning wind/solar projects. "
            "Replace ALL Finnish law references (MRL, YSL, YVA-laki, ELY-keskus, Tukes, Traficom, Fingrid) "
            "with Lithuanian equivalents listed above. "
            "Mark uncertain items: [Requires verification against Lithuanian regulations].\n\n"
        ),
    },
    "DE": {
        "name": "Germany / Deutschland",
        "authorities": [
            "Immissionsschutzbehörde der Länder (BImSchG-Genehmigung)",
            "Bundesnetzagentur – BNetzA (Netzanschluss, Marktprämie)",
            "Umweltbundesamt – UBA (UVP, Umweltgutachten)",
            "BSH – Bundesamt für Seeschifffahrt und Hydrographie (Offshore)",
            "Luftfahrtbundesamt / Bundeswehr (Hindernisbeurteilung, LuftVG)",
            "Untere Naturschutzbehörde (Artenschutzrechtliche Prüfung, BNatSchG § 44)",
            "Gemeinde / Landkreis (Bauleitplanung, Baugenehmigung)",
        ],
        "key_laws": [
            "Bundes-Immissionsschutzgesetz BImSchG § 4 (immission control permit)",
            "Gesetz über die Umweltverträglichkeitsprüfung UVPG (UVP / EIA)",
            "Bundesnaturschutzgesetz BNatSchG § 44 (species protection = Artenschutz)",
            "Windenergieflächenbedarfsgesetz WindBG (wind area targets per Land)",
            "Energiewirtschaftsgesetz EnWG (grid connection = Netzanschluss)",
            "Baugesetzbuch BauGB (land-use planning = Bauleitplanung)",
            "Atomgesetz AtG (nuclear facilities, SMR)",
            "Bundeswaldgesetz BWaldG (forest clearance)",
        ],
        "prompt_prefix": (
            "IMPORTANT — COUNTRY: This project is located in GERMANY (Deutschland). "
            "Apply German regulatory framework throughout — do NOT use Finnish law references.\n"
            "Key authorities: Immissionsschutzbehörde der Länder (main BImSchG permit authority), "
            "Bundesnetzagentur BNetzA (grid connection = Netzanschlussvertrag, market premium = Marktprämie), "
            "Umweltbundesamt UBA (environmental impact assessment support), "
            "BSH Bundesamt für Seeschifffahrt und Hydrographie (offshore wind only), "
            "Luftfahrtbundesamt / Bundeswehr (aviation obstacle assessment per LuftVG § 14–18.1), "
            "Untere Naturschutzbehörde (species protection = Artenschutzrechtliche Prüfung per BNatSchG § 44), "
            "Gemeinde / Landkreis (Bauleitplanung F-Plan / B-Plan, Baugenehmigung).\n"
            "Key laws: BImSchG § 4 (immission control permit = BImSchG-Genehmigung, replaces Finnish ympäristölupa), "
            "UVPG (environmental impact assessment = UVP-Bericht, replaces Finnish YVA-ohjelma/selostus), "
            "BNatSchG § 44 (species protection = artenschutzrechtliche Prüfung / saP), "
            "WindBG (wind area designation per Land, replaces Finnish kaavoitus context), "
            "EnWG (grid connection), BauGB + Bauordnungsrecht der Länder (building permit = Baugenehmigung), "
            "Atomgesetz AtG (nuclear only), BWaldG (forest clearance = Waldumwandlung).\n"
            "Required annexes for tuulivoima_maa (wind onshore): "
            "1. Übersichtslageplan und Detaillageplan (site location maps), "
            "2. Schallimmissionsprognose / Schallgutachten (noise impact study per TA Lärm), "
            "3. Schattenwurfgutachten (shadow flicker study), "
            "4. UVP-Bericht (environmental impact report per UVPG, if threshold triggered), "
            "5. Artenschutzrechtlicher Fachbeitrag / saP (species protection assessment per BNatSchG § 44), "
            "6. Avifaunistische Bestandserhebung (bird & bat survey), "
            "7. Flugsicherungsstellungnahme (aviation clearance letter, LuftVG § 18.1 / Luftfahrtbundesamt), "
            "8. Radar- und Richtfunkprüfung (radar / microwave link clearance, Bundeswehr / Bundesnetzagentur), "
            "9. Standsicherheitsnachweis (structural safety certificate, Bauordnungsrecht), "
            "10. Brandschutzkonzept (fire safety concept).\n"
            "Replace ALL Finnish law references (MRL, YSL, YVA-laki, ELY-keskus, Tukes, Traficom, Finavia, "
            "Fingrid) with the German equivalents listed above (Fingrid → Bundesnetzagentur/BNetzA for "
            "grid connection). Note: Germany has no single federal equivalent to Tukes (technical/electrical "
            "safety oversight is split between Länder-level Gewerbeaufsichtsämter and certified inspection "
            "bodies such as TÜV/DEKRA) — mark any Tukes-context item [Requires verification against "
            "applicable Landesrecht] rather than inventing a specific authority name. "
            "Mark items requiring local Land-level verification: "
            "[Requires verification against applicable Landesrecht].\n\n"
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
            ("Periaatepäätös (VN)",              "Energistyrelsen / Klima-, Energi- og Forsyningsministeriet", "Undergrundsloven (LBK nr. 1461/2023)"),
            ("YVA-menettely",                    "Miljøministeriet / Miljøstyrelsen",          "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Rakentamislupa (ydinlaitos)",       "Sundhedsstyrelsen / Statens Institut for Strålebeskyttelse (SIS)", "Strålebeskyttelsesloven (Lov nr. 23/2018)"),
            ("Käyttölupa (ydinlaitos)",           "Sundhedsstyrelsen / SIS",                   "Strålebeskyttelsesloven (Lov nr. 23/2018)"),
            ("Vesilupa (jäähdytysvesi)",          "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 245/2025)"),
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen (planafdelingen)",                 "Planloven (LBK nr. 572/2024)"),
        ],
        "BESS": [
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Ympäristölupa",                     "Kommunen / Miljøstyrelsen",                 "Miljøbeskyttelsesloven (LBK nr. 1742/2025)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 1248/2023)"),
            ("YVA-menettely (tarvitt.)",          "Miljøstyrelsen",                            "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
        ],
        "tuulivoima_maa": [
            ("YVA-menettely (≥10 MW / ≥5 voimalaa)", "Miljøstyrelsen",                        "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Vindmølletilladelse",                   "Energistyrelsen",                       "VE-loven (LBK nr. 1031/2024)"),
            ("Osayleiskaava tai asemakaava",           "Kommunen",                             "Planloven (LBK nr. 572/2024)"),
            ("Byggetilladelse",                          "Kommunen (teknik og miljø)",            "Byggeloven (LBK nr. 1178/2016)"),
            ("Ympäristölupa (tarvitt.)",              "Miljøstyrelsen",                        "Miljøbeskyttelsesloven (LBK nr. 1742/2025)"),
            ("Verkkoliityntäsopimus",                 "Energinet",                             "Elforsyningsloven (LBK nr. 1248/2023)"),
        ],
        "aurinkovoima": [
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 1248/2023)"),
            ("YVA-menettely (tarvitt.)",          "Miljøstyrelsen",                            "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
        ],
        "smr_bess": [
            ("Periaatepäätös (VN)",              "Energistyrelsen / Klima-, Energi- og Forsyningsministeriet", "Undergrundsloven (LBK nr. 1461/2023)"),
            ("YVA-menettely",                    "Miljøministeriet / Miljøstyrelsen",          "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Rakentamislupa (ydinlaitos)",       "Sundhedsstyrelsen / SIS",                   "Strålebeskyttelsesloven (Lov nr. 23/2018)"),
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Vesilupa (jäähdytysvesi)",          "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 245/2025)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 1248/2023)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
        ],
        "vesivoima": [
            ("Vesilupa (padotus, rakentaminen)",  "Kystdirektoratet / Miljøstyrelsen",         "Vandforsyningsloven (LBK nr. 1149/2024)"),
            ("Ympäristölupa",                     "Miljøstyrelsen",                            "Miljøbeskyttelsesloven (LBK nr. 1742/2025)"),
            ("YVA-menettely (tarvitt.)",          "Miljøstyrelsen",                            "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Byggetilladelse",                      "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 1248/2023)"),
        ],
        "tuulivoima_meri": [
            ("YVA-menettely",                    "Miljøstyrelsen",                             "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Havvindtilladelse",                 "Energistyrelsen",                           "VE-loven (LBK nr. 1031/2024)"),
            ("Ympäristölupa",                     "Miljøstyrelsen",                            "Miljøbeskyttelsesloven (LBK nr. 1742/2025)"),
            ("Vesilupa (merialue)",               "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 245/2025)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 1248/2023)"),
        ],
        "datakeskus": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Miljøgodkendelse (tarvitt.)",       "Kommunen / Miljøstyrelsen",                 "Miljøbeskyttelsesloven (LBK nr. 1742/2025)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 1248/2023)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
        ],
        "teollisuus": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Miljøgodkendelse",                  "Kommunen / Miljøstyrelsen",                 "Miljøbeskyttelsesloven (LBK nr. 1742/2025)"),
            ("YVA-menettely (tarvitt.)",          "Miljøstyrelsen",                            "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
        ],
        "asuinrakennus": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Lokalplan",                         "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
            ("Naapurikuuleminen / nabohøring",    "Kommunen / hakija",                        "Byggeloven (LBK nr. 1178/2016)"),
        ],
        "maatalous": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Miljøgodkendelse (husdyr)",         "Kommunen",                                  "Husdyrbrugloven (LBK nr. 1065/2025)"),
            ("Lokalplan",                         "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
        ],
        "liikerakennus": [
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Lokalplan",                         "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
            ("Maankäyttösopimus",                 "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
        ],
        "smr_da": [
            ("Periaatepäätös (VN)",              "Energistyrelsen / Klima-, Energi- og Forsyningsministeriet", "Undergrundsloven (LBK nr. 1461/2023)"),
            ("YVA-menettely",                    "Miljøministeriet / Miljøstyrelsen",          "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Rakentamislupa (ydinlaitos)",       "Sundhedsstyrelsen / Statens Institut for Strålebeskyttelse (SIS)", "Strålebeskyttelsesloven (Lov nr. 23/2018)"),
            ("Käyttölupa (ydinlaitos)",           "Sundhedsstyrelsen / SIS",                   "Strålebeskyttelsesloven (Lov nr. 23/2018)"),
            ("Vesilupa (jäähdytysvesi)",          "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 245/2025)"),
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen (planafdelingen)",                 "Planloven (LBK nr. 572/2024)"),
        ],
        "offshore_wind": [
            ("YVA-menettely",                    "Miljøstyrelsen",                             "Miljøvurderingsloven (LBK nr. 4/2023)"),
            ("Havvindtilladelse (flydende)",      "Energistyrelsen",                           "VE-loven (LBK nr. 1031/2024)"),
            ("Ympäristölupa",                     "Miljøstyrelsen",                            "Miljøbeskyttelsesloven (LBK nr. 1742/2025)"),
            ("Vesilupa (merialue)",               "Kystdirektoratet",                          "Kystbeskyttelsesloven (LBK nr. 245/2025)"),
            ("Verkkoliityntäsopimus",             "Energinet",                                 "Elforsyningsloven (LBK nr. 1248/2023)"),
        ],
        "egs": [
            ("Forundersøgelsestilladelse (geotermisk)", "Energistyrelsen",                     "Undergrundsloven (LBK nr. 1461/2023)"),
            ("Byggetilladelse",                   "Kommunen (teknik og miljø)",                "Byggeloven (LBK nr. 1178/2016)"),
            ("Miljøgodkendelse (tarvitt.)",       "Miljøstyrelsen",                            "Miljøbeskyttelsesloven (LBK nr. 1742/2025)"),
            ("Maankäyttösopimus / kaavoitus",     "Kommunen",                                  "Planloven (LBK nr. 572/2024)"),
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
    # ── Latvija ───────────────────────────────────────────────────────────────
    "LV": {
        # ── BESS ──────────────────────────────────────────────────────────────
        "BESS": [
            ("Detālplānojums (ja nepieciešams)",        "Pašvaldības dome (Municipal council)",              "Teritorijas attīstības plānošanas likums (2011)"),
            ("IVN / Sākotnējā izvērtēšana",             "VPVB (Vides pārraudzības valsts birojs)",           "Likums par ietekmes uz vidi novērtējumu (IVN likums)"),
            ("Ugunsdrošības atzinums (BESS)",           "VUGD (Valsts ugunsdzēsības un glābšanas dienests)", "Ugunsdrošības un ugunsdzēsības likums — obligāts"),
            ("Būvatļauja (būvniecības atļauja)",        "Pašvaldības būvvalde — PRASĪBAS VARIĒ PĒC PAŠVALDĪBAS", "Būvniecības likums (2013) — digitāli caur BIS (bis.gov.lv)"),
            ("SPRK reģistrācija vai ražošanas licence", "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija)", "ETL; MK noteikumi Nr. 631 — licence >1 MW, reģistrācija 50 kW–1 MW"),
            ("Pievienošanas līgums (tīkla pieslēgums)", "AST (Augstsprieguma tīkls) vai Sadales tīkls AS",   "Elektroenerģijas tirgus likums §§73-83; RfG 2016/631 + DCC 2016/1388"),
            ("Nodošana ekspluatācijā",                  "Pašvaldības būvvalde",                              "Būvniecības likums (2013)"),
        ],
        # ── Tuulivoima (onshore) ──────────────────────────────────────────────
        "tuulivoima_maa": [
            ("Detālplānojums",                          "Pašvaldības dome (Municipal council)",              "Teritorijas attīstības plānošanas likums (2011)"),
            ("IVN (pilna procedūra)",                   "VPVB (Vides pārraudzības valsts birojs)",           "IVN likums — obligāts >2 turbīnas vai >5 MW"),
            ("NBS radiolokācijas saskaņojums",          "NBS (Nacionālie bruņotie spēki / Aizsardzības min.)", "Nacionālās drošības prasības — obligāts"),
            ("LGS šķērsļa atzinums",                   "LGS (Latvijas gaisa satiksme)",                     "Aviācijas likums — turbīnām >60m kopgarums"),
            ("Būvatļauja (vēja elektrostacija)",        "Pašvaldības būvvalde — PRASĪBAS VARIĒ PĒC PAŠVALDĪBAS", "Būvniecības likums (2013); MK noteikumi par VES būvniecību"),
            ("SPRK ražošanas licence",                  "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija)", "Elektroenerģijas tirgus likums; MK noteikumi Nr. 631 — >1 MW"),
            ("Pievienošanas līgums (AST / tīkls)",      "AST (Augstsprieguma tīkls)",                        "ETL §§73-83; Komisijas regula (ES) 2016/631 (RfG)"),
            ("Nodošana ekspluatācijā",                  "Pašvaldības būvvalde",                              "Būvniecības likums (2013)"),
        ],
        # ── Aurinkovoima ──────────────────────────────────────────────────────
        "aurinkovoima": [
            ("Detālplānojums (ja nepieciešams)",        "Pašvaldības dome (Municipal council)",              "Teritorijas attīstības plānošanas likums (2011)"),
            ("IVN / Sākotnējā izvērtēšana (>50 ha)",   "VPVB (Vides pārraudzības valsts birojs)",           "IVN likums — screening ja laukums >50 ha"),
            ("Būvatļauja (saules paneļu parks)",        "Pašvaldības būvvalde — PRASĪBAS VARIĒ PĒC PAŠVALDĪBAS", "Būvniecības likums (2013)"),
            ("SPRK ražošanas licence vai reģistrācija", "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija)", "ETL; MK noteikumi Nr. 631 — licence >1 MW, reģistrācija 50 kW–1 MW"),
            ("Pievienošanas līgums (Sadales tīkls / AST)", "Sadales tīkls AS vai AST",                       "ETL §§73-83; Komisijas regula (ES) 2016/631 (RfG)"),
            ("Nodošana ekspluatācijā",                  "Pašvaldības būvvalde",                              "Būvniecības likums (2013)"),
        ],
        # ── Offshore wind ─────────────────────────────────────────────────────
        "tuulivoima_meri": [
            ("Jūras akvatorijas izmantošanas atļauja",  "Ekonomikas ministrija / Vides ministrija",          "Jūras kodekss; Teritorijas attīstības plānošanas likums"),
            ("IVN (pilna procedūra, obligāta)",         "VPVB (Vides pārraudzības valsts birojs)",           "IVN likums — obligāta visiem jūras vēja projektiem"),
            ("NBS un LGS saskaņojums",                  "NBS + LGS (Latvijas gaisa satiksme)",               "Nacionālās drošības un aviācijas prasības"),
            ("Būvatļauja (jūras būvniecība)",           "Pašvaldības būvvalde + jūras iestādes",             "Būvniecības likums + Jūras kodekss"),
            ("SPRK ražošanas licence",                  "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija)", "ETL; MK noteikumi Nr. 631"),
            ("Pievienošanas līgums (AST jūras kabelis)", "AST (Augstsprieguma tīkls)",                       "ETL §§73-83; Komisijas regula (ES) 2016/1447 (HVDC)"),
        ],
        # ── offshore_wind alias ───────────────────────────────────────────────
        "offshore_wind": [
            ("Jūras akvatorijas izmantošanas atļauja",  "Ekonomikas ministrija / Vides ministrija",          "Jūras kodekss; Teritorijas attīstības plānošanas likums"),
            ("IVN (pilna procedūra, obligāta)",         "VPVB (Vides pārraudzības valsts birojs)",           "IVN likums — obligāta visiem jūras vēja projektiem"),
            ("NBS un LGS saskaņojums",                  "NBS + LGS (Latvijas gaisa satiksme)",               "Nacionālās drošības un aviācijas prasības"),
            ("Būvatļauja (jūras būvniecība)",           "Pašvaldības būvvalde + jūras iestādes",             "Būvniecības likums + Jūras kodekss"),
            ("SPRK ražošanas licence",                  "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija)", "ETL; MK noteikumi Nr. 631"),
            ("Pievienošanas līgums (AST jūras kabelis)", "AST (Augstsprieguma tīkls)",                       "ETL §§73-83; Komisijas regula (ES) 2016/1447 (HVDC)"),
        ],
        # ── SMR — regWarning: no nuclear framework in Latvia ──────────────────
        "SMR": [
            ("⚠️ Kodolenerģijas likums neeksistē",      "Saeima (Parliament) — nav spēkā esoša tiesiskā regulējuma", "Latvijā nav kodolelektrostaciju un nav kodolenerģijas primārā likumdošanas"),
            ("⚠️ Principa lēmums (Saeimas lēmums, nepieciešams)", "Saeima / Ministru kabinets",            "Jāpieņem jauns primārais likums pirms jebkādas atļaujas piešķiršanas"),
            ("IVN (pilna procedūra, obligāta)",         "VPVB (Vides pārraudzības valsts birojs)",           "IVN likums — obligāta kodolobiektu gadījumā"),
            ("Būvatļauja (obligāti jauni nosacījumi)",  "Pašvaldības būvvalde + valsts iestāde (jānosaka)",  "Būvniecības likums — kodolobijektiem nav īpašu noteikumu"),
            ("Pievienošanas līgums (AST)",              "AST (Augstsprieguma tīkls)",                        "ETL §§73-83"),
        ],
        # ── Datakeskus ────────────────────────────────────────────────────────
        "datakeskus": [
            ("Detālplānojums (ja nepieciešams)",        "Pašvaldības dome (Municipal council)",              "Teritorijas attīstības plānošanas likums (2011)"),
            ("IVN / Sākotnējā izvērtēšana (lielie DC)", "VPVB (Vides pārraudzības valsts birojs)",          "IVN likums — lieliem enerģijas lietotājiem"),
            ("Vides atļauja (B/C kategorija)",          "Valsts vides dienests (VVD)",                       "Vides aizsardzības likums (1997)"),
            ("Būvatļauja",                              "Pašvaldības būvvalde — PRASĪBAS VARIĒ PĒC PAŠVALDĪBAS", "Būvniecības likums (2013) — caur BIS portālu"),
            ("Pievienošanas līgums (tīkla pieslēgums)", "AST (Augstsprieguma tīkls) vai Sadales tīkls AS",   "ETL §§73-83"),
            ("Nodošana ekspluatācijā",                  "Pašvaldības būvvalde",                              "Būvniecības likums (2013)"),
        ],
        # ── Teollisuus ────────────────────────────────────────────────────────
        "teollisuus": [
            ("Detālplānojums (ja nepieciešams)",        "Pašvaldības dome (Municipal council)",              "Teritorijas attīstības plānošanas likums (2011)"),
            ("Vides atļauja (A/B/C kategorija)",        "Valsts vides dienests (VVD) / VPVB",                "Vides aizsardzības likums (1997) — kategorija atkarīga no darbības veida"),
            ("IVN / Sākotnējā izvērtēšana",             "VPVB (Vides pārraudzības valsts birojs)",           "IVN likums"),
            ("Būvatļauja",                              "Pašvaldības būvvalde — PRASĪBAS VARIĒ PĒC PAŠVALDĪBAS", "Būvniecības likums (2013)"),
            ("Nodošana ekspluatācijā",                  "Pašvaldības būvvalde",                              "Būvniecības likums (2013)"),
        ],
        # ── Asuinrakennus ─────────────────────────────────────────────────────
        "asuinrakennus": [
            ("Detālplānojums (ja nepieciešams)",        "Pašvaldības dome (Municipal council)",              "Teritorijas attīstības plānošanas likums (2011)"),
            ("Būvatļauja vai paskaidrojuma raksts",     "Pašvaldības būvvalde — PRASĪBAS VARIĒ PĒC PAŠVALDĪBAS", "Būvniecības likums (2013) — mazām ēkām paskaidrojuma raksts"),
            ("Naapurikuuleminen (kaimiņu piekrišana)",  "Pašvaldības būvvalde / hakija",                     "Būvniecības likums (2013)"),
            ("Nodošana ekspluatācijā",                  "Pašvaldības būvvalde",                              "Būvniecības likums (2013)"),
        ],
        # ── Liikerakennus ─────────────────────────────────────────────────────
        "liikerakennus": [
            ("Detālplānojums (ja nepieciešams)",        "Pašvaldības dome (Municipal council)",              "Teritorijas attīstības plānošanas likums (2011)"),
            ("Būvatļauja",                              "Pašvaldības būvvalde — PRASĪBAS VARIĒ PĒC PAŠVALDĪBAS", "Būvniecības likums (2013)"),
            ("Nodošana ekspluatācijā",                  "Pašvaldības būvvalde",                              "Būvniecības likums (2013)"),
        ],
        # ── Maatalous ─────────────────────────────────────────────────────────
        "maatalous": [
            ("Būvatļauja vai apliecinājums",            "Pašvaldības būvvalde — PRASĪBAS VARIĒ PĒC PAŠVALDĪBAS", "Būvniecības likums (2013) §§ — mazajām ēkām apliecinājums"),
            ("Vides atļauja (lieliem lauksaimniecības uzņ.)", "Valsts vides dienests (VVD)",               "Vides aizsardzības likums — IPPC slieksnis lauksaimniecībā"),
            ("Zemes izmantošanas maiņas atļauja (ja nepieciešams)", "Zemkopības ministrija / pašvaldība",  "Likums par zemes dzīlēm / lauksaimnieciskās zemes aizsardzības likums"),
        ],
        # ── Hybridi (BESS + wind/solar) ───────────────────────────────────────
        "hybridi": [
            ("Detālplānojums",                          "Pašvaldības dome (Municipal council)",              "Teritorijas attīstības plānošanas likums (2011)"),
            ("IVN (pilna procedūra)",                   "VPVB (Vides pārraudzības valsts birojs)",           "IVN likums"),
            ("Ugunsdrošības atzinums (BESS daļa)",      "VUGD (Valsts ugunsdzēsības un glābšanas dienests)", "Ugunsdrošības un ugunsdzēsības likums — obligāts"),
            ("Būvatļauja",                              "Pašvaldības būvvalde — PRASĪBAS VARIĒ PĒC PAŠVALDĪBAS", "Būvniecības likums (2013)"),
            ("SPRK ražošanas licence vai reģistrācija", "SPRK (Sabiedrisko pakalpojumu regulēšanas komisija)", "ETL; MK noteikumi Nr. 631"),
            ("Pievienošanas līgums",                    "AST (Augstsprieguma tīkls) vai Sadales tīkls AS",   "ETL §§73-83; RfG 2016/631"),
            ("Nodošana ekspluatācijā",                  "Pašvaldības būvvalde",                              "Būvniecības likums (2013)"),
        ],
        # ── smr_lv alias ─────────────────────────────────────────────────────
        "smr_lv": [
            ("⚠️ Kodolenerģijas likums neeksistē",      "Saeima (Parliament) — nav spēkā esoša tiesiskā regulējuma", "Latvijā nav kodolelektrostaciju un nav kodolenerģijas primārā likumdošanas"),
            ("⚠️ Principa lēmums (Saeimas lēmums, nepieciešams)", "Saeima / Ministru kabinets",            "Jāpieņem jauns primārais likums pirms jebkādas atļaujas piešķiršanas"),
            ("IVN (pilna procedūra, obligāta)",         "VPVB (Vides pārraudzības valsts birojs)",           "IVN likums"),
            ("Pievienošanas līgums (AST)",              "AST (Augstsprieguma tīkls)",                        "ETL §§73-83"),
        ],
    },
    # ── Lietuva ───────────────────────────────────────────────────────────────
    "LT": {
        # ── BESS ──────────────────────────────────────────────────────────────
        "BESS": [
            ("Detalusis planas (jei reikalingas)",       "Savivaldybės taryba (Municipal council)",            "Teritorijų planavimo įstatymas (consolidated)"),
            ("PAV atranka arba PAV (stambūs projektai)", "AAA (Aplinkos apsaugos agentūra)",                  "PAV įstatymas — AAA screening rekomenduojamas visiems >10 MWh projektams"),
            ("Priešgaisrinės saugos ekspertizė (BESS)",  "PAGD (Priešgaisrinės apsaugos ir gelbėjimo departamentas)", "Priešgaisrinės saugos taisyklės — privaloma prieš statybos leidimą"),
            ("Statybos leidimas",                        "Savivaldybės administracija — REIKALAVIMAI SKIRIASI PAGAL SAVIVALDYBĘ", "Statybos įstatymas — teikiama per INFOSTATYBA portalą (infostatyba.planuojuinfo.lt)"),
            ("VERT gamybos licencija arba registracija", "VERT (Valstybinė energetikos reguliavimo taryba)",  "EEĮ — licencija >100 kW, registracija 10–100 kW"),
            ("Tinklų prisijungimo sutartis",             "LITGRID AB (>5 MW) arba ESO (<5 MW)",               "Elektros energetikos įstatymas; RfG 2016/631 + DCC 2016/1388"),
            ("Statinio pripažinimas tinkamu naudoti",    "Savivaldybės administracija",                       "Statybos įstatymas — per INFOSTATYBA"),
        ],
        # ── Tuulivoima (onshore) ──────────────────────────────────────────────
        "tuulivoima_maa": [
            ("Detalusis planas",                         "Savivaldybės taryba (Municipal council)",            "Teritorijų planavimo įstatymas — privalomas >1 turbinai daugelyje savivaldybių"),
            ("PAV (pilna procedūra)",                    "AAA (Aplinkos apsaugos agentūra)",                  "PAV įstatymas — privaloma >5 turbinos arba >30 MW; atranka 2–5 turbinoms"),
            ("Kariuomenės radiolokacinis suderinimas",   "Lietuvos kariuomenė / Krašto apsaugos ministerija", "Nacionalinio saugumo reikalavimai — privalomas visiems vėjo parkams"),
            ("Oro navigacijos suderinimas",              "ANCO / Civilinės aviacijos administracija (CAA)",   "Aviacijos taisyklės — turbinoms >45m aukštis"),
            ("Statybos leidimas (vėjo elektrinė)",       "Savivaldybės administracija — REIKALAVIMAI SKIRIASI PAGAL SAVIVALDYBĘ", "Statybos įstatymas; Vėjo energetikos specialusis planas (zonavimas)"),
            ("VERT gamybos licencija",                   "VERT (Valstybinė energetikos reguliavimo taryba)",  "EEĮ; AIEI — privaloma >100 kW"),
            ("Tinklų prisijungimo sutartis (LITGRID)",   "LITGRID AB (Lietuvos perdavimo sistemos operatorius)", "EEĮ; Komisijos reglamentas (ES) 2016/631 (RfG)"),
            ("Statinio pripažinimas tinkamu naudoti",    "Savivaldybės administracija",                       "Statybos įstatymas"),
        ],
        # ── Aurinkovoima ──────────────────────────────────────────────────────
        "aurinkovoima": [
            ("Detalusis planas (jei reikalingas)",       "Savivaldybės taryba (Municipal council)",            "Teritorijų planavimo įstatymas"),
            ("PAV atranka arba PAV (>50 MW / >100 ha)",  "AAA (Aplinkos apsaugos agentūra)",                  "PAV įstatymas — privaloma >50 MW (>100 ha); atranka >10 MW (>25 ha)"),
            ("Statybos leidimas (saulės parkas)",        "Savivaldybės administracija — REIKALAVIMAI SKIRIASI PAGAL SAVIVALDYBĘ", "Statybos įstatymas — per INFOSTATYBA"),
            ("VERT gamybos licencija arba registracija", "VERT (Valstybinė energetikos reguliavimo taryba)",  "EEĮ; AIEI — licencija >100 kW, registracija 10–100 kW"),
            ("Tinklų prisijungimo sutartis (ESO / LITGRID)", "ESO (Energijos skirstymo operatorius) arba LITGRID", "EEĮ; Komisijos reglamentas (ES) 2016/631 (RfG)"),
            ("Statinio pripažinimas tinkamu naudoti",    "Savivaldybės administracija",                       "Statybos įstatymas"),
        ],
        # ── Offshore wind ─────────────────────────────────────────────────────
        "tuulivoima_meri": [
            ("Jūrinės erdvės planavimo dokumentas",      "Aplinkos ministerija / Susisiekimo ministerija",    "Lietuvos Respublikos teritorijų planavimo įstatymas; Jūros kodeksas"),
            ("PAV (pilna procedūra, privaloma)",         "AAA (Aplinkos apsaugos agentūra)",                  "PAV įstatymas — privaloma visiems jūros vėjo projektams"),
            ("Kariuomenės ir oro navigacijos suderinimas", "Lietuvos kariuomenė + ANCO/CAA",                  "Nacionalinio saugumo ir aviacijos reikalavimai"),
            ("Statybos leidimas (jūros statiniai)",      "Savivaldybės administracija + jūros institucijos",  "Statybos įstatymas + Jūros kodeksas"),
            ("VERT gamybos licencija",                   "VERT (Valstybinė energetikos reguliavimo taryba)",  "EEĮ; AIEI"),
            ("Tinklų prisijungimo sutartis (LITGRID, jūros kabelis)", "LITGRID AB",                          "EEĮ; Komisijos reglamentas (ES) 2016/1447 (HVDC)"),
        ],
        # ── SMR — regWarning: no nuclear framework in Lithuania ───────────────
        "SMR": [
            ("⚠️ Branduolinės energetikos įstatymas neegzistuoja", "Seimas (Parliament) — nėra galiojančios teisinės bazės", "Lietuva neturi branduolinių elektrinių (Ignalinos AE uždaryta 2009) ir nėra pirminės branduolinės energetikos teisės aktų"),
            ("⚠️ Principinis sprendimas (Seimo sprendimas, būtinas)", "Seimas / Vyriausybė",                "Prieš bet kokio leidimo išdavimą būtina priimti naują pirminį teisės aktą"),
            ("PAV (pilna procedūra, privaloma)",         "AAA (Aplinkos apsaugos agentūra)",                  "PAV įstatymas — privaloma branduoliniams objektams"),
            ("Statybos leidimas (specialios sąlygos)",   "Savivaldybės administracija + valstybės institucija (nustatyti)", "Statybos įstatymas — branduoliniams objektams nėra specialių nuostatų"),
            ("Tinklų prisijungimo sutartis (LITGRID)",   "LITGRID AB",                                        "EEĮ §§73-83"),
        ],
        # ── Datakeskus ────────────────────────────────────────────────────────
        "datakeskus": [
            ("Detalusis planas (jei reikalingas)",       "Savivaldybės taryba (Municipal council)",            "Teritorijų planavimo įstatymas"),
            ("PAV atranka arba PAV (dideli DC)",         "AAA (Aplinkos apsaugos agentūra)",                  "PAV įstatymas — dideliems energijos vartotojams"),
            ("Taršos leidimas (B/C kategorija)",         "Aplinkos apsaugos agentūra (AAA) / rajonas",        "Aplinkos apsaugos įstatymas"),
            ("Statybos leidimas",                        "Savivaldybės administracija — REIKALAVIMAI SKIRIASI PAGAL SAVIVALDYBĘ", "Statybos įstatymas — per INFOSTATYBA"),
            ("Tinklų prisijungimo sutartis",             "LITGRID AB arba ESO",                               "EEĮ §§73-83"),
            ("Statinio pripažinimas tinkamu naudoti",    "Savivaldybės administracija",                       "Statybos įstatymas"),
        ],
        # ── Teollisuus ────────────────────────────────────────────────────────
        "teollisuus": [
            ("Detalusis planas (jei reikalingas)",       "Savivaldybės taryba (Municipal council)",            "Teritorijų planavimo įstatymas"),
            ("Taršos leidimas arba TIPK (A/B kategorija)", "AAA / rajono aplinkos apsaugos departamentas",    "Aplinkos apsaugos įstatymas — kategorija priklauso nuo veiklos tipo"),
            ("PAV atranka arba PAV",                     "AAA (Aplinkos apsaugos agentūra)",                  "PAV įstatymas"),
            ("Statybos leidimas",                        "Savivaldybės administracija — REIKALAVIMAI SKIRIASI PAGAL SAVIVALDYBĘ", "Statybos įstatymas"),
            ("Statinio pripažinimas tinkamu naudoti",    "Savivaldybės administracija",                       "Statybos įstatymas"),
        ],
        # ── Asuinrakennus ─────────────────────────────────────────────────────
        "asuinrakennus": [
            ("Detalusis planas (jei reikalingas)",       "Savivaldybės taryba (Municipal council)",            "Teritorijų planavimo įstatymas"),
            ("Statybos leidimas arba rašytinis pritarimas", "Savivaldybės administracija — REIKALAVIMAI SKIRIASI PAGAL SAVIVALDYBĘ", "Statybos įstatymas — supaprastintas kelias mažiems statiniams"),
            ("Kaimynų informavimas",                     "Savivaldybės administracija / pareiškėjas",         "Statybos įstatymas"),
            ("Statinio pripažinimas tinkamu naudoti",    "Savivaldybės administracija",                       "Statybos įstatymas"),
        ],
        # ── Liikerakennus ─────────────────────────────────────────────────────
        "liikerakennus": [
            ("Detalusis planas (jei reikalingas)",       "Savivaldybės taryba (Municipal council)",            "Teritorijų planavimo įstatymas"),
            ("Statybos leidimas",                        "Savivaldybės administracija — REIKALAVIMAI SKIRIASI PAGAL SAVIVALDYBĘ", "Statybos įstatymas — per INFOSTATYBA"),
            ("Statinio pripažinimas tinkamu naudoti",    "Savivaldybės administracija",                       "Statybos įstatymas"),
        ],
        # ── Hybridi (BESS + wind/solar) ───────────────────────────────────────
        "hybridi": [
            ("Detalusis planas",                         "Savivaldybės taryba (Municipal council)",            "Teritorijų planavimo įstatymas"),
            ("PAV (pilna procedūra)",                    "AAA (Aplinkos apsaugos agentūra)",                  "PAV įstatymas"),
            ("Priešgaisrinės saugos ekspertizė (BESS dalis)", "PAGD (Priešgaisrinės apsaugos ir gelbėjimo departamentas)", "Priešgaisrinės saugos taisyklės — privaloma"),
            ("Statybos leidimas",                        "Savivaldybės administracija — REIKALAVIMAI SKIRIASI PAGAL SAVIVALDYBĘ", "Statybos įstatymas — per INFOSTATYBA"),
            ("VERT gamybos licencija arba registracija", "VERT (Valstybinė energetikos reguliavimo taryba)",  "EEĮ; AIEI"),
            ("Tinklų prisijungimo sutartis",             "LITGRID AB arba ESO",                               "EEĮ; RfG 2016/631"),
            ("Statinio pripažinimas tinkamu naudoti",    "Savivaldybės administracija",                       "Statybos įstatymas"),
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
            "Brandsäkerhetsrapport BESS (NFPA 855 / EN 50604-1) — tillstånd enligt "
            "Lag (2010:1011) om brandfarliga och explosiva varor (LBE)",
            "Hydrogeologisk utredning (kylvattenresurs)",
            "Nätanslutningsplan (Svenska kraftnät)",
            "Detaljplan / kommunal markanvändningsplan",
            "Bolagsregistreringsutdrag (Bolagsverket)",
        ],
        "datakeskus": [
            "Lägeskarta (skala 1:20 000 eller vidare)",
            "Maankäyttöselvitys PDF (NCE)",
            "Situationsplan och baskarta (skala 1:500)",
            "Byggnadsritningar (tekniskt utrymme, kylsystem)",
            "Bullerutredning (kyl- och aggregatbuller)",
            "Värmelastutredning för kylning",
            "Elsystemplan (UPS, reservkraft, anslutning)",
            "Brandsläcknings- och brandsäkerhetsplan",
            "Nätanslutningsberäkning (Svenska kraftnät kapacitetsutredning)",
            "Miljökonsekvensbedömning (vid behov)",
            "PUE- och energieffektivitetsutredning",
            "Bolagsregistreringsutdrag (Bolagsverket)",
        ],
        # 2026-08-23 (item 4, SE research — see SE_LAW_CITATION_RESEARCH.md):
        # energy-project hanketyyppis added first per pilot-activity priority
        # (SE/Ecogain); construction types (asuinrakennus/liikerakennus/
        # maatalous/teollisuus) deliberately not yet added, lower priority.
        # All citations verified against Riksdagen/Boverket/MSB/
        # Naturvårdsverket -- see the research doc for sourcing.
        "BESS": [
            "Sijaintikartta / Lägesbeskrivning (skala 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Anmälan/tillstånd miljöfarlig verksamhet — Miljöbalken kap. 9 (anmälan till kommun "
            "eller tillstånd från Länsstyrelsen beroende på storlek)",
            "Brandskyddsdokumentation — Lag (2010:1011) om brandfarliga och explosiva varor (LBE)",
            "Bygglov / detaljplan — Plan- och bygglagen (2010:900)",
            "Nätanslutningsplan (Svenska kraftnät / lokalt elnätsbolag)",
            "Bolagsregistreringsutdrag (Bolagsverket)",
            "Fullmakt (om ombud företräder sökanden)",
        ],
        "tuulivoima_maa": [
            "Sijaintikartta / Lägesbeskrivning (skala 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Miljökonsekvensbeskrivning (MKB) — Miljöbalken kap. 6",
            "Tillståndsansökan miljöfarlig verksamhet — Miljöbalken kap. 9 (prövas av "
            "Miljöprövningsdelegationen vid Länsstyrelsen)",
            "Kommunalt tillstyrkande (kommunalt veto — Miljöbalken 16 kap. 4 §)",
            "Bygglov / detaljplan — Plan- och bygglagen (2010:900)",
            "Nätanslutningsplan (Svenska kraftnät / lokalt elnätsbolag)",
            "Bolagsregistreringsutdrag (Bolagsverket)",
            "Fullmakt (om ombud företräder sökanden)",
        ],
        "tuulivoima_meri": [
            "Sijaintikartta / Lägesbeskrivning (sjökort)",
            "Maankäyttöselvitys PDF (NCE)",
            "Miljökonsekvensbeskrivning (MKB) — Miljöbalken kap. 6",
            "Tillståndsansökan vattenverksamhet — Miljöbalken kap. 11 (prövas av mark- och "
            "miljödomstol)",
            "Kommunalt tillstyrkande (kommunalt veto — Miljöbalken 16 kap. 4 §, gäller även "
            "kustnära havsbaserad vindkraft)",
            "Nätanslutningsplan (Svenska kraftnät)",
            "Bolagsregistreringsutdrag (Bolagsverket)",
            "Fullmakt (om ombud företräder sökanden)",
        ],
        "aurinkovoima": [
            "Sijaintikartta / Lägesbeskrivning (skala 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Bygglov / detaljplan — Plan- och bygglagen (2010:900)",
            "Anmälan miljöfarlig verksamhet (vid behov, större anläggningar) — Miljöbalken kap. 9",
            "Nätanslutningsplan (lokalt elnätsbolag)",
            "Bolagsregistreringsutdrag (Bolagsverket)",
            "Fullmakt (om ombud företräder sökanden)",
        ],
        "vesivoima": [
            "Sijaintikartta / Lägesbeskrivning (skala 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Miljökonsekvensbeskrivning (MKB) — Miljöbalken kap. 6",
            "Tillståndsansökan vattenverksamhet — Miljöbalken kap. 11 (prövas av mark- och "
            "miljödomstol)",
            "Dammsäkerhetsdokumentation — Förordning (2014:214) om dammsäkerhet",
            "Nätanslutningsplan (Svenska kraftnät)",
            "Bolagsregistreringsutdrag (Bolagsverket)",
            "Fullmakt (om ombud företräder sökanden)",
        ],
        # hybridi (BESS + wind/solar): a real, de-duplicated combined list per
        # the user's explicit ask -- NOT a mechanical union of the BESS and
        # wind/solar lists above (which would duplicate bygglov/nätanslutning/
        # bolagsregistrering entries three times over).
        "hybridi": [
            "Sijaintikartta / Lägesbeskrivning (skala 1:20 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Miljökonsekvensbeskrivning (MKB, vid behov beroende på anläggningens storlek) — "
            "Miljöbalken kap. 6",
            "Tillståndsansökan/anmälan miljöfarlig verksamhet (BESS- och kraftproduktionsdel) — "
            "Miljöbalken kap. 9",
            "Brandskyddsdokumentation (BESS) — Lag (2010:1011) om brandfarliga och explosiva "
            "varor (LBE)",
            "Kommunalt tillstyrkande (vid vindkraftkomponent — kommunalt veto, Miljöbalken "
            "16 kap. 4 §)",
            "Bygglov / detaljplan — Plan- och bygglagen (2010:900)",
            "Integrerad nätanslutningsplan (Svenska kraftnät / lokalt elnätsbolag)",
            "Bolagsregistreringsutdrag (Bolagsverket)",
            "Fullmakt (om ombud företräder sökanden)",
        ],
        "ymparistolupa": [
            "Tillståndsansökan miljöfarlig verksamhet — Miljöbalken kap. 9",
            "Miljökonsekvensbeskrivning (MKB, vid behov) — Miljöbalken kap. 6",
            "Bolagsregistreringsutdrag (Bolagsverket)",
            "Fullmakt (om ombud företräder sökanden)",
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
        "datakeskus": [
            "Beliggenhedskort (1:20.000 eller bredere)",
            "Maankäyttöselvitys PDF (NCE)",
            "Situationsplan og grundkort (1:500)",
            "Byggetegninger (teknikrum, kølesystemer)",
            "Støjundersøgelse (køle- og aggregatstøj)",
            "Varmelastundersøgelse for køling",
            "Elsystemplan (UPS, nødstrøm, tilslutning)",
            "Brandslukning- og brandsikkerhedsplan",
            "Nettilslutningsberegning (Energinet kapacitetsundersøgelse)",
            "Miljøkonsekvensvurdering (hvis relevant)",
            "PUE- og energieffektivitetsundersøgelse",
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
        "datakeskus": [
            "Kart / Stedsbeskrivelse (1:20 000 eller bredere)",
            "Maankäyttöselvitys PDF (NCE)",
            "Situasjonsplan og grunnkart (1:500)",
            "Byggetegninger (teknisk rom, kjølesystemer)",
            "Støyutredning (kjøle- og aggregatstøy)",
            "Varmelastutredning for kjøling",
            "Elsystemplan (UPS, reservekraft, tilknytning)",
            "Brannslukking- og brannsikkerhetsplan",
            "Nettilknytningsberegning (Statnett kapasitetsutredning)",
            "Konsekvensutredning (ved behov)",
            "PUE- og energieffektivitetsutredning",
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
        "datakeskus": [
            "Mapa lokalizacyjna (skala 1:20 000 lub większa)",
            "Maankäyttöselvitys PDF (NCE)",
            "Plan sytuacyjny i mapa zasadnicza (skala 1:500)",
            "Projekty budowlane (pomieszczenie techniczne, systemy chłodzenia)",
            "Raport hałasowy (hałas chłodzenia i agregatów)",
            "Analiza obciążenia cieplnego chłodzenia",
            "Plan systemu elektrycznego (UPS, zasilanie awaryjne, przyłącze)",
            "Plan gaszenia pożaru i bezpieczeństwa pożarowego",
            "Plan przyłączenia do sieci (PSE S.A. — analiza zdolności przyłączeniowej)",
            "Ocena oddziaływania na środowisko (w razie potrzeby)",
            "Raport PUE i efektywności energetycznej",
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
            "Jäähavainnointisüsteemi kirjeldus (ice detection system description)",
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
            "Jäähavainnointisüsteemi kirjeldus (ice detection system description)",
        ],
        "datakeskus": [
            "Asukohakaart (mõõtkava 1:20 000 või laiem)",
            "Maankäyttöselvitys PDF (NCE)",
            "Situatsiooniplaan ja põhikaart (mõõtkava 1:500)",
            "Ehitusprojektid (tehniline ruum, jahutussüsteemid)",
            "Müraraport (jahutus- ja generaatorimüra)",
            "Jahutuse soojuskoormuse analüüs",
            "Elektrisüsteemi plaan (UPS, reservtoide, liitumine)",
            "Tuletõrje- ja tuleohutusplaan",
            "Liitumisleping Elering AS-iga (võimsusuuring)",
            "Keskkonnamõju hindamine (vajadusel)",
            "PUE ja energiatõhususe aruanne",
            "Äriregistri väljavõte (Company Registry extract)",
        ],
    },
    "DE": {
        "tuulivoima_maa": [
            "Übersichtslageplan und Detaillageplan (M 1:25 000 / 1:5 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "Schallimmissionsprognose / Schallgutachten (TA Lärm — BImSchG-Genehmigung)",
            "Schattenwurfgutachten (Schattenwurf-Analyse, automatische STF-Abschaltung)",
            "UVP-Bericht (Umweltverträglichkeitsprüfung nach UVPG, sofern erforderlich)",
            "Artenschutzrechtlicher Fachbeitrag / saP (BNatSchG § 44 — Artenschutzprüfung)",
            "Avifaunistische Bestandserhebung inkl. Fledermauserfassung (TAK-Methode)",
            "Flugsicherungsstellungnahme (Luftfahrtbundesamt / Bundeswehr — LuftVG § 18.1)",
            "Radar- und Richtfunkprüfung (Bundeswehr / Bundesnetzagentur — Standortkoordinaten)",
            "Standsicherheitsnachweis (prüfpflichtiger Statiker — Bauordnungsrecht der Länder)",
            "Brandschutzkonzept (Feuerwehr / Bauordnungsrecht)",
            "Netzanschlussbegehren (BNetzA / Netzbetreiber — EnWG § 5)",
            "Handelsregisterauszug des Antragstellers",
            "Vollmacht (sofern ein Bevollmächtigter den Antragsteller vertritt)",
            "Eiserkennungssystem-Beschreibung (Vereisungsrisiko / Eiswurf)",
        ],
        "tuulivoima_meri": [
            "Übersichtslageplan Offshore (M 1:50 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "BSH-Planfeststellungsantrag (Fachplanung nach WindSeeG)",
            "UVP-Bericht Offshore (UVPG — Schutzgüter Meeresumwelt, Vögel, Fledermäuse)",
            "Schallimmissionsprognose Offshore (Bau- und Betriebslärm inkl. Unterwasserschall)",
            "Flugsicherungsstellungnahme (Luftfahrtbundesamt / Bundeswehr — LuftVG § 18.1)",
            "Netzanschlussbegehren (Übertragungsnetzbetreiber — EnWG)",
            "Meeresökologisches Fachgutachten (Benthos, Fische, Meeressäuger — BNatSchG)",
            "Handelsregisterauszug des Antragstellers",
            "Vollmacht (sofern ein Bevollmächtigter den Antragsteller vertritt)",
            "Eiserkennungssystem-Beschreibung (Vereisungsrisiko / Eiswurf)",
        ],
        "BESS": [
            "Übersichtslageplan und Detaillageplan (M 1:25 000 / 1:5 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "BImSchG-Genehmigungsantrag oder Baugenehmigungsunterlagen (Länder-Bauordnung)",
            "Brandschutzkonzept BESS (NFPA 855 / VdS 3500 — Lithium-Ionen)",
            "Explosionsschutzzoneneinteilung (ATEX — BetrSichV § 5)",
            "Löschwasserrückhaltenachweis (WHG § 62 — Gewässerschutz)",
            "Schallemissionsprognose (TA Lärm — falls Wohnbebauung im Einwirkungsbereich)",
            "Netzanschlussbegehren (BNetzA / Netzbetreiber — EnWG § 5)",
            "Handelsregisterauszug des Antragstellers",
            "Vollmacht (sofern ein Bevollmächtigter den Antragsteller vertritt)",
        ],
        "aurinkovoima": [
            "Lageplan (M 1:5 000 oder kleiner)",
            "Maankäyttöselvitys PDF (NCE)",
            "Bauvoranfrage oder Baugenehmigungsunterlagen (Länder-Bauordnung)",
            "Artenschutzrechtliche Prüfung (sofern Außenbereich — BNatSchG § 44)",
            "Schallimmissionsprognose (TA Lärm — bei Wechselrichter-Immissionen)",
            "Netzanschlussbegehren (BNetzA / Netzbetreiber — EnWG § 5)",
            "Handelsregisterauszug des Antragstellers",
            "Vollmacht (sofern ein Bevollmächtigter den Antragsteller vertritt)",
            "Rückbauplan und Rekultivierungsverpflichtung (Rückbaubürgschaft)",
        ],
        "SMR": [
            "Übersichtslageplan (M 1:25 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "⚠️ Genehmigungsantrag nach Atomgesetz (AtG § 7) — Bundesaufsicht BMUV",
            "Vorläufige Sicherheitsanalyse (Probabilistische Sicherheitsanalyse PSA)",
            "UVP-Bericht (UVPG — Schutzgüter inkl. radioaktive Emissionen)",
            "Hydrogeologisches Gutachten (Kühlwasserbedarf / Grundwasserschutz)",
            "Netzanschlussplanung (Übertragungsnetzbetreiber — EnWG)",
            "Bauleitplanung / Raumordnungsverfahren (BauGB — Standortfestlegung)",
            "Handelsregisterauszug des Antragstellers",
            "⚠️ Bundestagsbeschluss ggf. erforderlich — kein vereinfachtes Genehmigungsverfahren",
        ],
        "smr_bess": [
            "Übersichtslageplan (M 1:25 000)",
            "Maankäyttöselvitys PDF (NCE)",
            "⚠️ Genehmigungsantrag nach Atomgesetz (AtG § 7) — Bundesaufsicht BMUV",
            "Vorläufige Sicherheitsanalyse PSA (SMR-Komponente)",
            "Brandschutzkonzept BESS (NFPA 855 / VdS 3500)",
            "UVP-Bericht (UVPG)",
            "Netzanschlussplanung (Übertragungsnetzbetreiber — EnWG)",
            "Handelsregisterauszug des Antragstellers",
        ],
        "datakeskus": [
            "Lageplan (Maßstab 1:20 000 oder weiter)",
            "Maankäyttöselvitys PDF (NCE)",
            "Lageplan und Katasterkarte (Maßstab 1:500)",
            "Baupläne (Technikraum, Kühlsysteme)",
            "Schallimmissionsprognose (Kühl- und Aggregatlärm)",
            "Wärmelastuntersuchung für Kühlung",
            "Elektrosystemplan (USV, Notstrom, Netzanschluss)",
            "Brandbekämpfungs- und Brandschutzkonzept",
            "Netzanschlussbegehren (BNetzA / Netzbetreiber — Kapazitätsnachweis, EnWG § 5)",
            "UVP-Bericht (falls erforderlich)",
            "PUE- und Energieeffizienznachweis",
            "Handelsregisterauszug des Antragstellers",
        ],
    },
}

# 2026-08-23 (manual QA item 4, priority 3): the country-scoped smr_XX
# hanketyyppis (smr_se/no/da/de/ee/lv) had ZERO _COUNTRY_LIITTEET coverage of
# their own, despite the target country being unambiguous from the
# hanketyyppi ID itself, so they fell through to the FI base SMR liite list.
# smr_ee already had its own real entry (unchanged here). For SE/DA/NO/DE,
# smr_XX is the exact same real-world project as that country's existing,
# already-researched "SMR" entry — just reached via the country-scoped
# hanketyyppi ID instead of the generic one — so it reuses that entry
# directly, same aliasing pattern already used for _HANKE_NIMI_TRANS's
# offshore_wind/smr_XX entries. LV is deliberately NOT included here: LV has
# zero _COUNTRY_LIITTEET coverage for ANY hanketyyppi (no baseline "SMR"
# entry to alias from), and Latvia has no nuclear regulatory framework at all
# (see the _STUK_REPLACEMENT LV hedge) — a real liite list would need
# genuine research, not a mechanical copy, so smr_lv is intentionally left
# for that follow-up work rather than aliased or fabricated here.
for _smr_c in ("SE", "DA", "NO", "DE"):
    _COUNTRY_LIITTEET[_smr_c][f"smr_{_smr_c.lower()}"] = _COUNTRY_LIITTEET[_smr_c]["SMR"]
del _smr_c

# 2026-08-23 (item 4, SE research): offshore_wind is the same real-world
# project as tuulivoima_meri (see the identical _HANKE_NIMI_TRANS aliasing
# from PR #103) -- SE's newly-added tuulivoima_meri liite entry applies
# equally to offshore_wind, same reasoning as above.
_COUNTRY_LIITTEET["SE"]["offshore_wind"] = _COUNTRY_LIITTEET["SE"]["tuulivoima_meri"]

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
    "Älä koskaan generoi sisältöä joka ei perustu haettuihin lähteisiin. "
    "TÄYDENNETTÄVÄ-MERKINTÄ: Kun jokin projektispesifinen tieto puuttuu hakijalta "
    "(esim. tarkat koordinaatit, kiinteistön pinta-ala, etäisyydet naapureihin, hankkeen tarkat "
    "mitat, vastuuhenkilön nimi tai Y-tunnus) — kirjoita kyseiseen kohtaan tekstissä täsmälleen "
    "muodossa [TÄYDENNETTÄVÄ – <lyhyt kuvaus puuttuvasta tiedosta>] ilman sisältöä sen ympärille. "
    "Käytä tätä merkintää VAIN silloin kun jokin konkreettinen hakijalta saatava syötetieto "
    "puuttuu — ÄLÄ käytä sitä epävarmoille faktoille (niille on ⚠️-merkintä) eikä yleisille "
    "arvauksille. Merkinnän lyhyt kuvaus kertoo täsmälleen mitä tietoa tarvitaan. "
    "MARKKINATILASTOSÄÄNTÖ: Älä koskaan mainitse tarkkoja ulkoisia markkinatilastoja, "
    "indeksejä, hintoja tai määrällisiä lukemia (esim. '€/MW/vuosi', 'reservimarkkinahinta X€', "
    "'kapasiteettimaksu Y%', tiettyä indeksipistettä tai raporttinimeä) elleivät ne löydy "
    "suoraan annetuista RAG-lähteistä. Tällainen tieto on usein koulutusaineistosta peräisin "
    "eikä välttämättä pidä paikkaansa — se on hallusinaatioriskin ydintyyppi. "
    "Jos haluat viitata markkinakontekstiin ilman vahvistettua RAG-lähdettä, "
    "käytä merkintää [TÄYDENNETTÄVÄ – ajankohtaiset markkinatiedot] tai jätä luku kokonaan mainitsematta. "
    "PAKOLLINEN SÄÄNTÖ: Kun esität tarkan teknisen, tieteellisen tai "
    "tilastollisen väitteen (mittausluku, aikaraja, prosenttiosuus tms.) "
    "NIMETYN ulkoisen lähteen kanssa (tutkimus, yritys, standardiorganisaatio, "
    "testilaboratorio), lähdenimi SAA tulla vain suoraan tälle kutsulle "
    "palautetusta RAG-kontekstista. Jos väite on perusteltu mutta sinulla ei "
    "ole RAG-kontekstissa nimettyä lähdettä sille: (a) poista nimetty "
    "attribuutio kokonaan ja esitä väite yleisenä alan käytäntönä, TAI "
    "(b) merkitse väite [Huom] Asiantuntijatarkistus suositellaan -tagilla. "
    "Älä koskaan keksi tutkimuksen, yrityksen tai standardin nimeä "
    "paikkaamaan puuttuvaa lähdettä."
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
    "DE": (
        "KRITISCHE SPRACHANFORDERUNG: Sie MÜSSEN JEDES Wort dieses Genehmigungsantrags auf Deutsch "
        "verfassen. ALLE Überschriften, Absätze, Aufzählungspunkte, Fußnoten und Anmerkungen müssen "
        "auf Deutsch sein. Fügen Sie KEINE finnischen Wörter oder Sätze in die Ausgabe ein. "
        "Finnische Gesetzesnummern (z. B. YSL 527/2014, MRL 132/1999) dürfen als juristische "
        "Kennungen erscheinen — fügen Sie stets den deutschen Gesetzesnamen daneben hinzu. Finnische "
        "Eigennamen wie Stadtnamen, Firmennamen und Behördenabkürzungen (ELY, STUK, Luova, Fingrid, "
        "Traficom) sind ausschließlich als Eigennamen zulässig.\n\n"
    ),
    "LV": (
        "KRITISKA VALODAS PRASĪBA: Jums IR JĀRAKSTA KATRS šī atļaujas pieteikuma vārds latviešu valodā. "
        "VISIEM virsrakstiem, rindkopām, saraksta punktiem, zemsvītras piezīmēm un piezīmēm ir jābūt "
        "latviešu valodā. NEIEKĻAUJIET somu vārdus vai teikumus izvadē. "
        "Somu likumu numuri (piem., YSL 527/2014, MRL 132/1999) drīkst parādīties kā juridiski "
        "identifikatori — vienmēr pievienojiet tiem blakus latviešu likuma nosaukumu. Somu īpašvārdi, "
        "piemēram, pilsētu nosaukumi, uzņēmumu nosaukumi un iestāžu saīsinājumi (ELY, STUK, Luova, "
        "Fingrid, Traficom) ir pieņemami tikai kā īpašvārdi.\n\n"
    ),
    "LT": (
        "KRITINIS KALBOS REIKALAVIMAS: JŪS PRIVALOTE parašyti KIEKVIENĄ šio leidimo paraiškos žodį "
        "lietuvių kalba. VISOS antraštės, pastraipos, sąrašo punktai, išnašos ir pastabos turi būti "
        "lietuvių kalba. NEĮTRAUKITE suomiškų žodžių ar sakinių į išvestį. "
        "Suomijos įstatymų numeriai (pvz., YSL 527/2014, MRL 132/1999) gali būti nurodyti kaip "
        "juridiniai identifikatoriai — visada pridėkite šalia lietuvišką įstatymo pavadinimą. Suomijos "
        "tikriniai vardai, tokie kaip miestų pavadinimai, įmonių pavadinimai ir institucijų "
        "santrumpos (ELY, STUK, Luova, Fingrid, Traficom), yra priimtini tik kaip tikriniai vardai.\n\n"
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
    # SMR later-lifecycle phases (2026-08-12, P3-3b -- CONTENT PENDING
    # NATIVE-FINNISH/SUBJECT-MATTER REVIEWER SIGN-OFF, per 2026-08-11
    # planning; NOT yet approved to ship, see this PR's commit message).
    # Every claim below traces to a specific STUK document/section
    # actually verified this session (real fetch of stuklex.fi and
    # ohjeisto.stuk.fi, real chunk sampling of the already-ingested
    # YVL_A.1 -- see the TASO-2 coverage report, 2026-08-11/12) --
    # deliberately hedged everywhere the real source coverage is thin or
    # SMR-specific detail doesn't exist yet, matching the same honest-
    # flagging discipline as every prior sourcing round this remediation
    # (maatalous/vesivoima, hybrid co-location).
    "kayttolupa": (
        "VAIHEEN KIRJOITUSOHJE — KÄYTTÖLUPAVAIHE (SMR):\n"
        "- Rakentamislupa [YEL 990/1987 § 18] on jo myönnetty; kyse on käyttöluvasta [YEL § 20]\n"
        "- Käytä sävyä: 'haetaan käyttölupaa', 'osoitetaan', 'todennetaan', 'raportoidaan'\n"
        "- Mainitse STUK YVL A.1 luku 3.9 (käyttöluvan uusiminen, määräaikainen "
        "turvallisuusarvio) ja luku 4.4 (käyttölupahakemuksen käsittely STUK:ssa)\n"
        "- Mainitse koekäyttövaihe (commissioning) ennen kaupallista tuotantoa\n"
        "- Älä väitä käyttöluvan olevan jo myönnetty ellei hankekohtaisesti vahvistettu — "
        "käytä muotoa '[TÄYDENNETTÄVÄ – käyttöluvan myöntämispäivämäärä]'\n"
        "- Älä väitä SMR-erityisiä (esim. porrastettuja) käyttölupavaatimuksia olevan "
        "olemassa ellei RAG-lähteistössä nimenomaisesti mainita — STUK ei ole toistaiseksi "
        "julkaissut erillistä SMR-kohtaista ohjetta tälle vaiheelle"
    ),
    "purku": (
        "VAIHEEN KIRJOITUSOHJE — PURKUVAIHE (KÄYTÖSTÄPOISTO + JÄTEHUOLTO, SMR):\n"
        "- Kyse on ydinlaitoksen käytöstäpoistosta ja jätehuollosta [YVL A.1 luku 3.10; YVL D.4]\n"
        "- Käytä sävyä: 'puretaan', 'käsitellään', 'luokitellaan', 'vapautetaan valvonnasta'\n"
        "- Mainitse STUK YVL D.4 (matala- ja keskiaktiivisten ydinjätteiden käsittely ja "
        "käytöstäpoisto, 15.12.2019) — annosrajat 0,1 mSv/v normaalikäytössä, "
        "0,01 mSv/v purkutilanteessa\n"
        "- Käytetyn ydinpolttoaineen loppusijoitus ON ERI ASIA kuin matala-/keskiaktiivisen "
        "jätteen käsittely — älä sekoita näitä; viittaa YVL D.5:een (loppusijoitus) ja "
        "YVL D.7:ään (vapautumisesteet) vain nimellä, älä keksi niiden sisältöä yksityis"
        "kohtaisesti — RAG-lähteistössä ei toistaiseksi ole näiden ohjeiden tekstiä\n"
        "- POSIVA: vastaa TÄLLÄ HETKELLÄ vain Loviisan ja Olkiluodon voimalaitosten "
        "käytetystä polttoaineesta (Fortumin ja TVO:n omistama yhtiö) — ÄLÄ VÄITÄ Posivan "
        "automaattisesti vastaavan tämän hankkeen jätehuollosta; SMR-hankkeen pääsy Posivan "
        "loppusijoituslaitokseen ei ole vahvistettu — käytä muotoa "
        "'[TÄYDENNETTÄVÄ – jätehuoltoratkaisu, mahdollinen Posiva-yhteistyö tai erillinen "
        "ratkaisu]'\n"
        "- IAEA SSR-5 (Disposal of Radioactive Waste) on kansainvälinen taustastandardi "
        "jonka periaatteet STUK:n YVL D -sarja toteuttaa Suomen lainsäädännössä — älä "
        "esitä SSR-5:ttä itsenäisenä Suomessa sitovana säädöksenä"
    ),
}
# Alias so both "Rakentaminen" and "Rakentamisvaihe" (frontend values) resolve correctly
_PHASE_INSTRUCTIONS["rakentamisvaihe"] = _PHASE_INSTRUCTIONS["rakentaminen"]
_HANKE_CFG["BESS"]["context_extra_phases"]["rakentamisvaihe"] = (
    _HANKE_CFG["BESS"]["context_extra_phases"]["rakentaminen"]
)

# Phase-aware retrieval scoping (2026-08-12, P3-3a). Same shape as
# _COUNTRY_RAG_QUERIES below, keyed hanketyyppi -> vaihe -> query list.
# ADDITIVE to whatever _rag_context() already collects for that
# hanketyyppi/country (Step 1/2 below) -- not a replacement -- so a
# phase-4/5 request retrieves both the base pre-license context AND
# phase-specific later-lifecycle content, not one or the other. Empty/
# absent for every hanketyyppi and phase except SMR's kayttolupa/purku
# today, which makes this a no-op everywhere else (confirmed in this
# PR's self-test).
#
# This dict ALSO doubles as the "declared phase-4/5 support" signal for
# _generate_sections()'s PhaseContentNotAvailableError guard below --
# deliberately not a separate structure. A vaihe key present here for a
# given hanketyyppi means that hanketyyppi has genuinely been engineered
# for this later phase (SMR/kayttolupa, SMR/purku today); a vaihe key
# absent means "not recognized for this hanketyyppi," which falls through
# to the existing esiselvitys-tone fallback completely unchanged, exactly
# as before this PR -- this is what keeps e.g. a stray
# hankkeen_vaihe="kayttolupa" for BESS (nonsensical, but nothing currently
# stops ApplicationInput.hankkeen_vaihe from being any free-text string)
# from being wrongly treated as "recognized but not ready." Caught by this
# PR's own self-test before it shipped -- see the commit message.
_PHASE_RAG_QUERIES: dict[str, dict[str, list[str]]] = {
    "SMR": {
        "kayttolupa": [
            "nuclear power plant operating licence application renewal periodic safety review STUK",
            "ydinvoimalaitoksen käyttölupa hakemus uusiminen määräaikainen turvallisuusarvio",
        ],
        "purku": [
            "nuclear facility decommissioning low intermediate level radioactive waste management clearance",
            "ydinlaitoksen käytöstäpoisto matala- ja keskiaktiivinen jäte jätehuolto vapauttaminen valvonnasta",
        ],
    },
}

# Country-specific RAG queries — override Finnish default queries for country chunk retrieval.
# Used in _rag_context step 1 so cross-lingual embedding similarity stays above threshold.
_COUNTRY_RAG_QUERIES: dict[str, dict[str, list[str]]] = {
    "SE": {
        "BESS": [
            "Batterilagring energilagring tillståndsansökan Sverige miljöbalken PBL",
            "Litiumjon batteri säkerhet brandskydd bygglov Länsstyrelsen",
            "Elnätsanslutning Svenska kraftnät nättillstånd energilagring",
        ],
        "tuulivoima_maa": [
            "Vindkraft landbaserad tillstånd Länsstyrelsen artskydd miljöbalken",
            "Vindpark bygglov PBL riksintresse Energimyndigheten",
        ],
        "tuulivoima_meri": [
            "Havsbaserad vindkraft tillståndsansökan Energimyndigheten Svenska kraftnät",
        ],
        "aurinkovoima": [
            "Solcellsanläggning solpark bygglov PBL nättillstånd Sverige",
        ],
        "SMR": [
            "Kärnkraft SMR tillstånd Strålsäkerhetsmyndigheten Sverige",
        ],
    },
    "DA": {
        "BESS": [
            "Batterilagringssystem BESS tilladelse byggetilladelse Danmark",
            "Energilagring VVM-screening Energistyrelsen Energinet nettilslutning",
            "Litiumbatteri brandsikkerhed sikkerhedskrav BESS Beredskabsstyrelsen",
        ],
        "tuulivoima_maa": [
            "Vindmølle landvindmølle tilladelse VVM Energistyrelsen arealtilladelse",
        ],
        "tuulivoima_meri": [
            "Havvind havmølle tilladelse Energistyrelsen udbud havområde",
        ],
        "aurinkovoima": [
            "Solcelleanlæg solpark byggetilladelse lokalplan Danmark",
        ],
    },
    "NO": {
        "BESS": [
            "Batterilagring energilagring konsesjon NVE Norge plan bygningsloven",
            "Nettilknytning Statnett energiloven lagringsanlegg",
            "Litiumbatteri brannsikkerhet sikkerhetskrav BESS DSB",
        ],
        "tuulivoima_maa": [
            "Vindkraft landbasert konsesjon NVE energiloven arealplan",
            "Vindkraftanlegg miljøkonsekvensutredning plan bygningsloven",
        ],
        "tuulivoima_meri": [
            "Havvind offshore konsesjon NVE havenergilova Statnett",
        ],
        "aurinkovoima": [
            "Solkraft solcelle solpark konsesjon NVE plan bygningsloven Norge",
        ],
    },
    "PL": {
        "BESS": [
            "Magazyn energii BESS pozwolenie na budowę Polska URE decyzja",
            "Akumulator litowy bateria energia elektryczna PSE warunki przyłączenia",
            "Ustawa prawo energetyczne magazyn elektrownia koncesja URE",
        ],
        "tuulivoima_maa": [
            "Elektrownia wiatrowa decyzja środowiskowa OOŚ URE odległość 10H",
            "Wiatraki farma wiatrowa pozwolenie budowlane Polska plan miejscowy",
        ],
        "tuulivoima_meri": [
            "Morska farma wiatrowa Polska ustawa offshore PSE koncesja",
        ],
        "aurinkovoima": [
            "Fotowoltaika farma PV pozwolenie na budowę Polska warunki zabudowy",
        ],
        "SMR": [
            "Reaktor jądrowy SMR prawo atomowe Polska UDT dozór jądrowy",
        ],
    },
    "EU": {
        "BESS": [
            "EU energy storage battery regulation directive grid connection",
            "European Network Code RfG generator connection requirement",
            "EU Battery Regulation 2023 stationary storage lifecycle",
        ],
        "tuulivoima_maa": [
            "EU wind energy onshore directive EIA environmental impact assessment",
            "EU taxonomy renewable energy permitting reform",
        ],
        "tuulivoima_meri": [
            "EU offshore wind directive maritime spatial planning grid connection",
        ],
        "aurinkovoima": [
            "EU solar energy photovoltaic rooftop directive permitting",
        ],
        "SMR": [
            "EU nuclear safety directive Euratom SMR regulation licensing",
        ],
        "datakeskus": [
            "EU BIM Building Information Modelling construction data centre",
            "EU data centre energy efficiency directive construction permitting",
        ],
    },
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
    "LV": {
        "BESS": [
            "enerģijas uzkrāšanas sistēma BESS akumulators Latvija licencēšana SPRK",
            "litija jonu baterija ugunsdrošība pieslegums Sadales tikls AST",
            "battery energy storage permit Latvia electricity market SPRK registration",
        ],
        "tuulivoima_maa": [
            "vēja elektrostacija izvietošana būvatļauja pašvaldība Latvija SPRK",
            "vēja parks IVN VPVB vides ietekmes novērtējums teritorijas plānojums",
            "wind farm permit Latvia EIA VPVB municipal building permit setback",
        ],
        "tuulivoima_meri": [
            "jūras vēja enerģija offshore vēja parks Latvija Jūras kodekss",
            "offshore wind energy permit Latvia maritime spatial planning",
        ],
        "aurinkovoima": [
            "saules enerģija fotoelements saules parks Latvija būvatļauja",
            "solar PV permit Latvia grid connection Sadales tikls SPRK registration",
        ],
    },
    "LT": {
        "BESS": [
            "baterinis energijos kaupiklis BESS akumuliatorius Lietuva VERT licencija",
            "ličio jonų baterija priešgaisrinė sauga prisijungimas ESO LITGRID",
            "battery energy storage permit Lithuania VERT registration fire safety PAGD",
        ],
        "tuulivoima_maa": [
            "vėjo elektrinė statybos leidimas savivaldybė Lietuva VERT licencija",
            "vėjo parkas PAV poveikio aplinkai vertinimas AAA detalusis planas",
            "wind farm permit Lithuania EIA PAV municipal building permit setback radar",
        ],
        "tuulivoima_meri": [
            "jūros vėjo energija offshore vėjo parkas Lietuva jūros erdvės planavimas",
            "offshore wind energy permit Lithuania maritime spatial planning LITGRID",
        ],
        "aurinkovoima": [
            "saulės energija fotovoltika saulės parkas Lietuva statybos leidimas",
            "solar PV permit Lithuania grid connection ESO VERT registration AIEI",
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
_HANKE_CFG["smr_lv"]       = _HANKE_CFG["SMR"]
_HANKE_CFG["offshore_wind"] = _HANKE_CFG["tuulivoima_meri"]
# egs (Enhanced Geothermal System) intentionally has no alias — the aurinkovoima
# placeholder it used to point to would serve wrong content to users, worse than
# an error. Frontend marks egs comingSoon so this is never reached. Real EGS
# content is a separate future task.


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
    "ET": ("Kirjutage järgmised neli osa eesti keeles, selgelt pealkirjadega eraldatuna. "
           "Iga osa sisaldab mõttekäigu etappe nähtavate alapealkirjadena (### ANALÜÜSI, ### HANGI jne.). "
           "Lisage seadusviited nurksulgudes, nt [Ehitusseadustik EhS] või [KMH-KSH seadus]. "
           "Kui mõni teave on ebakindel, puudub või nõuab eriteadmisi, "
           "lisage märgis '⚠️ Soovitatav ekspertiisi ülevaatus' kohe selle koha järel — "
           "ärge spekuleerige ega täitke teadmislünki oletustega. "
           "OLULINE: Rajage iga soovitus, riskihinnang ja elutsükli laiendus rangelt käesolevas kontekstis "
           "leiduvatele hangitud regulatiivsetele allikatele ja pretsedentidele. "
           "Kui mõne mõttekäigu sammu jaoks teave puudub, märkige selgesõnaliselt, mis puudub. "
           "Ärge kunagi looge sisu, mis ei põhine hangitud allikatel:"),
    "DE": ("Schreiben Sie die folgenden vier Abschnitte auf Deutsch, klar durch Überschriften getrennt. "
           "Jeder Abschnitt enthält die Schritte der Argumentationskette als sichtbare Unterüberschriften "
           "(### ANALYSIEREN, ### ABRUFEN usw.). "
           "Fügen Sie Gesetzeszitate in eckigen Klammern ein, z. B. [BImSchG] oder [BauGB]. "
           "Wenn eine Information unsicher ist, fehlt oder Fachwissen erfordert, "
           "fügen Sie unmittelbar danach die Markierung '⚠️ Fachliche Überprüfung empfohlen' hinzu — "
           "spekulieren Sie nicht und füllen Sie Wissenslücken nicht mit Annahmen. "
           "WICHTIG: Stützen Sie jede Empfehlung, Risikobewertung und Lebenszyklus-Erweiterung ausschließlich auf "
           "die in diesem Kontext gefundenen abgerufenen regulatorischen Quellen und Präzedenzfälle. "
           "Wenn für einen Argumentationsschritt Informationen fehlen, geben Sie explizit an, was fehlt. "
           "Erzeugen Sie niemals Inhalte, die nicht auf den abgerufenen Quellen basieren:"),
    "LV": ("Rakstiet šādas četras sadaļas latviešu valodā, skaidri atdalot ar virsrakstiem. "
           "Katra sadaļa satur argumentācijas ķēdes soļus kā redzamas apakšvirsraksti (### ANALIZĒT, ### IEGŪT utt.). "
           "Iekļaujiet likumu citātus kvadrātiekavās, piem., [Būvniecības likums] vai [IVN likums]. "
           "Ja kāda informācija ir nenoteikta, trūkst vai prasa specializētas zināšanas, "
           "pievienojiet atzīmi '⚠️ Ieteicama ekspertu pārbaude' tieši aiz attiecīgās vietas — "
           "nespekulējiet un neaizpildiet zināšanu robus ar pieņēmumiem. "
           "SVARĪGI: Balstiet katru ieteikumu, riska novērtējumu un dzīves cikla paplašinājumu stingri uz "
           "šajā kontekstā iegūtajiem regulatīvajiem avotiem un precedentu lietām. "
           "Ja informācija kādam argumentācijas solim trūkst, skaidri norādiet, kas trūkst. "
           "Nekad neģenerējiet saturu, kas nav pamatots ar iegūtajiem avotiem:"),
    "LT": ("Parašykite šias keturias dalis lietuvių kalba, aiškiai atskirtas antraštėmis. "
           "Kiekviena dalis apima argumentacijos grandinės žingsnius kaip matomas paantraštes (### ANALIZUOTI, ### GAUTI ir kt.). "
           "Įtraukite įstatymų nuorodas laužtiniuose skliaustuose, pvz., [Statybos įstatymas] arba [PAV įstatymas]. "
           "Jei kokia nors informacija yra neaiški, trūksta arba reikalauja specializuotų žinių, "
           "iškart po tos vietos pridėkite žymą '⚠️ Rekomenduojama ekspertų peržiūra' — "
           "nespėliokite ir nepildykite žinių spragų prielaidomis. "
           "SVARBU: kiekvieną rekomendaciją, rizikos vertinimą ir gyvavimo ciklo išplėtimą grinskite "
           "griežtai šiame kontekste gautais reguliavimo šaltiniais ir precedentinėmis bylomis. "
           "Jei kuriam nors argumentacijos žingsniui trūksta informacijos, aiškiai nurodykite, kas trūksta. "
           "Niekada negenerinkite turinio, kuris nėra pagrįstas gautais šaltiniais:"),
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
    "LT": ("⚠️ SPECIALI INSTRUKCIJA ŠIAM PROJEKTO TIPUI: {hanketyyppi} projektams reguliavimo reikalavimai, "
           "saugos taisyklės ir teisinis pagrindas yra ypač tikslūs ir kintantys. "
           "Naudokite žymą '⚠️ Rekomenduojama ekspertų peržiūra' VISADA, KAI: "
           "(a) reguliavimo reikalavimas ar leidimo procedūra yra neaiški arba galimai pasikeitusi, "
           "(b) techninė riba ar parametras nėra pagrįstas pateikta dokumentacija, "
           "(c) teisinė informacija yra neišsami arba gali būti interpretuojama įvairiai. "
           "Niekada negenerinkite skaičių, terminų ar reikalavimų be dokumentuoto pagrindo."),
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
    "ET": {
        "intro":        "Koostage loataotluse mustand järgmise projekti jaoks:",
        "rag_intro":    "Allpool on asjakohane dokumentatsioon (Fingrid, Tukes, Keskkonnaministeerium):",
        "kuvaus":       "PROJEKTI KIRJELDUS",
        "perustelut":   "PÕHJENDUS JA VAJADUS",
        "luvat":        "LOAMENETLUSTE KIRJELDUS",
        "toimenpiteet": "JÄRGMISED SAMMUD",
        "kuvaus_inst":  ("Kirjutage see osa kahes nähtavas etapis:\n\n"
                         "### ANALÜÜSI\n"
                         "Tuvastage projekti peamised omadused ja riskitegurid: tüüp, asukoht, suurus, "
                         "riik ja asjakohased ametiasutused. Lisage sellele projekti tüübile tüüpilised tehnilised parameetrid.\n\n"
                         "### HANGI\n"
                         "Tuvastage RAG-kontekstist selle projektiprofiili jaoks kõige asjakohasemad regulatiivsed "
                         "nõuded ja pretsedenditud juhtumid. Kirjutage 3–4 lõiguline kirjeldus: eesmärk, tehnilised "
                         "üksikasjad, asukoht, võrguühendus ja keskkonnamõjud. Osa peab olema piisavalt "
                         "põhjalik eelkonsultatsiooniks."),
        "kuvaus_extra": " Võtke arvesse esitatud asukoha- ja keskkonnamõju teavet.",
        "perustelut_inst": ("Kirjutage see osa kahes nähtavas etapis:\n\n"
                            "### VÕRDLE\n"
                            "Võrrelge seda projekti RAG-konteksti pretsedenditud juhtumitega: millised riskid olid "
                            "olemas, kuidas need lahendati, mis muutis projektid edukaks või ebaõnnestunuks.\n\n"
                            "### HINDA\n"
                            "Määrake, millised riskid on heakskiitmise tõenäosuse jaoks kõige kriitilisemad. "
                            "Kirjutage 2–3 lõiguline põhjendus selle kohta, miks projekt on vajalik "
                            "(energiasüsteemi vaatenurgast, Soome kliimaeesmärgid, "
                            "regionaalsed majanduslikud mõjud) ja nimetage suurim üksik heakskiitmisrisk."),
        "luvat_inst":   ("### HANGI — LOAD\n"
                         "Selgitage lühidalt (1–2 lauset loa kohta), mida iga vajalik luba hõlmab, "
                         "miks see on selle projekti jaoks vajalik ja milline ametiasutus seda menetleb. "
                         "Viidake vajadusel asjakohastele pretsedentidele või erinõuetele."),
        "luvat_extra":  " Viidake eelkõige sihtasutuse {auth} protsessidele ja nõuetele.",
        "toimenpiteet_first": ("Eelkonsultatsioon valla/linna ehitusjärelevalvega + planeeringu ülevaatus — "
                               "Taotleja / {kunta} ehitusjärelevalve — 1–2 nädala jooksul"),
        "toimenpiteet_inst": ("Kirjutage see osa kahes nähtavas etapis:\n\n"
                              "### SOOVITA\n"
                              "Esimene samm on ALATI: \"{first}\".\n"
                              "Seejärel loetlege täpselt 4 konkreetset meedet, mis parandavad heakskiitmise "
                              "tõenäosust (uuringud, arvamused, projektimuudatused, dokumendid). "
                              "Vorming: number. Meede – Vastutav pool – Ajakava\n\n"
                              "### ELUTSÜKKEL\n"
                              "Samm 6: Laiendage soovitusi projekti järgmisele elutsükli etapile "
                              "AINULT SIIS, KUI hangitud kontekst sisaldab piisavalt andmeid selle etapi kohta. "
                              "Kui allikad ei kata hilisemaid etappe, märkige selgesõnaliselt: "
                              "'Hangitud lähteandmed ei ole hilisema etapi soovitusteks piisavad.' "
                              "Vorming: 6. Meede – Vastutav pool – Ajakava"),
        "toimenpiteet_vaihe": " Võtke arvesse projekti praegust etappi: {vaihe}.",
        "phase_label":        "Projekti etapp",
        "viranomainen_ohje":  ("OLULINE: Taotlus on adresseeritud ametiasutusele '{auth}'. "
                               "Kohandage sisu, struktuur ja keel vastavalt selle nõuetele. "
                               "Viidake selle ametiasutuse juhistele, vormidele ja nõuetele."),
    },
    "DE": {
        "intro":        "Erstellen Sie einen Genehmigungsantragsentwurf für das folgende Projekt:",
        "rag_intro":    "Nachfolgend finden Sie relevante Dokumentation (Fingrid, Tukes, Umweltministerium):",
        "kuvaus":       "PROJEKTBESCHREIBUNG",
        "perustelut":   "BEGRÜNDUNG UND BEDARF",
        "luvat":        "BESCHREIBUNG DER GENEHMIGUNGSVERFAHREN",
        "toimenpiteet": "NÄCHSTE SCHRITTE",
        "kuvaus_inst":  ("Schreiben Sie diesen Abschnitt in zwei sichtbaren Schritten:\n\n"
                         "### ANALYSIEREN\n"
                         "Identifizieren Sie die wichtigsten Merkmale und Risikofaktoren des Projekts: Typ, Standort, "
                         "Größe, Land und relevante Behörden. Nennen Sie typische technische Parameter für diesen Projekttyp.\n\n"
                         "### ABRUFEN\n"
                         "Identifizieren Sie die relevantesten regulatorischen Anforderungen und Präzedenzfälle für dieses "
                         "Projektprofil aus dem RAG-Kontext. Schreiben Sie eine 3–4-Absatz-Beschreibung: Zweck, technische "
                         "Details, Standort, Netzanschluss und Umweltauswirkungen. Der Abschnitt muss für eine "
                         "Vorabkonsultation ausreichend umfassend sein."),
        "kuvaus_extra": " Berücksichtigen Sie die angegebenen Standort- und Umweltauswirkungsinformationen.",
        "perustelut_inst": ("Schreiben Sie diesen Abschnitt in zwei sichtbaren Schritten:\n\n"
                            "### VERGLEICHEN\n"
                            "Vergleichen Sie dieses Projekt mit Präzedenzfällen aus dem RAG-Kontext: welche Risiken "
                            "vorhanden waren, wie sie gelöst wurden, was Projekte erfolgreich oder erfolglos machte.\n\n"
                            "### BEWERTEN\n"
                            "Bestimmen Sie, welche Risiken für die Genehmigungswahrscheinlichkeit am kritischsten sind. "
                            "Schreiben Sie eine 2–3-Absatz-Begründung, warum das Projekt notwendig ist "
                            "(Perspektive des Energiesystems, Finnlands Klimaziele, "
                            "regionale wirtschaftliche Auswirkungen) und benennen Sie das größte Einzelrisiko."),
        "luvat_inst":   ("### ABRUFEN — GENEHMIGUNGEN\n"
                         "Erklären Sie kurz (1–2 Sätze pro Genehmigung), was jede erforderliche Genehmigung abdeckt, "
                         "warum sie für dieses Projekt erforderlich ist und welche Behörde sie bearbeitet. "
                         "Verweisen Sie bei Bedarf auf relevante Präzedenzfälle oder besondere Anforderungen."),
        "luvat_extra":  " Verweisen Sie insbesondere auf die Prozesse und Anforderungen der Zielbehörde {auth}.",
        "toimenpiteet_first": ("Vorabkonsultation mit der kommunalen Bauaufsicht + Bebauungsplanprüfung — "
                               "Antragsteller / Bauaufsicht {kunta} — innerhalb von 1–2 Wochen"),
        "toimenpiteet_inst": ("Schreiben Sie diesen Abschnitt in zwei sichtbaren Schritten:\n\n"
                              "### EMPFEHLEN\n"
                              "Der erste Schritt ist IMMER: \"{first}\".\n"
                              "Listen Sie anschließend genau 4 konkrete Maßnahmen auf, die die "
                              "Genehmigungswahrscheinlichkeit verbessern (Untersuchungen, Stellungnahmen, "
                              "Planänderungen, Dokumente). "
                              "Format: Nummer. Maßnahme – Verantwortliche Partei – Zeitplan\n\n"
                              "### LEBENSZYKLUS\n"
                              "Schritt 6: Erweitern Sie die Empfehlungen auf die nächste Lebenszyklusphase des Projekts "
                              "NUR WENN der abgerufene Kontext ausreichende Daten zu dieser Phase enthält. "
                              "Wenn die Quellen spätere Phasen nicht abdecken, geben Sie explizit an: "
                              "'Abgerufene Quelldaten reichen für Empfehlungen zu späteren Phasen nicht aus.' "
                              "Format: 6. Maßnahme – Verantwortliche Partei – Zeitplan"),
        "toimenpiteet_vaihe": " Berücksichtigen Sie die aktuelle Projektphase: {vaihe}.",
        "phase_label":        "Projektphase",
        "viranomainen_ohje":  ("WICHTIG: Der Antrag richtet sich an die Behörde '{auth}'. "
                               "Passen Sie Inhalt, Struktur und Sprache an deren Anforderungen an. "
                               "Verweisen Sie auf die Richtlinien, Formulare und Anforderungen dieser Behörde."),
    },
    "LV": {
        "intro":        "Sagatavojiet atļaujas pieteikuma projektu šādam projektam:",
        "rag_intro":    "Zemāk ir attiecīgā dokumentācija (Fingrid, Tukes, Vides ministrija):",
        "kuvaus":       "PROJEKTA APRAKSTS",
        "perustelut":   "PAMATOJUMS UN NEPIECIEŠAMĪBA",
        "luvat":        "ATĻAUJU PROCEDŪRU APRAKSTS",
        "toimenpiteet": "NĀKAMIE SOĻI",
        "kuvaus_inst":  ("Rakstiet šo sadaļu divos redzamos soļos:\n\n"
                         "### ANALIZĒT\n"
                         "Identificējiet projekta galvenās īpašības un riska faktorus: tips, atrašanās vieta, izmērs, "
                         "valsts un attiecīgās iestādes. Iekļaujiet šim projekta tipam raksturīgos tehniskos parametrus.\n\n"
                         "### IEGŪT\n"
                         "Identificējiet no RAG konteksta šim projekta profilam visatbilstošākās regulatīvās prasības un "
                         "precedentu lietas. Uzrakstiet 3–4 rindkopu aprakstu: mērķis, tehniskā informācija, atrašanās "
                         "vieta, tīkla pieslēgums un ietekme uz vidi. Sadaļai jābūt pietiekami izvērstai priekšapspriedei."),
        "kuvaus_extra": " Ņemiet vērā sniegto informāciju par atrašanās vietu un ietekmi uz vidi.",
        "perustelut_inst": ("Rakstiet šo sadaļu divos redzamos soļos:\n\n"
                            "### SALĪDZINĀT\n"
                            "Salīdziniet šo projektu ar precedentu lietām no RAG konteksta: kādi riski bija klātesoši, "
                            "kā tie tika atrisināti, kas padarīja projektus veiksmīgus vai neveiksmīgus.\n\n"
                            "### NOVĒRTĒT\n"
                            "Nosakiet, kuri riski ir vissvarīgākie apstiprināšanas varbūtībai. "
                            "Uzrakstiet 2–3 rindkopu pamatojumu, kāpēc projekts ir nepieciešams "
                            "(no enerģētikas sistēmas viedokļa, Somijas klimata mērķi, "
                            "reģionālā ekonomiskā ietekme) un nosauciet lielāko atsevišķo apstiprināšanas risku."),
        "luvat_inst":   ("### IEGŪT — ATĻAUJAS\n"
                         "Īsi paskaidrojiet (1–2 teikumi par atļauju), ko aptver katra nepieciešamā atļauja, "
                         "kāpēc tā ir nepieciešama šim projektam un kura iestāde to izskata. "
                         "Ja nepieciešams, atsaucieties uz attiecīgiem precedentiem vai īpašām prasībām."),
        "luvat_extra":  " Īpaši atsaucieties uz mērķa iestādes {auth} procesiem un prasībām.",
        "toimenpiteet_first": ("Priekšapspriede ar pašvaldības būvvaldi + teritorijas plānojuma pārskatīšana — "
                               "Pieteicējs / {kunta} būvvalde — 1–2 nedēļu laikā"),
        "toimenpiteet_inst": ("Rakstiet šo sadaļu divos redzamos soļos:\n\n"
                              "### IESAKĀT\n"
                              "Pirmais solis VIENMĒR ir: \"{first}\".\n"
                              "Pēc tam uzskaitiet tieši 4 konkrētas darbības, kas uzlabo apstiprināšanas varbūtību "
                              "(izpētes, atzinumi, projekta izmaiņas, dokumenti). "
                              "Formāts: numurs. Darbība – Atbildīgā puse – Grafiks\n\n"
                              "### DZĪVES CIKLS\n"
                              "6. solis: Paplašiniet ieteikumus uz nākamo projekta dzīves cikla posmu "
                              "TIKAI TAD, JA iegūtais konteksts satur pietiekamus datus par šo posmu. "
                              "Ja avoti neaptver vēlākos posmus, skaidri norādiet: "
                              "'Iegūtie avotu dati nav pietiekami vēlāka posma ieteikumiem.' "
                              "Formāts: 6. Darbība – Atbildīgā puse – Grafiks"),
        "toimenpiteet_vaihe": " Ņemiet vērā pašreizējo projekta posmu: {vaihe}.",
        "phase_label":        "Projekta posms",
        "viranomainen_ohje":  ("SVARĪGI: Pieteikums ir adresēts iestādei '{auth}'. "
                               "Pielāgojiet saturu, struktūru un valodu tās prasībām. "
                               "Atsaucieties uz šīs iestādes vadlīnijām, veidlapām un prasībām."),
    },
    "LT": {
        "intro":        "Parengkite leidimo paraiškos projektą šiam projektui:",
        "rag_intro":    "Žemiau pateikiama susijusi dokumentacija (Fingrid, Tukes, Aplinkos ministerija):",
        "kuvaus":       "PROJEKTO APRAŠYMAS",
        "perustelut":   "PAGRINDIMAS IR POREIKIS",
        "luvat":        "LEIDIMŲ PROCEDŪROS APRAŠYMAS",
        "toimenpiteet": "TOLESNI ŽINGSNIAI",
        "kuvaus_inst":  ("Parašykite šį skyrių dviem matomais žingsniais:\n\n"
                         "### ANALIZUOTI\n"
                         "Nustatykite pagrindines projekto charakteristikas ir rizikos veiksnius: tipą, vietą, dydį, "
                         "šalį, susijusias institucijas. Įtraukite šiam projekto tipui būdingus techninius parametrus.\n\n"
                         "### GAUTI\n"
                         "Nustatykite iš RAG konteksto šiam projekto profiliui aktualiausius reguliavimo reikalavimus "
                         "ir precedentines bylas. Parašykite 3–4 pastraipų aprašymą: tikslas, techninė informacija, "
                         "vieta, prijungimas prie tinklo ir poveikis aplinkai. Skyrius turi būti pakankamai išsamus "
                         "išankstinei konsultacijai."),
        "kuvaus_extra": " Atsižvelkite į pateiktą informaciją apie vietą ir poveikį aplinkai.",
        "perustelut_inst": ("Parašykite šį skyrių dviem matomais žingsniais:\n\n"
                            "### PALYGINTI\n"
                            "Palyginkite šį projektą su precedentinėmis bylomis iš RAG konteksto: kokios rizikos "
                            "buvo, kaip jos buvo išspręstos, kas lėmė projektų sėkmę ar nesėkmę.\n\n"
                            "### ĮVERTINTI\n"
                            "Nustatykite, kurios rizikos yra svarbiausios patvirtinimo tikimybei šiuo atveju. "
                            "Parašykite 2–3 pastraipų pagrindimą, kodėl projektas yra būtinas "
                            "(energetikos sistemos požiūriu, Suomijos klimato tikslai, "
                            "regioninis ekonominis poveikis), ir nurodykite didžiausią pavienę patvirtinimo riziką."),
        "luvat_inst":   ("### GAUTI — LEIDIMAI\n"
                         "Trumpai paaiškinkite (1–2 sakiniai kiekvienam leidimui), ką apima kiekvienas reikalingas "
                         "leidimas, kodėl jis reikalingas šiam projektui ir kuri institucija jį nagrinėja. "
                         "Prireikus nurodykite susijusius precedentus ar specialius reikalavimus."),
        "luvat_extra":  " Ypač atsižvelkite į paskirties institucijos {auth} procesus ir reikalavimus.",
        "toimenpiteet_first": ("Išankstinė konsultacija su savivaldybės statybos priežiūros institucija + "
                               "teritorijų planavimo patikra — Pareiškėjas / {kunta} statybos priežiūra — "
                               "per 1–2 savaites"),
        "toimenpiteet_inst": ("Parašykite šį skyrių dviem matomais žingsniais:\n\n"
                              "### REKOMENDUOTI\n"
                              "Pirmasis žingsnis VISADA yra: \"{first}\".\n"
                              "Tada išvardykite lygiai 4 konkrečius veiksmus, kurie padidina patvirtinimo "
                              "tikimybę (tyrimai, išvados, projekto pakeitimai, dokumentai). "
                              "Formatas: numeris. Veiksmas – Atsakinga šalis – Terminas\n\n"
                              "### GYVAVIMO CIKLAS\n"
                              "6 žingsnis: Išplėskite rekomendacijas į kitą projekto gyvavimo ciklo etapą "
                              "TIK TUO ATVEJU, JEI gautame kontekste yra pakankamai duomenų apie tą etapą. "
                              "Jei šaltiniai neapima vėlesnių etapų, aiškiai nurodykite: "
                              "'Nepakanka šaltinių duomenų vėlesnio etapo rekomendacijoms.' "
                              "Formatas: 6. Veiksmas – Atsakinga šalis – Terminas"),
        "toimenpiteet_vaihe": " Atsižvelkite į dabartinį projekto etapą: {vaihe}.",
        "phase_label":        "Projekto etapas",
        "viranomainen_ohje":  ("SVARBU: Paraiška adresuota institucijai '{auth}'. "
                               "Pritaikykite turinį, struktūrą ir kalbą pagal jos reikalavimus. "
                               "Remkitės tos institucijos gairėmis, formomis ir reikalavimais."),
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
    "YSL 527/2014":                                         {"EN": "Environmental Protection Act (YSL 527/2014)",                          "SE": "Miljöbalken kap. 9 (miljöfarlig verksamhet)",                      "DA": "Miljøbeskyttelsesloven (YSL 527/2014)",                   "NO": "Miljøvernloven (YSL 527/2014)",                    "PL": "Ustawa o ochronie środowiska (YSL 527/2014)"},
    "Rakentamislaki 751/2023 / MRL 132/1999":               {"EN": "Building Act / Land Use and Building Act (751/2023 / 132/1999)",       "SE": "Plan- och bygglagen (2010:900)",    "DA": "Byggelov / Planlægningslov (751/2023 / 132/1999)",        "NO": "Byggelov / Plan- og bygningsloven (751/2023 / 132/1999)", "PL": "Prawo budowlane / Ustawa o zagospodarowaniu przestrzennym (751/2023 / 132/1999)"},
    "Rakentamislaki 751/2023":                               {"EN": "Building Act 751/2023",                                                "SE": "Plan- och bygglagen (2010:900)",                                   "DA": "Byggeloven 751/2023",                                     "NO": "Byggeloven 751/2023",                              "PL": "Prawo budowlane 751/2023"},
    "Rakentamislaki 751/2023, 44 §":                        {"EN": "Building Act 751/2023, § 44",                                          "SE": "Plan- och bygglagen (2010:900)",                             "DA": "Byggeloven 751/2023, § 44",                               "NO": "Byggeloven 751/2023, § 44",                        "PL": "Prawo budowlane 751/2023, § 44"},
    "Pelastuslaki 379/2011, 15 §":                          {"EN": "Rescue Services Act 379/2011, § 15",                                   "SE": "Lag (2003:778) om skydd mot olyckor",                        "DA": "Redningstjenesteloven 379/2011, § 15",                    "NO": "Brannvernloven 379/2011, § 15",                    "PL": "Ustawa o ochronie przeciwpożarowej 379/2011, § 15"},
    "Sähkömarkkinalaki 588/2013":                           {"EN": "Electricity Market Act (588/2013)",                                    "SE": "Elmarknadslagen (588/2013)",                           "DA": "Elmarkedsloven (588/2013)",                               "NO": "Energiloven (588/2013)",                           "PL": "Ustawa o rynku energii elektrycznej (588/2013)"},
    "Maa-aineslaki 555/1981":                               {"EN": "Extractable Land Resources Act (555/1981)",                            "SE": "Marktäktslagen (555/1981)",                            "DA": "Råstofloven (555/1981)",                                  "NO": "Mineralressursloven (555/1981)",                   "PL": "Ustawa o kopalinach pospolitych (555/1981)"},
    "YVA-laki 252/2017":                                    {"EN": "EIA Act (252/2017)",                                                   "SE": "Miljöbalken kap. 6 (miljöbedömningar)",                                 "DA": "VVM-loven (252/2017)",                                    "NO": "KU-loven (252/2017)",                              "PL": "Ustawa OOŚ (252/2017)"},
    "YVA-laki 252/2017 (kynnykset ylittyessä)":            {"EN": "EIA Act 252/2017 (when thresholds exceeded)",                          "SE": "Miljöbalken kap. 6 (miljöbedömningar), vid tröskelöverskridning",        "DA": "VVM-loven 252/2017 (når grænseværdier overskrides)",      "NO": "KU-loven 252/2017 (når terskler overskrides)",    "PL": "Ustawa OOŚ 252/2017 (gdy progi są przekroczone)"},
    "YVA-laki 252/2017 (≥50 ha hankkeet)":                 {"EN": "EIA Act 252/2017 (≥50 ha projects)",                                   "SE": "Miljöbalken kap. 6 (miljöbedömningar)",                 "DA": "VVM-loven 252/2017 (≥50 ha projekter)",                   "NO": "KU-loven 252/2017 (≥50 ha prosjekter)",           "PL": "Ustawa OOŚ 252/2017 (≥50 ha projekty)"},
    "MRL 132/1999 § 77a":                                   {"EN": "Land Use and Building Act 132/1999, § 77a",                            "SE": "Plan- och bygglagen (2010:900)",                  "DA": "Planlægningsloven 132/1999, § 77a",                       "NO": "Plan- og bygningsloven 132/1999, § 77a",          "PL": "Ustawa o zagospodarowaniu przestrzennym 132/1999, § 77a"},
    "MRL 132/1999 § 137":                                   {"EN": "Land Use and Building Act 132/1999, § 137",                            "SE": "Plan- och bygglagen (2010:900)",                  "DA": "Planlægningsloven 132/1999, § 137",                       "NO": "Plan- og bygningsloven 132/1999, § 137",          "PL": "Ustawa o zagospodarowaniu przestrzennym 132/1999, § 137"},
    "MRL 197 §":                                            {"EN": "Land Use and Building Act, § 197",                                     "SE": "Plan- och bygglagen, § 197",                           "DA": "Planlægningsloven, § 197",                                "NO": "Plan- og bygningsloven, § 197",                    "PL": "Ustawa o zagospodarowaniu przestrzennym, § 197"},
    "MRL 132/1999 § 91a":                                   {"EN": "Land Use and Building Act 132/1999, § 91a",                            "SE": "Plan- och bygglagen (2010:900)",                  "DA": "Planlægningsloven 132/1999, § 91a",                       "NO": "Plan- og bygningsloven 132/1999, § 91a",          "PL": "Ustawa o zagospodarowaniu przestrzennym 132/1999, § 91a"},
    "MRL 132/1999 § 9":                                     {"EN": "Land Use and Building Act 132/1999, § 9",                              "SE": "Plan- och bygglagen (2010:900)",                    "DA": "Planlægningsloven 132/1999, § 9",                         "NO": "Plan- og bygningsloven 132/1999, § 9",            "PL": "Ustawa o zagospodarowaniu przestrzennym 132/1999, § 9"},
    "MRL 132/1999":                                         {"EN": "Land Use and Building Act (132/1999)",                                 "SE": "Plan- och bygglagen (2010:900)",                       "DA": "Planlægningsloven (132/1999)",                            "NO": "Plan- og bygningsloven (132/1999)",                "PL": "Ustawa o zagospodarowaniu przestrzennym (132/1999)"},
    "Ilmailulaki 864/2014":                                 {"EN": "Aviation Act (864/2014)",                                              "SE": "Luftfartslagen (864/2014)",                            "DA": "Luftfartsloven (864/2014)",                               "NO": "Luftfartsloven (864/2014)",                        "PL": "Ustawa lotnicza (864/2014)"},
    "Maakaari 540/1995":                                    {"EN": "Code of Real Estate (540/1995)",                                       "SE": "Jordabalken (540/1995)",                               "DA": "Tinglysningsloven (540/1995)",                            "NO": "Eiendomsloven (540/1995)",                         "PL": "Ustawa o nieruchomościach (540/1995)"},
    "Vesilaki 587/2011":                                    {"EN": "Water Act (587/2011)",                                                 "SE": "Miljöbalken kap. 11 (vattenverksamhet)",                               "DA": "Vandloven (587/2011)",                                    "NO": "Vannressursloven (587/2011)",                      "PL": "Prawo wodne (587/2011)"},
    "Vesilaki 587/2011 § 3:2":                              {"EN": "Water Act 587/2011, § 3:2",                                            "SE": "Miljöbalken kap. 11 (vattenverksamhet)",                          "DA": "Vandloven 587/2011, § 3:2",                               "NO": "Vannressursloven 587/2011, § 3:2",                 "PL": "Prawo wodne 587/2011, § 3:2"},
    "Merilaki 674/1994":                                    {"EN": "Maritime Act (674/1994)",                                              "SE": "Sjölagen (674/1994)",                                  "DA": "Søloven (674/1994)",                                      "NO": "Sjøloven (674/1994)",                              "PL": "Kodeks morski (674/1994)"},
    "Merenkulkulaki 1672/2009":                             {"EN": "Maritime Navigation Act (1672/2009)",                                  "SE": "Sjöfartslagen (1672/2009)",                            "DA": "Søfartsloven (1672/2009)",                                "NO": "Navigasjonsloven (1672/2009)",                     "PL": "Ustawa o żegludze morskiej (1672/2009)"},
    "Laki alueiden käytöstä":                               {"EN": "Act on Land Use",                                                      "SE": "Lagen om områdesanvändning",                           "DA": "Lov om arealanvendelse",                                  "NO": "Lov om arealbruk",                                 "PL": "Ustawa o użytkowaniu gruntów"},
    "Rakentamislaki 751/2023 / MRL 132/1999 § 125–126":    {"EN": "Building Act / Land Use and Building Act (751/2023 / 132/1999 §§ 125–126)", "SE": "Plan- och bygglagen (2010:900)", "DA": "Byggelov / Planlægningslov (751/2023 / 132/1999 §§ 125–126)", "NO": "Byggelov / Plan- og bygningsloven (751/2023 / 132/1999 §§ 125–126)", "PL": "Prawo budowlane (751/2023 / 132/1999 §§ 125–126)"},
    "Rakentamislaki 751/2023 / MRL 132/1999 § 126":         {"EN": "Building Act / Land Use and Building Act (751/2023 / 132/1999, § 126)", "SE": "Plan- och bygglagen (2010:900)", "DA": "Byggelov / Planlægningslov (751/2023 / 132/1999, § 126)", "NO": "Byggelov / Plan- og bygningsloven (751/2023 / 132/1999, § 126)", "PL": "Prawo budowlane (751/2023 / 132/1999, § 126)"},
    "Ydinenergialaki 990/1987 § 11":                        {"EN": "Nuclear Energy Act 990/1987, § 11",                                   "SE": "Lag (1984:3) om kärnteknisk verksamhet",                        "DA": "Kerneenergieloven 990/1987, § 11",                        "NO": "Atomenergiloven 990/1987, § 11",                   "PL": "Ustawa o energii jądrowej 990/1987, § 11"},
    "YEL 990/1987 § 18":                                    {"EN": "Nuclear Energy Act 990/1987, § 18",                                   "SE": "Lag (1984:3) om kärnteknisk verksamhet",                        "DA": "Kerneenergieloven 990/1987, § 18",                        "NO": "Atomenergiloven 990/1987, § 18",                   "PL": "Ustawa o energii jądrowej 990/1987, § 18"},
    "YEL 990/1987 § 20":                                    {"EN": "Nuclear Energy Act 990/1987, § 20",                                   "SE": "Lag (1984:3) om kärnteknisk verksamhet",                        "DA": "Kerneenergieloven 990/1987, § 20",                        "NO": "Atomenergiloven 990/1987, § 20",                   "PL": "Ustawa o energii jądrowej 990/1987, § 20"},
    "Kalastuslaki 379/2015":                                {"EN": "Fisheries Act (379/2015)",                                             "SE": "Fiskelagen (379/2015)",                                "DA": "Fiskeriloven (379/2015)",                                 "NO": "Fiskeloven (379/2015)",                            "PL": "Ustawa o rybołówstwie (379/2015)"},
    "Säteilylaki 859/2018":                                 {"EN": "Radiation Act (859/2018)",                                             "SE": "Strålningslagen (859/2018)",                           "DA": "Strålingsloven (859/2018)",                               "NO": "Strålevernloven (859/2018)",                       "PL": "Ustawa prawo atomowe (859/2018)"},
    "Kemikaaliturvallisuuslaki 390/2005":                   {"EN": "Chemicals Safety Act (390/2005)",                                      "SE": "Kemikaliesäkerhetslagen (390/2005)",                   "DA": "Kemikaliesikkerhedsloven (390/2005)",                     "NO": "Kjemikaliesikkerhetsloven (390/2005)",             "PL": "Ustawa o bezpieczeństwie chemicznym (390/2005)"},
    "Kemikaaliturvallisuuslaki 390/2005 (BESS)":           {"EN": "Chemicals Safety Act 390/2005 (BESS)",                                  "SE": "Kemikaliesäkerhetslagen 390/2005 (BESS)",              "DA": "Kemikaliesikkerhedsloven 390/2005 (BESS)",                "NO": "Kjemikaliesikkerhetsloven 390/2005 (BESS)",        "PL": "Ustawa o bezpieczeństwie chemicznym 390/2005 (BESS)"},
    "Luonnonsuojelulaki 9/2023":                            {"EN": "Nature Conservation Act (9/2023)",                                     "SE": "Naturvårdslagen (9/2023)",                             "DA": "Naturbeskyttelsesloven (9/2023)",                         "NO": "Naturmangfoldloven (9/2023)",                      "PL": "Ustawa o ochronie przyrody (9/2023)"},
    "Maantielaki 503/2005 (tiealueet)":                     {"EN": "Highways Act 503/2005 (road areas)",                                   "SE": "Väglagen 503/2005 (vägområden)",                       "DA": "Vejloven 503/2005 (vejarealer)",                          "NO": "Vegloven 503/2005 (vegarealer)",                   "PL": "Ustawa o drogach publicznych 503/2005 (obszary drogowe)"},
    "Patoturvallisuuslaki 494/2009":                        {"EN": "Dam Safety Act (494/2009)",                                            "SE": "Förordning (2014:214) om dammsäkerhet",                         "DA": "Dæmningssikkerhedsloven (494/2009)",                      "NO": "Damsikkerhetsloven (494/2009)",                    "PL": "Ustawa o bezpieczeństwie budowli piętrzących (494/2009)"},
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

def _t_liite(lang: str, fi: str, country: str = "FI") -> str:
    """Liitteen nimen käännös. Ajaa myös Traficom/Finavia-, STUK- ja
    ELY-keskus-korvaukset (ks. _fix_hardcoded_traficom, _fix_hardcoded_stuk,
    _fix_hardcoded_ely) — _LIITE_TRANS-taulukon kiinteät viranomaismaininnat
    ovat samat joka kielessä, koska _t_liite ei aiemmin saanut maatietoa
    lainkaan; sama pysyvä ongelma kuin context_extra-tekstissä, mutta eri
    koodipolku (staattinen käännöstaulukko, ei Claude-generoitu proosa) —
    ei siis liity RAG-kontekstiin lainkaan."""
    text = _t_str(lang, fi, _LIITE_TRANS)
    text = _fix_hardcoded_traficom(text, country)
    text = _fix_hardcoded_stuk(text, country)
    text = _fix_hardcoded_ely(text, country)
    return text


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
                            "(ks. Liite 2: Maankäyttöselvitys PDF). Selvitys sisältää kiinteistötiedot, "
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
        # RAQS self-assessment page strings
        "raqs_heading":    "RAQS-arviointi",
        "raqs_subtitle":   ("AI-itsearvio (Regulatory Assurance & Quality System) — ei korvaa asiantuntijatarkistusta. "
                            "Pisteet 1–5 per kriteeri; korkea pistemäärä = parempi laatu."),
        "raqs_overall":    "Kokonaispistemäärä",
        "raqs_disclaimer": ("RAQS-arvio tuotettu automaattisesti tekoälyn avulla. "
                            "Arvioi aina luonnos ennen viranomaisen käsittelyyn toimittamista."),
        "raqs_lbl_viittaukset":    "Lakiviittaukset",
        "raqs_lbl_lupakattavuus":  "Lupakattavuus",
        "raqs_lbl_epävarmuus":     "Epävarmuudenhallinta",
        "raqs_lbl_kattavuus":      "Sisällön kattavuus",
        "raqs_lbl_valmisteluaste": "Hakemusvalmisteluaste",
        # Täydennyslista (completion checklist) page strings
        "taydennys_heading":  "Täydennyslista",
        "taydennys_subtitle": ("Alla olevat kohdat vaativat täydentämistä ennen hakemuksen jättämistä "
                               "viranomaiselle. Löydät nämä kohdat myös tekstissä oransseina "
                               "[TÄYDENNETTÄVÄ]-merkintöinä."),
        "taydennys_osio":     "Osio",
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
                            "land use report (see Appendix 2: Land Use Report PDF). The report includes "
                            "property information, zoning status, protected areas and groundwater area data, "
                            "in accordance with the requirements for site surveys under applicable "
                            "national planning legislation."),
        "kaava_BESS":      ("<b>Zoning status (most critical pre-study item):</b> The zoning status of the "
                            "BESS project site must be determined first. In most municipalities, siting a "
                            "battery energy storage system requires a detailed plan or a planning permit. "
                            "Zoning status has the greatest impact on the total duration of the permit "
                            "process — pre-consultation with the building authority is the first step."),
        "kaava_tuuli":     ("<b>Zoning status and EIA requirement:</b> A wind power project almost always "
                            "requires a local master plan or detailed plan (Land Use and Building Act "
                            "132/1999, § 77a). The EIA procedure (EIA Act 252/2017) is mandatory for "
                            "projects ≥10 MW or ≥5 turbines — zoning and EIA processes often run in "
                            "parallel, taking 3–6 years combined. Zoning is resolved first before other "
                            "permits."),
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
        # RAQS self-assessment page strings
        "raqs_heading":    "RAQS Assessment",
        "raqs_subtitle":   ("AI self-assessment (Regulatory Assurance & Quality System) — does not replace expert review. "
                            "Score 1–5 per criterion; higher score = better quality."),
        "raqs_overall":    "Overall score",
        "raqs_disclaimer": ("RAQS assessment generated automatically using AI. "
                            "Always review the draft before submitting to the permitting authority."),
        "raqs_lbl_viittaukset":    "Legal References",
        "raqs_lbl_lupakattavuus":  "Permit Coverage",
        "raqs_lbl_epävarmuus":     "Uncertainty Management",
        "raqs_lbl_kattavuus":      "Content Depth",
        "raqs_lbl_valmisteluaste": "Application Readiness",
        # Completion checklist page strings
        "taydennys_heading":  "Completion Checklist",
        "taydennys_subtitle": ("The following items require completion before the application is submitted to the authority. "
                               "You will also find these items in the text as orange [REQUIRED] markers."),
        "taydennys_osio":     "Section",
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
                            "markanvändningsutredning (se Bilaga 2: Markanvändningsutredning PDF). "
                            "Utredningen innehåller fastighetsuppgifter, planläggningsstatus, "
                            "skyddsområden och grundvattenuppgifter, i enlighet med krav på "
                            "platsutredning enligt plan- och bygglagen (PBL 2010:900), 10 kap."),
        "kaava_BESS":      ("<b>Planläggningsstatus (viktigaste förundersökningspunkten):</b> "
                            "Planläggningsstatusen för BESS-projektplatsen måste utredas först. I de flesta "
                            "kommuner kräver placering av ett batterienergilager en detaljplan eller "
                            "planeringstillstånd. Planläggningsstatus påverkar mest den totala längden på "
                            "tillståndsprocessen — förkonsultation med byggnadstillsynen är det första steget."),
        "kaava_tuuli":     ("<b>Planläggningsstatus och MKB-krav:</b> Ett vindkraftsprojekt kräver nästan "
                            "alltid en lokal översiktsplan eller detaljplan (Plan- och bygglagen "
                            "2010:900). MKB-förfarandet (Miljöbalken kap. 6) är obligatoriskt för "
                            "projekt ≥10 MW eller ≥5 verk — plan- och MKB-processerna löper ofta "
                            "parallellt och tar sammanlagt 3–6 år. Planläggningsstatus klarläggs först."),
        "kaava_SMR":       ("<b>SSM-förlicensiering (viktigaste första steget):</b> För ett kärnkraftverk "
                            "krävs regeringens tillståndsprövning (Lag (1984:3) om kärnteknisk "
                            "verksamhet) och Strålsäkerhetsmyndighetens (SSM) förlicensieringsförfarande "
                            "innan alla andra tillstånd. SSM:s säkerhetsgranskning inleder processen. "
                            "Planläggning hanteras parallellt men kärnsäkerhetsförfarandet är "
                            "den dominerande faktorn."),
        "kaava_aurinkovoima": ("<b>Bygglov och planläggning:</b> En solcellsanläggning kräver normalt bygglov "
                            "eller anmälan enligt Plan- och bygglagen (2010:900) — mindre anläggningar kan i "
                            "vissa fall omfattas av bygglovsbefriade åtgärder. MKB krävs normalt inte för "
                            "mindre projekt. Planläggningsstatus måste ändå kontrolleras — "
                            "planeringstillstånd kan behövas utanför detaljplaneområden."),
        "kaava_generic":   ("<b>Planläggningsstatus:</b> Gällande planläggningsstatus för projektplatsen "
                            "måste verifieras i ett förkonsultationsmöte med byggnadstillsynen innan "
                            "tillståndsansökan lämnas in. Planläggning påverkar direkt varaktigheten och "
                            "kraven i tillståndsprocessen — byggande kräver ofta en detaljplan, en ändring "
                            "av en sådan eller ett planeringstillstånd."),
        "nce_info_desc":   ("NCE Permit AI är ett AI-drivet verktyg för att automatisera "
                            "tillståndsprocesser inom energiprojekt."),
        # RAQS self-assessment page strings
        "raqs_heading":    "RAQS-bedömning",
        "raqs_subtitle":   ("AI-självbedömning (Regulatory Assurance & Quality System) — ersätter inte expertgranskning. "
                            "Poäng 1–5 per kriterium; högre poäng = bättre kvalitet."),
        "raqs_overall":    "Totalpoäng",
        "raqs_disclaimer": ("RAQS-bedömning genererad automatiskt med AI. "
                            "Granska alltid utkastet innan det skickas till tillståndsmyndigheten."),
        "raqs_lbl_viittaukset":    "Lagreferenser",
        "raqs_lbl_lupakattavuus":  "Tillståndstäckning",
        "raqs_lbl_epävarmuus":     "Osäkerhetshantering",
        "raqs_lbl_kattavuus":      "Innehållsdjup",
        "raqs_lbl_valmisteluaste": "Ansökningsberedskap",
        # Completion checklist page strings
        "taydennys_heading":  "Kompletteringslista",
        "taydennys_subtitle": ("Följande punkter kräver komplettering innan ansökan lämnas in till myndigheten. "
                               "Du hittar även dessa punkter i texten som orange [KOMPLETTERA]-markeringar."),
        "taydennys_osio":     "Avsnitt",
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
        "raqs_heading":    "RAQS-vurdering",
        "raqs_subtitle":   ("AI-selvvurdering (Regulatory Assurance & Quality System) — erstatter ikke "
                            "ekspertgennemgang. Point 1–5 pr. kriterium; høj score = bedre kvalitet."),
        "raqs_overall":    "Samlet score",
        "raqs_disclaimer": ("RAQS-vurdering genereret automatisk med AI. "
                            "Gennemgå altid udkastet før indsendelse til myndighedsbehandling."),
        "raqs_lbl_viittaukset":    "Lovhenvisninger",
        "raqs_lbl_lupakattavuus":  "Tilladelsesdækning",
        "raqs_lbl_epävarmuus":     "Usikkerhedshåndtering",
        "raqs_lbl_kattavuus":      "Indholdsdybde",
        "raqs_lbl_valmisteluaste": "Ansøgningsberedskab",
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
                            "arealanvendelsesrapport (se Bilag 2: Arealanvendelsesrapport PDF). "
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
                            "en lokaloversigtsplan eller lokalplan (Planlægningsloven 132/1999, § 77a). "
                            "VVM-proceduren (VVM-loven 252/2017) er obligatorisk for projekter ≥10 MW "
                            "eller ≥5 møller — plan- og VVM-processerne forløber ofte parallelt og tager "
                            "tilsammen 3–6 år. Planlægningsstatus afklares først."),
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
        "raqs_heading":    "RAQS-vurdering",
        "raqs_subtitle":   ("AI-selvvurdering (Regulatory Assurance & Quality System) — erstatter ikke "
                            "ekspertgjennomgang. Poeng 1–5 per kriterium; høy poengsum = bedre kvalitet."),
        "raqs_overall":    "Total poengsum",
        "raqs_disclaimer": ("RAQS-vurdering generert automatisk med AI. "
                            "Gjennomgå alltid utkastet før innsending til myndighetsbehandling."),
        "raqs_lbl_viittaukset":    "Lovhenvisninger",
        "raqs_lbl_lupakattavuus":  "Tillatelsesdekning",
        "raqs_lbl_epävarmuus":     "Usikkerhetshåndtering",
        "raqs_lbl_kattavuus":      "Innholdsdybde",
        "raqs_lbl_valmisteluaste": "Søknadsberedskap",
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
                            "arealbruksrapport (se Vedlegg 2: Arealbruksrapport PDF). "
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
                            "en kommunedelplan eller reguleringsplan (Plan- og bygningsloven 132/1999, "
                            "§ 77a). KU-prosedyren (KU-loven 252/2017) er obligatorisk for prosjekter "
                            "≥10 MW eller ≥5 turbiner — plan- og KU-prosessene løper ofte parallelt og "
                            "tar til sammen 3–6 år. Reguleringstatus avklares først."),
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
        "raqs_heading":    "Ocena RAQS",
        "raqs_subtitle":   ("Samoocena AI (Regulatory Assurance & Quality System) — nie zastępuje "
                            "weryfikacji przez eksperta. Punktacja 1–5 na kryterium; wyższy wynik = lepsza jakość."),
        "raqs_overall":    "Wynik łączny",
        "raqs_disclaimer": ("Ocena RAQS wygenerowana automatycznie przy użyciu AI. "
                            "Zawsze zweryfikuj projekt przed złożeniem do organu."),
        "raqs_lbl_viittaukset":    "Odniesienia prawne",
        "raqs_lbl_lupakattavuus":  "Zakres zezwoleń",
        "raqs_lbl_epävarmuus":     "Zarządzanie niepewnością",
        "raqs_lbl_kattavuus":      "Głębokość treści",
        "raqs_lbl_valmisteluaste": "Gotowość wniosku",
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
                            "dotyczącym zagospodarowania terenu (zob. Załącznik 2: Raport PDF). Raport "
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
                            "wymaga planu miejscowego (Ustawa o zagospodarowaniu przestrzennym 132/1999, "
                            "§ 77a). Procedura OOŚ (Ustawa OOŚ 252/2017) jest obowiązkowa dla projektów "
                            "≥10 MW lub ≥5 turbin — procesy planistyczne i OOŚ przebiegają często "
                            "równolegle i trwają łącznie 3–6 lat. Status planistyczny ustala się "
                            "w pierwszej kolejności."),
        "kaava_SMR":       ("<b>Wstępne licencjonowanie STUK (najważniejszy pierwszy krok):</b> W przypadku "
                            "obiektu jądrowego decyzja zasadnicza Rady Stanu (ustawa o energii jądrowej "
                            "990/1987, § 11) i procedura wstępnego licencjonowania STUK są obowiązkowe "
                            "przed wszystkimi innymi zezwoleniami. Raport bezpieczeństwa STUK zgodny "
                            "z wytycznymi YVL inicjuje proces. Planowanie odbywa się równolegle, "
                            "ale procedura bezpieczeństwa jądrowego jest czynnikiem dominującym."),
        "kaava_aurinkovoima": ("<b>Pozwolenie na roboty budowlane lub pozwolenie na budowę — i planowanie:</b> "
                            "Dla małej elektrowni słonecznej (poniżej ok. 1 ha) często wystarczy zgłoszenie "
                            "robót budowlanych zamiast pełnego pozwolenia na budowę (Prawo budowlane "
                            "751/2023 / 132/1999, § 126). OOŚ nie jest wymagana dla projektów poniżej "
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
        "raqs_heading":    "RAQS-Bewertung",
        "raqs_subtitle":   ("KI-Selbstbewertung (Regulatory Assurance & Quality System) — ersetzt keine "
                            "fachkundige Prüfung. Punkte 1–5 pro Kriterium; hohe Punktzahl = bessere Qualität."),
        "raqs_overall":    "Gesamtpunktzahl",
        "raqs_disclaimer": ("RAQS-Bewertung automatisch mit KI erstellt. "
                            "Prüfen Sie den Entwurf stets vor der Einreichung bei der Behörde."),
        "raqs_lbl_viittaukset":    "Gesetzesverweise",
        "raqs_lbl_lupakattavuus":  "Genehmigungsabdeckung",
        "raqs_lbl_epävarmuus":     "Unsicherheitsmanagement",
        "raqs_lbl_kattavuus":      "Inhaltstiefe",
        "raqs_lbl_valmisteluaste": "Antragsreife",
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
    "LT": {
        "sub_title": "Statybos leidimo paraiškos projektas",
        "esiselvitys_sub": "Išankstinio tyrimo ir konsultacijos medžiaga — Parengta statybos leidimo išankstinei konsultacijai",
        "lupavaihe_sub": "Leidimo etapas — Statybos leidimo paraiškos projektas",
        "rakentaminen_sub": "Statybos etapas — Vykdymas ir priežiūra",
        "disclaimer_h": "DI SUGENERUOTAS PROJEKTAS — REIKALINGA EKSPERTŲ PERŽIŪRA",
        "disclaimer_b": "Šis dokumentas yra dirbtinio intelekto sugeneruotas projektas, kurį prieš teikiant institucijoms būtina peržiūrėti kvalifikuotam ekspertui.",
        "nce_speed_note": "NCE Permit AI sugeneruoja paraiškos projektą per kelias minutes. Institucijos nagrinėjimo laikas yra atskiras procesas ir priklauso nuo projekto tipo (žr. žemiau).",
        "arviointiviive_lbl": "Institucijos nagrinėjimo laikas",
        "m_hakija": "Pareiškėjas",
        "m_ytunnus": "Įmonės kodas",
        "m_hanketyyppi": "Projekto tipas",
        "m_teho": "Galia",
        "m_kunta": "Savivaldybė",
        "m_kt": "Kadastro numeris",
        "m_maa": "Šalis",
        "m_laadittu": "Parengta",
        "m_laatinut_lbl": "Parengė",
        "m_laatinut": "NCE Permit AI (su DI pagalba)",
        "sec1": "1. Projekto aprašymas",
        "sec2": "2. Pagrindimas ir poreikis",
        "sec3": "3. Reikalingi leidimai ir institucijos",
        "sec4": "4. Teisinis pagrindas",
        "sec5": "5. Priedų sąrašas",
        "sec6": "6. Tolesni žingsniai",
        "sec_standards": "Taikomi standartai (ES / tarptautiniai)",
        "th_std_code": "Standartas",
        "th_std_scope": "Taikymo sritis",
        "th_std_supervisor": "Priežiūros institucija",
        "liite_standards": "Atitikties standartams deklaracija",
        "liitteet_note": ("Kartu su paraiška privaloma pateikti šiuos priedus. "
                          "Pažymėkite langelį, kai priedas paruoštas."),
        "lahteet_h": "Šaltiniai ir nuorodos",
        "lahteet_laki_h": "Teisinis pagrindas",
        "lahteet_rag_h": "Institucijų šaltiniai",
        "lahteet_b": "Rengiant šį projektą buvo naudojami šie oficialūs dokumentai:",
        "yhteystiedot_h": "Pareiškėjo kontaktiniai duomenys",
        "yht_hakija": "Pareiškėjas",
        "yht_ytunnus": "Įmonės kodas",
        "yht_osoite": "Adresas",
        "yht_lisatietoja": "Papildoma informacija",
        "footer":          ("NCE Permit AI (Native Clean Energy)  ·  ncenergy.fi  ·  "
                            "info@ncenergy.fi"
                            "·  DI sugeneruotas projektas — reikalinga ekspertų peržiūra"),
        "th_lupa":  "Leidimas / Pranešimas", "th_viran": "Institucija", "th_laki": "Teisinis pagrindas",
        "th_nro":   "Nr.",  "th_liite": "Priedas",  "th_tila": "Būsena",
        "liite_toimitettu": "[ ] Pateikta",
        "toim_nro": "Nr.", "toim_toimenpide": "Veiksmas",
        "toim_vastuutaho": "Atsakingas", "toim_aikataulu": "Terminas",
        "hdr_draft": "leidimo paraiškos projektas", "hdr_right": "ncenergy.fi  |  DI projektas",
        "ftr_ai":    "DI projektas — reikalinga peržiūra", "ftr_sivu": "Puslapis",
        "esiselvitys_p":   ("Projektas yra išankstinio tyrimo etape. Galutinės techninės "
                            "specifikacijos, sklypo planai ir poveikio aplinkai vertinimai bus "
                            "patikslinti tolesnio planavimo metu."),
        "lupavaihe_p":     ("Projektas yra leidimo etape. Statybos leidimo paraiška ir priedai "
                            "rengiami pateikti institucijai."),
        "rakentaminen_p":  ("Projektas yra statybos etape. Įgyvendinimas vykdomas pagal "
                            "patvirtintus leidimus ir planus, prižiūrint atsakingoms institucijoms."),
        "bess_pintaala":   "Numatomas sklypo plotas yra 0,4–0,6 ha.",
        "mks_viittaus":    ("Projekto teritorijos žemės naudojimas ištirtas NCE žemės naudojimo "
                            "ataskaitoje (žr. 2 priedą: Žemės naudojimo ataskaitos PDF). Ataskaitoje "
                            "pateikiama informacija apie nekilnojamąjį turtą, teritorijų planavimo "
                            "statusą, saugomas teritorijas ir požeminio vandens telkinių duomenis, "
                            "atitinkančius taikomų nacionalinių teritorijų planavimo teisės aktų "
                            "reikalavimus sklypo tyrimams."),
        "kaava_BESS":      ("<b>Teritorijų planavimo statusas (svarbiausias išankstinio tyrimo "
                            "aspektas):</b> Pirmiausia būtina nustatyti BESS projekto sklypo "
                            "teritorijų planavimo statusą. Daugumoje savivaldybių baterijų energijos "
                            "kaupimo sistemos įrengimui reikalingas detalusis planas arba planavimo "
                            "leidimas. Teritorijų planavimo statusas labiausiai lemia bendrą leidimo "
                            "proceso trukmę — pirmasis žingsnis yra išankstinė konsultacija su "
                            "statybos priežiūros institucija."),
        "kaava_tuuli":     ("<b>Teritorijų planavimo statusas ir PAV reikalavimas:</b> Vėjo "
                            "elektrinių projektui beveik visada reikalingas vietinis bendrasis arba "
                            "detalusis planas (Žemės naudojimo ir statybos įstatymas 132/1999, "
                            "§ 77a). PAV procedūra (PAV įstatymas 252/2017) yra privaloma ≥10 MW "
                            "arba ≥5 vėjo jėgainių projektams — teritorijų planavimo ir PAV procesai "
                            "dažnai vyksta lygiagrečiai ir kartu trunka 3–6 metus. Teritorijų "
                            "planavimo statusas išsprendžiamas pirmiausia, prieš kitus leidimus."),
        "kaava_SMR":       ("<b>STUK išankstinis licencijavimas (svarbiausias pirmasis žingsnis):</b> "
                            "Branduolinio objekto projektui Valstybės tarybos principinis sprendimas "
                            "(Branduolinės energijos įstatymas 990/1987, § 11) ir STUK išankstinio "
                            "licencijavimo procedūra yra privalomi prieš visus kitus leidimus. "
                            "Procesą inicijuoja STUK YVL gairių saugos ataskaita. Teritorijų "
                            "planavimas vykdomas lygiagrečiai, tačiau branduolinės saugos procedūra "
                            "yra dominuojanti."),
        "kaava_aurinkovoima": ("<b>Statybos leidimas ir teritorijų planavimas:</b> Nedideliam saulės "
                            "parkui (mažesniam nei apie 1 ha) gali būti taikoma supaprastinta "
                            "procedūra pagal Statybos įstatymą 751/2023 (§ 49). PAV nereikalaujama "
                            "projektams, mažesniems nei 50 ha. Vis tiek būtina patikrinti teritorijų "
                            "planavimo statusą — už detaliojo plano ribų gali prireikti planavimo "
                            "leidimo."),
        "kaava_generic":   ("<b>Teritorijų planavimo statusas:</b> Prieš teikiant paraišką būtina "
                            "patikrinti dabartinį projekto sklypo teritorijų planavimo statusą "
                            "išankstinėje konsultacijoje su statybos priežiūros institucija. "
                            "Teritorijų planavimo statusas tiesiogiai veikia leidimo proceso trukmę "
                            "ir reikalavimus — statybai dažnai reikalingas detalusis planas, jo "
                            "pakeitimas arba planavimo leidimas."),
        "nce_info_desc":   ("NCE Permit AI yra dirbtinio intelekto įrankis, automatizuojantis "
                            "energetikos projektų leidimų procesus."),
        "raqs_heading":    "RAQS vertinimas",
        "raqs_subtitle":   ("Dirbtinio intelekto savęs vertinimas (Regulatory Assurance & Quality "
                            "System) — nepakeičia ekspertų peržiūros. Kiekvienas kriterijus "
                            "vertinamas 1–5 balais; didesnis balas reiškia geresnę kokybę."),
        "raqs_overall":    "Bendras įvertinimas",
        "raqs_disclaimer": ("RAQS vertinimas sugeneruotas automatiškai naudojant dirbtinį intelektą. "
                            "Prieš teikdami projektą leidimus išduodančiai institucijai, visada jį "
                            "peržiūrėkite."),
        "raqs_lbl_viittaukset":     "Teisinės nuorodos",
        "raqs_lbl_lupakattavuus":  "Leidimų aprėptis",
        "raqs_lbl_epävarmuus":     "Neapibrėžtumo valdymas",
        "raqs_lbl_kattavuus":      "Turinio išsamumas",
        "raqs_lbl_valmisteluaste": "Paraiškos parengtumas",
        "taydennys_heading":  "Papildymo kontrolinis sąrašas",
        "taydennys_subtitle": ("Šiuos punktus būtina papildyti prieš teikiant paraišką institucijai. "
                               "Šiuos punktus taip pat rasite tekste kaip oranžines [BŪTINA] žymas."),
        "taydennys_osio":     "Skyrius",
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
    # FI country, output_language=EN — keep entity names, translate parenthetical descriptors
    "FI_EN": {
        "BESS":           "Tukes (chemicals/electrical), Pelastuslaitos (fire safety)",
        "tuulivoima_maa": "ELY-keskus / Luova (EIA), Tukes (electrical safety)",
        "tuulivoima_meri":"ELY-keskus / Luova (EIA), Traficom, Tukes",
        "aurinkovoima":   "Tukes (electrical safety), Kunta (building control)",
        "SMR":            "STUK (nuclear safety, YVL guidelines), TEM (government resolution)",
        "smr_bess":       "STUK (nuclear safety), Tukes (BESS component)",
        "vesivoima":      "AVI (water permit), ELY-keskus, Tukes",
        "_generic":       "Tukes (safety), Kunta (building control)",
    },
    # FI country, output_language=SE
    "FI_SE": {
        "BESS":           "Tukes (kemikalier/el), Pelastuslaitos (brandsäkerhet)",
        "tuulivoima_maa": "ELY-keskus / Luova (MKB), Tukes (elsäkerhet)",
        "tuulivoima_meri":"ELY-keskus / Luova (MKB), Traficom, Tukes",
        "aurinkovoima":   "Tukes (elsäkerhet), Kunta (byggnadskontroll)",
        "SMR":            "STUK (kärnsäkerhet, YVL-anvisningar), TEM (principbeslut)",
        "smr_bess":       "STUK (kärnsäkerhet), Tukes (BESS-komponent)",
        "vesivoima":      "AVI (vattentillstånd), ELY-keskus, Tukes",
        "_generic":       "Tukes (säkerhet), Kunta (byggnadskontroll)",
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
    "DE": {
        "BESS":           "Immissionsschutzbehörde (Länder), BNetzA (Netz), Gewerbeaufsichtsamt",
        "tuulivoima_maa": "Immissionsschutzbehörde (Länder), Luftfahrtbundesamt/Bundeswehr (Hindernisbeurteilung), BNetzA",
        "tuulivoima_meri":"BSH (Bundesamt für Seeschifffahrt und Hydrographie), BNetzA, Wasser- und Schifffahrtsamt",
        "aurinkovoima":   "Untere Baubehörde (Landkreis/Stadt), BNetzA (Netz), Gewerbeaufsichtsamt",
        "SMR":            "BMUV / Länderaufsichtsbehörde (Atomgesetz AtG), BMWi, Bundestag (Genehmigung)",
        "smr_bess":       "BMUV / Länderaufsichtsbehörde (AtG), BNetzA, Immissionsschutzbehörde",
        "vesivoima":      "Wasserbehörde (Land), Umweltbundesamt (UBA), BNetzA",
        "_generic":       "Untere Baubehörde (Landkreis/Stadt), Immissionsschutzbehörde (Länder)",
    },
}


# KNOWN, LIVE INACCURACY (2026-08-23, LT language-gap investigation — see
# LT_LANGUAGE_GAP_INVESTIGATION.md): DE and LT have no entry here, so
# generate_pdf() (~line 8380: _BESS_MARKET_DATA.get(_country,
# _BESS_MARKET_DATA["FI"])) falls back to FINLAND'S market index and cites it
# as a real source in that country's own BESS report's "Lähteet" list — e.g.
# a Lithuanian BESS report currently cites "Clean Horizon Storage Index — 110
# €k/MW/year" as if it reflected the Lithuanian market. Labeled a "hardcoded
# estimate," so not false precision, but nothing marks it as Finland-specific.
# Needs real DE/LT market data (or an honest "not yet researched" placeholder,
# same pattern as the [TÄYDENNETTÄVÄ – ...] markers used elsewhere), not a
# fabricated figure — tracked as real per-country research alongside item
# 5(B)'s law-citation research and item 4's _COUNTRY_LIITTEET buildout. Do
# NOT add a plausible-looking DE/LT number here without that research.
_BESS_MARKET_DATA: dict[str, dict] = {
    "FI": {"index": 110, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "SE": {"index": 145, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "DA": {"index": 160, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "NO": {"index": 130, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "PL": {"index": 775, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index", "date": "Q1/2026"},
    "EE": {"index": 120, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index (estimate)", "date": "Q1/2026"},
    "LV": {"index": 115, "unit": "€k/MW/year", "source": "Clean Horizon Storage Index (estimate)", "date": "Q1/2026"},
}


# ─── Finnish statute name → English translation (applied to body text when lang=EN) ───
# Replaces Finnish law names that leak from the prompt into EN body text.
# Keys are word-boundary-anchored Finnish statute names; values are English equivalents.
_FI_STATUTE_NAME_EN: list[tuple[str, str]] = [
    (r"Rakentamislaki\b",                              "Building Act"),
    (r"Maankäyttö- ja rakennuslaki\b",                 "Land Use and Building Act"),
    (r"\bMRL\b(?!\s*\d)",                              "LUBA"),   # bare MRL abbreviation
    (r"Ympäristönsuojelulaki\b",                       "Environmental Protection Act"),
    (r"\bYSL\b(?!\s*\d)",                              "EPA"),
    (r"Pelastuslaki\b",                                "Rescue Act"),
    (r"Sähkömarkkinalaki\b",                           "Electricity Market Act"),
    (r"Kemikaaliturvallisuuslaki\b",                   "Chemical Safety Act"),
    (r"Luonnonsuojelulaki\b",                          "Nature Conservation Act"),
    (r"Vesilaki\b",                                    "Water Act"),
    (r"Säteilylaki\b",                                 "Radiation Act"),
    (r"Ydinenergiaalaki\b",                            "Nuclear Energy Act"),
    (r"Laki\s+ympäristövaikutusten\s+arviointimenettelystä\b", "EIA Act"),
    (r"\bYVA-laki\b",                                  "EIA Act"),
    (r"paloturvallisuusselvitys\b",                     "fire safety report"),
    (r"\bpaloturvallisuus\b",                           "fire safety"),
    (r"\brakennusvalvonta\b",                           "building control"),
    (r"\bLakiviittaukset\b",                            "Legal references"),
]
_FI_STATUTE_RE_EN: list[tuple] = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _FI_STATUTE_NAME_EN
]


def _translate_fi_statute_names(text: str) -> str:
    """Replace Finnish statute names with English equivalents in body text."""
    for rx, repl in _FI_STATUTE_RE_EN:
        text = rx.sub(repl, text)
    return text


def _dedup_statute_bullets(text: str) -> str:
    """Remove less-specific statute bullets when a more specific version of the same statute exists.
    E.g. '• Building Act 751/2023' is dropped when '• Building Act 751/2023, § 44' also appears."""
    statute_num_re = re.compile(r"\b(\d{2,4}/\d{4})\b")
    bullet_re = re.compile(r"^[-•*]\s+|^\d+[.)]\s+")
    lines = text.splitlines()
    # Map statute number → list of (line_idx, line_text) for bullet lines only
    num_to_idxs: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        if bullet_re.match(line.strip()):
            for num in statute_num_re.findall(line):
                num_to_idxs.setdefault(num, []).append(i)
    # For each statute with 2+ bullets: drop the shorter (less specific) ones
    remove_idxs: set[int] = set()
    for idxs in num_to_idxs.values():
        if len(idxs) > 1:
            by_len = sorted(idxs, key=lambda i: len(lines[i]), reverse=True)
            remove_idxs.update(by_len[1:])
    return "\n".join(line for i, line in enumerate(lines) if i not in remove_idxs)


def _apply_en_statute_fixes(sections: dict) -> dict:
    """Translate Finnish statute names and deduplicate bullets in body text for EN output."""
    result = {}
    for k, v in sections.items():
        if isinstance(v, str) and v.strip():
            v = _translate_fi_statute_names(v)
            v = _dedup_statute_bullets(v)
        result[k] = v
    return result


# ─── Language-consistency guardrail ───────────────────────────────────────────
# Markers that are distinctly Finnish compound words — should never appear in a
# non-FI language output. These are NOT proper nouns (authority names like "Tukes"
# are explicitly excluded). Update this list whenever a new leak is found.
_FI_LEAK_MARKERS: list[str] = [
    # RAQS page labels (caught in session 2026-07-03, now fixed via _PDF_STRINGS)
    "Lakiviittaukset", "Lupakattavuus", "Epävarmuudenhallinta",
    "Hakemusvalmisteluaste", "Kokonaispistemäärä", "RAQS-arviointi",
    # Standards table authority descriptors (caught 2026-07-03)
    "kemikaalit", "paloturvallisuus", "sähköturvallisuus",
    "paloturvallisuusselvitys",
    # Finnish statute names that should not appear in EN/SE output
    "Rakentamislaki", "Ympäristönsuojelulaki", "Pelastuslaki",
    "Sähkömarkkinalaki", "Kemikaaliturvallisuuslaki", "Luonnonsuojelulaki",
    # Common Finnish body-text words that shouldn't appear in non-FI output
    "rakennusvalvonta", "perustelu", "yhteenveto",
]
# Compile once for efficiency
_FI_LEAK_RE: re.Pattern = re.compile(
    "|".join(r"\b" + re.escape(m) + r"\b" for m in _FI_LEAK_MARKERS),
    re.IGNORECASE,
)


def check_fi_language_leak(text: str) -> list[str]:
    """Return list of Finnish marker matches found in text. Empty = clean.

    Call after generating a non-FI report to detect language-consistency regressions.
    Accepts either raw section text or PDF-extracted text.
    """
    found = []
    for m in _FI_LEAK_RE.finditer(text):
        found.append(f"'{m.group()}' at char {m.start()}")
    return found


def _sections_language_check(sections: dict, lang: str) -> list[str]:
    """Run Finnish-leak check over all text sections. Returns match list or []."""
    if lang == "FI":
        return []
    combined = " ".join(v for v in sections.values() if isinstance(v, str))
    return check_fi_language_leak(combined)


def _s(lang: str, key: str, country: str = "FI") -> str:
    """Hae käännetty merkkijono PDF-layoutille. EE/ET → EN fallback; muut → FI.

    2026-08-23 (architectural fix, folded into the SE research follow-up):
    the Traficom/STUK/ELY-keskus/law-citation deterministic backstops were
    previously only wired into _t_liite() (liite/attachment names) -- the
    _PDF_STRINGS "kaava_*" cards this function renders bypassed all of them
    entirely, which is exactly how "STUK" ended up named as the reviewing
    authority in a Swedish nuclear project's kaava_SMR card (found during SE
    research, PR #106) even though a STUK backstop already existed. Applying
    the same four backstops here means every future per-country research
    pass automatically protects this call site too, instead of needing the
    same architectural gap rediscovered and re-patched each time. `country`
    defaults to "FI", under which all four backstops are no-ops (verified) --
    so every other _s() call site is unaffected unless it opts in by passing
    country explicitly, same safe-by-default shape as _t_liite()."""
    _lang = "EN" if lang in ("EE", "ET", "LV") else lang
    d = _PDF_STRINGS.get(_lang) or _PDF_STRINGS["FI"]
    text = d.get(key) or _PDF_STRINGS["FI"].get(key, key)
    text = _fix_hardcoded_traficom(text, country)
    text = _fix_hardcoded_stuk(text, country)
    text = _fix_hardcoded_ely(text, country)
    text = _fix_hardcoded_law_citations(text, country)
    return text


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
            f"\nTAUSTATIETO (suuruusluokka-arvio, ei vahvistettua tuoretta dataa): "
            f"BESS-reservimarkkinat {country}:ssa ovat suuruusluokaltaan "
            f"satoja tuhansia euroja per MW per vuosi 2h-varastolle. "
            f"ÄLÄ mainitse tarkkoja lukemia (€/MW/vuosi, indeksipisteitä) eikä "
            f"ulkoisia raporttinimiä (esim. Clean Horizon Storage Index) tekstissä — "
            f"nämä tiedot eivät ole RAG-lähteistä vahvistettuja. "
            f"Jos markkina-arvo on olennainen, käytä [TÄYDENNETTÄVÄ – ajankohtainen reservimarkkinahinta-arvio]."
        )

    critical_block = ""
    if inp.hanketyyppi in _CRITICAL_HANKE_TYPES:
        _crit_tmpl = _CRITICAL_EXTRA.get(lang, _CRITICAL_EXTRA["FI"])
        critical_block = "\n\n" + _crit_tmpl.format(hanketyyppi=inp.hanketyyppi)

    context_extra_block = ""
    if cfg.get("context_extra"):
        context_extra_block = "\n\n" + cfg["context_extra"]

    _vaihe_key = (inp.hankkeen_vaihe or "esiselvitys").lower()
    # A phase this SPECIFIC hanketyyppi has declared support for (via
    # _PHASE_RAG_QUERIES[hanketyyppi]) but has no writing guidance
    # published yet -- stop cleanly here rather than falling through to
    # the esiselvitys-tone fallback below, which would silently produce a
    # pre-study-toned report for what should be an operating-license or
    # decommissioning document. See PhaseContentNotAvailableError's
    # docstring. Hanketyyppi-scoped, not just vaihe-name-scoped: a vaihe
    # name not declared for THIS hanketyyppi (e.g. "kayttolupa" requested
    # for BESS, which was never declared) falls through to the existing
    # esiselvitys fallback completely unchanged, same as before this PR --
    # caught by this PR's own self-test before shipping, see commit message.
    if _vaihe_key in _PHASE_RAG_QUERIES.get(inp.hanketyyppi, {}) and _vaihe_key not in _PHASE_INSTRUCTIONS:
        raise PhaseContentNotAvailableError(inp.hanketyyppi, _vaihe_key)
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

    # Block 1: stable context — same for identical hanketyyppi/country/phase within 5 min
    # RAG context here is large (~25k tokens) and cached across concurrent requests
    prompt_context = (
        f"{lang_prefix}{country_prefix}"
        f"{standards_block}{bess_market_block}{critical_block}{context_extra_block}{phase_block}"
        f"\n\n{ph['rag_intro']}\n{rag_context}{_prec_block}"
        f"\n\n{write_instr}"
    )

    # Known fields guard — prevents [TÄYDENNETTÄVÄ] for already-provided input values
    _known_parts = [
        f"kiinteistötunnus={_clean_kt(inp.kiinteistotunnus)}",
        f"sijaintikunta={inp.kunta}",
        f"hakija={inp.hakija}",
        f"teho={inp.teho_mw} MW",
    ]
    if inp.kapasiteetti_mwh:
        _known_parts.append(f"kapasiteetti={inp.kapasiteetti_mwh} MWh")
    if inp.sijainti_ymparistovaikutukset:
        _known_parts.append("sijainti- ja ympäristötiedot (annettu yllä)")
    _known_block = (
        "\n\nTUNNETUT SYÖTETIEDOT — ÄLÄ merkitse näitä [TÄYDENNETTÄVÄ]-tagilla:\n"
        + ", ".join(_known_parts)
        + "\n"
    )

    # Missing fields — Claude MUST insert [TÄYDENNETTÄVÄ] wherever these would be needed
    _missing_parts: list[str] = []
    if not (inp.y_tunnus or "").strip():
        _missing_parts.append("Y-tunnus (rekisteröintinumero)")
    if not (inp.osoite or "").strip():
        _missing_parts.append("hankkeen tarkka osoite tai koordinaatit")
    if not inp.sijainti_ymparistovaikutukset:
        _missing_parts.append("sijainti- ja ympäristövaikutustiedot")
    if inp.hanketyyppi == "BESS" and not inp.kapasiteetti_mwh:
        _missing_parts.append("varastointikapasiteetti (MWh)")
    if not inp.kohdeviranomainen:
        _missing_parts.append("kohdeviranomainen / lupaa myöntävä viranomainen")
    _missing_block = ""
    if _missing_parts:
        _missing_block = (
            "\n\nPUUTTUVAT HAKIJAN SYÖTETIEDOT — PAKOLLINEN TÄYDENNYS:\n"
            "Hakija EI ole toimittanut seuraavia tietoja. "
            "Kirjoita [TÄYDENNETTÄVÄ – <tiedon nimi>] JOKAISEEN kohtaan tekstissä, "
            "jossa kyseinen tieto olisi välttämätön. ÄLÄ täytä näitä arvauksilla:\n"
            + "".join(f"  • {f}\n" for f in _missing_parts)
        )

    # Block 2: project-specific — varies per user request, not cached
    prompt_task = (
        f"{ph['intro']}\n\n"
        f"Hanketyyppi: {inp.hanketyyppi} ({cfg['nimi_fi']})\n"
        f"Kiinteistötunnus: {_clean_kt(inp.kiinteistotunnus)}\n"
        f"Teho: {inp.teho_mw} MW{kap_lisatieto}\n"
        f"Kunta: {inp.kunta}\n"
        f"Hakija: {inp.hakija}"
        f"{sijainti_lisatieto}{vaihe_lisatieto}{viranomainen_lisatieto}{ifc_block}\n"
        f"Päivämäärä: {now}{_known_block}{_missing_block}{viranomainen_ohje}\n\n"
        f"## {ph['kuvaus']}\n{kuvaus_inst}\n\n"
        f"## {ph['perustelut']}\n{perustelut_inst}\n\n"
        f"## {ph['luvat']}\n{luvat_inst}\n\n"
        f"## {ph['toimenpiteet']}\n{toim_inst}"
    )

    # Streaming, not a blocking create(): this prompt is ~30K+ input tokens with
    # max_tokens=8000 output, and a non-streaming call can legitimately run past
    # a per-call timeout while still making progress (confirmed via instrumented
    # reproduction 2026-07-25 — httpx measures time between chunks on a stream,
    # not total call duration). timeout=600.0 is generous because it now only
    # matters as a stall detector. The SDK's default retry (max_retries=2, not
    # overridden here) is kept deliberately: with streaming, a retry now only
    # triggers on a genuine multi-second stall, which is exactly the case
    # retries should handle — before this change, retries were instead
    # multiplying a single slow-but-fine ~120-200s call into a cascading
    # 2-3x wait that occasionally exhausted all attempts as a hard failure.
    # Cost & resource guardrail (TASO 1): fail cleanly before spending on a Claude
    # call this generation has no budget left for. See GenerationCapError.
    _generation_id = getattr(inp, "generation_id", "") or ""
    if _generation_id:
        _n_calls = _retrieval_trace.count_claude_calls(_generation_id)
        if _n_calls >= _CLAUDE_CALL_CAP:
            _retrieval_trace.log_guardrail_hit(
                generation_id=_generation_id,
                guard_type="claude_call_cap",
                count_at_trip=_n_calls,
                cap=_CLAUDE_CALL_CAP,
                detail="call_type=draft",
            )
            raise GenerationCapError("claude_call", _generation_id, _n_calls, _CLAUDE_CALL_CAP)

    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=600.0)
    _LAST_TIMING["t3a_claude_call_start"] = _time.monotonic()
    _LAST_TIMING["prompt_context_chars"] = len(prompt_context)
    _LAST_TIMING["prompt_task_chars"] = len(prompt_task)
    try:
        with claude.messages.stream(
            model=_MODEL_ID,
            max_tokens=8000,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt_context, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt_task},
            ]}],
        ) as stream:
            raw_parts = [text for text in stream.text_stream]
            resp = stream.get_final_message()
    except anthropic.RateLimitError:
        raise RuntimeError(
            "API rate limit (429) — pyyntöjä liikaa, yritä hetken kuluttua"
        )
    except anthropic.APIStatusError as _ae:
        if _ae.status_code == 402:
            raise RuntimeError(
                "Anthropic-krediitit lopussa — lisää krediittejä Consolessa "
                "(info@ncenergy.fi)"
            )
        if _ae.status_code in (529, 503):
            raise RuntimeError(
                f"Anthropic API ylikuormittunut ({_ae.status_code}) — yritä uudelleen"
            )
        raise RuntimeError(f"Anthropic API virhe ({_ae.status_code}): {_ae.message}")
    except anthropic.APITimeoutError:
        _LAST_TIMING["t3b_claude_call_TIMED_OUT"] = _time.monotonic()
        raise RuntimeError(
            "Claude API -aikakatkaisu (>600s) — palvelin ei vastannut, yritä uudelleen"
        )
    except anthropic.APIConnectionError as _ae:
        raise RuntimeError(f"Anthropic yhteysvirhe: {_ae}")
    _LAST_TIMING["t3c_claude_call_succeeded"] = _time.monotonic()
    raw = unicodedata.normalize("NFC", "".join(raw_parts))
    _u = resp.usage
    logger.warning(
        "[cache] creation=%s read=%s input=%s output=%s stop=%s raw_len=%d",
        getattr(_u, "cache_creation_input_tokens", 0),
        getattr(_u, "cache_read_input_tokens", 0),
        _u.input_tokens, _u.output_tokens, resp.stop_reason, len(raw),
    )
    if _generation_id:
        _retrieval_trace.log_api_call(
            generation_id=_generation_id,
            call_type="draft",
            model=_MODEL_ID,
            input_tokens=_u.input_tokens,
            cache_creation_input_tokens=getattr(_u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(_u, "cache_read_input_tokens", 0) or 0,
            output_tokens=_u.output_tokens,
        )
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

    # Strip [TÄYDENNETTÄVÄ] tags for fields already provided in inp
    result = _strip_known_gap_tags(result, inp)

    return result


def _strip_known_gap_tags(sections: dict, inp: "ApplicationInput") -> dict:
    """Remove spurious [TÄYDENNETTÄVÄ] tags for values already given in inp."""
    # Build a set of lower-case keywords from the description that indicate
    # the field is actually already known from the input.
    known_kw: set[str] = set()
    if inp.kiinteistotunnus:
        # Use precise compound terms — do NOT add bare "tunnus" as it matches Y-tunnus
        known_kw |= {"kiinteistötunnus", "kiinteistotunnus"}
    if inp.kunta:
        known_kw |= {"sijaintikunta"}  # "kunta" alone is too broad (e.g. "kuntakohtainen")
    if inp.hakija:
        known_kw |= {"hakija", "hakijan nimi", "hakijan"}
    if inp.teho_mw:
        known_kw |= {"teho mw", "megawatti"}  # avoid bare "teho" matching "tehostaa" etc.
    if inp.kapasiteetti_mwh:
        known_kw |= {"kapasiteetti mwh", "megawattitunti"}

    def _filter(m: re.Match) -> str:
        desc = m.group(1).lower()
        if any(kw in desc for kw in known_kw):
            return ""
        return m.group(0)

    def _clean(v: str) -> str:
        v = re.sub(r'\[TÄYDENNETTÄVÄ\s*–\s*([^\]]+)\]', _filter, v)
        v = re.sub(r' {2,}', ' ', v)   # collapse double spaces left by removed tags
        v = re.sub(r'\n {2,}', '\n', v)
        return v

    return {k: (_clean(v) if isinstance(v, str) else v)
            for k, v in sections.items()}


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
            Paragraph(_t_liite(lang, liite, country), ParagraphStyle("tl", fontSize=8.5, leading=12)),
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
    # Prefer lang-qualified key (e.g. "FI_EN") so descriptor words are in the output language
    _sup_key = f"{country}_{lang}" if lang != "FI" else country
    sup_country = _NATIONAL_SUPERVISORS.get(_sup_key) or _NATIONAL_SUPERVISORS.get(country, _NATIONAL_SUPERVISORS["FI"])
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
    text = _fix_unicode_subscripts(text)
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
            clean = re.sub(
                r'\[TÄYDENNETTÄVÄ\s*–\s*([^\]]+)\]',
                r'<font color="#e65100"><b>[TÄYDENNETTÄVÄ – \1]</b></font>',
                clean,
            )
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


class _PDFDocTemplate(SimpleDocTemplate):
    """SimpleDocTemplate subclass that feeds h2 headings into a TableOfContents widget."""

    def __init__(self, *args, toc: "TableOfContents | None" = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._toc = toc

    def afterFlowable(self, flowable):
        if self._toc is not None and isinstance(flowable, Paragraph):
            if getattr(flowable, "_toc_key", None):
                key  = flowable._toc_key
                text = flowable.getPlainText()
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (0, text, self.page, key))


def _toc_h2(text: str, style, key: str) -> Paragraph:
    """Return an h2 Paragraph marked for ToC registration."""
    p = Paragraph(text, style)
    p._toc_key = key
    return p


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


# ─────────────────────────────────────────────────────────────────────────────
# Gap analysis — [TÄYDENNETTÄVÄ] extraction and Täydennyslista page
# ─────────────────────────────────────────────────────────────────────────────

_GAP_RE = re.compile(r'\[TÄYDENNETTÄVÄ\s*–\s*([^\]]+)\]')

_GAP_SECTION_LABELS: dict[str, str] = {
    "kuvaus":       "Hankkeen kuvaus",
    "perustelut":   "Perustelut",
    "luvat_teksti": "Tarvittavat luvat ja ilmoitukset",
    "toimenpiteet": "Seuraavat toimenpiteet",
}


def _extract_gaps(sections: dict) -> list[tuple[str, str]]:
    """Return list of (section_label, description) for every gap found.

    Detects two kinds of gaps:
    - Explicit: [TÄYDENNETTÄVÄ – ...] tags in the AI output
    - Heuristic: thin sections (<150 chars) or placeholder text patterns
    """
    _PLACEHOLDER_RE = re.compile(
        r'\bToimenpide\s+[A-Z]\b'       # "Toimenpide A", "Toimenpide B"
        r'|\[lisää\s'                    # [lisää jotain]
        r'|\bXXX\b'
        r'|\bTBD\b(?!\s*\w)',
        re.IGNORECASE,
    )
    _THIN_THRESHOLD = 150  # chars

    gaps: list[tuple[str, str]] = []
    seen_heuristic: set[str] = set()  # avoid duplicate heuristic entries per section

    for key, text in sections.items():
        if not isinstance(text, str):
            continue
        label = _GAP_SECTION_LABELS.get(key, key)

        # Explicit [TÄYDENNETTÄVÄ] tags
        for m in _GAP_RE.finditer(text):
            gaps.append((label, m.group(1).strip()))

        # Heuristic: thin section
        stripped = text.strip()
        if stripped and len(stripped) < _THIN_THRESHOLD and label not in seen_heuristic:
            seen_heuristic.add(label)
            gaps.append((label, "Osio on epätyypillisen lyhyt — lisää asianmukainen sisältö ennen jättämistä"))

        # Heuristic: placeholder text
        ph_m = _PLACEHOLDER_RE.search(stripped)
        if ph_m and label not in seen_heuristic:
            seen_heuristic.add(label)
            gaps.append((label, f"Placeholder-teksti havaittu: '{ph_m.group()}' — korvaa todellisella sisällöllä"))

    return gaps


def _taydennyslista_page(gaps: list[tuple[str, str]], st: dict, lang: str = "FI") -> list:
    """Build Täydennyslista PDF elements (PageBreak + orange box + checklist)."""
    C_FILL = colors.HexColor("#fff3e0")
    C_BD   = colors.HexColor("#e65100")
    C_HDR  = colors.HexColor("#e65100")

    heading_style = ParagraphStyle(
        "tl_hdr", fontSize=13, fontName=PDF_FONT_BOLD,
        textColor=C_HDR, spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "tl_sub", fontSize=8.5, textColor=colors.HexColor("#555555"),
        leading=12, spaceAfter=10,
    )
    item_style = ParagraphStyle(
        "tl_item", fontSize=9, leading=13, leftIndent=6, spaceAfter=4,
    )
    sec_style = ParagraphStyle(
        "tl_sec", fontSize=7.5, textColor=colors.HexColor("#888888"),
        leading=10, leftIndent=6, spaceAfter=8,
    )

    body_elems = [
        Paragraph(_s(lang, "taydennys_heading"), heading_style),
        Paragraph(_s(lang, "taydennys_subtitle"), sub_style),
    ]
    for i, (sec_label, desc) in enumerate(gaps, 1):
        body_elems.append(
            Paragraph(f"<b>{i}.</b> ☐ {_latin1_safe(desc)}", item_style)
        )
        body_elems.append(
            Paragraph(f"{_s(lang, 'taydennys_osio')}: {_latin1_safe(sec_label)}", sec_style)
        )

    box = Table(
        [[body_elems]],
        colWidths=[16.2 * cm],
        splitByRow=1,
    )
    box.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 2.0, C_BD),
        ("BACKGROUND", (0, 0), (-1, -1), C_FILL),
        ("PADDING",    (0, 0), (-1, -1), 14),
    ]))
    return [PageBreak(), box]


# ─── RAQS: Regulatory Assurance & Quality System — reviewing agent ───────────

# ── RAQS Full Phase 1: single structured criterion source ──────────────────
# Previously the 5 criteria were hand-duplicated across three places (the
# prompt text below, _RAQS_CRITERION_KEY, and _RAQS_ORDER) — changing or
# adding a criterion meant editing all three in sync, with no check that
# they'd stayed consistent. _RAQS_CRITERIA is now the single source of
# truth: order, prompt wording (always Finnish — RAQS is always scored by
# reading Finnish-labeled criteria; only its JSON *output* text is
# translated via _RAQS_LANG_INSTRUCTION below) and PDF label key all live
# in one place, and everything else is derived from it.
_RAQS_CRITERIA: list[dict] = [
    {
        "id": "viittaukset",
        "pdf_label_key": "raqs_lbl_viittaukset",
        "prompt": "Lakiviittausten relevanttius ja tarkkuus (1=puuttuu/virheelliset, 5=täsmälliset ja kattavat)",
        # numbered_pad/json_pad: exact original whitespace (list alignment
        # was hand-typed, not a clean formula) — kept explicit per criterion
        # so the generated prompt stays byte-identical to the pre-refactor
        # text (verified: no behavioral risk from an accidental wording/
        # whitespace drift in a prompt this scoring-sensitive).
        "numbered_pad": 2, "json_pad": 4,
    },
    {
        "id": "lupakattavuus",
        "pdf_label_key": "raqs_lbl_lupakattavuus",
        "prompt": "Tarvittavien lupamenettelyjen kattavuus (1=merkittäviä aukkoja, 5=kaikki luvat käsitelty)",
        "numbered_pad": 1, "json_pad": 2,
    },
    {
        "id": "epävarmuus",
        "pdf_label_key": "raqs_lbl_epävarmuus",
        "prompt": (
            "Epävarmuuden hallinta. PUNAINEN LIPPU: jos tekstissä esiintyy tarkka määrällinen "
            "lukema (€/MW/vuosi, %, €/MWh, indeksipiste tms.) yhdistettynä nimettyyn ulkoiseen "
            "lähteeseen tai raporttiin (esim. \"Clean Horizon Storage Index\", \"BNEF\", "
            "\"IEA Storage Tracker\") — mutta kyseinen lähde EI löydy raportin lähdeluettelosta "
            "— laske 2 pistettä ja mainitse tämä perustelussa nimenomaisesti. "
            "(1=vahvistamattomia tarkkoja ulkoisia lukuja esitetty faktana ilman lähdettä, "
            "5=⚠️-merkinnät asiallisesti käytetty eikä vahvistamattomia tarkkoja markkinaviittauksia)"
        ),
        "numbered_pad": 3, "json_pad": 5,
    },
    {
        "id": "kattavuus",
        "pdf_label_key": "raqs_lbl_kattavuus",
        "prompt": "Sisällön syvyys ja pituus (1=pintapuolinen, 5=perusteellinen)",
        "numbered_pad": 4, "json_pad": 6,
    },
    {
        "id": "valmisteluaste",
        "pdf_label_key": "raqs_lbl_valmisteluaste",
        "prompt": "Hakemusvalmisteluaste (1=raakaaineisto, 5=lähes jätettävissä sellaisenaan)",
        "numbered_pad": 1, "json_pad": 1,
    },
]

# Derived, not hand-duplicated — see _RAQS_CRITERIA above.
_RAQS_ORDER: list[str] = [c["id"] for c in _RAQS_CRITERIA]
_RAQS_CRITERION_KEY: dict[str, str] = {c["id"]: c["pdf_label_key"] for c in _RAQS_CRITERIA}


def _raqs_build_system_base() -> str:
    """Generate _RAQS_SYSTEM_BASE's text from _RAQS_CRITERIA — the numbered
    criteria list and the JSON schema example are both built from the same
    list, so adding/reordering a criterion here can't desync them."""
    numbered = "\n".join(
        f"{i}. {c['id']}{' ' * c['numbered_pad']}— {c['prompt']}"
        for i, c in enumerate(_RAQS_CRITERIA, 1)
    )
    json_fields = "\n".join(
        f'  "{c["id"]}":{" " * c["json_pad"]}{{"pisteet": <1-5>, "perustelu": "<max 15 sanaa>"}},'
        for c in _RAQS_CRITERIA
    )
    return (
        "Olet sääntelylaadunvarmistusagentti (RAQS). Arvioi annettu energialupaluonnos "
        f"viidellä kriteerillä.\nAnna kokonaislukupisteytys asteikolla 1–5 jokaiselle "
        "kriteerille ja yksi lyhyt perustelu (max 15 sanaa).\n\n"
        f"Kriteerit:\n{numbered}\n\n"
        "Vastaa VAIN validilla JSON-objektilla, ei muuta tekstiä:\n{\n"
        f"{json_fields}\n"
        '  "yhteenveto":     "<max 25 sanaa>"\n}\n'
    )


_RAQS_SYSTEM_BASE = _raqs_build_system_base()

# RAQS's JSON *output* text (perustelu/yhteenveto) is translated per output
# language — the criteria themselves are always read in Finnish (see above).
# Previously only EN/SE had this; DA/NO/PL/ET/DE/LV/LT fell through to no
# instruction at all, meaning RAQS's own justification/summary text stayed
# in whatever language the model defaulted to (usually Finnish) even when
# every other part of the report was correctly translated — same class of
# gap as P3-5's UI-string gap, but for an AI-facing prompt instead of a
# static string table.
_RAQS_LANG_INSTRUCTION = {
    "EN": (
        "CRITICAL OUTPUT RULE: All text values in the JSON response MUST be in English. "
        "This means every 'perustelu' justification and the 'yhteenveto' summary must be "
        "written in English — regardless of the language of the content you are reviewing. "
        "JSON keys (viittaukset, lupakattavuus, etc.) stay unchanged; only the text values change language."
    ),
    "SE": (
        "KRITISK UTDATAREGEL: Alla textvärden i JSON-svaret MÅSTE vara på svenska. "
        "Det innebär att varje 'perustelu'-motivering och 'yhteenveto'-sammanfattning ska "
        "skrivas på svenska — oavsett vilket språk det granskade innehållet är på. "
        "JSON-nycklar (viittaukset, lupakattavuus osv.) förblir oförändrade; bara textvärdena byter språk."
    ),
    "DA": (
        "KRITISK UDDATAREGEL: Alle tekstværdier i JSON-svaret SKAL være på dansk. "
        "Det betyder, at hver 'perustelu'-begrundelse og 'yhteenveto'-sammenfatning skal "
        "skrives på dansk — uanset hvilket sprog det gennemgåede indhold er på. "
        "JSON-nøgler (viittaukset, lupakattavuus osv.) forbliver uændrede; kun tekstværdierne skifter sprog."
    ),
    "NO": (
        "KRITISK UTDATAREGEL: Alle tekstverdier i JSON-svaret MÅ være på norsk. "
        "Det betyr at hver 'perustelu'-begrunnelse og 'yhteenveto'-sammendrag skal "
        "skrives på norsk — uansett hvilket språk det gjennomgåtte innholdet er på. "
        "JSON-nøkler (viittaukset, lupakattavuus osv.) forblir uendret; bare tekstverdiene bytter språk."
    ),
    "PL": (
        "KRYTYCZNA ZASADA WYJŚCIA: Wszystkie wartości tekstowe w odpowiedzi JSON MUSZĄ być "
        "w języku polskim. Oznacza to, że każde uzasadnienie 'perustelu' i podsumowanie "
        "'yhteenveto' muszą być napisane po polsku — niezależnie od języka ocenianej treści. "
        "Klucze JSON (viittaukset, lupakattavuus itd.) pozostają niezmienione; zmienia się "
        "tylko język wartości tekstowych."
    ),
    "ET": (
        "KRIITILINE VÄLJUNDIREEGEL: Kõik JSON-vastuse tekstiväärtused PEAVAD olema eesti "
        "keeles. See tähendab, et iga 'perustelu' põhjendus ja 'yhteenveto' kokkuvõte tuleb "
        "kirjutada eesti keeles — sõltumata hinnatava sisu keelest. JSON-võtmed (viittaukset, "
        "lupakattavuus jne) jäävad muutumatuks; ainult tekstiväärtuste keel muutub."
    ),
    "DE": (
        "KRITISCHE AUSGABEREGEL: Alle Textwerte in der JSON-Antwort MÜSSEN auf Deutsch sein. "
        "Das bedeutet, dass jede 'perustelu'-Begründung und die 'yhteenveto'-Zusammenfassung "
        "auf Deutsch verfasst werden müssen — unabhängig von der Sprache des geprüften Inhalts. "
        "JSON-Schlüssel (viittaukset, lupakattavuus usw.) bleiben unverändert; nur die "
        "Textwerte wechseln die Sprache."
    ),
    "LV": (
        "KRITISKS IZVADES NOTEIKUMS: Visām teksta vērtībām JSON atbildē IR JĀBŪT latviešu "
        "valodā. Tas nozīmē, ka katram 'perustelu' pamatojumam un 'yhteenveto' kopsavilkumam "
        "jābūt rakstītam latviešu valodā — neatkarīgi no izvērtētā satura valodas. JSON atslēgas "
        "(viittaukset, lupakattavuus utt.) paliek nemainītas; mainās tikai teksta vērtību valoda."
    ),
    "LT": (
        "SVARBI IŠVESTIES TAISYKLĖ: Visos teksto reikšmės JSON atsakyme TURI būti lietuvių "
        "kalba. Tai reiškia, kad kiekvienas 'perustelu' pagrindimas ir 'yhteenveto' santrauka "
        "turi būti parašyti lietuvių kalba — nepriklausomai nuo vertinamo turinio kalbos. "
        "JSON raktai (viittaukset, lupakattavuus ir kt.) lieka nepakitę; keičiasi tik teksto "
        "reikšmių kalba."
    ),
}


def _raqs_system_prompt(lang: str) -> str:
    instr = _RAQS_LANG_INSTRUCTION.get(lang, "")
    if instr:
        # Prepend the language rule so the model sees it before the Finnish criteria text
        return instr + "\n\n" + _RAQS_SYSTEM_BASE
    return _RAQS_SYSTEM_BASE


def _raqs_init_db() -> None:
    import sqlite3
    os.makedirs(os.path.dirname(_RAQS_DB), exist_ok=True)
    with sqlite3.connect(_RAQS_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raqs_reviews (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at       TEXT    NOT NULL,
                kiinteistotunnus TEXT,
                kunta            TEXT,
                hanketyyppi      TEXT,
                teho_mw          REAL,
                scores_json      TEXT    NOT NULL,
                overall          REAL
            )
        """)


def _raqs_store(review: dict, inp: "ApplicationInput") -> None:
    import sqlite3, json
    from datetime import datetime, timezone
    try:
        _raqs_init_db()
        scores = [review[k]["pisteet"] for k in _RAQS_ORDER if k in review]
        overall = round(sum(scores) / len(scores), 2) if scores else None
        with sqlite3.connect(_RAQS_DB) as conn:
            conn.execute(
                "INSERT INTO raqs_reviews "
                "(created_at, kiinteistotunnus, kunta, hanketyyppi, teho_mw, scores_json, overall) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    getattr(inp, "kiinteistotunnus", None),
                    getattr(inp, "kunta", None),
                    getattr(inp, "hanketyyppi", None),
                    getattr(inp, "teho_mw", None),
                    json.dumps(review, ensure_ascii=False),
                    overall,
                ),
            )
    except Exception:
        pass  # never block PDF generation


# ─── Citation gap flagging (TASO 2, supplementary — hybrid design, 2026-08-16) ─
#
# Section 4 ("Lakiviitteet") is populated from a static, hanketyyppi+country
# -keyed config list (cfg["luvat"]/cfg.get("laki_extra", []), or a
# _COUNTRY_LUVAT override) — see generate_pdf(). That list stays the
# GUARANTEED FLOOR: a full manual regulatory audit across all 135 real
# (hanketyyppi, country) combinations with independently-defined law data
# is real, valuable expert work, but too large to front-load before this
# lands (tracked separately — see project_manual_sourcing_backlog.md).
#
# This adds a SUPPLEMENTARY, non-blocking layer instead: extract the law
# references the model actually cited inline, and flag any that don't
# appear (by numeric core) anywhere in the static list, as a signal for
# human review. This NEVER changes what section 4 actually renders —
# pure logging, same "never block generation" discipline as RAQS itself.
#
# Matching key is the bare numeric law identifier (e.g. "751/2023",
# "2014/34/EU") ONLY — not the surrounding law-name text. Confirmed
# during scoping against 4 real generated Finnish reports: name-based
# matching produced real false positives from (a) the same law cited in a
# differently-punctuated form, (b) config entries that embed free-text
# annotations in the value itself (e.g. "YVA-laki 252/2017 (kynnykset
# ylittyessä)"), and (c) Finnish grammatical case inflection — e.g.
# "Säteilylakia"/"Säteilylain" vs. the base form "Säteilylaki" — which
# produces spurious near-duplicate "different" citations. Keying on the
# numeric core alone sidesteps all three; the same 4-report test then
# found 0 false positives.
#
# FI-ONLY FOR NOW — DELIBERATE, TRACKED, NOT SILENT. The regex below
# matches Finnish/EU numbering conventions (NNN/YYYY, YYYY/NN/EU) only.
# Sweden (SFS YYYY:NNN), Denmark/Norway/Germany/Poland/Estonia/Latvia/
# Lithuania each use different citation conventions and are completely
# UNTESTED against this extractor — gated off below rather than run
# unvalidated. This is a known, accepted, not-yet-built limitation (same
# tracking treatment as the DA JS-SPA gap / LV wind coverage backlog
# items), not something left implicit in the code.

_CITATION_NUM_RE = re.compile(r"(\d{1,4}/\d{4}|\d{4}/\d{1,3}/EU)")

# Countries this extractor is validated for. Extending to the other 8
# means designing + testing a citation-number pattern per country's own
# legal-citation convention (see module comment above) — not just adding
# a country code here.
_CITATION_GAP_SUPPORTED_COUNTRIES = {"FI"}


def _extract_cited_law_numbers(sections: dict) -> set[str]:
    """Real numeric law-reference cores (e.g. '751/2023', '2014/34/EU')
    found anywhere in the generated section text. FI/EU numbering
    conventions only — see module comment above."""
    full_text = "\n".join(v for v in sections.values() if isinstance(v, str))
    return set(_CITATION_NUM_RE.findall(full_text))


def _flag_citation_gaps(sections: dict, laki_set: set) -> list[str]:
    """Numeric law references cited in the generated text but not present
    (by numeric core) anywhere in the static section-4 list for this
    hanketyyppi/country. Supplementary signal for human review only —
    never changes what section 4 renders."""
    cited = _extract_cited_law_numbers(sections)
    config_nums: set[str] = set()
    for entry in laki_set:
        config_nums.update(_CITATION_NUM_RE.findall(entry))
    return sorted(cited - config_nums)


def _citation_gap_init_db() -> None:
    import sqlite3
    os.makedirs(os.path.dirname(_CITATION_GAP_DB), exist_ok=True)
    with sqlite3.connect(_CITATION_GAP_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS citation_gap_flags (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at       TEXT    NOT NULL,
                kiinteistotunnus TEXT,
                kunta            TEXT,
                hanketyyppi      TEXT,
                country          TEXT,
                flagged_json     TEXT    NOT NULL
            )
        """)


def _citation_gap_store(flags: list[str], inp: "ApplicationInput") -> None:
    """Only ever called with a non-empty `flags` list — no row means no
    gap was found, so nothing to review. Same never-block-generation
    discipline as _raqs_store."""
    import sqlite3, json
    from datetime import datetime, timezone
    try:
        _citation_gap_init_db()
        with sqlite3.connect(_CITATION_GAP_DB) as conn:
            conn.execute(
                "INSERT INTO citation_gap_flags "
                "(created_at, kiinteistotunnus, kunta, hanketyyppi, country, flagged_json) "
                "VALUES (?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    getattr(inp, "kiinteistotunnus", None),
                    getattr(inp, "kunta", None),
                    getattr(inp, "hanketyyppi", None),
                    getattr(inp, "country", None),
                    json.dumps(flags, ensure_ascii=False),
                ),
            )
    except Exception:
        pass  # never block PDF generation


def _raqs_review(
    sections: dict,
    inp: "ApplicationInput",
    sources: list | None = None,
    laki_set: set | None = None,
) -> dict | None:
    """Call Haiku to score sections on 5 RAQS criteria. Returns parsed dict or None.

    2026-08-15 (TASO 1, real-report investigation): this used to send only
    4 of the generated sections, each hard-capped at 800 characters, and
    NEVER included the source list or lakiviitteet at all — RAQS was
    structurally incapable of seeing what a human reader of the actual PDF
    sees, which produced a long-standing, systemic pattern of false
    "sections cut off"/"source not in reference list" flags (confirmed
    against real RAQS history back to 2026-07-03, not just the report that
    triggered this investigation). Now sends the FULL text of all 4
    sections plus the real source list and lakiviitteet — the exact same
    content the assembled PDF shows, built from `sources`/`laki_set`
    (both already computed in generate_pdf() before this call, passed
    through rather than recomputed). Real measured cost impact is
    negligible (input tokens roughly triple, from ~2200 to an estimated
    ~4000-5000 — still a fraction of a cent per call on Haiku).
    """
    import json as _json

    lang = getattr(inp, "lang", "FI") or "FI"

    # Full section text — no truncation. This alone fixes the
    # "sections cut off mid-sentence" class of false flags.
    summary_parts = []
    for key in ["kuvaus", "perustelut", "luvat_teksti", "toimenpiteet"]:
        v = sections.get(key, "")
        if isinstance(v, str) and v.strip():
            summary_parts.append(f"## {key}\n{v}")
    sections_text = "\n\n".join(summary_parts)

    # Source list — same shape/content as the PDF's own "Lähteet ja
    # tietolähteet" section (src.get("display") or src.get("id")), so RAQS
    # checks citations against exactly what the human reader will see.
    sources_text = ""
    if sources:
        country = getattr(inp, "country", "FI") or "FI"
        lines = []
        for src in sources[:8]:
            src_display = src.get("display") or src.get("id", "–")
            src_country = src.get("country", country)
            lines.append(f"- {src_display} ({src_country})")
        sources_text = "## Lähteet ja tietolähteet\n" + "\n".join(lines)

    # Lakiviitteet — same list the PDF's own section 4 renders (translated
    # via _t_law, same as the PDF, not the raw internal reference strings).
    laki_text = ""
    if laki_set:
        laki_lines = [f"- {_t_law(lang, ref)}" for ref in sorted(laki_set)]
        laki_text = "## Lakiviitteet\n" + "\n".join(laki_lines)

    full_content = "\n\n".join(p for p in [sections_text, sources_text, laki_text] if p)

    # Translate the raw internal hanketyyppi ID (e.g. "tuulivoima_maa") before
    # it goes into the prompt. Left untranslated, this raw ID was leaking into
    # RAQS's own "yhteenveto" output even for non-FI reports, since nothing
    # in _raqs_system_prompt()'s language instruction can translate a string
    # the model is simply asked to echo/reference. FI is left as the raw ID
    # (same as before) since it reads as a normal Finnish hanketyyppi slug
    # there; only non-FI languages need the _HANKE_NIMI_TRANS lookup.
    _raqs_hanketyyppi_raw = getattr(inp, "hanketyyppi", "?") or "?"
    _raqs_hanketyyppi_nimi = (
        _HANKE_NIMI_TRANS.get(_raqs_hanketyyppi_raw, {}).get(lang, _raqs_hanketyyppi_raw)
        if lang != "FI"
        else _raqs_hanketyyppi_raw
    )

    prompt = (
        f"Hanketyyppi: {_raqs_hanketyyppi_nimi} | "
        f"Teho: {getattr(inp, 'teho_mw', '?')} MW | "
        f"Kunta: {getattr(inp, 'kunta', '?')}\n\n"
        f"LUONNOKSEN SISÄLTÖ:\n{full_content}"
    )

    # Cost & resource guardrail (TASO 1): checked outside the try/except below.
    # _raqs_review()'s own `except Exception: return None` exists so an ordinary
    # RAQS hiccup degrades gracefully (PDF still builds, just without the RAQS
    # page) — a cap hit must not be swallowed into that same silent-degradation
    # path, so it must fail the whole generation cleanly. See GenerationCapError.
    _generation_id = getattr(inp, "generation_id", "") or ""
    if _generation_id:
        _n_calls = _retrieval_trace.count_claude_calls(_generation_id)
        if _n_calls >= _CLAUDE_CALL_CAP:
            _retrieval_trace.log_guardrail_hit(
                generation_id=_generation_id,
                guard_type="claude_call_cap",
                count_at_trip=_n_calls,
                cap=_CLAUDE_CALL_CAP,
                detail="call_type=raqs",
            )
            raise GenerationCapError("claude_call", _generation_id, _n_calls, _CLAUDE_CALL_CAP)

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=30.0)
        resp = client.messages.create(
            model=_MODEL_ID_FAST,
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _raqs_system_prompt(lang),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        _ru = resp.usage
        if _generation_id:
            _retrieval_trace.log_api_call(
                generation_id=_generation_id,
                call_type="raqs",
                model=_MODEL_ID_FAST,
                input_tokens=_ru.input_tokens,
                cache_creation_input_tokens=getattr(_ru, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(_ru, "cache_read_input_tokens", 0) or 0,
                output_tokens=_ru.output_tokens,
            )
        raw = resp.content[0].text.strip()
        # strip optional markdown code fences
        raw = re.sub(r"^```[a-z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        if not raw:
            return None
        result = _json.loads(raw)
        _raqs_store(result, inp)
        # Retrieval trace (TASO 1, purely additive) — record the final RAQS outcome
        # against this generation's trace, if a generation_id was set. "flagged" =
        # the criteria RAQS itself scored <=2 (its own low-confidence signal, e.g.
        # the "PUNAINEN LIPPU" uncited-external-figure rule for 'epävarmuus') —
        # RAQS reviews generated text sections, not individual chunks, so there is
        # no chunk-ID-level flag to log; this is the closest honest equivalent.
        try:
            _flagged = [
                {"criterion": k, "pisteet": v.get("pisteet"), "perustelu": v.get("perustelu", "")}
                for k, v in result.items()
                if isinstance(v, dict) and isinstance(v.get("pisteet"), (int, float)) and v["pisteet"] <= 2
            ]
            _retrieval_trace.log_raqs_outcome(
                generation_id=getattr(inp, "generation_id", "") or "",
                review=result,
                raqs_order=_RAQS_ORDER,
                flagged=_flagged,
            )
        except Exception:
            pass
        result["_lang"] = lang  # carry lang through to _raqs_page
        return result
    except Exception:
        return None


def _raqs_page(review: dict, st: dict, lang: str = "FI") -> list:
    """Build RAQS score page PDF elements (PageBreak + navy-bordered box)."""
    C_FILL = colors.HexColor("#f0f4ff")
    C_BD   = colors.HexColor("#1a3a5c")
    C_HDR  = colors.HexColor("#1a3a5c")

    heading_style = ParagraphStyle(
        "raqs_hdr", fontSize=13, fontName=PDF_FONT_BOLD,
        textColor=C_HDR, spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "raqs_sub", fontSize=8, textColor=colors.HexColor("#555555"),
        leading=11, spaceAfter=10,
    )
    row_label_style = ParagraphStyle(
        "raqs_lbl", fontSize=9, fontName=PDF_FONT_BOLD, leading=12,
    )
    row_score_style = ParagraphStyle(
        "raqs_sc", fontSize=11, fontName=PDF_FONT_BOLD,
        textColor=C_HDR, alignment=TA_CENTER, leading=12,
    )
    row_note_style = ParagraphStyle(
        "raqs_nt", fontSize=8, textColor=colors.HexColor("#444444"),
        leading=11, leftIndent=4,
    )
    summary_style = ParagraphStyle(
        "raqs_sum", fontSize=8.5, textColor=colors.HexColor("#333333"),
        leading=12, spaceAfter=0, spaceBefore=8,
    )
    disc_style = ParagraphStyle(
        "raqs_disc", fontSize=7, textColor=colors.HexColor("#888888"),
        leading=10, spaceBefore=6,
    )

    def _dots(score: int) -> str:
        score = max(1, min(5, score))
        return "●" * score + "○" * (5 - score)

    scores = [review[k]["pisteet"] for k in _RAQS_ORDER if k in review]
    overall = round(sum(scores) / len(scores), 1) if scores else None

    body_elems: list = [
        Paragraph(_s(lang, "raqs_heading"), heading_style),
        Paragraph(_s(lang, "raqs_subtitle"), sub_style),
    ]

    rows = []
    for key in _RAQS_ORDER:
        if key not in review:
            continue
        entry = review[key]
        sc = int(entry.get("pisteet", 0))
        perustelu_raw = entry.get("perustelu", "")
        if lang != "FI":
            perustelu_raw = _translate_fi_statute_names(perustelu_raw)
        note = _latin1_safe(perustelu_raw)
        lbl_key = _RAQS_CRITERION_KEY.get(key, key)
        rows.append([
            Paragraph(_s(lang, lbl_key), row_label_style),
            Paragraph(_dots(sc), row_score_style),
            Paragraph(f"{sc}/5", row_score_style),
            Paragraph(note, row_note_style),
        ])

    if rows:
        score_tbl = Table(
            rows,
            colWidths=[4.8 * cm, 2.4 * cm, 1.2 * cm, 7.8 * cm],
            repeatRows=0,
        )
        score_tbl.setStyle(TableStyle([
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.HexColor("#e8edf8"), colors.HexColor("#f0f4ff")]),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        body_elems.append(score_tbl)

    if overall is not None:
        yhteenveto_raw = review.get("yhteenveto", "")
        if lang != "FI":
            yhteenveto_raw = _translate_fi_statute_names(yhteenveto_raw)
        body_elems.append(Paragraph(
            f"<b>{_s(lang, 'raqs_overall')}: {overall}/5</b> — "
            + _latin1_safe(yhteenveto_raw),
            summary_style,
        ))

    body_elems.append(Paragraph(_s(lang, "raqs_disclaimer"), disc_style))

    box = Table([[body_elems]], colWidths=[16.2 * cm], splitByRow=1)
    box.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 2.0, C_BD),
        ("BACKGROUND", (0, 0), (-1, -1), C_FILL),
        ("PADDING",    (0, 0), (-1, -1), 14),
    ]))
    return [PageBreak(), box]


def generate_pdf(
    inp: ApplicationInput,
    sections: dict,
    sources: list[dict],
    warning_flag: bool = False,
    prec_chunks: Optional[list] = None,
    prec_sources: Optional[list] = None,
    logo_path: Optional[str] = None,
    footer_name: Optional[str] = None,
    is_final: bool = True,
) -> bytes:
    """Rakenna PDF ja palauta bytes.

    `is_final`: whether this is the human-facing deliverable PDF (default True).
    Controls whether the RAQS outcome gets a "raqs_final" rollback checkpoint
    (TASO 1). Historically also had an is_final=False caller — the old
    generate_application_draft() draft-PDF build, whose bytes were discarded by
    every caller and whose RAQS pass fed nothing downstream — removed in the
    RAQS-removal task (2026-08-09); no caller currently passes False, but the
    parameter and this gating are kept since is_final's meaning (and the
    checkpoint-vs-no-checkpoint distinction) still applies to this function on
    its own terms.
    """
    prec_chunks  = prec_chunks  or []
    prec_sources = prec_sources or []

    # Inject hardcoded market data as an explicit source for transparency
    _country = getattr(inp, "country", "FI") or "FI"
    if inp.hanketyyppi == "BESS" and not any(s.get("id") == "bess_market_index" for s in sources):
        _md = _BESS_MARKET_DATA.get(_country, _BESS_MARKET_DATA["FI"])
        sources = list(sources) + [{
            "id":      "bess_market_index",
            "display": f"{_md['source']} — {_md['index']} {_md['unit']} ({_md['date']}, kovakoodattu arvio)",
            "country": "EU",
        }]

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

    # For non-FI output: translate Finnish statute names and deduplicate bullets
    _lang_pdf = getattr(inp, "lang", "FI") or "FI"
    if _lang_pdf != "FI":
        sections = _apply_en_statute_fixes(sections)

    # Language-consistency guardrail: log any Finnish markers that leaked into body text
    _lang_leaks = _sections_language_check(sections, _lang_pdf)
    if _lang_leaks:
        logger.warning(
            "[lang-leak] %s/%s/%s — %d Finnish marker(s) found in non-FI output: %s",
            getattr(inp, "hanketyyppi", "?"), getattr(inp, "country", "?"), _lang_pdf,
            len(_lang_leaks), _lang_leaks[:10],
        )

    buf    = io.BytesIO()
    now    = datetime.now().strftime("%d.%m.%Y")
    cfg    = _HANKE_CFG[inp.hanketyyppi]
    st     = _st()
    margin = 2.2 * cm

    canvas_cls = _make_canvas_cls(inp, now)

    _toc = TableOfContents()
    _toc.levelStyles = [
        ParagraphStyle(
            "toc_lvl0",
            fontName=PDF_FONT, fontSize=9.5, leading=16, spaceAfter=3,
            leftIndent=0, rightIndent=1.2*cm, firstLineIndent=0,
        ),
    ]
    _toc.dotsMinLevel = 0

    doc = _PDFDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        toc=_toc,
    )

    lang    = inp.lang or "FI"
    country = getattr(inp, "country", "FI") or "FI"
    story   = []

    # ── Branding: logo_path / footer_name — direct param overrides inp field ─
    _logo_path   = logo_path   or getattr(inp, "logo_path",   None) or _LOGO_PATH
    _footer_name = footer_name or getattr(inp, "footer_name", None)

    # ── Kansilehti ────────────────────────────────────────────────────────────
    # Logo top-right of cover page
    import os as _os
    if _logo_path and _os.path.exists(_logo_path):
        _logo_img = Image(_logo_path, width=3.5 * cm, height=1.5 * cm, kind="proportional")
        _logo_tbl = Table([[None, _logo_img]], colWidths=[12.0 * cm, 4.5 * cm])
        _logo_tbl.setStyle(TableStyle([
            ("ALIGN",  (1, 0), (1, 0), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(_logo_tbl)

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

    # ── Sisällysluettelo ──────────────────────────────────────────────────────
    _toc_title = {
        "FI": "Sisällysluettelo", "EN": "Table of Contents",
        "SE": "Innehållsförteckning", "DA": "Indholdsfortegnelse",
        "NO": "Innholdsfortegnelse", "PL": "Spis treści",
        "DE": "Inhaltsverzeichnis", "EE": "Sisukord",
    }.get(lang, "Table of Contents")
    story.append(PageBreak())
    story.append(Paragraph(_toc_title, ParagraphStyle(
        "toc_title", fontName=PDF_FONT_BOLD, fontSize=13, textColor=C_NAVY,
        spaceAfter=4, leading=17,
    )))
    story.append(_hr(C_NAVY, 1.0))
    story.append(Spacer(1, 3*mm))
    story.append(_toc)
    story.append(Spacer(1, 6*mm))

    # ── 1. Hankkeen kuvaus ────────────────────────────────────────────────────
    # PageBreak takaa puhtaan aloituksen. KeepTogether sisältää vain otsikon + HR +
    # ensimmäisen kappaleen — [:2] ylitti sivun korkeuden pitkällä AI-sisällöllä ja
    # aiheutti otsikon jäämisen yksin sivulle.
    story.append(PageBreak())
    _kuvaus_elems = _para_text(sections.get("kuvaus", "–"), st)
    story.append(KeepTogether([
        _toc_h2(_s(lang, "sec1"), st["h2"], "sec1"),
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
        _toc_h2(_s(lang, "sec2"), st["h2"], "sec2"),
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
        _toc_h2(_s(lang, "sec3"), st["h2"], "sec3"),
        _hr(),
        _luvat_tbl,
    ]))
    story.append(Spacer(1, 5*mm))
    _kaava_key = _KAAVA_KEY.get(inp.hanketyyppi, "kaava_generic")
    story.append(Paragraph(_s(lang, _kaava_key, country), st["body"]))
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
        _toc_h2(_s(lang, "sec4"), st["h2"], "sec4"),
        _hr(),
        *laki_bullets[:2],
    ]))
    for b in laki_bullets[2:]:
        story.append(b)
    story.append(Spacer(1, 4*mm))

    # Citation gap flagging (TASO 2, supplementary, FI-only for now — see
    # _flag_citation_gaps' module comment for the full reasoning and the
    # other-8-countries tracking note). Never affects what section 4 above
    # renders; purely a stored signal for human review, wrapped so it can
    # never block generation.
    if country in _CITATION_GAP_SUPPORTED_COUNTRIES:
        try:
            _gap_flags = _flag_citation_gaps(sections, laki_set)
            if _gap_flags:
                _citation_gap_store(_gap_flags, inp)
        except Exception:
            pass

    # ── 5. Liiteluettelo ──────────────────────────────────────────────────────
    story.append(PageBreak())
    _liite_tbl = _liitteet_table(inp.hanketyyppi, lang, country)
    story.append(KeepTogether([
        _toc_h2(_s(lang, "sec5"), st["h2"], "sec5"),
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
        _toc_h2(_s(lang, "sec6"), st["h2"], "sec6"),
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

    # ── Kannattavuusarvio (kannattavuuslaskenta v1, permit_ai/feasibility.py) ──
    from feasibility import calculate_feasibility as _calc_feasibility
    from xml.sax.saxutils import escape as _xml_escape
    _feas = _calc_feasibility(
        inp.hanketyyppi, inp.teho_mw, inp.kapasiteetti_mwh, getattr(inp, "country", "FI") or "FI"
    )
    if _feas:
        # feasibility.py's strings are plain text (may contain "&") — escape before
        # handing to Paragraph, which parses its content as a small XML-like markup.
        _feas = {k: (_xml_escape(v) if isinstance(v, str) else v) for k, v in _feas.items()}
        _feas_h = {
            "FI": "Alustava kannattavuusarvio",
            "EN": "Indicative Feasibility Estimate",
            "SE": "Preliminär lönsamhetsbedömning",
            "DA": "Foreløbig rentabilitetsvurdering",
            "NO": "Foreløpig lønnsomhetsvurdering",
            "PL": "Wstępna ocena opłacalności",
            "DE": "Vorläufige Wirtschaftlichkeitsschätzung",
        }.get(lang, "Alustava kannattavuusarvio")
        story.append(PageBreak())
        story.append(KeepTogether([
            Paragraph(_feas_h, st["h2"]),
            _hr(),
        ]))
        _feas_note_en = (
            "INDICATIVE ONLY — NOT INVESTMENT-GRADE. Derived from public benchmark ranges "
            "(IRENA, NREL ATB, BloombergNEF, Ember), not project-specific quotes, grid-"
            "connection assessments, or market forecasts. Actual project economics depend on "
            "site-specific factors not captured here. Do not use for investment decisions "
            "without professional financial due diligence."
        )
        _feas_note = {"FI": _feas["disclaimer"], "EN": _feas_note_en}.get(lang, _feas["disclaimer"])
        story.append(Paragraph(
            _feas_note,
            ParagraphStyle("feas_disclaimer", fontSize=8, leading=11.5, textColor=C_RED,
                           fontName=PDF_FONT_BOLD, spaceAfter=6),
        ))

        def _fmt_eur(v) -> str:
            return f"{v:,.0f}".replace(",", " ") + " €"

        def _fmt_range_eur(lo, hi) -> str:
            return _fmt_eur(lo) if lo == hi else f"{_fmt_eur(lo)} – {_fmt_eur(hi)}"

        _feas_th = ParagraphStyle("feas_th", fontSize=8.5, fontName=PDF_FONT_BOLD, textColor=C_WHITE)
        _feas_td = ParagraphStyle("feas_td", fontSize=8.5, leading=12)
        _feas_src = ParagraphStyle("feas_src", fontSize=7.5, leading=10, textColor=C_GRAY)

        _item_h    = {"FI": "Erä", "EN": "Item"}.get(lang, "Erä")
        _est_h     = {"FI": "Arvio", "EN": "Estimate"}.get(lang, "Arvio")
        _src_h     = {"FI": "Lähde", "EN": "Source"}.get(lang, "Lähde")
        _capex_h   = {"FI": "Investointikustannus (capex)", "EN": "Capital cost (capex)"}.get(lang, "Investointikustannus (capex)")
        _opex_h    = {"FI": "Käyttökustannus (opex, €/v)", "EN": "Operating cost (opex, €/yr)"}.get(lang, "Käyttökustannus (opex, €/v)")
        _price_h   = {"FI": "Sähkön hinta (historiallinen, €/MWh)", "EN": "Electricity price (historical, €/MWh)"}.get(lang, "Sähkön hinta (historiallinen, €/MWh)")
        _payback_h = {"FI": "Takaisinmaksuaika (arvio)", "EN": "Payback period (estimate)"}.get(lang, "Takaisinmaksuaika (arvio)")
        _yr        = {"FI": "v", "EN": "yr"}.get(lang, "v")
        _not_est   = {"FI": "Ei arvioitu (katso huomautus alla)", "EN": "Not estimated (see note below)"}.get(lang, "Ei arvioitu (katso huomautus alla)")

        capex_lo, capex_hi = _feas["capex_eur"]
        opex_lo, opex_hi = _feas["opex_eur_per_year"]

        _feas_rows = [
            [Paragraph(_item_h, _feas_th), Paragraph(_est_h, _feas_th), Paragraph(_src_h, _feas_th)],
            [Paragraph(_capex_h, _feas_td), Paragraph(_fmt_range_eur(capex_lo, capex_hi), _feas_td),
             Paragraph(_feas["capex_source"], _feas_src)],
            [Paragraph(_opex_h, _feas_td), Paragraph(_fmt_range_eur(opex_lo, opex_hi), _feas_td),
             Paragraph(_feas["opex_source"], _feas_src)],
        ]

        if _feas.get("payback_years") is not None:
            payback_fast, payback_slow = _feas["payback_years"]
            _payback_txt = (
                f"{payback_fast}–{payback_slow} {_yr}"
                if payback_fast is not None and payback_slow is not None
                else _feas.get("payback_note", "—")
            )
            if _feas.get("technology") == "battery_storage":
                # BESS payback (countries with a real
                # _ANCILLARY_REVENUE_EUR_MW_YEAR entry — FI/DA/DE/EE/LV/LT/
                # SE/NO/PL as of Phase 3) is driven by ancillary-market
                # revenue, not a flat electricity price — different
                # supporting field, different row, no
                # electricity_price_eur_mwh key exists in this branch.
                rev_lo, rev_hi = _feas["ancillary_revenue_eur_mw_year"]
                _rev_h = {
                    "FI": "Verkkopalveluiden tuotto (aFRR, arvio, €/MW/v)",
                    "EN": "Ancillary-market revenue (aFRR, estimate, €/MW/yr)",
                }.get(lang, "Verkkopalveluiden tuotto (aFRR, arvio, €/MW/v)")
                _feas_rows.append([
                    Paragraph(_rev_h, _feas_td),
                    Paragraph(_fmt_range_eur(rev_lo, rev_hi), _feas_td),
                    Paragraph(_feas["ancillary_revenue_source"], _feas_src),
                ])
            else:
                price_lo, price_hi = _feas["electricity_price_eur_mwh"]
                _feas_rows.append([
                    Paragraph(_price_h, _feas_td),
                    Paragraph(f"{price_lo:.0f} – {price_hi:.0f} €/MWh", _feas_td),
                    Paragraph(_feas["electricity_price_source"], _feas_src),
                ])
            _feas_rows.append([
                Paragraph(_payback_h, _feas_td),
                Paragraph(_payback_txt, _feas_td),
                Paragraph(_feas.get("fx_rate_note", ""), _feas_src),
            ])
        else:
            _feas_rows.append([
                Paragraph(_payback_h, _feas_td),
                Paragraph(_not_est, _feas_td),
                Paragraph(_feas.get("fx_rate_note", ""), _feas_src),
            ])

        _feas_tbl = Table(_feas_rows, colWidths=[5.5*cm, 4.5*cm, 6.5*cm], repeatRows=1)
        _feas_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
            ("GRID",          (0, 0), (-1, -1), 0.4, C_DGRAY),
            ("PADDING",       (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(_feas_tbl)

        # BESS's payback_note is a MANDATORY disclaimer (aFRR-only scope, source
        # + computation date, DA zone caveat, revenue-stacking caveat) even when
        # payback_years computed successfully — unlike generation techs, where
        # the note only exists for the negative-cash-flow edge case and is
        # otherwise absent, so "payback_years is None" isn't a safe proxy here.
        _feas_show_note = (
            _feas.get("payback_years") is None or _feas.get("technology") == "battery_storage"
        ) and _feas.get("payback_note")
        if _feas_show_note:
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                _feas["payback_note"],
                ParagraphStyle("feas_note2", fontSize=7.5, leading=10.5, textColor=C_GRAY, fontName=PDF_FONT),
            ))

        # ENTSO-E cannibalization disclaimer — MANDATORY whenever the price row
        # above came from real day-ahead market data rather than the static
        # benchmark table (electricity_price_note is only ever set in that
        # case, see feasibility.py). Independent of payback_note/_feas_show_note
        # above — solar/wind can carry both notes at once (a negative-cashflow
        # payback_note AND the cannibalization caveat aren't mutually exclusive).
        if _feas.get("electricity_price_note"):
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                _feas["electricity_price_note"],
                ParagraphStyle("feas_note3", fontSize=7.5, leading=10.5, textColor=C_GRAY, fontName=PDF_FONT),
            ))
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

    # ── Täydennyslista (jos TÄYDENNETTÄVÄ-merkintöjä löytyi) ─────────────────
    _gaps = _extract_gaps(sections)
    if _gaps:
        for _elem in _taydennyslista_page(_gaps, st, lang):
            story.append(_elem)

    # ── RAQS: reviewing agent — Haiku-arvio sisällön laadusta ────────────────
    _raqs_result = _raqs_review(sections, inp, sources=sources, laki_set=laki_set)
    if _raqs_result and is_final and getattr(inp, "generation_id", ""):
        # Rollback checkpoint (TASO 1): only the final, human-facing RAQS pass —
        # this is the terminal step before the human approval gate (see the
        # rollback task's investigation; the draft-PDF RAQS pass is excluded,
        # is_final=False there).
        _retrieval_trace.save_checkpoint(
            generation_id=inp.generation_id,
            step="raqs_final",
            state={k: v for k, v in _raqs_result.items() if k != "_lang"},
        )
    if _raqs_result:
        _raqs_lang = _raqs_result.pop("_lang", lang)
        for _elem in _raqs_page(_raqs_result, st, _raqs_lang):
            story.append(_elem)

    # ── Loppumerkintä ─────────────────────────────────────────────────────────
    story.append(_hr(C_NAVY, 1.0))
    story.append(Paragraph(
        _footer_name if _footer_name else _s(lang, "footer"),
        ParagraphStyle("end", fontSize=7.5, textColor=C_GRAY, alignment=TA_CENTER, leading=11),
    ))

    doc.multiBuild(story, canvasmaker=canvas_cls)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Pääfunktio
# ─────────────────────────────────────────────────────────────────────────────

def generate_application_draft(inp: ApplicationInput) -> tuple:
    """Generoi luonnos (sections + sources) ilman oikolukua. Palauttaa
    (None, sections, sources).

    RAQS-removal task (2026-08-09): this used to also render a draft PDF (via
    generate_pdf(..., is_final=False)) and run a RAQS review on it, purely so
    the first tuple element could hold pdf_bytes. Confirmed via grep that no
    caller — backend/main.py's four call sites or test_cot_validation.py —
    ever reads that first element; only `sections`/`sources` propagate forward
    into apply_proofread_to_pdf(). That made the draft-PDF render and its RAQS
    pass (a Haiku call) pure dead work every generation paid for. Removed;
    first element is now always None but kept so every caller's existing
    `pdf_bytes, sections, sources = ...`-style unpack still works unchanged.
    """
    _LAST_TIMING.clear()
    _LAST_TIMING["hanketyyppi"] = inp.hanketyyppi
    _LAST_TIMING["country"] = inp.country
    _LAST_TIMING["t0_start"] = _time.monotonic()
    _LAST_TIMING["t0a_rag_lock_wait_start"] = _time.monotonic()
    with _RAG_LOCK:
        _LAST_TIMING["t1_rag_lock_acquired"] = _time.monotonic()
        rag_ctx, sources, warning_flag, prec_chunks, prec_sources, _ = \
            _rag_context(inp.hanketyyppi, inp.country or "FI", generation_id=inp.generation_id, vaihe=inp.hankkeen_vaihe)
        _LAST_TIMING["t2_rag_context_done"] = _time.monotonic()
    _LAST_TIMING["t3_before_generate_sections"] = _time.monotonic()
    try:
        sections = _generate_sections(inp, rag_ctx, prec_chunks, prec_sources)
    except Exception as exc:
        _LAST_TIMING["t4_generate_sections_FAILED"] = _time.monotonic()
        _LAST_TIMING["error"] = f"{type(exc).__name__}: {exc}"
        raise
    _LAST_TIMING["t4_generate_sections_done"] = _time.monotonic()
    _lang = inp.lang or "FI"
    sections = _final_polish(sections, _lang, inp.country or "FI", inp.hanketyyppi)
    _LAST_TIMING["t5_final_polish_done"] = _time.monotonic()
    if inp.generation_id:
        # Rollback checkpoint (TASO 1): the exact `sections` state that feeds
        # proofread next — post-final_polish, since that's what actually
        # propagates forward (not the raw _generate_sections() output).
        _retrieval_trace.save_checkpoint(
            generation_id=inp.generation_id, step="draft", state={"sections": sections},
        )
    return None, sections, sources


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
    sections = _proofread_sections(sections, generation_id=inp.generation_id)
    # 2026-08-23: granular checkpoints added below, narrowing the stuck-job
    # investigation past t7c_proofread_succeeded (see _LAST_TIMING) -- a job
    # was caught live completing _proofread_sections() successfully (t7c
    # written) but never reaching arq_task_generate_permit's own "pdf done"
    # log line or _proofread_store status update, with no error surfaced
    # anywhere. This tail (second _final_polish(), the checkpoint save,
    # generate_pdf()) has no explicit timeout the way the two Claude calls
    # do -- these checkpoints exist purely to show which of the three it is
    # the next time a job gets caught stuck here. Purely additive, no
    # behavior change.
    _LAST_TIMING["t8_final_polish2_start"] = _time.monotonic()
    sections = _final_polish(sections, _lang, inp.country or "FI", inp.hanketyyppi)
    _LAST_TIMING["t8c_final_polish2_done"] = _time.monotonic()
    if inp.generation_id:
        # Rollback checkpoint (TASO 1): the exact `sections` state that feeds
        # the final PDF next.
        _LAST_TIMING["t9_checkpoint_save_start"] = _time.monotonic()
        _retrieval_trace.save_checkpoint(
            generation_id=inp.generation_id, step="proofread", state={"sections": sections},
        )
        _LAST_TIMING["t9c_checkpoint_save_done"] = _time.monotonic()
    _LAST_TIMING["t10_generate_pdf_start"] = _time.monotonic()
    pdf = generate_pdf(
        inp, sections, sources,
        warning_flag, prec_chunks or [], prec_sources or [],
        # is_final defaults to True — this is the human-facing deliverable PDF.
    )
    _LAST_TIMING["t10c_generate_pdf_done"] = _time.monotonic()
    return pdf


def generate_application(inp: ApplicationInput) -> str:
    """Generoi lupahakemus-PDF ja palauta tallennuspolku."""
    logger.warning("DEBUG TEST: äö toimii - raw=%s", repr("testäö"))
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    print(f"[1/3] Haetaan RAG-konteksti ({inp.hanketyyppi}, maa={inp.country or 'FI'})…")
    rag_ctx, sources, warning_flag, prec_chunks, prec_sources, _ = \
        _rag_context(inp.hanketyyppi, inp.country or "FI", generation_id=inp.generation_id, vaihe=inp.hankkeen_vaihe)
    print(f"      {len(rag_ctx.split())} sanaa, lähteet: {[s['display'] for s in sources]}")
    if warning_flag:
        print("      ⚠️  RAG_WARN: rajallinen lähdeaineisto")

    print("[2/4] Generoidaan hakemusteksti (Claude)…")
    sections = _generate_sections(inp, rag_ctx, prec_chunks, prec_sources)
    print(f"      Osiot: {list(sections.keys())}")
    if inp.generation_id:
        # Rollback checkpoint (TASO 1): this single-pass path has no _final_polish
        # between draft generation and proofread, unlike generate_application_draft()
        # — checkpoint whatever state actually feeds proofread next in this path.
        _retrieval_trace.save_checkpoint(
            generation_id=inp.generation_id, step="draft", state={"sections": sections},
        )

    print("[3/4] Oikoluku ja tekstikorjaus (Claude + säännöt)…")
    _lang = inp.lang or "FI"
    sections = _proofread_sections(sections, generation_id=inp.generation_id)
    sections = _final_polish(sections, _lang, inp.country or "FI", inp.hanketyyppi)
    if inp.generation_id:
        _retrieval_trace.save_checkpoint(
            generation_id=inp.generation_id, step="proofread", state={"sections": sections},
        )

    print("[4/4] Rakennetaan PDF…")
    pdf_bytes = generate_pdf(inp, sections, sources, warning_flag, prec_chunks, prec_sources)
    # is_final defaults to True — this single-pass function's PDF is the deliverable.

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
    import uuid as _uuid
    test_inp = ApplicationInput(
        hanketyyppi      = "BESS",
        kiinteistotunnus = "636-439-4-711",
        teho_mw          = 1.0,
        kunta            = "Pöytyä",
        hakija           = "Carbon Zero Finland Oy",
        generation_id    = _uuid.uuid4().hex[:10],
    )
    print(f"[trace] generation_id = {test_inp.generation_id}")
    path = generate_application(test_inp)
    os.system(f"open '{path}'")
