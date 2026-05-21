"""PDF-raporttigenerointi BESS-kaavoituskartoitukselle (ReportLab)."""

import base64
import io
import os
import re
from datetime import datetime
from typing import Optional

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "nce_energy_logo.png")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable, Image as RLImage, KeepTogether, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

C_RED    = colors.HexColor("#e94560")
C_NAVY   = colors.HexColor("#16213e")
C_GRAY   = colors.HexColor("#8899aa")
C_GREEN  = colors.HexColor("#4caf50")
C_ORANGE = colors.HexColor("#ff9800")
C_WHITE  = colors.white

# ─────────────────────────────────────────────────────────────────────────────
# Tekstikorjaus — vanhat viranomaisnimet → 2026-nimet
# ─────────────────────────────────────────────────────────────────────────────

_POSTPROCESS_RULES: list[tuple[str, str]] = [
    (r'\bAVI:sta\b',   'Lupa- ja valvontavirastosta'),
    (r'\bAVI:ssa\b',   'Lupa- ja valvontavirastossa'),
    (r'\bAVI:lta\b',   'Lupa- ja valvontavirastolta'),
    (r'\bAVI:lle\b',   'Lupa- ja valvontavirastolle'),
    (r'\bAVI:ksi\b',   'Lupa- ja valvontavirastoksi'),
    (r'\bAVI:n\b',     'Lupa- ja valvontaviraston'),
    (r'\bAVI\b',       'Lupa- ja valvontavirasto'),
    (r'\b[Aa]luehallintovirastosta\b',  'Lupa- ja valvontavirastosta'),
    (r'\b[Aa]luehallintovirastossa\b',  'Lupa- ja valvontavirastossa'),
    (r'\b[Aa]luehallintovirastolta\b',  'Lupa- ja valvontavirastolta'),
    (r'\b[Aa]luehallintovirastolle\b',  'Lupa- ja valvontavirastolle'),
    (r'\b[Aa]luehallintoviraston\b',    'Lupa- ja valvontaviraston'),
    (r'\b[Aa]luehallintovirasto\b',     'Lupa- ja valvontavirasto'),
    (r'\bELY\b(?!-)',  'ELY-keskus'),
    (r'(?<!/ )MRL\s+132/1999',  'Rakentamislaki (751/2023) / MRL 132/1999'),
]


def _postprocess_text(text: str) -> str:
    """Korjaa vanhat viranomaisnimet ja lakiviitteet automaattisesti."""
    for pattern, replacement in _POSTPROCESS_RULES:
        text = re.sub(pattern, replacement, text)
    return text


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title":      ParagraphStyle("title",      parent=base["Title"], fontSize=22,
                                     textColor=C_NAVY, spaceAfter=2, fontName="Helvetica-Bold"),
        "title_sub":  ParagraphStyle("title_sub",  fontSize=11, textColor=C_RED,
                                     spaceAfter=2, fontName="Helvetica-Bold"),
        "subtitle":   ParagraphStyle("subtitle",   fontSize=9,  textColor=C_GRAY, spaceAfter=6),
        "h2":         ParagraphStyle("h2",         fontSize=12, textColor=C_RED,
                                     spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold"),
        "body":       ParagraphStyle("body",       fontSize=9,  spaceAfter=4, leading=13),
        "small":      ParagraphStyle("small",      fontSize=8,  textColor=C_GRAY, leading=11),
        "footer":     ParagraphStyle("footer",     fontSize=7,  textColor=C_GRAY, alignment=TA_CENTER),
        "disclaimer": ParagraphStyle("disclaimer", fontSize=7,  textColor=C_GRAY, leading=11,
                                     alignment=TA_CENTER, spaceBefore=4, spaceAfter=2),
        "disc_label": ParagraphStyle("disc_label", fontSize=6,  textColor=C_GRAY, alignment=TA_CENTER,
                                     fontName="Helvetica-Bold", spaceAfter=4),
        "confid":     ParagraphStyle("confid",     fontSize=8,  textColor=C_RED,
                                     alignment=TA_RIGHT, fontName="Helvetica-Bold"),
        "contact":    ParagraphStyle("contact",    fontSize=8,  textColor=C_NAVY, leading=12),
    }


def _score_color(score: int) -> colors.Color:
    if score >= 70: return C_GREEN
    if score >= 40: return C_ORANGE
    return C_RED


def _grid_operator(grid_connection: str) -> tuple[str, str]:
    """Palauttaa (operaattorin nimi, liityntätarjous-URL tai '')."""
    gc = grid_connection.lower()
    if "caruna" in gc:
        return ("Caruna Oy", "https://www.caruna.fi/asiakaspalvelu/liittyminen/uusi-liittyman")
    if "elenia" in gc:
        return ("Elenia Oy", "https://www.elenia.fi/sahko/liittyminen")
    if "fingrid" in gc:
        return ("Fingrid Oyj", "https://www.fingrid.fi/kantaverkko/liittyminen-kantaverkkoon/")
    return ("Jakeluverkkoyhtiö", "")


def generate_bess_report(
    kiinteistotunnus: str,
    property_data: Optional[dict] = None,
    analysis_data: Optional[dict] = None,
    map_image_b64: Optional[str] = None,
    project_owner:   str = "Carbon Zero Finland",
    project_name:    str = "Standalone BESS 1 MW",
    power_mw:        float = 1.0,
    grid_connection: str = "Caruna 20 kV (Jakeluverkko)",
    market:          str = "FCR (Frequency Containment Reserve)",
) -> bytes:
    """
    Luo BESS-kaavoituskartoitusraportin PDF-muodossa.
    map_image_b64: valinnainen base64-koodattu PNG karttakuvasta.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
        title=f"BESS-kaavoituskartoitus {kiinteistotunnus}", author="BESS Tool")

    st = _styles()
    story = []
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    p = property_data or {}
    a = analysis_data or {}
    op_name, op_url = _grid_operator(grid_connection)

    kuntanimi      = a.get("kuntanimi", p.get("kuntanimi", "Pöytyä"))
    kuntanimi_gen  = a.get("kuntanimi_gen", kuntanimi + "n")
    pelastuslaitos = a.get("pelastuslaitos", "Paikallinen pelastuslaitos")
    ely_center     = a.get("ely_center", "Paikallinen ELY-keskus")
    muni_code      = a.get("muni_code", kiinteistotunnus.split("-")[0].zfill(3))
    lupapiste_url  = a.get("lupapiste_url", f"https://www.lupapiste.fi/?municipality={muni_code}")
    def sec(*items):
        return KeepTogether([i for i in items if i is not None])

    # ── Otsikkorivi: logo vasemmalla + otsikko oikealla ─────────────────────
    logo_cell: object = Spacer(1, 1*mm)
    if os.path.exists(_LOGO_PATH):
        logo_w = 7.0 * cm
        logo_h = logo_w / 3.0
        _logo = RLImage(_LOGO_PATH, width=logo_w, height=logo_h)
        _logo.hAlign = "LEFT"
        logo_cell = _logo

    title_cell = [
        Paragraph("Akkuvarastohankkeen sijaintianalyysi", st["title"]),
        Paragraph("BESS-kaavoituskartoitus", st["title_sub"]),
        Paragraph(f"{kiinteistotunnus} · {kuntanimi} · {now}", st["subtitle"]),
    ]

    hdr = Table(
        [[logo_cell, title_cell]],
        colWidths=[7.5 * cm, 9.5 * cm],
    )
    hdr.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (0, -1),  8),
        ("RIGHTPADDING",  (1, 0), (1, -1),  0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(hdr)
    story.append(HRFlowable(width="100%", thickness=2, color=C_RED, spaceAfter=6))

    # ── Tekijätiedot ja luottamuksellisuus ───────────────────────────────────
    author_left = [
        Paragraph("<b>NCE Energy</b>", st["contact"]),
        Paragraph("ncenergy.fi  |  info@ncenergy.fi", st["contact"]),
    ]
    author_right = [
        Paragraph("LUOTTAMUKSELLINEN", st["confid"]),
        Paragraph("Vain vastaanottajan käyttöön", ParagraphStyle(
            "confid_sub", fontSize=7, textColor=C_GRAY, alignment=TA_RIGHT)),
    ]
    author_tbl = Table([[author_left, author_right]], colWidths=[9*cm, 8*cm])
    author_tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    story.append(author_tbl)

    # ── Etusivun vastuuvapauslauseke ─────────────────────────────────────────
    C_DISC_BG = colors.HexColor("#f8f8f8")
    disc_notice = Table([[Paragraph(
        "Tämä raportti on laadittu esiselvityskäyttöön ja perustuu julkisiin avoimiin tietoihin. "
        "Raportti ei korvaa virallisia lupaselvityksiä eikä ole sitova. "
        "NCE Energy ei vastaa mahdollisten tietojen puutteellisuudesta tai muutoksista.",
        ParagraphStyle("front_disc", fontSize=7.5, textColor=C_GRAY, leading=11),
    )]], colWidths=[17*cm])
    disc_notice.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 0.5, C_GRAY),
        ("BACKGROUND", (0, 0), (-1, -1), C_DISC_BG),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(disc_notice)
    story.append(Spacer(1, 6*mm))

    # ── Hankkeen tiedot ──────────────────────────────────────────────────────
    story.append(sec(
        Paragraph("Hankkeen tiedot", st["h2"]),
        _table([
            ["Kenttä", "Arvo"],
            ["Omistaja / kehittäjä", project_owner],
            ["Hanke",                project_name],
            ["Teho",                 f"{power_mw} MW"],
            ["Verkkoliityntä",       grid_connection],
            ["Tavoitemarkkinat",     market],
            ["Sijaintikunta",        kuntanimi],
            ["Kiinteistötunnus",     kiinteistotunnus],
            ["Raportin päivämäärä",  now],
        ]),
        Spacer(1, 6*mm),
    ))

    # ── Karttakuva ───────────────────────────────────────────────────────────
    if map_image_b64:
        map_items = [Paragraph("Karttanäkymä", st["h2"])]
        try:
            img_bytes = base64.b64decode(map_image_b64)
            from PIL import Image as PILImage
            pil = PILImage.open(io.BytesIO(img_bytes))
            w_px, h_px = pil.size
            aspect = h_px / w_px if w_px else 0.625
            img_w = 17 * cm
            img_h = min(img_w * aspect, 12 * cm)
            img_buf = io.BytesIO(img_bytes)
            img = RLImage(img_buf, width=img_w, height=img_h)
            img.hAlign = "CENTER"
            map_items.append(img)
        except Exception:
            map_items.append(Paragraph("Karttakuva ei saatavilla.", st["small"]))
        map_items.append(Spacer(1, 6*mm))
        story.append(sec(*map_items))

    # ── 1. Kiinteistötiedot ──────────────────────────────────────────────────
    manual_pinta_ala_ha = a.get("manual_pinta_ala_ha")
    area_ha = manual_pinta_ala_ha if manual_pinta_ala_ha else p.get("area_ha")
    if area_ha is not None:
        if area_ha < 1.0:
            varattu_ha = round(area_ha, 2)
            varattu_note = "koko kiinteistö (alle 1 ha)"
        else:
            tarvittava = round(max(0.25, power_mw * 0.5), 2)
            varattu_ha = round(min(area_ha, tarvittava), 2)
            varattu_note = "tarvittava osa"
        varattu_str   = f"{varattu_ha:.2f} ha ({varattu_note})"
        calc_area_str = f"{area_ha:.2f} ha (laskennallinen, INSPIRE WFS polygon)"
        if manual_pinta_ala_ha:
            reg_area_str = f"{manual_pinta_ala_ha:.2f} ha (Manuaalinen syöte)"
        else:
            reg_area_str = "Ei saatavilla ilman MML API-avainta"
    else:
        varattu_str   = "–"
        calc_area_str = "–"
        reg_area_str  = "–"

    story.append(sec(
        Paragraph("1. Kiinteistötiedot", st["h2"]),
        _table([
            ["Kenttä", "Arvo"],
            ["Kiinteistötunnus", kiinteistotunnus],
            ["Kunta", kuntanimi],
            ["Kylä", p.get("kylanimi", "Kyrö")],
            ["Rekisteripinta-ala (MML)", reg_area_str],
            ["Laskennallinen pinta-ala", calc_area_str],
            ["Hankkeelle varattu määräala", varattu_str],
            ["Koordinaatit", (
                f"{float(a['center_lat']):.6f}, {float(a['center_lon']):.6f}"
                if a.get('center_lat') is not None and a.get('center_lon') is not None
                else "–"
            )],
            ["Lähde", "MML INSPIRE WFS (cp:CadastralParcel)"],
        ]),
        Paragraph(
            f"Hankkeelle varataan {varattu_str}. "
            "Lopullinen pinta-ala ja lohkominen sovitaan kunnan kanssa lupaprosessin yhteydessä.",
            st["body"]),
    ))

    # ── 2. Sähköverkko + tiesuoja-alue ───────────────────────────────────────
    grid_m   = a.get("nearest_grid_m")
    grid_str = f"{grid_m} m" if grid_m is not None else "Ei tiedossa"
    _gm = grid_m or 9999
    grid_eval = "✓ Erinomainen" if _gm < 1000 else ("⚠ Hyvä" if _gm < 2000 else "✗ Tarkista")

    buf_ok   = a.get("powerline_buffer_ok", True)
    buf_str  = "OK (≥25 m)" if buf_ok else f"⚠ VAROITUS — {grid_m} m (<25 m suojavyöhyke)"
    buf_eval = "✓ OK" if buf_ok else "✗ Rakentamiskielto"

    road_m    = a.get("nearest_road_m")
    road_name = a.get("nearest_road_name", "")
    road_ok   = a.get("road_protection_ok", True)
    road_str  = (f"{road_m} m ({road_name})" if road_m is not None and road_name
                 else (f"{road_m} m" if road_m is not None else "Ei tiedossa"))
    road_eval = "✓ OK (>20 m)" if road_ok else "✗ Suoja-alueella (<20 m)"

    grid_items = [
        Paragraph("2. Sähköverkon liitynnän soveltuvuus ja suoja-alueet", st["h2"]),
        _table([
            ["Parametri", "Arvo", "Arviointi"],
            [f"Lähin johto ({op_name})",         grid_str,  grid_eval],
            ["Johtotyyppi",                       a.get("grid_type", "–"), ""],
            ["Voimajohdon suojavyöhyke (25 m)",   buf_str,  buf_eval],
            ["Lähin valtatie (suoja 20 m)",        road_str, road_eval],
            ["Verkonhaltija",                     op_name,  ""],
            ["Liityntätarjous",                   op_url if op_url else "Ota yhteyttä paikalliseen jakeluverkkoyhtiöön", ""],
            ["Datalähde",                         "OpenStreetMap Overpass", ""],
            ["BESS-suositus",                     "< 1 km jakeluverkkoon", ""],
        ]),
        Spacer(1, 4*mm),
        Paragraph(
            f"Hanke liitetään {op_name}:n verkkoon. "
            "Etäisyys <1 km on erinomainen; 1–2 km on toteutettavissa; "
            ">2 km nostaa liityntäkustannuksia (10 000–50 000 €). ",
            st["body"]),
    ]
    if not buf_ok:
        grid_items.append(_alert(
            f"Voimajohdon suojavyöhyke: kiinteistö on {grid_m} m päässä voimajohdosta "
            "(rakentamiskielto <25 m, Sähköturvallisuuslaki 1135/2016 § 100 ja "
            "maankäyttö- ja rakennuslaki). Neuvottele verkonhaltijan kanssa ennen suunnittelun jatkamista."
        ))
    if not road_ok:
        grid_items.append(_alert(
            f"Tiesuoja-alue: kiinteistö on {road_m} m päässä valtatien reunasta "
            "(rakentamiskielto <20 m, Maantielaki 503/2005 § 44). "
            "Pyydä ELY-keskukselta poikkeuslupa tai sijoita laitos kauemmas."
        ))
    story.append(sec(*grid_items))

    # ── 3. Pohjavesi ─────────────────────────────────────────────────────────
    gw = a.get("groundwater_overlap")
    gw_na = a.get("groundwater_unavailable", False)
    gw_cls = a.get("groundwater_class", "")
    if gw_na:
        gw_str = "Ei saatavilla (SYKE offline)"
        gw_eval = "– Ei dataa"
    elif gw:
        cls_label = f" — Luokka {gw_cls}" if gw_cls else ""
        gw_str = f"Kyllä{cls_label} (!)"
        gw_eval = "(!) Selvitä (kriittinen)" if gw_cls in ("1", "1E") else "⚠ Selvitä"
    else:
        gw_str = "Ei ✓"
        gw_eval = "✓ OK"
    story.append(sec(
        Paragraph("3. Pohjavesialueet", st["h2"]),
        _table([
            ["Tarkasteltava tekijä", "Tulos", "Arviointi"],
            ["Pohjavesialue (SYKE)", gw_str, gw_eval],
            ["Datalähde", "SYKE paikkatiedot.ymparisto.fi (syke_vhspohjavesi)", ""],
        ]),
        Spacer(1, 4*mm),
        Paragraph(
            "Pohjavesialueella sijaitsevan BESS-laitoksen suunnittelussa on huomioitava "
            "akkuvaraston sähkökemiallisten aineiden varastointimääräykset ja ympäristölupa.",
            st["body"]),
    ))

    # ── 4. Muinaismuistot ─────────────────────────────────────────────────────
    h_overlap = a.get("heritage_overlap", False)
    h_na      = a.get("heritage_unavailable", False)
    h_src     = a.get("heritage_source", "none")
    if h_na:
        h_str     = "Ei saatavilla"
        h_eval    = "– Ei dataa"
        h_src_str = "Museovirasto WFS + OSM (molemmat epäonnistuivat)"
    elif h_overlap:
        h_str     = "Kyllä ⚠"
        h_eval    = "⚠ Selvitä (Muinaismuistolaki 295/1963)"
        h_src_str = ("Museovirasto INSPIRE WFS" if h_src == "museovirasto"
                     else "OpenStreetMap (epävirallinen)")
    else:
        h_str     = "Ei ✓"
        h_eval    = "✓ OK"
        h_src_str = ("Museovirasto INSPIRE WFS" if h_src == "museovirasto"
                     else "OpenStreetMap (epävirallinen — tarkista kyppi.fi/palveluikkuna/mjreki/)")
    heritage_items = [
        Paragraph("4. Muinaismuistot ja kulttuuriympäristö", st["h2"]),
        _table([
            ["Tarkasteltava tekijä", "Tulos", "Arviointi"],
            ["Muinaismuistot / RKY (Museovirasto)", h_str, h_eval],
            ["Datalähde", h_src_str, ""],
        ]),
        Spacer(1, 4*mm),
        Paragraph(
            "Kiinteät muinaisjäännökset (Muinaismuistolaki 295/1963) ja RKY-alueet ovat "
            "automaattisesti rauhoitettuja. BESS-laitos ei saa sijoittua muinaismuistokohteen "
            "välittömään läheisyyteen ilman Museoviraston lupaa.",
            st["body"]),
    ]
    if h_overlap and not h_na:
        heritage_items.append(_alert(
            "Muinaismuisto tai RKY-kohde havaittu alueella! Pyydä Museoviraston lausunto ennen "
            "rakentamislupahakemuksen jättämistä (Muinaismuistolaki 295/1963)."
        ))
    if h_src == "osm" and not h_na:
        heritage_items.append(Paragraph(
            "Huom: Tiedot OSM:sta — virallinen tarkistus: kyppi.fi/palveluikkuna/mjreki/",
            st["small"]
        ))
    story.append(sec(*heritage_items))

    # ── 5. Ympäristö ─────────────────────────────────────────────────────────
    natura = a.get("natura_overlap", False)
    story.append(sec(
        Paragraph("5. Ympäristö- ja suojelualueet", st["h2"]),
        _table([
            ["Tarkasteltava tekijä", "Tulos", "Arviointi"],
            ["Natura 2000 -alue", "Kyllä ⚠" if natura else "Ei ✓", "⚠ YVA mahdollinen" if natura else "✓ OK"],
            ["Datalähde", "SYKE sy:natura2000_sac_fi", ""],
        ]),
    ))

    # ── 6. Asutus ────────────────────────────────────────────────────────────
    bldg_m   = a.get("nearest_building_m")
    bldg_str = f"{bldg_m} m" if bldg_m is not None else "Ei tiedossa"
    bldg_ok  = bldg_m is None or bldg_m > 300
    asutus_items = [
        Paragraph("6. Asutuksen etäisyys", st["h2"]),
        _table([
            ["Parametri", "Arvo", "Arviointi"],
            ["Lähin rakennus (OSM)", bldg_str, "✓ OK (>300 m)" if bldg_ok else "⚠ Lähellä (<300 m)"],
            ["Etäisyyssuositus", "> 300 m asutuksesta", ""],
            ["Datalähde", "OpenStreetMap Overpass", ""],
        ]),
    ]
    if not bldg_ok and bldg_m is not None:
        asutus_items += [
            Spacer(1, 2*mm),
            _alert(
                f"Meluselvitys ja naapurikuuleminen pakollinen – lähin rakennus {bldg_m} m "
                f"(<300 m asutuksesta). BESS-laitoksen jäähdytysjärjestelmä voi tuottaa "
                f"melua 45–60 dB(A). Selvitys on toimitettava lupahakemuksen liitteenä "
                f"(YSL 527/2014, ympäristönsuojeluasetus)."
            ),
            Spacer(1, 2*mm),
            _warning(
                f"Lähin rakennus sijaitsee {bldg_m} m etäisyydellä. "
                f"Pelastusopiston LION-suositus (12/2025) edellyttää selvitystä onko rakennus asuttu. "
                f"Jos asuttu, meluselvitys ja pelastuslausunto ovat pakollisia "
                f"ennen rakentamislupahakemuksen jättämistä."
            ),
        ]
    story.append(sec(*asutus_items))

    # ── 7. Kaavoitus ─────────────────────────────────────────────────────────
    zoning_ok          = a.get("zoning_ok", True)
    zoning_unavailable = a.get("zoning_unavailable", False)
    manual_kaavoitus   = a.get("manual_kaavoitus")
    is_rural           = zoning_ok
    zoning_eval = ("– Ei dataa" if zoning_unavailable
                   else ("✓ Sopii (STR tarvitaan)" if zoning_ok else "⚠ Asemakaava – selvitettävä"))
    if manual_kaavoitus:
        zoning_src = "Manuaalinen syöte"
    elif not zoning_unavailable:
        zoning_src = "MML Maastotiedot WFS"
    else:
        zoning_src = "Ei saatavilla (MML API-avain puuttuu)"
    zoning_items = [
        Paragraph("7. Kaavoitustilanne ja maankäyttölupa", st["h2"]),
        _table([
            ["Kaavatyyppi", "Tilanne", "Huomio"],
            ["Kaavatilanne",           a.get("zoning_status", "–"), zoning_eval],
            ["Suunnittelutarveratkaisu",
             "Vaaditaan (maaseuturakentaminen)" if is_rural else "Ei koske asemakaava-aluetta",
             "Pakollinen ⚠" if is_rural else "–"],
            ["Naapurikuuleminen",      "Pakollinen lupaprosessissa", "Pakollinen"],
            ["Maankäyttö",             a.get("land_use", "–"),       ""],
            ["Datalähde",              zoning_src,                   ""],
        ]),
        Spacer(1, 4*mm),
        (Paragraph(
            f"Maaseutualueella (ei asemakaavaa) BESS-laitos edellyttää suunnittelutarveratkaisua "
            f"(STR) {kuntanimi_gen} kunnalta ennen rakennuslupaa (Rakentamislaki 751/2023, § 46). "
            "STR:n myöntäminen edellyttää, ettei hanke aiheuta haittaa kaavoitukselle, "
            "liikenteelle tai luonnonarvoille.",
            st["body"]) if is_rural else None),
    ]
    story.append(sec(*zoning_items))

    # ── 7b. Maaperä ja tulvavaara ────────────────────────────────────────────
    maaperalaaji     = a.get("maaperalaaji", "Ei tiedossa")
    maapera_source   = a.get("maaperalaaji_source", "unavailable")
    manual_maapera   = a.get("manual_maapera")
    flood_overlap    = a.get("flood_overlap", False)
    flood_unavailable = a.get("flood_unavailable", False)
    manual_tulvavaara = a.get("manual_tulvavaara")

    if maapera_source == "manual":
        maapera_src_str = "Manuaalinen syöte"
    elif maapera_source and maapera_source != "unavailable":
        maapera_src_str = f"GTK maaperäkartta ({maapera_source})"
    else:
        maapera_src_str = "GTK ei saatavilla"

    if flood_unavailable and not manual_tulvavaara:
        flood_str  = "Ei saatavilla (SYKE offline)"
        flood_eval = "– Ei dataa"
        flood_src  = "SYKE tulvavaara-aineisto"
    elif flood_overlap:
        flood_str  = "Kyllä ⚠"
        flood_eval = "⚠ Selvitä"
        flood_src  = "Manuaalinen syöte" if manual_tulvavaara else "SYKE tulvavaara-aineisto"
    else:
        flood_str  = "Ei ✓"
        flood_eval = "✓ OK"
        flood_src  = "Manuaalinen syöte" if manual_tulvavaara else "SYKE tulvavaara-aineisto"

    if not manual_maapera and (not maaperalaaji or maaperalaaji == "Ei tiedossa"):
        maapera_eval = "– Ei dataa"
    elif maaperalaaji in ("Kallio", "Moreeni"):
        maapera_eval = "✓ Hyvä maaperä BESS:lle"
    elif maaperalaaji == "Turve":
        maapera_eval = "⚠ Turve – rakennettavuusselvitys"
    else:
        maapera_eval = "ℹ Tarkista kantavuus"

    story.append(sec(
        Paragraph("7b. Maaperä ja tulvavaara", st["h2"]),
        _table([
            ["Tarkasteltava tekijä", "Tulos", "Arviointi"],
            ["Maaperä", maaperalaaji or "Ei tiedossa", maapera_eval],
            ["Datalähde (maaperä)", maapera_src_str, ""],
            ["Tulvavaara", flood_str, flood_eval],
            ["Datalähde (tulvavaara)", flood_src, ""],
        ]),
        Spacer(1, 4*mm),
    ))

    # ── 8. Hankkeen toteutettavuusindeksi ────────────────────────────────────
    score = a.get("bess_score", 0)
    story.append(sec(
        Paragraph("8. Hankkeen toteutettavuusindeksi (0–100)", st["h2"]),
        _score_table(a, score),
    ))

    # ── 8b. Lupaprosessianalyysi ─────────────────────────────────────────────
    # Ei sec()/KeepTogether — pitkä AI-teksti voi jakautua sivunvaihdossa
    story.append(Paragraph("8b. NCE Energy — Lupaprosessianalyysi (tekoälyavusteinen)", st["h2"]))
    for _ai_item in _analysis_section(a, kuntanimi_gen):
        story.append(_ai_item)

    # ── 8c. Lupaprosessin aikajana ──────────────────────────────────────────
    story.append(sec(
        Paragraph("8c. Lupaprosessin aikajana", st["h2"]),
        _timeline_table(a, op_name, op_url, lupapiste_url, kuntanimi_gen, power_mw),
    ))

    # ── 9. Suositukset ───────────────────────────────────────────────────────
    recs = _recommendations(a, score, op_name=op_name, kuntanimi_gen=kuntanimi_gen, ely_center=ely_center)
    story.append(sec(
        Paragraph("9. Suositukset", st["h2"]),
        *[Paragraph(f"{i}. {rec}", st["body"]) for i, rec in enumerate(recs, 1)],
    ))

    # ── 10. Lakisääteiset vaatimukset ─────────────────────────────────────────
    story.append(sec(
        Paragraph("10. Lakisääteiset vaatimukset ja viranomaisprosessi", st["h2"]),
        _regulatory_table(
            bldg_m,
            gw_overlap=a.get("groundwater_overlap", False),
            heritage_overlap=h_overlap,
            natura_overlap=a.get("natura_overlap", False),
            road_protection_ok=road_ok,
            power_mw=power_mw,
            op_name=op_name,
            op_url=op_url,
            kuntanimi_gen=kuntanimi_gen,
            pelastuslaitos=pelastuslaitos,
            ely_center=ely_center,
            zoning_ok=a.get("zoning_ok", True),
            lupapiste_url=lupapiste_url,
        ),
    ))

    # ── 11. Seuraavat toimenpiteet ───────────────────────────────────────────
    _steps_rows = [
        [f"{op_name} liityntähakemus\n"
         f"Täytä liityntätarjouspyyntö: {op_url if op_url else 'Ota yhteyttä paikalliseen jakeluverkkoyhtiöön'}\n"
         "Arvioi liityntäkustannus (10 000–50 000 €)",
         f"Carbon Zero Finland / {op_name}", "KRIITTINEN"],
    ]
    if zoning_ok:
        _steps_rows.append([
            f"Suunnittelutarveratkaisu (STR)\nHae STR {kuntanimi_gen} kunnalta ennen rakennuslupaa – maaseuturakentaminen\n"
            f"Lupapiste: {lupapiste_url}",
            f"Carbon Zero Finland /\n{kuntanimi_gen} kunta", "KRIITTINEN",
        ])
    _steps_rows += [
        ["Rakennuslupa + naapurikuuleminen\nRakentamislupahakemus ja naapurien kuuleminen",
         f"Carbon Zero Finland /\n{kuntanimi_gen} kunta", "KORKEA"],
        ["Sähköturvallisuustarkastus (Tukes)\nBESS = sähkölaitos — käyttöönottotarkastus ennen verkkoliityntää "
         "(Sähköturvallisuuslaki 1135/2016)",
         "Sertifioitu tarkastuslaitos /\nTukes", "KORKEA"],
        ["Meluselvitys (YSL)\nAkustiset mittaukset, lähin rakennus <300 m – pakollinen liite",
         "Akustikkokonsultti", "KORKEA"],
        ["Pelastuslausunto + kemikaali-ilmoitus\nLION-suositus 12/2025; Li-ion = vaaralliset kemikaalit",
         pelastuslaitos, "KORKEA"],
        ["Ympäristöselvitykset\nPohjavesi, Natura tarvittaessa, YVA-tarveharkinta (ELY)",
         f"Ympäristökonsultti / {ely_center}", "KORKEA"],
        [f"Maanomistus ja määräalan lohkominen\nVuokra-/kauppaneuvottelut, {varattu_str} lohkominen MML",
         "Carbon Zero Finland / MML", "NORMAALI"],
    ]
    if h_overlap and not h_na:
        _steps_rows.append([
            "Museoviraston lausunto\nMuinaismuisto tai RKY havaittu — pyydä lausunto ennen lupahakemuksen jättämistä",
            "Museovirasto", "KORKEA",
        ])
    steps = [["Nro", "Toimenpide", "Vastuutaho", "Prioriteetti"]] + [
        [str(i), row[0], row[1], row[2]] for i, row in enumerate(_steps_rows, 1)
    ]
    story.append(sec(
        Paragraph("11. Seuraavat toimenpiteet", st["h2"]),
        _next_steps_table(steps),
    ))

    # ── Loppudisclaimer ───────────────────────────────────────────────────────
    story.append(sec(
        Spacer(1, 10*mm),
        HRFlowable(width="100%", thickness=0.5, color=C_GRAY, spaceAfter=5),
        Paragraph("VASTUUVAPAUSLAUSEKE", st["disc_label"]),
        Paragraph(
            "Tämä raportti on laadittu esiselvityskäyttöön ja perustuu julkisiin avoimiin tietoihin "
            f"(MML INSPIRE WFS, OSM Overpass, SYKE) raportin päivämääränä {now}. "
            "Raportti ei korvaa virallisia lupaselvityksiä eikä ole sitova. "
            "NCE Energy ei vastaa mahdollisten tietojen puutteellisuudesta tai muutoksista. "
            "Lopullinen rakentamislupahakemus edellyttää viranomaisten edellyttämiä lisäselvityksiä "
            f"kuten suunnittelutarveratkaisu, meluselvitys, paloturvallisuussuunnitelma ja "
            f"{op_name}-liityntähakemus.",
            st["disclaimer"]),
        Spacer(1, 3*mm),
        HRFlowable(width="100%", thickness=0.3, color=C_GRAY, spaceAfter=4),
        Paragraph(
            f"© NCE Energy · ncenergy.fi · info@ncenergy.fi · {now} · Luottamuksellinen",
            st["footer"]),
    ))

    # ── Sivunumero-footer (joka sivu) ─────────────────────────────────────────
    def _page_footer(canv, doc):
        canv.saveState()
        page_w, _ = A4
        margin = 2 * cm
        y_line = 1.45 * cm
        y_text = 0.9 * cm
        canv.setStrokeColor(C_GRAY)
        canv.setLineWidth(0.3)
        canv.line(margin, y_line, page_w - margin, y_line)
        canv.setFont("Helvetica", 6.5)
        canv.setFillColor(C_GRAY)
        canv.drawString(margin, y_text,
                        f"Akkuvarastohankkeen sijaintianalyysi  |  {kiinteistotunnus}  |  {kuntanimi}")
        canv.drawRightString(page_w - margin, y_text,
                             f"{now}  |  Luottamuksellinen  |  Sivu {doc.page}")
        canv.restoreState()

    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return buf.getvalue()


# ── Taulukkorakentajat ────────────────────────────────────────────────────────

def _table(data: list) -> Table:
    col_count = len(data[0]) if data else 2
    widths = {2: [6*cm, 11*cm], 3: [6*cm, 6*cm, 5.5*cm]}.get(col_count)
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), C_NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0), C_WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("GRID",        (0,0), (-1,-1), 0.5, C_GRAY),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("PADDING",     (0,0), (-1,-1), 5),
    ]))
    return t


def _score_table(a: dict, score: int) -> Table:
    def _pts(key):
        v = a.get(key)
        return "N/A" if v is None else str(v)

    def _man(flag_key: str) -> str:
        return " ★" if a.get(flag_key) else ""

    data = [
        ["Kriteeri", "Pisteet", "Max", "Osuus"],
        ["Verkkoliityntä (<1km=30p, 1-2km=20p, >2km=5p)",
         str(a.get("score_grid", "–")), "30", "30 %"],
        ["Pohjavesialue (ei=20p, luokka2/E=8p, luokka1=0p, N/A=poissa)",
         _pts("score_groundwater"), "20", "20 %"],
        ["Natura 2000 (ei=15p, kyllä=0p)",
         str(a.get("score_natura", "–")), "15", "15 %"],
        ["Muinaismuistot (ei=10p, kyllä=0p, N/A=poissa)",
         _pts("score_heritage"), "10", "10 %"],
        ["Asutus >300m (>300m=10p, 150-300m=5p, <150m=0p)",
         str(a.get("score_settlement", "–")), "10", "10 %"],
        [f"Kaavoitustilanne (ei kaavaa=10p, kaava=3p, N/A=poissa){_man('manual_kaavoitus')}",
         _pts("score_zoning"), "10", "10 %"],
        ["Tiesuoja-alue OK (ok=5p, ei ok=0p)",
         str(a.get("score_road", "–")), "5", " 5 %"],
        [f"Ei tulvavaaraa (ok=5p, kyllä=0p, N/A=poissa){_man('manual_tulvavaara')}",
         _pts("score_flood"), "5", " 5 %"],
        [f"Maaperä (kallio=5p, moreeni=4p, hiekka=3p, savi=1p, turve=0p, N/A=poissa){_man('manual_maapera')}",
         _pts("score_soil"), "5", " 5 %"],
        ["KOKONAISINDEKSI (normalisoitu saatavilla olevien kriteerien mukaan)",
         str(score), "100", ""],
    ]
    t = Table(data, colWidths=[10*cm, 1.8*cm, 1.4*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), C_NAVY),
        ("TEXTCOLOR",      (0,0), (-1,0), C_WHITE),
        ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 7.5),
        ("ALIGN",          (1,0), (-1,-1), "CENTER"),
        ("GRID",           (0,0), (-1,-1), 0.5, C_GRAY),
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.white, colors.HexColor("#f5f5f5")]),
        ("BACKGROUND",     (0,-1), (-1,-1), _score_color(score)),
        ("TEXTCOLOR",      (0,-1), (-1,-1), C_WHITE),
        ("FONTNAME",       (0,-1), (-1,-1), "Helvetica-Bold"),
        ("PADDING",        (0,0), (-1,-1), 4),
    ]))

    has_manual = any(a.get(k) for k in ("manual_kaavoitus", "manual_tulvavaara", "manual_maapera"))
    if has_manual:
        footnote = Table([[Paragraph(
            "★  Manuaalinen syöte — ei automaattinen rajapintadata",
            ParagraphStyle("fn", fontSize=7, textColor=C_GRAY, leading=10,
                           leftIndent=2),
        )]], colWidths=[15.7*cm])
        footnote.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ]))
        return KeepTogether([t, footnote])
    return t


def _alert(text: str) -> Table:
    """Punainen huomiolaatikko kriittisille viesteille."""
    C_ALERT_BG = colors.HexColor("#fff3f4")
    C_ALERT_BD = colors.HexColor("#e94560")
    t = Table([[Paragraph(f"⚠ {text}", ParagraphStyle(
        "alert", fontSize=8, textColor=colors.HexColor("#c0001a"), leading=12
    ))]], colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BOX",        (0,0), (-1,-1), 1.5, C_ALERT_BD),
        ("BACKGROUND", (0,0), (-1,-1), C_ALERT_BG),
        ("PADDING",    (0,0), (-1,-1), 7),
    ]))
    return t


def _warning(text: str) -> Table:
    """Oranssi VAROITUS-laatikko LION-suosituksia ja turvallisuushuomioita varten."""
    C_WARN_BG = colors.HexColor("#fff8e1")
    C_WARN_BD = colors.HexColor("#ff9800")
    t = Table([[Paragraph(
        f"<b>VAROITUS:</b> {text}",
        ParagraphStyle("warning", fontSize=8, textColor=colors.HexColor("#e65100"), leading=13)
    )]], colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BOX",        (0,0), (-1,-1), 2.0, C_WARN_BD),
        ("BACKGROUND", (0,0), (-1,-1), C_WARN_BG),
        ("PADDING",    (0,0), (-1,-1), 8),
    ]))
    return t


def _regulatory_table(
    bldg_m,
    gw_overlap: bool = False,
    heritage_overlap: bool = False,
    natura_overlap: bool = False,
    road_protection_ok: bool = True,
    power_mw: float = 1.0,
    op_name: str = "Verkonhaltija",
    op_url: str = "",
    kuntanimi_gen: str = "Pöytyän",
    pelastuslaitos: str = "Paikallinen pelastuslaitos",
    ely_center: str = "Paikallinen ELY-keskus",
    zoning_ok: bool = True,
    lupapiste_url: str = "https://www.lupapiste.fi/",
) -> Table:
    """
    Lakisääteiset vaatimukset kahdessa kategoriassa:
    A) Aina vaaditaan  B) Sijaintiriippuvaiset
    """
    liitynta_url = op_url if op_url else "Ota yhteyttä paikalliseen jakeluverkkoyhtiöön"

    C_PAKOLLINEN = colors.HexColor("#fdecea")
    C_WARN_ROW   = colors.HexColor("#fff3e0")

    # ── A) AINA VAADITAAN ─────────────────────────────────────────────────────
    always_rows = [
        ["1",
         "Rakentamislupa\nBESS-konttiasema tai tekninen laitos",
         f"{kuntanimi_gen} kunta\nRakentamislaki 751/2023",
         "Pakollinen"],
        ["2",
         f"{op_name} liityntähakemus\n{liitynta_url}",
         f"{op_name}\nSähkömarkkinalaki 588/2013",
         "Pakollinen ennen rakentamista"],
        ["3",
         "Fingrid SVJ2024 tekninen vaatimustenmukaisuus\nKoskee ≥0.8 kW laitoksia — verkko- ja järjestelmävaatimukset",
         "Fingrid / Energiavirasto\nEnergiavirasto 20.3.2025 (SVJ2024)",
         "Pakollinen ennen liityntää"],
        ["4",
         "Kemikaali-ilmoitus (Li-ion)\nLi-ion akustot = vaaralliset kemikaalit, ilmoitus pelastuslaitokselle",
         f"{pelastuslaitos}\nLaki vaarallisista kemikaaleista 390/2005, § 24",
         "Pakollinen"],
        ["5",
         "Pelastuslausunto (LION-suositus 12/2025)\nLi-ion akkuvaraston palotarkastus ja turvallisuuslausunto",
         f"{pelastuslaitos}\nPelL 379/2011, § 81a",
         "Pakollinen"],
        ["6",
         "Sähköturvallisuustarkastus (Tukes)\nBESS = sähkölaitos — käyttöönottotarkastus ennen verkkoliityntää",
         "Sertifioitu tarkastuslaitos / Tukes\nSähköturvallisuuslaki 1135/2016",
         "Pakollinen ennen käyttöönottoa"],
        ["7",
         "Naapurikuuleminen\nKuuleminen lupaprosessissa",
         f"{kuntanimi_gen} kunta\nRakentamislaki 751/2023",
         "Pakollinen"],
    ]

    # ── B) SIJAINTIRIIPPUVAISET ────────────────────────────────────────────────
    cond_rows = []
    row_num = 8

    # STR: vaaditaan vain maaseutualueella (ei asemakaavaa)
    if zoning_ok:
        cond_rows.append([
            str(row_num),
            f"Suunnittelutarveratkaisu (STR) — maaseutualue\n"
            f"STR ennen rakennuslupaa, hae: {lupapiste_url}",
            f"{kuntanimi_gen} kunta\nRakentamislaki 751/2023, § 46",
            "Pakollinen ennen rakentamista",
        ])
        row_num += 1

    # Ympäristölupa: pohjavesialue TAI asutus <300 m
    if gw_overlap or ((bldg_m or 999) < 300):
        reasons = []
        if gw_overlap:
            reasons.append("pohjavesialue")
        if (bldg_m or 999) < 300:
            reasons.append(f"asutus {bldg_m} m (<300 m)")
        cond_rows.append([
            str(row_num),
            f"Ympäristölupa\nEhto: {', '.join(reasons)}. "
            "Akkukemikaalien varastointi edellyttää ympäristölupaa.",
            f"{kuntanimi_gen} ympäristöviranomainen\nYSL 527/2014, § 27",
            "PAKOLLINEN ⚠",
        ])
        row_num += 1

    # Meluselvitys: asutus <300 m
    if (bldg_m or 999) < 300:
        cond_rows.append([
            str(row_num),
            f"Meluselvitys\nLähin rakennus {bldg_m} m (<300 m). "
            "Jäähdytysjärjestelmä 45–60 dB(A).",
            f"{kuntanimi_gen} ympäristöviranomainen\nYSL 527/2014, YSA",
            "PAKOLLINEN ⚠",
        ])
        row_num += 1

    # Muinaismuistoselvitys: heritage_overlap
    if heritage_overlap:
        cond_rows.append([
            str(row_num),
            "Muinaismuistoselvitys\nMuinaismuisto tai RKY havaittu — automaattinen rauhoitus, "
            "Museoviraston lupa pakollinen",
            "Museovirasto\nMuinaismuistolaki 295/1963, § 11",
            "PAKOLLINEN ⚠",
        ])
        row_num += 1

    # Tiesuoja-alue: valtatie <20 m
    if not road_protection_ok:
        cond_rows.append([
            str(row_num),
            "Tiesuoja-alue — rakentamiskielto\nValtatie <20 m — poikkeuslupa ELY-keskukselta",
            f"{ely_center}\nMaantielaki 503/2005, § 44",
            "PAKOLLINEN ⚠",
        ])
        row_num += 1

    # Natura-arviointi: jos Natura-alue havaittu
    if natura_overlap:
        cond_rows.append([
            str(row_num),
            "Natura-arviointi (LSL 66 §)\nNatura 2000 -alue havaittu — poikkeusmenettely tai arviointi pakollinen",
            f"SYKE / {ely_center}\nLuonnonsuojelulaki 9/2023, 66 §",
            "PAKOLLINEN ⚠",
        ])
        row_num += 1

    # YVA: teho >30 MW
    if power_mw > 30:
        cond_rows.append([
            str(row_num),
            f"Ympäristövaikutusten arviointi (YVA)\nTeho {power_mw} MW > 30 MW — YVA-menettely pakollinen",
            f"{ely_center}\nYVA-asetus 713/2006",
            "PAKOLLINEN ⚠",
        ])

    if not cond_rows:
        cond_rows.append([
            "–",
            "Ei sijaintiriippuvaisia vaatimuksia tällä sijainnilla\n"
            "Pohjavesi: ei  ·  Asutus: >300 m  ·  Muinaismuistot: ei  ·  "
            "Tiesuoja: ok  ·  Teho: ≤30 MW",
            "–",
            "✓ Ei sovellu",
        ])

    # Row indices for section headers
    sec_a_idx = 1
    sec_b_idx = 2 + len(always_rows)

    data = [
        ["Nro", "Vaatimus", "Viranomainen / Säädös", "Tila"],
        ["", "A) AINA VAADITAAN — kaikki BESS-hankkeet", "", ""],
        *always_rows,
        ["", "B) SIJAINTIRIIPPUVAISET — näytetään vain ehdon täyttyessä", "", ""],
        *cond_rows,
    ]

    t = Table(data, colWidths=[0.7*cm, 7.0*cm, 5.5*cm, 3.5*cm])
    style = [
        # Column header
        ("BACKGROUND",  (0, 0),         (-1, 0),         C_NAVY),
        ("TEXTCOLOR",   (0, 0),         (-1, 0),         C_WHITE),
        ("FONTNAME",    (0, 0),         (-1, 0),         "Helvetica-Bold"),
        # Section A header row
        ("BACKGROUND",  (0, sec_a_idx), (-1, sec_a_idx), C_NAVY),
        ("TEXTCOLOR",   (0, sec_a_idx), (-1, sec_a_idx), colors.HexColor("#f4a261")),
        ("FONTNAME",    (0, sec_a_idx), (-1, sec_a_idx), "Helvetica-Bold"),
        ("SPAN",        (0, sec_a_idx), (-1, sec_a_idx)),
        # Section B header row
        ("BACKGROUND",  (0, sec_b_idx), (-1, sec_b_idx), colors.HexColor("#0f3460")),
        ("TEXTCOLOR",   (0, sec_b_idx), (-1, sec_b_idx), colors.HexColor("#64b5f6")),
        ("FONTNAME",    (0, sec_b_idx), (-1, sec_b_idx), "Helvetica-Bold"),
        ("SPAN",        (0, sec_b_idx), (-1, sec_b_idx)),
        # Global
        ("FONTSIZE",    (0, 0),         (-1, -1),        8),
        ("GRID",        (0, 0),         (-1, -1),        0.5, C_GRAY),
        ("VALIGN",      (0, 0),         (-1, -1),        "TOP"),
        ("PADDING",     (0, 0),         (-1, -1),        5),
        ("ALIGN",       (0, 0),         (0, -1),         "CENTER"),
    ]

    # Color always rows
    for i, row in enumerate(always_rows):
        ri = sec_a_idx + 1 + i
        tila = row[-1]
        if "PAKOLLINEN ⚠" in tila:
            style += [("BACKGROUND", (0,ri), (-1,ri), C_PAKOLLINEN),
                      ("FONTNAME",   (0,ri), (-1,ri), "Helvetica-Bold")]
        elif "ennen" in tila:
            style.append(("BACKGROUND", (0,ri), (-1,ri), C_WARN_ROW))
        else:
            style.append(("BACKGROUND", (0,ri), (-1,ri), colors.HexColor("#f5f5f5")))

    # Color conditional rows
    for i, row in enumerate(cond_rows):
        ri = sec_b_idx + 1 + i
        tila = row[-1]
        if "PAKOLLINEN ⚠" in tila:
            style += [("BACKGROUND", (0,ri), (-1,ri), C_PAKOLLINEN),
                      ("FONTNAME",   (0,ri), (-1,ri), "Helvetica-Bold")]
        else:
            style.append(("BACKGROUND", (0,ri), (-1,ri), colors.HexColor("#f0fff4")))

    t.setStyle(TableStyle(style))
    return t


def _build_analysis_narrative(a: dict, kuntanimi_gen: str) -> list[tuple[str, str]]:
    """
    Rakentaa datapohjaisen lupaprosessianalyysin konsulttiraporttityylisenä tekstinä.
    Palauttaa listan (teksti, tyyppi)-tupleista, tyyppi = "heading" | "body" | "disclaimer".
    """
    score          = a.get("bess_score", 0)
    grid_m         = a.get("nearest_grid_m")
    gw_overlap     = a.get("groundwater_overlap", False)
    gw_class       = a.get("groundwater_class", "")
    zoning_ok      = a.get("zoning_ok", True)
    natura_overlap = a.get("natura_overlap", False)

    is_gw_class1 = gw_class in ("1", "1E")

    if score >= 70:
        score_label = "Korkea"
    elif score >= 50:
        score_label = "Kohtalainen"
    else:
        score_label = "Matala"

    # Verkkoetäisyys suomenkielisellä desimaalipistekäytännöllä
    if grid_m is not None:
        if grid_m >= 1000:
            km_val = grid_m / 1000
            # 1 des. riittää jos < 10 km
            grid_str = f"n. {km_val:.1f}".replace(".", ",") + " km"
            grid_long = f"Pitkä verkkoliityntämatka ({grid_str}) nostaa liityntäkustannuksia merkittävästi."
        else:
            grid_str  = f"n. {grid_m} m"
            grid_long = f"Verkkoliityntämatka ({grid_str}) on kohtuullinen." if grid_m > 500 \
                        else f"Lyhyt verkkoliityntämatka ({grid_str}) on erinomainen."
    else:
        grid_str  = ""
        grid_long = ""

    items: list[tuple[str, str]] = []

    # ── Kriittiset pullonkaulat ──────────────────────────────────────────────
    items.append(("Kriittiset pullonkaulat", "heading"))

    risk_parts = []
    if gw_overlap and is_gw_class1:
        risk_parts.append(
            f"Pohjavesialue (Luokka {gw_class}) on suurin riskitekijä. "
            "BESS-laitoksen sijoittaminen tärkeälle pohjavesialueelle edellyttää "
            "poikkeuksellisen kattavia ympäristöselvityksiä sekä erityisiä suojarakenteita."
        )
    elif gw_overlap:
        risk_parts.append(
            "Pohjavesialue edellyttää ympäristöselvityksiä akkukemikaalien varastoinnin osalta."
        )
    if natura_overlap:
        risk_parts.append(
            "Natura 2000 -alue voi edellyttää luonnonsuojeluarviointia (LSL 66 §)."
        )
    if grid_long:
        risk_parts.append(grid_long)

    has_risks = bool(risk_parts)
    risk_parts.append(
        f"{score_label} toteutettavuusindeksi ({score}/100) heijastaa näitä haasteita."
        if has_risks else
        f"{score_label} toteutettavuusindeksi ({score}/100); analyysi ei tunnista kriittisiä esteitä."
    )
    items.append((" ".join(risk_parts), "body"))

    # ── Suositus ─────────────────────────────────────────────────────────────
    items.append(("Suositus", "heading"))
    if score < 40 or (gw_overlap and is_gw_class1):
        rec = "Harkitse vahvasti vaihtoehtoista sijaintia."
    elif score < 60:
        rec = (
            "Sijainnilla on potentiaalia, mutta tunnistetut rajoitukset edellyttävät "
            "lisäselvityksiä ennen sitoutumista hankkeeseen."
        )
    else:
        rec = "Sijainti vaikuttaa toteuttamiskelpoiselta. Eteneminen esiselvitysvaiheeseen on perusteltua."
    items.append((rec, "body"))

    # ── Vaiheistus ───────────────────────────────────────────────────────────
    items.append(("Jos hanketta kuitenkin edetään:", "heading"))

    # Vaihe 1
    phase1 = (
        f"Vaihe 1 – Ennakkoselvitykset (1–2 kk): "
        f"Ennakkoneuvottelu {kuntanimi_gen} kunnan kaavoituksen ja ympäristötoimen kanssa "
        f"ennen virallisia hakemuksia."
    )
    if gw_overlap:
        phase1 += " Pohjavesiasiantuntijan riskiarvio on vahvasti suositeltava."
    items.append((phase1, "body"))

    # Vaihe 2
    phase2_months = "6–12" if is_gw_class1 else "3–6"
    phase2_parts = []
    if zoning_ok:
        phase2_parts.append("Suunnittelutarveratkaisu (STR)")
    if gw_overlap:
        phase2_parts.append("ympäristölupa")
    phase2_parts.append("verkkoliityntähakemus")
    phase2_list = ", ".join(phase2_parts) + " rinnakkain."
    items.append((
        f"Vaihe 2 – Lupahakemukset ({phase2_months} kk): {phase2_list}",
        "body",
    ))

    # Kokonaisaikataulu
    total_months = "12–24" if (is_gw_class1 or natura_overlap) else ("6–18" if gw_overlap else "4–12")
    pohj_ref = "pohjavesikysymykseen" if gw_overlap else "hankkeeseen"
    items.append((
        f"Arvioitu kokonaisaikataulu: {total_months} kuukautta "
        f"(riippuen {kuntanimi_gen} kunnan kannasta {pohj_ref}).",
        "body",
    ))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    items.append((
        "Tämä on yleinen analyysi perustuen saatavilla oleviin tietoihin. "
        "Lopullinen lupastrategia tulee aina laatia yhdessä kokeneen lupa-asiantuntijan kanssa.",
        "disclaimer",
    ))

    return [(_postprocess_text(t), kind) for t, kind in items]


def _analysis_section(a: dict, kuntanimi_gen: str) -> list:
    """Renderöi datapohjaisen lupaprosessianalyysin PDF-elementeiksi."""
    C_AI_BG = colors.HexColor("#f0f4ff")
    C_AI_BD = colors.HexColor("#3a7bd5")

    st_head = ParagraphStyle("an_h",    fontSize=9,   fontName="Helvetica-Bold",
                             textColor=C_AI_BD, spaceAfter=4)
    st_sec  = ParagraphStyle("an_sec",  fontSize=8.5, fontName="Helvetica-Bold",
                             spaceBefore=6, spaceAfter=2)
    st_body = ParagraphStyle("an_body", fontSize=8.5, leading=13, spaceAfter=4)
    st_disc = ParagraphStyle("an_disc", fontSize=7.5, fontName="Helvetica-BoldOblique",
                             textColor=colors.HexColor("#333333"), leading=11, spaceBefore=6)

    narrative = _build_analysis_narrative(a, kuntanimi_gen)

    rows = [[
        Paragraph("NCE Energy — Lupaprosessianalyysi (tekoälyavusteinen)", st_head),
        "",
    ]]
    for text, kind in narrative:
        if kind == "heading":
            rows.append([Paragraph(text, st_sec), ""])
        elif kind == "disclaimer":
            rows.append([Paragraph(text, st_disc), ""])
        else:
            rows.append([Paragraph(text, st_body), ""])

    box = Table(rows, colWidths=[15.5*cm, 1.2*cm], splitByRow=1)
    box.setStyle(TableStyle([
        ("SPAN",       (0, 0),  (-1, 0)),
        ("BOX",        (0, 0),  (-1, -1), 1.5, C_AI_BD),
        ("BACKGROUND", (0, 0),  (-1, -1), C_AI_BG),
        ("PADDING",    (0, 0),  (-1, -1), 8),
        ("TOPPADDING", (0, 0),  (-1, 0),  10),
    ]))
    return [box, Spacer(1, 6*mm)]


def _timeline_table(
    a: dict,
    op_name: str,
    op_url: str,
    lupapiste_url: str,
    kuntanimi_gen: str,
    power_mw: float,
) -> Table:
    """Lupaprosessin aikajana — näyttää vain relevantit askeleet."""
    gw_overlap     = a.get("groundwater_overlap", False)
    gw_class       = a.get("groundwater_class", "")
    bldg_m         = a.get("nearest_building_m")
    natura_overlap = a.get("natura_overlap", False)
    heritage_overlap = a.get("heritage_overlap", False)
    road_ok        = a.get("road_protection_ok", True)
    zoning_ok      = a.get("zoning_ok", True)
    is_gw_class1   = gw_class in ("1", "1E")

    C_IMM   = colors.HexColor("#e8f5e9")  # vihreä — välitön
    C_W2    = colors.HexColor("#fff8e1")  # keltainen — viikot 2-4
    C_M1    = colors.HexColor("#e3f2fd")  # sininen — kuukausi 1-2
    C_M3    = colors.HexColor("#fce4ec")  # punainen — kuukausi 3+
    C_HEAD  = colors.HexColor("#0f3460")
    C_WARN  = colors.HexColor("#fdecea")

    def row(phase, step, law, note=""):
        return [phase, step, law, note]

    rows = [
        ["Vaihe", "Toimenpide", "Laki / Viite", "Huomio"],
        # VÄLITÖN
        row("VÄLITÖN\npäivä 1",
            f"1. Sijaintianalyysi valmis ✓\n2. Lähetä {op_name} liityntähakemus\n   {op_url if op_url else 'Ota yhteys paikalliseen verkonhaltijaan'}",
            "Sähkömarkkinalaki 588/2013",
            "KRIITTINEN"),
    ]
    if zoning_ok:
        rows.append(row(
            "VÄLITÖN\npäivä 1",
            f"3. Jätä STR Lupapiste.fi:ssä\n   {lupapiste_url}",
            "Rakentamislaki 751/2023, § 46",
            "Maaseutu-alue",
        ))

    # VIIKKO 2-4
    rows.append(row(
        "VIIKKO 2–4",
        f"Liityntätarjous saapuu ({op_name})\n2–4 viikkoa hakemuksesta",
        "Sähkömarkkinalaki 588/2013",
        "",
    ))
    if bldg_m is not None and bldg_m < 300:
        rows.append(row(
            "VIIKKO 2–4",
            f"Tilaa meluselvitys (akustikkokonsultti)\nLähin rakennus {bldg_m} m — pakollinen liite",
            "YSL 527/2014",
            "PAKOLLINEN ⚠",
        ))

    # KUUKAUSI 1-2
    if zoning_ok:
        rows.append(row(
            "KUUKAUSI 1–2",
            "STR/Sijoittamislupa päätös\n1–2 kuukautta hakemuksesta",
            "Rakentamislaki 751/2023",
            "",
        ))
    if gw_overlap:
        rows.append(row(
            "KUUKAUSI 1–2" if not is_gw_class1 else "KUUKAUSI 3–6",
            "Ympäristölupa käsittelyssä (pohjavesialue)\n3–6 kuukautta — kriittinen polku",
            "YSL 527/2014, § 27",
            "KRIITTINEN POLKU ⚠" if is_gw_class1 else "PAKOLLINEN ⚠",
        ))
    if natura_overlap:
        rows.append(row(
            "KUUKAUSI 1–3",
            "Natura-arviointi (SYKE / ELY)\nVoi viivyttää 3–12 kk",
            "Luonnonsuojelulaki 9/2023, 66 §",
            "KRIITTINEN POLKU ⚠",
        ))
    if heritage_overlap:
        rows.append(row(
            "KUUKAUSI 1–2",
            "Museoviraston lausunto\nMuinaismuisto tai RKY havaittu",
            "Muinaismuistolaki 295/1963, § 11",
            "PAKOLLINEN ⚠",
        ))

    # KUUKAUSI 3
    rows.append(row(
        "KUUKAUSI 3",
        "Rakentamislupa haettavissa\n" + ("(STR:n jälkeen)" if zoning_ok else "(muiden lupien jälkeen)"),
        "Rakentamislaki 751/2023",
        f"Lupapiste: {lupapiste_url}",
    ))
    rows.append(row(
        "KUUKAUSI 3",
        "Pelastuslausunto + kemikaali-ilmoitus\nLi-ion LION-suositus 12/2025",
        "PelL 379/2011  ·  Laki 390/2005",
        "",
    ))
    rows.append(row(
        "KUUKAUSI 3",
        "Sähköturvallisuustarkastus (Tukes)\nKäyttöönottotarkastus ennen liityntää",
        "Sähköturvallisuuslaki 1135/2016",
        "",
    ))
    rows.append(row(
        "KUUKAUSI 3",
        "Fingrid SVJ2024 vaatimustenmukaisuus",
        "Energiavirasto 20.3.2025",
        "",
    ))
    if power_mw > 30:
        rows.append(row(
            "KUUKAUSI 2–4",
            f"YVA-menettely pakollinen ({power_mw} MW > 30 MW)",
            "YVA-asetus 713/2006",
            "KRIITTINEN POLKU ⚠",
        ))

    # KUUKAUSI 3-4
    aloitus_kk = "KUUKAUSI 6–8" if is_gw_class1 or natura_overlap else "KUUKAUSI 3–4"
    aloitus_note = "POIKKEUS: luokka 1 pohjavesi tai Natura → kriittinen polku" if (is_gw_class1 or natura_overlap) else ""
    rows.append(row(
        aloitus_kk,
        "Rakentamisaloitus mahdollinen\nKaikki luvat myönnetty",
        "–",
        aloitus_note,
    ))

    t = Table(rows, colWidths=[2.5*cm, 7.5*cm, 4.0*cm, 2.7*cm])
    style = [
        ("BACKGROUND", (0,0), (-1,0), C_HEAD),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7.5),
        ("GRID",       (0,0), (-1,-1), 0.5, C_GRAY),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("PADDING",    (0,0), (-1,-1), 4),
    ]
    phase_colors = {
        "VÄLITÖN": C_IMM,
        "VIIKKO":  C_W2,
        "KUUKAUSI 1": C_M1,
        "KUUKAUSI 2": C_M1,
        "KUUKAUSI 3": C_M3,
        "KUUKAUSI 6": C_WARN,
    }
    for ri, row_data in enumerate(rows[1:], start=1):
        phase = row_data[0] if row_data else ""
        note  = row_data[3] if len(row_data) > 3 else ""
        bg = colors.white
        for k, c in phase_colors.items():
            if phase.startswith(k):
                bg = c; break
        if "KRIITTINEN" in note or "PAKOLLINEN" in note:
            bg = C_WARN
        style.append(("BACKGROUND", (0, ri), (-1, ri), bg))
        if "KRIITTINEN" in note:
            style.append(("FONTNAME", (0, ri), (-1, ri), "Helvetica-Bold"))

    t.setStyle(TableStyle(style))
    return t


def _next_steps_table(data: list) -> Table:
    C_KRIT  = colors.HexColor("#fdecea")
    C_HIGH  = colors.HexColor("#fff8e1")
    C_NORM  = colors.white

    priority_colors = {"KRIITTINEN": C_KRIT, "KORKEA": C_HIGH, "NORMAALI": C_NORM}

    t = Table(data, colWidths=[0.8*cm, 8.5*cm, 4.5*cm, 2.5*cm])
    style = [
        ("BACKGROUND",  (0,0), (-1,0), C_NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0), C_WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("GRID",        (0,0), (-1,-1), 0.5, C_GRAY),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("PADDING",     (0,0), (-1,-1), 5),
        ("ALIGN",       (0,0), (0,-1), "CENTER"),
    ]
    for row_i, row in enumerate(data[1:], start=1):
        prio = row[-1] if row else ""
        bg = priority_colors.get(prio, C_NORM)
        style.append(("BACKGROUND", (0, row_i), (-1, row_i), bg))
        if prio == "KRIITTINEN":
            style.append(("FONTNAME", (0, row_i), (-1, row_i), "Helvetica-Bold"))

    t.setStyle(TableStyle(style))
    return t


def _recommendations(
    a: dict,
    score: int,
    op_name: str = "verkonhaltijalta",
    kuntanimi_gen: str = "Pöytyän",
    ely_center: str = "Paikallinen ELY-keskus",
) -> list[str]:
    recs = []
    grid_m = a.get("nearest_grid_m")
    if grid_m is not None:
        if grid_m < 500:
            recs.append(f"Sähköliityntä: Lähin johto {grid_m} m – erinomainen sijainti, edullinen liityntä ({op_name}).")
        elif grid_m < 2_000:
            recs.append(f"Sähköliityntä: Lähin johto {grid_m} m – toteutettavissa, arvioi liityntäkustannus {op_name}:lta (10 000–50 000 €).")
        else:
            recs.append(f"Sähköliityntä: Lähin johto {grid_m} m – pyydä liityntätarjous ({op_name}), OSM-data voi olla puutteellinen.")
    if a.get("natura_overlap"):
        recs.append("Natura 2000: Luonnonsuojelulain 65 § mukainen Natura-arviointi on todennäköisesti tarpeen.")
    if a.get("groundwater_overlap"):
        recs.append("Pohjavesi: Alue sijaitsee pohjavesialueella – tarkista ympäristölupavaatimukset ja akkukemikaalien varastointiehdot.")
    if a.get("heritage_overlap") and not a.get("heritage_unavailable"):
        recs.append("Muinaismuistot: Alueella havaittu muinaismuisto tai RKY-kohde – pyydä Museoviraston lausunto ennen hakemuksen jättämistä (Muinaismuistolaki 295/1963).")
    if not a.get("road_protection_ok", True):
        road_m = a.get("nearest_road_m")
        recs.append(f"Tiesuoja-alue: Kiinteistö on {road_m} m päässä valtatien reunasta (<20 m) – neuvottele ELY-keskuksen kanssa tai sijoita laitos kauemmas (Maantielaki 503/2005 § 44).")
    if a.get("zoning_ok"):
        recs.append(f"Suunnittelutarveratkaisu (STR): Maaseutualue – hae STR {kuntanimi_gen} kunnalta ennen rakennuslupaa (Rakentamislaki 751/2023, § 46).")
    else:
        recs.append(f"Kaavoitus: Asemakaava-alue – tee kaavoitusselvitys {kuntanimi_gen} kunnan kanssa ennen hankkeen eteenpäinviemistä.")
    bldg_m = a.get("nearest_building_m")
    if bldg_m is not None and bldg_m < 300:
        recs.append(f"Asutus: Lähin rakennus {bldg_m} m päässä – melun ja turvallisuusvyöhykkeen vaatimukset on selvitettävä.")
    recs.append("Sähköturvallisuus: BESS-laitos on sähkölaitos – käyttöönottotarkastus Tukesin sertifioimalla tarkastuslaitoksella pakollinen ennen verkkoliityntää (Sähköturvallisuuslaki 1135/2016).")
    recs.append(f"YVA: Yli 30 MW akkuvarastot voivat edellyttää ympäristövaikutusten arviointia (YVA-asetus 713/2006). Ota yhteyttä {ely_center}een.")
    recs.append("Maanomistus: Varmista kiinteistön omistus- tai vuokrausmahdollisuus ja selvitä rasitteet.")
    if score >= 70:
        recs.append(f"Soveltuvuusindeksi {score}/100 – sijainti vaikuttaa lupaavalta. Eteneminen hankesuunnitteluvaiheeseen on perusteltua.")
    elif score < 40:
        recs.append(f"Soveltuvuusindeksi {score}/100 – useita rajoittavia tekijöitä. Harkitse vaihtoehtoisia sijainteja.")
    return recs
