"""
Indeksoi Suomen ympäristölainsäädäntö ChromaDB:hen.

Finlex on täysin JS-renderöity (Next.js) eikä sovellu yksinkertaiseen
HTTP-hakuun. Tämä skripti käyttää kahta lähdettä:

1. INLINE-tekstit  — kuratoitu kooste YSL 527/2014- ja YVA-laki 252/2017
   -avainkohdista, kirjoitettu suoraan tähän skriptiin.
2. WEB-haku       — lvv.fi (Lupa- ja valvontavirasto) ympäristösivut.

Käyttö:
    python3 permit_ai/ingest_fi_env.py

Ajaa myös:
    python3 permit_ai/ingest_web.py --country FI   (lvv.fi osio)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

HERE   = Path(__file__).parent
DB_DIR = HERE / "embeddings"

EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION  = "permit_docs_v2"
CHUNK_CHARS = 1500
OVERLAP     = 200
BATCH       = 64

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NCE-RAG-Ingest/1.0; "
        "+https://github.com/jeremakela-web/nce-energy-permit-ai)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fi-FI,fi;q=0.9,en;q=0.5",
}

# ─────────────────────────────────────────────────────────────────────────────
# INLINE-dokumentit — kuratoitu kooste suomalaisesta ympäristölainsäädännöstä
# ─────────────────────────────────────────────────────────────────────────────

_FI_ENV_LAW_DOCS: list[tuple[str, str, str]] = [
    # (doc_id, source_label, text)
    (
        "ysl_527_2014_luvantarve",
        "YSL 527/2014 — Ympäristölupa: luvantarve",
        """\
YMPÄRISTÖNSUOJELULAKI 527/2014 — LUVANTARVE JA SOVELTAMISALA

Ympäristönsuojelulaissa (YSL 527/2014) säädetään ympäristön pilaantumisen
ehkäisemisestä ja toiminnanharjoittajan velvollisuuksista.

§ 27 — YMPÄRISTÖLUVANVARAISUUS
Ympäristölupa tarvitaan seuraaviin toimintoihin:
- Teollisuuslaitosten ja suurten tuotantolaitosten toiminta (yli tiettyjen
  kapasiteettirajojen)
- Energiantuotantolaitokset: polttoaineteholtaan yli 50 MW:n laitos
  (ISO 14064 -standardin mukainen laskenta)
- Polttoaineiden varastointi: nestemäiset polttoaineet yli 10 000 l,
  kiinteät polttoaineet yli 50 000 tn
- Kemikaaliteollisuuden laitokset (Seveso-direktiivin kynnykset)
- Kaivostoiminta: louhinta ja malmin rikastus
- Jätteen käsittely: kaatopaikat, poltto, kompostointi yli tiettyjen rajojen
- Kotieläintalous: lypsylehmät ≥ 150 paikkaa, lihasiat ≥ 2 000 paikkaa,
  siipikarja ≥ 40 000 paikkaa (liite 1, taulukko 2)
- Akkuenergiavarastot (BESS): merkittävät kemikaalit, pohjavesialue tai
  Natura-alue lähellä voi edellyttää ympäristölupaa tai ilmoitusmenettelyä
- Toiminta pohjavesialueella tai vesistön läheisyydessä, jos riski
  pilaantumiseen
- Pilaantunut maaperä — puhdistaminen

§ 28 — KUNTIEN YMPÄRISTÖLUPA
Kunta (ympäristönsuojeluviranomainen) myöntää luvan pienemmille laitoksille,
jotka on lueteltu liitteessä 2.

§ 29 — ILMOITUSMENETTELY (REKISTERÖINTI)
Eräät toiminnot eivät tarvitse varsinaista lupaa vaan pelkän
rekisteröinti-ilmoituksen kunnalle (YSL 116 §):
- Pienemmät polttoaineen jakeluasemat
- Pienet eläinsuojat
- Asfalttiasemat
- Nestemäisten polttoaineiden ja kemikaalien jakeluasemat

§ 39 — LUPAHAKEMUKSEN SISÄLTÖ
Hakemuksessa on esitettävä:
1. Hakijan nimi ja yhteystiedot
2. Toiminnan sijainti ja kiinteistötunnus
3. Toiminnan yleiskuvaus (tuotantoprosessi, raaka-aineet, tuotteet)
4. Arvio ympäristövaikutuksista: päästöt ilmaan, veteen, maaperään
5. Jätehuoltosuunnitelma (jätejakeet, määrät, käsittely)
6. Poikkeustilanteiden toimintaohje (häiriöt, onnettomuudet)
7. Meluselvitys, jos toiminta aiheuttaa melua asuinalueelle
8. Vesistövaikutukset ja pohjavesiriskiarvio
9. Luontovaikutukset (Natura, luonnonsuojelulaki 9/2023)
10. Parhaan käyttökelpoisen tekniikan (BAT) kuvaus

§ 47 — LUPAVIRANOMAINEN
- Lupa- ja valvontavirasto (LVV, ent. AVI): suuret laitokset, liitteen 1
  toiminnot, valtion ympäristölupa
- Kunta: pienemmät toiminnot (liite 2)
- ELY-keskus: neuvoo ja lausuu, ei myönnä ympäristölupia

§ 86 — LUPAMÄÄRÄYKSET
Luvassa on annettava tarpeelliset määräykset:
- Päästöraja-arvot (ilma, vesi, maaperä)
- Jätteiden käsittely
- Toiminnan tarkkailu ja raportointi
- Toiminnan lopettaminen ja jälkihoito

§ 118 — MELUILMOITUS
Rakentaminen tai muu tilapäinen toiminta, josta aiheutuu erityistä meluhaittaa,
edellyttää ilmoituksen kunnalle. Koske myös tilapäisiä energiaprojekteja.

LIITE 2 — TOIMINNOT JOTKA EIVÄT VAADI YMPÄRISTÖLUPAA (YSL 527/2014)
Direktiivilaitokset (ns. IE-laitokset), joita koskee EU:n
teollisuuspäästödirektiivi 2010/75/EU, ovat aina ympäristölupavelvoitteen
piirissä riippumatta koosta.
""",
    ),
    (
        "ysl_527_2014_hakeminen",
        "YSL 527/2014 — Ympäristöluvan hakeminen: prosessi ja liitteet",
        """\
YMPÄRISTÖLUVAN HAKEMINEN — PROSESSI (YSL 527/2014)

HAKEMUKSEN JÄTTÄMINEN
- Hakemus jätetään toimivaltaiselle viranomaiselle:
  Lupa- ja valvontavirasto (LVV) tai kunnan ympäristöviranomainen
- Sähköinen asiointi: lvv.fi — Sähköinen asiointijärjestelmä
- Kirjallinen hakemus: Luvan hakeminen -lomake (Luova-lomake)
- Käsittelyaika: yleensä 6–18 kuukautta toiminnon kompleksisuuden mukaan

HAKEMUKSEN PAKOLLISET LIITTEET
1. Sijaintikartta (1:20 000 tai laajempi)
2. Asema- tai tonttipiirrustus (1:500 tai tarkempi)
3. Prosessikaavio tai toimintakuvaus
4. Päästöluettelo: ilma, vesi, maaperä (mitatut tai lasketut arvot)
5. Jätteen käsittelysuunnitelma
6. Pohjavesialueen kartta (SYKE:n Vesistörekisteri)
7. Natura 2000 -arviointi tarvittaessa (luonnonsuojelulaki 9/2023 §30)
8. Meluselvitys (Sosiaali- ja terveysministeriön ohjeet)
9. Naapuritilojen omistajatiedot (kuuleminen)
10. Kaupparekisteriote tai muu oikeushenkilötodistus
11. YVA-selostus jos YVA on tehty (YVA-laki 252/2017)

KUULEMINEN JA LAUSUNNOT
- Naapurit ja asianosaiset kuullaan (hallintolaki 434/2003)
- Lausunto pyritään hakemaan: ELY-keskus, kunta, SYKE, Traficom
- Hakija voi pyytää ennakkoneuvottelua (§ 44)

YMPÄRISTÖLUPAPÄÄTÖS
- Sisältää lupamääräykset päästöistä, tarkkailusta ja raportoinnista
- Muutoksenhaku: Vaasan hallinto-oikeus (ympäristöasiat)
- Luvassa on usein myös vakuusvaatimus (§ 59)
- Toimintaa ei saa aloittaa ennen luvan lainvoimaisuutta, jollei
  aloituslupaa (§ 199) haeta erikseen

NEUVONTA
- Lupa- ja valvontavirasto (LVV): https://lvv.fi/ymparisto
- SYKE ympäristölupa-asiat: ympäristö.fi
- ELY-keskus: neuvoo YVA-menettelyissä
""",
    ),
    (
        "yva_laki_252_2017",
        "YVA-laki 252/2017 — Ympäristövaikutusten arviointi",
        """\
LAKI YMPÄRISTÖVAIKUTUSTEN ARVIOINTIMENETTELYSTÄ 252/2017 (YVA-LAKI)

YVA-MENETTELY — MILLOIN PAKOLLINEN?
YVA-menettely (ympäristövaikutusten arviointi) on pakollinen liitteen 1
hankkeille. Energiasektoria koskevia kynnysarvoja:

LIITE 1 — AINA YVA-VELVOLLISET HANKKEET (energia):
- Ydinvoimala tai muu ydinreaktorit (SMR, kaikki koot)
- Tuulivoimala: vähintään 10 voimalaa TAI kokonaisteho ≥ 45 MW
- Aurinkovoimala: pinta-ala ≥ 350 ha TAI teho ≥ 200 MW (MWp)
- Voimalaitokset > 300 MW (terminen teho), pl. ydinvoimalat
- Vesivoimalat: yli 20 MW
- Akkuenergiavarastot (BESS): ei suoraan YVA-velvoitetta,
  mutta liittyviä hankkeita (voimalinjat > 220 kV, suuret
  vesihankkeet) voidaan arvioida

LIITE 2 — TAPAUSKOHTAINEN YVA-HARKINTA (ELY-keskus päättää):
- Tuulivoimala: 3–9 voimalaa tai 10–44 MW
- Voimalinnat ≥ 110 kV, pituus yli 15 km
- Teollisuuslaitokset liitteen 1 alapuolella
- Kaivoshankkeet pienemmät kuin liitteen 1 kynnykset
- Satamat, lentokentät pienemmät kuin liitteen 1 kynnykset

YVA-MENETTELYN VAIHEET
1. YVA-ohjelma (scoping)
   - Hankevastaava tekee YVA-ohjelman
   - Yhteysviranomainen (ELY-keskus tai Lupa- ja valvontavirasto/LVV)
     kuuluttaa ja pyytää lausunnot
2. YVA-selostus (assessment report)
   - Hankevastaava teettää YVA-selostuksen
   - Yhteysviranomainen arvioi ja antaa perustellun päätelmän
3. Perusteltu päätelmä (reasoned conclusion)
   - ELY-keskuksen tai LVV:n antama kirjallinen arvio
   - Otettava huomioon lupakäsittelyssä
4. YVA-selostus liitetään ympäristölupa-/rakentamislupahakemukseen

YHTEYSVIRANOMAISET
- ELY-keskus: useimmat hanketyypit
- Lupa- ja valvontavirasto (LVV): eräät suuret hankkeet (mm. ydinvoima)

YHTEYS YMPÄRISTÖLUPAAN
- Ympäristölupaa ei myönnetä ennen kuin YVA-menettely on päättynyt
- Perusteltu päätelmä on voimassa 5 vuotta
- YVA-selostus on keskeinen liite ympäristölupahakemuksessa

LAKIVIITTEET
- YVA-laki 252/2017 (laki ympäristövaikutusten arviointimenettelystä)
- EU: Direktiivi 2014/52/EU (YVA-direktiivin muutos)
- Luonnonsuojelulaki 9/2023 § 30 (Natura-arviointi)
""",
    ),
    (
        "ysl_527_2014_bess_energia",
        "YSL 527/2014 — Akkuvarasto (BESS) ja energiantuotanto: ympäristölupatarve",
        """\
YMPÄRISTÖLUPATARVE — AKKUENERGIAVARASTOT (BESS) JA ENERGIANTUOTANTO

BESS — AKKUENERGIAVARASTOT
Akkuenergiavarastojen (BESS) ympäristölupatarve riippuu sijainnista,
kemikaaleista ja kapasiteetista.

YMPÄRISTÖLUPA TAI ILMOITUSMENETTELY TARVITAAN KUN:
- Akusto sijoittuu pohjavesialueelle tai sen lähelle
- Akusto sijoittuu Natura 2000 -alueelle tai sen läheisyyteen
- Kemikaalimäärät ylittävät Tukesin ilmoitusvelvollisuuden rajat:
  * Litiumioni: palava neste, jos kapasiteetti yli 100 kWh samassa
    rakennuksessa (Tukesin ohje akkuenergiavarastoille)
  * Lyijyakut: ympäristölupa jos kaikkiaan > 30 tn lyijyä
- Jätevesiä tai sammutusvedesä pääsee ympäristöön (pohjavesiriski)
- Asutuksen lähellä ja merkittävä meluhaitta

PELKKÄ RAKENNUSLUPA (rakentamislaki 751/2023) riittää kun:
- Litiumioniakkuvarasto sijoittuu asemakaavoitetulle teollisuustonteille
  ilman pohjavesi- tai luontovaikutuksia
- Kemikaalimäärät alle Tukesin ilmoitusvelvollisuuden

ENERGIANTUOTANTOLAITOKSET — YMPÄRISTÖLUVAN KYNNYSARVOT (YSL liite 1):
- Polttoaineteholtaan > 50 MW: ympäristölupa LVV:ltä (kaikki polttoaineet)
- Polttoaineteholtaan 1–50 MW: rekisteröintimenettely (YSL 116 §)
  (kiinteä biomassa, turve, öljy, kaasu)
- Kaukolämpökattila ≥ 5 MW: rekisteröinti-ilmoitus
- Tuulivoima: ei YSL-lupatarvetta yleensä (ympäristölupa vain erikoistilanteet)
- Aurinkovoima: ei YSL-lupatarvetta yleensä
- SMR / ydinvoima: ydinenergialaki 990/1987, STUK-luvat; YSL-lupa
  BESS-komponentille tarvittaessa

POHJAVESIALUE — ERITYISSÄÄNNÖKSET (YSL § 17)
- Pohjavesialueelle ei saa sijoittaa toimintaa, joka vaarantaa pohjaveden
  laadun tai määrän
- Varsinkin 1. ja 2. luokan pohjavesialueet: tiukat rajoitukset
- Poikkeuslupa mahdollinen perustelluista syistä (YSL § 52)

LUPA- JA VALVONTAVIRASTO (LVV) — YMPÄRISTÖLUPAYHTEYS
Website: https://lvv.fi/ymparisto
Sähköinen asiointi: https://lvv.fi/yhteystiedot/sahkoinen-asiointi
Toimivaltainen viranomainen: suuret hankkeet, direktiivilaitokset,
pohjavesialueelle sijoittuvat laitokset
""",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Apufunktiot
# ─────────────────────────────────────────────────────────────────────────────

def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunks.append(text[start:end].strip())
        start += CHUNK_CHARS - OVERLAP
    return [c for c in chunks if len(c) > 100]


def _fetch_html(url: str, session) -> str | None:
    try:
        r = session.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        ct = r.headers.get("content-type", "")
        if r.status_code == 200 and "html" in ct:
            return r.text
        print(f"    HTTP {r.status_code} / ct={ct}: {url}")
    except Exception as exc:
        print(f"    virhe ({exc.__class__.__name__}): {url}")
    return None


def _extract_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "noscript", "meta", "link"]):
        tag.decompose()
    main = (soup.find("main") or soup.find("article")
            or soup.find(id="content") or soup.find(class_=re.compile(r"content|main|article", re.I)))
    target = main if main else (soup.body if soup.body else soup)
    lines = [l.strip() for l in target.get_text(separator="\n").splitlines()]
    return "\n".join(l for l in lines if l)


def _collect_links(html: str, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup
    parsed_base = urlparse(base_url)
    base_path   = parsed_base.path.rstrip("/")
    soup        = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    result: list[str] = []
    for a in soup.find_all("a", href=True):
        href  = a["href"].split("#")[0].split("?")[0]
        if not href:
            continue
        full  = urljoin(base_url, href)
        p     = urlparse(full)
        if p.scheme not in ("http", "https") or p.netloc != parsed_base.netloc:
            continue
        if not p.path.startswith(base_path):
            continue
        clean = p._replace(fragment="", query="").geturl()
        if clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Päälogiikka
# ─────────────────────────────────────────────────────────────────────────────

def ingest() -> None:
    import requests
    from sentence_transformers import SentenceTransformer
    import chromadb

    if not DB_DIR.exists():
        print(f"[ingest_fi_env] VIRHE: {DB_DIR} puuttuu.\n"
              f"                Aja ensin: python3 permit_ai/build_index.py")
        sys.exit(1)

    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION)

    existing_ids: set[str] = set(col.get()["ids"])
    print(f"[ingest_fi_env] Olemassaolevia chunkkeja: {len(existing_ids)}")

    new_docs:  list[str]  = []
    new_ids:   list[str]  = []
    new_metas: list[dict] = []

    # ── 1. Inline-lakitekstit ──────────────────────────────────────────────
    print("\n[1/2] Inline-lakitekstit (YSL 527/2014, YVA-laki 252/2017)…")
    for doc_id, source_label, text in _FI_ENV_LAW_DOCS:
        chunks = _chunk(text)
        for i, chunk in enumerate(chunks):
            id_ = f"fi_env_inline__{doc_id}__{i}"
            if id_ in existing_ids:
                continue
            new_docs.append(chunk)
            new_ids.append(id_)
            new_metas.append({
                "country":     "FI",
                "lang":        "fi",
                "source":      source_label,
                "source_type": "inline_law",
            })
        print(f"  {doc_id}: {len(chunks)} chunkkia")

    # ── 2. LVV web-sivut ──────────────────────────────────────────────────
    print("\n[2/2] LVV web-sivut (lvv.fi/ymparisto)…")
    session   = requests.Session()
    start_url = "https://lvv.fi/ymparisto"
    to_visit  = [start_url]
    visited:  set[str] = set()
    MAX_PAGES = 25
    DELAY_S   = 1.0

    while to_visit and len(visited) < MAX_PAGES:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        if len(visited) > 1:
            time.sleep(DELAY_S)

        html = _fetch_html(url, session)
        if not html:
            continue

        text   = _extract_text(html)
        chunks = _chunk(text)
        added  = 0
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        path_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", urlparse(url).path)[:40]

        for i, chunk in enumerate(chunks):
            id_ = f"web__FI__{path_safe}__{url_hash}__{i}"
            if id_ in existing_ids:
                continue
            new_docs.append(chunk)
            new_ids.append(id_)
            new_metas.append({
                "country":     "FI",
                "lang":        "fi",
                "source":      "lvv.fi",
                "url":         url,
                "source_type": "web",
            })
            added += 1

        short = url.replace("https://", "")[:70]
        print(f"  [{len(visited):2d}] {short}  →  {len(chunks)} chunkkia, {added} uutta")

        for link in _collect_links(html, start_url):
            if link not in visited and link not in to_visit:
                to_visit.append(link)

    # ── Kirjoita ChromaDB ──────────────────────────────────────────────────
    if not new_docs:
        print("\n[ingest_fi_env] Ei uusia chunkkeja — kaikki jo indeksoitu.")
        return

    print(f"\nLisätään {len(new_docs)} chunkkia ChromaDB:hen…")
    for i in range(0, len(new_docs), BATCH):
        b    = slice(i, i + BATCH)
        embs = model.encode(new_docs[b], show_progress_bar=False).tolist()
        col.add(
            documents  = new_docs[b],
            embeddings = embs,
            ids        = new_ids[b],
            metadatas  = new_metas[b],
        )
        pct = min(100, (i + len(new_docs[b])) * 100 // len(new_docs))
        print(f"  {i + len(new_docs[b])}/{len(new_docs)} ({pct}%)")

    print(f"\n✅ Valmis — {len(new_docs)} uutta chunkkia. Koko indeksi: {col.count()} chunkkia")


if __name__ == "__main__":
    ingest()
