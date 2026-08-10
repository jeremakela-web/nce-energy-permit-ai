"""
Priority-2 remediation: maatalous + vesivoima named-but-unsupported content
gap (see the full-platform coverage audit, 2026-08-09/10). Both hanketyyppi
previously had ZERO dedicated source content -- their permit tables/liitteet
named specific requirements (manure storage sizing, Ruokavirasto/EU CAP
building-subsidy conditions, Vesilaki/Patoturvallisuuslaki citations, fish-
passage obligations) with nothing behind them.

Finlex is fully JS-rendered and does not support simple HTTP fetch (same
constraint documented in ingest_fi_env.py). Every text below was sourced by
directly reading real primary/authoritative documents -- verbatim statutory
text where a primary source was reachable (YSL 527/2014 Liite 1 via a
FAOLEX PDF mirror, Vesilaki 587/2011 Ch.3 via a FAOLEX PDF mirror,
Patoturvallisuuslaki 494/2009 via Finlex's own official unofficial-
translation PDF export, MMM 610/2023 via a FAOLEX PDF mirror of the alkup
page), and REAL VERBATIM PAGE TEXT (not an LLM summary/paraphrase) for the
Ruokavirasto CAP page (fetched and stripped of markup directly) and the
nitrate-decree compliance-guidance deck. None of these are auto-fetchable,
so this follows ingest_fi_env.py's established pattern: curated INLINE
text, source_type="manual" + last_verified so a future freshness check can
find it, never auto-refreshable since there is no fetch path.

IMPORTANT, flagged per user instruction: nitraattiasetus_1250_2014_
valvontaohje is NOT the bare pykala-level statute text of VNa 1250/2014 --
no official English translation or accessible PDF mirror of the primary
decree text was found despite genuine effort (see the PR's commit message
for the full search trail). What IS ingested here is a real, detailed
compliance/enforcement guidance document (looks Ruokavirasto/ELY-keskus-
authored) that explicitly and repeatedly cites "Nitraattiasetus (1250/2014)"
and gives specific, real numeric requirements (spreading dates, buffer
zones, nitrogen limits, manure-analysis intervals, storage exemptions).
Traceable via its source_label and the doc_type="viranomaisohje"
classification (not "laki") -- if the primary pykala text is later located,
it should be ingested as a SEPARATE source, not merged into this one.

hanketyyppi_tag is deliberately NOT set in this ingestion's metadata --
per explicit user instruction, tagging goes exclusively through
source_policy.py's SOURCE_HANKETYYPPI_TAG (single source of truth), not
duplicated into chunk metadata the way ingest_iaea.py and ingest_fi_env.py
do it. See source_policy.py's PR-D... er, Priority-2 entries for the tag
mapping.

Käyttö:
    python3 permit_ai/ingest_maatalous_vesivoima.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DB_DIR = HERE / "embeddings"

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION  = "permit_docs"
CHUNK_CHARS = 1500
OVERLAP     = 200
BATCH       = 32

# ─────────────────────────────────────────────────────────────────────────────
# INLINE documents -- (doc_id, source_label, url_or_None, text)
# ─────────────────────────────────────────────────────────────────────────────

DOCS: list[tuple[str, str, str | None, str]] = [
    (
        "ysl_527_2014_liite1_elainsuoja",
        "YSL 527/2014, Liite 1 kohta 11 -- Eläinsuojien ympäristölupakynnykset",
        None,
        """\
YMPÄRISTÖNSUOJELULAKI 527/2014

=== 27 § Yleinen luvanvaraisuus ===

Ympäristön pilaantumisen vaaraa aiheuttavaan toimintaan, josta säädetään liitteen 1 taulukossa 1
(direktiivilaitos) ja taulukossa 2, on oltava lupa (ympäristölupa). Eläinsuojan luvanvaraisuuden
määrittämiseen eräissä tapauksissa käytettävistä eläinyksikkökertoimista säädetään liitteessä 3.
(10.4.2015/423)
Ympäristölupa on lisäksi oltava:
1) toimintaan, josta saattaa aiheutua vesistön pilaantumista eikä kyse ole vesilain mukaan luvanvaraisesta
hankkeesta;
2) jätevesien johtamiseen, josta saattaa aiheutua ojan, lähteen tai vesilain 1 luvun 3 §:n 1 momentin 6
kohdassa tarkoitetun noron pilaantumista;
3) toimintaan, josta saattaa ympäristössä aiheutua eräistä naapuruussuhteista annetun lain (26/1920) 17 §:n 1
momentissa tarkoitettua kohtuutonta rasitusta.
28 § (19.12.2018/1166)
Luvanvaraisuus pohjavesialueilla
Liitteessä 2 tarkoitetun energiantuotantolaitoksen, asfalttiaseman, jakeluaseman, betoniaseman,
betonituotetehtaan ja liitteen 2 kohdassa 5–7 mainittuun toimintaan, kun orgaanisten liuottimien kulutus on
enemmän kuin 10 tonnia vuodessa sekä liitteessä 4 tarkoitettuun toimintaan on oltava ympäristölupa, jos
toiminta sijoitetaan tärkeälle tai muulle vedenhankintakäyttöön soveltuvalle pohjavesialueelle.
Lisäksi liitteessä 1, liitteen 2 kohdassa 1 ja 3 sekä liitteessä 4 tarkoitettuun, mutta niitä vähäisempään
toimintaan ja liitteen 2 kohdassa 4 tarkoitetun kemiallisen pesulan toimintaan on oltava ympäristölupa, jos
toiminta sijoitetaan tärkeälle tai muulle vedenhankintakäyttöön soveltuvalle pohjavesialueelle ja toiminnasta

=== Liite 1, kohta 11: Eläinsuojat tai kalankasvatus (luvanvaraisuuden kynnysarvot) ===

11. Eläinsuojat tai kalankasvatus 11. Eläinsuojat tai kalankasvatus
a) Siipikarjakasvattamot, kun siipikarjapaikkoja on
yli 40 000 ja sikalat, kun tuotantosikojen (yli
30 kg:n painoisia) paikkoja on yli 2 000 tai kun
emakkopaikkoja on yli 750; siipikarjalla
tarkoitetaan kanoja, kalkkunoita, helmikanoja,
ankkoja, sorsia, hanhia, viiriäisiä, kyyhkysiä,
fasaaneja, peltopyitä ja muita lintuja
a) Eläinsuoja, joka on tarkoitettu vähintään
300 lypsylehmälle, 500 lihanaudalle tai
600 emolehmälle taikka muu eläinsuoja, jonka
kokonaiseläinyksikkömäärä liitteen 3 taulukon 1
eläinyksikkökertoimilla laskettuna on vähintään 3 000
b) Turkistarha, joka on tarkoitettu vähintään
500 siitosnaarasminkille tai -hillerille taikka vähintään
250 siitosnaarasketulle tai -supikoiralle taikka muu
turkistarha, jonka kokonaiseläinyksikkömäärä liitteen 3
taulukon 1 eläinyksikkökertoimilla laskettuna on
vähintään 100
c) Kalankasvatus- tai kalanviljelylaitos, jossa käytetään
vähintään 2 000 kg vuodessa kuivarehua tai sitä
ravintoarvoltaan vastaava määrä muuta rehua taikka
jossa kalan lisäkasvu on vähintään 2 000 kg vuodessa,
taikka kooltaan vähintään 20 hehtaarin
luonnonravintolammikko tai lammikkoryhmä
""",
    ),
    (
        "ruokavirasto_maatalouden_investointituet",
        "Ruokavirasto -- Maatalouden investointituet (CAP 2023-2027)",
        'https://www.ruokavirasto.fi/tuet/maatalous/investoinnit/maatalouden-investointituet/',
        """\
Maatalouden investointituet - Ruokavirasto
Maatalouden investointituet
Metsäkatoasetuksen tulkinta tarkentunut – navettainvestoinnit voivat jatkua
Maa- ja metsätalousministeriön tiedote 16.1.2025
Maatalouden investointitukea voi hakea esimerkiksi navetan, sikalan tai kasvihuoneen rakentamiskustannuksiin, salaojitukseen ja yhteisiin ojitusinvestointeihin tai esimerkiksi korjauksiin, jotka parantavat eläinten hyvinvointia tai ympäristön tilaa. Myös maatilan energiainvestointeihin, kuten aurinkopaneelien tai led-valaistuksen hankkimiseen, voidaan myöntää investointitukea.
Kuka voi saada tukea?
Voit saada maatalouden investointitukea, jos olet 18 vuotta täyttänyt viljelijä, joka elinkeinonaan harjoittaa tai ryhtyy harjoittamaan maatilalla maataloutta ja jolla on riittävä ammattitaito. Ammattitaitovaatimus ei koske investointeja, jotka edistävät ympäristön tilaa, kestävää tuotantotapaa, eläinten hyvinvointia ja bioturvallisuutta eikä energiainvestointeja (tukikohteet 2–4).
Jos tilan hallinta kuuluu kahdelle tai useammalle henkilölle, hakekaa tukea yhdessä. Tällöin vähintään kolmasosa tilasta on oltava henkilöllä, joka täyttää tuen myöntämisen edellytykset.
Tukea voi saada myös yksityisoikeudellinen yhteisö tai maatalousyrittäjien yhteenliittymä (esimerkiksi kuivuriosakeyhtiö). Huomioi tällöin:
Jos tukea hakee yhteisö, osake-enemmistön ja määräysvallan tulee olla henkilöllä tai henkilöillä, jotka täyttävät hakijaa koskevat ehdot.
Jos haette tukea maatalousyrittäjien yhteenliittymänä, kaikkien yhtymän jäsenten on täytettävä ikä- ja ammattitaitovaatimukset sekä vähintään puolella osakkaiden maatiloista tulee täyttyä edellytykset jatkuvaan kannattavuuteen.
Elinvoimakeskus voi tarkistaa, täyttyvätkö lain vaatimukset tarkastusten tai valvontojen perusteella, viranomaisten päätösten perusteella, oman selvityksesi perusteella tai tekemällä maatilakäynnin.
Tuen määrä
Rakentamisinvestoinnissa tuen määrä on aina yli 7 000 euroa. Sen pienempään investointiin ei myönnetä tukea. Muissa toimenpiteissä vastaava määrä on 3 000 euroa.
Tukea voidaan kolmen verovuoden jakson aikana myöntää enintään 1 500 000 euroa, josta kilpailukykyyn ja nykyaikaistamiseen liittyviin investointeihin (tukikohde 1) voi kohdistua enintään 1 200 000 euroa maatilaa kohti.
Voit saada valtiontakausta seuraaviin kohteisiin:
lypsy- ja nautakarjanavetat
sikalat
lihasiipikarjan tuotantorakennukset
lammas- ja vuohinavetat
hevoskasvatuksen tuotantorakennukset
kasvihuoneet ja elintarviketuotantoon viljeltävän puutarhakasvien kasvutunnelit
energiantuotantoinvestoinnit
tilanpidon aloittaminen.
Voit saada valtiontakausta yhtä investointia kohden korkeintaan 800 000 euroa. Takaus on voimassa koko laina-ajan ja sen määrä pienenee samassa suhteessa kuin lainakin. Voit saada valtiontakausta vain tasalyhennyslainaan.
Valtiontakauksen määrä
Valtiontakaukseen sisältyvä tuki on 0,15 % takauksen määrästä.
Valtiontakaus voi koskea enintään 80 % takauksen kohteena olevan lainan määrästä koko laina-aikana.
Valtiontakaus voi ainoastaan erityisestä syystä olla suurempi kuin 30 % toimenpiteen kokonaisrahoituksesta.
Myönnetty avustus ja valtiontakaus yhteensä eivät saa ylittää 70 % tuen kohteena olevan investoinnin kokonaisrahoituksesta.
Yhtä maatilaa kohden saa olla valtiontakauksia voimassa enintään 2,5 miljoonaa euroa.
Vastavakuus vaaditaan
Sinun on annettava valtiontakaukselle vastavakuus, joka voi olla kiinteistö- tai yrityskiinnitys. Yhteisömuotoiselta yritykseltä elinvoimakeskus voi edellyttää myös henkilötakausta yhteisön osakkailta.
Pyydä lainatarjous vähintään kolmelta pankilta
Pyydä lainatarjous vähintään kolmelta pankilta ja liitä saamasi lainatarjoukset hakemukseesi. Voit pyytää lainatarjouksia samaan pankkiryhmään (esim. OP-ryhmä) kuuluvista pankeista, mutta ei saman pankin eri konttoreista. Varaudu tarvittaessa perustelemaan valitsemasi lainatarjous.
Valtiontakauksesta perittävät maksut
Kun sinulle on myönnetty valtiontakaus, pankki perii sinulta valtiolle maksettavaksi kertamaksun, jonka suuruus on 0,75 prosenttia takauksen määrästä, kuitenkin enintään 200 euroa. Lisäksi sinun on maksettava puolivuosittain maksu, jonka suuruus on 0,75 prosenttia takauksen kulloinkin jäljellä olevasta määrästä. Takauksesta perittävien maksujen eräpäivät ovat vuosittain huhtikuun ja lokakuun viimeisenä päivänä.
Tukikohteet
Navetat
Lypsy- ja nautakarjataloudessa tarvittavat rakentamisinvestoinnit
Tukea voidaan myöntää lypsy- ja nautakarjataloudessa tarvittavaan rakentamisinvestointiin. Uuden lypsykarjapihaton rakentamisinvestointiin voidaan tukea myöntää, jos pihaton yhteydessä on käytettävissä jaloittelutarha tai laidun. Parsinavettaan kohdistuvassa rakentamisinvestoinnissa tukea voidaan myöntää vain peruskorjaukseen. Tukea peruskorjaukseen ei kuitenkaan myönnetä kustannuksiin, jotka lisäävät parsipaikkojen lukumäärää.
Ota selvää, että investointitukihakemuksen kohteena olevan nautakarjarakennuksen ja jaloittelutarhan rakennuspaikka ei tule aiheuttamaan metsäkatoa. Huomioi, että sen tahon, joka ensimmäistä kertaa saattaa tuotteet markkinoille, on toimitettava metsäkatoasetuksen asianmukaista huolellisuutta koskeva vakuutus asetuksen 4 artiklan mukaisesti. (Euroopan parlamentin ja neuvoston asetus (EU) 2023/1115).
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 35 %
Nuoren viljelijän avustuskorotus: 10 %
Sikalat
Sikataloudessa tarvittavat rakentamisinvestoinnit.
Emakkosikalan uudisrakentamisessa tai laajennuksessa tukea voidaan myöntää, jos investoinnin kohteena olevassa sikalassa tai sen laajennusosassa on käytössä vapaaporsitus. Emakkosikalan peruskorjaukseen voidaan myöntää tukea, jos porsitushäkkien lukumäärää ei lisätä.
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 35 %
Nuoren viljelijän avustuskorotus: 10 %
Lihasiipikarjatalous
Siipikarjanlihan tuotannossa tarvittavat rakentamisinvestoinnit.
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 35 %
Nuoren viljelijän avustuskorotus: 10 %
Lammas- ja vuohitalous
Lammas- ja vuohitaloudessa tarvittavat rakentamisinvestoinnit.
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 35 %
Nuoren viljelijän avustuskorotus: 10 %
Hevostalouden investoinnit
Hevosten kasvattamisessa tarvittavat rakentamisinvestoinnit.
Tukea ei myönnetä investointiin, joka liittyy hevostalouden palvelutoimintaan.
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 35 %
Nuoren viljelijän avustuskorotus: 10 %
Mehiläistalous
Mehiläistaloudessa tarvittavat rakentamisinvestoinnit sekä kone- ja laitehankinnat.
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 25 %
Nuoren viljelijän avustuskorotus: 10 %
Kasvihuonetuotanto
Kasvihuonetuotannossa tarvittavat rakentamisinvestoinnit sekä elintarvikekäyttöön viljeltävän puutarhakasvin tuotannossa tarvittavan kasvutunnelin hankinnat.
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 25 %
Nuoren viljelijän avustuskorotus: 10 %
Kuivaamot
Viljan tai heinän kuivaamiseen tarkoitetun kuivaamon rakentamisinvestoinnit sekä viljan vaunukuivureiden hankinnat ja kuivaavan siilon hankinnat. Tukea kuivaamisessa tarvittavaan lämmöntuotanto- ja puhallinjärjestelmään voidaan myöntää energiatuotanto-kohdan mukaisesti.
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 25 %
Nuoren viljelijän avustuskorotus: 10 %
Varastot
Maataloustuotannossa välttämättömät tuote-, tuotantopanos- ja tarvikevarastojen sekä maatalouskonevarastojen rakentamisinvestoinnit
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 25 %
Nuoren viljelijän avustuskorotus: 10 %
Myyntikunnostus
Tukea voidaan myöntää maataloustuotteiden myyntikunnostamisessa tarvittavaan rakentamisinvestointiin ja koneen tai laitteen hankintaan. Tukea ei kuitenkaan myönnetä poronlihan myyntikunnostukseen liittyviin eikä kananmunapakkaamojen investointeihin.
Tuen myöntämisen edellytyksenä on, että:
myyntikunnostamisessa hyödynnetään pääosin tuen kohteena olevan maatilan raaka-aineita
tuotteet valmistetaan myytäviksi jälleenmyyjille tai jatkojalostajille
tuote on myyntikunnostuksen jälkeen edelleen Euroopan unionin toiminnasta tehdyn sopimuksen liitteessä I tarkoitettu maataloustuote.
Korkotukilainan määrä hyväksyttävistä kustannuksista: 50 %
Korkotuen määrä hyväksyttävistä kustannuksista: 5 %
Avustuksen määrä hyväksyttävistä kustannuksista: 25 %
Nuoren viljelijän avustuskorotus: 10 %
Työympäristöä ja tuotantohygieniaa edistävät investoinnit
Tukea voidaan myöntää rakentamisinvestointiin tai koneen tai laitteen hankintaan, jonka tarkoituksena on parantaa maatalouden tuotantorakennuksessa työskentelevän henkilön työoloja tai maatilan tuotantohygieniaa
Korkotukilainan määrä hyväksyttävistä kustannuksista: -
Korkotuen määrä hyväksyttävistä kustannuksista: -
Avustuksen määrä hyväksyttävistä kustannuksista: 25 %
Investoinnin pitää edistää ympäristöystävällisemmän tuotantotavan ja teknologian käyttöönottoa. Investoinnilla on oltava lisäksi vähintään yksi myönteinen vaikutus johonkin seuraavista kohteista:
Maan kasvukunto
Vesitalous
Kasvinsuojeluaineiden käytön turvallisuus
Ravinteiden hyötykäyttö ja kierrätys
Lannan tehokas käsittely
Kasvihuonekaasupäästöjen vähennys.
Korkotukilainan määrä hyväksyttävistä kustannuksista: -
Korkotuen määrä hyväksyttävistä kustannuksista: -
Avustuksen määrä hyväksyttävistä kustannuksista: 40 %
Tukea voidaan myöntää energiantuotantoon, energian säästöön tai energiatehokkuuden parantamiseen liittyviin investointeihin.
Tuen myöntämisen edellytyksenä on, että energialaitoksessa hyödynnetään uusiutuvaa energialähdettä.
Tuki voidaan myöntää biokaasulaitokseen, jos tuotetusta energiasta enintään puolet myydään markkinoille.
Korkotukilainan määrä hyväksyttävistä kustannuksista: -
Korkotuen määrä hyväksyttävistä kustannuksista: -
Avustuksen määrä hyväksyttävistä kustannuksista: biokaasulaitokset 50 %, muut energiainvestoinnit 40 %.
Maatilan investointitukea voidaan myöntää eläinten hyvinvointia edistävään rakentamisinvestointiin ja koneen ja laitteen hankintaan edellyttäen, että investointi parantaa jo olemassa olevaa tuotantorakennusta eikä lisää olemassa olevaa tuotantokapasiteettia.
Bioturvallisuutta edistävät investoinnit voivat kohdistua olemassa olevan tuotantorakennuksen bioturvallisuuden parantamiseen. Tukea voidaan myöntää suoja-aitoihin, joilla suojataan sikaloita afrikkalaiselta sikarutolta. Turkistarhoille voidaan myöntää tukea verkottamiseen tai muun vastaavan suojamateriaalin hankintaan ja asentamiseen, jolla estetään lintujen pääsy tiloihin, joissa turkiseläimiä pidetään. Turkistuottajille myönnettävä tuki on väliaikainen ja voimassa 15.4.2026 saakka.
Jos rakennat navetan yhteyteen jaloittelutarhan, selvitä, että investointitukihakemuksen kohteena olevan jaloittelutarhan rakennuspaikka ei tule aiheuttamaan metsäkatoa. Toimita metsäkatoasetuksen asianmukaista huolellisuutta koskeva vakuutus asetuksen 4 artiklan mukaisesti. (Euroopan parlamentin ja neuvoston asetus (EU) 2023/1115).
Korkotukilainan määrä hyväksyttävistä kustannuksista: -
Korkotuen määrä hyväksyttävistä kustannuksista: -
Avustuksen määrä hyväksyttävistä kustannuksista: 40 %
Tukikohteet taulukkomuodossa
Tukikohde
Korkotukilainan määrä hyväksyttävistä kustannuksista, prosenttia
Korkotuen määrä hyväksyttävistä kustannuksista, prosenttia
Avustuksen määrä hyväksyttävistä kustannuksista, prosenttia
Korotus
Lypsy- ja nautakarjatalous
50
5
35
*)
Sikatalous
50
5
35
*)
Lihasiipikarjatalous
50
5
35
*)
Lammas- ja vuohitalous
50
5
35
*)
Hevostalous
50
5
35
*)
Mehiläistalous
50
5
25
*)
Kasvihuonetuotanto
50
5
25
*)
Kuivaamo
50
5
25
*)
Varasto
50
5
25
*)
Myyntikunnostus
50
5
25
*)
Työympäristöä ja tuotantohygieniaa edistävät investoinnit
25
Ympäristön tilaa ja kestävää tuotantotapaa edistävät investoinnit, esim. salaojainvestoinnit
40
Maatilojen energiainvestoinnit
- biokaasuntuotantoon liittyvät investoinnit
50
- muut maatilojen energiainvestoinnit
40
Eläinten hyvinvointia ja bioturvallisuutta edistävät investoinnit
40
*) Enintään 40-vuotiaan viljelijän avustuksen enimmäismäärää voidaan korottaa 10 prosenttiyksikköä, jos tilanpidon aloittamisesta kulunut on enintään 7 vuotta.
Tuen ehdot
Omistat tai olet vuokrannut tuen kohteena olevan maatilan.
Sinulla on Maanmittauslaitoksen kirjauspäätös vuokrasopimuksestasi. Liitä kirjauspäätös tukihakemukseen.
Vuokraoikeuden tulee olla siirrettävissä kolmannelle kiinteistön omistajaa kuulematta, ja sen voimassaolon tulee jatkua vähintään 10 vuotta.
Saat vähintään 25 000 euroa maatalouden yrittäjätuloa viimeistään viidentenä kalenterivuonna tuen myöntämisestä (kilpailukykyyn ja nykyaikaistamiseen liittyvät investoinnit). Yrittäjätulo lasketaan maataloudesta saatavista tuotoista vähentämällä maatalouteen kohdistuvat muuttuvat ja kiinteät kulut, poistot ja velkojen korot. Noudatat tilallasi pakollisia vaatimuksia, jotka perustuvat ympäristöä, hygieniaa ja eläinten hyvinvointia koskevaan Euroopan unionin ja kansalliseen lainsäädäntöön.
Näin haet tukea
Hae investointitukea Hyrrä-asiointipalvelussa.
Huomioi, että hanke on toteutettava viimeistään 30.6.2029 mennessä EU-rahoituskauden päättymisen vuoksi.
Jos haet rahoitusta salaoja- tai yhteisojitusinvestointiin, selvitä, vaaditaanko ojituksesta ilmoitusta. Lisätietoa ojitusilmoituksesta
Hyrrä-asiointipalvelu
Ohjeet
Näin haet tukea Hyrrä-asiointipalvelussa
Näin haet lainan nostolupaa Hyrrä-asiointipalvelussa
Hakemuksen liitteet
Tähdellä (*) merkityt liitteet ovat pakollisia.
Kotieläinrakennukset
Mehiläistalous
Kasvihuonetuotanto
Kuivaamot
Myyntikunnostus
Varastot
Työympäristöä ja tuotantohygieniaa edistävät investoinnit
1. liiketoimintasuunnitelma ( lomake 3430 ) ja siihen liittyvät laskelmat ( täyttöohje )*
2. pankin luottolupaus ( lomake 3311 ), jos haetaan korkotukilainaa*
3. jäljennös hakijan verolomakkeesta 2 (maatalouden veroilmoitus)*
4. jäljennös hakijan viimeisimmästä verotuspäätöksestä*
5. jos on yritystoimintaa, niin silloin myös elinkeinon veroilmoitus
6. luotettava selvitys tulojen muutoksista, jos ne ovat olennaisesti muuttuneet tai tulevat muuttumaan toimitetun verotuksen mukaisista tuloista
7. koulu- tai tutkintotodistukset, jos tuotantosuunta muuttuu (muuten selvitys työkokemuksesta esim. liiketoimintasuunnitelmassa)
8. jäljennökset vuokrasopimuksista, jos tuettava rakennus sijaitsee alueella, jonka hallinta perustuu vuokrasopimukseen. Tuen myöntämisen edellytyksenä on, että vuokrasopimukset ovat voimassa 10 vuotta ja ne on kirjattu.
9. lomakkeelle ( nro 500 ) laadittu luettelo yrityksen veloista, joiden vakuutena yritysomaisuus on, tai vastaavat tiedot muuten sekä tarvittaessa rasitustodistus maatilan tiloista
10. rakentamista koskevat suunnitelmat, jotka on tehty Suomen rakentamismääräyskokoelman A2 ja maa- ja metsätalousministeriön rakentamista koskevien asetusten mukaisesti:
a. pääpiirustukset b. rakennusselostus (suositus: Talo-nimikkeistön mukaan eriteltynä) c. rakennusselostukseen perustuva eritelty kustannusarvio tai laskelma, joka laaditaan rakennusalalla yleisesti käytössä olevin menetelmin d. erikoissuunnitelmat, kuten rakenne-, LVI-, sähkö- ja muut vastaavat suunnitelmat, jos niillä on merkitystä rakennuksen toimivuutta ja hyväksyttäviä kustannuksia arvioitaessa
11. jäljennökset investoinnin edellyttämistä viranomaisluvista liiteasiakirjoineen
12. koneista, laitteista tms. hankinnoista myyjän vähintään kolme vertailukelpoista tarjousta.
Ympäristön tilaa ja kestävää tuotantotapaa edistävät investoinnit, maatilojen energiainvestoinnit, eläinten hyvinvointia ja bioturvallisuutta edistävät investoinnit
ei tarvita liiketoimintasuunnitelmia, mutta ainakin isoissa investoinneissa olisi ne kuitenkin hyvä toimittaa
ei ole ammattitaitovaatimusta eli ei tarvita todistuksia
salaojista ja muista rakentamiseen liittyvistä kohteista riittävällä ammattitaidolla laaditut suunnitelmat.
luottolaitoksen täyttämä vakuusarviolomake 4447*
maatilan omistusselvitys (jäljennös lainhuutorekisterikortista sekä niistä saantokirjoista, joiden perusteella omistuksessa on lainhuudon jälkeen tapahtunut muutoksia; esimerkiksi kauppa-, vaihto- tai lahjakirja, ositus-, perinnönjako- tai yhtiösopimus), jos takausluoton kohteena oleva hanke toteutetaan hakijan omistamalla tilalla*
rasitustodistukset kiinteistöistä, jotka kuuluvat hakijan maatilaan*
pankin antama luottolupaus ja kaksi kilpailevaa lainatarjousta tai jos niitä ole ei saatu, tarjouspyynnöt tai kielteiset lainatarjoukset*.
Tähdellä (*) merkityt liitteet ovat pakollisia.
Jos hakijana on yhteisö, on hakemukseen lisäksi liitettävä:
a. tuloslaskelma ja tase
b. tuenhakijan kaupparekisteriote ja oikeushenkilön osalta yhteisön säännöt, jos hakija on kaupparekisteriin merkitty yritys
c. muun yksityisen tai julkisen yhteisön osalta selvitys nimenkirjoitusoikeudesta
d. jäljennös yhteisön sen kokouksen kokouspöytäkirjasta, jossa hankkeesta ja sitä koskevasta tukihakemuksesta on päätetty.
Jos kyseessä on yhteisojitusinvestointi, hakemukseen lisäksi liitettävä:
jäljennös hanketta varten perustetun ojitusyhteisön perustamisasiakirjasta ja säännöistä tai, jos tällaista yhteisöä ei ole perustettu, hyödynsaajien keskinäinen sopimus hankkeen toteuttamisesta
ojitusilmoituksen vastaus, ojitustoimituksen päätös tai vesilupa, jos nämä ovat vesilain (587/2011) perusteella tarpeen hankkeen toteuttamiseksi
vesilain 5 luvun 15 §:n mukainen ojitussuunnitelma, kustannusarvio ja kustannusten osittelu
ilmoitus hankkeen toimitsijoista
yhteisojitusinvestoinneissa vähintään puolelta osakkaista on oltava kannattavaa maataloustoimintaa, joten näiltä vaaditaan verotuspäätös, maatilan veroilmoitus ja velkaluettelo
selvitys arvonlisäveron hyväksyttävyydestä tukikelpoiseksi (koskee ojitusyhteisöjä)
pöytäkirja kokouksesta, jossa on päätetty tuen hakemisesta.
Tee muusta kuin vähäisestä ojituksesta ojitusilmoitus Lupa- ja valvontavirastolle ennen tuen hakemista. Lupa- ja valvontavirasto käsittelee ilmoitukset 60 vuorokauden kuluessa. Ojitusilmoituksen vastaus tarvitaan hakemuksen liitteeksi. Ojitusilmoitusta ei tarvitse tehdä, mikäli kyseessä on ojitustoimituksessa tai Lupa- ja valvontavirastossa vahvistetusta ojitussuunnitelmasta.
Lisätietoa ojitusinvestoinneista:
Maatalousmaan kuivatus (vesi.fi)
Peruskuivatushankkeen rahoitus (vesi.fi)
Lisätietopyynnöt
Näin vastaat täydennyspyyntöön
Muutoshakemus
Näin teet muutoshakemuksen
Kun olet lähettänyt hakemuksesi
Hakemukset ratkaistaan tukijaksoittain, jotka ovat:
16.10.–15.1.
16.1.–15.3.
16.3.–15.8.
16.8.–15.10.
Maatalouden investointitukia koskevat valintaperusteet . Valintaperusteita sovelletaan yhdenmukaisesti kaikilla elinvoimakeskuksen alueilla.
Älä aloita ennen vireilletuloa
Älä aloita rakentamisinvestointia tai muuta toimenpidettä äläkä allekirjoita sopimusta (esimerkiksi lopullinen luovutuskirja, tilaussopimus tai urakkasopimus) ennen hakemuksen vireilletuloa! Investointitukea ei myönnetä lainkaan sellaiseen toimenpiteeseen, jonka toteuttaminen on aloitettu ennen hakemuksen vireilletuloa.
Mikä on aloittamista?
Kun rakennat tai laajennat rakennusta:
Perustustyö on aloitettu valamalla tai muulla vastaavalla kestävällä tavalla tai jos perustustyön toteuttaa urakoitsija, lopullinen urakkasopimus on allekirjoitettu.
Kun peruskorjaat rakennusta:
Työn tekeminen on aloitettu tai jos se teetetään, lopullinen sopimus työn tekemisestä on allekirjoitettu.
Kun hankit koneen tai laitteen:
Tilaus on tehty tai sopimus hankinnasta on allekirjoitettu. Jos tilausta ei edellytetä, hankintahinta tai sen ensimmäinen erä on maksettu.
Rakennussuunnitelmasta ja salaojasuunnitelmasta johtuvat kustannukset ovat tukikelpoisia, vaikka ne ovat syntyneet ennen tukihakemuksen jättämistä.
Toteuta investointi määräajassa
Toteuta investointisi kahden vuoden kuluessa tukipäätöksestä.
Voit saada toteutukselle jatkoaikaa kaksi kertaa. Jatkoaikaa voi saada enintään vuodeksi kerrallaan. Hae jatkoaikaa Hyrrä-asiointipalvelussa ennen kuin päätöksen voimassaoloaika päättyy.
Edellytyksenä jatkoajan saamiselle on, että olet aloittanut tuettavan toimenpiteen toteuttamisen määräajassa ja että määräajan pidentämiseen on hyväksyttävä syy.
Muista viestintävelvoitteet
Verkkosivustot ja sosiaalisen median tilit
On tärkeää, että tuensaajan verkkosivulla ja sosiaalisen median kanavilla mainitaan saatu EU-rahoitus. Verkkosivustolla tai sosiaalisen median kanavilla tarkoitetaan yrityksen, organisaation tai hankkeen virallisia kanavia ja verkkosivustoja.
Tuensaajan, jolla on verkkosivusto tai sosiaalisen median kanava, tulee lisätä niihin  lyhyt kuvaus hankkeesta .
Verkkosivuilla ja sosiaalisesta mediassa olevasta kuvauksesta tulee käydä ilmi hankkeen tavoitteet, tulokset ja maininta EU-rahoituksesta. Kuvauksen yhteydessä on oltava EU-tunnus (EU-lippu ja lause: ”Euroopan unionin osarahoittama”). Sosiaalisessa mediassa tiedot voivat olla esillä esimerkiksi sosiaalisen median kanavan esittelytekstissä tai hanketta koskevassa yksittäisessä julkaisussa.
Liitä maksuhakemukseen kuvakaappaus esittelytekstistä tai julkaisusta, kun haet hankkeen maksua.
Kuvaus saadusta tuesta tulee sisällyttää www-sivuille tai sosiaaliseen mediaan vähintään loppumaksuun asti.
ESIMERKKI:   Jos on kyse investointiin saadusta tuesta, voi kuvaus olla julkaisun ohessa, jossa esimerkiksi esitellään, mitä tuella on tehty tai hankittu (esim. laitehankinta).
Jos hankkeen aikana perustetaan uusia verkkosivustoja tai sosiaalisen median tilejä, ilmoita niistä elinvoimakeskukselle. Viestintävelvoite ei koske tuensaajan henkilökohtaisia sosiaalisen median kanavia.
Tiedotuskyltti yli 50 000 euroa tukea saaneeseen investointiin
Jos julkisen tuen kokonaismäärä ylittää 50 000 euroa, hankkeen toteuttamisen ajaksi on sijoitettava investointia ja sille saatavaa EU-tukea esittelevä A4-kokoinen tiedotuskyltti näkyvälle paikalle: investoinnin yhteyteen, rakennuskohteeseen tai toimitiloihin. Kyltin tulee olla paikallaan vähintään hankkeen toteuttamisen ajan eli vähintään siihen saakka, kunnes olet saanut loppumaksun. Tiedot voidaan esittää myös sähköisellä näytöllä.
Pysyvä tiedotuskyltti yli 500 000 euroa tukea saaneeseen investointiin
Jos kyseessä on investointi, jonka julkisen tuen kokonaismäärä ylittää 500 000 euroa ja joka käsittää fyysisen kohteen hankinnan tai rakennushankkeen rahoittamisen, aseta näkyvälle paikalle pysyvä tiedotuskyltti. Kyltin tulee olla paikallaan vähintään viisi vuotta viimeisen maksun saamisesta tai korkotukilainan nostosta.
Saat kyltin elinvoimakeskukselta.
Tiedotuskyltit ja -julisteet (ennen 1.1.2023 saadut päätökset)
Kilpailuta hankinnat tarvittaessa
Kilpailuta hankinnat, jos:
rakennusurakkaa koskevan hankinnan arvo on 150 000 euroa tai sen yli
tai tarvike-, kone- ja laitehankinnan arvo on 60 000 euroa tai sen yli
ja samalla myönnetyn investointituen tukitaso ylittää tai saattaa ylittää 50 prosenttia.
Jos palvelu- tai tavarahankintojen arvo ylittää 209 000 euroa, tulee kilpailutus tehdä EU-kynnysarvot ylittäviä hankintoja koskevien menettelyjen mukaisesti.
Kilpailuttamisella tarkoitetaan sitä, että järjestät hankinnasta avoimen tarjouskilpailun sekä julkaiset hankintaa koskevan ilmoituksen julkisessa Hilma-ilmoituskanavassa osoitteessa www.hankintailmoitukset.fi .
Lisätietoja hankintamenettelyistä:
Maatalousinvestointien kilpailuttamisohje
Suomen Kuntaliiton julkisten hankintojen neuvontayksikkö:  hankinnat.fi
työ- ja elinkeinoministeriön sivuilta: tem.fi/julkisethankinnat .
Tuen maksaminen
Hae maatalouden investointituen maksamista ja lainan nostolupaa sähköisesti Hyrrä-palvelussa. Tukipäätöksen tehnyt elinvoimakeskus opastaa tarvittaessa hakemuksen täyttämisessä.
Näin teet maksuhakemuksen
Näin teet nostolupahakemuksen
Perehdy maksu- ja nostolupahakemukseen huolellisesti ja toimita myös tarvittavat liitteet. Lue tukipäätös ja toimita myös tukipäätöksellä mahdollisesti vaaditut dokumentit. Elinvoimakeskus voi pyytää tarvittaessa lisädokumentteja.
Investointiin voidaan myöntää enintään 5 maksupäätöstä ja 5 nostolupaa. Loppumaksun ja viimeisen nostoluvan on oltava vähintään 20 % tuen määrästä. Hae viimeinen maksuerä 2 kuukauden kuluessa toimenpiteen toteutusajan jälkeen. Investoinnin on oltava valmis ja tukipäätöksen mukainen ennen viimeisen maksupäätöksen tekemistä. Liitä viimeiseen maksuhakemukseen rakennusinvestointien osalta rakennusvalvonnan loppukatselmuspöytäkirja.
Elinvoimakeskus maksaa avustuksen ja myöntää lainan nostoluvan toteutuneiden tukikelpoisten kustannusten tai toteutuksen osoittamien dokumenttien perusteella.
Tukipäätöksestä näet, pitääkö sinun liittää maksuhakemukseen dokumentti kustannuksista vai valmiusastetodistus. Tukipäätöksessä on lueteltu myös mahdolliset muut maksuhakemukseen vaadittavat dokumentit.
Kustannusperusteinen maksuhakemus
Jos maksuhakemuksessa pyydetään esittämään kustannukset, liitä hakemukseen koontiluettelo ( lomake 3331 ) tai muut vastaavat dokumentit, joista voidaan todeta kustannusten hyväksyttävyys. Kustannusten tulee olla tuensaajan maksamia, kohdistua tuettuun toimenpiteeseen ja ne tulee olla kirjattuna tuensaajan kirjanpitoon tai muistiinpanovelvollisen verovelvollisen osalta muistiinpanoihin.
Ilmoita kustannukset ilman arvonlisäveroa. Jos kustannus jää sinulle lopulliseksi menoksi etkä saa sitä palautuksena, esimerkiksi ojitusyhteisössä, voit ilmoittaa kustannukset arvonlisäverollisina. Jos ilmoitat tukeen oikeuttamattomia, hylättäviä kustannuksia liikaa, maksettavaa tukisummaa pienennetään eikä tätä vähennettyä määrää makseta myöhemmissä maksuissa. Tämän verkkosivun alaosassa on esimerkkejä hylättävistä kustannuksista ja muista mahdollisista seuraamuksista.
Kustannusperusteisessa maksuhakemuksessa voidaan esittää kohtuullinen määrä tuensaajan tai -saajien tekemää omaa työtä, lomakkeella 3506 . Kustannuksissa voidaan esittää myös kohtuullinen määrä omien puuainesten ja maa-ainesten käyttöä, lomakkeella 3322L .
Jos olet hankkinut koneen tai laitteen osamaksulla, voit ilmoittaa kustannusperusteisessa maksuhakemuksessa kustannukseksi myös rahoitusvelan osuuden, jos seuraavat ehdot täyttyvät:
rahoitusyhtiö on maksanut rahoitusvelan myyjälle
sinulla on sopimukseen perustuva velvollisuus maksaa rahoitusvelka rahoitusyhtiölle
omaisuus on toimitettu sinulle
sinulla on oikeus tehdä poistoja omaisuudesta sekä mahdollisuus vähentää arvonlisävero.
Tuotosperusteinen maksuhakemus
Jos tuki maksetaan toteutuksen etenemisen osoittamien dokumenttien perusteella, liitä hakemukseen selvitys investoinnin valmiusasteesta ( lomake 3508 ) ja mahdollisesti muita toteutuksen osoittavia dokumentteja. Joidenkin hankkeiden osalta voi olla helpointa osoittaa toteutuminen ostodokumentilla tai esimerkiksi valokuvilla, jotka sisältävät paikkatiedot.
Aurinkosähköjärjestelmän osalta helpoin tapa todentaa investoinnin toteutus on liittää maksuhakemukseen valokuva käyttöönottotarkastuspöytäkirjasta, josta nähdään myös asennetun voimalan nimellisteho. Toinen valokuva kannattaa ottaa valmiiksi asennetuista aurinkopaneeleista (mahdollisella paikkatiedolla varustettuna).
Tukikelvottomat kustannukset
Näitä kustannuksia et voi ilmoittaa maksuhakemuksessasi, koska niistä ei makseta tukea.
Esimerkkejä tukeen oikeuttamattomista, maksuhakemuksessa hylättävistä kustannuksista:
tuotanto- tai tukioikeuksien hankinta
eläinten tai kasvien hankinta
traktorin tai pienkuormaajan hankinta
käytettyjen koneiden ja laitteiden hankinta
suunnittelukustannukset, jotka eivät liity rakennussuunnitelmaan
hankinnat, jotka on tehty hakijalta, hänen perheenjäseneltään, hakijan tai hänen perheenjäsenensä määräysvallassa olevalta yritykseltä tai hakijana olevan yrityksen johtavassa asemassa olevalta henkilöltä, jos hankinnasta ei ole hankittu riittävästi tarjouksia muilta asianmukaisilta tarjoajilta
investoinnin tai muun toimenpiteen rahoitus
arvonlisävero, jos se ei jää hakijan lopulliseksi menoksi
hakijan itsensä suorittaman kuljetuksen kustannukset, kuten esimerkiksi materiaalien noutaminen tavarantoimittajalta
investoinnin toteuttamiseen vaadittavan viranomaisen myöntämän luvan maksu
kustannukset, jotka aiheutuvat maatilayrityksen tavanomaiseen liiketoimintaan kuuluvasta toiminnasta tai siihen liittyvästä suhdetoiminnasta, kuten esimerkiksi työvaatteet ja työkalut tai majoituskustannukset
kustannukset, jotka ovat syntyneet ennen tukihakemuksen vireilletuloa, jos kustannukset eivät liity rakennussuunnitelmaan
liittymämaksut tai vastaavat kulut.
Investoinnin tarkastuskäynti
Elinvoimakeskus käy tarkastamassa paikan päällä, että olet toteuttanut investoinnin tukipäätöksen mukaisesti. Elinvoimakeskus ilmoittaa tarkastuskäynnin ajankohdan etukäteen. Käynnillä elinvoimakeskus tarkastaa muun muassa rakennukset, koneet ja laitteet, eläinten hyvinvoinnin, eri viranomaisten dokumentit, hankintasopimukset, kauppakirjat ja kirjanpidon.
Tuen takaisinperintä tai vähennys
Tuen maksatus keskeytetään ja maksettu tuki peritään takaisin, jos elinvoimakeskus huomaa maksuhakemusta käsitellessään, että edellytykset tuen saamiseksi ovat keinotekoiset tai olet antanut virheellisen tai puutteellisen tiedon, joka on olennaisesti vaikuttanut tuen myöntämiseen tai maksamiseen.
Maksettavaa avustussummaa pienennetään, jos maksuhakemuksessasi ilmoittamistasi kustannuksista yli 10 prosenttia hylätään. Summaa pienennetään siten, että hyväksyttävistä kustannuksista vähennetään kaksi kertaa hylättävien kustannusten määrä.
ESIMERKKI: Olet saanut tukea rakentamiseen, jonka hyväksyttävät kokonaiskustannukset ovat 200 000 € ja avustusta 60 000 € (avustustaso on 30 %). Esität ensimmäisellä maksuhakemuksella kustannuksia 100 000 € ja haet avustusta 30 000 €. Kustannuksista hylätään 30 000 €, eli 30 % on hylättäviä kustannuksia. Koska hylättäviä kustannuksia on yli 10 %, hylätään kustannuksista lisäksi toisen kerran 30 000 €, joten hyväksyttäviä kustannuksia jää jäljelle 40 000 € (esitetyistä 100 000 € vähennetään hylättävät 30 000 € ja lisävähennyksenä toiset 30 000 €). Avustusta maksetaan jäljelle jäävien kustannusten määrää vastaava summa, eli 40 000 € kertaa avustusprosentti (30 %) = 12 000 €. Ilman vähennystä olisit saanut 30 000 €. Tätä vähennettyä 18 000 € ei makseta myöhemmissä maksuerissä.
Maksettavaa avustussummaa pienennetään myös seuraavista syistä:
Tiedotuskyltin tai hankkeen kuvauksen puuttuminen verkkosivuilta tai sosiaalisen median kanavista. Jos tuensaaja on laiminlyönyt tuen ehtoihin sisältyneen tiedotusvelvoitteen noudattamisen, tukeen voidaan tehdä vähennys, joka on vähintään prosentti ja enintään viisi prosenttia myönnetystä tuesta.
Puutteet hankintojen kilpailutuksessa aiheuttavat kyseisten hankintojen osalta avustussumman pienentämisen. Tukipäätöksestä näet mahdolliset kilpailuttamisen velvoitteet.
Ennen tukihakemuksen tekemistä syntyneet kustannukset. Voit aloittaa investoinnin omalla riskillä heti tukihakemuksen jättämisen jälkeen. Jos kustannuksia on syntynyt ennen tukihakemusta, ne hylätään ja ne saattavat aiheuttaa vähennyksen.
Vähennettyä avustussummaa ei makseta myöhemmissä maksuerissä eli lopullinen avustusmäärä tulee jäämään vähennyksen verran pienemmäksi.
Tarvitsetko apua?
Hae elinvoimakeskuksesi yhteystiedot
Ohje: paikkatiedon lisääminen valokuvaan
Lomakkeet
3322L Oma puutavara ja/tai maa-aines 3331 Tositekohtainen koontiluettelo 3506 Tuntikirjanpito (oma työ) 3508 Investoinnin valmiusaste
Lainsäädäntö
Laki maatalouden rakennetuista
Valtioneuvoston asetus maatalouden rakennetuesta
Valtioneuvoston asetus maatilan investointituen kohdentamisesta
Maa- ja metsätalousministeriön asetus maatalouden investointien hyväksyttävistä yksikkökustannuksista
Sivu on viimeksi päivitetty 30.7.2026
""",
    ),
    (
        "mmm_610_2023_lypsykarjarakennukset",
        "MMM asetus 610/2023 -- Lypsykarjarakennusten rakennustekniset ja toiminnalliset vaatimukset",
        None,
        """\
610/2023 
Dokumentin versiot 
• Viitetiedot 
• På svenska 
Helsingissä 29.3.2023 
Maa- ja metsätalousministeriön 
asetustuettavaa rakentamista koskevista 
lypsykarjarakennusten rakennusteknisistä ja 
toiminnallisista vaatimuksista 
Maa- ja metsätalousministeriön päätöksen mukaisesti säädetään porotalouden ja 
luontaiselinkeinojen rakennetuista annetun lain (986/2011) 12 §:n 4 momentin sekä 
maatalouden rakennetuista annetun lain (1476/2007) 13 §:n 5 momentin, sellaisena kuin 
se on laissa 1328/2022, nojalla: 
1 § 
Soveltamisala 
Sen lisäksi, mitä rakennusten suunnittelusta, rakentamisesta ja paloturvallisuudesta sekä 
eläinsuojelusta on muutoin säädetty tai määrätty, tätä asetusta sovelletaan myönnettäessä 
tukea maatalouden rakennetuista annetun lain (1476/2007) tai porotalouden ja 
luontaiselinkeinojen rakennetuista annetun lain (986/2011) nojalla lypsykarjarakennusten 
uudisrakentamiseen, siihen verrattavaan laajennukseen tai laajaan peruskorjaukseen. 
2 § 
Määritelmät 
Tässä asetuksessa tarkoitetaan: 
1) vasikalla enintään kuuden kuukauden ikäistä nautaa sukupuolesta riippumatta; 
2) hieholla vähintään kahdeksan kuukauden ikäistä poikimatonta naaraspuolista nautaa; 
3) lehmällä poikinutta naaraspuolista nautaa; 
4) lypsylehmällä pääasiassa maidontuotantoa varten pidettävää lehmää; 
5) sonnilla vähintään kuuden kuukauden ikäistä urospuolista nautaa; 
6) nuorkarjalla 6–22 kuukauden ikäisiä nautoja; 
7) imettäjälehmä, vasikkaa imettävä lehmä, joka ei ole vasikan oma emä; 
8) lypsykarjarakennuksella rakennusta, jossa pidetään nautaeläimiä maidontuotantoa 
varten; 
9) parsinavetalla lypsykarjarakennusta, jossa lehmät ovat kytkettyinä parsiin; 
10) pihatolla lypsykarjarakennusta, jossa nautaeläimet voivat liikkua vapaasti syömään, 
makuulle ja lypsylle tai niillä on erillinen lypsyasema; 
11) eläintilalla lypsykarjarakennuksessa olevaa tilaa, joka muodostuu nautaeläinten parsi- 
ja karsina-alueesta, lypsyosasto ruokinta- ja lantakäytävät mukaan lukien; 
12) lypsyosastolla lypsyasemaa tai automaattilypsypaikkaa ja lehmien odotustilaa sekä 
siihen liittyvää käytävätilaa; 
13) kiinteäpohjaisella lattialla joko rei’ittämätöntä tasapintaista lattiaa tai harvaan 
rei'itettyä lattiaa, jossa on keskimäärin enintään 15 prosentin viemäröintireikäosuus; 
14) lantakäytävällä pihaton makuuparsien ja ruokintapöydän yhdistävää lehmien kulku-
väylää, josta lanta poistuu valumalla tai se poistetaan koneellisesti; 
15) jaloittelualueella eläinsuojan välittömässä yhteydessä sijaitsevaa aluetta, jota käytetään 
säännöllisesti eläinten jaloitteluun, ja jolta kerätään lanta ja valumavedet talteen; 
16) laitumella pelto-tai metsäpohjaista kasvipeitteistä aidattua aluetta, jolla eläimet saavat 
liikkua vapaasti. Laitumelta syötävä ruoho voi olla myös osa eläinten ruokintaa. 
3 § 
Yleisiä vaatimuksia 
Lypsykarjarakennusten paloturvallisuuteen sovelletaan tuettavaa rakentamista koskevista 
paloteknisistä vaatimuksista annettua maa- ja metsätalousministeriön asetusta 
(265/2019). Tuotanto ja eläinten hyvinvointi on voitava järjestää sähkökatkoksen sattuessa 
varasähköjärjestelmän avulla ja turvaamalla vedensaanti. Jos eläintilassa on 
sähköriippuvainen ilmanvaihto, on käytössä oltava hälytyslaitteisto, joka varoittaa eläinten 
hoitajaa sähkökatkosta. 
Uudisrakentamisessa eläintilan huonekorkeuden on oltava keskimäärin vähintään 2,7 
metriä. Peruskorjauksessa eläintilan huonekorkeuden on oltava riittävä eläimen kokoon 
nähden. 
Peruskorjauksessa ja sellaisessa laajennuksessa, joissa eläintilan laajennus on alle 
50 prosenttia olemassa olevan eläintilan pinta-alasta ja enintään 500 neliömetriä, voidaan 
käyttää olemassa olevia parsi- ja käytäväleveyksiä, ellei eläinsuojelu-, työsuojelu- tai muista 
säädöksistä muuta johdu. 
Parsien, makuualueiden, karsinoiden, lantakäytävien ja eläinten muiden kulkuväylien sekä 
ruokintapöytätilan mitoitusvaatimukset ovat liitteen taulukoissa 1–6. 
4 § 
Lypsykarjarakennuksen sijoitus ja kuljetusreitit 
Rakennuspaikka on valittava siten, että eläimillä on pääsy laitumelle tai jaloittelualueelle. 
Jaloittelualueen on täytettävä Maa- ja metsätalousministeriön asetuksen tuettavaa 
rakentamista koskevista ympäristönsuojeluvaatimuksista (606/2023) mukaiset tekniset 
vaatimukset. 
Maitoauton kulkureitin on oltava eriävä lannan kuljetusreitistä. Maatilan sisäinen sekä 
maatilalle tuleva ja sieltä lähtevä liikenne on esitettävä asemapiirroksessa. 
Rakennettavat tuotantopiha-alueet, alueiden pinta-ala, rakennekerrokset ja pintavesien 
johtaminen on esitettävä piirustuksissa. 
Eläintila on suunniteltava siten, että rehunkuljetusreitit ja ruokintapöydät eivät mene ristiin 
eläinten liikkumisväylien ja lannan kuljetusreittien kanssa. Jos tätä ei navetan 
toiminnallisuuden vuoksi voida toteuttaa, on risteyspaikalle rakennettava nostosilta tai 
vastaava rehuhygienian turvaava järjestely. 
5 § 
Ikkunat, muut valoaukot ja keinovalaistus 
Luonnonvalon saamiseksi eläintilassa on oltava ikkunoita tai muita niitä vastaavia 
valoaukkoja vähintään viisi prosenttia eläintilan lattiapinta-alasta. Valaistusvoimakkuuden 
vaatimukset ovat liitteen taulukossa 9. 
6 § 
Pihatto 
Pihaton lehmille ja muille nautaeläimille on oltava vähintään eläinten lukumäärää vastaava 
määrä makuuparsia tai karsinapaikkoja. Lypsylehmäpaikaksi lasketaan maidontuotannossa 
tai ummessa oleville lehmille pohjapiirustuksessa merkityt makuuparret tai karsinapaikat. 
Makuuparsien välissä on oltava parrenerottaja. 
Pihatto, jossa ei ole makuuparsia, on suunniteltava siten, että lehmille on järjestetty 
kuivikepohjainen makuualue ja erillinen ruokinta- ja lannankeräysalue. 
Suunnitelmassa on esitettävä eläinten kytkentämahdollisuus hoito- tai muita toimenpiteitä 
varten. 
Eläinten parsi- ja karsinajako sekä ryhmäkarsinoissa pidettävien eläinten määrä ja 
ikäluokka on merkittävä pohjapiirrokseen. Eläintilassa on oltava poikima- ja sairastilat, 
jotka on merkittävä pohjapiirrokseen mitoitettuina, myös jos nämä karsinatilat sijoitetaan 
toiseen rakennukseen. 
7 § 
Eläintilan lattia 
Eläinten liikkuma- ja makuutilan lattiarakenteen on oltava niin tiivis, että lietettä ei pääse 
ympäristöön. Eläinten makuualueella on oltava kiinteäpohjainen lattia. 
Lantakäytävien ja muiden rakolattia-alueiden palkki- ja rakoleveyden mitoitusvaatimukset 
ovat liitteen taulukossa 8. 
8 § 
Sairas-, hoito- ja poikimakarsinat 
Jokaista alkavaa 20 lypsylehmän ryhmää kohden on oltava vähintään yksi karsinatila 
sairastunutta, hoidettavaa tai poikivaa lehmää varten. Karsinatilan on sijaittava 
vedottomassa paikassa lypsyaseman läheisyydessä tai muutoin siten, että lehmä voidaan 
lypsää helposti. Karsinatilasta on oltava esteetön kulkuväylä ulko-ovelle. 
Lisäksi on oltava yksi hoitopaikka jokaista alkavaa 50 hiehoa tai muuta nautaeläintä 
kohden. 
Makuuparsipaikkaa ei lasketa sairas- tai poikimapaikaksi. Parsinavetassa enintään puolet 
vaadituista paikoista voi olla sairasparsia. 
Karsinatila voi olla yksittäis- tai ryhmäkarsina. Rakennussuunnitelmassa on esitettävä 
mahdollisuus jakaa ryhmäkarsina erilliseksi sairas- ja poikimatilaksi, ellei poikimatila ole 
erillisessä karsinassa. Karsinan jakaminen ei saa estää eläimen pääsyä ruokinta- ja 
juomapaikalle. Ruokinta- ja juomapaikat sekä eläinten kytkentämahdollisuus 
hoitotoimenpiteiden ajaksi on esitettävä pohjapiirustuksessa. 
Karsinan makuualueella on oltava kiinteäpohjainen lattia. Makuualueella ei saa olla 
lannanpoistoraappaa eikä rako- tai ritilälattiaa. 
Karsinan mitoitusvaatimukset ovat liitteen taulukossa 3. 
9 § 
Juomapaikat 
Kaikkien eläinten saatavilla on aina oltava puhdasta ja sulaa juomavettä. Juomapaikka voi 
olla juomakuppi tai -allas. Pihatossa lehmillä on oltava vähintään yksi juomapaikka jokaista 
alkavaa kuuden lypsylehmän ryhmää kohden. Muilla nautaeläimillä on oltava vähintään 
yksi juomapaikka jokaista alkavaa 20 eläimen ryhmää kohden, poikkeuksena 10–20 
eläimen ryhmä, jolla on oltava vähintään kaksi juomapaikkaa. 
Juomapaikkojen vähimmäismäärät ovat liitteen taulukossa 7. 
10 § 
Lypsyosasto 
Lypsyaseman edessä on oltava kerralla lypsylle tulevaa lehmämäärää kohden kokooma- ja 
odotustilaa vähintään 1,5 neliömetriä yksittäistä lehmää kohden. Kokooma- ja odotustila on 
mitoitettava lypsyaseman kapasiteetin mukaan siten, että lehmän odotusaika ei ylitä yhtä 
tuntia. Alle 60 lypsylehmän yksikössä ja automaattilypsyaseman yhteydessä kokooma- ja 
odotustila voi olla osa eläintilan liikunta-aluetta. 
11 § 
Maitohygienia 
Maitohuone ei saa olla suoraan yhteydessä eläintiloihin tai muihin sellaisiin tiloihin, joista 
voi siirtyä lantaa tai likaa maidonkäsittelytiloihin. Maidonkäsittelytilat voivat olla suoraan 
yhteydessä lypsyasemaan, jos lypsyasema on erotettu eläintiloista. 
Maidonkäsittelytilojen seinien ja lattioiden on oltava sellaisia, että ne voidaan puhdistaa ja 
desinfioida helposti. 
Lypsykarjarakennus on eläintautien ehkäisemistä varten varustettava tautisulun 
mahdollistavalla erillisellä sisäänkäynnillä. 
12 § 
Jaloittelualueet 
Jaloittelualueiden sijoittamisessa on huomioitava valtioneuvoston asetuksessa eräiden 
maa- ja puutarhataloudesta peräisin olevien päästöjen rajoittamisesta (1250/2014) 4 §:ssä 
esitetyt vaatimukset. 
Minimipinta-ala nautojen jaloittelualueelle on 6 m2 nautaa kohden, mutta tarhan 
kokonaisalan on aina oltava vähintään 50 m2. Mikäli eläimiä ulkoilutetaan ryhmissä, 
suunnitelmasta on ilmettävä, minkä kokoiselle ryhmälle jaloittelualue on mitoitettu. 
Jaloittelualueen aitauksen on kestettävä eläinten rasitusta ja pidettävä eläimet turvallisesti 
aitauksessa. 
13 § 
Voimaantulo 
Tämä asetus tulee voimaan 31 päivänä maaliskuuta 2023. 
Tällä asetuksella kumotaan maa- ja metsätalousministeriön asetus tuettavaa rakentamista 
koskevista lypsykarjatalousrakennusten rakennusteknisistä ja toiminnallisista 
vaatimuksista (405/2017). 
Helsingissä 29.3.2023 
Maa- ja metsätalousministeri 
Antti Kurvinen 
Projektipäällikkö 
Maarit Hellstedt 
Liite 
Taulukko 1.   Parsinavetan parsimitoitus. Parren kallistus lantakäytävälle 2-3 % 
Eläin Parren 
vähimmäisleveys1), 
mm 
Lyhytparren 
vähimmäispituus 2), mm 
Pitkäparren 
vähimmäispituus 2), mm 
Lypsylehmä 1 200 1 650 2 000 
Hiehot, 
nuorkarja, ikä 
> 22 kk 
1 100 1 550 1 800 
18–22 kk 1 000 1 500 1 700 
12–18 kk 900 1 200 1 500 
6–12 kk 800 1 000 1 200 
1) parsileveys tarkoittaa parrenerottajien vaakasuoraa mittaa keskeltä keskelle mitattuna. 
Parrenerottajan paksuus voi kaventaa parsileveyttä enintään 65 mm. 
2) parren pituus tarkoittaa ruokintapöydän tai -kourun ja parren rajaviivan ja parren ja 
lantakäytävän tai -kourun välisen rajaviivan etäisyyttä toisistaan. 
Taulukko 2.   Alle 2 kuukautta vanhan vasikan yksilökarsinan mitoitus  
Vähimmäispinta-ala, m2 Lyhimmän sivun vähimmäispituus, mm1) 
Vasikat < 2 kk 1,6 1 100 
1) Sisämittana: esitetty arvo miinus enintään 5 prosenttia 
Taulukko 3.   Nautakarjan ryhmäkarsina, vähimmäispinta-ala, karsinan sisämitoilla 
laskettuna, m 2 .  
Karsina ilman 
lantakäytävää, 
Kuivikepohja ja 
lantakäytävä 
m2/eläin 
 
kuivikepohja, 
m2/eläin 
Makuualue Kokonaispinta-ala, 
sis. lantakäytävä 
Lypsylehmä 
 
6,0 8,5 
Hiehot, nuorkarja, > 22 
kk 
6,0 4,0 6,0 
Nuorkarja, 18–22 kk 4,4 3,1 4,4 
Nuorkarja, 9–18 kk 3,7 2,6 3,7 
Nuorkarja, 6–9 kk 3,0 2,0 3,0 
Vasikka, 3–6 kk 2,3 1,1 2,5 
Vasikka 1–3 kk 1,8 1,0 2,0 
Lehmien sairas-, hoito- 
ja poikimakarsina 1) 
11,0 7,0 11,0 
Imettäjälehmä ja 2–3 
vasikkaa1) 
16 10 16 
Sairaskarsina, hiehot, 
nuorkarja > 12 kk 2) 
10,0 6,0 10,0 
Sairaskarsina, nuorkarja 
< 12 kk 3) 
6,0 3,4 6,0 
1) lyhimmän sivun pituuden on oltava vähintään 3,0 metriä, 
2) lyhimmän sivun pituuden on oltava vähintään 2,7 metriä 
3) lyhimmän sivun pituuden on oltava vähintään 2,2 metriä 
Taulukko 4.   Lypsykarjapihaton makuuparsimitoitus. Parren kallistus lantakäytävälle 2-5 % 
Eläinten 
ikä, kk 
Makuuparren 
vähimmäisleveys 1), 
mm 
Makuuparren 
vähimmäispituus2), yksi 
parsirivi, mm 
Makuuparret päät 
vastakkain 
vähimmäispituus 3), mm 
> 22 1 200 2 800 2 600 
18–22 1 100 2 600 2 200 
12–18 1 000 2 450 2 100 
6–12 900 2 150 1 950 
2–6 800 1 700 1 600 
1) parsileveys tarkoittaa parrenerottajien vaakasuoraa mittaa keskeltä keskelle mitattuna. 
Parrenerottajan paksuus voi kaventaa parsileveyttä enintään 65 mm. 
2) makuuparren pituus tarkoittaa parren etuosan tilarajoittimen tai seinämän sisäpinnan ja 
parren ja lantakäytävän välisen rajaviivan etäisyyttä toisistaan; 
3) makuuparsi on avoin pääpuolella siten, että lehmän pää voi ulottua vastakkaiseen 
parteen, ruokintapöytään tai lantakäytävään. 
Taulukko 5.   Lypsykarja- ja nuorkarjapihaton vähimmäiskäytävämitoitus, ahtautta lisäävä 
kaluste esim. vesiallas tai karjaharja lisää poikittaisen käytävän leveysvaatimusta 1 200 mm 
ja pitkittäisen käytävän leveysvaatimusta 600 mm. 
Eläinten ikä, kk > 22 
kk 
18–22 
kk 
12–18 
kk 
6–12 
kk 
2–6 
kk 
Lantakäytävä, yhden tai kahden parsirivin 
välillä, mm 
2 600 2 100 1 800 1 500 1 200 
Ruokintapöydän lantakäytävä, 
1–2 makuuparsiriviä, mm 
3 600 3 100 2 800 2 400 2 100 
Ruokintapöydän lantakäytävä, 
3 makuuparsiriviä, +2 rivinen jos parsi 
avautuu ruokintapöytään, mm 
4 000 3 200 3 000 2 600 2 300 
Poikittainen käytävä eläinosaston päädyssä 
tai päätyseinän vieressä, vähintään, mm 1) 
1 800 1 500 1 200 1 200 1 200 
Poikittainen käytävä mm 1) 2 400 2 100 1 800 1 800 1 200 
1) enintään 25 vierekkäistä makuupartta/ poikittainen käytävä 
Taulukko 6.   Pihattojen ruokintapöydän eläinkohtaisen vähimmäisleveysvaatimukset, mm  
Ruokintapöydän 
reunan pituus eläintä 
kohti, mm 
Ruokinta-
aita-aukon 
Ruokintapöydän 
syöttöparsi 
 
Rehun saanti 
aikavälinen tai 
rajoitettu 
Rehun saanti 
jatkuva, ei 
rajoitettu 
vähimmäisleveys leveys, 
mm 
pituus, 
mm 
Lehmät ja 
hiehot 
> 600 kg 
750 400 220 800 1 600–
1 650 
Nuorkarja 500 300 150 600 1 500 
Taulukko 7.   Juomapaikkojen vähimmäismäärät, lehmille 100 mm ja muille nautaeläimille 50 
mm altaan pituutta vastaa yhden eläimen juomapaikkaa 
Lypsylehmät, 
  
Lehmien määrä, 
kpl 
Juomapaikkoja vähintään, 
juomakuppi tai vastaava, kpl 
Juoma-altaan reunapituus vähintään 
100 mm per lehmä, mm 
1–6 1 600 
6–12 2 600–1 200 
12–18 3 1 200–1 800 
18–24 4 1 800–2 400 
jne. 
  
Muut 
nautaeläimet, 
  
Nuorkarjan 
määrä, kpl 
Juomapaikkoja vähintään, 
juomakuppi tai vastaava, kpl 
Juoma-altaan reunapituus 
vähintään 50 mm per eläin, mm 
1–10 1 500 
11–20 2 550–1 000 
21–40 2 1 050–2 000 
41–60 3 2 050–3 000 
61–80 4 3 050–4 000 
jne. 
  
Taulukko 8.   Rakolattian mitoitus 
Eläinten ikä, kk Palkin leveys, vähintään, mm Raon leveys, mm 
Täysikasvuiset >22kk 120 35–40 
18–22 kk 110 35 
12–18 kk 100 35 
6–12 kk 90 30–35 
< 6 kk 70 25–30 
Palkki- ja rakoleveyden poikkeama saa olla enintään 5 millimetriä. Vierekkäisten 
rakolattiapalkkien yläpintakorkeudet saavat poiketa toisistaan enintään 5 millimetriä. 
Taulukko 9.   Lypsykarjarakennusten eläin- ja lypsyosastojen valaistuksen 
vähimmäisvoimakkuus 
Kohde Lux 
[lx] 
Eläintilan yleisvalaistus, 150 1) 
Lypsyasema ja -robotti 250 2) 
Nuoren karjan tila 100 1) 
1) 2 metrin korkeudella 
2) utarekorkeudella
""",
    ),
    (
        "nitraattiasetus_1250_2014_valvontaohje",
        "VNa 1250/2014 (nitraattiasetus) -- valvontaohje-esitys, ei pykalatason perusteksti",
        None,
        """\
Nitraattiasetuksen valvonta
• Vuonna 2020 Ruokaviraston otannoissa  162 tilaa, joista 
45  täydentävien ehtojen valvonnassa (rehut, ilmasto ja 
hyvä maatalous sekä ympäristö)
• Nitraattiasetus osana täydentävien ehtojen valvontaa 27 
tilalla
• Jos normaalissa peltoalavalvonnassa havaitaan 
puutteita nitraattiasetuksen noudattamisessa, 
valvontalaajennus täydentäviin ehtoihin 
• Talviaikaiset asiakirjavalvonnat
• Lisäksi mahdolliset ilmiannot (esim.ymp.viranomainen)
• Seuraamus yleensä 1-5% kaikista EU:n osittain tai 
kokonaan maksamista tuista
Materiaali perustuu julkaisuhetken tietoihin 1
”Nitraattiasetus” (1250/2014) 
• Lantalan tilavuuden riittävyys arvioidaan pääasiassa silmämääräisesti
• Lantalan vähimmäistilavuuden laskemisessa voidaan ottaa huomioon 
viljelijöiden yhteiset lantalat ja pihattojen kuivikepohjat ja laitumelle jäävä 
lanta (suppeita jaloittelualueita ei oteta enää huomioon). 
• Ympärivuotisesti ulkona kasvatettavilla naudoilla voidaan huomioida koko 
ulkona vietetty kausi lantalatilavuusvaatimusta laskettaessa.
• Jos tilalla kertyy varastoitavaa kuivalantaa tai jos tilalla varastoidaan 
kerrallaan enintään 25 m3 kuivalantaa vuodessa, voidaan lanta varastoida 
esim. tiiviillä siirtolavalla joka katetaan peitteellä.
• Tarkastetaan tilakäynnin yhteydessä
• Tarvittaessa pyydetään ympäristöviranomaiselta lausunto lantalan 
riittävyydestä
2
Lannan varastointi: 
•Kuivalantaa, jonka ka-pitoisuus on vähintään 30 %, voidaan 
työteknisestä tai hygieenisestä syystä varastoida aumassa.
Aumauksen periaatteet:
- Alustan muotoilu ja pohjalle 20 cm:n nestettä sitova kerros
- Yhteen aumaan on sijoitettava vähintään yhden hehtaarin 
alalle tai enintään koko lohkolle ja siihen rajautuville lohkoille 
levitettävä määrä. Auman päälle tiivis peite!
- Aumaan varastoitu lanta on levitettävä viimeistään vuoden 
kuluttua auman perustamisesta.
- Paikalle, jolla auma on sijainnut, saa sijoittaa uuden auman 
kahden välivuoden jälkeen
- EI pohjavesialueelle tai tulvanalaiselle alueelle
- Vesistöstä tai valtaojasta vähintään 100m
3
Kuivalannan varastointi poikkeustilanteessa
”Nitraattiasetus” (1250/2014) 
Materiaali perustuu julkaisuhetken tietoihin 4

• Lannan ja orgaanisten lannoitevalmisteiden levittäminen pellolle on
kielletty 1.11. – 31.3. (Maasto/asiakirja)
• Em. ajankohdasta voidaan kuitenkin poiketa marraskuun loppuun 
asti poikkeuksellisen sääolosuhteen vuoksi. Tällaisena pidetään 
tilannetta, jossa pitkään jatkuneiden runsaiden sateiden ja vähäisen 
haihdunnan vuoksi pellon märkyys on estänyt lannan syyslevityksen 
viimeistään lokakuussa. Ilmoitus tästä kunnan 
ympäristöviranomaiselle!
• Pellon pintaan levitetty lanta ja orgaaniset lannoitevalmisteet on 
muokattava maahan vuorokauden sisällä levityksestä (myös 
keväällä). (Maasto)
5
Lannoitteiden käyttö
”Nitraattiasetus” (1250/2014) 

• Kasvipeitteisenä talven yli pidettäville peltolohkoille lantaa ja 
orgaanista lannoitevalmistetta saa syyskuun 15. päivästä 
eteenpäin levittää vain sijoittamalla, ellei kyseessä ole syksyllä 
kylvettävän kasvin kylvöä edeltävä lannan levitys. (Maasto/asiakirja)
• Lannoitus on kielletty viisi metriä lähempänä vesistöä. Seuraavan 
viiden metrin vyöhykkeellä vesistöstä lannan ja orgaanisten 
lannoitevalmisteiden pintalevitys on kielletty, ellei peltoa muokata 
vuorokauden kuluessa levityksestä. (Maasto)
• Peltolohkon osilla, joiden kaltevuus on vähintään 15 prosenttia, 
lietelannan, virtsan ja nestemäisten orgaanisten 
lannoitevalmisteiden levittäminen muulla tavoin kuin sijoittamalla 
on aina kielletty. Kalteville peltolohkon osille levitettävät muut lannat 
ja orgaaniset lannoitevalmisteet on muokattava maahan kahdentoista 
tunnin sisällä levityksestä. (Maasto)
6
Lannoitteiden käyttö
”Nitraattiasetus” (1250/2014) 
• Liukoisen typen vuotuiset enimmäiskäyttömäärät (kg/ha) kasvikohtaisia
• Jos liukoisen typen lannoitusmäärä ylittää 150 kg/ha vuodessa, määrä 
on jaettava vähintään kahteen erään, joiden levittämisen välisen ajan on 
oltava vähintään kaksi viikkoa.
• Tuotantoeläinten lannassa ja lantaa sisältävissä orgaanisissa 
lannoitevalmisteissa vuosittain levitettävä kokonaistypen määrä saa olla 
enintään 170 kg/ha.
• Syyskuun alusta alkaen tuotantoeläinten lannassa ja orgaanisissa 
lannoitevalmisteissa levitettävän liukoisen typen määrä saa olla enintään 
35 kg/ha.
7
Typpilannoitemäärät (asiakirjavalvonnassa)
”Nitraattiasetus” (1250/2014) 
• Lanta-analyysi on otettava 5 vuoden välein
- Liukoinen typpi
- Kokonaistyppi
- Kokonaisfosfori
• Lanta-analyysivelvoite koskee tiloja, joilla syntyy ja/tai joka käyttää 
lantaa enemmän kuin 25 m3 vuodessa
• Lannoitus suunnitellaan joko lanta-analyysin tai taulukkoarvojen 
perusteella
Valvotaan viimeistään talvella asiakirjavalvonnassa!
8
Lannan ravinnepitoisuuden määrittäminen:
”Nitraattiasetus” (1250/2014)

• Kirjanpitovelvollisuutta täsmennetään:
- Peltojen ravinnelisäykseen käytetyn lannan ja orgaanisten 
lannoitevalmisteiden ja typpilannoitteiden määrä sekä niiden 
sisältämä liukoinen typpi ja kokonaistyppi
- Satotasot
- Lannan ja orgaanisten lannoitevalmisteiden levitysajankohta
M 9
Kirjanpitovelvollisuus (valvotaan talvella)
”Nitraattiasetus” (1250/2014)

Kiitos
10
""",
    ),
    (
        "vesilaki_587_2011_kalatalousvelvoite",
        "Vesilaki 587/2011, 3 luku 11-15 pykala -- Kalatalousvelvoite ja kalatalousmaksu",
        None,
        """\
VESILAKI 587/2011, 3 LUKU

=== 11-15 §: Kalatalousvelvoite ja kalatalousmaksu ===

lain mukaisessa seurannassa ja vesienhoitosuunnitelman laadinnassa.
11 §
Tarkkailuvelvoite
Yhteistarkkailua tai tarkkailusuunnitelman hyväksymistä taikka näiden muuttamista koskeva
päätös on tehtävä noudattaen soveltuvin osin, mitä hallintolaissa säädetään, jollei päätöstä
tehdä lupaa myönnettäessä tai muutettaessa. Päätös annetaan tiedoksi ja siitä tiedotetaan
noudattaen, mitä 11 luvun 22 §:ssä säädetään. Päätöksestä tehtävästä oikaisuvaatimuksesta
säädetään 15 luvun 1 §:n 3 momentissa.
Valtaväylään tai yleiseen kulkuväylään vaikuttava hanke on lupapäätöksessä määrättävä
toteutettavaksi niin, että liikennettä voidaan vesistössä harjoittaa edelleen ilman
huomattavaa haittaa.
Jos hanke toteutetaan sellaisessa vesistössä, jolla on merkitystä uiton kannalta, hankkeesta
vastaava on lupapäätöksessä velvoitettava tekemään tarvittavat laitteet ja rakennelmat
puutavaran kulun turvaamiseksi sekä suojaamaan hankkeeseen liittyvät rakennelmat
uitosta aiheutuvilta vahingoilta.
Jos hankkeen seurauksena on vesialueella, jäällä tai rannalla yleistä tai yksityistä käyttöä
palvelevan kulkuyhteyden katkeaminen tai huomattava huonontuminen, hankkeesta
vastaava on lupapäätöksessä velvoitettava tekemään tie tai vastaamaan sen tekemisen
kustannuksista taikka ryhtymään muihin toimenpiteisiin kohtuulliset vaatimukset täyttävän
kulkumahdollisuuden järjestämiseksi sen tarvitsijoille.
Jos vesitaloushankkeesta aiheutuu kalakannoille tai kalastukselle vahinkoa, hankkeesta
vastaava on velvoitettava ryhtymään toimenpiteisiin vahinkojen ehkäisemiseksi tai
vähentämiseksi (kalatalousvelvoite) taikka määrättävä maksamaan tällaisten toimenpiteiden
kohtuullisia kustannuksia vastaava maksu kalataloustehtäviä hoitavalle
elinvoimakeskukselle (kalatalousmaksu). (27.6.2025/572)
Kalatalousvelvoitetta, kalatalousmaksua tai näiden yhdistelmää määrättäessä on otettava
huomioon hankkeen ja sen vaikutusten laatu, muut haitta-alueella toteutettavat
hoitotoimenpiteet ja kalastuksen järjestely. Kalatalousvelvoitteen toimenpiteiden
suorittamisesta ei saa aiheutua niillä saavutettavaan hyötyyn verrattuna hankkeesta
vastaavalle kohtuuttomia kustannuksia.
Kalatalousvelvoite voi olla kalatie, kalataloudellinen kunnostustoimenpide, istutus tai muu
kalataloudellinen hoitotoimenpide taikka näiden yhdistelmä. Kalatalousvelvoitteeseen
voidaan tarvittaessa sisällyttää toimenpiteiden tuloksellisuuden tarkkailu sillä vesialueella,
johon hankkeen vahingollinen vaikutus ulottuu.
12 §(12.4.2019/505)
Tarkkailuvelvoitteen määrääminen
13 §
Kulkuyhteydet
14 §
Kalatalousvelvoite ja kalatalousmaksu
Kalatalousmaksu käytetään 1 momentissa tarkoitettujen toimenpiteiden suunnitteluun ja
toteuttamiseen sekä niiden tuloksellisuuden seurantaan sillä vesialueella, johon hankkeen
vahingollinen vaikutus ulottuu. Lupa- ja valvontavirasto voi antaa kalataloustehtäviä
hoitavalle elinvoimakeskukselle määräyksiä maksun käytöstä.(27.6.2025/572)
Hankkeesta vastaavan on laadittava yksityiskohtainen suunnitelma luvassa määrätyn
kalatalousvelvoitteen toteuttamiseksi (kalatalousvelvoitteen toteuttamissuunnitelma).
Kalataloustehtäviä hoitava elinvoimakeskus hyväksyy kalatalousvelvoitteen
toteuttamissuunnitelman. Suunnitelman laatiminen ei kuitenkaan ole tarpeen, jos
kalatalousvelvoite on vähäinen ja sen sisällöstä määrätään yksityiskohtaisesti luvassa.
Kalataloustehtäviä hoitava elinvoimakeskus vahvistaa suunnitelman, jossa yksilöidään
kalatalousmaksulla tehtävät toimenpiteet (kalatalousmaksun käyttösuunnitelma). Saman
vesialueen haittojen ehkäisemiseksi määrättyjen kalatalousmaksujen käytöstä voidaan
laatia yhteinen suunnitelma.
Päätös kalatalousvelvoitteen toteuttamissuunnitelman hyväksymisestä ja kalatalousmaksun
käyttösuunnitelman vahvistamisesta on tehtävä noudattaen, mitä hallintolaissa säädetään.
""",
    ),
    (
        "patoturvallisuuslaki_494_2009",
        "Patoturvallisuuslaki 494/2009 (Dam Safety Act, unofficial MMM translation)",
        None,
        """\
Ministry of Agriculture and Forestry, Finland 
NB: Unofficial translation; legally binding texts are those in Finnish and Swedish. 
 
 
 
Dam Safety Act  
494/2009 
(amendments up to 1511/2009 included) 
 
 
Chapter 1 – General provisions    
 
Section 1 – Objective  
(1) The objective of this Act is to ensure safety i n the construction, maintenance and 
 
operation of a dam and reduce the hazard that may be caused by a dam. 
 
Section 2 – Scope of application  
(1) This Act applies to dams and the structures and  equipment which belong to these 
independent of the material of which the dam is constructed or how the dam has been 
constructed or the substance impounded by the dam. 
(2) The provisions concerning a dam laid down in th is Act also apply to flood embankments. 
(3) This Act does not apply to sluice gate structur es of canals. 
 
Section 3 – Relationship with other legislation 
(1) In addition to this Act, the provisions of the Water Act (264/1961) and under it apply to 
construction in watercourses. 
(2) In addition to this Act, the provisions of the Environmental Protection Act (86/2000 and 
under it concerning the prevention of environmental pollution and the provisions of the 
Waste Act (1072/1965) and under it on preventing and combating the risk to health and 
the environment arising from wastes apply to waste dams. 
(3) As regards mine safety, the provisions of the M ining Act (503/1965) and under it also 
apply. 
(4)  In addition to this Act, the provisions of the  Land Use and Building Act (132/1999) 
concerning the permits required for building activities, structures and other action apply 
to dams. 
(5) The provisions on rescue service arrangements a re laid down in the Rescue Act 
(468/2003). 
(6)  The provisions of this Act shall be taken into  account when making an official decision 
on the construction and use of a dam under the Water Act, Environmental Protection Act 
and Land Use and Building Act.  
 
 Section 4 – Definitions  
(1) In this Act: 
1)  dam  means a structure such as a wall or embankment the purpose of which it is to  
permanently or temporarily prevent the spread of a liquid or substance that behaves 
like a liquid impounded by the dam or to regulate the surface level of the impounded 
substance; 
2)  watercourse dam  means a dam in a watercourse; 
3)  waste dam  means a dam for impounding substances that are harmful or hazardous to 
health or the environment; 
4)  flood embankment  means a structure the purpose of which is to prevent the spread of 
water at times when the water level of a watercourse or sea level is unusually high; 
5)  owner of a dam  means the owner or manager of the dam or one whose task it is to see 
to the design, construction, operation and maintenance of the dam. 
 
 Section 5 – Authorities (1511/2009)  
(1) The Ministry of Agriculture and Forestry is res ponsible for the general steering, follow-
up and development of activities under this Act. 
(2) The Centre for Economic Development, Transport and the Environment which is 
competent in dam safety matters functions as the dam safety authority referred to in this 
Act. 
(3)  The Ministry of Agriculture and Forestry may o rder that a Centre for Economic 
Development, Transport and the Environment functions as the dam safety authority 
within the territory of another Centre for Economic Development, Transport and the 
Environment. 
 
Section 6 –
 Competence requirements  
(1) A person who prepares the plan concerning the c onstruction of a dam and a person who 
is responsible for the  operation, monitoring and inspections of the dam must possess 
sufficient expertise in dam safety matters, taking account of the type of the dam and the 
hazard it may cause. Further provisions on the  competence requirements are issued by 
Government Decree.   
 
Chapter 2 – Planning, design and construction of a dam  
 
Section 7 – General obligation  
(1) A dam must be designed and constructed so that its use does not cause any danger to 
safety. 
(2) Further provisions on the hydrological dimensio ning and technical safety requirements 
for the construction of a dam are issued by Government Decree. 
 
Section 8 – Planning and design of a dam  
(1) The plan and design prepared for constructing a  dam must show how the dam safety 
requirements under this Act have been taken into account. 
 
Section 9 – Dam safety studies in the case of permit  
(1) In a permit application concerning the construc tion of a dam under another Act the owner 
of the dam must describe sufficiently the dam hazard and its impact on the dam 
dimensioning and design criteria. 
(2) When resolving a matter referred to in section 3(6) concerning the construction of a dam 
the authority shall request a statement from the dam safety authority concerning the 
fulfilment of the dam safety requirements laid down in this Act. 
 (3) In the statement the dam safety authority shal l, where necessary, present an estimation of 
the design criteria from the dam safety perspective. 
 
Chapter 3 – Classification of a dam and dam safety documents  
 
Section 10 – Classification obligation 
(1)  Before bringing into operation the dam must be  classified and a dam break hazard 
analysis and monitoring programme must be approved for it as laid down in this Chapter. 
 
Section 11 – Classification of a dam 
(1) Based on the hazard 
 the dam is placed in one of the following classes: 
1)  Class 1 dam, which in the event of an accident causes danger to human life and health 
or considerable danger to the environment or property; 
2)  Class 2 dam, which in the event of an accident may cause danger  to health or greater 
than minor danger to the environment or property; 
3)  Class 3 dam, which in the event of an accident may cause only a minor danger. 
 (2) Classification need not be made if the dam saf ety authority considers that the dam does 
not cause any danger. However, the provisions of section 15 on the maintenance of a 
dam, section 16 on the operation  of a dam, section 24 on preventing accidents and 
Chapter 6 on the control of these provisions apply to such a dam. 
 
Section 12 – Dam break hazard analysis and emergency action plan 
(1) To establish the hazard caused by a dam, the ow ner of a class 1 dam must prepare a more 
detailed analysis than that referred to in section 9 of the dam hazard to humans and 
property as well as the environment ( dam break hazard analysis ). 
(2) The dam safety authority may decide that the da m break hazard analysis must also be 
prepared on a dam other than class 1 dam if this is necessary for the classification of the 
dam or assessment of the need to change the class. 
(3) The owner of dam must prepare and regularly upd ate a plan of measures in case of 
accidents and disturbances concerning a class 1 dam (emergency action plan of a dam ). 
(4) Further provisions on preparing and updating of  a dam break hazard analysis and 
emergency action plan of a dam are issued by Government Decree. 
 
Section 13 – Monitoring programme  
(1) The owner of a dam must prepare a programme con cerning a classified dam on the 
monitoring of factors which may impact on dam safety when the dam is brought into 
operation and during 
 operation (monitoring programme ). 
(2) A specific monitoring programme is, however, no t needed if similar factors are being 
monitored under other law in a way that is approved by the dam safety authority. 
(3) Further provisions on the preparation and conte nt of the monitoring programme are 
issued by Government Decree. 
 
Section 14 – Classification decision and approval of documents  
(1) The dam safety authority makes a decision on th e classification of a dam and approval of 
documents referred to in sections 12(1) and 13. 
(2) The owner of a dam must submit the explanatory note needed for the classification 
decision and monitoring programme as well as, where necessary, a dam break hazard 
analysis and emergency action plan of a dam 
 to the dam safety authority well before the 
dam is to be  brought into operation. 
(3) Before making the classification decisions and approval of documents referred to in 
subsection 2 the dam safety authority must give an opportunity to be heard to the owner 
of the dam and the rescue authority of the region concerned. 
(4) The decision shall be delivered to the owner of  the dam,  regional rescue authority and 
municipalities of the area affected by the dam. 
 
Chapter 4 – Maintenance, operation and monitoring of a dam  
 
Section 15 –Maintenance obligation  
(1) The owner of a dam is obligated to keep the dam  in such a condition that it functions as 
intended and is safe. 
  
Section 16 – Operation of a dam   
(1) A dam shall be operated 
 in such a way that it causes no danger to human life and health. 
(2) Sufficient safety arrangements shall be in plac e for class 1 and 2 dams to ensure the 
safety of the operation of the dam. Further provisions on the safety arrangements are 
issued by Government Decree. 
 
Section 17 – Monitoring  
(1) The owner of a dam must organise the monitoring  of the condition and functioning of a 
classified dam in accordance with the monitoring programme. 
 
Section 18 – Annual inspection 
(1) The owner of a dam shall inspect the condition and safety of a class1 and 2 dam at least 
once a year. The owner of a dam must notify the written report prepared on the 
inspection of a class 1 dam to the dam safety authority.  
 
Section 19 – Periodic inspection   
(1) The owner of a dam must organise a periodic ins pection of class 1-3 dams at least every 
 five years and, where necessary, more frequently, to which the dam safety authority and 
 rescue authority has the right to participate. 
(2) A summary of the dam monitoring data from the p ast five years and a preliminary 
assessment of the condition of the dam by an expert who fulfils the competence 
requirements laid down in section 6 must be presented to the dam safety authority in 
good time before the inspection. 
 (3) In the periodic inspection changes in the cond itions of the dam and factors which impact 
on its safety are studied, with due account for the changes in land use and weather and 
hydrological conditions. If in the periodic inspection it cannot be established with 
sufficient certainty that the dam fulfils the safety requirements set for it, the owner of the 
dam must prepare a thorough study of the condition of the dam or its part ( condition 
study ).  
(4)  The owner of a dam must notify the written rep ort prepared on the inspection of a class 1 
and 2 dam to the dam safety authority. 
  
Section 20 – Updating a dam break hazard analysis 
(1) Based on a periodic inspection the dam safety a uthority may order the owner of a dam to 
update a dam break hazard analysis prepared for the dam. 
(2) The owner of a dam must deliver the updated  dam break hazard analysis 
 to the dam 
safety authority for approval in connection with the periodic inspection or separately.  
(3) The dam safety authority shall notify the decis ion on the approval of an updated dam 
break hazard analysis to the owner of the dam, rescue authority of the region concerned 
and municipalities of the area affected by the dam. 
 
Section 21 – Change of class  
(1) The class of a dam may be changed by a decision  of the dam safety authority if 
 on the 
 grounds detected in the inspection of the dam or o therwise the dam hazard can be 
 considered to have changed in an essential way due  to a change in circumstances.     
(2) The provisions of sections 11 and 14 on the cla sses and classification decision apply to a 
decision concerning the change of class.  
(3) The provisions of subsections 1 and 2 on the ch ange of class also apply to a dam which 
has not been classified before by virtue of this Act. The owner of such a dam must 
deliver a report needed for the classification to the dam safety authority upon request. 
 
Section 22 – Alteration and repair works 
(1) In addition to the provisions on the repair and  alteration works of a dam laid down in 
other law, the provisions of Chapter 2 on the planning, design and construction of a dam 
and Chapter 3 on the classification and dam safety documents apply, as appropriate, to 
alteration and repair works which significantly impact on the structures of the dam or are 
otherwise significant as regards dam safety. Such alteration and repair works must be 
notified to the dam safety authority before they are implemented. 
 
Section 23 – Dam decommissioning   
(1) A dam is recorded as removed from service to th e information system of the dam safety 
authority when it is established in the inspection that the dam structure has been pulled 
down or the dam has been decommissioned in such a way that it can no longer cause 
hazard referred to in this Act. The inspection is performed in the presence of the dam 
safety authority after the obligations relating to pulling down a dam structure or dam 
decommissioning under other law have been fulfilled. The obligations under this Act 
cease to be applicable when the dam has been recorded as removed from service. 
 
Chapter 5 – Preparing for dam accidents and action in the event of accidents 
 
Section 24 – Preventing accidents  
 (1) The owner of a dam must, with due consideratio n of the dam hazard, take the necessary 
actions to prevent a dam accident and to limit the damages caused by an accident.  
 
Section 25 – Rescue service plans 
(1) Provisions on rescue service are laid down in s ection 9 of the Rescue Act. The dam 
safety authority delivers the information in its possession necessary for preparing the 
rescue service plans as requested by the rescue authority. 
 
  
Section 26 – Rescue activity 
(1) Provisions on rescue activity are laid down in the Rescue Act. The owner of a dam and 
dam safety authority must assist the head of the rescue activity in performing rescue 
activity. In addition, the dam safety authority participates, where necessary, in the work 
of the steering group referred to in section 44(3) of the Rescue Act. 
 
Section 27 – Emergency call and notice of an exceptional situation as regards safety  
(1) Provisions on an emergency call to the Emergenc y Response Centre are laid down in 
section 28 of the Rescue Act. The owner of the dam must notify the emergency call made 
to the dam safety authority without delay. 
(2) The owner of the dam must give notice concernin g an exceptional situation as regards 
dam safety which has occurred at the dam other than those referred to in subsection 1 to 
the dam safety authority without delay. The notice 
 must describe the situation and give 
the necessary accounts for control measures to the dam safety authority. Where necessary, 
the dam safety authority delivers the notice to the regional rescue authority. 
 
Chapter 6 – Control and coercive measures  
 
Section 28 – Communication  
(1) In addition to provisions laid down in the Act on the Openness of Government Activities 
(621/1999), the dam safety authority shall communicate and keep available information 
on the dam hazard. 
 
Section 29 – Right of inspection 
(1)  The dam safety authority has the right to perf orm the necessary inspections at the dam to 
control the compliance with this Act and provisions issued under it. 
 
Section 30 – Remedying an infringement or neglect 
 
(1)  As far as the infringement does not concern ot her law as well, the dam safety authority 
may prohibit one who violates this Act or provisions issued under it from continuing or 
repeating an action which violates the provision or order that the obligation shall be 
fulfilled. 
(2) Before issuing a prohibition or order the dam s afety authority shall, as far as possible, 
negotiate with the party who has violated this Act or provisions under it. 
 
Section 31 – Order to remedy or rectify 
(1)  In addition to the provisions of section 30, t he dam safety authority may order that 
rectifying, remedial or communication measures are to be implemented within the time 
and in a manner determined by the dam safety authority if an immediate danger related to 
the dam can be efficiently prevented or the magnitude of the danger essentially reduced 
through such a measure. 
 
Section 32– Penalty payment and threat of interruption and of having action taken at the 
defaulter's expense 
(1) The dam safety authority may reinforce a prohib ition or order it has issued by a penalty 
payment or threat of taking the neglected action at the defaulter's expense or that the 
activity is interrupted. 
(2)  Otherwise the provisions of the Penalty Paymen t Act (1113/1990) apply to the penalty 
payment and threat of interruption and of having action taken at the defaulter's expense. 
 
Chapter 7 – Miscellaneous provisions  
 
Section 33 – Information systems  
(1) The Finnish Environment Institute maintains an information system for control of dams 
under this Act. (1511/2009)  
(2) The owner of a dam must provide his or her cont act information and information on the 
staff operating the dam as well as technical information concerning the dam laid down by 
Government Decree to the dam safety authority to be entered to the information system. 
(3)  The dam safety authority and owner of the dam must keep up-to-date printouts from the 
information system for each dam as well as other important documents as regards dam 
safety so that these are readily available in case of disturbances ( dam safety file ). 
(4)  The owner of a dam must notify essential chang es in information referred to in subsection 
2 to the dam safety authority. When the owner of the dam changes the owner who gives 
up the ownership must deliver the dam safety file to the new owner and notify the dam 
safety authority of the change of the owner. 
 
Section 34 – Environmental violation and violation involving public danger 
 
(1) The penalty for impairment of the environment, aggravated impairment of the 
environment, environmental infraction and negligent impairment of the environment 
committed contrary to this Act or provisions or orders issued under it is laid down in 
Chapter 48, sections 1–4 of the Criminal Code (39/1889). 
(2) The penalty for criminal mischief, aggravated c riminal mischief, negligent endangerment 
or gross negligent endangerment committed by creating a water flood is laid down in 
Chapter 34, sections 1, 3, 7 and 8 of the Criminal Code. 
 
Section 35 – Dam safety offence 
 
(1) One who intentionally or through negligence 
1)  neglects the preparation of  a dam break hazard analysis referred to in section 12(1), 
emergency action plan of a dam referred to in section 12(3) or monitoring programme 
referred to in section 13(1) or the delivery of these documents or an explanatory note 
referred to in section 14(2) to the dam safety authority, 
2)  brings a dam into operation  contrary to section 10 before the classification of the dam 
or approval of dam safety documents, 
3)  operates the dam contrary to section 16(1) or neglects the technical safety 
arrangement referred to in section 16(2), 
4)  neglects the maintenance of a dam referred to in section 15, monitoring referred to in 
section 17, annual inspection referred to in section 18, organisation of a periodic 
inspection referred to in section 19 or the updating of  a dam break hazard analysis  
and its delivery to the dam safety authority under section 20, 
5)  neglects the obligation  to take the necessary action  to prevent a dam accident and to 
limit the damage caused by an accident laid down in section 24, 
6)  neglects giving  a notice referred to in section 27(2) or the delivery of information 
referred to in section 33(2) to the dam safety authority, or 
7)  undertakes repair or alteration works of a dam contrary to section 22, 
shall be sentenced to a fine for a dam safety offence, unless a more severe penalty is 
laid down elsewhere in the law. 
 
Section 36 – Entry into force and transitional provisions 
(1)  This Act enters into force on 1 October 2009. 
(2) This Act repeals the Dam Safety Act of 1 June 1 984 (413/1984), as amended. 
(3) The Ministry of Agriculture and Forestry decide s on the placement of dams constructed 
before the entry into force of this Act into classes referred to in section 11. 
(4) Periodic inspection referred to in section 19 s hall be performed on dams constructed 
before the entry into force of this Act within five years from the previous periodic 
inspection under the safety monitoring programme to which the dam safety authority has 
participated, but no more than five years from the entry into force of this Act. Documents 
referred to in sections 12 and 13 above shall be delivered to the dam safety authority for 
approval in the first periodic inspection organised after the entry into force of this Act. 
(5) If in other law reference is made to the Dam Sa fety Act in force upon the entry into force 
of this Act, this Act is applied instead.
""",
    ),
]


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        start += CHUNK_CHARS - OVERLAP
    return chunks


def ingest(dry_run: bool = False) -> None:
    from sentence_transformers import SentenceTransformer
    import chromadb

    if not DB_DIR.exists():
        print(f"[ingest_maatalous_vesivoima] ERROR: {DB_DIR} missing. Run build_index.py first.")
        sys.exit(1)

    print(f"[ingest_maatalous_vesivoima] Connecting to ChromaDB: {DB_DIR}")
    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    existing_ids: set[str] = set(col.get()["ids"])
    print(f"[ingest_maatalous_vesivoima] Existing chunks: {len(existing_ids)}")
    print()

    verified_today = time.strftime("%Y-%m-%d", time.gmtime())
    grand_new = 0

    for doc_id, source_label, url, text in DOCS:
        chunks = _chunk(text)
        new_docs:  list[str]  = []
        new_ids:   list[str]  = []
        new_metas: list[dict] = []

        for i, chunk in enumerate(chunks):
            id_ = f"maatalous_vesivoima_inline__{doc_id}__{i}"
            if id_ in existing_ids:
                continue
            new_docs.append(chunk)
            new_ids.append(id_)
            meta = {
                "country":       "FI",
                "lang":          "fi",
                "source":        doc_id,
                "source_type":   "manual",
                "last_verified": verified_today,
                # hanketyyppi_tag deliberately NOT set here -- resolved
                # exclusively via source_policy.py's SOURCE_HANKETYYPPI_TAG.
            }
            if url:
                meta["url"] = url
            new_metas.append(meta)

        print(f"[{doc_id}] {source_label}")
        print(f"  Chunks: {len(chunks)} total, {len(new_docs)} new")

        if not new_docs:
            print("  -> Nothing new to add\n")
            continue

        if dry_run:
            print(f"  DRY-RUN: would add {len(new_docs)} chunks\n")
            grand_new += len(new_docs)
            continue

        for i in range(0, len(new_docs), BATCH):
            b = slice(i, i + BATCH)
            embs = model.encode(new_docs[b], show_progress_bar=False).tolist()
            col.add(
                documents=new_docs[b],
                embeddings=embs,
                ids=new_ids[b],
                metadatas=new_metas[b],
            )
        existing_ids.update(new_ids)
        grand_new += len(new_docs)
        print(f"  Added {len(new_docs)} chunks\n")

    print(f"{'-'*55}")
    print("Summary (maatalous/vesivoima ingest):")
    print(f"  New chunks added: {grand_new}")
    print(f"  Total index size: {col.count()}")
    print(f"{'-'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index maatalous/vesivoima Priority-2 sources into ChromaDB")
    parser.add_argument("--dry-run", action="store_true", help="Show chunk counts, do not write")
    args = parser.parse_args()
    ingest(dry_run=args.dry_run)
