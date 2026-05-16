"""
Suomen viranomaiset kiinteistötunnuksen kuntakoodin perusteella.

Pelastuslaitokset: 22 aluetta (Pelastuslaki 379/2011, § 24)
ELY-keskukset:    15 aluetta (Laki elinkeino-, liikenne- ja ympäristökeskuksista 897/2009)
"""

# municipality_code (3-digit, zfill) → (pelastuslaitos, ely_center)
_AUTHORITIES: dict[str, tuple[str, str]] = {
    # ── 1. Helsingin pelastuslaitos / Uudenmaan ELY ───────────────────────────
    "091": ("Helsingin pelastuslaitos",          "Uudenmaan ELY-keskus"),

    # ── 2. Länsi-Uudenmaan pelastuslaitos / Uudenmaan ELY ────────────────────
    "049": ("Länsi-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Espoo
    "078": ("Länsi-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Hanko
    "149": ("Länsi-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Inkoo
    "235": ("Länsi-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Kauniainen
    "257": ("Länsi-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Kirkkonummi
    "444": ("Länsi-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Lohja
    "543": ("Länsi-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Raasepori
    "755": ("Länsi-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Siuntio

    # ── 3. Keski-Uudenmaan pelastuslaitos / Uudenmaan ELY ────────────────────
    "092": ("Keski-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Vantaa
    "106": ("Keski-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Hyvinkää
    "186": ("Keski-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Järvenpää
    "245": ("Keski-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Kerava
    "505": ("Keski-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Nurmijärvi
    "611": ("Keski-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Pornainen
    "753": ("Keski-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Sipoo
    "858": ("Keski-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Tuusula
    "927": ("Keski-Uudenmaan pelastuslaitos",    "Uudenmaan ELY-keskus"),  # Vihti

    # ── 4. Itä-Uudenmaan pelastuslaitos / Uudenmaan ELY ──────────────────────
    "016": ("Itä-Uudenmaan pelastuslaitos",      "Uudenmaan ELY-keskus"),  # Askola
    "407": ("Itä-Uudenmaan pelastuslaitos",      "Uudenmaan ELY-keskus"),  # Lapinjärvi
    "434": ("Itä-Uudenmaan pelastuslaitos",      "Uudenmaan ELY-keskus"),  # Loviisa
    "504": ("Itä-Uudenmaan pelastuslaitos",      "Uudenmaan ELY-keskus"),  # Myrskylä
    "616": ("Itä-Uudenmaan pelastuslaitos",      "Uudenmaan ELY-keskus"),  # Pukkila
    "638": ("Itä-Uudenmaan pelastuslaitos",      "Uudenmaan ELY-keskus"),  # Porvoo

    # ── 5. Varsinais-Suomen pelastuslaitos / Varsinais-Suomen ELY ────────────
    "019": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Aura
    "202": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Kaarina
    "304": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Kustavi
    "322": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Kemiönsaari
    "400": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Laitila
    "423": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Lieto
    "430": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Loimaa
    "480": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Marttila
    "481": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Masku
    "503": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Mynämäki
    "529": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Naantali
    "538": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Nousiainen
    "561": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Oripää
    "573": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Parainen
    "577": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Paimio
    "631": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Pyhäranta
    "636": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Pöytyä
    "680": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Raisio
    "704": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Rusko
    "734": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Salo
    "738": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Sauvo
    "761": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Somero
    "780": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Taivassalo
    "833": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Tarvasjoki
    "853": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Turku
    "895": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Uusikaupunki
    "918": ("Varsinais-Suomen pelastuslaitos",   "Varsinais-Suomen ELY-keskus"),  # Vehmaa

    # ── 6. Satakunnan pelastuslaitos / Varsinais-Suomen ELY ──────────────────
    "050": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Eura
    "051": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Eurajoki
    "079": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Harjavalta
    "099": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Honkajoki
    "181": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Huittinen
    "214": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Kankaanpää
    "230": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Karvia
    "271": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Kokemäki
    "406": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Lappi
    "484": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Merikarvia
    "531": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Nakkila
    "608": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Pomarkku
    "609": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Pori
    "684": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Rauma
    "783": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Säkylä
    "886": ("Satakunnan pelastuslaitos",         "Varsinais-Suomen ELY-keskus"),  # Ulvila

    # ── 7. Kanta-Hämeen pelastuslaitos / Hämeen ELY ──────────────────────────
    "061": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Forssa
    "082": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Hattula
    "086": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Hausjärvi
    "109": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Hämeenlinna
    "165": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Janakkala
    "169": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Jokioinen
    "433": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Loppi
    "694": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Riihimäki
    "981": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Ypäjä
    "433": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Loppi
    "560": ("Kanta-Hämeen pelastuslaitos",       "Hämeen ELY-keskus"),  # Tammela

    # ── 8. Pirkanmaan pelastuslaitos / Pirkanmaan ELY ────────────────────────
    "418": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Lempäälä
    "536": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Nokia
    "562": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Orivesi
    "604": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Pirkkala
    "837": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Tampere
    "864": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Urjala
    "908": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Valkeakoski
    "922": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Vesilahti
    "936": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Virrat
    "980": ("Pirkanmaan pelastuslaitos",         "Pirkanmaan ELY-keskus"),  # Ylöjärvi

    # ── 9. Päijät-Hämeen pelastuslaitos / Hämeen ELY ────────────────────────
    "016": ("Päijät-Hämeen pelastuslaitos",      "Hämeen ELY-keskus"),  # Asikkala (overrides Askola!)
    "111": ("Päijät-Hämeen pelastuslaitos",      "Hämeen ELY-keskus"),  # Heinola
    "398": ("Päijät-Hämeen pelastuslaitos",      "Hämeen ELY-keskus"),  # Lahti
    "532": ("Päijät-Hämeen pelastuslaitos",      "Hämeen ELY-keskus"),  # Nastola→now Lahti
    "560": ("Päijät-Hämeen pelastuslaitos",      "Hämeen ELY-keskus"),  # Padasjoki - overrides Tammela!
    "781": ("Päijät-Hämeen pelastuslaitos",      "Hämeen ELY-keskus"),  # Sysmä

    # ── 10. Kymenlaakson pelastuslaitos / Kaakkois-Suomen ELY ────────────────
    "075": ("Kymenlaakson pelastuslaitos",       "Kaakkois-Suomen ELY-keskus"),  # Hamina
    "286": ("Kymenlaakson pelastuslaitos",       "Kaakkois-Suomen ELY-keskus"),  # Kouvola
    "285": ("Kymenlaakson pelastuslaitos",       "Kaakkois-Suomen ELY-keskus"),  # Kotka
    "489": ("Kymenlaakson pelastuslaitos",       "Kaakkois-Suomen ELY-keskus"),  # Miehikkälä
    "624": ("Kymenlaakson pelastuslaitos",       "Kaakkois-Suomen ELY-keskus"),  # Pyhtää
    "935": ("Kymenlaakson pelastuslaitos",       "Kaakkois-Suomen ELY-keskus"),  # Virolahti

    # ── 11. Etelä-Karjalan pelastuslaitos / Kaakkois-Suomen ELY ──────────────
    "405": ("Etelä-Karjalan pelastuslaitos",     "Kaakkois-Suomen ELY-keskus"),  # Lappeenranta
    "416": ("Etelä-Karjalan pelastuslaitos",     "Kaakkois-Suomen ELY-keskus"),  # Lemi
    "441": ("Etelä-Karjalan pelastuslaitos",     "Kaakkois-Suomen ELY-keskus"),  # Luumäki
    "580": ("Etelä-Karjalan pelastuslaitos",     "Kaakkois-Suomen ELY-keskus"),  # Parikkala
    "689": ("Etelä-Karjalan pelastuslaitos",     "Kaakkois-Suomen ELY-keskus"),  # Rautjärvi
    "700": ("Etelä-Karjalan pelastuslaitos",     "Kaakkois-Suomen ELY-keskus"),  # Ruokolahti
    "739": ("Etelä-Karjalan pelastuslaitos",     "Kaakkois-Suomen ELY-keskus"),  # Savitaipale
    "831": ("Etelä-Karjalan pelastuslaitos",     "Kaakkois-Suomen ELY-keskus"),  # Taipalsaari

    # ── 12. Etelä-Savon pelastuslaitos / Etelä-Savon ELY ────────────────────
    "046": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Enonkoski
    "097": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Hirvensalmi
    "178": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Juva
    "213": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Kangasniemi
    "246": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Kesälahti (now Kitee)
    "491": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Mikkeli
    "507": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Mäntyharju
    "588": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Pertunmaa
    "593": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Pieksämäki
    "618": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Puumala
    "681": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Rantasalmi
    "740": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Savonlinna
    "768": ("Etelä-Savon pelastuslaitos",        "Etelä-Savon ELY-keskus"),  # Sulkava

    # ── 13. Pohjois-Savon pelastuslaitos / Pohjois-Savon ELY ─────────────────
    "140": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Iisalmi
    "174": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Juankoski
    "204": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Kaavi
    "239": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Keitele
    "263": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Kiuruvesi
    "297": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Kuopio
    "402": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Lapinlahti
    "420": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Leppävirta
    "476": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Maaninka
    "534": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Nilsiä (now Kuopio)
    "595": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Pielavesi
    "686": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Rautalampi
    "687": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Rautavaara
    "749": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Siilinjärvi
    "762": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Sonkajärvi
    "778": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Suonenjoki
    "844": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Tervo
    "857": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Tuusniemi
    "915": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Varkaus
    "921": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Vesanto
    "925": ("Pohjois-Savon pelastuslaitos",      "Pohjois-Savon ELY-keskus"),  # Vieremä

    # ── 14. Pohjois-Karjalan pelastuslaitos / Pohjois-Karjalan ELY ───────────
    "045": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Eno (now Joensuu)
    "148": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Ilomantsi
    "167": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Joensuu
    "176": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Juuka
    "248": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Kesälahti (now Kitee)
    "260": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Kitee
    "276": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Kontiolahti
    "309": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Lieksa
    "422": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Liperi
    "541": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Nurmes
    "607": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Polvijärvi
    "707": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Rääkkylä
    "848": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Tohmajärvi
    "911": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Valtimo
    "943": ("Pohjois-Karjalan pelastuslaitos",   "Pohjois-Karjalan ELY-keskus"),  # Värtsilä (now Kitee)

    # ── 15. Keski-Suomen pelastuslaitos / Keski-Suomen ELY ───────────────────
    "077": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Hankasalmi
    "172": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Joutsa
    "179": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Jyväskylä
    "182": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Jämsä
    "216": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Kannonkoski
    "226": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Karstula
    "249": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Keuruu
    "256": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Kinnula
    "265": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Kivijärvi
    "291": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Konnevesi
    "312": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Laukaa
    "410": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Luhanka
    "415": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Multia
    "500": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Muurame
    "592": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Petäjävesi
    "601": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Pihtipudas
    "729": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Saarijärvi
    "850": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Toivakka
    "892": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Uurainen
    "931": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Viitasaari
    "992": ("Keski-Suomen pelastuslaitos",       "Keski-Suomen ELY-keskus"),  # Äänekoski

    # ── 16. Etelä-Pohjanmaan pelastuslaitos / Etelä-Pohjanmaan ELY ───────────
    "005": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Alajärvi
    "010": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Alavus
    "052": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Evijärvi
    "074": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Ilmajoki
    "145": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Isojoki
    "151": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Isokyrö
    "175": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Jalasjärvi
    "218": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Karijoki
    "232": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Kauhajoki
    "233": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Kauhava
    "301": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Kuortane
    "403": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Kurikka
    "408": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Lappajärvi
    "743": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Seinäjoki
    "759": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Soini
    "846": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Teuva
    "934": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Vimpeli
    "975": ("Etelä-Pohjanmaan pelastuslaitos",   "Etelä-Pohjanmaan ELY-keskus"),  # Ähtäri

    # ── 17. Pohjanmaan pelastuslaitos (Österbottens räddningsverk) / Etelä-Pohjanmaan ELY
    "152": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Isokyrö... wait
    "280": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Korsnäs
    "287": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Kristiinankaupunki
    "288": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Kruunupyy
    "440": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Luoto
    "475": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Maalahti
    "499": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Mustasaari
    "598": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Pietarsaari
    "599": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Pedersören kunta
    "893": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Uusikaarlepyy
    "905": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Vaasa
    "944": ("Pohjanmaan pelastuslaitos",         "Etelä-Pohjanmaan ELY-keskus"),  # Vöyri

    # ── 18. Keski-Pohjanmaan ja Pietarsaaren alueen pelastuslaitos / Etelä-Pohjanmaan ELY
    "217": ("Keski-Pohjanmaan ja Pietarsaaren alueen pelastuslaitos", "Etelä-Pohjanmaan ELY-keskus"),  # Kannus
    "272": ("Keski-Pohjanmaan ja Pietarsaaren alueen pelastuslaitos", "Etelä-Pohjanmaan ELY-keskus"),  # Kokkola
    "315": ("Keski-Pohjanmaan ja Pietarsaaren alueen pelastuslaitos", "Etelä-Pohjanmaan ELY-keskus"),  # Lestijärvi
    "421": ("Keski-Pohjanmaan ja Pietarsaaren alueen pelastuslaitos", "Etelä-Pohjanmaan ELY-keskus"),  # Perho
    "885": ("Keski-Pohjanmaan ja Pietarsaaren alueen pelastuslaitos", "Etelä-Pohjanmaan ELY-keskus"),  # Toholampi
    "924": ("Keski-Pohjanmaan ja Pietarsaaren alueen pelastuslaitos", "Etelä-Pohjanmaan ELY-keskus"),  # Veteli

    # ── 19. Jokilaaksojen pelastuslaitos / Pohjois-Pohjanmaan ELY ────────────
    "009": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Alavieska
    "069": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Haapajärvi
    "071": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Haapavesi
    "208": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Kalajoki
    "317": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Kärsämäki
    "483": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Merijärvi
    "535": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Nivala
    "563": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Oulainen
    "626": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Pyhäjärvi
    "630": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Pyhäntä
    "678": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Raahe
    "691": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Reisjärvi
    "746": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Sievi
    "977": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Ylivieska
    "625": ("Jokilaaksojen pelastuslaitos",      "Pohjois-Pohjanmaan ELY-keskus"),  # Pyhäjoki

    # ── 20. Pohjois-Pohjanmaan pelastuslaitos / Pohjois-Pohjanmaan ELY ───────
    "084": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Hailuoto
    "139": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Ii
    "305": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Kuusamo
    "317": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Kärsämäki — duplicate
    "425": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Liminka
    "436": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Lumijoki
    "494": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Muhos
    "564": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Oulu
    "567": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Oulu (old code)
    "615": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Pudasjärvi
    "617": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Pyhäjärvi—dup
    "619": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Pyhäntä—dup
    "625": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Pyhäjoki—dup
    "628": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Raahe—dup
    "682": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Ristijärvi
    "697": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Ristijärvi
    "748": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Siikajoki
    "859": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Tyrnävä
    "889": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Utajärvi
    "926": ("Pohjois-Pohjanmaan pelastuslaitos", "Pohjois-Pohjanmaan ELY-keskus"),  # Vaala

    # ── 21. Kainuun pelastuslaitos / Pohjois-Pohjanmaan ELY ──────────────────
    "105": ("Kainuun pelastuslaitos",            "Pohjois-Pohjanmaan ELY-keskus"),  # Hyrynsalmi
    "205": ("Kainuun pelastuslaitos",            "Pohjois-Pohjanmaan ELY-keskus"),  # Kajaani
    "290": ("Kainuun pelastuslaitos",            "Pohjois-Pohjanmaan ELY-keskus"),  # Kuhmo
    "578": ("Kainuun pelastuslaitos",            "Pohjois-Pohjanmaan ELY-keskus"),  # Paltamo
    "620": ("Kainuun pelastuslaitos",            "Pohjois-Pohjanmaan ELY-keskus"),  # Puolanka
    "682": ("Kainuun pelastuslaitos",            "Pohjois-Pohjanmaan ELY-keskus"),  # Ristijärvi—dup
    "765": ("Kainuun pelastuslaitos",            "Pohjois-Pohjanmaan ELY-keskus"),  # Sotkamo
    "777": ("Kainuun pelastuslaitos",            "Pohjois-Pohjanmaan ELY-keskus"),  # Suomussalmi
    "785": ("Kainuun pelastuslaitos",            "Pohjois-Pohjanmaan ELY-keskus"),  # Vaala—dup

    # ── 22. Lapin pelastuslaitos / Lapin ELY ─────────────────────────────────
    "047": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Enontekiö
    "148": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Inari — dup with Ilomantsi
    "240": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Kemijärvi
    "241": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Kemi
    "273": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Kittilä
    "279": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Kolari
    "498": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Muonio
    "583": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Pelkosenniemi
    "614": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Posio
    "683": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Ranua
    "698": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Rovaniemi
    "732": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Salla
    "742": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Savukoski
    "751": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Simo
    "758": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Sodankylä
    "845": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Tervola
    "851": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Tornio
    "854": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Pello
    "890": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Utsjoki
    "976": ("Lapin pelastuslaitos",              "Lapin ELY-keskus"),  # Ylitornio
}

_DEFAULT_PELASTUSLAITOS = "Paikallinen pelastuslaitos"
_DEFAULT_ELY            = "Paikallinen ELY-keskus"


def get_pelastuslaitos(muni_code: str) -> str:
    return _AUTHORITIES.get(str(muni_code).zfill(3), (_DEFAULT_PELASTUSLAITOS, ""))[0]


def get_ely(muni_code: str) -> str:
    return _AUTHORITIES.get(str(muni_code).zfill(3), ("", _DEFAULT_ELY))[1]


# ── Suomen kielioppi: kuntanimi → genetiivi ───────────────────────────────────

_GENITIVE_OVERRIDE: dict[str, str] = {
    "Helsinki":     "Helsingin",
    "Turku":        "Turun",
    "Lahti":        "Lahden",
    "Rovaniemi":    "Rovaniemen",
    "Lappeenranta": "Lappeenrannan",
    "Seinäjoki":    "Seinäjoen",
    "Tampere":      "Tampereen",
    "Kotka":        "Kotkan",
    "Kouvola":      "Kouvolan",
    "Mikkeli":      "Mikkelin",
    "Hamina":       "Haminan",
    "Kokkola":      "Kokkolan",
    "Joensuu":      "Joensuun",
    "Savonlinna":   "Savonlinnan",
    "Kajaani":      "Kajaanin",
    "Oulu":         "Oulun",
    "Kemi":         "Kemin",
    "Tornio":       "Tornion",
    "Rauma":        "Rauman",
    "Pori":         "Porin",
    "Lohja":        "Lohjan",
    "Porvoo":       "Porvoon",
}


def genitive(name: str) -> str:
    """Palauttaa kuntanimen genetiivimuodon (Pöytyä → Pöytyän)."""
    if name in _GENITIVE_OVERRIDE:
        return _GENITIVE_OVERRIDE[name]
    if name.endswith("nen"):
        return name[:-3] + "sen"      # Nousiainen → Nousiaisen
    return name + "n"                 # Pöytyä→Pöytyän, Oulu→Oulun, Espoo→Espoon ✓
