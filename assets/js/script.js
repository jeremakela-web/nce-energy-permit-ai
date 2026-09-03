/* ==========================================================
   NCE ENERGY — Vanilla JS
   ========================================================== */
(function () {
  "use strict";

  /* ---------- 1. Anti-copy / rights protection ---------- */
  window.addEventListener("contextmenu", function (e) {
    e.preventDefault();
  });
  // soft-block common save shortcuts
  window.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && ["s", "u", "p"].includes(e.key.toLowerCase())) {
      e.preventDefault();
    }
  });

  /* ---------- 2. i18n dictionary ---------- */
  const I18N = {
    en: {
      "nav.item1": "About",
      "nav.item2": "How it works",
      "nav.item3": "Projects",
      "nav.item4": "Permit AI",

      "cta.openTool": "Open Tool",
      "cta.discover": "See how it works",
      "hero.title1": "Permit documentation for energy projects",
      "hero.title2": "— in minutes, not weeks",
      "hero.sub": "NCE Permit AI helps energy developers and engineering firms generate professional, source-traceable permit documentation with built-in regulatory quality validation.",
      "hero.tag1": "BESS",
      "hero.tag2": "Wind",
      "hero.tag3": "Solar",
      "hero.tag4": "SMR",
      "hero.tag5": "Hydro",
      "hero.tagMore": "20+ Types",

      "hero.statLabel": "LIVE REGULATORY DATA",
      "hero.statText1": "Regulatory chunks indexed",
      "hero.statText2": "Project types",
      "hero.statText3": "Countries",

      "subhero.tag": "Regulatory Intelligence Platform",
      "subhero.title.html": "Built for energy, construction, industrial and infrastructure projects <em>the Nordic and European market</em>",
      "subhero.text": "NCE Permit AI is a regulatory intelligence platform built for the energy sector. It combines AI-assisted drafting, source-traced regulatory content and RAQS quality validation into one structured workflow.",
      "subhero.country": "Finland · Sweden · Denmark · Norway · Poland · Germany",

      "stats.minute":"in minutes",
      "stats.draft":"Draft permit application",
      "stats.types":"Project types",
      "stats.mw":"MW in the BESS pipeline in Finland",
      "stats.cost":"Cost vs. traditional consulting",
      "stats.summaryTitle":"From Idea To Application",
      "stats.summaryText":"Significant time and cost savings compared to traditional permit preparation and consulting workflows",

      "s1.tag": "01 — TRUST LINE",
      "s1.h2.html": "From contract negotiations <em>to building permits</em>",
      "s1.p1": "Every report is grounded in official sources: Finlex, STUK YVL guidelines, EUR-Lex, Fingrid and SYKE. RAQS validation ensures structure, traceability and regulatory compliance before any document is delivered.",
      "s1.cta": "Open permit AI",

      "s2.tag": "02 — HOW IT WORKS",
      "s2.h2.html": "A thousand pages, <em>distilled</em> into one application",
      "s2.step1Title": "Enter project details",
      "s2.step1Text": "Add basic project information, location and project type.",
      "s2.step2Title": "AI retrieves regulatory requirements",
      "s2.step2Text": "NCE Permit AI automatically retrieves relevant legislation, authority requirements and source references for your project type and jurisdiction.",
      "s2.step3Title": "Generate structured draft",
      "s2.step3Text": "A structured permit application draft is generated in PDF format, ready for professional review and submission.",
      "s2.step4Title": "RAQS validates before delivery",
      "s2.step4Text": "Every document goes through the RAQS validation process (Regulatory Assurance Quality System). This automated quality assurance step ensures structural consistency, source traceability, and regulatory compliance before the document reaches the user. This guarantees professional quality and reduces the risk of errors or missing regulatory references.",
      "s2.cta": "Try the Permit AI",
      
      "s3.tag": "03 — SUPPORTED PROJECT TYPES",
      "s3.h2.html": "20+ project types. <em>One development standard</em>",
      //CARDS
      
      //Card 1
      "s3.card1Title": "BESS",
      "s3.card1Meta": "Battery Energy Storage · 50–800 MWh",
      "s3.card1Text": "Grid connection, environmental review and building permitting. Designed for frequency response, energy arbitrage and renewable integration.",
      
      //Card 2
      "s3.card2Title": "Wind — Land",
      "s3.card2Meta": "Onshore Wind · 50–600 MW",
      "s3.card2Text": "Large-scale wind developments across forest and agricultural land. Includes EIA, landowner agreements and transmission planning.",
      
      //Card 3
      "s3.card3Title": "Wind — Sea",
      "s3.card3Meta": "Offshore Wind · 500 MW–2 GW",
      "s3.card3Text": "Maritime permitting, seabed surveys and offshore grid integration. Built for utility-scale renewable generation in Nordic waters.",
      
      //Card 4
      "s3.card4Title": "Solar",
      "s3.card4Meta": "Utility PV · 10–400 MW",
      "s3.card4Text": "Ground-mounted solar parks with optional battery storage. Fast permitting path and scalable deployment model.",
      
      //Card 5
      "s3.card5Title": "SMR",
      "s3.card5Meta": "Small Modular Reactor · 50–300 MW",
      "s3.card5Text": "Advanced nuclear projects for district heating, hydrogen production and long-term baseload electricity supply.",
      
      //Card 6
      "s3.card6Title": "Hydro",
      "s3.card6Meta": "Hydropower · 5–60 MW",
      "s3.card6Text": "Water permit management, environmental assessments and modernisation of existing hydroelectric infrastructure.",
      
      //Card 7
      "s3.card7Title": "Residential",
      "s3.card7Meta": "Housing Development",
      "s3.card7Text": "Apartment complexes, residential districts and mixed-use communities. Building permits, zoning compliance and infrastructure coordination.",
      
      //Card 8
      "s3.card8Title": "Industrial",
      "s3.card8Meta": "Manufacturing & Processing",
      "s3.card8Text": "Factories, production facilities and logistics hubs. Environmental permits, utility connections and site approvals.",
      
      //Card 9
      "s3.card9Title": "Agriculture",
      "s3.card9Meta": "Farm Infrastructure",
      "s3.card9Text": "Barns, storage facilities and agricultural processing buildings. Permitting adapted for rural development projects.",
      
      //Card 10
      "s3.card10Title": "Commercial",
      "s3.card10Meta": "Office & Retail Development",
      "s3.card10Text": "Business parks, office buildings and commercial real estate. Planning approvals, construction permits and utility integration.",
            
      //Card 11
      "s3.card11Title": "Other Project",
      "s3.card11Meta": "Custom Development",
      "s3.card11Text": "Infrastructure, public-sector and special-purpose developments. Flexible permitting workflows tailored to project requirements.",
      
      //Card more
      "s3.cardMoreTitle": "More Project Types",
      "s3.cardMoreMeta": "Infrastructure · Energy · Industrial",
      "s3.cardMoreText": "Data centres, logistics hubs, public infrastructure, commercial developments and sector-specific projects. NCE Permit AI currently supports more than 20 project categories.",


      "s4.tag": "04 — SMR / NEXT-GENERATION NUCLEAR",
      "s4.h2.html": "Small Modular Reactors, <em>European-scaled</em>",
      "s4.p1": "NCE Permit AI supports the permitting process for Small Modular Reactors across Nordic and European jurisdictions, including pre-licensing, site characterisation and safety documentation.",
      "s4.p2": "Designed to manage the regulatory complexity of new nuclear technologies under STUK (Finland), SSM (Sweden), DSA (Norway) and EURATOM frameworks.",

      "s5.tag": "05 — FROM PERMITTING TO OPERATIONS",
      "s5.h2.html": "A single pane of glass for <em>every megawatt</em>",
      "s5.p1": "NCE Permit AI provides structure across the full permitting lifecycle — from initial application through construction and into operations and compliance monitoring. Regulatory requirements, documentation and compliance tracking in a single auditable workflow.",

      "s6.tag": "06 — WHY NCE",
      "s6.h2.html": "Built for permitting. <em>Nothing else</em>",
      "s6.p1": "NCE Permit AI is not a generic AI writing tool. It is built for one purpose: helping energy projects move through permitting faster, with full regulatory traceability and less manual work.",
      "s6.p2": "Your team works from a single structured workflow instead of searching through regulations, managing document versions and rewriting the same content across projects.",

      "s7.tag": "07 — WHO IT IS FOR",
      "s7.p1": "Energy developers · Engineering firms · Permit specialists · Project managers · Regulatory and compliance teams",

      "s8.tag": "08 — CTA / ONBOARDING",
      "s8.h2.html": "In minutes, <em>Professional license application</em>.",
      "s8.p1": "We are currently onboarding the first pilot customers.",
      "s8.p2": "If you are working on BESS, wind, solar, SMR or construction, industrial or infrastructure projects — get in touch to discuss a pilot.",
      "cta.openToolRequest": "Request access",

      "footer.title": "NCE — Native Clean Energy",
      "footer.p1": "Regulatory Intelligence for Energy, Construction and Infrastructure Projects",
      "footer.pp": "Privacy Policy",
      "footer.tag": "Finland · Sweden · Denmark · Norway · Poland · Germany",

      "tool.tag": "Permit AI · v1.4",
      "tool.h2.html": "Generate your <em>permit application</em>.",
      "tool.sub": "Select a project type — the assistant will tailor jurisdictional flow, filings and timeline.",
      "tool.t.wind": "Wind",
      "tool.t.solar": "Solar",
      "tool.t.bess": "BESS",
      "tool.t.smr": "SMR",


"tool.more": "+ MORE...",
"tool.moreLabel": "MORE",

"tool.selector.kicker": "ADDITIONAL PROJECT TYPES",
"tool.selector.title": "Select project type",
"tool.selector.text": "Choose the asset category for permit generation.",
"tool.selector.back": "← Back",

"tool.form.applicant": "Applicant",
"tool.form.business": "Business ID",
"tool.form.address": "Address",
"tool.form.property": "Property ID",
"tool.form.municipality": "Municipality",
"tool.form.power": "Total Power (MW)",
"tool.form.location": "Location / Environmental Impacts",
"tool.form.capacity": "Capacity (MWh)",
"tool.form.phase": "Project Phase",
"tool.form.authority": "Target Authority",
"tool.form.projectId": "Project ID",

"tool.form.preStudy": "Pre-study",
"tool.form.permitting": "Permitting",
"tool.form.construction": "Construction",

"tool.generate": "Generate Permit Application",



"tool.generation.title": "AI GENERATION IN PROGRESS",
"tool.generation.subtitle": "RAG + Claude is analyzing your project and preparing the regulatory draft",

"tool.generation.analyze": "Analyze",
"tool.generation.analyze.sub": "Understanding project",

"tool.generation.retrieve": "Retrieve",
"tool.generation.retrieve.sub": "Searching regulations",

"tool.generation.compare": "Compare",
"tool.generation.compare.sub": "Matching requirements",

"tool.generation.assess": "Assess",
"tool.generation.assess.sub": "Risk & impact",

"tool.generation.recommend": "Recommend",
"tool.generation.recommend.sub": "Building draft",

"tool.generation.lifecycle": "Lifecycle",
"tool.generation.lifecycle.sub": "Phase-lock check",

"tool.generation.waiting": "WAITING",

"tool.generation.liveActivity": "LIVE ACTIVITY",
"tool.generation.starting": "Starting generation...",

"tool.generation.dataSources": "DATA SOURCES",
"tool.generation.regulatoryChunks": "Regulatory chunks",
"tool.generation.countries": "Countries",
"tool.generation.euLayer": "EU layer",
"tool.generation.active": "active",

"tool.generation.status": "Starting generation...",
"tool.generation.jobId": "Job ID:",


"tool.raqs.sectionTitle": "2. RAQS — QUALITY ASSESSMENT",
"tool.raqs.sectionSubtitle": "5 quality criteria with overall score",

"tool.raqs.title": "RAQS QUALITY SCORE",
"tool.raqs.live": "LIVE",

"tool.raqs.citations": "Citations",
"tool.raqs.coverage": "Permit Coverage",
"tool.raqs.uncertainty": "Uncertainty",
"tool.raqs.comprehensiveness": "Comprehensiveness",
"tool.raqs.readiness": "Readiness",

"tool.raqs.warning": "UNCERTAINTY DETECTED",
"tool.raqs.warningText": "Expert verification may be required.",


"tool.result.sectionTitle": "3. RESULT & ACTIONS",
"tool.result.sectionSubtitle": "Generated draft + mandatory expert review",

"tool.result.title": "PERMIT APPLICATION DRAFT",
"tool.result.review": "EXPERT REVIEW",

"tool.result.fileTitle": "Permit Application Draft",
"tool.result.generated": "Generated:",

"tool.result.download": "↓ DOWNLOAD PDF",
"tool.result.summary": "VIEW SUMMARY ◉",

"tool.result.warning": "AI-GENERATED DRAFT",
"tool.result.warningText": "This document requires expert review before official submission.",

"tool.result.markReviewed": "✓ MARK AS REVIEWED",


"tool.project.wind": "Wind",
"tool.project.wind.title": "Wind Power (Onshore)",
"tool.project.wind.meta": "Onshore Wind · 50–600 MW",
"tool.project.wind.description": "Wind Power (onshore) — Land-based wind power project",

"tool.project.solar": "Solar",
"tool.project.solar.title": "Solar Power",
"tool.project.solar.meta": "Utility PV · 10–400 MW",
"tool.project.solar.description": "Solar panel project",

"tool.project.bess": "BESS",
"tool.project.bess.title": "Battery Energy Storage Project",
"tool.project.bess.meta": "50–800 MWh",
"tool.project.bess.description": "Battery Energy Storage Project",

"tool.project.smr": "SMR",
"tool.project.smr.title": "Small Modular Reactor",
"tool.project.smr.meta": "50–300 MW",
"tool.project.smr.description": "Small Modular Reactor project (pre-licensing)",

"tool.project.windsea": "Wind — Sea",
"tool.project.windsea.title": "Offshore Wind",
"tool.project.windsea.meta": "500 MW–2 GW",
"tool.project.windsea.description": "Offshore wind power project",

"tool.project.hydro": "Hydro",
"tool.project.hydro.title": "Hydropower",
"tool.project.hydro.meta": "5–60 MW",
"tool.project.hydro.description": "Hydroelectric project",

"tool.project.residential": "Residential",
"tool.project.residential.title": "Housing Development",
"tool.project.residential.meta": "5–6 MW",
"tool.project.residential.description": "Residential Building — permit application",

"tool.project.industrial": "Industrial",
"tool.project.industrial.title": "Manufacturing & Processing",
"tool.project.industrial.meta": "5–6 MW",
"tool.project.industrial.description": "Industrial Building — permit application",

"tool.project.agriculture": "Agriculture",
"tool.project.agriculture.title": "Farm Infrastructure",
"tool.project.agriculture.meta": "5–6 MW",
"tool.project.agriculture.description": "Agriculture Building — permit application",

"tool.project.commercial": "Commercial",
"tool.project.commercial.title": "Office & Retail Development",
"tool.project.commercial.meta": "5–6 MW",
"tool.project.commercial.description": "Commercial Building — permit application",

"tool.project.other": "Other Project",
"tool.project.other.title": "Custom Development",
"tool.project.other.meta": "5–6 MW",
"tool.project.other.description": "Custom Development Building — permit application",

"tool.dynamic.wind.power": "Total Power (MW)",
"tool.dynamic.wind.turbines": "Turbine Count",

"tool.dynamic.solar.capacity": "Installed Capacity (MWp)",
"tool.dynamic.solar.area": "Area (ha)",

"tool.dynamic.bess.power": "Power (MW)",
"tool.dynamic.bess.capacity": "Capacity (MWh)",
"tool.dynamic.bess.grid": "Grid Connection",

"tool.dynamic.smr.thermalPower": "Thermal Power (MW)",
"tool.dynamic.smr.reactorType": "Reactor Type",

"tool.dynamic.hydro.reactorType": "Water Power Type",

"tool.dynamic.residential.floors": "Floors",

"tool.dynamic.industrial.intendedUse": "Intended Use",
"tool.dynamic.industrial.buildingVolume": "Building Volume (m³)",

"tool.dynamic.select": "Select...",

      "tool.f.name": "Project name",
      "tool.f.jur": "Jurisdiction",
      "tool.f.cap": "Capacity (MW)",
      "tool.f.brief": "Project brief",
      "tool.f.run": "Run permit analysis",
      "tool.back": "← Back to site",

      "modal.title":    "Request Access",
      "modal.sub":      "We are onboarding pilot customers. Fill in your details and we'll be in touch.",
      "modal.company":  "Company",
      "modal.contact":  "Contact person",
      "modal.email":    "Email",
      "modal.phone":    "Phone",
      "modal.optional": "(optional)",
      "modal.desc":     "Description of operations",
      "modal.submit":   "Submit",

      "hero.slides": [
        "Onshore Wind Development",
        "Renewable Infrastructure",
        "Offshore Wind Energy",
        "Battery Energy Storage",
        "Solar Power Generation",
        "Nordic Environment",
        "Community Integration",
        "Project Development",
        "Regulatory Compliance",
        "SMR & Nuclear Energy"
      ],
    },

  fi: {
    "nav.item1": "Tietoa",
    "nav.item2": "Miten se toimii",
    "nav.item3": "Projektit",
    "nav.item4": "Permit AI",

    "cta.openTool": "Avaa työkalu",
    "cta.discover": "Katso miten se toimii",

    "hero.title1": "Lupadokumentaatio energiahankkeille",
    "hero.title2": "— minuuteissa, ei viikoissa",

    "hero.sub": "NCE Permit AI auttaa energiakehittäjiä ja insinööritoimistoja tuottamaan ammattitasoisia, lähdejäljitettäviä lupadokumentteja sisäänrakennetulla sääntelyn laadunvarmistuksella.",

    "hero.tag1": "BESS",
    "hero.tag2": "Tuulivoima",
    "hero.tag3": "Aurinkovoima",
    "hero.tag4": "SMR",
    "hero.tag5": "Vesivoima",
    "hero.tagMore": "Yli 20 tyyppiä",

    "hero.statLabel": "AJANTASAINEN SÄÄNTELYDATA",
    "hero.statText1": "Indeksoituja sääntelylähteitä",
    "hero.statText2": "Projektityyppejä",
    "hero.statText3": "Maita",

    "subhero.tag": "Regulatory Intelligence Platform",
    "subhero.title.html": "Rakennettu energia-, rakennus-, teollisuus- ja infrastruktuurihankkeille <em>Pohjoismaissa ja Euroopassa</em>",
    "subhero.text": "NCE Permit AI on energia-alalle suunniteltu sääntelyä hyödyntävä tekoälyalusta. Se yhdistää tekoälyavusteisen dokumentoinnin, lähdejäljitettävän sääntelysisällön ja RAQS-laadunvarmistuksen yhdeksi yhtenäiseksi työnkuluksi.",
    "subhero.country": "Suomi · Ruotsi · Tanska · Norja · Puola · Saksa",

    "stats.minute": "minuuteissa",
    "stats.draft": "Lupahakemusluonnos",
    "stats.types": "Projektityyppiä",
    "stats.mw": "MW BESS-putkessa Suomessa",
    "stats.cost": "Kustannukset verrattuna perinteiseen konsultointiin",
    "stats.summaryTitle": "Ideasta hakemukseksi",
    "stats.summaryText": "Merkittäviä aika- ja kustannussäästöjä verrattuna perinteiseen lupavalmisteluun ja konsultointiin",

    "s1.tag": "01 — LUOTETTAVUUS",
    "s1.h2.html": "Sopimusneuvotteluista <em>rakennuslupiin</em>",
    "s1.p1": "Jokainen raportti perustuu virallisiin lähteisiin: Finlexiin, STUKin YVL-ohjeisiin, EUR-Lexiin, Fingridiin ja SYKEen. RAQS-validointi takaa rakenteen, jäljitettävyyden ja sääntelyn mukaisuuden ennen dokumentin toimitusta.",
    "s1.cta": "Avaa Permit AI",

    "s2.tag": "02 — NÄIN SE TOIMII",
    "s2.h2.html": "Tuhat sivua sääntelyä, <em>yhdeksi hakemukseksi tiivistettynä</em>",
    "s2.step1Title": "Syötä projektin tiedot",
    "s2.step1Text": "Lisää perustiedot, sijainti ja projektityyppi.",
    "s2.step2Title": "Tekoäly hakee sääntelyvaatimukset",
    "s2.step2Text": "NCE Permit AI hakee automaattisesti hankkeeseen liittyvän lainsäädännön, viranomaisvaatimukset ja lähdeviitteet.",
    "s2.step3Title": "Luo rakenteellinen luonnos",
    "s2.step3Text": "Järjestelmä luo PDF-muotoisen lupahakemusluonnoksen valmiina asiantuntijatarkastukseen ja jättämiseen.",
    "s2.step4Title": "RAQS tarkistaa ennen toimitusta",
    "s2.step4Text": "Jokainen dokumentti käy läpi RAQS-validointiprosessin (Regulatory Assurance Quality System). Tämä automaattinen laadunvarmistus varmistaa dokumentin rakenteellisen johdonmukaisuuden, lähdejäljitettävyyden sekä sääntelyn mukaisuuden ennen kuin se saavuttaa käyttäjän. Näin varmistetaan ammattitasoinen laatu ja vähennetään virheiden riskiä.",
    "s2.cta": "Kokeile Permit AI:ta",

    "s3.tag": "03 — TUETUT PROJEKTITYYPIT",
    "s3.h2.html": "Yli 20 hanketyyppiä. <em>Yksi kehitysstandardi</em>",

    // Card 1
    "s3.card1Title": "BESS",
    "s3.card1Meta": "Akkuenergiavarasto · 50–800 MWh",
    "s3.card1Text": "Verkkoliitynnät, ympäristöselvitykset ja rakennusluvat. Suunniteltu taajuussäätöön, energian varastointiin ja uusiutuvan energian integrointiin.",

    // Card 2
    "s3.card2Title": "Tuulivoima — Maa",
    "s3.card2Meta": "Maatuulivoima · 50–600 MW",
    "s3.card2Text": "Laajamittaiset tuulivoimahankkeet metsä- ja maatalousalueilla. Sisältää YVA-menettelyn, maanomistajasopimukset ja sähkönsiirron suunnittelun.",

    // Card 3
    "s3.card3Title": "Tuulivoima — Meri",
    "s3.card3Meta": "Merituulivoima · 500 MW–2 GW",
    "s3.card3Text": "Merialueiden luvitus, merenpohjatutkimukset ja offshore-verkkoliitynnät. Rakennettu suuren mittakaavan uusiutuvan energian tuotantoon Pohjoismaissa.",

    // Card 4
    "s3.card4Title": "Aurinkovoima",
    "s3.card4Meta": "Aurinkopuistot · 10–400 MW",
    "s3.card4Text": "Maahan asennettavat aurinkopuistot valinnaisella akkuvarastoinnilla. Nopea luvituspolku ja skaalautuva toteutusmalli.",

    // Card 5
    "s3.card5Title": "SMR",
    "s3.card5Meta": "Pienydinreaktori · 50–300 MW",
    "s3.card5Text": "Edistyneet ydinenergiahankkeet kaukolämpöön, vedyn tuotantoon ja pitkäaikaiseen perusvoiman tuotantoon.",

    // Card 6
    "s3.card6Title": "Vesivoima",
    "s3.card6Meta": "Vesivoimalaitokset · 5–60 MW",
    "s3.card6Text": "Vesiluvat, ympäristöarvioinnit ja olemassa olevan vesivoimainfrastruktuurin modernisointi.",

    // Card 7
    "s3.card7Title": "Asuminen",
    "s3.card7Meta": "Asuntorakentaminen",
    "s3.card7Text": "Asuinkerrostalot, asuinalueet ja monikäyttöiset korttelit. Rakennusluvat, kaavoituksen vaatimukset ja infrastruktuurin yhteensovittaminen.",

    // Card 8
    "s3.card8Title": "Teollisuus",
    "s3.card8Meta": "Valmistus ja tuotanto",
    "s3.card8Text": "Tehtaat, tuotantolaitokset ja logistiikkakeskukset. Ympäristöluvat, liittymät ja alueen hyväksynnät.",

    // Card 9
    "s3.card9Title": "Maatalous",
    "s3.card9Meta": "Maatalousinfrastruktuuri",
    "s3.card9Text": "Navetat, varastorakennukset ja maatalouden tuotantotilat. Lupaprosessit maaseudun kehityshankkeille.",

    // Card 10
    "s3.card10Title": "Liikekiinteistöt",
    "s3.card10Meta": "Toimisto- ja liiketilahankkeet",
    "s3.card10Text": "Yrityspuistot, toimistorakennokset ja kaupalliset kiinteistöt. Kaavoitus, rakennusluvat ja tekniset liittymät.",

    // Card 11
    "s3.card11Title": "Muu projekti",
    "s3.card11Meta": "Räätälöity kehityshanke",
    "s3.card11Text": "Infrastruktuuri-, julkisen sektorin ja erityiskohteiden hankkeet. Joustavat lupaprosessit projektin tarpeiden mukaan.",

    // Card More
    "s3.cardMoreTitle": "Lisää hanketyyppejä",
    "s3.cardMoreMeta": "Infrastruktuuri · Energia · Teollisuus",
    "s3.cardMoreText": "Datakeskukset, logistiikkakeskukset, julkinen infrastruktuuri, liikekiinteistöt ja toimialakohtaiset hankkeet. NCE Permit AI tukee tällä hetkellä yli 20 projektikategoriaa.",

    "s4.tag": "04 — SMR / SEURAAVAN SUKUPOLVEN YDINVOIMA",
    "s4.h2.html": "Pienydinreaktorit, <em>Euroopan mittakaavassa</em>",
    "s4.p1": "NCE Permit AI tukee pienydinreaktorihankkeiden lupaprosesseja Pohjoismaissa ja Euroopassa, mukaan lukien esiluvitus, sijoituspaikkaselvitykset ja turvallisuusdokumentaatio.",
    "s4.p2": "Suunniteltu hallitsemaan uusien ydinteknologioiden sääntelyvaatimuksia STUKin, SSM:n, DSA:n ja EURATOMin viitekehyksissä.",

    "s5.tag": "05 — LUVITUKSESTA KÄYTTÖTOIMINTAAN",
    "s5.h2.html": "Yksi näkymä <em>jokaiselle megawatille</em>",
    "s5.p1": "NCE Permit AI tuo rakenteen koko luvitusprosessiin ensimmäisestä hakemuksesta rakentamiseen, käyttöön ja vaatimustenmukaisuuden seurantaan. Sääntelyvaatimukset, dokumentaatio ja compliance-seuranta yhdessä auditoitavassa työnkulussa.",

    "s6.tag": "06 — MIKSI NCE",
    "s6.h2.html": "Rakennettu luvitukseen. <em>Ei mihinkään muuhun</em>",
    "s6.p1": "NCE Permit AI ei ole yleiskäyttöinen tekoälykirjoitustyökalu. Se on rakennettu auttamaan energiahankkeita etenemään luvituksessa nopeammin, täydellä sääntelyn jäljitettävyydellä ja vähemmällä manuaalisella työllä.",
    "s6.p2": "Tiimisi työskentelee yhden rakenteisen työnkulun kautta sen sijaan, että etsisi säädöksiä, hallitsisi dokumenttiversioita ja kirjoittaisi samaa sisältöä uudelleen projektista toiseen.",

    "s7.tag": "07 — KENELLE",
    "s7.p1": "Energiakehittäjät · Insinööritoimistot · Lupaprosessien asiantuntijat · Projektipäälliköt · Sääntely- ja compliance-tiimit",

    "s8.tag": "08 — PILOTTIASIAKKAAT",
    "s8.h2.html": "Minuuteissa, <em>ammattitasoinen lupahakemus</em>",
    "s8.p1": "Otamme parhaillaan mukaan ensimmäisiä pilottiasiakkaita.",
    "s8.p2": "Jos työskentelet BESS-, tuuli-, aurinko-, SMR- tai rakennus-, teollisuus- tai infrastruktuurihankkeiden parissa, ota yhteyttä keskustellaksesi pilotista.",
    "cta.openToolRequest": "Pyydä käyttöoikeutta",

    "footer.title": "NCE — Native Clean Energy",
    "footer.p1": "Regulatory Intelligence energia-, rakennus- ja infrastruktuurihankkeille",
    "footer.pp": "Tietosuojaseloste",
    "footer.tag": "Suomi · Ruotsi · Tanska · Norja · Puola · Saksa",

    "tool.tag": "Permit AI · v1.4",
    "tool.h2.html": "Luo <em>lupahakemuksesi</em>.",
    "tool.sub": "Valitse projektityyppi — järjestelmä mukauttaa luvituspolun, vaatimukset ja aikataulun automaattisesti.",

    "tool.t.wind": "Tuulivoima",
    "tool.t.solar": "Aurinkovoima",
    "tool.t.bess": "BESS",
    "tool.t.smr": "SMR",


"tool.more": "+ LISÄÄ...",
"tool.moreLabel": "LISÄÄ",

"tool.selector.kicker": "MUUT PROJEKTITYYPIT",
"tool.selector.title": "Valitse projektityyppi",
"tool.selector.text": "Valitse hanketyyppi lupahakemuksen luomista varten.",
"tool.selector.back": "← Takaisin",


"tool.form.applicant": "Hakija",
"tool.form.business": "Y-tunnus",
"tool.form.address": "Osoite",
"tool.form.property": "Kiinteistötunnus",
"tool.form.municipality": "Kunta",
"tool.form.power": "Kokonaisteho (MW)",
"tool.form.location": "Sijainti / ympäristövaikutukset",
"tool.form.capacity": "Kapasiteetti (MWh)",
"tool.form.phase": "Hankkeen vaihe",
"tool.form.authority": "Kohdeviranomainen",
"tool.form.projectId": "Hanketunnus",

"tool.form.preStudy": "Esiselvitys",
"tool.form.permitting": "Lupavaihe",
"tool.form.construction": "Rakentaminen",

"tool.generate": "Luo lupahakemus",

"tool.generation.title": "AI-GENEROINTI KÄYNNISSÄ",
"tool.generation.subtitle": "RAG + Claude analysoi hankettasi ja valmistelee lupaluonnosta",

"tool.generation.analyze": "Analysoi",
"tool.generation.analyze.sub": "Hankkeen ymmärtäminen",

"tool.generation.retrieve": "Hae",
"tool.generation.retrieve.sub": "Säädösten haku",

"tool.generation.compare": "Vertaa",
"tool.generation.compare.sub": "Vaatimusten täsmäytys",

"tool.generation.assess": "Arvioi",
"tool.generation.assess.sub": "Riskit ja vaikutukset",

"tool.generation.recommend": "Suosittele",
"tool.generation.recommend.sub": "Luonnoksen laatiminen",

"tool.generation.lifecycle": "Elinkaari",
"tool.generation.lifecycle.sub": "Vaihelukituksen tarkistus",

"tool.generation.waiting": "ODOTTAA",

"tool.generation.liveActivity": "REAALIAIKAINEN TOIMINTA",
"tool.generation.starting": "Generointi käynnistyy...",

"tool.generation.dataSources": "TIETOLÄHTEET",
"tool.generation.regulatoryChunks": "Sääntelyaineiston osiot",
"tool.generation.countries": "Maat",
"tool.generation.euLayer": "EU-TASO",
"tool.generation.active": "aktiivinen",

"tool.generation.status": "Generointi käynnistyy...",
"tool.generation.jobId": "Työtunnus:",

"tool.raqs.sectionTitle": "2. RAQS — LAADUNARVIOINTI",
"tool.raqs.sectionSubtitle": "5 laatukriteeriä ja kokonaispistemäärä",

"tool.raqs.title": "RAQS-LAATUPISTEET",
"tool.raqs.live": "LIVE",

"tool.raqs.citations": "Lähdeviitteet",
"tool.raqs.coverage": "Lupavaatimusten kattavuus",
"tool.raqs.uncertainty": "Epävarmuus",
"tool.raqs.comprehensiveness": "Kattavuus",
"tool.raqs.readiness": "Valmius",

"tool.raqs.warning": "EPÄVARMUUTTA HAVAITTU",
"tool.raqs.warningText": "Asiantuntijan tarkistus voi olla tarpeen.",

"tool.result.sectionTitle": "3. TULOS JA TOIMENPITEET",
"tool.result.sectionSubtitle": "Luotu luonnos + pakollinen asiantuntijatarkistus",

"tool.result.title": "LUPAHAKEMUSLUONNOS",
"tool.result.review": "ASIANTUNTIJATARKISTUS",

"tool.result.fileTitle": "Lupahakemusluonnos",
"tool.result.generated": "Luotu:",

"tool.result.download": "↓ LATAA PDF",
"tool.result.summary": "NÄYTÄ YHTEENVETO ◉",

"tool.result.warning": "TEKOÄLYN TUOTTAMA LUONNOS",
"tool.result.warningText": "Asiantuntijan on tarkistettava tämä asiakirja ennen virallista jättämistä.",

"tool.result.markReviewed": "✓ MERKITSE TARKISTETUKSI",

"tool.project.wind": "Tuulivoima",
"tool.project.wind.title": "Maatuulivoima",
"tool.project.wind.meta": "Maatuulivoima · 50–600 MW",
"tool.project.wind.description": "Maatuulivoimahanke — maalla sijaitseva tuulivoimahanke",

"tool.project.solar": "Aurinkovoima",
"tool.project.solar.title": "Aurinkovoima",
"tool.project.solar.meta": "Aurinkovoima · 10–400 MW",
"tool.project.solar.description": "Aurinkopaneelihanke",

"tool.project.bess": "BESS",
"tool.project.bess.title": "Akk energiavarastohanke",
"tool.project.bess.meta": "50–800 MWh",
"tool.project.bess.description": "Akk energiavarastohanke",

"tool.project.smr": "SMR",
"tool.project.smr.title": "Pieni modulaarinen reaktori",
"tool.project.smr.meta": "50–300 MW",
"tool.project.smr.description": "Pienen modulaarisen reaktorin hanke (ennakkolupavaihe)",

"tool.project.windsea": "Merituulivoima",
"tool.project.windsea.title": "Merituulivoima",
"tool.project.windsea.meta": "500 MW–2 GW",
"tool.project.windsea.description": "Merituulivoimahanke",

"tool.project.hydro": "Vesivoima",
"tool.project.hydro.title": "Vesivoima",
"tool.project.hydro.meta": "5–60 MW",
"tool.project.hydro.description": "Vesivoimahanke",

"tool.project.residential": "Asuinrakentaminen",
"tool.project.residential.title": "Asuinrakennushanke",
"tool.project.residential.meta": "5–6 MW",
"tool.project.residential.description": "Asuinrakennus — lupahakemus",

"tool.project.industrial": "Teollisuus",
"tool.project.industrial.title": "Teollisuus- ja tuotantorakentaminen",
"tool.project.industrial.meta": "5–6 MW",
"tool.project.industrial.description": "Teollisuusrakennus — lupahakemus",

"tool.project.agriculture": "Maatalous",
"tool.project.agriculture.title": "Maatalouden infrastruktuuri",
"tool.project.agriculture.meta": "5–6 MW",
"tool.project.agriculture.description": "Maatalousrakennus — lupahakemus",

"tool.project.commercial": "Liikerakentaminen",
"tool.project.commercial.title": "Toimisto- ja liiketilarakentaminen",
"tool.project.commercial.meta": "5–6 MW",
"tool.project.commercial.description": "Liikerakennus — lupahakemus",

"tool.project.other": "Muu hanke",
"tool.project.other.title": "Muu kehityshanke",
"tool.project.other.meta": "5–6 MW",
"tool.project.other.description": "Muu rakennushanke — lupahakemus",

"tool.dynamic.wind.power": "Kokonaisteho (MW)",
"tool.dynamic.wind.turbines": "Turbiinien määrä",

"tool.dynamic.solar.capacity": "Asennettu kapasiteetti (MWp)",
"tool.dynamic.solar.area": "Pinta-ala (ha)",

"tool.dynamic.bess.power": "Teho (MW)",
"tool.dynamic.bess.capacity": "Kapasiteetti (MWh)",
"tool.dynamic.bess.grid": "Verkkoliitäntä",

"tool.dynamic.smr.thermalPower": "Lämpöteho (MW)",
"tool.dynamic.smr.reactorType": "Reaktorityyppi",

"tool.dynamic.hydro.reactorType": "Vesivoiman tyyppi",

"tool.dynamic.residential.floors": "Kerrosten määrä",

"tool.dynamic.industrial.intendedUse": "Käyttötarkoitus",
"tool.dynamic.industrial.buildingVolume": "Rakennuksen tilavuus (m³)",

"tool.dynamic.select": "Valitse...",

    "tool.f.name": "Projektin nimi",
    "tool.f.jur": "Toimivalta-alue",
    "tool.f.cap": "Kapasiteetti (MW)",
    "tool.f.brief": "Projektikuvaus",
    "tool.f.run": "Suorita lupa-analyysi",

    "tool.back": "← Takaisin sivustolle",

    "modal.title":    "Pyydä käyttöoikeutta",
    "modal.sub":      "Otamme mukaan pilottiasiakkaita. Täytä tietosi ja otamme sinuun yhteyttä.",
    "modal.company":  "Yritys",
    "modal.contact":  "Yhteyshenkilö",
    "modal.email":    "Sähköposti",
    "modal.phone":    "Puhelin",
    "modal.optional": "(valinnainen)",
    "modal.desc":     "Kuvaus toiminnasta",
    "modal.submit":   "Lähetä",
    "modal.credNote": "Lähetämme käyttöoikeustunnukset antamaasi sähköpostiosoitteeseen.",

    "hero.slides": [
      "Maatuulivoimahanke",
      "Uusiutuva infrastruktuuri",
      "Merituulivoima",
      "Akkuenergiavarasto",
      "Aurinkovoimatuotanto",
      "Pohjoismainen ympäristö",
      "Yhteisöintegraatio",
      "Projektikehitys",
      "Sääntelyn vaatimustenmukaisuus",
      "SMR ja ydinenergia"
    ],
  },

  
  };

  let currentLang = "en";

  window.translate = function(key) {
      return I18N[currentLang]?.[key] || key;
  };

  function applyI18n() {

    document.documentElement.lang = currentLang;
    document.body.className = document.body.className
            .replace(/\blang-\S+/g, '')
            .trim();
    document.body.classList.add('lang-' + currentLang);

    const dict = I18N[currentLang];

    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.dataset.i18n;
        if (dict[key] == null) return;
        if (key.endsWith(".html")) {
          el.innerHTML = dict[key];
        } else {
          el.textContent = dict[key];
        }
    });

    // hero slides label
    // const idx = parseInt(document.getElementById("pagerIndex").textContent, 10) || 1;
    // document.getElementById("pagerName").textContent = dict["hero.slides"][idx - 1];
    
    // hero slides label
    const pagerIndex = document.getElementById("pagerIndex");
    const pagerName = document.getElementById("pagerName");
    
    if (pagerIndex && pagerName && dict["hero.slides"]) {
        const idx = parseInt(pagerIndex.textContent, 10) || 1;
        pagerName.textContent = dict["hero.slides"][idx - 1];
    }

    // cards
    // renderCards();
    // lang labels
    document.getElementById("langLabel").textContent = currentLang.toUpperCase();
    const ll2 = document.getElementById("langLabelTool");
    if (ll2) ll2.textContent = currentLang.toUpperCase();
  }

  // Wire data-i18n-html on the serif H2s that need <em>
  document.addEventListener("DOMContentLoaded", () => {
    const map = {
      "s1.h2": "s1.h2.html",
      "s2.h2": "s2.h2.html",
      "s3.h2": "s3.h2.html",
      "s4.h2": "s4.h2.html",
      "s5.h2": "s5.h2.html",
      "s6.h2": "s6.h2.html",
      "subhero.title": "subhero.title.html",
      "tool.h2": "tool.h2.html"
    };
    // Find each serif-h2 by the textContent placeholder we shipped in HTML and map.
    // Simpler: find by data attribute on shipped h2s — assign manually:
    const h2s = document.querySelectorAll(".serif-h2");
    // For each h2, set data-i18n-html via heuristic: take parent section's first .tag text key.
    h2s.forEach((h2) => {
      const section = h2.closest("section");
      const tag = section && section.querySelector(".tag[data-i18n]");
      if (!tag) return;
      const tagKey = tag.getAttribute("data-i18n"); // e.g. "s1.tag"
      const base = tagKey.replace(".tag", ".h2");
      if (map[base]) h2.setAttribute("data-i18n-html", map[base]);
    });
    init();
  });

  /* ---------- 4. Init ---------- */
  function init() {
    setupHeaderScroll();
    setupLangSwitcher();
    initHeroCounters();
    setupAccessModal();
    // setupHeroSlides();
    // setupViewSwitcher(); 111111
    setupJumpButtons();
    setupParallax();
    setupTypePicker();
    // renderCards();
    // applyI18n();

    if (document.querySelector('.hero')) {
        setupHeroSlides();
        // setupParallax();
    }

    if (document.querySelector('#typePicker')) {
        setupTypePicker();
    }


  }

  /* ---------- 5. Header scroll transition ---------- */
  function setupHeaderScroll() {
    const header = document.getElementById("siteHeader");
    const onScroll = () => {
      if (window.scrollY > 60) header.classList.add("scrolled");
      else header.classList.remove("scrolled");
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  /* ---------- 6. Hero slides (8 crossfade) ---------- */
  function setupHeroSlides() {
    const slides = document.querySelectorAll("#heroBg .slide");
    const dotsContainer = document.getElementById("heroDots");
    const pagerIndex = document.getElementById("pagerIndex");
    const pagerName = document.getElementById("pagerName");
    let current = 0;
    let timer = null;

    slides.forEach((_, i) => {
      const dot = document.createElement("button");
      dot.className = "dot" + (i === 0 ? " active" : "");
      dot.setAttribute("aria-label", "Slide " + (i + 1));
      dot.addEventListener("click", () => go(i));
      dotsContainer.appendChild(dot);
    });
    const dots = dotsContainer.querySelectorAll(".dot");

    function go(i) {
        current = i;
        slides.forEach((s, idx) => {
            s.classList.toggle("active", idx === i);
            const video = s.querySelector("video");
            if(video){
                if(idx === i){
                    clearTimeout(video.pauseTimer);
                    video.currentTime = 0;
                    video.play().catch(() => {});
                }else{
                    clearTimeout(video.pauseTimer);
                    video.pauseTimer = setTimeout(() => { video.pause(); }, 2500);
                }
            }
        });
    
        dots.forEach((d, idx) => d.classList.toggle("active", idx === i));
        pagerIndex.textContent = String(i + 1).padStart(2, "0");
        pagerName.textContent =  I18N[currentLang]["hero.slides"][i];
        restart();
    }

    function next() { go((current + 1) % slides.length); }
    function restart() { clearInterval(timer); timer = setInterval(next, 6000); }
    restart();
  }

  /* ---------- z. autoplay video ---------- */

  document.querySelectorAll('.section-video').forEach(video => {
      const observer = new IntersectionObserver(
          entries => {
              entries.forEach(entry => {
                  if(entry.isIntersecting){
                      video.play().catch(() => {});
                  }else{
                      video.pause();
                  }
              });
          },
          {
              threshold: 0.4
          }
      );
      observer.observe(video);
  });

  function setupJumpButtons() {
    document.querySelectorAll("[data-jump]").forEach((b) => {
      b.addEventListener("click", (e) => {
        e.preventDefault();
        switchView(b.getAttribute("data-jump"));
      });
    });
  }

  /* ---------- 8. Language switcher ---------- */
  function setupLangSwitcher() {
    document.querySelectorAll(".lang-switcher").forEach((sw) => {
      const btn = sw.querySelector(".lang-current");
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        sw.classList.toggle("open");
        btn.setAttribute("aria-expanded", sw.classList.contains("open") ? "true" : "false");
      });
      sw.querySelectorAll("[data-lang]").forEach((li) => {
        li.addEventListener("click", () => {
          currentLang = li.getAttribute("data-lang");
          sw.classList.remove("open");
          applyI18n();
        });
      });
    });
    document.addEventListener("click", () => {
      document.querySelectorAll(".lang-switcher.open").forEach((sw) => sw.classList.remove("open"));
    });
  }

  /* ---------- 9. Parallax ---------- */
  function setupParallax() {
    const targets = document.querySelectorAll("[data-parallax] .media-container");
    if (!targets.length) return;
    let ticking = false;
    function update() {
      const vh = window.innerHeight;
      targets.forEach((el) => {
        const r = el.getBoundingClientRect();
        if (r.bottom < 0 || r.top > vh) return;
        const progress = (r.top + r.height / 2 - vh / 2) / vh; // -0.5..0.5
        el.style.transform = `translateY(${progress * -66}px)`;
      });
      ticking = false;
    }
    window.addEventListener("scroll", () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  }

  /* ---------- 10. Tool type picker (dynamic background) ---------- */
  function setupTypePicker() {
    const picker = document.getElementById("typePicker");
    if (!picker) return;
    const bg = document.getElementById("toolBg");
    const grid = document.getElementById("toolGridOverlay");
    picker.querySelectorAll(".tp-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        picker.querySelectorAll(".tp-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const type = btn.getAttribute("data-type");
        bg.setAttribute("data-accent", type);
        grid.classList.toggle("show", type === "wind" || type === "smr");
      });
    });
  }


  function createCounter(el, value){
      const str = String(value);
      el.innerHTML = '';
      [...str].forEach(finalDigit => {
  
          const digitWrap = document.createElement('div');
          digitWrap.className = 'digit';
          const wheel = document.createElement('div');
          wheel.className = 'wheel';
  
          for(let i=0;i<=9;i++){
              const span = document.createElement('span');
              span.textContent = i;
              wheel.appendChild(span);
          }
  
          digitWrap.appendChild(wheel);
          el.appendChild(digitWrap);
  
          const startDigit = Math.floor(Math.random()*10);
  
          wheel.style.transform = `translateY(-${startDigit*34}px)`;
  
          setTimeout(() => {
              wheel.style.transform = `translateY(-${finalDigit*34}px)`;
          }, 300);
      });
  }

  function initHeroCounters(){
      const stats = document.getElementById('heroStats');
      if(!stats) return;

      // Fetch live stats and update data-value before animation fires at 4 s.
      // Counter order: [0] chunks, [1] project_types, [2] countries.
      fetch('/api/stats')
          .then(r => r.json())
          .then(data => {
              const counters = stats.querySelectorAll('.counter');
              if(counters[0] && data.chunks_total)  counters[0].dataset.value = data.chunks_total;
              if(counters[1] && data.project_types) counters[1].dataset.value = data.project_types;
              if(counters[2] && data.countries)     counters[2].dataset.value = data.countries;
          })
          .catch(() => {}); // keep hardcoded fallback values on network error

      setTimeout(() => { stats.classList.add('visible'); }, 3000);
      setTimeout(() => {
          document
              .querySelectorAll('.counter')
              .forEach(counter => {
                  createCounter(
                      counter,
                      counter.dataset.value
                  );
              });
      }, 4000);
  }


// moreTypesBtn
// document
// .getElementById('moreTypesBtn')
// .addEventListener('click', () => {

//     document
//         .getElementById('moreTypesPanel')
//         .classList
//         .toggle('open');

// });


  /* ---------- Access Request Modal ---------- */
  function setupAccessModal() {
      const overlay = document.getElementById('accessModalOverlay');
      if (!overlay) return;
      const closeBtn = document.getElementById('accessModalClose');
      const form     = document.getElementById('accessForm');

      function openModal() {
          overlay.classList.add('open');
          document.body.style.overflow = 'hidden';
      }
      function closeModal() {
          overlay.classList.remove('open');
          document.body.style.overflow = '';
      }

      // All buttons/links with data-modal="access" open the modal
      document.querySelectorAll('[data-modal="access"]').forEach(el => {
          el.addEventListener('click', e => { e.preventDefault(); openModal(); });
      });

      closeBtn.addEventListener('click', closeModal);
      overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });
      document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

      form.addEventListener('submit', e => {
          e.preventDefault();
          const company = document.getElementById('af-company').value.trim();
          const contact = document.getElementById('af-contact').value.trim();
          const email   = document.getElementById('af-email').value.trim();
          const phone   = document.getElementById('af-phone').value.trim();
          const desc    = document.getElementById('af-desc').value.trim();

          if (!company || !contact || !email || !desc) return;

          const lang       = document.body.classList.contains('lang-fi') ? 'fi' : 'en';
          const successMsg = lang === 'fi'
              ? 'Kiitos! Otamme yhteyttä pian.'
              : 'Thank you! We will be in touch soon.';
          const errorMsg   = lang === 'fi'
              ? 'Lähetys epäonnistui. Lähetä sähköposti osoitteeseen info@ncenergy.fi'
              : 'Sending failed. Please email us at info@ncenergy.fi';

          const submitBtn = form.querySelector('.access-submit');
          submitBtn.disabled = true;
          const origText = submitBtn.textContent;
          submitBtn.textContent = '…';

          fetch('/api/access-request', {
              method:  'POST',
              headers: {'Content-Type': 'application/json'},
              body:    JSON.stringify({
                  yritys:        company,
                  yhteyshenkilo: contact,
                  sahkoposti:    email,
                  puhelin:       phone,
                  kuvaus:        desc,
              }),
          })
          .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
          .then(() => {
              form.innerHTML = '<p style="color:#00d9c7;font-size:16px;margin:0;line-height:1.5">'
                  + successMsg + '</p>';
          })
          .catch(() => {
              let errEl = form.querySelector('.access-err');
              if (!errEl) {
                  errEl = document.createElement('p');
                  errEl.className = 'access-err';
                  errEl.style.cssText = 'color:#f87171;font-size:13px;margin:10px 0 0';
                  form.appendChild(errEl);
              }
              errEl.textContent = errorMsg;
              submitBtn.disabled  = false;
              submitBtn.textContent = origText;
          });
      });
  }


})();