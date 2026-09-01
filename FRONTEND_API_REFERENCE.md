# NCE Permit AI — Frontend API Reference

**Base URL:** `https://ai.ncenergy.fi`  
**Auth:** HTTP Basic (user: `nce`, pass: `BASIC_AUTH_PASS` env var). Endpoints marked **[public]** skip auth.  
**Content-Type:** `application/json; charset=utf-8` (requests and responses unless noted)

---

## Authentication

All endpoints except the four public ones require an `Authorization: Basic <base64>` header.  
On `ai.ncenergy.fi`, unauthenticated requests return **401** with an HTML page (not JSON).  
Requests to `ncenergy.fi` (landing page domain) bypass auth entirely.

---

## Public endpoints (no auth required)

### `GET /api/health` [public]

Service liveness check.

**Response 200**
```json
{ "status": "ok", "mml_key_set": true }
```

---

### `GET /api/rag-status` [public]

RAG embedding database status. Use this to show users whether the AI knowledge base is fully loaded.

**Response 200**
```json
{
  "active_collection":      "permit_docs",
  "active_model":           "all-MiniLM-L6-v2",
  "v2_ready":               false,
  "permit_docs_count":      10316,
  "permit_docs_v2_count":   null,
  "db_path":                "/root/bess_tool/permit_ai/embeddings",
  "db_path_exists":         true,
  "db_path_files":          ["chroma.sqlite3", "permit_docs"]
}
```

| Field | Meaning |
|---|---|
| `active_collection` | `permit_docs` (V1) or `permit_docs_v2` (V2 mpnet) |
| `active_model` | Current embedding model name |
| `v2_ready` | `true` once V2 re-index completes (≥10,000 chunks) |
| `permit_docs_count` | Chunks in V1 collection (`null` = collection missing) |
| `permit_docs_v2_count` | Chunks in V2 collection (`null` = not yet created) |
| `db_error` | Present only if ChromaDB client init fails |

---

### `POST /api/access-request` [public]

Sends an access request email to `info@ncenergy.fi`. Used from the landing page.

**Request body**
```json
{
  "yritys":         "Acme Oy",
  "yhteyshenkilo":  "Matti Meikäläinen",
  "sahkoposti":     "matti@acme.fi",
  "puhelin":        "+358 40 123 4567",
  "kuvaus":         "Haluamme käyttää BESS-lupahakemuksen generointia."
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `yritys` | string | yes | Company name |
| `yhteyshenkilo` | string | yes | Contact person |
| `sahkoposti` | string | yes | Email |
| `puhelin` | string | no | Phone (defaults to `""`) |
| `kuvaus` | string | yes | Request description |

**Response 200**
```json
{ "ok": true }
```

---

### `GET /api/stats` [public]

Summary counts for display on the landing page.

**Response 200**
```json
{
  "chunks_total":  10316,
  "countries":     6,
  "project_types": 20,
  "languages":     7
}
```

---

## Tool endpoints (auth required)

### `POST /api/generate-application`

Generate a permit application PDF in the background. Returns immediately with a `job_id`; poll `/api/proofread/{job_id}` until done, then download.

**Rate limit:** 5 calls / hour per IP

**Request body**
```json
{
  "hanketyyppi":                  "BESS",
  "kiinteistotunnus":             "636-439-4-711",
  "teho_mw":                      5.0,
  "kapasiteetti_mwh":             20.0,
  "y_tunnus":                     "1234567-8",
  "osoite":                       "Teollisuustie 1",
  "kunta":                        "Pöytyä",
  "hakija":                       "Acme Energia Oy",
  "sijainti_ymparistovaikutukset": "Sijaitsee teollisuusalueella, ei Natura-alueita lähellä.",
  "hankkeen_vaihe":               "lupavaihe",
  "kohdeviranomainen":            "Lupa- ja valvontavirasto (Luova)",
  "lang":                         "FI",
  "country":                      "FI",
  "session_id":                   "abc123",
  "hanke_id":                     "",
  "ifc_floor_area":               0.0,
  "ifc_building_height":          0.0,
  "ifc_fire_rating":              "",
  "ifc_materials":                "",
  "ifc_storeys":                  0,
  "ifc_compliance_flags":         ""
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `hanketyyppi` | string | **yes** | One of the 23 project type IDs below |
| `kiinteistotunnus` | string | **yes** | Finnish property ID, e.g. `636-439-4-711` |
| `kunta` | string | **yes** | Municipality name |
| `hakija` | string | **yes** | Applicant name |
| `teho_mw` | float | no | Power capacity in MW (default 0.0) |
| `kapasiteetti_mwh` | float | no | Energy capacity in MWh (default 0.0) |
| `y_tunnus` | string | no | Finnish business ID |
| `osoite` | string | no | Street address |
| `sijainti_ymparistovaikutukset` | string | no | Site description / environmental notes |
| `hankkeen_vaihe` | string | no | Project phase — see Phase values below |
| `kohdeviranomainen` | string | no | Target authority |
| `lang` | string | no | Output language: `FI` (default), `EN`, `SE`, `DA`, `NO`, `PL`, `DE`, `ET`, `LV`, `LT` |
| `country` | string | no | Regulatory context: `FI` (default), `SE`, `DA`, `NO`, `PL`, `DE`, `EE`, `LV`, `LT` |
| `session_id` | string | no | Session ID for phase-lock tracking |
| `hanke_id` | string | no | RTB cockpit linkage ID (default `""`) |
| `ifc_floor_area` | float | no | From IFC file parse (default 0.0) |
| `ifc_building_height` | float | no | From IFC file parse (default 0.0) |
| `ifc_fire_rating` | string | no | From IFC file parse |
| `ifc_materials` | string | no | From IFC file parse |
| `ifc_storeys` | int | no | From IFC file parse (default 0) |
| `ifc_compliance_flags` | string | no | From IFC file parse |

**Response 202** — PDF generation started
```json
{ "job_id": "3f9a2b1c4d" }
```
Also includes header `X-Job-Id: 3f9a2b1c4d`.

**Error 400** — invalid `hanketyyppi`:
```json
{ "detail": "hanketyyppi oltava: BESS, SMR, asuinrakennus, ..." }
```

---

### `GET /api/proofread/{job_id}`

Poll job status after `POST /api/generate-application`. Recommended poll interval: 5-10s — a
real generation can now legitimately take up to ~20-25 minutes (SMR/smr_bess in particular;
see the YVL Compliance Memo note under `stage` below), so don't apply a short client-side
give-up timeout — keep polling until a terminal `status` arrives.

**Response 200 — full shape (2026-08-31; `phase_status`/`stage`/`raqs`/`late_completion_available`
are additive — always present, `null`/`false` when not yet applicable, never removed)**
```json
{
  "status": "done",
  "error": null,
  "debug_sections": { "kuvaus": 1842, "perustelut": 1204, "luvat_teksti": 980, "toimenpiteet": 640 },
  "phase_status": null,
  "stage": "complete",
  "raqs": {
    "id": 28,
    "generation_id": "60a92f2509",
    "created_at": "2026-08-31T07:51:54.677442+00:00",
    "overall": 3.2,
    "scores": {
      "viittaukset":    { "pisteet": 3, "perustelu": "..." },
      "lupakattavuus":  { "pisteet": 4, "perustelu": "..." },
      "epävarmuus":     { "pisteet": 2, "perustelu": "..." },
      "kattavuus":      { "pisteet": 4, "perustelu": "..." },
      "valmisteluaste": { "pisteet": 3, "perustelu": "..." }
    },
    "flagged": [ { "criterion": "epävarmuus", "pisteet": 2, "perustelu": "..." } ]
  },
  "late_completion_available": false
}
```

| Field | Meaning |
|---|---|
| `status` | See status value table below |
| `error` | `null`, or a message string for any non-success terminal status |
| `debug_sections` | Character counts per drafted section; `null` until drafting starts |
| `phase_status` | Phase-lock auto-advance result, if `session_id`/`hankkeen_vaihe` were sent and phase-lock is enabled; otherwise `null` |
| `stage` | Real, computed progress marker — see values below. Use this for a progress UI, not a fixed-duration client-side timer |
| `raqs` | RAQS quality self-review (5 criteria, 1-5 each, `overall` = average, `flagged` = any criterion ≤2). `null` until `status=="done"`, or if RAQS logging itself failed for this generation (rare, never blocks the PDF) |
| `late_completion_available` | Almost always `false`. `true` only in the rare case a background thread kept running past this job's own already-set terminal status and genuinely finished anyway (e.g. an internal cooperative deadline check missed a narrow window). If `true`, fetch `GET /api/proofread/{job_id}/late-completion` — the normal `/download` endpoint won't have it, since the job's official status was never `"done"` |

**`status` values**

| `status` value | Meaning |
|---|---|
| `pending` | Queued, not started yet |
| `running` | Currently generating |
| `done` | PDF ready for download |
| `error` | Generation failed; `error` field contains message |
| `insufficient_sources` | RAG knowledge base lacked sources for this type/country |
| `cap_exceeded` | A per-generation Claude-API-call safety cap tripped (very rare — a resource guardrail, not a normal outcome) |
| `timeout_soft_abort` | An internal generation-time budget was exceeded and the job stopped itself cleanly (as opposed to a hard failure) — currently only reachable for SMR/smr_bess, `country=="FI"` generations, which run a much longer YVL Compliance Memo pass; `error` contains details, `guides_completed` (if present) lists which parts finished first |

**`stage` values** (only meaningful while `status=="running"`; once `status` reaches any
terminal value, treat that as authoritative over `stage`)

| `stage` value | Meaning |
|---|---|
| `retrieval` | Fetching regulatory source material |
| `draft` | Drafting the application sections |
| `proofread` | AI proofreading pass |
| `raqs_final` | Final quality review — for SMR/smr_bess FI generations, this stage also covers the YVL Compliance Memo (3 regulatory-guide sections), which is why this stage can run considerably longer than the others for that project type |
| `finalizing` | PDF assembly, right before completion |
| `complete` | Same meaning as `status=="done"` |

**Response 422 — insufficient_sources**
```json
{
  "detail": {
    "error":         "insufficient_sources",
    "message":       "Riittämätön lähdeaineisto — ...",
    "chunks_found":  3,
    "avg_relevance": 0.41
  }
}
```

---

### `GET /api/proofread/{job_id}/late-completion`

Rare-path endpoint — only relevant if `late_completion_available: true` was seen on a prior
poll (see above). Retrieves a background generation that kept running after its own job
already reported a terminal (non-`done`) status, and then genuinely finished. Not part of the
normal completion flow — don't poll this speculatively.

**Response 200** — if the late result was a PDF: `Content-Type: application/pdf`, same
`Content-Disposition` shape as `/download`. If it was only the draft stage finishing late (no
PDF), returns JSON: `{ "kind": "draft", "completed_at": "...", "note": "draft-stage late completion only — no PDF to download" }`.

**Response 404** — no late completion recorded for this job (the normal case).

---

### `GET /api/proofread/{job_id}/download`

Download the completed PDF. Only available when `status == "done"`.

**Response 200** — `Content-Type: application/pdf`  
`Content-Disposition: attachment; filename="hakemus_BESS_Pöytyä.pdf"`

Filename prefix per output language:
| `lang` | Prefix |
|---|---|
| `FI` | `hakemus` |
| `EN` | `application` |
| `SE` | `ansökan` |
| `DA` | `ansøgning` |
| `NO` | `søknad` |
| `PL` | `wniosek` |

**Response 404** — PDF not ready or job not found.

---

### `POST /api/permit-ai`

Direct RAG question answering (not application generation). Returns an AI answer with sources.

**Rate limit:** 50 calls / hour per IP

**Request body**
```json
{
  "question":  "Mitä lupia BESS-akkuvarasto tarvitsee Suomessa?",
  "n_results": 5
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | **yes** | Question in any language (Finnish recommended) |
| `n_results` | int | no | Number of RAG chunks to retrieve (default 5) |

**Response 200**
```json
{
  "answer":  "BESS-akkuvarasto tarvitsee seuraavat luvat: ...",
  "sources": ["fingrid_sjv2024.pdf", "pelastusopisto_ohjeet.pdf"]
}
```

---

### `GET /api/phase-status`

Returns which project phases are completed and which is next. Used to enforce the sequential 3-phase workflow.

**Query parameters**

| Param | Type | Required |
|---|---|---|
| `session_id` | string | yes |
| `hanketyyppi` | string | yes |

**Response 200 — phase lock disabled (current production state)**
```json
{
  "completed_phase": 0,
  "next_phase": 1,
  "phase_lock_disabled": true,
  "phases": [
    { "name": "esiselvitys",  "phase": 1, "state": "active" },
    { "name": "lupavaihe",    "phase": 2, "state": "active" },
    { "name": "rakentaminen", "phase": 3, "state": "active" }
  ]
}
```

When `phase_lock_disabled: true`, all three phases are open — no phase gating enforced.

---

### `POST /api/complete-phase`

Mark a phase as complete and unlock the next one.

**Request body**
```json
{
  "session_id":  "abc123",
  "hanketyyppi": "BESS",
  "phase":       1
}
```

| Field | Type | Notes |
|---|---|---|
| `session_id` | string | Session identifier |
| `hanketyyppi` | string | Project type ID |
| `phase` | int | `1`, `2`, or `3` |

**Response 200**
```json
{ "ok": true, "next_phase": 2 }
```

---

### `POST /api/parse-ifc`

Parse an IFC (BIM) file and extract permit-relevant fields. Returns pre-filled form values.

**Rate limit:** 20 calls / hour per IP  
**Content-Type:** `multipart/form-data`  
**Max file size:** 50 MB

**Form fields**

| Field | Type | Notes |
|---|---|---|
| `file` | file upload | `.ifc` file |
| `project_type` | string (query param) | One of: `BESS`, `AURINKO`, `TUULI`, `SMR`, `DATAKESKUS`, `SCO2`, `VESIVOIMA`, `YVA`, `VERKKO` |
| `country` | string (query param) | `FI`, `SE`, `DA`, `NO`, `PL`, `DE`, `LV`, `EE` — **this endpoint's own allow-list, checked separately from `/api/generate-application`'s. It includes `LV`/`EE` but not `LT`** (confirmed directly against `parse_ifc()`'s validation in `backend/main.py`, spotted while correcting the country/lang tables below — not one of the five corrections requested, included since it's directly adjacent). If you need IFC prefill for Lithuania, flag it to backend first. |

**Response 200**
```json
{
  "prefilled_fields": {
    "teho_mw":           { "value": 5.0,    "confidence": 0.92 },
    "ifc_floor_area":    { "value": 1200.0, "confidence": 0.87 }
  },
  "missing_fields":  ["y_tunnus", "kunta"],
  "compliance_flags": ["P3 fire load class — automatic suppression required"],
  "summary":         "BESS installation, 5 MW, 2 storeys, fire rating P3",
  "parse_errors":    [],
  "ifc_schema":      "IFC4",
  "filename":        "project.ifc"
}
```

---

### `POST /api/approve-ifc`

Submit engineer-reviewed IFC fields to generate a final PDF with audit trail.

**Rate limit:** 10 calls / hour per IP

**Request body**
```json
{
  "hanketyyppi":       "BESS",
  "kiinteistotunnus":  "636-439-4-711",
  "teho_mw":           5.0,
  "kapasiteetti_mwh":  20.0,
  "kunta":             "Pöytyä",
  "hakija":            "Acme Energia Oy",
  "lang":              "FI",
  "country":           "FI",
  "hankkeen_vaihe":    "lupavaihe",
  "kohdeviranomainen": "",
  "approved_fields":   { "teho_mw": 5.0, "ifc_floor_area": 1200.0 },
  "reviewer_name":     "Jyrki Rintanen",
  "review_notes":      "Fire suppression system confirmed."
}
```

**Response 200** — `Content-Type: application/pdf`  
Headers include:
- `X-NCE-Audit-Timestamp`
- `X-NCE-Audit-Reviewer`
- `X-NCE-Audit-Fields`

---

### `POST /api/optimize-bess`

Site optimization — scores candidate locations within a bounding box.

**Rate limit:** 20 calls / hour per IP

**Request body**
```json
{
  "bbox":         [60.4, 22.3, 60.8, 23.1],
  "project_type": "bess",
  "power_mw":     5.0,
  "min_area_ha":  2.0
}
```

| Field | Type | Notes |
|---|---|---|
| `bbox` | array[4] | `[lat_min, lon_min, lat_max, lon_max]` — must overlap Finland |
| `project_type` | string | `bess`, `tuulivoima`, `aurinkovoima`, or `smr` |
| `power_mw` | float | Target power (default 5.0) |
| `min_area_ha` | float | Minimum site area in hectares (default 2.0) |

**Error 400** — bbox outside Finland lat 59.5–70.1, lon 19.5–31.6

---

### `GET /api/property/{kiinteistotunnus}`

Property boundaries from the Finnish National Land Survey (MML) INSPIRE WFS.

**Path param:** `kiinteistotunnus` — e.g. `636-439-4-711`  
**Query param:** `api_key` (optional, defaults to server-side MML key)

**Response 200** — GeoJSON FeatureCollection

---

### `GET /api/fingrid/lines`

Transmission lines from OpenStreetMap Overpass.

**Query params**

| Param | Default | Notes |
|---|---|---|
| `bbox` | `22.5,60.6,23.0,60.9` | `minlon,minlat,maxlon,maxlat` |
| `min_voltage_kv` | `0` | Filter by minimum voltage |

**Response 200** — GeoJSON FeatureCollection

---

### `GET /api/groundwater`

Groundwater protection areas from SYKE Hakku.

**Query param:** `bbox` — `minlon,minlat,maxlon,maxlat` (default `22.5,60.6,23.0,60.9`)

**Response 200** — GeoJSON FeatureCollection

---

### `GET /api/buildings/nearest`

Nearest building from OSM within a radius.

**Query params**

| Param | Required | Default | Notes |
|---|---|---|---|
| `lat` | yes | — | Latitude |
| `lon` | yes | — | Longitude |
| `radius_km` | no | 1.0 | Search radius in km |

**Response 200**
```json
{
  "nearest_building_m": 342,
  "buildings_found": 7,
  "geojson": { ... }
}
```

---

### `GET /api/natura`

Natura 2000 protected areas from Syke.

**Query param:** `bbox` — `minlon,minlat,maxlon,maxlat` (default `22.5,60.6,23.0,60.9`)

**Response 200** — GeoJSON FeatureCollection

---

## Reference tables

### Project type identifiers (`hanketyyppi`)

These are the exact backend values to send. 23 total (confirmed against the
`allowed` set in `backend/main.py`'s `/api/generate-application` handler).

| `hanketyyppi` value | Finnish display name | Notes |
|---|---|---|
| `BESS` | Akkuenergiavarastohanke | Battery Energy Storage |
| `tuulivoima_maa` | Maalle sijoitettava tuulivoimahanke | Onshore wind |
| `tuulivoima_meri` | Merelle sijoitettava tuulivoimahanke (offshore) | Offshore wind |
| `aurinkovoima` | Aurinkovoimalahanke | Solar |
| `SMR` | Pienydinreaktori (SMR) — ennakkolupahakemus | Small Modular Reactor, FI |
| `smr_se` | Pienydinreaktori (SMR) | SMR, Sweden regulatory context |
| `smr_no` | Pienydinreaktori (SMR) | SMR, Norway regulatory context |
| `smr_da` | Pienydinreaktori (SMR) | SMR, Denmark regulatory context |
| `smr_de` | Pienydinreaktori (SMR) | SMR, Germany regulatory context |
| `smr_ee` | Pienydinreaktori (SMR) | SMR, Estonia regulatory context |
| `smr_lv` | Pienydinreaktori (SMR) | SMR, Latvia regulatory context |
| `smr_bess` | SMR + BESS -hybridienergiajärjestelmä | SMR + Battery hybrid |
| `vesivoima` | Vesivoimalahanke | Hydropower |
| `hybridi` | Hybridivoimalahanke (BESS + tuuli/aurinko) | Hybrid power plant |
| `ymparistolupa` | Ympäristölupahakemus | Environmental permit |
| `datakeskus` | Datakeskushanke | Data centre |
| `egs` | Enhanced Geothermal System | **No backend config exists for this value** — see note below |
| `offshore_wind` | Merelle sijoitettava tuulivoimahanke | Maps to `tuulivoima_meri` |
| `asuinrakennus` | Asuinrakennus | Residential construction |
| `teollisuus` | Teollisuus | Industrial construction |
| `maatalous` | Maatalous | Agricultural construction |
| `liikerakennus` | Liikerakennus | Commercial building |
| `muu` | Muu | Other — **see note below** |

> **Note:** `asuinrakennus`, `teollisuus`, `maatalous`, and `liikerakennus` use generic permit templates without energy-specific regulatory content, but do have real per-type backend config.
>
> **`egs` — no backend config, do not offer it as a normal option.** The backend accepts `egs` as a valid `hanketyyppi` (validation passes), but it is not aliased to any content — a real generation request with `egs` would produce a document with no hanketyyppi-specific permit list, law citations, or prompt content. There used to be an alias to `aurinkovoima`, but it was deliberately removed (per an explicit code comment in `generate_application.py`) because serving solar-permit content under an EGS label was judged worse than an outright error. The backend's own assumption is that the **frontend** gates `egs` behind a "coming soon" state and never actually submits it — confirm your intake form does this before exposing it as selectable.
>
> **`muu` — unconfirmed, no backend config found either.** Unlike `asuinrakennus`/`teollisuus`/`maatalous`/`liikerakennus` (which all have real `_HANKE_CFG` entries), `muu` has no config entry in `generate_application.py` and, unlike `egs`, there is no code comment confirming this is intentional. A generation request with `muu` would likely produce a similarly empty/generic document, but this has not been confirmed with a live test. Treat `muu` as unconfirmed rather than a normal type until this is checked — flag to backend before relying on it in the intake form.

---

### Country codes (`country`)

9 total — confirmed against `_COUNTRY_CONFIG`'s keys in `generate_application.py`.

| Value | Label (in UI) | Regulatory context |
|---|---|---|
| `FI` | Suomi / Finland | Finnish legislation (default) |
| `SE` | Sverige / Sweden | Swedish MB, PBL, Energilag |
| `DA` | Danmark / Denmark | Danish PBL, VE-loven |
| `NO` | Norge / Norway | Norwegian PBL, NVE-forskrift |
| `PL` | Polska / Poland | Polish Prawo budowlane, OZE |
| `DE` | Deutschland / Germany | German EnWG, EEG, BImSchG |
| `EE` | Eesti / Estonia | Estonian Elektrituruseadus, KeÜS, Ehitusseadustik |
| `LV` | Latvija / Latvia | Latvian Elektroenerģijas tirgus likums, Atjaunojamās enerģijas likums, Būvniecības likums |
| `LT` | Lietuva / Lithuania | Lithuanian Teritorijų planavimo įstatymas, Statybos įstatymas, Aplinkos apsaugos įstatymas |

`country` controls which legislation is cited and which authority names appear in the generated document. The language of the document is controlled separately by `lang`.

---

### Output language codes (`lang`)

10 total — confirmed against `_LANG_INSTRUCTIONS`'s keys in `generate_application.py`.

| Value | Language |
|---|---|
| `FI` | Finnish (default) |
| `EN` | English |
| `SE` | Swedish |
| `DA` | Danish |
| `NO` | Norwegian |
| `PL` | Polish |
| `DE` | German |
| `ET` | Estonian |
| `LV` | Latvian |
| `LT` | Lithuanian |

> **Note the asymmetry:** Estonia's `country` code is `EE`, but its `lang` code is `ET` (not `EE`) — `LV`/`LT` are used unchanged for both `country` and `lang`. Confirmed directly against `_COUNTRY_CONFIG` (uses `EE`) vs. `_LANG_INSTRUCTIONS` (uses `ET`) in `generate_application.py` — not a typo, don't "fix" it to match.

---

### Project phase values (`hankkeen_vaihe`)

Three valid values. Both `rakentaminen` and `rakentamisvaihe` are accepted (aliases).

| Value | Phase | Description |
|---|---|---|
| `esiselvitys` | 1 — Pre-study | Preliminary feasibility, no binding applications yet. Tone: exploratory. |
| `lupavaihe` | 2 — Permit | Active permit applications. Tone: concrete, citing specific permits and attachments. |
| `rakentaminen` | 3 — Construction | Post-permit, under construction. Tone: compliance, inspections, commissioning. |
| `rakentamisvaihe` | 3 — Construction | Alias for `rakentaminen` (both accepted) |

`hankkeen_vaihe` is optional. If omitted, the backend defaults to `esiselvitys` tone.

---

### Phase-lock workflow

Phase-lock is **currently disabled** in production (`PHASE_LOCK_ENABLED=false`). When disabled:
- All three phases are always open
- `GET /api/phase-status` returns `phase_lock_disabled: true`
- `POST /api/complete-phase` always returns `{ "ok": true, "next_phase": N+1 }`
- `POST /api/generate-application` ignores `session_id` for phase validation

When phase-lock is enabled (future), the workflow enforces: esiselvitys → lupavaihe → rakentaminen, with each phase requiring explicit completion before the next is unlocked.

---

### RAG confidence thresholds

The backend may reject document generation if the knowledge base returns too few relevant sources:

| Condition | Result |
|---|---|
| `chunks_found < 5` OR `avg_relevance < 0.65` | **422** `insufficient_sources` — hard stop |
| `chunks_found < 12` OR `avg_relevance < 0.75` | **200** with `⚠️ Expert review recommended` markers in document |
| Normal | **200**, clean document |

When you receive a 422 `insufficient_sources`, do not retry automatically — prompt the user to select a different project type or contact NCE support.

---

### Error format

All 4xx/5xx errors return:
```json
{ "detail": "Error message here" }
```
Except 422 `insufficient_sources` which returns a structured object in `detail` (see above).
