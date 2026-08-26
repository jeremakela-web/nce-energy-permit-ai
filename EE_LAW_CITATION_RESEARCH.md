# Estonia — real statutory equivalents (research pass, not yet built)

Research findings for the backlog's priority item 1 (2026-08-26), EE leg of
the EE/DE/LV pass — continued in a fresh session after [[LV_LAW_CITATION_RESEARCH]].
**This file is research only.** Per instruction, no code changes were made —
findings below are for review before any `_LAW_TRANS`/`_LAW_CITATION_REPLACEMENT`/
`_COUNTRY_LIITTEET` build-out.

Current gap state (re-verified directly against `permit_ai/generate_application.py`,
not assumed from the earlier snapshot):

| | value |
|---|---|
| `_LAW_TRANS` EE entries | 0 |
| `_LAW_CITATION_REPLACEMENT` EE entries | 0 |
| `_STUK_REPLACEMENT["EE"]` | absent |
| `_COUNTRY_LIITTEET["EE"]` hanketyyppi coverage | 5 keys present: `SMR`, `smr_ee`, `tuulivoima_maa`, `tuulivoima_meri`, `datakeskus` — missing `BESS`, `aurinkovoima`, `vesivoima`, `hybridi`, `ymparistolupa`, `offshore_wind` alias |
| `_COUNTRY_LUVAT["EE"]` (permit/authority/law table) | **already fully built**, unlike LV's 0/10 starting point — 11 hanketyyppi + 2 aliases present (`tuulivoima_maa`, `aurinkovoima`, `tuulivoima_meri`+`offshore_wind`, `BESS`, `SMR`+`smr_ee`, `datakeskus`, `teollisuus`, `asuinrakennus`, `liikerakennus`, `maatalous`, `hybridi`) — this table is what's below getting corrected, not built from scratch |

Every statute verified via live web search against primary/near-primary
sources. **Riigiteataja.ee itself is JS-rendered** (same known blocker
pattern as LT's e-seimas.lrs.lt and DA's retsinformation.dk, logged in
[[project-manual-sourcing-backlog]]) — `WebFetch` against `riigiteataja.ee/akt/*`
and `riigiteataja.ee/en/eli/*` URLs returns only a loading shell ("Laeb...")
with no text. Verification instead relied on: Riigi Teataja's own indexed
search-result snippets (title + adoption/in-force metadata, which Google's
index captures even though the live page doesn't render for a fetcher),
Kliimaministeerium's (Ministry of Climate) own pages, Riigikogu's own press
releases, k6k.ee (Keskkonnaõiguse Keskus — Estonian Environmental Law
Centre, a real specialist secondary source), and one successful FAOLEX PDF
mirror fetch (`faolex.fao.org`, a UN FAO legal database that mirrors
national environmental laws — not JS-rendered, but this file's specific PDF
was compressed/non-extractable, so it confirmed existence, not
did not on its own yield full text). Cross-checked every date against at
least two independent snippets before treating it as confirmed.

---

## 1. The five items from the status check

**KeÜS — confirmed real, no correction needed.**
Keskkonnaseadustiku üldosa seadus (General Part of the Environmental Code
Act). Adopted 16.02.2011, published RT I, 28.02.2011, 1. Entry into force
was phased: main body 01.08.2014, parts 01.01.2015 and 01.08.2017. Matches
the code's existing "(KeÜS, RT I 2011)" citation for publication year. Its
real role: the general framework act — fundamental environmental-protection
principles, general permit concepts, EIA general provisions — same
structural role as Latvia's "Vides aizsardzības likums" (2006), **not** the
operative permit-granting law for industrial/IPPC-threshold facilities. See
next item.

**Tööstusheite seadus — confirmed real, and it's the actual YSL-equivalent
(same correction pattern as LV's "Par piesārņojumu").**
Adopted 24.04.2013, published RT I 16.05.2013, in force 01.06.2013.
Implements EU Industrial Emissions Directive 2010/75/EU (confirmed directly
from Kliimaministeerium's own "Tööstusheide ja kemikaalid" page, which
quotes the directive by number and date). Governs the actual
permit-to-operate for industrial installations: "käitistel on õigus
tegutseda ainult juhul, kui neil on selleks luba" (facilities may operate
only if authorised) — the permit is either an integrated permit
(**kompleksluba**) above the IED capacity threshold, or an environmental
permit (**keskkonnaluba**) below it. This is the direct structural analog
to YSL 527/2014 and to LV's "Par piesārņojumu": KeÜS defines the general
concept of *keskkonnaluba*, but Tööstusheite seadus is the specific
operative law that actually triggers and governs the permit for
industrial-scale facilities (the code's current `teollisuus`/`maatalous`
rows cite only KeÜS for this — same wrong-law-as-primary-citation pattern
LV had, not yet corrected here).

**Veeseadus — confirmed real, genuinely new to this codebase (not cited
anywhere currently).**
Passed by Riigikogu 30.01.2019, entered into force 01.10.2019 (a real 2019
recodification, not the older 1994 Veeseadus it replaced — worth noting
explicitly since a stale search hit could easily surface the repealed law's
name unchanged). A later substantive amendment (bureaucracy reduction,
clarified "water body" definition, nitrate-sensitive-area restrictions) was
adopted 13.09.2021, in force 01.10.2021 — this is very likely the date a
less careful search would surface as "the" Veeseadus date; the real
original enactment is 2019. Direct FI-equivalent of Vesilaki 587/2011; not
currently present in `_COUNTRY_LUVAT["EE"]` at all — a real content gap,
not just a citation-accuracy issue.

**Päästeseadus — confirmed real, but it is *not* the law the code actually
needs; flagging a conflation risk rather than a wrong citation.**
Adopted 05.05.2010, published RT I 2010, 24, 115, in force 01.09.2010 (a
companion act to Tuleohutuse seadus below — both passed the same day as
part of the same 2010 rescue-law reform). Its real scope: the Rescue
Board's (Päästeamet) powers and duties in actual rescue operations and
emergency response — organisational/operational law, not fire-safety
technical/permit requirements. There is also a **third**, newer, easily
confused law: Päästeteenistuse seadus (Rescue Service Act, 2022) — governs
rescue-service personnel/organisation, also not the fire-safety-permit law.
None of these three is currently cited in the code.

**The law actually needed for BESS/hybridi fire-safety rows is Tuleohutuse
seadus** (Fire Safety Act) — adopted 05.05.2010, published RT I 2010, 24,
116, in force 01.09.2010. Sets fire-safety technical requirements,
self-inspection/reporting obligations, and inspection cadence (e.g.
buildings >750–1000 m² inspected every 3 years) — the correct analog to
what the code is trying to cite. **Real, small correction needed**: the
code currently spells it `"Tuleohutusseadus"` (one word, no space) at
lines 4555/4621 of `generate_application.py` — the real name is two words,
genitive case: **"Tuleohutuse seadus"**. Same category of fix as LV's
Pelastuslaki correction (a real law, just imprecisely cited) — not
discovered by stopping at the first plausible-looking hit.

**Nuclear framework — confirmed, and this is a major, time-sensitive
correction (same shape as LV's nuclear-framework bug, arguably bigger).**
The code's current `prompt_prefix` and `_COUNTRY_LUVAT["EE"]["SMR"]` both
assert "no nuclear power law yet (draft Tuumaenergia seadus under
development)". **This is now stale, not just conservative.** Verified via
Riigikogu's own press materials and independent news coverage (ERR,
Kliimaministeerium, toostusuudised.ee, world-nuclear-news.org — cross-checked
across 4+ independent sources, all agreeing on the vote count):

- **Tuumaenergia ja -ohutuse seadus** (Nuclear Energy and Safety Act,
  abbreviated **TEOS**, bill 856 SE) passed its **third reading in the
  Riigikogu on 17 June 2026**, 63 in favour, 10 against, 1 abstention —
  a real, decisive, already-completed legislative event, not a live draft.
- **Not yet in force as of today (2026-08-26)** — scheduled entry into
  force is **1 January 2027**. The accurate status is "adopted, not yet in
  force," a third state the current binary "law exists" / "no framework"
  prompt language doesn't have room for.
- Creates a **tiered licensing system**: preliminary assessment →
  construction permit → testing permit → operating permit → decommissioning
  permit, plus a national decommissioning fund.
- Creates an **independent nuclear regulator inside TTJA** (Tarbijakaitse ja
  Tehnilise Järelevalve Amet / Consumer Protection and Technical Regulatory
  Authority) — beginning operations 1 January 2027, same date as the law.
  **TTJA is already in the code's EE authorities list** (as the
  electrical/technical-equipment-safety regulator, Tukes-equivalent) — the
  nuclear regulator is a new function added to an existing, already-cited
  authority, not a brand-new body needing its own line.
- Notable amendment during passage: any actual decision to *build* a plant
  now requires separate Riigikogu approval — the law creates the licensing
  framework but does not itself authorise construction.
- Realistic commercial timeline still long — construction/first power not
  expected before "the mid-2030s at the earliest," per the law's own
  supporting coverage. Same "real framework, no first-of-kind commercial
  process yet" shape as LV's and NO's nuclear findings — not evidence this
  is imminent, just evidence the *legal* gap the code currently asserts is
  closed.

---

## 2. Scope boundary — what this pass did **not** cover

Per the user's explicit list, only the five items above were in scope this
session. Not researched this pass (flagging so it isn't silently assumed
done, same discipline as the LV report's own scoping note):

- Patoturvallisuuslaki 494/2009 equivalent (dam safety) — no EE search done.
- Maa-aineslaki 555/1981 equivalent (extractable land resources).
- Sähkömarkkinalaki 588/2013 — EE's Elektrituruseadus (ETS) is already
  cited extensively throughout `_COUNTRY_LUVAT["EE"]`, but wasn't
  independently re-verified against Riigi Teataja this pass (inherited as
  presumed-correct from the pre-existing content, same as MRL/Rakentamislaki/
  YVA-laki above).
- `_PDF_STRINGS["EE"]` / `kaava_SMR` / `kaava_aurinkovoima` card-text audit
  (the check LV's report did in its own §5) — not done for EE this pass.
- The actual `_LAW_TRANS`/`_LAW_CITATION_REPLACEMENT`/`_STUK_REPLACEMENT`/
  `_COUNTRY_LIITTEET` build-out — explicitly deferred, awaiting review.

---

**Summary table for review:**

| FI statute | EE equivalent | Status |
|---|---|---|
| YSL 527/2014 | **Tööstusheite seadus** (2013, in force 01.06.2013) | Correction — KeÜS is currently over-cited as the permit-granting law; Tööstusheite seadus is the real IPPC/industrial-permit analog |
| Vesilaki 587/2011 | **Veeseadus** (2019, in force 01.10.2019) | New — not cited anywhere in the code currently |
| Pelastuslaki 379/2011 | **Tuleohutuse seadus** (2010, in force 01.09.2010) | Correction — spelling/naming fix only (`"Tuleohutusseadus"` → `"Tuleohutuse seadus"`); concept was already right |
| Ydinenergialaki 990/1987 | **Tuumaenergia ja -ohutuse seadus (TEOS)**, adopted 17.06.2026, in force 01.01.2027 | Major correction — code asserts "no law," real law is adopted and has a firm in-force date |
| (general framework, not FI-mapped 1:1) | Keskkonnaseadustiku üldosa seadus (KeÜS) | Confirmed, no change — correctly cited already, just not the operative permit law on its own |
| (not FI-mapped — flag only) | Päästeseadus (2010) / Päästeteenistuse seadus (2022) | Confirmed real, but neither is what the code needs — flagged to prevent future conflation with Tuleohutuse seadus |

Ready for review before any build step.
