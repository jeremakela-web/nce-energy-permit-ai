"""
kannattavuuslaskenta (feasibility/ROI) — v1.

Scope, deliberately narrow: indicative capex/opex range + a simple payback
estimate. NOT NPV/IRR, NOT a market-price forecast, NOT a substitute for
professional financial due diligence or a site-specific grid-connection quote.

All benchmark figures are public-source ranges (IRENA, NREL ATB, BloombergNEF,
Ember), not project-specific data. See DISCLAIMER, returned in every result.

Wired into the PDF report as an optional "Kannattavuusarvio" section (see
generate_application.py). BESS gets a computed payback estimate only for
countries with real sourced ancillary-market revenue data (FI/DA/DE as of
Phase 1b, see _ANCILLARY_REVENUE_EUR_MW_YEAR) — everywhere else keeps the
original v1 "not provided" behaviour (BESS_PAYBACK_NOTE), same as generation
techs without a sourced electricity price.
"""
from __future__ import annotations

# ── Currency ──────────────────────────────────────────────────────────────
# All source data is published in USD. Converted at display-time with an
# explicit, separately-visible approximate rate rather than baking a
# silently-chosen FX assumption into the benchmark numbers themselves —
# keeps "the cost range is approximate" and "the FX rate is approximate"
# as two distinct, visible approximations instead of one compounded one.
USD_TO_EUR = 0.92
USD_TO_EUR_ASOF = "2026-07-25"  # update this line periodically; not live-fetched

# Poland's PSE ancillary-market data (see _ANCILLARY_REVENUE_EUR_MW_YEAR, "PL")
# is published in PLN, the only entry in that table needing FX conversion —
# same "separate visible approximation" principle as USD_TO_EUR above.
PLN_TO_EUR = 1 / 4.3135
PLN_TO_EUR_ASOF = "2026-07-31"  # Frankfurter/ECB reference rate; not live-fetched

# ── hanketyyppi → technology category ───────────────────────────────────
# Hanketyyppis with no renewable-generation cost profile (datakeskus,
# asuinrakennus, ...) are intentionally absent from this map.
_HANKETYYPPI_TECH: dict[str, str] = {
    "BESS":            "battery_storage",
    "tuulivoima_maa":  "onshore_wind",
    "tuulivoima_meri": "offshore_wind",
    "aurinkovoima":    "solar_pv",
}

# ── CAPEX benchmarks (USD; single-point published averages, not ranges) ──
_CAPEX_USD_PER_KW: dict[str, tuple[float, str]] = {
    "solar_pv":      (691,  "IRENA, Renewable Power Generation Costs in 2024 (global weighted-average installed cost)"),
    "onshore_wind":  (1041, "IRENA, Renewable Power Generation Costs in 2024 (global weighted-average installed cost)"),
    "offshore_wind": (2852, "IRENA, Renewable Power Generation Costs in 2024 (global weighted-average installed cost)"),
}
# BESS is priced per kWh of storage capacity, not per kW of power — kept separate.
_CAPEX_USD_PER_KWH_BESS: tuple[tuple[float, float], str] = (
    (150, 250),
    "NREL ATB 2025 (current utility-scale turnkey system cost range, incl. balance-of-system) "
    "and BloombergNEF (Europe-specific $180-260/kWh range for 4h LFP systems)",
)

# ── OPEX benchmarks (% of capex per year) ────────────────────────────────
_OPEX_PCT_PER_YEAR: dict[str, tuple[tuple[float, float], str]] = {
    "solar_pv": (
        (1.5, 3.0),
        "Industry O&M benchmark surveys (moderate confidence)",
    ),
    "onshore_wind": (
        (2.0, 4.0),
        "WIDE RANGE — industry sources conflict on wind O&M costs. One IRENA-attributed "
        "figure found ~4%/year; general industry expectation is closer to 2-3%/year. "
        "Presented as a wide hedge rather than resolved (low confidence).",
    ),
    "offshore_wind": (
        (2.0, 3.0),
        "WIDE RANGE — industry sources conflict on wind O&M costs. One IRENA-attributed "
        "figure found 2.0-2.2%/year, counter-intuitively lower than the onshore figure "
        "from the same source; widened rather than trusted at face value (low confidence).",
    ),
    "battery_storage": (
        (1.2, 2.5),
        "Industry O&M benchmark surveys — 1.2-1.5%/year for top-performing fleets, "
        "2-2.5%/year typical (moderate confidence)",
    ),
}

# ── Electricity price benchmarks (HISTORICAL, EUR/MWh — not a forecast) ──
# Sparse by design: only countries with a specifically-sourced figure get an
# entry here. Everything else uses an explicit generic placeholder rather
# than a guessed number. Extend this table opportunistically as real
# country-specific figures are sourced — do not fabricate entries.
_ELECTRICITY_PRICE_EUR_MWH: dict[str, tuple[tuple[float, float], str]] = {
    "FI": (
        (40, 50),
        "Finland Q4 2025 wholesale day-ahead average ~44.3 EUR/MWh (Elenger power market overview, Q4 2025)",
    ),
    "EE": (
        (85, 111),
        "Baltic states (EE/LV/LT) Q4 2025 wholesale range 86-111 EUR/MWh — source did not break this "
        "out per country, same figure used for EE/LV/LT (Elenger power market overview, Q4 2025)",
    ),
    "LV": (
        (85, 111),
        "See EE — same shared Baltic-region source, not broken out per country",
    ),
    "LT": (
        (85, 111),
        "See EE — same shared Baltic-region source, not broken out per country",
    ),
}
_ELECTRICITY_PRICE_FALLBACK: tuple[tuple[float, float], str] = (
    (40, 90),
    "No country-specific historical price sourced for this country in v1 — generic wide "
    "placeholder band covering typical Nordic-to-Central-European wholesale ranges. "
    "Needs a dedicated per-country data pass before this should be trusted.",
)

# ── Ancillary/balancing-market revenue benchmarks (EUR/MW/year) ─────────
# BESS ancillary-revenue sourcing effort — replaces the Clean Horizon
# Storage Index (third-party commercial index) with official/primary TSO
# and ENTSO-E-family data. Phase 1: FI, DA, DE. Phase 2: EE, LV, LT.
# Phase 3: SE, NO, PL (this batch) — table is now complete for all ten
# countries in scope; no fabricated entries were ever added.
#
# METHODOLOGY (read before adding countries or using these numbers
# elsewhere — this is the single most important decision in this table):
#
# - Product: aFRR (automatic Frequency Restoration Reserve) CAPACITY
#   market, UP and DOWN directions summed, for every country in this
#   table — including EE/LV/LT (Phase 2), for consistency with the
#   FI/DA/DE (Phase 1) methodology. Chosen because it is the one product
#   with genuinely accessible, no-registration-barrier official data
#   across all six countries so far via one consistent methodology — not
#   because it is necessarily the single largest real BESS revenue
#   stream in any of these markets.
# - FI was originally planned as FCR-N (the Nordic-specific reserve
#   product, and historically the more commonly cited BESS revenue
#   reference in Finland specifically) but Fingrid's Open Data API
#   requires a free, self-service API key (email registration) that
#   wasn't available for this pass. FI uses the same aFRR product as
#   DA/DE instead, sourced from the pan-Nordic Energinet dataset (which
#   happens to also publish FI's aFRR capacity prices). Revisit with a
#   real FCR-N figure once a Fingrid key exists — FCR-N and aFRR are
#   different markets and are NOT expected to give the same number.
# - EE/LV/LT (Phase 2): sourced from the joint Baltic Transparency
#   Dashboard (api-baltic.transparency-dashboard.eu), which covers all
#   three countries from one API — treated as one combined effort, per
#   the original sourcing investigation's recommendation. The Baltic
#   states adopted the continental aFRR/mFRR capacity-market structure
#   only in 2025, after leaving BRELL — this is a genuinely young market
#   and each entry below carries an explicit, stronger maturity caveat
#   than FI/DA/DE. The series' actual first valid data point is
#   2025-09-29, not the Baltic Balancing Capacity Market's nominal
#   4-Feb-2025 FCR+mFRR launch or its 15-Apr-2025 aFRR-specific launch —
#   real usable history is ~10 months as of this pass, even less than
#   the ~18 months a naive launch-date estimate would suggest. Note also
#   that LV and LT cleared at (near-)identical marginal prices for most
#   of the period — the joint Baltic auction mechanism optimises across
#   all three countries simultaneously, so this is expected, not a
#   data error.
# - This is ONE product's capacity revenue, NOT full BESS revenue. Real
#   BESS assets typically stack multiple products (FCR + aFRR + mFRR +
#   arbitrage) simultaneously and dynamically. This figure is what a
#   battery bidding symmetric aFRR capacity year-round would earn from
#   THAT product alone. This data existing does not, by itself, make a
#   BESS payback estimate defensible — see BESS_PAYBACK_NOTE for
#   countries without an entry here, and the mandatory payback_note text
#   built in calculate_feasibility() for countries with one.
# - SE/NO (Phase 3): sourced from the same Energinet 'AfrrReservesNordic'
#   dataset already used for FI/DA — it covers all Nordic price zones, so
#   no new data source or registration was needed (Mimer/eSett and
#   Statnett's own portal, both originally flagged as unknowns, turned out
#   to be unnecessary). Like DA, both are genuine multi-zone markets — SE
#   has 4 zones (SE1-SE4), NO has 5 (NO1-NO5). Same "don't average
#   distinct zones into a false single number" principle as DA: reported
#   as (lowest zone, highest zone), NOT a confidence range, with the
#   BESS payback calculation defaulting to the lower/conservative zone
#   (see _ZONE_NOTE_BY_COUNTRY below). SE's spread (~19%) and NO's
#   (~64%) are materially different in character — SE's zones sit
#   moderately apart, NO's reflect a genuine outlier (NO5) rather than a
#   tight cluster; each entry's source string says so explicitly rather
#   than presenting both spreads the same way.
# - PL (Phase 3): sourced from PSE's (Polskie Sieci Elektroenergetyczne,
#   the Polish TSO) public raporty.pse.pl API, report 'cmbp-tp' (base-mode
#   procured balancing capacity prices). CORRECTION to this project's own
#   prior investigation-phase finding: a small sample had suggested
#   afrr_d (down) and afrr_g (up) price identically, which would have
#   allowed a single figure with no UP/DOWN summing. A full check across
#   the entire 12-month, 8,784-hour pull found they match on only 5 rows
#   (0.06%) — genuinely asymmetric, same UP+DOWN-summed treatment as
#   every other country in this table. Published in PLN — the only
#   country here needing FX conversion (see PLN_TO_EUR). A second
#   'uzupełniający' (supplementary) procurement mode exists as a
#   separate product (report 'cmbu-tu') and is deliberately NOT included,
#   same single-market discipline as elsewhere in this table. Poland's
#   figure is notably higher than every other country here — plausibly
#   real reserve scarcity (a documented characteristic of Poland's grid),
#   not a data error, but treated as less certain than the mature
#   markets' figures precisely because it's an outlier.
# - Annualization: raw 15-minute (Baltic Transparency Dashboard), hourly
#   (Energinet, PSE), or 4-hour-block (regelleistung.net) capacity
#   clearing prices, averaged over the sourced period, multiplied by
#   8760 h/year. Assumes constant year-round capacity commitment;
#   real-world availability/eligibility varies and isn't modeled here.
_ANCILLARY_REVENUE_EUR_MW_YEAR: dict[str, tuple[tuple[float, float], str]] = {
    "FI": (
        (113800, 113800),
        "aFRR capacity market (UP+DOWN summed), 2025-08-01 to 2026-08-01, "
        "Energinet Energi Data Service 'AfrrReservesNordic' dataset (FI price "
        "area) — computed 2026-08-01. NOT FCR-N (the originally-planned "
        "Nordic product); substituted because Fingrid's own API requires a "
        "registration key unavailable for this pass — see methodology note "
        "above. Nordic aFRR capacity market is a relatively young product "
        "(live since ~2024) — moderate maturity, not a multi-year track record.",
    ),
    "DA": (
        (129700, 155900),
        "aFRR capacity market (UP+DOWN summed), 2025-08-01 to 2026-08-01, "
        "Energinet Energi Data Service 'AfrrReservesNordic' dataset — "
        "computed 2026-08-01. Range reflects Denmark's two price zones "
        "(DK2 129,700 / DK1 155,900 EUR/MW/year) — NOT a confidence range. "
        "DK1 is continental-synchronous, DK2 is Nordic-synchronous — "
        "genuinely different markets under one country code. Denmark's aFRR "
        "capacity market has several years of history — moderate-to-high "
        "maturity (DK1's local capacity market specifically only launched "
        "Oct 2024, so DK1's figure carries less history than DK2's).",
    ),
    "DE": (
        (257300, 257300),
        "aFRR capacity market (UP+DOWN summed), 2025-08-01 to 2026-08-01, "
        "365/365 days of daily auction data from regelleistung.net (joint "
        "50Hertz/Amprion/TenneT/TransnetBW platform), no gaps — computed "
        "2026-08-01. Germany's aFRR market has 5+ years of history — the "
        "most stable and best-established of the three Phase-1 countries.",
    ),
    "EE": (
        (200800, 200800),
        "aFRR capacity market (UP+DOWN summed), joint Baltic Transparency "
        "Dashboard (api-baltic.transparency-dashboard.eu), 'price_procured_"
        "reserves' series — computed 2026-08-01. Data period used: "
        "2025-09-29 to 2026-07-31 (~10 months) — this is the series' actual "
        "first valid data point, not the Baltic capacity market's nominal "
        "Feb/Apr-2025 launch dates. NEW MARKET (~10 months of history as of "
        "2026-08-01, post-BRELL 2025) — moderate-to-low confidence, NOT "
        "comparable to FI/DA/DE's multi-year track record. Treat this figure "
        "as substantially more likely to shift as the market matures than "
        "the Phase-1 countries' figures.",
    ),
    "LV": (
        (322300, 322300),
        "aFRR capacity market (UP+DOWN summed), joint Baltic Transparency "
        "Dashboard (api-baltic.transparency-dashboard.eu), 'price_procured_"
        "reserves' series — computed 2026-08-01. Data period used: "
        "2025-09-29 to 2026-07-31 (~10 months) — this is the series' actual "
        "first valid data point, not the Baltic capacity market's nominal "
        "Feb/Apr-2025 launch dates. NEW MARKET (~10 months of history as of "
        "2026-08-01, post-BRELL 2025) — moderate-to-low confidence, NOT "
        "comparable to FI/DA/DE's multi-year track record. Notably higher "
        "than FI/DA/DE — plausibly reflects genuine scarcity in a young, "
        "still-forming market rather than a stable long-run price; treat "
        "this figure as substantially more likely to shift than the Phase-1 "
        "countries' figures. LV and LT cleared at near-identical prices for "
        "most of the period (joint Baltic auction, not a data error).",
    ),
    "LT": (
        (322600, 322600),
        "aFRR capacity market (UP+DOWN summed), joint Baltic Transparency "
        "Dashboard (api-baltic.transparency-dashboard.eu), 'price_procured_"
        "reserves' series — computed 2026-08-01. Data period used: "
        "2025-09-29 to 2026-07-31 (~10 months) — this is the series' actual "
        "first valid data point, not the Baltic capacity market's nominal "
        "Feb/Apr-2025 launch dates. NEW MARKET (~10 months of history as of "
        "2026-08-01, post-BRELL 2025) — moderate-to-low confidence, NOT "
        "comparable to FI/DA/DE's multi-year track record. Notably higher "
        "than FI/DA/DE — plausibly reflects genuine scarcity in a young, "
        "still-forming market rather than a stable long-run price; treat "
        "this figure as substantially more likely to shift than the Phase-1 "
        "countries' figures. LV and LT cleared at near-identical prices for "
        "most of the period (joint Baltic auction, not a data error).",
    ),
    "SE": (
        (192500, 229900),
        "aFRR capacity market (UP+DOWN summed), 2025-08-01 to 2026-08-01, "
        "Energinet Energi Data Service 'AfrrReservesNordic' dataset — "
        "computed 2026-08-01. Range reflects Sweden's four price zones "
        "(SE1 192,500 low / SE3 229,900 high EUR/MW/year; SE2 210,800 and "
        "SE4 216,800 fall in between) — NOT a confidence range. Spread is "
        "~19% low-to-high — moderately, not tightly, clustered (above the "
        "~10-15% threshold that would justify treating this as one "
        "number), so presented as a genuine zone range. Sweden's aFRR "
        "capacity market has several years of history — mature, "
        "comparable to FI/DA/DE.",
    ),
    "NO": (
        (116700, 191700),
        "aFRR capacity market (UP+DOWN summed), 2025-08-01 to 2026-08-01, "
        "Energinet Energi Data Service 'AfrrReservesNordic' dataset — "
        "computed 2026-08-01. Range reflects Norway's five price zones "
        "(NO5 116,700 low / NO3 191,700 high EUR/MW/year; NO1 160,500, "
        "NO2 173,900, NO4 184,900 fall in between) — NOT a confidence "
        "range. Spread is ~64% low-to-high — genuinely wide, not "
        "clustered: NO5 (Bergen area, hydro-dominated, historically "
        "prone to decoupling from the rest of Norway's grid) clears "
        "notably lower than the other four zones, which sit closer "
        "together. Norway's aFRR capacity market has several years of "
        "history — mature, comparable to FI/DA/DE.",
    ),
    "PL": (
        (664600, 664600),
        "aFRR capacity market (UP+DOWN summed), 2025-08-01 to 2026-08-01, "
        "PSE (Polskie Sieci Elektroenergetyczne) public raporty.pse.pl "
        "API, 'cmbp-tp' report (base-mode procured balancing capacity "
        "prices) — computed 2026-08-01. CORRECTION to this project's own "
        "investigation-phase finding: afrr_d/afrr_g were assumed to "
        "price symmetrically from a small sample; a full check of all "
        "8,784 hourly rows found only 5 matches (0.06%) — genuinely "
        "asymmetric, so this figure sums UP+DOWN like every other "
        "country here, not a shared single-direction number. Published "
        "in PLN, converted at 1 EUR ≈ 4.3135 PLN (Frankfurter/ECB "
        "reference rate, 2026-07-31) — the only country in this table "
        "needing FX conversion (see PLN_TO_EUR), kept as a separate "
        "visible approximation rather than folded into the benchmark "
        "number. A second 'uzupełniający' (supplementary) "
        "procurement mode exists as a separate product and is "
        "deliberately excluded — single-market discipline, same as "
        "elsewhere in this table. Poland's capacity market has ~14 "
        "months of history (from 2024-06-14) as of this pass — a "
        "genuine intermediate maturity tier, more established than the "
        "Baltic states' ~10 months but well short of FI/DA/DE/SE/NO's "
        "multi-year track record. Notably higher than every other "
        "country in this table (~2.6x DE's rate) — plausibly reflects "
        "real reserve scarcity, a widely-documented characteristic of "
        "Poland's balancing market, but treat this as more likely to "
        "shift than the mature markets' figures.",
    ),
}

# Per-country zone_note text for the BESS payback calculation below — only
# for countries whose _ANCILLARY_REVENUE_EUR_MW_YEAR entry is a genuine
# multi-zone range (rev_lo != rev_hi), not a confidence interval. Absent
# entries (single-point countries) get no zone_note, since there's no zone
# choice to explain.
_ZONE_NOTE_BY_COUNTRY: dict[str, str] = {
    "DA": (
        " Uses the DK2 (Nordic-synchronous) price zone — the lower and "
        "more conservative of Denmark's two aFRR markets. DK1 "
        "(continental-synchronous) is higher; no per-project zone "
        "selector exists yet, so this is a conservative default, not a "
        "determination of which zone this specific project is in. A "
        "proper zone selector is a future improvement."
    ),
    "SE": (
        " Uses the SE1 (northernmost) price zone — the lower and more "
        "conservative of Sweden's four aFRR zones (SE1-SE4). SE3 is "
        "highest; no per-project zone selector exists yet, so this is a "
        "conservative default, not a determination of which zone this "
        "specific project is in. A proper zone selector is a future "
        "improvement."
    ),
    "NO": (
        " Uses the NO5 (Bergen area, hydro-dominated) price zone — the "
        "lower and by far the most conservative of Norway's five aFRR "
        "zones (NO1-NO5); NO5 clears notably lower than the other four. "
        "NO3 is highest; no per-project zone selector exists yet, so "
        "this is a conservative default, not a determination of which "
        "zone this specific project is in. A proper zone selector is a "
        "future improvement."
    ),
}

# ── Typical capacity factors (generation techs only — indicative) ────────
_CAPACITY_FACTOR: dict[str, float] = {
    "solar_pv":      0.12,  # Nordic/Baltic-latitude solar — not a Southern-Europe figure
    "onshore_wind":  0.30,
    "offshore_wind": 0.45,
}

DISCLAIMER = (
    "INDICATIVE ONLY — NOT INVESTMENT-GRADE. These figures are derived from public "
    "benchmark ranges (IRENA, NREL ATB, BloombergNEF, Ember), not project-specific quotes, "
    "grid-connection assessments, or market forecasts. Actual project economics depend on "
    "site-specific factors (grid connection cost, land cost, financing terms, local support "
    "schemes) not captured here. Do not use for investment decisions without professional "
    "financial due diligence."
)

BESS_PAYBACK_NOTE = (
    "Payback estimate not provided for battery storage in v1. BESS revenue comes from "
    "arbitrage spread (charge low, discharge high) or ancillary-services/capacity-market "
    "payments — not a flat energy price times output, the way generation-asset revenue "
    "can be roughly approximated. Modeling that correctly needs market-spread data this "
    "version doesn't have; a naive flat-price estimate would be actively misleading rather "
    "than just imprecise, so it's omitted rather than guessed."
)

# Mandatory whenever electricity_price_source comes from ENTSO-E (real day-ahead
# market data, see entsoe_prices.py) rather than the static IRENA/NREL/BloombergNEF
# benchmark table. Solar/wind generation clusters when many producers generate
# simultaneously — exactly when day-ahead prices tend to be lowest ("cannibalization
# effect", well-documented and worsening with rising renewable penetration). A flat
# average price systematically OVERSTATES achievable revenue. For solar, this figure
# already uses a daylight-hours proxy (see entsoe_prices.daylight_hours_average) as a
# partial, honestly-bounded mitigation — a real time-of-day filter computed from real
# fetched data, NOT a true generation-weighted curve (that needs irradiance data this
# project doesn't have). Wind gets no such proxy — there is no defensible time-of-day
# window for wind's generation pattern, and inventing one would be worse than not
# having it.
ENTSOE_CANNIBALIZATION_NOTE = (
    "This price is real historical day-ahead market data (ENTSO-E Transparency "
    "Platform), not a static benchmark — but it is still a FLAT AVERAGE price, and "
    "solar/wind revenue in practice is affected by the 'cannibalization effect': "
    "generation clusters exactly when many other solar/wind producers are also "
    "generating, which is typically when prices are lowest. A flat-average estimate "
    "therefore tends to OVERSTATE achievable revenue, more so for solar than wind and "
    "more so in markets with high renewable penetration. For solar, this figure uses "
    "a daylight-hours-only average as a partial mitigation (a simplified time-of-day "
    "proxy, not a true generation-weighted curve — see methodology). For wind, no "
    "such adjustment is applied. Treat this as a better-than-static-benchmark "
    "estimate, not a precise revenue forecast."
)


def calculate_feasibility(
    hanketyyppi: str,
    teho_mw: float | None,
    kapasiteetti_mwh: float | None,
    country: str,
) -> dict | None:
    """
    Compute an indicative capex/opex range and (for generation techs only) a
    simple payback estimate. Returns None if hanketyyppi has no renewable-
    generation cost profile (e.g. datakeskus, asuinrakennus) or if the
    required scale input (teho_mw / kapasiteetti_mwh) is missing.
    """
    tech = _HANKETYYPPI_TECH.get(hanketyyppi)
    if tech is None:
        return None

    result: dict = {
        "hanketyyppi": hanketyyppi,
        "technology": tech,
        "disclaimer": DISCLAIMER,
        "fx_rate_note": f"1 USD ≈ {USD_TO_EUR} EUR (approximate, as of {USD_TO_EUR_ASOF})",
        "sources": [],
    }

    # ── CAPEX ──
    if tech == "battery_storage":
        if not kapasiteetti_mwh or kapasiteetti_mwh <= 0:
            return None
        (lo_usd_kwh, hi_usd_kwh), capex_source = _CAPEX_USD_PER_KWH_BESS
        capacity_kwh = kapasiteetti_mwh * 1000
        capex_lo_eur = capacity_kwh * lo_usd_kwh * USD_TO_EUR
        capex_hi_eur = capacity_kwh * hi_usd_kwh * USD_TO_EUR
    else:
        if not teho_mw or teho_mw <= 0:
            return None
        capex_usd_kw, capex_source = _CAPEX_USD_PER_KW[tech]
        capacity_kw = teho_mw * 1000
        # Single-point published average, not a sourced range — capex_lo == capex_hi.
        # Kept as a (lo, hi) tuple for a uniform result shape across technologies.
        capex_lo_eur = capacity_kw * capex_usd_kw * USD_TO_EUR
        capex_hi_eur = capex_lo_eur

    result["capex_eur"] = (round(capex_lo_eur), round(capex_hi_eur))
    result["capex_source"] = capex_source
    result["sources"].append(capex_source)

    # ── OPEX ──
    (opex_lo_pct, opex_hi_pct), opex_source = _OPEX_PCT_PER_YEAR[tech]
    opex_lo_eur = capex_lo_eur * opex_lo_pct / 100
    opex_hi_eur = capex_hi_eur * opex_hi_pct / 100
    result["opex_eur_per_year"] = (round(opex_lo_eur), round(opex_hi_eur))
    result["opex_pct_range"] = (opex_lo_pct, opex_hi_pct)
    result["opex_source"] = opex_source
    result["sources"].append(opex_source)

    # ── Payback ──
    if tech == "battery_storage":
        # Only countries with a real, sourced _ANCILLARY_REVENUE_EUR_MW_YEAR
        # entry get a computed payback (FI/DA/DE as of Phase 1b). Everyone
        # else keeps the original v1 behaviour unchanged — no fabricated
        # numbers, same BESS_PAYBACK_NOTE as before.
        ancillary = _ANCILLARY_REVENUE_EUR_MW_YEAR.get(country)
        if ancillary is None or not teho_mw or teho_mw <= 0:
            result["payback_years"] = None
            result["payback_note"] = BESS_PAYBACK_NOTE
            if ancillary is not None and (not teho_mw or teho_mw <= 0):
                # Ancillary data exists for this country but we can't use it —
                # power rating (teho_mw) is required and wasn't provided.
                result["payback_note"] += (
                    " (Ancillary-revenue data is available for this country, but "
                    "computing a payback estimate from it requires the project's "
                    "power rating (teho_mw), which was not provided for this request.)"
                )
            return result

        (rev_lo, rev_hi), ancillary_source = ancillary
        # DA/SE/NO are genuine multi-zone markets (rev_lo != rev_hi is a real
        # zone split, not a confidence range — see the dict's own source
        # comment for each). No per-project zone input exists yet, so every
        # country here defaults to rev_lo — the lower, more conservative zone
        # — rather than averaging, which would misrepresent both/all zones.
        # For FI/DE/EE/LV/LT/PL, rev_lo == rev_hi (single-point estimate) so
        # this is a no-op.
        annual_revenue_eur_mw = rev_lo
        zone_note = _ZONE_NOTE_BY_COUNTRY.get(country, "")

        annual_revenue_eur = teho_mw * annual_revenue_eur_mw
        result["ancillary_revenue_eur_mw_year"] = (rev_lo, rev_hi)
        result["ancillary_revenue_source"] = ancillary_source
        result["sources"].append(ancillary_source)

        # Revenue is a single figure (no sourced range) — capex/opex ranges
        # (a real range for BESS, unlike generation capex) still give a
        # fast/slow spread. Fast: cheap capex + cheap opex. Slow: opposite.
        net_cf_fast = annual_revenue_eur - opex_lo_eur
        net_cf_slow = annual_revenue_eur - opex_hi_eur
        payback_fast = (capex_lo_eur / net_cf_fast) if net_cf_fast > 0 else None
        payback_slow = (capex_hi_eur / net_cf_slow) if net_cf_slow > 0 else None
        result["payback_years"] = (
            round(payback_fast, 1) if payback_fast is not None else None,
            round(payback_slow, 1) if payback_slow is not None else None,
        )

        result["payback_note"] = (
            "BESS payback estimate reflects ONLY aFRR capacity-market revenue "
            "(UP+DOWN summed) — NOT energy arbitrage, NOT FCR or mFRR, NOT any "
            "other stacked revenue stream. Real BESS assets typically combine "
            "multiple revenue streams simultaneously and dynamically; this is a "
            "single-market estimate, not full project economics." + zone_note +
            f" Source: {ancillary_source}"
        )
        if payback_fast is None or payback_slow is None:
            result["payback_note"] += (
                " Net cash flow was negative in at least one scenario (opex "
                "exceeding ancillary revenue at these assumptions) — payback not "
                "computable for that scenario; treat as a signal the assumptions "
                "need review, not a literal 'never pays back' claim."
            )
        return result

    # Three-tier price lookup: real ENTSO-E day-ahead data (if a fresh cache
    # entry exists — see entsoe_prices.py) > static per-country benchmark >
    # generic wide placeholder. Lazy, try/except import: feasibility.py stays
    # usable with zero dependencies (no Redis needed) when the ENTSO-E cache
    # simply isn't populated yet or Redis is unreachable — same "never let an
    # optional data source break the whole calculation" discipline as the
    # BESS ancillary-revenue lookup above.
    _entsoe_result = None
    try:
        from entsoe_prices import get_cached_price_eur_mwh as _get_entsoe_price
        _entsoe_result = _get_entsoe_price(country, tech=tech)
    except Exception:
        _entsoe_result = None

    if _entsoe_result is not None:
        price_range, price_source = _entsoe_result
        result["electricity_price_note"] = ENTSOE_CANNIBALIZATION_NOTE
    else:
        price_range, price_source = _ELECTRICITY_PRICE_EUR_MWH.get(country, _ELECTRICITY_PRICE_FALLBACK)
    price_lo, price_hi = price_range
    result["electricity_price_eur_mwh"] = price_range
    result["electricity_price_source"] = price_source
    result["sources"].append(price_source)

    capacity_factor = _CAPACITY_FACTOR[tech]
    annual_output_mwh = teho_mw * 8760 * capacity_factor
    result["annual_output_mwh_estimate"] = round(annual_output_mwh)
    result["capacity_factor_assumed"] = capacity_factor

    # Fast case: low capex, low opex, high price. Slow case: opposite.
    revenue_hi = annual_output_mwh * price_hi
    revenue_lo = annual_output_mwh * price_lo
    net_cf_fast = revenue_hi - opex_lo_eur
    net_cf_slow = revenue_lo - opex_hi_eur

    payback_fast = (capex_lo_eur / net_cf_fast) if net_cf_fast > 0 else None
    payback_slow = (capex_hi_eur / net_cf_slow) if net_cf_slow > 0 else None
    result["payback_years"] = (
        round(payback_fast, 1) if payback_fast is not None else None,
        round(payback_slow, 1) if payback_slow is not None else None,
    )
    if payback_fast is None or payback_slow is None:
        result["payback_note"] = (
            "Net cash flow was negative in at least one scenario (opex exceeding revenue "
            "at the assumed price/capacity-factor combination) — payback not computable "
            "for that scenario; treat as a strong signal the assumptions need review, not "
            "a literal 'never pays back' claim."
        )
    return result
