"""
kannattavuuslaskenta (feasibility/ROI) — v1.

Scope, deliberately narrow: indicative capex/opex range + a simple payback
estimate. NOT NPV/IRR, NOT a market-price forecast, NOT a substitute for
professional financial due diligence or a site-specific grid-connection quote.

All benchmark figures are public-source ranges (IRENA, NREL ATB, BloombergNEF,
Ember), not project-specific data. See DISCLAIMER, returned in every result.

Not wired into report generation yet — this module is calculation logic only,
tested in isolation. Integration (PDF section vs. separate output) is a
separate, not-yet-decided step.
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

    # ── Payback (generation techs only — see BESS_PAYBACK_NOTE for why not BESS) ──
    if tech == "battery_storage":
        result["payback_years"] = None
        result["payback_note"] = BESS_PAYBACK_NOTE
        return result

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
