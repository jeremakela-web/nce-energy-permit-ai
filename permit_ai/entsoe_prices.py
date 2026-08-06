"""
ENTSO-E Transparency Platform day-ahead electricity prices (documentType A44).

Sourcing effort parallel to feasibility.py's BESS ancillary-revenue table
(_ANCILLARY_REVENUE_EUR_MW_YEAR) but for solar/wind's existing
_ELECTRICITY_PRICE_EUR_MWH benchmark — same country set (FI/SE/DA/NO/PL/
DE/EE/LV/LT), same "real primary-source data, not fabricated" discipline.

ARCHITECTURE (read before touching this file):

- Fetch is a *background* concern, not a request-path concern. Nothing in
  this module makes a live HTTP call from calculate_feasibility()'s call
  path — a live day-ahead fetch is a multi-second, multi-request, possibly-
  failing operation and has no business blocking a user-facing PDF request.
  refresh_all_prices() is invoked out-of-band (ARQ daily cron + an admin
  endpoint for manual triggering, see backend/main.py) and writes results
  into Redis. get_cached_price_eur_mwh() is the ONLY function feasibility.py
  calls, and it is a synchronous, non-raising Redis GET with a staleness
  check — safe to call inline, always falls back to None (never an
  exception) so feasibility.py's existing static-benchmark fallback chain
  keeps working even if Redis is down or has never been populated.

- Persistence: Redis, not a local file. Render's only persistent disk is
  already claimed by ChromaDB (see render.yaml) and is mounted on exactly
  one service — the web service's own filesystem is ephemeral and gets
  wiped on every deploy. Redis (REDIS_URL) is already provisioned for ARQ
  and is the only durable, cross-restart store actually available here.

- Multi-zone countries (SE/NO/DK — same countries the BESS ancillary-
  revenue table already had to handle): fetched per-zone, reported as
  (lowest zone avg, highest zone avg) with an explicit zone_note, same
  shape and same "not a confidence range" discipline as
  _ANCILLARY_REVENUE_EUR_MW_YEAR / _ZONE_NOTE_BY_COUNTRY in feasibility.py.
  No single-zone "default" is picked arbitrarily — seehe investigation
  that preceded this module for why an arbitrary single-zone default was
  rejected in favour of reusing the already-shipped range pattern.

- TWO REAL METHODOLOGY HAZARDS, both handled explicitly (not glossed over):

  (a) GRANULARITY CHANGE, 2025-09-30/2025-10-01: the pan-European SDAC
      day-ahead market switched from 60-minute to 15-minute price
      resolution. Any "most recent 12 months" pull straddles this. A
      NAIVE flat average over raw rows silently mis-weights the two
      regimes: e.g. for a Aug2025-Aug2026 pull, ~2 months are PT60M
      (1,464 rows) and ~10 months are PT15M (~29,184 rows) — a flat
      per-row average gives the PT15M regime ~95% of the weight when the
      correct time-weighted split is ~83%/17%. _time_weighted_average()
      below weights every point by its own interval duration
      (resolution), not by row count, which is the only correct way to
      average a mixed-resolution series. See test_time_weighted_average_*
      in this file's self-test block for a worked numeric proof this
      actually matters (not just a theoretical concern).

  (b) CANNIBALIZATION EFFECT: solar (and to a lesser, less time-of-day-
      predictable extent, wind) generation clusters when many producers
      are generating simultaneously, which is exactly when day-ahead
      prices tend to be lowest — a flat average price systematically
      OVERSTATES achievable solar/wind revenue. This module does NOT
      pretend to solve this with a real generation-weighted curve (that
      needs irradiance/wind-speed data this project doesn't have — would
      be fabricated precision, not a real fix). What it DOES do,
      honestly bounded: for solar specifically, alongside the flat
      time-weighted average it also computes a "daylight-hours" average
      using each country's approximate sunrise/sunset (a standard NOAA
      simplified solar-position formula, stdlib math only, no new
      dependency) — a real statistic computed from real fetched data,
      clearly labeled as a simplified time-of-day proxy, NOT a true
      generation-weighted figure. feasibility.py uses this daylight
      figure as solar's primary price input specifically because the
      task calls for the revenue estimate itself to be weighted "where
      feasible" — but the mandatory disclaimer (ENTSOE_CANNIBALIZATION_
      NOTE in feasibility.py) says explicitly what this is and isn't.
      Wind gets NO such proxy — there's no defensible time-of-day window
      for wind, and inventing one would be worse than not having it.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from xml.etree import ElementTree as ET

# ── Bidding zones (EIC codes) ────────────────────────────────────────────
# Confirmed against entsoe-py's maintained mappings.py (the most widely
# used unofficial ENTSO-E client) during the investigation pass that
# preceded this module — not guessed. Multi-zone countries list every
# zone; get_cached_price_eur_mwh() reports the low/high pair across them.
EIC_ZONES: dict[str, list[tuple[str, str]]] = {
    "FI": [("FI", "10YFI-1--------U")],
    "SE": [
        ("SE1", "10Y1001A1001A44P"),
        ("SE2", "10Y1001A1001A45N"),
        ("SE3", "10Y1001A1001A46L"),
        ("SE4", "10Y1001A1001A47J"),
    ],
    "DA": [
        ("DK1", "10YDK-1--------W"),
        ("DK2", "10YDK-2--------M"),
    ],
    "NO": [
        ("NO1", "10YNO-1--------2"),
        ("NO2", "10YNO-2--------T"),
        ("NO3", "10YNO-3--------J"),
        ("NO4", "10YNO-4--------9"),
        ("NO5", "10Y1001A1001A48H"),
    ],
    "PL": [("PL", "10YPL-AREA-----S")],
    "DE": [("DE-LU", "10Y1001A1001A82H")],
    "EE": [("EE", "10Y1001A1001A39I")],
    "LV": [("LV", "10YLV-1001A00074")],
    "LT": [("LT", "10YLT-1001A0008Q")],
}

# Approximate reference coordinates (capital / major population centre) —
# used ONLY for the daylight-hours proxy window (see module docstring,
# hazard (b)), never for anything price-bearing. Deliberately coarse: one
# point per country regardless of zone, since this is a rough time-of-day
# filter, not a precision solar-position calculation.
_REFERENCE_LATLON: dict[str, tuple[float, float]] = {
    "FI": (60.17, 24.94),  # Helsinki
    "SE": (59.33, 18.07),  # Stockholm
    "DA": (55.68, 12.57),  # Copenhagen
    "NO": (59.91, 10.75),  # Oslo
    "PL": (52.23, 21.01),  # Warsaw
    "DE": (52.52, 13.40),  # Berlin
    "EE": (59.44, 24.75),  # Tallinn
    "LV": (56.95, 24.11),  # Riga
    "LT": (54.69, 25.28),  # Vilnius
}

_BASE_URL = "https://web-api.tp.entsoe.eu/api"
_REDIS_KEY_PREFIX = "entsoe:price:"
_DEFAULT_MAX_AGE_HOURS = 26.0  # daily refresh cron + a bit of slack
_CHUNK_DAYS = 90  # conservative — real per-call range limit not empirically
                   # confirmed (no live token during the investigation pass
                   # that preceded this module); trivial to widen once tested
                   # live against the real API.

_RESOLUTION_RE = re.compile(r"^PT(\d+)([MH])$")


def _resolution_to_hours(resolution: str) -> float:
    """ISO 8601 duration (PT60M, PT15M, PT30M, PT1H, ...) -> hours (float)."""
    m = _RESOLUTION_RE.match(resolution.strip())
    if not m:
        raise ValueError(f"Unrecognized resolution string: {resolution!r}")
    n, unit = int(m.group(1)), m.group(2)
    return n / 60.0 if unit == "M" else float(n)


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


# ── XML parsing ───────────────────────────────────────────────────────────

def parse_day_ahead_xml(xml_text: str) -> list[dict]:
    """
    Parse an A44 Publication_MarketDocument response into a flat list of
    {"start": datetime (UTC), "duration_hours": float, "price_eur_mwh": float}.

    Namespace-agnostic (ENTSO-E's XML namespace is versioned and changes
    between document-type revisions — stripping it is the standard,
    robust approach rather than pinning an exact namespace URI).

    Handles multiple <TimeSeries>/<Period> blocks with DIFFERENT
    resolutions in one document (e.g. a date range straddling the
    2025-09-30 PT60M->PT15M transition) — each Period's points are
    computed from THAT period's own start + resolution, never assumed
    uniform across the whole document. This is the load-bearing behaviour
    for hazard (a) in the module docstring.
    """
    root = ET.fromstring(xml_text)
    points: list[dict] = []

    for ts in root:
        if _strip_ns(ts.tag) != "TimeSeries":
            continue
        currency = None
        for child in ts:
            if _strip_ns(child.tag) == "currency_Unit.name":
                currency = child.text
        for period in ts:
            if _strip_ns(period.tag) != "Period":
                continue
            time_interval = next(
                (c for c in period if _strip_ns(c.tag) == "timeInterval"), None
            )
            resolution_el = next(
                (c for c in period if _strip_ns(c.tag) == "resolution"), None
            )
            if time_interval is None or resolution_el is None:
                continue
            start_el = next(
                (c for c in time_interval if _strip_ns(c.tag) == "start"), None
            )
            if start_el is None:
                continue
            period_start = datetime.strptime(
                start_el.text.strip(), "%Y-%m-%dT%H:%MZ"
            ).replace(tzinfo=timezone.utc)
            duration_hours = _resolution_to_hours(resolution_el.text.strip())

            for point in period:
                if _strip_ns(point.tag) != "Point":
                    continue
                position = price = None
                for c in point:
                    tag = _strip_ns(c.tag)
                    if tag == "position":
                        position = int(c.text)
                    elif tag == "price.amount":
                        price = float(c.text)
                if position is None or price is None:
                    continue
                point_start = period_start + timedelta(
                    hours=duration_hours * (position - 1)
                )
                points.append({
                    "start": point_start,
                    "duration_hours": duration_hours,
                    "price_eur_mwh": price,
                    "currency": currency or "EUR",
                })

    return points


# ── Aggregation ───────────────────────────────────────────────────────────

def time_weighted_average(points: list[dict]) -> float:
    """
    Correct fix for hazard (a): average weighted by each point's own
    interval duration, NOT by row count. A flat `sum(p)/len(p)` average
    over a mixed PT60M/PT15M series silently over-weights whichever
    regime has more (shorter) rows — see module docstring for the numeric
    argument and test_time_weighted_average_matches_manual_calc below for
    a worked proof.
    """
    if not points:
        raise ValueError("time_weighted_average() called with no points")
    total_weight = sum(p["duration_hours"] for p in points)
    if total_weight <= 0:
        raise ValueError("total duration is zero — malformed points")
    weighted_sum = sum(p["price_eur_mwh"] * p["duration_hours"] for p in points)
    return weighted_sum / total_weight


def _sunrise_sunset_utc(d: date, lat: float, lon: float) -> tuple[float | None, float | None]:
    """
    Approximate sunrise/sunset in UTC decimal hours for a given date and
    lat/lon — standard NOAA simplified solar-position formula (declination
    + hour angle), stdlib math only. A real approximation (checked against
    known Helsinki solstice times to within ~30-60 min during development),
    NOT a precision astronomical calculation — good enough for a coarse
    "which hours are plausibly daylight" filter, nothing more.
    Returns (None, None) for polar night (sun never rises).
    """
    day_of_year = d.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    lat_rad = math.radians(lat)
    cos_hour_angle = -math.tan(lat_rad) * math.tan(decl)
    if cos_hour_angle > 1:
        return None, None  # polar night
    if cos_hour_angle < -1:
        return 0.0, 24.0  # polar day
    hour_angle = math.degrees(math.acos(cos_hour_angle))
    solar_noon_utc = 12 - lon / 15
    sunrise_utc = (solar_noon_utc - hour_angle / 15) % 24
    sunset_utc = (solar_noon_utc + hour_angle / 15) % 24
    return sunrise_utc, sunset_utc


def daylight_hours_average(points: list[dict], country: str) -> float | None:
    """
    Solar-only proxy for hazard (b) — see module docstring. Filters points
    to those whose start time falls within that date's approximate
    sunrise-sunset window (per _REFERENCE_LATLON), then time-weighted-
    averages just those. Returns None if no reference coordinates exist
    for `country` or no points fall in daylight (e.g. empty input).
    """
    latlon = _REFERENCE_LATLON.get(country)
    if latlon is None or not points:
        return None
    lat, lon = latlon
    daylight_points = []
    for p in points:
        sunrise, sunset = _sunrise_sunset_utc(p["start"].date(), lat, lon)
        if sunrise is None:  # polar night that day — no daylight hours to include
            continue
        hour = p["start"].hour + p["start"].minute / 60.0
        if sunrise <= sunset:
            in_daylight = sunrise <= hour < sunset
        else:  # window wraps midnight (not expected at these latitudes, defensive)
            in_daylight = hour >= sunrise or hour < sunset
        if in_daylight:
            daylight_points.append(p)
    if not daylight_points:
        return None
    return time_weighted_average(daylight_points)


# ── Fetch (network) ──────────────────────────────────────────────────────

async def _fetch_zone_chunk(
    client, zone_eic: str, period_start: datetime, period_end: datetime, token: str
) -> list[dict]:
    params = {
        "securityToken": token,
        "documentType": "A44",
        "in_Domain": zone_eic,
        "out_Domain": zone_eic,
        "periodStart": period_start.strftime("%Y%m%d%H%M"),
        "periodEnd": period_end.strftime("%Y%m%d%H%M"),
    }
    resp = await client.get(_BASE_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    return parse_day_ahead_xml(resp.text)


async def fetch_zone_prices(
    zone_eic: str, period_start: datetime, period_end: datetime, token: str
) -> list[dict]:
    """
    Fetch the full [period_start, period_end) range for one bidding zone,
    chunked into _CHUNK_DAYS-day windows (see module-level constant note
    on why this is conservative rather than empirically tuned). Raises on
    any chunk's HTTP failure — callers (refresh_all_prices) decide whether
    a partial-country failure should abort or be logged and skipped.
    """
    import httpx

    all_points: list[dict] = []
    async with httpx.AsyncClient() as client:
        chunk_start = period_start
        while chunk_start < period_end:
            chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), period_end)
            points = await _fetch_zone_chunk(client, zone_eic, chunk_start, chunk_end, token)
            all_points.extend(points)
            chunk_start = chunk_end
    return all_points


async def refresh_all_prices(
    token: str | None = None,
    countries: list[str] | None = None,
    days_back: int = 365,
    redis_url: str | None = None,
) -> dict:
    """
    Out-of-band refresh entry point — invoked by the ARQ daily cron job and
    the /api/admin/refresh-entsoe-prices endpoint (backend/main.py), NOT
    by calculate_feasibility()'s request path (see module docstring).

    Fetches the trailing `days_back` days for every zone of every country
    in `countries` (default: all of EIC_ZONES), aggregates per-zone, and
    writes one Redis key per country. A single zone's fetch failure is
    logged and that zone is skipped (marked in the returned summary) —
    doesn't abort the whole refresh, same "partial failure, not silent,
    not fatal" discipline as poland_ingestion.py's per-source handling.
    """
    token = token or os.environ.get("ENTSOE_API_TOKEN", "")
    if not token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN not set — refresh_all_prices() needs a real "
            "ENTSO-E RESTful API security token (see module docstring / "
            "PR description for the registration process)."
        )
    countries = countries or list(EIC_ZONES.keys())
    period_end = _utcnow_no_seconds()
    period_start = period_end - timedelta(days=days_back)

    fetched_at = period_end.strftime("%Y-%m-%dT%H:%M:%SZ")
    summary: dict = {"fetched_at": fetched_at, "countries": {}}

    for country in countries:
        zones = EIC_ZONES[country]
        zone_results = []
        for zone_label, eic in zones:
            try:
                points = await fetch_zone_prices(eic, period_start, period_end, token)
                if not points:
                    zone_results.append({"zone": zone_label, "status": "NO_DATA"})
                    continue
                avg = time_weighted_average(points)
                daylight_avg = daylight_hours_average(points, country)
                zone_results.append({
                    "zone": zone_label,
                    "status": "OK",
                    "avg_eur_mwh": round(avg, 2),
                    "daylight_avg_eur_mwh": round(daylight_avg, 2) if daylight_avg is not None else None,
                    "n_points": len(points),
                    "currency": points[0].get("currency", "EUR"),
                })
            except Exception as exc:
                zone_results.append({"zone": zone_label, "status": "FAIL", "reason": str(exc)})

        ok_zones = [z for z in zone_results if z["status"] == "OK"]
        country_entry = {
            "zones": zone_results,
            "period_start": period_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "period_end": period_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if ok_zones:
            lo = min(ok_zones, key=lambda z: z["avg_eur_mwh"])
            hi = max(ok_zones, key=lambda z: z["avg_eur_mwh"])
            country_entry.update({
                "lo_zone": lo["zone"], "lo_avg_eur_mwh": lo["avg_eur_mwh"],
                "hi_zone": hi["zone"], "hi_avg_eur_mwh": hi["avg_eur_mwh"],
                # Solar daylight proxy: average across zones that have one
                # (all zones share the same country-level reference lat/lon,
                # so these are directly comparable — not mixing methodologies).
                "daylight_avg_eur_mwh": (
                    round(sum(z["daylight_avg_eur_mwh"] for z in ok_zones if z.get("daylight_avg_eur_mwh") is not None)
                          / max(1, len([z for z in ok_zones if z.get("daylight_avg_eur_mwh") is not None])), 2)
                    if any(z.get("daylight_avg_eur_mwh") is not None for z in ok_zones) else None
                ),
            })
            _write_cache(country, country_entry, fetched_at, redis_url)
        summary["countries"][country] = country_entry

    return summary


def _utcnow_no_seconds() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)


# ── Redis cache (read side — this is what feasibility.py actually calls) ──

def _redis_client(redis_url: str | None = None):
    import redis  # already a production dependency (ARQ's backing store)
    url = redis_url or os.environ.get("REDIS_URL", "")
    if not url:
        return None
    return redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)


def _write_cache(country: str, entry: dict, fetched_at: str, redis_url: str | None = None) -> None:
    client = _redis_client(redis_url)
    if client is None:
        return
    payload = {"fetched_at": fetched_at, **entry}
    try:
        client.set(_REDIS_KEY_PREFIX + country, json.dumps(payload))
    except Exception:
        pass  # best-effort — a failed cache write shouldn't abort the whole refresh


def get_cached_price_eur_mwh(
    country: str,
    max_age_hours: float = _DEFAULT_MAX_AGE_HOURS,
    tech: str | None = None,
    redis_url: str | None = None,
) -> tuple[tuple[float, float], str] | None:
    """
    THE function feasibility.py calls. Synchronous, never raises, never
    blocks on the network — pure Redis GET with a staleness check. Returns
    None (not an exception) on ANY failure: no Redis configured, Redis
    unreachable, key missing, key malformed, or data older than
    `max_age_hours` — feasibility.py's existing static-benchmark fallback
    chain handles None exactly like "no ENTSO-E data was ever available"
    for this country, which is the correct behaviour in all those cases.

    tech="solar_pv" returns the daylight-hours proxy average when present
    (see daylight_hours_average / hazard (b)) instead of the flat
    time-weighted average — this is where the "revenue estimate itself
    weighted by generation profile" requirement actually takes effect.
    Any other tech (or tech=None) gets the flat time-weighted average.
    """
    client = _redis_client(redis_url)
    if client is None:
        return None
    try:
        raw = client.get(_REDIS_KEY_PREFIX + country)
        if raw is None:
            return None
        data = json.loads(raw)
        fetched_at = datetime.strptime(data["fetched_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        if age_hours > max_age_hours:
            return None

        if tech == "solar_pv" and data.get("daylight_avg_eur_mwh") is not None:
            lo = hi = data["daylight_avg_eur_mwh"]
            granularity_note = (
                "daylight-hours proxy average (approximate sunrise/sunset window "
                "per country, NOT a true generation-weighted figure — see "
                "ENTSOE_CANNIBALIZATION_NOTE)"
            )
        else:
            lo, hi = data["lo_avg_eur_mwh"], data["hi_avg_eur_mwh"]
            granularity_note = (
                f"zone range {data['lo_zone']} (low) to {data['hi_zone']} (high) — "
                "NOT a confidence range, see zone_note"
                if data["lo_zone"] != data["hi_zone"]
                else "single zone"
            )

        source = (
            f"ENTSO-E Transparency Platform, day-ahead prices (A44), "
            f"{data['period_start'][:10]} to {data['period_end'][:10]}, "
            f"{granularity_note} — fetched {data['fetched_at'][:10]}."
        )
        return (round(lo, 1), round(hi, 1)), source
    except Exception:
        return None


# ── Self-tests ────────────────────────────────────────────────────────────
# Run directly: venv312/bin/python3 permit_ai/entsoe_prices.py
# No live network/token required — these exercise the parsing/aggregation
# logic against synthetic-but-schema-accurate fixtures (schema confirmed
# against entsoe-py's maintained parser during the investigation pass) and
# the Redis-unavailable graceful-fallback path. A live fetch against the
# real API (fetch_zone_prices / refresh_all_prices with a real token) is
# NOT exercised here — that needs a real ENTSOE_API_TOKEN, deliberately
# left as a separate, explicit follow-up rather than blocking this PR.

_SAMPLE_A44_PT60M = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <currency_Unit.name>EUR</currency_Unit.name>
    <Period>
      <timeInterval><start>2025-09-01T00:00Z</start><end>2025-09-01T02:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>40.0</price.amount></Point>
      <Point><position>2</position><price.amount>60.0</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""

_SAMPLE_A44_PT15M = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <currency_Unit.name>EUR</currency_Unit.name>
    <Period>
      <timeInterval><start>2025-10-01T00:00Z</start><end>2025-10-01T01:00Z</end></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><price.amount>0.0</price.amount></Point>
      <Point><position>2</position><price.amount>0.0</price.amount></Point>
      <Point><position>3</position><price.amount>0.0</price.amount></Point>
      <Point><position>4</position><price.amount>400.0</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""


def _test_parse_pt60m():
    points = parse_day_ahead_xml(_SAMPLE_A44_PT60M)
    assert len(points) == 2, points
    assert points[0]["start"] == datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc)
    assert points[0]["duration_hours"] == 1.0
    assert points[0]["price_eur_mwh"] == 40.0
    assert points[1]["start"] == datetime(2025, 9, 1, 1, 0, tzinfo=timezone.utc)
    print("PASS: parse PT60M — position->timestamp, price, duration all correct")


def _test_parse_pt15m():
    points = parse_day_ahead_xml(_SAMPLE_A44_PT15M)
    assert len(points) == 4, points
    assert points[0]["duration_hours"] == 0.25
    assert points[3]["start"] == datetime(2025, 10, 1, 0, 45, tzinfo=timezone.utc)
    print("PASS: parse PT15M — quarter-hour positions and durations correct")


def _test_time_weighted_average_matches_manual_calc():
    # 2 hours @ PT60M (40, 60) + 1 hour of PT15M (0,0,0,400) = the SAME
    # underlying "1 cheap hour + 1 hour that's cheap-then-spikes" shape,
    # deliberately built so naive-vs-weighted give visibly different answers.
    hourly = parse_day_ahead_xml(_SAMPLE_A44_PT60M)   # 40, 60 @ 1h each
    quarter = parse_day_ahead_xml(_SAMPLE_A44_PT15M)  # 0,0,0,400 @ 0.25h each
    mixed = hourly + quarter

    naive_flat_avg = sum(p["price_eur_mwh"] for p in mixed) / len(mixed)
    weighted_avg = time_weighted_average(mixed)

    # Correct time-weighted answer, hand-computed:
    #   (40*1 + 60*1 + 0*.25 + 0*.25 + 0*.25 + 400*.25) / (1+1+.25+.25+.25+.25)
    # = (40 + 60 + 100) / 3 = 200/3 = 66.667
    expected_weighted = (40 * 1 + 60 * 1 + 0 * 0.25 + 0 * 0.25 + 0 * 0.25 + 400 * 0.25) / 3.0
    assert abs(weighted_avg - expected_weighted) < 1e-9, (weighted_avg, expected_weighted)

    # Naive flat average treats the 4 quarter-hour rows as 4 independent
    # samples equal in weight to the 2 hourly rows: (40+60+0+0+0+400)/6 = 83.33
    # — visibly different from the correct 66.67, and in the WRONG direction
    # (overstates the price here because it over-weights the PT15M spike row).
    assert abs(naive_flat_avg - weighted_avg) > 10, (
        f"naive ({naive_flat_avg}) and time-weighted ({weighted_avg}) averages "
        "should differ meaningfully on this fixture — if they don't, the "
        "weighting fix isn't actually doing anything"
    )
    print(
        f"PASS: time-weighted average ({weighted_avg:.2f}) differs from naive "
        f"flat average ({naive_flat_avg:.2f}) by {naive_flat_avg - weighted_avg:.2f} "
        "EUR/MWh on a mixed-resolution fixture — proves the resolution-change "
        "fix is load-bearing, not cosmetic"
    )


def _test_sunrise_sunset_sanity():
    # Helsinki: known approximate local sunrise/sunset (EEST=UTC+3 in June,
    # EET=UTC+2 in December). Checked to within ~1h — this is a coarse
    # proxy, not a precision astronomical calculation (see docstring).
    lat, lon = _REFERENCE_LATLON["FI"]
    sr_summer, ss_summer = _sunrise_sunset_utc(date(2026, 6, 21), lat, lon)
    sr_winter, ss_winter = _sunrise_sunset_utc(date(2026, 12, 21), lat, lon)
    assert sr_summer is not None and sr_winter is not None
    daylight_summer = (ss_summer - sr_summer) % 24
    daylight_winter = (ss_winter - sr_winter) % 24
    assert daylight_summer > 15, daylight_summer  # known ~18.5h real daylight in June
    assert daylight_winter < 8, daylight_winter    # known ~6h real daylight in December
    assert daylight_summer > daylight_winter
    print(
        f"PASS: Helsinki daylight-hours proxy — summer {daylight_summer:.1f}h, "
        f"winter {daylight_winter:.1f}h (seasonally correct direction and "
        "plausible magnitude)"
    )


def _test_daylight_hours_average_excludes_night():
    # Build a fake day: cheap at night (hour 2), expensive at midday (hour 12).
    # Helsinki reference — midsummer, so daylight covers both hours anyway;
    # use midwinter so hour 2 (night) is reliably excluded and hour 12
    # (near solar noon) is reliably included.
    d = date(2026, 12, 21)
    points = [
        {"start": datetime(2026, 12, 21, 2, 0, tzinfo=timezone.utc), "duration_hours": 1.0, "price_eur_mwh": 5.0},
        {"start": datetime(2026, 12, 21, 10, 0, tzinfo=timezone.utc), "duration_hours": 1.0, "price_eur_mwh": 95.0},
    ]
    avg = daylight_hours_average(points, "FI")
    assert avg == 95.0, (
        f"expected the night-hour (5.0) to be excluded and only the daylight "
        f"hour (95.0) counted, got {avg}"
    )
    print("PASS: daylight_hours_average correctly excludes a night-hour point")


def _test_cache_miss_returns_none_not_exception():
    # No REDIS_URL / unreachable Redis must degrade to None, never raise —
    # this is what lets feasibility.py call this inline without a try/except
    # of its own around every call site.
    result = get_cached_price_eur_mwh("FI", redis_url="redis://localhost:1/0")
    assert result is None
    result2 = get_cached_price_eur_mwh("FI", redis_url="")
    assert result2 is None
    print("PASS: get_cached_price_eur_mwh() degrades to None on Redis failure, no exception")


def _test_eic_zones_cover_all_nine_countries():
    expected = {"FI", "SE", "DA", "NO", "PL", "DE", "EE", "LV", "LT"}
    assert set(EIC_ZONES.keys()) == expected, set(EIC_ZONES.keys())
    assert len(EIC_ZONES["SE"]) == 4
    assert len(EIC_ZONES["NO"]) == 5
    assert len(EIC_ZONES["DA"]) == 2
    print("PASS: EIC_ZONES covers all 9 countries with correct zone counts (SE=4, NO=5, DA=2)")


if __name__ == "__main__":
    _test_eic_zones_cover_all_nine_countries()
    _test_parse_pt60m()
    _test_parse_pt15m()
    _test_time_weighted_average_matches_manual_calc()
    _test_sunrise_sunset_sanity()
    _test_daylight_hours_average_excludes_night()
    _test_cache_miss_returns_none_not_exception()
    print("\nAll entsoe_prices.py self-tests passed.")
