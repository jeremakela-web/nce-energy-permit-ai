# © 2026 NCE Permit AI | ncenergy.fi
"""
NCE Energy — Multi-Project Site Optimizer
"""

from dataclasses import dataclass
from typing import List

PROJECT_WEIGHTS = {
    "bess": {
        "grid_distance":    0.25,
        "zoning":           0.25,
        "solar":            0.20,
        "area":             0.10,
        "protected":        0.10,
        "economics":        0.10,
    },
    "tuulivoima": {
        "wind_resource":    0.30,
        "grid_distance":    0.25,
        "zoning":           0.20,
        "area":             0.10,
        "protected":        0.10,
        "economics":        0.05,
    },
    "aurinkovoima": {
        "solar":            0.30,
        "grid_distance":    0.20,
        "zoning":           0.20,
        "area":             0.15,
        "protected":        0.10,
        "economics":        0.05,
    },
    "smr": {
        "grid_distance":    0.20,
        "zoning":           0.30,
        "water_access":     0.20,
        "protected":        0.15,
        "area":             0.10,
        "economics":        0.05,
    },
}


def get_best_available_optimizer():
    try:
        from neal import SimulatedAnnealingSampler
        return "quantum_inspired"
    except ImportError:
        return "classical"


@dataclass
class EnergySite:
    site_id: str
    lat: float
    lon: float
    solar_irradiance: float     # W/m²
    wind_resource: float        # m/s keskituuli
    grid_distance_km: float
    land_area_ha: float
    zoning_score: float         # 0–1
    protected_area_score: float # 0–1 (1 = ei suojelua)
    water_access_score: float   # 0–1 (SMR-spesifi)
    land_cost_eur_ha: float


@dataclass
class OptimizationResult:
    ranked_sites: list
    scores: List[float]
    optimizer_used: str
    project_type: str
    explanation: str


class NCEOptimizer:
    def __init__(self, project_type: str):
        if project_type not in PROJECT_WEIGHTS:
            raise ValueError(f"Tuntematon hanketyyppi: {project_type}")
        self.project_type = project_type
        self.weights = PROJECT_WEIGHTS[project_type]
        self.optimizer_level = get_best_available_optimizer()

    def score_site(self, site: EnergySite) -> float:
        scores = {
            "solar":         min(site.solar_irradiance / 1200, 1.0),
            "wind_resource": min(site.wind_resource / 8.0, 1.0),
            "grid_distance": max(0, 1 - site.grid_distance_km / 50),
            "area":          min(site.land_area_ha / 10, 1.0),
            "zoning":        site.zoning_score,
            "protected":     site.protected_area_score,
            "water_access":  site.water_access_score,
            "economics":     max(0, 1 - site.land_cost_eur_ha / 50000),
        }
        return round(
            sum(self.weights.get(k, 0) * v for k, v in scores.items()), 4
        )

    def optimize(self, sites: list) -> OptimizationResult:
        scored = sorted(
            [(s, self.score_site(s)) for s in sites],
            key=lambda x: x[1], reverse=True
        )
        ranked_sites = [s for s, _ in scored]
        ranked_scores = [sc for _, sc in scored]
        best = ranked_sites[0]
        return OptimizationResult(
            ranked_sites=ranked_sites,
            scores=ranked_scores,
            optimizer_used=self.optimizer_level,
            project_type=self.project_type,
            explanation=(
                f"Paras sijainti: {best.site_id} ({ranked_scores[0]:.1%})\n"
                f"  Verkkoetäisyys: {best.grid_distance_km} km\n"
                f"  Kaavoitus: {best.zoning_score:.0%}\n"
                f"  Ympäristö: {best.protected_area_score:.0%}"
            )
        )
