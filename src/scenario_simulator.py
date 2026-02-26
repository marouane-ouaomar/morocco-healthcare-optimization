"""
src/scenario_simulator.py
==========================
Scenario simulation using greedy constrained facility placement.

Three optimization modes (only one active per run):
  - coverage_target   : place facilities until X% within coverage_radius_km
  - budget_mad        : place facilities until budget exhausted
  - max_avg_distance  : place facilities until mean distance <= target km

All placed facilities are guaranteed inside Morocco mainland polygon.

Public API:
    run_scenario(pop_gdf, facilities_gdf, mode, ...)  -> dict
    run_scenario_from_files(...)                      -> dict
"""

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.kmeans_placement import (
    greedy_facility_placement,
    compute_improvement_metrics,
    _metric_coords,
    _nearest_km,
    _coverage_pct,
    _pop_weighted_distance,
    COST_PER_FACILITY_MAD,
    COST_PER_MOBILE_MAD,
    COST_PER_KIOSK_MAD,
    DEFAULT_COVERAGE_RADIUS_KM,
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_BUDGET_MAD,
    DEFAULT_MAX_AVG_DISTANCE,
)
from src.spatial_utils import enforce_spatial_integrity

logger = logging.getLogger(__name__)

CRS_WGS84 = "EPSG:4326"

FACILITIES_PATH = Path("data/processed/facilities.geojson")
POPGRID_PATH    = Path("data/processed/popgrid.geojson")
SCENARIO_PATH   = Path("data/processed/scenario_results.json")

OptimizationMode = Literal["coverage_target", "budget", "max_avg_distance"]


def run_scenario(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    # Optimization mode
    mode: OptimizationMode = "coverage_target",
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
    budget_mad: float = DEFAULT_BUDGET_MAD,
    max_avg_distance_km: float = DEFAULT_MAX_AVG_DISTANCE,
    coverage_radius_km: float = DEFAULT_COVERAGE_RADIUS_KM,
    # Facility mix (fractions of total placed facilities)
    mobile_unit_fraction: float = 0.2,
    kiosk_fraction: float = 0.3,
) -> dict:
    """
    Run a scenario simulation using greedy spatially-constrained placement.

    Args:
        pop_gdf: Population grid GeoDataFrame with 'population' column.
        facilities_gdf: Existing facility GeoDataFrame.
        mode: One of 'coverage_target', 'budget', 'max_avg_distance'.
        coverage_target: Target coverage % (mode='coverage_target').
        budget_mad: Total budget in MAD (mode='budget').
        max_avg_distance_km: Max acceptable mean distance km (mode='max_avg_distance').
        coverage_radius_km: Radius used for coverage % calculations.
        mobile_unit_fraction: Of placed sites, this fraction become mobile units.
        kiosk_fraction: Of placed sites, this fraction become telemedicine kiosks.

    Returns:
        Structured result dict with scenario_config, baseline, after,
        delta, interventions, combined_results, cost_analysis, metadata.
    """
    logger.info(f"═══════════ Scenario Simulation [mode={mode}] ═══════════")

    # ── Validate inputs ───────────────────────────────────────────────────
    if mode not in ("coverage_target", "budget", "max_avg_distance"):
        raise ValueError(f"Invalid mode: {mode}. Choose from coverage_target, budget, max_avg_distance")

    # ── Run greedy placement ──────────────────────────────────────────────
    placed_gdf = greedy_facility_placement(
        pop_gdf=pop_gdf,
        facilities_gdf=facilities_gdf,
        coverage_target=coverage_target      if mode == "coverage_target"   else None,
        budget_mad=budget_mad                if mode == "budget"             else None,
        max_avg_distance_km=max_avg_distance_km if mode == "max_avg_distance" else None,
        coverage_radius_km=coverage_radius_km,
        facility_cost_mad=COST_PER_FACILITY_MAD,
    )

    n_placed = len(placed_gdf)

    # ── Assign intervention types to placed sites ─────────────────────────
    # Highest-need sites (placed first) become permanent facilities
    # Mid-priority sites become mobile units
    # Lower-priority sites become telemedicine kiosks
    if n_placed > 0:
        n_kiosks  = max(0, int(np.floor(n_placed * kiosk_fraction)))
        n_mobile  = max(0, int(np.floor((n_placed - n_kiosks) * mobile_unit_fraction)))
        n_fixed   = n_placed - n_mobile - n_kiosks

        # Sort by step (greedy order = highest need first)
        placed_sorted = placed_gdf.sort_values("step").reset_index(drop=True)
        fixed_sites  = placed_sorted.iloc[:n_fixed]
        mobile_sites = placed_sorted.iloc[n_fixed: n_fixed + n_mobile]
        kiosk_sites  = placed_sorted.iloc[n_fixed + n_mobile:]
    else:
        fixed_sites = mobile_sites = kiosk_sites = gpd.GeoDataFrame(crs=CRS_WGS84)
        n_fixed = n_mobile = n_kiosks = 0

    # ── Improvement metrics ───────────────────────────────────────────────
    improvement = compute_improvement_metrics(
        pop_gdf=pop_gdf,
        existing_facilities_gdf=facilities_gdf,
        candidate_sites_gdf=placed_gdf,
        radii_km=[5.0, 10.0, 20.0],
    )

    # ── Cost analysis ─────────────────────────────────────────────────────
    total_cost_mad = (
        n_fixed  * COST_PER_FACILITY_MAD +
        n_mobile * COST_PER_MOBILE_MAD   +
        n_kiosks * COST_PER_KIOSK_MAD
    )
    newly_covered   = improvement["population_newly_covered_10km"]
    cost_per_person = round(total_cost_mad / newly_covered, 1) if newly_covered > 0 else None

    # ── Mode-specific config ──────────────────────────────────────────────
    mode_config: dict = {"mode": mode, "coverage_radius_km": coverage_radius_km}
    if mode == "coverage_target":
        mode_config["coverage_target_pct"] = coverage_target
    elif mode == "budget":
        mode_config["budget_mad"]         = budget_mad
        mode_config["budget_usd_approx"]  = round(budget_mad / 10)
    elif mode == "max_avg_distance":
        mode_config["max_avg_distance_km"] = max_avg_distance_km

    # ── Build interventions dict ──────────────────────────────────────────
    def _sites_to_records(gdf: gpd.GeoDataFrame, itype: str) -> list[dict]:
        records = []
        for _, row in gdf.iterrows():
            records.append({
                "site_id":               row.get("site_id", "?"),
                "lat":                   round(row.geometry.y, 5),
                "lon":                   round(row.geometry.x, 5),
                "step":                  int(row.get("step", 0)),
                "population_served":     int(row.get("population_served_10km", 0)),
                "coverage_after_pct":    float(row.get("coverage_pct_after", 0)),
                "cumulative_cost_mad":   int(row.get("cumulative_cost_mad", 0)),
                "type":                  itype,
            })
        return records

    interventions: dict = {}
    if n_fixed > 0:
        interventions["new_facilities"] = {
            "count":          n_fixed,
            "sites":          _sites_to_records(fixed_sites, "permanent"),
            "cost_mad":       n_fixed * COST_PER_FACILITY_MAD,
        }
    if n_mobile > 0:
        interventions["mobile_units"] = {
            "count":          n_mobile,
            "sites":          _sites_to_records(mobile_sites, "mobile"),
            "cost_mad":       n_mobile * COST_PER_MOBILE_MAD,
        }
    if n_kiosks > 0:
        interventions["telemedicine_kiosks"] = {
            "count":          n_kiosks,
            "sites":          _sites_to_records(kiosk_sites, "telemedicine"),
            "cost_mad":       n_kiosks * COST_PER_KIOSK_MAD,
        }

    # ── Compile result ────────────────────────────────────────────────────
    before = improvement["before"]
    after  = improvement["after"]
    delta  = improvement["delta"]

    result = {
        "scenario_config": mode_config,
        "baseline":   before,
        "after":      after,
        "delta":      delta,
        "interventions": interventions,
        "combined_results": {
            "n_facilities_placed":          n_placed,
            "n_permanent":                  n_fixed,
            "n_mobile_units":               n_mobile,
            "n_telemedicine_kiosks":        n_kiosks,
            "avg_distance_before_km":       before["avg_distance_km"],
            "avg_distance_after_km":        after["avg_distance_km"],
            "distance_reduction_km":        abs(delta["avg_distance_km"]),
            "distance_reduction_pct":       round(
                abs(delta["avg_distance_km"]) / before["avg_distance_km"] * 100, 1
            ) if before["avg_distance_km"] > 0 else 0,
            "population_newly_covered_10km": newly_covered,
            "coverage_gain_10km_pct":       delta.get("coverage_10km", 0),
        },
        "cost_analysis": {
            "total_cost_mad":             total_cost_mad,
            "total_cost_usd_approx":      round(total_cost_mad / 10),
            "cost_per_person_reached_mad": cost_per_person,
            "breakdown": {
                "permanent_facilities": n_fixed  * COST_PER_FACILITY_MAD,
                "mobile_units":         n_mobile * COST_PER_MOBILE_MAD,
                "telemedicine_kiosks":  n_kiosks * COST_PER_KIOSK_MAD,
            },
        },
        "metadata": {
            "total_population":  round(float(pop_gdf["population"].sum())),
            "existing_facilities": len(facilities_gdf),
            "algorithm":          "greedy_underserved_targeting",
            "spatial_constraint": "morocco_mainland_polygon",
        },
    }

    logger.info(f"Scenario complete: {n_placed} facilities placed")
    logger.info(f"  Coverage 10km: {before.get('coverage_10km',0):.1f}% → {after.get('coverage_10km',0):.1f}%")
    logger.info(f"  Avg distance:  {before['avg_distance_km']:.2f} → {after['avg_distance_km']:.2f} km")
    logger.info(f"  Total cost:    {total_cost_mad/1e6:.1f}M MAD")
    return result


def run_scenario_from_files(
    mode: OptimizationMode = "coverage_target",
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
    budget_mad: float = DEFAULT_BUDGET_MAD,
    max_avg_distance_km: float = DEFAULT_MAX_AVG_DISTANCE,
    coverage_radius_km: float = DEFAULT_COVERAGE_RADIUS_KM,
    facilities_path: Path = FACILITIES_PATH,
    popgrid_path: Path = POPGRID_PATH,
    output_path: Path = SCENARIO_PATH,
) -> dict:
    """
    Load processed data, run scenario, save results to JSON.
    """
    for p in [facilities_path, popgrid_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run data_prep.run_pipeline() first.")

    facilities = gpd.read_file(facilities_path)
    pop_grid   = gpd.read_file(popgrid_path)

    results = run_scenario(
        pop_gdf=pop_grid,
        facilities_gdf=facilities,
        mode=mode,
        coverage_target=coverage_target,
        budget_mad=budget_mad,
        max_avg_distance_km=max_avg_distance_km,
        coverage_radius_km=coverage_radius_km,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Scenario results saved → {output_path}")
    return results
