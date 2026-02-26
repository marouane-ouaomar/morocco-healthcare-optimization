"""
src/kmeans_placement.py
========================
Greedy underserved population targeting for optimal facility placement.

Replaces KMeans with a geographically-constrained greedy algorithm that
mirrors how public health planners actually think:

  "Place the next facility where the most underserved people live."

Algorithm:
  1. Filter population grid to Morocco mainland polygon
  2. Compute distance from every cell to nearest existing facility
  3. Sort underserved cells (distance > threshold) by population desc
  4. Greedily place facility at highest-need cell
  5. Recompute distances including the new facility
  6. Repeat until stopping criterion met

Stopping criteria (one active at a time):
  - coverage_target : stop when X% of population is within radius
  - budget_mad      : stop when cumulative cost exceeds budget
  - max_avg_distance: stop when population-weighted mean distance <= target

Spatial guarantees:
  - Candidates come from valid population grid cells (already on land)
  - Every output point is inside Morocco mainland polygon
  - No ocean / Algeria / border artifacts possible

Public API:
    greedy_facility_placement(pop_gdf, facilities_gdf, ...)  -> GeoDataFrame
    compute_improvement_metrics(pop_gdf, existing, candidates) -> dict
"""

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely.geometry import Point

from src.spatial_utils import enforce_spatial_integrity

logger = logging.getLogger(__name__)

CRS_WGS84  = "EPSG:4326"
CRS_METRIC = "EPSG:3857"

# Cost model (MAD)
COST_PER_FACILITY_MAD   = 15_000_000
COST_PER_MOBILE_MAD     =    800_000
COST_PER_KIOSK_MAD      =    200_000

# Coverage radius used internally when no explicit threshold is given
DEFAULT_COVERAGE_RADIUS_KM = 10.0
DEFAULT_COVERAGE_TARGET    = 85.0   # %
DEFAULT_BUDGET_MAD         = 75_000_000
DEFAULT_MAX_AVG_DISTANCE   = 8.0    # km

# Hard cap to prevent infinite loops
MAX_ITERATIONS = 50


# ── Internal helpers ──────────────────────────────────────────────────────────

def _metric_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([gdf.to_crs(CRS_METRIC).geometry.x,
                            gdf.to_crs(CRS_METRIC).geometry.y])


def _nearest_km(pop_coords_m: np.ndarray, fac_coords_m: np.ndarray) -> np.ndarray:
    tree = KDTree(fac_coords_m)
    d, _ = tree.query(pop_coords_m, k=1, workers=-1)
    return d / 1000.0


def _coverage_pct(distances_km: np.ndarray,
                  population: np.ndarray,
                  radius_km: float) -> float:
    total = population.sum()
    if total == 0:
        return 0.0
    return float(population[distances_km <= radius_km].sum() / total * 100)


def _pop_weighted_distance(distances_km: np.ndarray,
                           population: np.ndarray) -> float:
    total = population.sum()
    if total == 0:
        return 0.0
    return float(np.average(distances_km, weights=population))


# ── Main algorithm ────────────────────────────────────────────────────────────

def greedy_facility_placement(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    # Stopping criteria — exactly one should be set
    coverage_target: Optional[float] = None,
    budget_mad: Optional[float] = None,
    max_avg_distance_km: Optional[float] = None,
    # Shared parameters
    coverage_radius_km: float = DEFAULT_COVERAGE_RADIUS_KM,
    facility_cost_mad: float = COST_PER_FACILITY_MAD,
    min_distance_from_existing_km: float = 2.0,
) -> gpd.GeoDataFrame:
    """
    Place new facilities greedily at highest-need inhabited locations.

    Spatial guarantee: every placed facility is a real population grid
    centroid that has already passed Morocco polygon validation —
    mathematically impossible to place in the ocean or Algeria.

    Args:
        pop_gdf: Population grid GeoDataFrame ('population' column required).
        facilities_gdf: Existing facility GeoDataFrame.
        coverage_target: Stop when this % of population is within
            coverage_radius_km. E.g. 85.0 means 85%.
        budget_mad: Stop when cumulative facility cost exceeds this
            amount in MAD. E.g. 75_000_000 = 75M MAD.
        max_avg_distance_km: Stop when population-weighted mean distance
            to nearest facility falls below this value (km).
        coverage_radius_km: Radius used for coverage % calculation.
        facility_cost_mad: Cost per new facility (MAD).
        min_distance_from_existing_km: Skip candidate cells that are
            already within this distance of an existing facility.

    Returns:
        GeoDataFrame of placed facility sites with columns:
            [site_id, lon, lat, step, population_served_10km,
             cumulative_cost_mad, coverage_pct_after,
             avg_distance_after_km, geometry]

    Raises:
        ValueError: If no stopping criterion is set, or inputs are empty.
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    n_criteria = sum([
        coverage_target is not None,
        budget_mad is not None,
        max_avg_distance_km is not None,
    ])
    if n_criteria == 0:
        raise ValueError(
            "Set exactly one stopping criterion: "
            "coverage_target, budget_mad, or max_avg_distance_km"
        )
    if len(pop_gdf) == 0:
        raise ValueError("pop_gdf is empty")
    if len(facilities_gdf) == 0:
        raise ValueError("facilities_gdf is empty")

    mode = ("coverage"  if coverage_target     is not None else
            "budget"    if budget_mad           is not None else
            "distance")
    logger.info(
        f"greedy_facility_placement: mode={mode}, "
        f"radius={coverage_radius_km}km, "
        f"pop_cells={len(pop_gdf)}, existing={len(facilities_gdf)}"
    )

    # ── Step 1: Enforce Morocco polygon on ALL inputs ─────────────────────
    facilities_valid = enforce_spatial_integrity(
        facilities_gdf, label="existing facilities", use_polygon=True
    )
    pop_valid = enforce_spatial_integrity(
        pop_gdf, label="population grid", use_polygon=True
    )
    if len(pop_valid) == 0:
        raise ValueError("No population cells inside Morocco polygon.")

    # ── Step 2: Prepare population arrays ────────────────────────────────
    population     = pop_valid["population"].values.astype(float)
    pop_coords_m   = _metric_coords(pop_valid)
    pop_lons       = pop_valid.geometry.x.values
    pop_lats       = pop_valid.geometry.y.values

    # ── Step 3: Initial distances ─────────────────────────────────────────
    # Build mutable list of facility coords (starts with existing)
    fac_coords_list = list(_metric_coords(facilities_valid))
    current_distances_km = _nearest_km(pop_coords_m, np.array(fac_coords_list))

    # ── Step 4: Compute initial stopping-criterion value ──────────────────
    initial_coverage  = _coverage_pct(current_distances_km, population, coverage_radius_km)
    initial_avg_dist  = _pop_weighted_distance(current_distances_km, population)
    cumulative_cost   = 0.0

    logger.info(
        f"Baseline: coverage={initial_coverage:.1f}%, "
        f"avg_dist={initial_avg_dist:.2f}km"
    )

    # Already meeting target?
    if mode == "coverage" and initial_coverage >= coverage_target:
        logger.info(f"Coverage target {coverage_target}% already met — no new facilities needed.")
        return gpd.GeoDataFrame(
            columns=["site_id","lon","lat","step","population_served_10km",
                     "cumulative_cost_mad","coverage_pct_after",
                     "avg_distance_after_km","geometry"],
            crs=CRS_WGS84,
        )
    if mode == "distance" and initial_avg_dist <= max_avg_distance_km:
        logger.info(f"Distance target {max_avg_distance_km}km already met — no new facilities needed.")
        return gpd.GeoDataFrame(
            columns=["site_id","lon","lat","step","population_served_10km",
                     "cumulative_cost_mad","coverage_pct_after",
                     "avg_distance_after_km","geometry"],
            crs=CRS_WGS84,
        )

    # ── Step 5: Greedy placement loop ────────────────────────────────────
    placed_sites = []
    iteration    = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # Check stopping criterion BEFORE placing
        if mode == "coverage":
            current = _coverage_pct(current_distances_km, population, coverage_radius_km)
            if current >= coverage_target:
                logger.info(f"Coverage target {coverage_target}% reached at step {iteration-1}.")
                break

        elif mode == "budget":
            if cumulative_cost + facility_cost_mad > budget_mad:
                logger.info(f"Budget {budget_mad:,.0f} MAD exhausted at step {iteration-1}.")
                break

        elif mode == "distance":
            current = _pop_weighted_distance(current_distances_km, population)
            if current <= max_avg_distance_km:
                logger.info(f"Distance target {max_avg_distance_km}km reached at step {iteration-1}.")
                break

        # ── Find highest-need candidate cell ──────────────────────────────
        # Only consider cells that are genuinely underserved
        underserved_mask = current_distances_km > coverage_radius_km

        # Also exclude cells already very close to existing facilities
        # (placing there would be wasteful)
        not_redundant = current_distances_km > min_distance_from_existing_km

        candidate_mask = underserved_mask & not_redundant

        if not candidate_mask.any():
            logger.info(f"No more underserved cells — stopping at step {iteration}.")
            break

        # Among candidates, find the cell with highest population
        candidate_populations = np.where(candidate_mask, population, -1)
        best_cell_idx = int(np.argmax(candidate_populations))

        lon = float(pop_lons[best_cell_idx])
        lat = float(pop_lats[best_cell_idx])
        best_coords_m = pop_coords_m[best_cell_idx]

        # ── Place the facility ─────────────────────────────────────────────
        fac_coords_list.append(best_coords_m)
        fac_array = np.array(fac_coords_list)
        current_distances_km = _nearest_km(pop_coords_m, fac_array)

        cumulative_cost += facility_cost_mad

        # ── Compute post-placement metrics ────────────────────────────────
        cov_after      = _coverage_pct(current_distances_km, population, coverage_radius_km)
        avg_dist_after = _pop_weighted_distance(current_distances_km, population)

        # Population newly served within coverage radius by THIS facility
        tree_new   = KDTree([best_coords_m])
        dist_to_new_m, _ = tree_new.query(pop_coords_m, k=1)
        dist_to_new_km   = dist_to_new_m / 1000.0
        pop_served = float(population[dist_to_new_km <= coverage_radius_km].sum())

        placed_sites.append({
            "site_id":               f"site_{iteration:02d}",
            "lon":                   round(lon, 5),
            "lat":                   round(lat, 5),
            "step":                  iteration,
            "population_served_10km": round(pop_served),
            "cumulative_cost_mad":   round(cumulative_cost),
            "coverage_pct_after":    round(cov_after, 2),
            "avg_distance_after_km": round(avg_dist_after, 3),
        })

        logger.info(
            f"  Step {iteration}: placed at ({lat:.3f}N, {lon:.3f}E) | "
            f"coverage={cov_after:.1f}% | avg_dist={avg_dist_after:.2f}km | "
            f"cost={cumulative_cost/1e6:.1f}M MAD"
        )

    if iteration >= MAX_ITERATIONS:
        logger.warning(f"Reached MAX_ITERATIONS ({MAX_ITERATIONS}) without meeting target.")

    if not placed_sites:
        logger.info("No facilities placed.")
        return gpd.GeoDataFrame(
            columns=["site_id","lon","lat","step","population_served_10km",
                     "cumulative_cost_mad","coverage_pct_after",
                     "avg_distance_after_km","geometry"],
            crs=CRS_WGS84,
        )

    result = gpd.GeoDataFrame(
        placed_sites,
        geometry=[Point(s["lon"], s["lat"]) for s in placed_sites],
        crs=CRS_WGS84,
    )

    # ── Final spatial integrity check (belt and suspenders) ───────────────
    result = enforce_spatial_integrity(result, label="placed sites", use_polygon=True)
    logger.info(f"Placed {len(result)} facilities inside Morocco.")
    return result


# ── Improvement metrics ───────────────────────────────────────────────────────

def compute_improvement_metrics(
    pop_gdf: gpd.GeoDataFrame,
    existing_facilities_gdf: gpd.GeoDataFrame,
    candidate_sites_gdf: gpd.GeoDataFrame,
    radii_km: list = [5.0, 10.0, 20.0],
) -> dict:
    """
    Quantify before/after access improvement from placed facilities.

    Returns dict with keys: before, after, delta, n_new_facilities,
    population_newly_covered_10km.
    """
    if len(pop_gdf) == 0 or len(existing_facilities_gdf) == 0:
        raise ValueError("pop_gdf and existing_facilities_gdf must not be empty")

    population    = pop_gdf["population"].values.astype(float)
    total_pop     = population.sum()
    pop_coords_m  = _metric_coords(pop_gdf)
    exist_m       = _metric_coords(existing_facilities_gdf)

    dist_before     = _nearest_km(pop_coords_m, exist_m)
    avg_dist_before = float(np.average(dist_before, weights=population))
    before_coverage = {
        f"coverage_{int(r)}km": round(float(population[dist_before <= r].sum() / total_pop * 100), 2)
        for r in radii_km
    }

    if len(candidate_sites_gdf) == 0:
        return {
            "before":  {**before_coverage, "avg_distance_km": round(avg_dist_before, 3)},
            "after":   {**before_coverage, "avg_distance_km": round(avg_dist_before, 3)},
            "delta":   {f"coverage_{int(r)}km": 0.0 for r in radii_km} | {"avg_distance_km": 0.0},
            "n_new_facilities": 0,
            "population_newly_covered_10km": 0,
        }

    combined_m   = np.vstack([exist_m, _metric_coords(candidate_sites_gdf)])
    dist_after   = _nearest_km(pop_coords_m, combined_m)
    avg_dist_after = float(np.average(dist_after, weights=population))
    after_coverage = {
        f"coverage_{int(r)}km": round(float(population[dist_after <= r].sum() / total_pop * 100), 2)
        for r in radii_km
    }
    delta = {
        f"coverage_{int(r)}km": round(after_coverage[f"coverage_{int(r)}km"] - before_coverage[f"coverage_{int(r)}km"], 2)
        for r in radii_km
    }
    delta["avg_distance_km"] = round(avg_dist_after - avg_dist_before, 3)

    newly = float(population[(dist_after <= 10.0) & (dist_before > 10.0)].sum())

    return {
        "before":  {**before_coverage, "avg_distance_km": round(avg_dist_before, 3)},
        "after":   {**after_coverage,  "avg_distance_km": round(avg_dist_after, 3)},
        "delta":   delta,
        "n_new_facilities": len(candidate_sites_gdf),
        "population_newly_covered_10km": round(newly),
    }
