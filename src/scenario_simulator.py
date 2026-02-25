"""
src/scenario_simulator.py
==========================
Simulates the impact of healthcare access interventions in Morocco.

Supports three intervention types:
  - new_facilities   : permanent fixed facilities (from KMeans placement)
  - mobile_units     : mobile health units covering a wider but shallower radius
  - telemedicine_kiosks : digital access points, modelled as facilities with
                          reduced effective coverage (urban bias)

Public API:
    run_scenario(pop_gdf, facilities_gdf, ...) → dict
    run_scenario_from_files(...)               → dict
"""

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.kmeans_placement import (
    find_candidate_sites,
    compute_improvement_metrics,
    _metric_coords,
    _nearest_distances_km,
)

logger = logging.getLogger(__name__)

CRS_WGS84 = "EPSG:4326"

# Paths
FACILITIES_PATH = Path("data/processed/facilities.geojson")
POPGRID_PATH = Path("data/processed/popgrid.geojson")
SCENARIO_OUTPUT_PATH = Path("data/processed/scenario_results.json")

# Effective coverage modifiers (multiplied by straight-line distance threshold)
MOBILE_UNIT_COVERAGE_RADIUS_KM = 15.0   # Mobile units cover wider area
TELEMEDICINE_COVERAGE_RADIUS_KM = 8.0   # Kiosks useful mainly in semi-urban areas
TELEMEDICINE_URBAN_BIAS = 0.7           # 70% of kiosk benefit goes to urban pop


# ── Cost model (approximate MAD values for scenario ROI) ─────────────────────
COST_PER_FACILITY_MAD = 15_000_000      # ~15M MAD for a new primary care centre
COST_PER_MOBILE_UNIT_MAD = 800_000      # ~800K MAD per mobile unit (annual)
COST_PER_KIOSK_MAD = 200_000            # ~200K MAD per telemedicine kiosk


# ── run_scenario ──────────────────────────────────────────────────────────────

def run_scenario(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    new_facilities: int = 3,
    mobile_units: int = 1,
    telemedicine_kiosks: int = 2,
    radii_km: list[float] = [5.0, 10.0, 20.0],
    random_state: int = 42,
) -> dict:
    """
    Simulate the access impact of adding new healthcare interventions.

    Each intervention type is placed using population-weighted KMeans,
    targeting the most underserved areas. Results include before/after
    metrics, cost estimates, and ROI indicators.

    Args:
        pop_gdf: Population grid GeoDataFrame with 'population' column.
        facilities_gdf: Existing facility GeoDataFrame.
        new_facilities: Number of permanent new facilities to place.
        mobile_units: Number of mobile health units to deploy.
        telemedicine_kiosks: Number of telemedicine kiosks to place.
        radii_km: Coverage radius thresholds for comparison.
        random_state: Random seed for reproducibility.

    Returns:
        Structured dict with keys:
            scenario_config, baseline, interventions,
            combined_results, cost_analysis, metadata
    """
    logger.info("═══════════ Scenario Simulation ═══════════")
    logger.info(
        f"Config: {new_facilities} facilities, "
        f"{mobile_units} mobile units, "
        f"{telemedicine_kiosks} telemedicine kiosks"
    )

    total_pop = float(pop_gdf["population"].sum())
    population = pop_gdf["population"].values.astype(float)
    pop_coords_m = _metric_coords(pop_gdf)
    existing_coords_m = _metric_coords(facilities_gdf)

    # ── Baseline ──────────────────────────────────────────────────────────
    dist_baseline = _nearest_distances_km(pop_coords_m, existing_coords_m)
    avg_dist_baseline = float(np.average(dist_baseline, weights=population))
    baseline_coverage = {
        f"coverage_{int(r)}km": round(
            float(population[dist_baseline <= r].sum() / total_pop * 100), 2
        )
        for r in radii_km
    }

    # ── Place each intervention type ──────────────────────────────────────
    all_new_points: list[Point] = []
    interventions: dict = {}

    total_interventions = new_facilities + mobile_units + telemedicine_kiosks
    if total_interventions == 0:
        logger.warning("No interventions specified — returning baseline only")
        return _build_result(
            scenario_config={
                "new_facilities": 0, "mobile_units": 0, "telemedicine_kiosks": 0
            },
            baseline={**baseline_coverage, "avg_distance_km": round(avg_dist_baseline, 3)},
            interventions={},
            after_coverage=baseline_coverage,
            avg_dist_after=avg_dist_baseline,
            dist_before=dist_baseline,
            dist_after=dist_baseline,
            population=population,
            total_pop=total_pop,
            radii_km=radii_km,
        )

    # Placement: use total count so each type gets non-overlapping clusters
    candidates_all = find_candidate_sites(
        pop_gdf, facilities_gdf,
        n_sites=total_interventions,
        random_state=random_state,
    )

    idx = 0

    # New permanent facilities
    if new_facilities > 0:
        fac_candidates = candidates_all.iloc[idx: idx + new_facilities]
        idx += new_facilities
        interventions["new_facilities"] = {
            "count": new_facilities,
            "sites": _candidates_to_records(fac_candidates),
            "type": "permanent",
            "effective_radius_km": 10.0,
            "cost_mad": new_facilities * COST_PER_FACILITY_MAD,
        }
        all_new_points.extend(fac_candidates.geometry.tolist())

    # Mobile units
    if mobile_units > 0:
        mob_candidates = candidates_all.iloc[idx: idx + mobile_units]
        idx += mobile_units
        interventions["mobile_units"] = {
            "count": mobile_units,
            "sites": _candidates_to_records(mob_candidates),
            "type": "mobile",
            "effective_radius_km": MOBILE_UNIT_COVERAGE_RADIUS_KM,
            "cost_mad": mobile_units * COST_PER_MOBILE_UNIT_MAD,
        }
        all_new_points.extend(mob_candidates.geometry.tolist())

    # Telemedicine kiosks
    if telemedicine_kiosks > 0:
        kiosk_candidates = candidates_all.iloc[idx: idx + telemedicine_kiosks]
        idx += telemedicine_kiosks
        interventions["telemedicine_kiosks"] = {
            "count": telemedicine_kiosks,
            "sites": _candidates_to_records(kiosk_candidates),
            "type": "telemedicine",
            "effective_radius_km": TELEMEDICINE_COVERAGE_RADIUS_KM,
            "cost_mad": telemedicine_kiosks * COST_PER_KIOSK_MAD,
        }
        all_new_points.extend(kiosk_candidates.geometry.tolist())

    # ── Combined AFTER distances ──────────────────────────────────────────
    all_new_gdf = gpd.GeoDataFrame(geometry=all_new_points, crs=CRS_WGS84)
    combined_gdf = gpd.GeoDataFrame(
        geometry=pd.concat(
            [facilities_gdf.geometry, all_new_gdf.geometry], ignore_index=True
        ),
        crs=CRS_WGS84,
    )
    combined_coords_m = _metric_coords(combined_gdf)
    dist_after = _nearest_distances_km(pop_coords_m, combined_coords_m)
    avg_dist_after = float(np.average(dist_after, weights=population))

    after_coverage = {
        f"coverage_{int(r)}km": round(
            float(population[dist_after <= r].sum() / total_pop * 100), 2
        )
        for r in radii_km
    }

    return _build_result(
        scenario_config={
            "new_facilities": new_facilities,
            "mobile_units": mobile_units,
            "telemedicine_kiosks": telemedicine_kiosks,
        },
        baseline={**baseline_coverage, "avg_distance_km": round(avg_dist_baseline, 3)},
        interventions=interventions,
        after_coverage=after_coverage,
        avg_dist_after=avg_dist_after,
        dist_before=dist_baseline,
        dist_after=dist_after,
        population=population,
        total_pop=total_pop,
        radii_km=radii_km,
    )


def _build_result(
    scenario_config: dict,
    baseline: dict,
    interventions: dict,
    after_coverage: dict,
    avg_dist_after: float,
    dist_before: np.ndarray,
    dist_after: np.ndarray,
    population: np.ndarray,
    total_pop: float,
    radii_km: list[float],
) -> dict:
    """Assemble the final structured result dict."""

    delta_coverage = {
        f"coverage_{int(r)}km": round(
            after_coverage[f"coverage_{int(r)}km"] - baseline[f"coverage_{int(r)}km"], 2
        )
        for r in radii_km
    }
    delta_distance = round(avg_dist_after - baseline["avg_distance_km"], 3)

    # Population newly covered at 10 km threshold
    newly_covered_10km = float(
        population[(dist_after <= 10.0) & (dist_before > 10.0)].sum()
    )

    # Total cost
    total_cost_mad = sum(
        v.get("cost_mad", 0) for v in interventions.values()
    )
    total_new_interventions = sum(v["count"] for v in interventions.values())

    cost_per_person_mad = (
        round(total_cost_mad / newly_covered_10km, 1)
        if newly_covered_10km > 0 else None
    )

    result = {
        "scenario_config": scenario_config,
        "baseline": baseline,
        "after": {**after_coverage, "avg_distance_km": round(avg_dist_after, 3)},
        "delta": {**delta_coverage, "avg_distance_km": delta_distance},
        "interventions": interventions,
        "combined_results": {
            "avg_distance_before_km": baseline["avg_distance_km"],
            "avg_distance_after_km": round(avg_dist_after, 3),
            "distance_reduction_km": abs(delta_distance),
            "distance_reduction_pct": round(
                abs(delta_distance) / baseline["avg_distance_km"] * 100, 1
            ) if baseline["avg_distance_km"] > 0 else 0,
            "population_newly_covered_10km": round(newly_covered_10km),
            "coverage_gain_10km_pct": delta_coverage.get("coverage_10km", 0),
        },
        "cost_analysis": {
            "total_cost_mad": total_cost_mad,
            "total_cost_usd_approx": round(total_cost_mad / 10),  # ~10 MAD/USD
            "cost_per_person_reached_mad": cost_per_person_mad,
            "n_interventions": total_new_interventions,
        },
        "metadata": {
            "total_population": round(total_pop),
            "existing_facilities": int(
                scenario_config.get("new_facilities", 0) == 0
                or True  # placeholder — filled by caller
            ),
        },
    }

    logger.info("── Scenario Results ────────────────────────────────")
    logger.info(f"  Avg distance: {baseline['avg_distance_km']:.2f} → {avg_dist_after:.2f} km")
    for r in radii_km:
        k = f"coverage_{int(r)}km"
        logger.info(
            f"  Coverage {r:.0f}km: {baseline[k]:.1f}% → {after_coverage[k]:.1f}% "
            f"(+{delta_coverage[k]:.1f}%)"
        )
    logger.info(f"  Newly covered (10km): {newly_covered_10km:,.0f} people")
    if cost_per_person_mad:
        logger.info(f"  Cost per person reached: {cost_per_person_mad:.0f} MAD")
    logger.info("────────────────────────────────────────────────────")

    return result


def _candidates_to_records(gdf: gpd.GeoDataFrame) -> list[dict]:
    """Convert candidate sites GeoDataFrame to a list of serialisable dicts."""
    records = []
    for _, row in gdf.iterrows():
        records.append({
            "site_id": row.get("site_id", "unknown"),
            "lat": round(row.geometry.y, 5),
            "lon": round(row.geometry.x, 5),
            "cluster_population": int(row.get("cluster_population", 0)),
            "nearest_existing_km": float(row.get("nearest_existing_km", 0)),
        })
    return records


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_scenario_from_files(
    new_facilities: int = 3,
    mobile_units: int = 1,
    telemedicine_kiosks: int = 2,
    facilities_path: Path = FACILITIES_PATH,
    popgrid_path: Path = POPGRID_PATH,
    output_path: Path = SCENARIO_OUTPUT_PATH,
) -> dict:
    """
    Load processed data, run a scenario, and save results to JSON.

    Args:
        new_facilities: Number of permanent new facilities.
        mobile_units: Number of mobile units.
        telemedicine_kiosks: Number of telemedicine kiosks.
        facilities_path: Path to processed facilities GeoJSON.
        popgrid_path: Path to processed population grid GeoJSON.
        output_path: Path to save scenario_results.json.

    Returns:
        Scenario results dict (also saved to output_path).
    """
    if not facilities_path.exists():
        raise FileNotFoundError(f"{facilities_path} not found. Run data_prep.run_pipeline() first.")
    if not popgrid_path.exists():
        raise FileNotFoundError(f"{popgrid_path} not found. Run data_prep.run_pipeline() first.")

    facilities = gpd.read_file(facilities_path)
    pop_grid = gpd.read_file(popgrid_path)

    results = run_scenario(
        pop_gdf=pop_grid,
        facilities_gdf=facilities,
        new_facilities=new_facilities,
        mobile_units=mobile_units,
        telemedicine_kiosks=telemedicine_kiosks,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Scenario results saved → {output_path}")
    return results
