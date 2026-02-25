"""
src/kmeans_placement.py
========================
Optimal facility placement using population-weighted KMeans clustering.

The core idea: underserved population cells are clustered, and cluster
centroids become candidate locations for new facilities. Cells are weighted
by population × distance-to-nearest-facility, so the algorithm naturally
targets dense, underserved areas.

Public API:
    find_candidate_sites(pop_gdf, facilities_gdf, n_sites)  → GeoDataFrame
    compute_improvement_metrics(pop_gdf, facilities_gdf,
                                candidate_gdf)              → dict
"""

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely.geometry import Point
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:3857"

# Morocco bounding box — candidate sites must stay within
MOROCCO_LON_MIN, MOROCCO_LON_MAX = -13.2, -0.99
MOROCCO_LAT_MIN, MOROCCO_LAT_MAX = 27.6, 35.95


# ── Internal helpers ──────────────────────────────────────────────────────────

def _metric_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Return (N, 2) array of coordinates in metres (EPSG:3857)."""
    projected = gdf.to_crs(CRS_METRIC)
    return np.column_stack([projected.geometry.x, projected.geometry.y])


def _nearest_distances_km(
    pop_coords: np.ndarray,
    fac_coords: np.ndarray,
) -> np.ndarray:
    """KD-tree nearest-neighbour distances in kilometres."""
    tree = KDTree(fac_coords)
    distances_m, _ = tree.query(pop_coords, k=1, workers=-1)
    return distances_m / 1000.0


# ── 1. find_candidate_sites ───────────────────────────────────────────────────

def find_candidate_sites(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    n_sites: int = 5,
    min_distance_from_existing_km: float = 5.0,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    """
    Identify optimal locations for new healthcare facilities using
    population-weighted KMeans clustering.

    The sample weight for each population cell is:
        weight = population × distance_to_nearest_facility

    This concentrates candidate sites in areas that are simultaneously
    densely populated AND far from existing facilities.

    Args:
        pop_gdf: Population grid GeoDataFrame with 'population' column.
        facilities_gdf: Existing facility GeoDataFrame.
        n_sites: Number of candidate sites to generate.
        min_distance_from_existing_km: Candidate sites closer than this
            to an existing facility are flagged (not removed — the caller
            decides). Default 5 km.
        random_state: KMeans random seed for reproducibility.

    Returns:
        GeoDataFrame of candidate sites with columns:
            [site_id, lon, lat, cluster_population, cluster_size,
             nearest_existing_km, too_close_to_existing, geometry]
        Sorted by cluster_population descending (highest-impact first).

    Raises:
        ValueError: If n_sites <= 0 or either GeoDataFrame is empty.
    """
    if n_sites <= 0:
        raise ValueError(f"n_sites must be > 0, got {n_sites}")
    if len(pop_gdf) == 0:
        raise ValueError("pop_gdf is empty")
    if len(facilities_gdf) == 0:
        raise ValueError("facilities_gdf is empty")

    logger.info(f"find_candidate_sites: n_sites={n_sites}, pop_cells={len(pop_gdf)}, facilities={len(facilities_gdf)}")

    pop_coords_m = _metric_coords(pop_gdf)
    fac_coords_m = _metric_coords(facilities_gdf)

    # ── Compute sample weights ─────────────────────────────────────────────
    distances_km = _nearest_distances_km(pop_coords_m, fac_coords_m)
    population = pop_gdf["population"].values.astype(float)

    # weight = population × distance (high pop far from care = high priority)
    weights = population * distances_km
    weights_sum = weights.sum()
    if weights_sum == 0:
        logger.warning("All weights are zero — falling back to population-only weights")
        weights = population.copy()

    # Normalise to avoid numerical issues in KMeans
    weights_norm = weights / weights.max()

    # ── Weighted KMeans ────────────────────────────────────────────────────
    kmeans = KMeans(
        n_clusters=n_sites,
        random_state=random_state,
        n_init=20,          # More initialisations → more stable results
        max_iter=500,
    )
    kmeans.fit(pop_coords_m, sample_weight=weights_norm)

    # Cluster centres are in EPSG:3857 (metres) — convert back to WGS84
    centres_m = kmeans.cluster_centers_
    labels = kmeans.labels_

    # ── Build candidate GeoDataFrame ──────────────────────────────────────
    # Project centres back to WGS84
    centres_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in centres_m],
        crs=CRS_METRIC,
    ).to_crs(CRS_WGS84)

    records = []
    for site_idx in range(n_sites):
        cluster_mask = labels == site_idx
        cluster_pop = float(population[cluster_mask].sum())
        cluster_size = int(cluster_mask.sum())

        lon = centres_gdf.geometry.iloc[site_idx].x
        lat = centres_gdf.geometry.iloc[site_idx].y

        # Snap out-of-bounds centroids to Morocco bbox
        lon = float(np.clip(lon, MOROCCO_LON_MIN, MOROCCO_LON_MAX))
        lat = float(np.clip(lat, MOROCCO_LAT_MIN, MOROCCO_LAT_MAX))

        records.append({
            "site_id": f"candidate_{site_idx + 1:02d}",
            "lon": round(lon, 5),
            "lat": round(lat, 5),
            "cluster_population": round(cluster_pop),
            "cluster_size": cluster_size,
        })

    candidates = gpd.GeoDataFrame(
        records,
        geometry=[Point(r["lon"], r["lat"]) for r in records],
        crs=CRS_WGS84,
    )

    # ── Distance from each candidate to nearest existing facility ─────────
    cand_coords_m = _metric_coords(candidates)
    nearest_existing_km = _nearest_distances_km(cand_coords_m, fac_coords_m)
    candidates["nearest_existing_km"] = nearest_existing_km.round(2)
    candidates["too_close_to_existing"] = nearest_existing_km < min_distance_from_existing_km

    # Sort by impact (cluster population)
    candidates = candidates.sort_values("cluster_population", ascending=False)
    candidates = candidates.reset_index(drop=True)

    n_flagged = candidates["too_close_to_existing"].sum()
    if n_flagged:
        logger.warning(
            f"{n_flagged} candidate site(s) are within {min_distance_from_existing_km} km "
            f"of an existing facility — consider increasing n_sites."
        )

    logger.info(f"Generated {len(candidates)} candidate sites")
    for _, row in candidates.iterrows():
        logger.info(
            f"  {row['site_id']}: ({row['lat']:.3f}, {row['lon']:.3f}) "
            f"| pop={row['cluster_population']:,.0f} "
            f"| nearest_existing={row['nearest_existing_km']:.1f} km"
        )

    return candidates


# ── 2. compute_improvement_metrics ───────────────────────────────────────────

def compute_improvement_metrics(
    pop_gdf: gpd.GeoDataFrame,
    existing_facilities_gdf: gpd.GeoDataFrame,
    candidate_sites_gdf: gpd.GeoDataFrame,
    radii_km: list[float] = [5.0, 10.0, 20.0],
) -> dict:
    """
    Quantify the access improvement from adding candidate sites.

    Computes before/after metrics by combining existing facilities
    with the candidate sites.

    Args:
        pop_gdf: Population grid GeoDataFrame with 'population' column.
        existing_facilities_gdf: Current facility GeoDataFrame.
        candidate_sites_gdf: Output of find_candidate_sites().
        radii_km: Coverage radius thresholds for before/after comparison.

    Returns:
        Dict with keys:
            before: {avg_distance_km, coverage_5km, coverage_10km, coverage_20km}
            after:  {avg_distance_km, coverage_5km, coverage_10km, coverage_20km}
            delta:  {avg_distance_km, coverage_5km, coverage_10km, coverage_20km}
            n_new_facilities: int
            population_newly_covered_10km: float
    """
    if len(pop_gdf) == 0 or len(existing_facilities_gdf) == 0:
        raise ValueError("pop_gdf and existing_facilities_gdf must not be empty")

    logger.info("compute_improvement_metrics: calculating before/after access...")

    population = pop_gdf["population"].values.astype(float)
    total_pop = population.sum()

    pop_coords_m = _metric_coords(pop_gdf)
    existing_coords_m = _metric_coords(existing_facilities_gdf)

    # ── BEFORE ────────────────────────────────────────────────────────────
    dist_before = _nearest_distances_km(pop_coords_m, existing_coords_m)
    avg_dist_before = float(np.average(dist_before, weights=population))

    before_coverage = {}
    for r in radii_km:
        pct = float((population[dist_before <= r].sum() / total_pop) * 100)
        before_coverage[f"coverage_{int(r)}km"] = round(pct, 2)

    # ── AFTER (existing + candidates combined) ────────────────────────────
    # Build combined facility set
    candidate_geom = candidate_sites_gdf[["geometry"]].copy()
    candidate_geom["facility_type"] = "new"
    existing_geom = existing_facilities_gdf[["geometry"]].copy()
    if "facility_type" not in existing_geom.columns:
        existing_geom["facility_type"] = "existing"
    combined = gpd.GeoDataFrame(
        pd.concat([existing_geom, candidate_geom], ignore_index=True),
        crs=CRS_WGS84,
    )

    combined_coords_m = _metric_coords(combined)
    dist_after = _nearest_distances_km(pop_coords_m, combined_coords_m)
    avg_dist_after = float(np.average(dist_after, weights=population))

    after_coverage = {}
    for r in radii_km:
        pct = float((population[dist_after <= r].sum() / total_pop) * 100)
        after_coverage[f"coverage_{int(r)}km"] = round(pct, 2)

    # ── Delta ─────────────────────────────────────────────────────────────
    delta = {
        f"coverage_{int(r)}km": round(
            after_coverage[f"coverage_{int(r)}km"] - before_coverage[f"coverage_{int(r)}km"], 2
        )
        for r in radii_km
    }
    delta["avg_distance_km"] = round(avg_dist_after - avg_dist_before, 3)

    # Population newly covered within 10 km
    newly_covered_10km = float(
        population[(dist_after <= 10.0) & (dist_before > 10.0)].sum()
    )

    result = {
        "before": {**before_coverage, "avg_distance_km": round(avg_dist_before, 3)},
        "after": {**after_coverage, "avg_distance_km": round(avg_dist_after, 3)},
        "delta": delta,
        "n_new_facilities": len(candidate_sites_gdf),
        "population_newly_covered_10km": round(newly_covered_10km),
    }

    logger.info("── Improvement Summary ─────────────────────────────")
    logger.info(f"  Avg distance: {avg_dist_before:.2f} km → {avg_dist_after:.2f} km "
                f"(Δ {avg_dist_after - avg_dist_before:+.2f} km)")
    for r in radii_km:
        k = f"coverage_{int(r)}km"
        logger.info(f"  Coverage {r:.0f}km: {before_coverage[k]:.1f}% → {after_coverage[k]:.1f}% "
                    f"(Δ +{delta[k]:.1f}%)")
    logger.info(f"  Newly covered within 10km: {newly_covered_10km:,.0f} people")
    logger.info("────────────────────────────────────────────────────")

    return result
