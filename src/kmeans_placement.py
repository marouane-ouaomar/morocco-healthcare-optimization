"""
src/kmeans_placement.py
========================
Optimal facility placement using population-weighted KMeans clustering.

All candidate sites are guaranteed to lie within Morocco's actual boundary
polygon — never in the ocean, Algeria, or Spain.

Public API:
    find_candidate_sites(pop_gdf, facilities_gdf, n_sites)  -> GeoDataFrame
    compute_improvement_metrics(...)                        -> dict
"""

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely.geometry import Point
from sklearn.cluster import KMeans

from src.spatial_utils import enforce_spatial_integrity

logger = logging.getLogger(__name__)

CRS_WGS84  = "EPSG:4326"
CRS_METRIC = "EPSG:3857"


def _metric_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Return (N, 2) coordinate array in metres (EPSG:3857)."""
    projected = gdf.to_crs(CRS_METRIC)
    return np.column_stack([projected.geometry.x, projected.geometry.y])


def _nearest_distances_km(pop_coords: np.ndarray, fac_coords: np.ndarray) -> np.ndarray:
    """KD-tree nearest-neighbour distances in kilometres."""
    tree = KDTree(fac_coords)
    distances_m, _ = tree.query(pop_coords, k=1, workers=-1)
    return distances_m / 1000.0


def _filter_pop_to_morocco(pop_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep only population cells inside Morocco's boundary polygon.
    Running KMeans only on valid interior points ensures all centroids
    are snapped to inhabited Moroccan land.
    """
    filtered = enforce_spatial_integrity(pop_gdf, label="population grid", use_polygon=True)
    if len(filtered) == 0:
        logger.warning("No cells inside Morocco polygon — using bbox only.")
        filtered = enforce_spatial_integrity(pop_gdf, label="population grid (bbox)", use_polygon=False)
    logger.info(f"Population grid after Morocco filter: {len(filtered)} / {len(pop_gdf)} cells")
    return filtered


def find_candidate_sites(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    n_sites: int = 5,
    min_distance_from_existing_km: float = 5.0,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    """
    Identify optimal new facility locations using population-weighted KMeans,
    strictly constrained to Morocco's boundary polygon.

    Spatial guarantee:
      1. KMeans runs ONLY on population cells inside Morocco polygon
      2. Each centroid is snapped to its cluster's highest-weight cell
      3. Final polygon containment check validates every output point

    Sample weight = population x distance_to_nearest_facility

    Args:
        pop_gdf: Population grid GeoDataFrame with 'population' column.
        facilities_gdf: Existing facility GeoDataFrame.
        n_sites: Number of candidate sites to generate.
        min_distance_from_existing_km: Flag sites closer than this.
        random_state: KMeans seed for reproducibility.

    Returns:
        GeoDataFrame with columns: site_id, lon, lat, cluster_population,
        cluster_size, nearest_existing_km, too_close_to_existing, geometry

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

    # Step 1 — validate existing facilities
    facilities_gdf = enforce_spatial_integrity(facilities_gdf, label="existing facilities")

    # Step 2 — constrain population grid to Morocco interior
    # This is the KEY fix: KMeans input is only valid Moroccan land cells
    pop_morocco = _filter_pop_to_morocco(pop_gdf)

    if len(pop_morocco) < n_sites:
        raise ValueError(
            f"Only {len(pop_morocco)} valid population cells inside Morocco — "
            f"cannot generate {n_sites} sites."
        )

    # Step 3 — compute sample weights
    pop_coords_m = _metric_coords(pop_morocco)
    fac_coords_m = _metric_coords(facilities_gdf)

    distances_km = _nearest_distances_km(pop_coords_m, fac_coords_m)
    population   = pop_morocco["population"].values.astype(float)

    weights = population * distances_km
    if weights.sum() == 0:
        logger.warning("All weights are zero — using population-only weights")
        weights = population.copy()
    weights_norm = weights / weights.max()

    # Step 4 — weighted KMeans on Morocco-interior points only
    kmeans = KMeans(n_clusters=n_sites, random_state=random_state, n_init=20, max_iter=500)
    kmeans.fit(pop_coords_m, sample_weight=weights_norm)
    labels = kmeans.labels_

    # Step 5 — snap centroid to highest-weight cell in cluster
    # Raw centroids are geometric means and can fall in the sea even if all
    # inputs are on land (e.g. cluster spans a peninsula).
    # Snapping to the best cell guarantees land placement.
    pop_lons = pop_morocco.geometry.x.values
    pop_lats = pop_morocco.geometry.y.values

    records = []
    for site_idx in range(n_sites):
        cluster_mask    = labels == site_idx
        cluster_pop     = float(population[cluster_mask].sum())
        cluster_size    = int(cluster_mask.sum())
        cluster_indices = np.where(cluster_mask)[0]
        best_idx        = cluster_indices[np.argmax(weights_norm[cluster_mask])]

        records.append({
            "site_id":            f"candidate_{site_idx + 1:02d}",
            "lon":                round(float(pop_lons[best_idx]), 5),
            "lat":                round(float(pop_lats[best_idx]), 5),
            "cluster_population": round(cluster_pop),
            "cluster_size":       cluster_size,
        })

    candidates = gpd.GeoDataFrame(
        records,
        geometry=[Point(r["lon"], r["lat"]) for r in records],
        crs=CRS_WGS84,
    )

    # Step 6 — final polygon containment check (belt and suspenders)
    candidates = enforce_spatial_integrity(candidates, label="candidate sites", use_polygon=True)

    if len(candidates) < n_sites:
        logger.warning(f"Only {len(candidates)}/{n_sites} sites passed spatial check.")

    # Step 7 — distance to nearest existing facility
    if len(candidates) > 0:
        cand_coords_m = _metric_coords(candidates)
        nearest_km    = _nearest_distances_km(cand_coords_m, fac_coords_m)
        candidates["nearest_existing_km"]   = nearest_km.round(2)
        candidates["too_close_to_existing"] = nearest_km < min_distance_from_existing_km

    candidates = candidates.sort_values("cluster_population", ascending=False).reset_index(drop=True)

    n_flagged = int(candidates.get("too_close_to_existing", pd.Series(dtype=bool)).sum())
    if n_flagged:
        logger.warning(f"{n_flagged} site(s) within {min_distance_from_existing_km} km of existing facility.")

    logger.info(f"Generated {len(candidates)} validated candidate sites inside Morocco")
    return candidates


def compute_improvement_metrics(
    pop_gdf: gpd.GeoDataFrame,
    existing_facilities_gdf: gpd.GeoDataFrame,
    candidate_sites_gdf: gpd.GeoDataFrame,
    radii_km: list = [5.0, 10.0, 20.0],
) -> dict:
    """
    Quantify access improvement from adding candidate sites.

    Returns dict with keys: before, after, delta, n_new_facilities,
    population_newly_covered_10km.
    """
    if len(pop_gdf) == 0 or len(existing_facilities_gdf) == 0:
        raise ValueError("pop_gdf and existing_facilities_gdf must not be empty")

    population    = pop_gdf["population"].values.astype(float)
    total_pop     = population.sum()
    pop_coords_m  = _metric_coords(pop_gdf)
    exist_coords_m = _metric_coords(existing_facilities_gdf)

    dist_before     = _nearest_distances_km(pop_coords_m, exist_coords_m)
    avg_dist_before = float(np.average(dist_before, weights=population))
    before_coverage = {
        f"coverage_{int(r)}km": round(float(population[dist_before <= r].sum() / total_pop * 100), 2)
        for r in radii_km
    }

    combined = gpd.GeoDataFrame(
        pd.concat([
            existing_facilities_gdf[["geometry"]].assign(facility_type="existing"),
            candidate_sites_gdf[["geometry"]].assign(facility_type="new"),
        ], ignore_index=True),
        crs=CRS_WGS84,
    )
    dist_after     = _nearest_distances_km(pop_coords_m, _metric_coords(combined))
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

    newly_covered_10km = float(population[(dist_after <= 10.0) & (dist_before > 10.0)].sum())

    return {
        "before":  {**before_coverage, "avg_distance_km": round(avg_dist_before, 3)},
        "after":   {**after_coverage,  "avg_distance_km": round(avg_dist_after, 3)},
        "delta":   delta,
        "n_new_facilities": len(candidate_sites_gdf),
        "population_newly_covered_10km": round(newly_covered_10km),
    }
