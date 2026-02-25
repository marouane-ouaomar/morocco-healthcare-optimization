"""
src/access_metrics.py
======================
Computes healthcare access metrics for Morocco using spatial algorithms.

All functions are pure — they accept pandas/geopandas objects and return
structured outputs with no side effects.

Public API:
    nearest_facility_distance(pop_gdf, facilities_gdf)       → pd.Series
    population_weighted_distance(pop_gdf, facilities_gdf)    → float
    coverage_within_radius(pop_gdf, facilities_gdf, radii)   → dict
    population_per_facility_ratio(pop_gdf, facilities_gdf)   → pd.DataFrame
    compute_all_metrics(pop_gdf, facilities_gdf)             → pd.DataFrame
    run_metrics()                                             → pd.DataFrame
"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:3857"          # Web Mercator — metres, good enough for Morocco

# Coverage thresholds in kilometres
DEFAULT_RADII_KM = [5.0, 10.0, 20.0]

# Paths
FACILITIES_PATH = Path("data/processed/facilities.geojson")
POPGRID_PATH = Path("data/processed/popgrid.geojson")
OUTPUT_PATH = Path("data/processed/access_metrics.csv")

# OSRM public server (optional road-network routing)
OSRM_DEFAULT_URL = "http://router.project-osrm.org"


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _to_metric_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Reproject GeoDataFrame to metric CRS and return (N, 2) coordinate array.

    Args:
        gdf: GeoDataFrame in any CRS.

    Returns:
        NumPy array of shape (N, 2) with (x, y) in metres.
    """
    projected = gdf.to_crs(CRS_METRIC)
    return np.column_stack([projected.geometry.x, projected.geometry.y])


# ── 1. nearest_facility_distance ─────────────────────────────────────────────

def nearest_facility_distance(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    facility_type: Optional[str] = None,
) -> pd.Series:
    """
    Compute the straight-line distance (km) from each population grid cell
    to the nearest healthcare facility using a KD-tree.

    Args:
        pop_gdf: Population grid GeoDataFrame with geometry column.
        facilities_gdf: Facility GeoDataFrame with geometry column.
        facility_type: If provided, filter facilities to this type only
            (e.g. 'hospital', 'pharmacy'). None = all facility types.

    Returns:
        pd.Series of distances in kilometres, indexed like pop_gdf.

    Raises:
        ValueError: If either GeoDataFrame is empty.
    """
    if len(pop_gdf) == 0:
        raise ValueError("pop_gdf is empty")
    if len(facilities_gdf) == 0:
        raise ValueError("facilities_gdf is empty")

    fac = facilities_gdf.copy()
    if facility_type is not None:
        fac = fac[fac["facility_type"] == facility_type]
        if len(fac) == 0:
            raise ValueError(f"No facilities found for type '{facility_type}'")
        logger.debug(f"Filtered to {len(fac)} facilities of type '{facility_type}'")

    # Build KD-tree on facility coordinates (metric CRS)
    fac_coords = _to_metric_coords(fac)
    pop_coords = _to_metric_coords(pop_gdf)

    tree = KDTree(fac_coords)
    distances_m, _ = tree.query(pop_coords, k=1, workers=-1)

    distances_km = pd.Series(
        distances_m / 1000.0,
        index=pop_gdf.index,
        name="nearest_facility_km",
    )

    logger.info(
        f"nearest_facility_distance: "
        f"mean={distances_km.mean():.2f} km, "
        f"median={distances_km.median():.2f} km, "
        f"max={distances_km.max():.2f} km"
    )
    return distances_km


# ── 2. population_weighted_distance ──────────────────────────────────────────

def population_weighted_distance(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    facility_type: Optional[str] = None,
) -> float:
    """
    Compute the population-weighted mean distance to the nearest facility.

    This metric answers: "What is the average distance a randomly chosen
    Moroccan citizen must travel to reach the nearest facility?"

    Args:
        pop_gdf: Population grid GeoDataFrame with 'population' column.
        facilities_gdf: Facility GeoDataFrame.
        facility_type: Optional facility type filter.

    Returns:
        Population-weighted mean distance in kilometres (scalar float).

    Raises:
        ValueError: If pop_gdf lacks 'population' column.
    """
    if "population" not in pop_gdf.columns:
        raise ValueError("pop_gdf must have a 'population' column")

    distances_km = nearest_facility_distance(pop_gdf, facilities_gdf, facility_type)
    weights = pop_gdf["population"].values.astype(float)

    # Avoid division by zero
    total_pop = weights.sum()
    if total_pop == 0:
        raise ValueError("Total population is zero — check pop_gdf['population'] values")

    weighted_mean = float(np.average(distances_km.values, weights=weights))
    logger.info(f"population_weighted_distance: {weighted_mean:.3f} km")
    return weighted_mean


# ── 3. coverage_within_radius ─────────────────────────────────────────────────

def coverage_within_radius(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    radii_km: list[float] = DEFAULT_RADII_KM,
    facility_type: Optional[str] = None,
) -> dict[str, float]:
    """
    Compute the percentage of population within each radius of a facility.

    Args:
        pop_gdf: Population grid GeoDataFrame with 'population' column.
        facilities_gdf: Facility GeoDataFrame.
        radii_km: List of distance thresholds in kilometres.
        facility_type: Optional facility type filter.

    Returns:
        Dict mapping radius label → coverage percentage.
        Example: {'coverage_5km': 72.3, 'coverage_10km': 88.1, 'coverage_20km': 95.4}
    """
    if "population" not in pop_gdf.columns:
        raise ValueError("pop_gdf must have a 'population' column")

    distances_km = nearest_facility_distance(pop_gdf, facilities_gdf, facility_type)
    total_pop = pop_gdf["population"].sum()

    coverage = {}
    for radius in sorted(radii_km):
        within_mask = distances_km <= radius
        pop_within = pop_gdf.loc[within_mask, "population"].sum()
        pct = float(pop_within / total_pop * 100) if total_pop > 0 else 0.0
        key = f"coverage_{int(radius)}km" if radius == int(radius) else f"coverage_{radius}km"
        coverage[key] = round(pct, 2)
        logger.info(f"  Coverage within {radius} km: {pct:.1f}%")

    return coverage


# ── 4. population_per_facility_ratio ─────────────────────────────────────────

def population_per_facility_ratio(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    region_col: str = "region",
) -> pd.DataFrame:
    """
    Compute population-per-facility ratio, optionally broken down by region.

    A WHO benchmark is roughly 1 doctor per 1,000 population and
    1 hospital bed per 1,000 population; higher ratios indicate underservice.

    Args:
        pop_gdf: Population grid GeoDataFrame with 'population' column.
        facilities_gdf: Facility GeoDataFrame, optionally with region_col.
        region_col: Column name for administrative region in facilities_gdf.

    Returns:
        DataFrame with columns:
            [region, total_population, facility_count,
             pop_per_facility, facility_type, underserved]
        Sorted by pop_per_facility descending (most underserved first).
    """
    if "population" not in pop_gdf.columns:
        raise ValueError("pop_gdf must have a 'population' column")

    total_pop = pop_gdf["population"].sum()
    total_facilities = len(facilities_gdf)

    records = []

    # ── National aggregate ────────────────────────────────────────────────
    for ftype in ["all"] + list(facilities_gdf["facility_type"].unique()):
        if ftype == "all":
            fac_count = total_facilities
        else:
            fac_count = (facilities_gdf["facility_type"] == ftype).sum()

        ratio = float(total_pop / fac_count) if fac_count > 0 else float("inf")
        records.append({
            "region": "National",
            "facility_type": ftype,
            "total_population": round(total_pop),
            "facility_count": int(fac_count),
            "pop_per_facility": round(ratio),
            "underserved": ratio > 10_000,  # WHO-inspired threshold
        })

    # ── Regional breakdown ────────────────────────────────────────────────
    if region_col in facilities_gdf.columns:
        # Approximate region populations using Voronoi / nearest cell approach
        # Here we assign each pop cell to its nearest facility's region
        if len(facilities_gdf) > 0:
            fac_coords = _to_metric_coords(facilities_gdf)
            pop_coords = _to_metric_coords(pop_gdf)
            tree = KDTree(fac_coords)
            _, indices = tree.query(pop_coords, k=1, workers=-1)

            fac_regions = facilities_gdf[region_col].values
            pop_gdf = pop_gdf.copy()
            pop_gdf["assigned_region"] = fac_regions[indices]

            region_pops = pop_gdf.groupby("assigned_region")["population"].sum()
            region_fac_counts = facilities_gdf.groupby(region_col).size()

            for region in region_pops.index:
                reg_pop = region_pops.get(region, 0)
                reg_fac = region_fac_counts.get(region, 0)
                ratio = float(reg_pop / reg_fac) if reg_fac > 0 else float("inf")
                records.append({
                    "region": region,
                    "facility_type": "all",
                    "total_population": round(reg_pop),
                    "facility_count": int(reg_fac),
                    "pop_per_facility": round(ratio),
                    "underserved": ratio > 10_000,
                })

    result = pd.DataFrame(records).sort_values("pop_per_facility", ascending=False)
    logger.info(f"population_per_facility_ratio: {len(result)} rows computed")
    return result.reset_index(drop=True)


# ── 5. OSRM road-network wrapper (optional) ───────────────────────────────────

def road_network_distance(
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    osrm_url: str = OSRM_DEFAULT_URL,
    max_batch: int = 10,
) -> Optional[pd.Series]:
    """
    Compute road-network distances via OSRM Table API (optional).

    Falls back gracefully to None if OSRM is unavailable, allowing the
    caller to use straight-line KD-tree distances instead.

    Args:
        origins: GeoDataFrame of origin points (e.g., population cells).
        destinations: GeoDataFrame of destination points (facilities).
        osrm_url: OSRM server base URL.
        max_batch: Maximum origins per OSRM request (public server limit).

    Returns:
        pd.Series of road distances in kilometres indexed like origins,
        or None if OSRM is unreachable.
    """
    try:
        import requests

        all_distances: list[float] = []

        dest_coords = ";".join(
            f"{row.geometry.x},{row.geometry.y}"
            for _, row in destinations.iterrows()
        )

        for start in range(0, len(origins), max_batch):
            batch = origins.iloc[start : start + max_batch]
            src_coords = ";".join(
                f"{row.geometry.x},{row.geometry.y}"
                for _, row in batch.iterrows()
            )
            n_src = len(batch)
            n_dst = len(destinations)

            src_indices = ";".join(str(i) for i in range(n_src))
            dst_indices = ";".join(str(n_src + i) for i in range(n_dst))

            url = (
                f"{osrm_url}/table/v1/driving/"
                f"{src_coords};{dest_coords}"
                f"?sources={src_indices}&destinations={dst_indices}&annotations=duration"
            )

            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != "Ok":
                logger.warning(f"OSRM error: {data.get('message')}")
                return None

            durations = np.array(data["durations"])  # seconds
            # Nearest destination per origin
            min_durations = durations.min(axis=1)
            # Approx distance: assume 40 km/h average in Morocco
            distances_km = min_durations / 3600.0 * 40.0
            all_distances.extend(distances_km.tolist())

        return pd.Series(all_distances, index=origins.index, name="road_distance_km")

    except Exception as e:
        logger.warning(
            f"OSRM unavailable ({e}) — falling back to straight-line distances. "
            f"Set OSRM_SERVER_URL in .env to use road distances."
        )
        return None


# ── 6. compute_all_metrics ────────────────────────────────────────────────────

def compute_all_metrics(
    pop_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    radii_km: list[float] = DEFAULT_RADII_KM,
    osrm_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute the full suite of access metrics and return a structured DataFrame.

    Combines:
    - Nearest facility distance per population cell
    - Road-network distance (if OSRM available, else None)
    - Coverage percentages within each radius
    - Population-weighted mean distance
    - Population-per-facility ratio by region

    Args:
        pop_gdf: Population grid GeoDataFrame with 'population' column.
        facilities_gdf: Cleaned facility GeoDataFrame.
        radii_km: Coverage radius thresholds in kilometres.
        osrm_url: Optional OSRM server URL for road distances.

    Returns:
        DataFrame with one row per population cell and columns:
            [cell_id, lon, lat, population,
             nearest_facility_km, road_distance_km (if available),
             within_5km, within_10km, within_20km, ...]
    """
    logger.info("Computing full access metrics suite...")

    result = pop_gdf[["cell_id", "lon", "lat", "population"]].copy()

    # Straight-line distances
    result["nearest_facility_km"] = nearest_facility_distance(pop_gdf, facilities_gdf)

    # Road-network distances (optional)
    if osrm_url:
        road_dist = road_network_distance(pop_gdf, facilities_gdf, osrm_url=osrm_url)
        if road_dist is not None:
            result["road_distance_km"] = road_dist.values

    # Coverage flags per cell
    for radius in radii_km:
        col = f"within_{int(radius)}km"
        result[col] = result["nearest_facility_km"] <= radius

    logger.info(
        f"compute_all_metrics: {len(result)} cells, "
        f"{len(facilities_gdf)} facilities"
    )
    return result


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_metrics(
    facilities_path: Path = FACILITIES_PATH,
    popgrid_path: Path = POPGRID_PATH,
    output_path: Path = OUTPUT_PATH,
    osrm_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load processed data, compute all metrics, and save to CSV.

    Args:
        facilities_path: Path to processed facilities GeoJSON.
        popgrid_path: Path to processed population grid GeoJSON.
        output_path: Where to save access_metrics.csv.
        osrm_url: Optional OSRM server URL.

    Returns:
        Access metrics DataFrame.
    """
    logger.info("═══════════ Access Metrics Pipeline ═══════════")

    if not facilities_path.exists():
        raise FileNotFoundError(f"Facilities not found: {facilities_path}\nRun data_prep.run_pipeline() first.")
    if not popgrid_path.exists():
        raise FileNotFoundError(f"Population grid not found: {popgrid_path}\nRun data_prep.run_pipeline() first.")

    facilities = gpd.read_file(facilities_path)
    pop_grid = gpd.read_file(popgrid_path)
    logger.info(f"Loaded {len(facilities)} facilities, {len(pop_grid)} population cells")

    metrics_df = compute_all_metrics(pop_grid, facilities, osrm_url=osrm_url)

    # ── Summary stats ─────────────────────────────────────────────────────
    coverage = coverage_within_radius(pop_grid, facilities)
    pwd = population_weighted_distance(pop_grid, facilities)
    ratio_df = population_per_facility_ratio(pop_grid, facilities)

    logger.info("── Summary ──────────────────────────────────────")
    logger.info(f"  Population-weighted mean distance : {pwd:.2f} km")
    for label, pct in coverage.items():
        logger.info(f"  {label:<25}: {pct:.1f}%")
    national = ratio_df[(ratio_df["region"] == "National") & (ratio_df["facility_type"] == "all")]
    if len(national):
        logger.info(f"  Pop per facility (national)       : {national.iloc[0]['pop_per_facility']:,.0f}")
    logger.info("─────────────────────────────────────────────────")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved access metrics → {output_path}")

    return metrics_df
