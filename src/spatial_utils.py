"""
src/spatial_utils.py
=====================
Shared spatial utility functions for Morocco boundary validation.

Used by data_prep.py, kmeans_placement.py, and streamlit_app.py to
enforce that all points stay within Morocco's actual polygon boundary.
"""

import logging
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, shape

logger = logging.getLogger(__name__)

# Path to the bundled Morocco boundary GeoJSON
BOUNDARY_PATH = Path(__file__).parent.parent / "data/boundaries/morocco_boundary.geojson"

# Tight Morocco bounding box (WGS84) — first-pass filter before polygon check
MOROCCO_LON_MIN, MOROCCO_LON_MAX = -17.1, -0.99
MOROCCO_LAT_MIN, MOROCCO_LAT_MAX =  20.77, 35.95


@lru_cache(maxsize=1)
def load_morocco_polygon() -> Polygon | MultiPolygon:
    """
    Load and cache the Morocco boundary polygon.

    Returns:
        Shapely Polygon or MultiPolygon for Morocco.
    """
    if BOUNDARY_PATH.exists():
        gdf = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")
        geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
        logger.info(f"Loaded Morocco boundary from {BOUNDARY_PATH}")
        return geom
    else:
        logger.warning(
            f"Morocco boundary file not found at {BOUNDARY_PATH}. "
            "Falling back to bounding box validation only."
        )
        return Polygon([
            [MOROCCO_LON_MIN, MOROCCO_LAT_MIN],
            [MOROCCO_LON_MAX, MOROCCO_LAT_MIN],
            [MOROCCO_LON_MAX, MOROCCO_LAT_MAX],
            [MOROCCO_LON_MIN, MOROCCO_LAT_MAX],
            [MOROCCO_LON_MIN, MOROCCO_LAT_MIN],
        ])


def enforce_spatial_integrity(
    gdf: gpd.GeoDataFrame,
    label: str = "features",
    use_polygon: bool = True,
) -> gpd.GeoDataFrame:
    """
    Remove any features that fall outside Morocco's boundary.

    Two-step filter:
    1. Fast bounding box check (drops obvious outliers in Algeria, Spain, sea)
    2. Polygon containment check (drops anything outside Morocco's actual shape)

    Args:
        gdf: GeoDataFrame in any CRS.
        label: Human-readable label for logging (e.g. "facilities", "kiosks").
        use_polygon: If True, also run polygon containment check.

    Returns:
        Filtered GeoDataFrame in EPSG:4326.
    """
    n_input = len(gdf)

    # Step 1 — ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Step 2 — bounding box filter
    lons = gdf.geometry.x
    lats = gdf.geometry.y
    in_bbox = (
        lons.between(MOROCCO_LON_MIN, MOROCCO_LON_MAX) &
        lats.between(MOROCCO_LAT_MIN, MOROCCO_LAT_MAX)
    )
    gdf = gdf[in_bbox].copy()
    n_after_bbox = len(gdf)

    if n_input - n_after_bbox > 0:
        logger.warning(
            f"enforce_spatial_integrity [{label}]: "
            f"dropped {n_input - n_after_bbox} points outside Morocco bounding box"
        )

    # Step 3 — polygon containment
    if use_polygon and len(gdf) > 0:
        morocco = load_morocco_polygon()
        # Vectorised containment check
        within_mask = gdf.geometry.within(morocco)
        n_outside_polygon = (~within_mask).sum()
        if n_outside_polygon > 0:
            logger.warning(
                f"enforce_spatial_integrity [{label}]: "
                f"dropped {n_outside_polygon} points outside Morocco polygon "
                f"(Algeria / ocean / border artifacts)"
            )
        gdf = gdf[within_mask].copy()

    logger.info(
        f"enforce_spatial_integrity [{label}]: "
        f"{n_input} → {len(gdf)} features retained"
    )
    return gdf.reset_index(drop=True)


def snap_to_nearest_population_cell(
    candidate_points: list[tuple[float, float]],
    pop_gdf: gpd.GeoDataFrame,
) -> list[tuple[float, float]]:
    """
    Snap a list of (lon, lat) points to the nearest populated grid cell.

    Used to ensure KMeans centroids always fall on inhabited land.

    Args:
        candidate_points: List of (lon, lat) tuples.
        pop_gdf: Population grid GeoDataFrame in EPSG:4326.

    Returns:
        List of (lon, lat) tuples snapped to nearest population cell.
    """
    from scipy.spatial import KDTree

    pop_lons = pop_gdf.geometry.x.values
    pop_lats = pop_gdf.geometry.y.values
    pop_coords = np.column_stack([pop_lons, pop_lats])
    tree = KDTree(pop_coords)

    snapped = []
    for lon, lat in candidate_points:
        _, idx = tree.query([lon, lat], k=1)
        snapped.append((float(pop_lons[idx]), float(pop_lats[idx])))
    return snapped
