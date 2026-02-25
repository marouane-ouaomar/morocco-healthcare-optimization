"""
src/data_prep.py
=================
Cleans, validates, and enriches raw OSM facility data and builds
a lightweight population grid for Morocco.

Public API:
    normalize_facility_schema(gdf)  → GeoDataFrame
    assign_admin_region(gdf)        → GeoDataFrame
    validate_coordinates(gdf)       → GeoDataFrame
    grid_population(pop_geojson)    → GeoDataFrame
    run_pipeline()                  → tuple[GeoDataFrame, GeoDataFrame]
"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.validation import make_valid

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"

# Morocco bounding box (WGS84)
MOROCCO_LON_MIN, MOROCCO_LON_MAX = -13.2, -0.99
MOROCCO_LAT_MIN, MOROCCO_LAT_MAX = 27.6, 35.95

RAW_FACILITIES_PATH = Path("data/raw/facilities_raw.geojson")
PROCESSED_FACILITIES_PATH = Path("data/processed/facilities.geojson")
PROCESSED_POPGRID_PATH = Path("data/processed/popgrid.geojson")

# Standardised facility type mapping
AMENITY_TYPE_MAP: dict[str, str] = {
    "hospital": "hospital",
    "clinic": "clinic",
    "doctors": "doctor",
    "doctor": "doctor",
    "pharmacy": "pharmacy",
    "health_post": "clinic",
    "health_centre": "clinic",
    "healthcare": "clinic",
}

# Morocco administrative regions (12 regions since 2015 reform)
MOROCCO_REGIONS = [
    "Tanger-Tétouan-Al Hoceïma",
    "L'Oriental",
    "Fès-Meknès",
    "Rabat-Salé-Kénitra",
    "Béni Mellal-Khénifra",
    "Casablanca-Settat",
    "Marrakech-Safi",
    "Drâa-Tafilalet",
    "Souss-Massa",
    "Guelmim-Oued Noun",
    "Laâyoune-Sakia El Hamra",
    "Dakhla-Oued Ed-Dahab",
]


# ── 1. normalize_facility_schema ──────────────────────────────────────────────

def normalize_facility_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardise column names, types, and facility categories.

    Steps:
    - Rename raw OSM columns to consistent schema
    - Map amenity tags to standardised facility_type values
    - Fill missing names with a sensible default
    - Cast columns to appropriate dtypes
    - Remove duplicates on (osm_id) keeping first occurrence

    Args:
        gdf: Raw GeoDataFrame loaded from facilities_raw.geojson.

    Returns:
        Cleaned GeoDataFrame with standardised schema.
    """
    logger.info(f"normalize_facility_schema: input rows = {len(gdf)}")
    gdf = gdf.copy()

    # Ensure geometry column is present
    if "geometry" not in gdf.columns:
        raise ValueError("Input GeoDataFrame has no geometry column")

    # ── Standardise facility_type ─────────────────────────────────────────
    if "amenity" in gdf.columns:
        gdf["facility_type"] = (
            gdf["amenity"]
            .str.lower()
            .map(AMENITY_TYPE_MAP)
            .fillna("other")
        )
    else:
        gdf["facility_type"] = "unknown"

    # ── Standardise name ──────────────────────────────────────────────────
    # Prefer French name for consistency (official government language)
    name_cols = ["name_fr", "name", "name_ar"]
    gdf["name_clean"] = None
    for col in name_cols:
        if col in gdf.columns:
            mask = gdf["name_clean"].isna() & gdf[col].notna()
            gdf.loc[mask, "name_clean"] = gdf.loc[mask, col]

    gdf["name_clean"] = gdf["name_clean"].fillna("Unnamed Facility")

    # ── Operator type: public / private / ngo / unknown ───────────────────
    if "operator_type" in gdf.columns:
        gdf["sector"] = (
            gdf["operator_type"]
            .str.lower()
            .map({"public": "public", "government": "public",
                  "private": "private", "ngo": "ngo", "religious": "ngo"})
            .fillna("unknown")
        )
    else:
        gdf["sector"] = "unknown"

    # ── Cast osm_id to string ─────────────────────────────────────────────
    if "osm_id" in gdf.columns:
        gdf["osm_id"] = gdf["osm_id"].astype(str)

    # ── Remove duplicates ─────────────────────────────────────────────────
    before = len(gdf)
    if "osm_id" in gdf.columns:
        gdf = gdf.drop_duplicates(subset=["osm_id"], keep="first")
    dupes_removed = before - len(gdf)
    if dupes_removed:
        logger.info(f"  Removed {dupes_removed} duplicate OSM IDs")

    # ── Select and order final columns ────────────────────────────────────
    keep_cols = [
        "osm_id", "osm_type", "name_clean", "facility_type", "sector",
        "name_ar", "name_fr", "operator", "beds", "emergency",
        "opening_hours", "phone", "addr_city", "addr_region", "geometry",
    ]
    existing = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[existing]

    logger.info(f"normalize_facility_schema: output rows = {len(gdf)}, cols = {list(gdf.columns)}")
    return gdf


# ── 2. assign_admin_region ────────────────────────────────────────────────────

def assign_admin_region(
    gdf: gpd.GeoDataFrame,
    admin_boundaries: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoDataFrame:
    """
    Assign each facility to a Moroccan administrative region.

    If admin_boundaries GeoDataFrame is provided, performs a spatial join.
    Otherwise, falls back to a lightweight rule-based assignment using
    approximate longitude/latitude ranges for Morocco's 12 regions.

    Args:
        gdf: Normalised facility GeoDataFrame (EPSG:4326).
        admin_boundaries: Optional GeoDataFrame with admin region polygons.
            Must have a 'region_name' column.

    Returns:
        GeoDataFrame with 'region' column added.
    """
    logger.info(f"assign_admin_region: assigning regions to {len(gdf)} facilities")
    gdf = gdf.copy()

    if admin_boundaries is not None:
        # Ensure matching CRS
        if admin_boundaries.crs != gdf.crs:
            admin_boundaries = admin_boundaries.to_crs(gdf.crs)

        # Spatial join: each facility gets the region whose polygon it falls in
        joined = gpd.sjoin(
            gdf,
            admin_boundaries[["region_name", "geometry"]],
            how="left",
            predicate="within",
        )
        gdf["region"] = joined["region_name"].values
        gdf["region"] = gdf["region"].fillna("Unknown")
        logger.info("  Used spatial join with provided admin boundaries")
    else:
        # ── Fallback: approximate bounding-box heuristic ──────────────────
        # Based on approximate centroids of Morocco's 12 regions
        logger.warning(
            "No admin boundaries provided — using approximate coordinate heuristic. "
            "For accurate results, provide GADM level-1 boundaries for Morocco."
        )
        gdf["region"] = gdf.apply(_region_from_coords, axis=1)

    n_unknown = (gdf["region"] == "Unknown").sum()
    if n_unknown:
        logger.warning(f"  {n_unknown} facilities could not be assigned to a region")

    region_counts = gdf["region"].value_counts()
    for region, count in region_counts.items():
        logger.info(f"  {region}: {count}")

    return gdf


def _region_from_coords(row: pd.Series) -> str:
    """
    Approximate region assignment based on coordinate ranges.
    Used as fallback when no admin boundary file is available.
    """
    try:
        lon = row.geometry.x
        lat = row.geometry.y
    except Exception:
        return "Unknown"

    # Approximate region bounding boxes [lon_min, lon_max, lat_min, lat_max]
    region_boxes: list[tuple[str, float, float, float, float]] = [
        ("Tanger-Tétouan-Al Hoceïma",   -6.0, -1.0, 34.5, 36.0),
        ("L'Oriental",                   -3.5, -0.99, 32.0, 35.5),
        ("Fès-Meknès",                   -6.5, -3.0, 33.0, 35.0),
        ("Rabat-Salé-Kénitra",           -7.5, -5.5, 33.5, 35.0),
        ("Béni Mellal-Khénifra",         -7.0, -4.0, 32.0, 33.5),
        ("Casablanca-Settat",            -8.5, -5.5, 32.5, 34.0),
        ("Marrakech-Safi",               -9.5, -6.0, 30.5, 33.0),
        ("Drâa-Tafilalet",               -6.0, -2.0, 29.5, 32.5),
        ("Souss-Massa",                  -10.0, -6.5, 29.0, 31.0),
        ("Guelmim-Oued Noun",            -11.0, -7.0, 27.5, 29.5),
        ("Laâyoune-Sakia El Hamra",      -14.0, -8.0, 24.5, 27.5),
        ("Dakhla-Oued Ed-Dahab",         -17.1, -14.0, 20.8, 24.5),
    ]

    for region_name, lon_min, lon_max, lat_min, lat_max in region_boxes:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return region_name

    return "Unknown"


# ── 3. validate_coordinates ───────────────────────────────────────────────────

def validate_coordinates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate and clean coordinate data.

    Checks:
    - Geometry is not null
    - Geometry is valid (repairs if possible)
    - Coordinates are within Morocco's bounding box
    - CRS is set to EPSG:4326
    - No duplicate geometries (exact same point)

    Args:
        gdf: GeoDataFrame after normalisation.

    Returns:
        GeoDataFrame with invalid/out-of-bounds rows removed and CRS enforced.

    Raises:
        ValueError: If GeoDataFrame has no CRS and cannot be set.
    """
    logger.info(f"validate_coordinates: input rows = {len(gdf)}")
    gdf = gdf.copy()

    # ── Ensure CRS is set ─────────────────────────────────────────────────
    if gdf.crs is None:
        logger.warning("CRS not set — assuming EPSG:4326")
        gdf = gdf.set_crs(CRS_WGS84)
    elif gdf.crs.to_epsg() != 4326:
        logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs(CRS_WGS84)

    # ── Remove null geometries ────────────────────────────────────────────
    null_mask = gdf.geometry.isna() | gdf.geometry.is_empty
    n_null = null_mask.sum()
    if n_null:
        logger.warning(f"  Dropping {n_null} rows with null/empty geometry")
        gdf = gdf[~null_mask]

    # ── Repair invalid geometries ─────────────────────────────────────────
    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = invalid_mask.sum()
    if n_invalid:
        logger.info(f"  Repairing {n_invalid} invalid geometries")
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)

    # ── Validate lat/lon ranges (Morocco bounding box) ────────────────────
    lons = gdf.geometry.x
    lats = gdf.geometry.y

    out_of_bounds = (
        (lons < MOROCCO_LON_MIN) | (lons > MOROCCO_LON_MAX) |
        (lats < MOROCCO_LAT_MIN) | (lats > MOROCCO_LAT_MAX)
    )
    n_oob = out_of_bounds.sum()
    if n_oob:
        logger.warning(
            f"  Dropping {n_oob} facilities outside Morocco bounding box "
            f"[{MOROCCO_LON_MIN},{MOROCCO_LON_MAX}] × [{MOROCCO_LAT_MIN},{MOROCCO_LAT_MAX}]"
        )
        gdf = gdf[~out_of_bounds]

    # ── Remove exact duplicate geometries ─────────────────────────────────
    coord_key = gdf.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6)))
    before = len(gdf)
    gdf = gdf[~coord_key.duplicated(keep="first")]
    n_geo_dupes = before - len(gdf)
    if n_geo_dupes:
        logger.info(f"  Removed {n_geo_dupes} exact duplicate geometries")

    logger.info(f"validate_coordinates: output rows = {len(gdf)}")
    return gdf.reset_index(drop=True)


# ── 4. grid_population ────────────────────────────────────────────────────────

def grid_population(
    tiff_path: Optional[Path] = None,
    grid_resolution_deg: float = 0.1,
    output_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Build an aggregated population grid for Morocco.

    If a WorldPop TIFF is available, it is aggregated to the specified
    grid resolution. Otherwise, a synthetic grid is generated using
    Morocco's approximate population distribution (urban centres weighted).

    The TIFF is never committed to the repository — only the output
    GeoJSON sample is saved.

    Args:
        tiff_path: Path to WorldPop Morocco population raster (.tif).
            If None or file not found, falls back to synthetic grid.
        grid_resolution_deg: Grid cell size in decimal degrees (default 0.1°
            ≈ 11 km at Morocco's latitude). Smaller = more detail but larger file.
        output_path: Where to save the output GeoJSON. If None, uses
            PROCESSED_POPGRID_PATH.

    Returns:
        GeoDataFrame with columns: [cell_id, lon, lat, population, geometry]
    """
    output_path = output_path or PROCESSED_POPGRID_PATH
    logger.info(f"grid_population: resolution={grid_resolution_deg}°")

    if tiff_path and Path(tiff_path).exists():
        grid_gdf = _grid_from_tiff(tiff_path, grid_resolution_deg)
    else:
        if tiff_path:
            logger.warning(f"TIFF not found at {tiff_path} — using synthetic grid")
        else:
            logger.info("No TIFF provided — generating synthetic population grid")
        grid_gdf = _synthetic_population_grid(grid_resolution_deg)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_gdf.to_file(output_path, driver="GeoJSON")
    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(
        f"Saved population grid: {len(grid_gdf)} cells → {output_path} ({size_kb:.1f} KB)"
    )
    logger.info(f"  Total population estimate: {grid_gdf['population'].sum():,.0f}")
    return grid_gdf


def _grid_from_tiff(tiff_path: Path, resolution: float) -> gpd.GeoDataFrame:
    """Aggregate a WorldPop raster to a coarser grid. Requires rasterio."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        raise ImportError(
            "rasterio is required to read population TIFFs. "
            "Install it with: pip install rasterio\n"
            "Or omit tiff_path to use the synthetic grid."
        )

    records = []
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata

        rows, cols = data.shape
        for r in range(rows):
            for c in range(cols):
                val = data[r, c]
                if nodata is not None and val == nodata:
                    continue
                if val <= 0:
                    continue
                lon, lat = rasterio.transform.xy(transform, r, c)
                records.append({"lon": lon, "lat": lat, "population": float(val)})

    df = pd.DataFrame(records)
    # Aggregate to coarser grid
    df["grid_lon"] = (df["lon"] // resolution) * resolution + resolution / 2
    df["grid_lat"] = (df["lat"] // resolution) * resolution + resolution / 2
    grid = df.groupby(["grid_lon", "grid_lat"])["population"].sum().reset_index()
    grid.columns = ["lon", "lat", "population"]

    return _df_to_grid_gdf(grid)


def _synthetic_population_grid(resolution: float) -> gpd.GeoDataFrame:
    """
    Generate a synthetic population grid approximating Morocco's distribution.

    Uses a Gaussian mixture to represent major population centres:
    Casablanca, Rabat, Fès, Marrakech, Agadir, Tanger, Meknès, Oujda.
    Total population is normalised to approximately 37 million (2024 estimate).
    """
    np.random.seed(42)

    # (name, lon, lat, relative_weight)
    centres = [
        ("Casablanca",  -7.59, 33.57, 4.0),
        ("Rabat",       -6.85, 34.02, 1.8),
        ("Fès",         -4.99, 34.04, 1.5),
        ("Marrakech",   -8.00, 31.63, 1.4),
        ("Agadir",      -9.59, 30.42, 0.9),
        ("Tanger",      -5.80, 35.77, 1.0),
        ("Meknès",      -5.55, 33.90, 0.8),
        ("Oujda",       -1.91, 34.68, 0.7),
        ("Kenitra",     -6.57, 34.25, 0.6),
        ("Tétouan",     -5.37, 35.57, 0.5),
        ("Safi",        -9.23, 32.30, 0.4),
        ("El Jadida",   -8.51, 33.26, 0.4),
        ("Béni Mellal", -6.36, 32.34, 0.4),
        ("Laâyoune",   -13.20, 27.15, 0.2),
    ]

    # Build grid
    lons = np.arange(MOROCCO_LON_MIN, MOROCCO_LON_MAX, resolution) + resolution / 2
    lats = np.arange(MOROCCO_LAT_MIN, MOROCCO_LAT_MAX, resolution) + resolution / 2
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    grid_lons = grid_lons.ravel()
    grid_lats = grid_lats.ravel()

    population = np.zeros(len(grid_lons))
    spread = resolution * 3  # Gaussian spread

    for _, c_lon, c_lat, weight in centres:
        dist_sq = (grid_lons - c_lon) ** 2 + (grid_lats - c_lat) ** 2
        population += weight * np.exp(-dist_sq / (2 * spread ** 2))

    # Add baseline rural population
    population += 0.05

    # Normalise to ~37M total
    total_target = 37_000_000
    population = (population / population.sum()) * total_target

    df = pd.DataFrame({"lon": grid_lons, "lat": grid_lats, "population": population})
    df = df[df["population"] > 10].reset_index(drop=True)  # Remove near-zero cells

    return _df_to_grid_gdf(df)


def _df_to_grid_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert a population DataFrame to a GeoDataFrame with Point geometry."""
    df = df.copy()
    df["cell_id"] = [f"cell_{i:05d}" for i in range(len(df))]
    df["population"] = df["population"].round(1)
    geometry = [Point(row.lon, row.lat) for row in df.itertuples()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_WGS84)
    return gdf[["cell_id", "lon", "lat", "population", "geometry"]]


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(
    raw_path: Path = RAW_FACILITIES_PATH,
    pop_tiff_path: Optional[Path] = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Run the full data preparation pipeline.

    Steps:
    1. Load raw facilities GeoJSON
    2. Normalize schema
    3. Validate coordinates
    4. Assign admin regions
    5. Save to data/processed/facilities.geojson
    6. Build population grid
    7. Save to data/processed/popgrid.geojson

    Args:
        raw_path: Path to raw facilities GeoJSON.
        pop_tiff_path: Optional path to WorldPop TIFF.

    Returns:
        Tuple of (facilities GeoDataFrame, population grid GeoDataFrame).
    """
    logger.info("═══════════ Data Prep Pipeline ═══════════")

    # Load raw
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw facilities file not found: {raw_path}\n"
            "Run: python scripts/fetch_osm_facilities.py"
        )
    logger.info(f"Loading raw facilities from {raw_path}")
    raw_gdf = gpd.read_file(raw_path)
    logger.info(f"Loaded {len(raw_gdf)} raw features")

    # Phase 1: normalize
    facilities = normalize_facility_schema(raw_gdf)

    # Phase 2: validate
    facilities = validate_coordinates(facilities)

    # Phase 3: assign regions
    facilities = assign_admin_region(facilities)

    # Save facilities
    PROCESSED_FACILITIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    facilities.to_file(PROCESSED_FACILITIES_PATH, driver="GeoJSON")
    logger.info(f"✅ Saved {len(facilities)} facilities → {PROCESSED_FACILITIES_PATH}")

    # Population grid
    pop_grid = grid_population(tiff_path=pop_tiff_path)

    logger.info("═══════════ Pipeline Complete ═══════════")
    return facilities, pop_grid
