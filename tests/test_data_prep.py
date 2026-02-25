"""
tests/test_data_prep.py
========================
Unit tests for src/data_prep.py using synthetic fixtures.
No real API calls or file I/O required.
"""

import pytest
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from src.data_prep import (
    normalize_facility_schema,
    assign_admin_region,
    validate_coordinates,
    grid_population,
    _region_from_coords,
    _synthetic_population_grid,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_facilities_gdf() -> gpd.GeoDataFrame:
    """Synthetic facility GeoDataFrame mimicking raw OSM output."""
    data = {
        "osm_id": [1001, 1002, 1003, 1004, 1005, 1001],  # 1001 duplicated
        "osm_type": ["node", "way", "node", "node", "node", "node"],
        "amenity": ["hospital", "clinic", "doctors", "pharmacy", "hospital", "hospital"],
        "name": ["Hôpital Ibn Sina", None, "Cabinet Médical", "Pharmacie Atlas", "Clinique Noor", "Hôpital Ibn Sina"],
        "name_fr": ["Hôpital Ibn Sina", "Clinique Al Majd", None, None, "Clinique Noor", "Hôpital Ibn Sina"],
        "name_ar": ["مستشفى ابن سينا", None, None, None, "عيادة نور", "مستشفى ابن سينا"],
        "operator": ["Ministry of Health", "Private", None, "Private", None, "Ministry of Health"],
        "operator_type": ["public", "private", None, "private", None, "public"],
        "beds": ["250", None, None, None, "50", "250"],
        "emergency": ["yes", None, None, None, "yes", "yes"],
        "opening_hours": [None, "Mo-Fr 08:00-18:00", None, "24/7", None, None],
        "phone": ["+212 5 37 00 00 00", None, None, None, None, "+212 5 37 00 00 00"],
        "addr_city": ["Rabat", "Casablanca", "Fès", "Marrakech", "Tanger", "Rabat"],
        "addr_region": [None, None, None, None, None, None],
        "geometry": [
            Point(-6.85, 34.02),   # Rabat
            Point(-7.59, 33.57),   # Casablanca
            Point(-4.99, 34.04),   # Fès
            Point(-8.00, 31.63),   # Marrakech
            Point(-5.80, 35.77),   # Tanger
            Point(-6.85, 34.02),   # Duplicate of row 0
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def out_of_bounds_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with some points outside Morocco."""
    data = {
        "osm_id": [10, 11, 12],
        "amenity": ["hospital", "clinic", "pharmacy"],
        "name": ["Valid", "Outside Morocco", "Also valid"],
        "geometry": [
            Point(-6.85, 34.02),   # Rabat — valid
            Point(2.35, 48.85),    # Paris — invalid
            Point(-7.59, 33.57),   # Casablanca — valid
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def no_crs_gdf(sample_facilities_gdf) -> gpd.GeoDataFrame:
    """GeoDataFrame with CRS stripped."""
    gdf = sample_facilities_gdf.copy()
    return gdf.set_crs(None, allow_override=True)


# ── normalize_facility_schema ─────────────────────────────────────────────────

class TestNormalizeFacilitySchema:
    def test_output_has_required_columns(self, sample_facilities_gdf):
        result = normalize_facility_schema(sample_facilities_gdf)
        for col in ["facility_type", "name_clean", "sector", "osm_id"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_amenity_mapped_to_facility_type(self, sample_facilities_gdf):
        result = normalize_facility_schema(sample_facilities_gdf)
        assert "hospital" in result["facility_type"].values
        assert "clinic" in result["facility_type"].values
        assert "doctor" in result["facility_type"].values
        assert "pharmacy" in result["facility_type"].values

    def test_duplicate_osm_ids_removed(self, sample_facilities_gdf):
        # Input has 6 rows with osm_id 1001 duplicated
        result = normalize_facility_schema(sample_facilities_gdf)
        assert result["osm_id"].duplicated().sum() == 0

    def test_missing_name_filled_with_default(self, sample_facilities_gdf):
        result = normalize_facility_schema(sample_facilities_gdf)
        assert result["name_clean"].isna().sum() == 0
        assert (result["name_clean"] == "Unnamed Facility").sum() >= 0

    def test_french_name_preferred_over_arabic(self, sample_facilities_gdf):
        result = normalize_facility_schema(sample_facilities_gdf)
        # Row 1 has name_fr = "Clinique Al Majd" but name = None
        clinique_row = result[result["osm_id"] == "1002"]
        if len(clinique_row):
            assert clinique_row.iloc[0]["name_clean"] == "Clinique Al Majd"

    def test_operator_type_mapped_to_sector(self, sample_facilities_gdf):
        result = normalize_facility_schema(sample_facilities_gdf)
        assert "public" in result["sector"].values
        assert "private" in result["sector"].values

    def test_osm_id_cast_to_string(self, sample_facilities_gdf):
        result = normalize_facility_schema(sample_facilities_gdf)
        assert result["osm_id"].dtype == object  # string

    def test_geometry_preserved(self, sample_facilities_gdf):
        result = normalize_facility_schema(sample_facilities_gdf)
        assert result.geometry.notna().all()

    def test_raises_on_missing_geometry_column(self):
        df = pd.DataFrame({"osm_id": [1], "amenity": ["hospital"]})
        gdf = gpd.GeoDataFrame(df)
        with pytest.raises(ValueError, match="geometry"):
            normalize_facility_schema(gdf)


# ── validate_coordinates ──────────────────────────────────────────────────────

class TestValidateCoordinates:
    def test_removes_out_of_bounds_points(self, out_of_bounds_gdf):
        result = validate_coordinates(out_of_bounds_gdf)
        assert len(result) == 2  # Only Rabat and Casablanca

    def test_sets_crs_to_4326(self, sample_facilities_gdf):
        result = validate_coordinates(sample_facilities_gdf)
        assert result.crs.to_epsg() == 4326

    def test_handles_missing_crs(self, no_crs_gdf):
        # Should not raise; should assign EPSG:4326
        result = validate_coordinates(no_crs_gdf)
        assert result.crs.to_epsg() == 4326

    def test_removes_null_geometries(self, sample_facilities_gdf):
        gdf = sample_facilities_gdf.copy()
        gdf.loc[0, "geometry"] = None
        result = validate_coordinates(gdf)
        assert result.geometry.isna().sum() == 0

    def test_removes_exact_duplicate_geometries(self, sample_facilities_gdf):
        # sample_facilities_gdf has two rows with Point(-6.85, 34.02)
        result = validate_coordinates(sample_facilities_gdf)
        # Check no two rows have identical coordinates
        coords = result.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6)))
        assert coords.duplicated().sum() == 0

    def test_all_coordinates_in_morocco_bbox(self, sample_facilities_gdf):
        result = validate_coordinates(sample_facilities_gdf)
        assert (result.geometry.x >= -13.2).all()
        assert (result.geometry.x <= -0.99).all()
        assert (result.geometry.y >= 27.6).all()
        assert (result.geometry.y <= 35.95).all()

    def test_output_is_reset_index(self, sample_facilities_gdf):
        result = validate_coordinates(sample_facilities_gdf)
        assert list(result.index) == list(range(len(result)))


# ── assign_admin_region ───────────────────────────────────────────────────────

class TestAssignAdminRegion:
    def test_region_column_added(self, sample_facilities_gdf):
        result = assign_admin_region(sample_facilities_gdf)
        assert "region" in result.columns

    def test_no_null_regions(self, sample_facilities_gdf):
        result = assign_admin_region(sample_facilities_gdf)
        # May have "Unknown" but not NaN
        assert result["region"].isna().sum() == 0

    def test_rabat_assigned_correctly(self):
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(-6.85, 34.02)]},  # Rabat
            crs="EPSG:4326"
        )
        result = assign_admin_region(gdf)
        assert "Rabat" in result.iloc[0]["region"]

    def test_casablanca_assigned_correctly(self):
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(-7.59, 33.57)]},  # Casablanca
            crs="EPSG:4326"
        )
        result = assign_admin_region(gdf)
        assert "Casablanca" in result.iloc[0]["region"]

    def test_spatial_join_with_admin_boundaries(self, sample_facilities_gdf):
        """Test that spatial join is used when boundaries are provided."""
        # Create a synthetic boundary covering Rabat area
        rabat_poly = Polygon([
            (-7.5, 33.5), (-6.0, 33.5), (-6.0, 35.0), (-7.5, 35.0), (-7.5, 33.5)
        ])
        admin_gdf = gpd.GeoDataFrame(
            {"region_name": ["Test Region"], "geometry": [rabat_poly]},
            crs="EPSG:4326"
        )
        result = assign_admin_region(sample_facilities_gdf, admin_boundaries=admin_gdf)
        assert "region" in result.columns
        # Points inside the polygon should get "Test Region"
        rabat_row = result[result.geometry == Point(-6.85, 34.02)]
        if len(rabat_row):
            assert rabat_row.iloc[0]["region"] == "Test Region"


# ── _region_from_coords helper ────────────────────────────────────────────────

class TestRegionFromCoords:
    def test_marrakech_coords(self):
        row = pd.Series({"geometry": Point(-8.00, 31.63)})
        assert "Marrakech" in _region_from_coords(row)

    def test_tangier_coords(self):
        row = pd.Series({"geometry": Point(-5.80, 35.77)})
        assert "Tanger" in _region_from_coords(row)

    def test_outside_morocco_returns_unknown(self):
        row = pd.Series({"geometry": Point(2.35, 48.85)})  # Paris
        assert _region_from_coords(row) == "Unknown"


# ── grid_population ───────────────────────────────────────────────────────────

class TestGridPopulation:
    def test_returns_geodataframe(self, tmp_path):
        result = grid_population(output_path=tmp_path / "test_pop.geojson")
        assert isinstance(result, gpd.GeoDataFrame)

    def test_required_columns_present(self, tmp_path):
        result = grid_population(output_path=tmp_path / "test_pop.geojson")
        for col in ["cell_id", "lon", "lat", "population", "geometry"]:
            assert col in result.columns

    def test_population_values_positive(self, tmp_path):
        result = grid_population(output_path=tmp_path / "test_pop.geojson")
        assert (result["population"] > 0).all()

    def test_total_population_reasonable(self, tmp_path):
        result = grid_population(output_path=tmp_path / "test_pop.geojson")
        total = result["population"].sum()
        # Morocco population ~37M — synthetic should be within 10%
        assert 33_000_000 <= total <= 41_000_000, f"Unexpected total: {total:,.0f}"

    def test_all_coords_within_morocco(self, tmp_path):
        result = grid_population(output_path=tmp_path / "test_pop.geojson")
        assert (result["lon"] >= -13.2).all()
        assert (result["lon"] <= -0.99).all()
        assert (result["lat"] >= 27.6).all()
        assert (result["lat"] <= 35.95).all()

    def test_output_file_saved(self, tmp_path):
        out = tmp_path / "pop.geojson"
        grid_population(output_path=out)
        assert out.exists()
        assert out.stat().st_size > 1000  # Non-trivial file

    def test_crs_is_4326(self, tmp_path):
        result = grid_population(output_path=tmp_path / "test_pop.geojson")
        assert result.crs.to_epsg() == 4326

    def test_synthetic_grid_reproducible(self):
        """Two runs with same seed should produce identical output."""
        g1 = _synthetic_population_grid(0.1)
        g2 = _synthetic_population_grid(0.1)
        pd.testing.assert_frame_equal(
            g1[["lon", "lat", "population"]].reset_index(drop=True),
            g2[["lon", "lat", "population"]].reset_index(drop=True),
        )

    def test_finer_resolution_more_cells(self, tmp_path):
        coarse = grid_population(
            grid_resolution_deg=0.5, output_path=tmp_path / "coarse.geojson"
        )
        fine = grid_population(
            grid_resolution_deg=0.1, output_path=tmp_path / "fine.geojson"
        )
        assert len(fine) > len(coarse)

    def test_missing_tiff_falls_back_gracefully(self, tmp_path):
        result = grid_population(
            tiff_path=tmp_path / "nonexistent.tif",
            output_path=tmp_path / "pop.geojson",
        )
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
