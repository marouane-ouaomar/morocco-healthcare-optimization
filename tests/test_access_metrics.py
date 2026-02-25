"""
tests/test_access_metrics.py
=============================
Unit tests for src/access_metrics.py.
Uses synthetic population grids and facility locations with
known-answer geometry so results can be verified exactly.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.access_metrics import (
    nearest_facility_distance,
    population_weighted_distance,
    coverage_within_radius,
    population_per_facility_ratio,
    compute_all_metrics,
    _to_metric_coords,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def single_facility_gdf() -> gpd.GeoDataFrame:
    """One facility at Casablanca centre."""
    return gpd.GeoDataFrame(
        {"facility_type": ["hospital"], "region": ["Casablanca-Settat"]},
        geometry=[Point(-7.59, 33.57)],
        crs="EPSG:4326",
    )


@pytest.fixture
def two_facilities_gdf() -> gpd.GeoDataFrame:
    """Two facilities: Casablanca and Rabat."""
    return gpd.GeoDataFrame(
        {
            "facility_type": ["hospital", "clinic"],
            "region": ["Casablanca-Settat", "Rabat-Salé-Kénitra"],
        },
        geometry=[Point(-7.59, 33.57), Point(-6.85, 34.02)],
        crs="EPSG:4326",
    )


@pytest.fixture
def pop_grid_near_casa() -> gpd.GeoDataFrame:
    """Three population cells all within ~5 km of Casablanca facility."""
    return gpd.GeoDataFrame(
        {
            "cell_id": ["c0", "c1", "c2"],
            "lon": [-7.59, -7.60, -7.58],
            "lat": [33.57, 33.58, 33.56],
            "population": [1000.0, 2000.0, 3000.0],
        },
        geometry=[
            Point(-7.59, 33.57),
            Point(-7.60, 33.58),
            Point(-7.58, 33.56),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def pop_grid_far() -> gpd.GeoDataFrame:
    """Population cells far from any facility (Sahara region)."""
    return gpd.GeoDataFrame(
        {
            "cell_id": ["f0", "f1"],
            "lon": [-3.0, -4.0],
            "lat": [28.5, 28.0],
            "population": [500.0, 800.0],
        },
        geometry=[Point(-3.0, 28.5), Point(-4.0, 28.0)],
        crs="EPSG:4326",
    )


@pytest.fixture
def pop_grid_mixed(pop_grid_near_casa, pop_grid_far) -> gpd.GeoDataFrame:
    """Mix of near and far cells."""
    return pd.concat(
        [pop_grid_near_casa, pop_grid_far], ignore_index=True
    ).pipe(gpd.GeoDataFrame, crs="EPSG:4326")


@pytest.fixture
def multi_type_facilities_gdf() -> gpd.GeoDataFrame:
    """Facilities of different types in Morocco."""
    return gpd.GeoDataFrame(
        {
            "facility_type": ["hospital", "hospital", "clinic", "pharmacy"],
            "region": [
                "Casablanca-Settat", "Rabat-Salé-Kénitra",
                "Fès-Meknès", "Marrakech-Safi",
            ],
        },
        geometry=[
            Point(-7.59, 33.57),
            Point(-6.85, 34.02),
            Point(-4.99, 34.04),
            Point(-8.00, 31.63),
        ],
        crs="EPSG:4326",
    )


# ── _to_metric_coords ─────────────────────────────────────────────────────────

class TestToMetricCoords:
    def test_returns_2d_array(self, single_facility_gdf):
        coords = _to_metric_coords(single_facility_gdf)
        assert coords.ndim == 2
        assert coords.shape[1] == 2

    def test_row_count_matches_input(self, two_facilities_gdf):
        coords = _to_metric_coords(two_facilities_gdf)
        assert coords.shape[0] == 2

    def test_values_are_metres(self, single_facility_gdf):
        coords = _to_metric_coords(single_facility_gdf)
        # Casablanca in Web Mercator should be around -845000, 3980000 m
        assert abs(coords[0, 0]) > 100_000   # x in metres, not degrees
        assert abs(coords[0, 1]) > 100_000   # y in metres


# ── nearest_facility_distance ─────────────────────────────────────────────────

class TestNearestFacilityDistance:
    def test_returns_series(self, pop_grid_near_casa, single_facility_gdf):
        result = nearest_facility_distance(pop_grid_near_casa, single_facility_gdf)
        assert isinstance(result, pd.Series)

    def test_length_matches_pop_grid(self, pop_grid_near_casa, single_facility_gdf):
        result = nearest_facility_distance(pop_grid_near_casa, single_facility_gdf)
        assert len(result) == len(pop_grid_near_casa)

    def test_distances_are_positive(self, pop_grid_near_casa, single_facility_gdf):
        result = nearest_facility_distance(pop_grid_near_casa, single_facility_gdf)
        assert (result >= 0).all()

    def test_collocated_point_has_zero_distance(self, single_facility_gdf):
        """A population cell at the exact facility location → 0 km."""
        pop = gpd.GeoDataFrame(
            {"cell_id": ["x"], "lon": [-7.59], "lat": [33.57], "population": [1.0]},
            geometry=[Point(-7.59, 33.57)],
            crs="EPSG:4326",
        )
        result = nearest_facility_distance(pop, single_facility_gdf)
        assert result.iloc[0] == pytest.approx(0.0, abs=0.01)

    def test_nearby_cells_have_small_distances(self, pop_grid_near_casa, single_facility_gdf):
        result = nearest_facility_distance(pop_grid_near_casa, single_facility_gdf)
        assert result.max() < 20.0  # All cells within 20 km

    def test_far_cells_have_large_distances(self, pop_grid_far, single_facility_gdf):
        result = nearest_facility_distance(pop_grid_far, single_facility_gdf)
        assert result.min() > 50.0  # Sahara far from Casablanca

    def test_two_facilities_closer_than_one(self, pop_grid_mixed, single_facility_gdf, two_facilities_gdf):
        dist_one = nearest_facility_distance(pop_grid_mixed, single_facility_gdf)
        dist_two = nearest_facility_distance(pop_grid_mixed, two_facilities_gdf)
        # Adding a facility should never increase distances
        assert (dist_two <= dist_one + 0.001).all()

    def test_facility_type_filter(self, pop_grid_near_casa, multi_type_facilities_gdf):
        result = nearest_facility_distance(
            pop_grid_near_casa, multi_type_facilities_gdf, facility_type="hospital"
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(pop_grid_near_casa)

    def test_invalid_facility_type_raises(self, pop_grid_near_casa, multi_type_facilities_gdf):
        with pytest.raises(ValueError, match="No facilities found"):
            nearest_facility_distance(
                pop_grid_near_casa, multi_type_facilities_gdf, facility_type="nonexistent"
            )

    def test_empty_pop_gdf_raises(self, single_facility_gdf):
        empty = gpd.GeoDataFrame({"population": []}, geometry=[], crs="EPSG:4326")
        with pytest.raises(ValueError, match="empty"):
            nearest_facility_distance(empty, single_facility_gdf)

    def test_empty_facilities_gdf_raises(self, pop_grid_near_casa):
        empty = gpd.GeoDataFrame({"facility_type": []}, geometry=[], crs="EPSG:4326")
        with pytest.raises(ValueError, match="empty"):
            nearest_facility_distance(pop_grid_near_casa, empty)

    def test_index_preserved(self, pop_grid_near_casa, single_facility_gdf):
        result = nearest_facility_distance(pop_grid_near_casa, single_facility_gdf)
        assert list(result.index) == list(pop_grid_near_casa.index)


# ── population_weighted_distance ──────────────────────────────────────────────

class TestPopulationWeightedDistance:
    def test_returns_float(self, pop_grid_near_casa, single_facility_gdf):
        result = population_weighted_distance(pop_grid_near_casa, single_facility_gdf)
        assert isinstance(result, float)

    def test_result_is_positive(self, pop_grid_near_casa, single_facility_gdf):
        result = population_weighted_distance(pop_grid_near_casa, single_facility_gdf)
        assert result >= 0.0

    def test_weighted_closer_than_unweighted_when_high_pop_near_facility(self):
        """
        If high-population cells are closer to the facility,
        weighted mean < unweighted mean.
        """
        fac = gpd.GeoDataFrame(
            {"facility_type": ["hospital"]},
            geometry=[Point(-7.59, 33.57)],
            crs="EPSG:4326",
        )
        pop = gpd.GeoDataFrame(
            {
                "cell_id": ["near", "far"],
                "lon": [-7.59, -5.0],
                "lat": [33.57, 33.57],
                "population": [100_000.0, 1.0],  # almost all pop near facility
            },
            geometry=[Point(-7.59, 33.57), Point(-5.0, 33.57)],
            crs="EPSG:4326",
        )
        result = population_weighted_distance(pop, fac)
        assert result < 100.0  # Should be close to 0, not 130+ km

    def test_missing_population_column_raises(self, two_facilities_gdf):
        pop_no_col = gpd.GeoDataFrame(
            {"cell_id": ["x"]},
            geometry=[Point(-7.59, 33.57)],
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="population"):
            population_weighted_distance(pop_no_col, two_facilities_gdf)

    def test_all_zero_population_raises(self, single_facility_gdf):
        pop = gpd.GeoDataFrame(
            {"cell_id": ["x"], "population": [0.0]},
            geometry=[Point(-7.59, 33.57)],
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="zero"):
            population_weighted_distance(pop, single_facility_gdf)


# ── coverage_within_radius ────────────────────────────────────────────────────

class TestCoverageWithinRadius:
    def test_returns_dict(self, pop_grid_near_casa, single_facility_gdf):
        result = coverage_within_radius(pop_grid_near_casa, single_facility_gdf)
        assert isinstance(result, dict)

    def test_keys_match_radii(self, pop_grid_near_casa, single_facility_gdf):
        result = coverage_within_radius(
            pop_grid_near_casa, single_facility_gdf, radii_km=[5.0, 10.0, 20.0]
        )
        assert set(result.keys()) == {"coverage_5km", "coverage_10km", "coverage_20km"}

    def test_coverage_increases_with_radius(self, pop_grid_mixed, single_facility_gdf):
        result = coverage_within_radius(
            pop_grid_mixed, single_facility_gdf, radii_km=[5.0, 10.0, 50.0, 500.0]
        )
        values = list(result.values())
        assert values == sorted(values), "Coverage should be non-decreasing with radius"

    def test_coverage_100_percent_at_very_large_radius(self, pop_grid_mixed, single_facility_gdf):
        result = coverage_within_radius(
            pop_grid_mixed, single_facility_gdf, radii_km=[10_000.0]
        )
        assert list(result.values())[0] == pytest.approx(100.0, abs=0.1)

    def test_near_facility_has_high_coverage(self, pop_grid_near_casa, single_facility_gdf):
        result = coverage_within_radius(
            pop_grid_near_casa, single_facility_gdf, radii_km=[20.0]
        )
        assert result["coverage_20km"] > 80.0

    def test_values_between_0_and_100(self, pop_grid_mixed, two_facilities_gdf):
        result = coverage_within_radius(pop_grid_mixed, two_facilities_gdf)
        for v in result.values():
            assert 0.0 <= v <= 100.0

    def test_missing_population_raises(self, two_facilities_gdf):
        pop = gpd.GeoDataFrame(
            {"cell_id": ["x"]}, geometry=[Point(-7.59, 33.57)], crs="EPSG:4326"
        )
        with pytest.raises(ValueError, match="population"):
            coverage_within_radius(pop, two_facilities_gdf)


# ── population_per_facility_ratio ─────────────────────────────────────────────

class TestPopulationPerFacilityRatio:
    def test_returns_dataframe(self, pop_grid_near_casa, multi_type_facilities_gdf):
        result = population_per_facility_ratio(pop_grid_near_casa, multi_type_facilities_gdf)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, pop_grid_near_casa, multi_type_facilities_gdf):
        result = population_per_facility_ratio(pop_grid_near_casa, multi_type_facilities_gdf)
        for col in ["region", "facility_type", "facility_count", "pop_per_facility"]:
            assert col in result.columns

    def test_national_row_present(self, pop_grid_near_casa, multi_type_facilities_gdf):
        result = population_per_facility_ratio(pop_grid_near_casa, multi_type_facilities_gdf)
        assert "National" in result["region"].values

    def test_ratio_is_positive(self, pop_grid_near_casa, multi_type_facilities_gdf):
        result = population_per_facility_ratio(pop_grid_near_casa, multi_type_facilities_gdf)
        finite_mask = np.isfinite(result["pop_per_facility"].values.astype(float))
        assert (result.loc[finite_mask, "pop_per_facility"] > 0).all()

    def test_more_facilities_lower_ratio(self):
        pop = gpd.GeoDataFrame(
            {"cell_id": ["x"], "population": [100_000.0]},
            geometry=[Point(-7.59, 33.57)],
            crs="EPSG:4326",
        )
        one_fac = gpd.GeoDataFrame(
            {"facility_type": ["hospital"], "region": ["R1"]},
            geometry=[Point(-7.59, 33.57)],
            crs="EPSG:4326",
        )
        two_fac = gpd.GeoDataFrame(
            {"facility_type": ["hospital", "hospital"], "region": ["R1", "R1"]},
            geometry=[Point(-7.59, 33.57), Point(-7.60, 33.57)],
            crs="EPSG:4326",
        )
        r1 = population_per_facility_ratio(pop, one_fac)
        r2 = population_per_facility_ratio(pop, two_fac)
        national_1 = r1[(r1["region"] == "National") & (r1["facility_type"] == "all")]["pop_per_facility"].iloc[0]
        national_2 = r2[(r2["region"] == "National") & (r2["facility_type"] == "all")]["pop_per_facility"].iloc[0]
        assert national_2 < national_1


# ── compute_all_metrics ───────────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_returns_dataframe(self, pop_grid_near_casa, single_facility_gdf):
        result = compute_all_metrics(pop_grid_near_casa, single_facility_gdf)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_pop_grid(self, pop_grid_near_casa, single_facility_gdf):
        result = compute_all_metrics(pop_grid_near_casa, single_facility_gdf)
        assert len(result) == len(pop_grid_near_casa)

    def test_required_columns_present(self, pop_grid_near_casa, single_facility_gdf):
        result = compute_all_metrics(pop_grid_near_casa, single_facility_gdf)
        for col in ["cell_id", "population", "nearest_facility_km"]:
            assert col in result.columns

    def test_coverage_flag_columns_present(self, pop_grid_near_casa, single_facility_gdf):
        result = compute_all_metrics(
            pop_grid_near_casa, single_facility_gdf, radii_km=[5.0, 10.0]
        )
        assert "within_5km" in result.columns
        assert "within_10km" in result.columns

    def test_coverage_flags_are_boolean(self, pop_grid_near_casa, single_facility_gdf):
        result = compute_all_metrics(
            pop_grid_near_casa, single_facility_gdf, radii_km=[10.0]
        )
        assert result["within_10km"].dtype == bool

    def test_no_osrm_column_without_url(self, pop_grid_near_casa, single_facility_gdf):
        result = compute_all_metrics(pop_grid_near_casa, single_facility_gdf)
        assert "road_distance_km" not in result.columns

    def test_distances_non_negative(self, pop_grid_near_casa, single_facility_gdf):
        result = compute_all_metrics(pop_grid_near_casa, single_facility_gdf)
        assert (result["nearest_facility_km"] >= 0).all()
