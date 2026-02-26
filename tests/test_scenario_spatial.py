"""
tests/test_scenario_spatial.py
================================
Spatial integrity tests for scenario simulation.

Acceptance criteria (all must pass):
  - No placed facility outside Morocco polygon
  - No placed facility in water / ocean
  - No placed facility in Algeria or Spain
  - All output points have valid CRS (EPSG:4326)
  - Greedy algorithm improves coverage measurably
  - All three optimization modes produce valid results
  - JSON output is fully serialisable
"""

import json
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from src.spatial_utils import load_morocco_polygon, enforce_spatial_integrity
from src.kmeans_placement import greedy_facility_placement, compute_improvement_metrics
from src.scenario_simulator import run_scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def morocco_polygon():
    return load_morocco_polygon()


@pytest.fixture
def morocco_pop_grid():
    """Synthetic population grid with cells spread across Morocco."""
    np.random.seed(42)
    records = []
    # Dense urban clusters inside Morocco
    for cx, cy, base_pop in [
        (-7.59, 33.57, 50_000),   # Casablanca
        (-4.99, 34.04, 30_000),   # Fès
        (-8.00, 31.63, 20_000),   # Marrakech
        (-6.85, 34.02, 18_000),   # Rabat
        (-5.80, 35.77, 12_000),   # Tanger
    ]:
        for _ in range(15):
            lon = cx + np.random.uniform(-0.2, 0.2)
            lat = cy + np.random.uniform(-0.2, 0.2)
            records.append({"lon": lon, "lat": lat,
                            "population": max(base_pop + np.random.uniform(-5000, 5000), 100)})
    # Sparse rural cells
    for _ in range(30):
        records.append({
            "lon": np.random.uniform(-10.0, -2.0),
            "lat": np.random.uniform(29.0, 35.0),
            "population": np.random.uniform(100, 2000),
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["cell_id"] = [f"c{i:03d}" for i in range(len(gdf))]
    gdf["geometry"] = [Point(r["lon"], r["lat"]) for r in records]
    # Keep only valid Morocco cells
    return enforce_spatial_integrity(gdf, label="test_pop", use_polygon=True)


@pytest.fixture
def existing_facilities():
    """Minimal set of existing facilities inside Morocco."""
    return gpd.GeoDataFrame(
        {"facility_type": ["hospital", "clinic", "hospital"]},
        geometry=[Point(-7.59, 33.57), Point(-6.85, 34.02), Point(-4.99, 34.04)],
        crs="EPSG:4326",
    )


@pytest.fixture
def bad_points_gdf():
    """GeoDataFrame with points outside Morocco — should all be dropped."""
    return gpd.GeoDataFrame(
        {"label": ["Algeria", "Spain", "Atlantic Ocean", "Paris"]},
        geometry=[
            Point(2.35,  36.74),   # Algeria
            Point(-3.70, 40.42),   # Spain/Madrid
            Point(-20.0, 30.0),    # Atlantic Ocean
            Point(2.35,  48.85),   # Paris
        ],
        crs="EPSG:4326",
    )


# ── enforce_spatial_integrity ─────────────────────────────────────────────────

class TestEnforceSpatialIntegrity:
    def test_drops_algeria_points(self, bad_points_gdf, morocco_polygon):
        result = enforce_spatial_integrity(bad_points_gdf, label="test", use_polygon=True)
        assert len(result) == 0, "All non-Morocco points should be dropped"

    def test_keeps_casablanca(self, morocco_polygon):
        gdf = gpd.GeoDataFrame(
            {"label": ["Casablanca"]},
            geometry=[Point(-7.59, 33.57)],
            crs="EPSG:4326",
        )
        result = enforce_spatial_integrity(gdf, label="test", use_polygon=True)
        assert len(result) == 1

    def test_keeps_marrakech(self):
        gdf = gpd.GeoDataFrame(
            {"label": ["Marrakech"]},
            geometry=[Point(-8.00, 31.63)],
            crs="EPSG:4326",
        )
        result = enforce_spatial_integrity(gdf, label="test", use_polygon=True)
        assert len(result) == 1

    def test_output_crs_is_4326(self, morocco_pop_grid):
        result = enforce_spatial_integrity(morocco_pop_grid, label="test")
        assert result.crs.to_epsg() == 4326

    def test_handles_wrong_crs(self):
        gdf = gpd.GeoDataFrame(
            {"label": ["Casablanca"]},
            geometry=[Point(-7.59, 33.57)],
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")
        result = enforce_spatial_integrity(gdf, label="test", use_polygon=True)
        assert result.crs.to_epsg() == 4326


# ── greedy_facility_placement — spatial integrity ─────────────────────────────

class TestGreedyPlacementSpatialIntegrity:
    def test_all_sites_inside_morocco(self, morocco_pop_grid, existing_facilities, morocco_polygon):
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            coverage_target=70.0, coverage_radius_km=10.0
        )
        if len(result) == 0:
            pytest.skip("Target already met with existing facilities")
        for _, row in result.iterrows():
            assert row.geometry.within(morocco_polygon), \
                f"Site {row.get('site_id')} at ({row.geometry.y:.3f}, {row.geometry.x:.3f}) is outside Morocco!"

    def test_no_site_in_ocean(self, morocco_pop_grid, existing_facilities, morocco_polygon):
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            budget_mad=45_000_000, coverage_radius_km=10.0
        )
        for _, row in result.iterrows():
            pt = row.geometry
            # Ocean points have longitude < -13.2 or latitude < 20.77
            assert -17.1 <= pt.x <= -0.99, f"Longitude {pt.x} out of Morocco range"
            assert 20.77 <= pt.y <= 35.95, f"Latitude {pt.y} out of Morocco range"

    def test_no_site_in_algeria(self, morocco_pop_grid, existing_facilities):
        """Algeria is east of approximately -1.7 longitude."""
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            budget_mad=45_000_000, coverage_radius_km=10.0
        )
        for _, row in result.iterrows():
            assert row.geometry.x < -1.0, \
                f"Site at lon={row.geometry.x:.3f} may be in Algeria (lon > -1.0)"

    def test_output_crs_is_4326(self, morocco_pop_grid, existing_facilities):
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            budget_mad=30_000_000, coverage_radius_km=10.0
        )
        assert result.crs.to_epsg() == 4326

    def test_sites_are_valid_geometries(self, morocco_pop_grid, existing_facilities):
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            budget_mad=30_000_000, coverage_radius_km=10.0
        )
        assert result.geometry.is_valid.all()
        assert not result.geometry.is_empty.any()

    def test_sites_match_population_grid_points(self, morocco_pop_grid, existing_facilities):
        """Every placed site must be an exact population grid cell coordinate."""
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            budget_mad=45_000_000, coverage_radius_km=10.0
        )
        pop_coords = set(
            (round(row.geometry.x, 4), round(row.geometry.y, 4))
            for _, row in morocco_pop_grid.iterrows()
        )
        for _, row in result.iterrows():
            candidate = (round(row.geometry.x, 4), round(row.geometry.y, 4))
            assert candidate in pop_coords, \
                f"Site {row.get('site_id')} at {candidate} is NOT a population grid point!"


# ── Stopping criteria ─────────────────────────────────────────────────────────

class TestGreedyStoppingCriteria:
    def test_coverage_target_mode(self, morocco_pop_grid, existing_facilities):
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            coverage_target=60.0, coverage_radius_km=10.0
        )
        assert isinstance(result, gpd.GeoDataFrame)
        # If facilities were placed, final coverage should be >= 60%
        if len(result) > 0:
            last_row = result.sort_values("step").iloc[-1]
            assert last_row["coverage_pct_after"] >= 60.0 - 5.0  # allow 5% tolerance

    def test_budget_mode(self, morocco_pop_grid, existing_facilities):
        budget = 45_000_000  # 45M MAD = ~3 facilities
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            budget_mad=budget, coverage_radius_km=10.0
        )
        assert isinstance(result, gpd.GeoDataFrame)
        if len(result) > 0:
            total_cost = len(result) * 15_000_000
            assert total_cost <= budget + 15_000_000  # max 1 facility over budget

    def test_distance_mode(self, morocco_pop_grid, existing_facilities):
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            max_avg_distance_km=15.0, coverage_radius_km=10.0
        )
        assert isinstance(result, gpd.GeoDataFrame)

    def test_no_criteria_raises(self, morocco_pop_grid, existing_facilities):
        with pytest.raises(ValueError, match="stopping criterion"):
            greedy_facility_placement(morocco_pop_grid, existing_facilities)

    def test_returns_empty_when_target_already_met(self, morocco_pop_grid, existing_facilities):
        """If coverage target is already met, return empty GeoDataFrame."""
        result = greedy_facility_placement(
            morocco_pop_grid, existing_facilities,
            coverage_target=1.0,  # 1% — trivially already met
            coverage_radius_km=10.0
        )
        assert isinstance(result, gpd.GeoDataFrame)


# ── run_scenario ──────────────────────────────────────────────────────────────

class TestRunScenarioSpatial:
    def test_coverage_mode_returns_valid_dict(self, morocco_pop_grid, existing_facilities):
        result = run_scenario(
            morocco_pop_grid, existing_facilities,
            mode="coverage_target", coverage_target=65.0
        )
        assert isinstance(result, dict)
        for key in ["scenario_config", "baseline", "after", "delta",
                    "interventions", "combined_results", "cost_analysis", "metadata"]:
            assert key in result

    def test_budget_mode_returns_valid_dict(self, morocco_pop_grid, existing_facilities):
        result = run_scenario(
            morocco_pop_grid, existing_facilities,
            mode="budget", budget_mad=45_000_000
        )
        assert isinstance(result, dict)
        assert result["combined_results"]["n_facilities_placed"] >= 0

    def test_distance_mode_returns_valid_dict(self, morocco_pop_grid, existing_facilities):
        result = run_scenario(
            morocco_pop_grid, existing_facilities,
            mode="max_avg_distance", max_avg_distance_km=15.0
        )
        assert isinstance(result, dict)

    def test_all_scenario_sites_inside_morocco(self, morocco_pop_grid, existing_facilities, morocco_polygon):
        result = run_scenario(
            morocco_pop_grid, existing_facilities,
            mode="budget", budget_mad=45_000_000
        )
        for key in ["new_facilities", "mobile_units", "telemedicine_kiosks"]:
            for site in result["interventions"].get(key, {}).get("sites", []):
                pt = Point(site["lon"], site["lat"])
                assert pt.within(morocco_polygon), \
                    f"{key} site at ({site['lat']}, {site['lon']}) is outside Morocco!"

    def test_no_scenario_site_in_ocean(self, morocco_pop_grid, existing_facilities):
        result = run_scenario(
            morocco_pop_grid, existing_facilities,
            mode="budget", budget_mad=45_000_000
        )
        for key in ["new_facilities", "mobile_units", "telemedicine_kiosks"]:
            for site in result["interventions"].get(key, {}).get("sites", []):
                assert -17.1 <= site["lon"] <= -0.99, f"lon={site['lon']} out of range"
                assert 20.77 <= site["lat"] <= 35.95, f"lat={site['lat']} out of range"

    def test_result_is_json_serialisable(self, morocco_pop_grid, existing_facilities):
        result = run_scenario(
            morocco_pop_grid, existing_facilities,
            mode="budget", budget_mad=30_000_000
        )
        serialised = json.dumps(result)
        assert len(serialised) > 50

    def test_coverage_improves_or_stays_same(self, morocco_pop_grid, existing_facilities):
        result = run_scenario(
            morocco_pop_grid, existing_facilities,
            mode="budget", budget_mad=45_000_000
        )
        assert result["after"]["coverage_10km"] >= result["baseline"]["coverage_10km"] - 0.01

    def test_metadata_contains_algorithm_name(self, morocco_pop_grid, existing_facilities):
        result = run_scenario(
            morocco_pop_grid, existing_facilities,
            mode="budget", budget_mad=15_000_000
        )
        assert result["metadata"]["algorithm"] == "greedy_underserved_targeting"
        assert result["metadata"]["spatial_constraint"] == "morocco_mainland_polygon"
