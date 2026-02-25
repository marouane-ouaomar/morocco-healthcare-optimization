"""
tests/test_optimization.py
===========================
Unit tests for src/kmeans_placement.py and src/scenario_simulator.py.
All tests use synthetic data — no file I/O or API calls.
"""

import json
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from src.kmeans_placement import find_candidate_sites, compute_improvement_metrics
from src.scenario_simulator import run_scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def morocco_pop_grid() -> gpd.GeoDataFrame:
    """
    Synthetic population grid: 3 dense clusters + sparse background.
    Mimics Morocco's urban concentration pattern.
    """
    np.random.seed(42)
    records = []

    # Dense urban clusters
    clusters = [
        (-7.59, 33.57, 50_000),   # Casablanca area
        (-4.99, 34.04, 30_000),   # Fès area
        (-8.00, 31.63, 20_000),   # Marrakech area
    ]
    for cx, cy, base_pop in clusters:
        for _ in range(20):
            lon = cx + np.random.uniform(-0.3, 0.3)
            lat = cy + np.random.uniform(-0.3, 0.3)
            pop = base_pop + np.random.uniform(-5000, 5000)
            records.append({"lon": lon, "lat": lat, "population": max(pop, 100)})

    # Sparse rural background
    for _ in range(40):
        lon = np.random.uniform(-11.0, -2.0)
        lat = np.random.uniform(28.0, 35.5)
        records.append({"lon": lon, "lat": lat, "population": np.random.uniform(100, 2000)})

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["cell_id"] = [f"c{i:03d}" for i in range(len(gdf))]
    gdf["geometry"] = [Point(r["lon"], r["lat"]) for r in records]
    return gdf


@pytest.fixture
def existing_facilities_gdf() -> gpd.GeoDataFrame:
    """A handful of existing facilities in Morocco's main cities."""
    return gpd.GeoDataFrame(
        {
            "facility_type": ["hospital", "clinic", "hospital", "pharmacy"],
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


@pytest.fixture
def remote_facilities_gdf() -> gpd.GeoDataFrame:
    """Facilities far from population clusters — should generate high improvement."""
    return gpd.GeoDataFrame(
        {"facility_type": ["hospital"]},
        geometry=[Point(-2.0, 28.5)],  # Deep south, away from population
        crs="EPSG:4326",
    )


# ── find_candidate_sites ──────────────────────────────────────────────────────

class TestFindCandidateSites:
    def test_returns_geodataframe(self, morocco_pop_grid, existing_facilities_gdf):
        result = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=3)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_correct_number_of_sites(self, morocco_pop_grid, existing_facilities_gdf):
        for n in [1, 3, 5]:
            result = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=n)
            assert len(result) == n

    def test_required_columns_present(self, morocco_pop_grid, existing_facilities_gdf):
        result = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=2)
        for col in ["site_id", "lon", "lat", "cluster_population",
                    "cluster_size", "nearest_existing_km", "too_close_to_existing"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_sites_within_morocco_bbox(self, morocco_pop_grid, existing_facilities_gdf):
        result = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=5)
        assert (result["lon"] >= -13.2).all()
        assert (result["lon"] <= -0.99).all()
        assert (result["lat"] >= 27.6).all()
        assert (result["lat"] <= 35.95).all()

    def test_cluster_population_positive(self, morocco_pop_grid, existing_facilities_gdf):
        result = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=3)
        assert (result["cluster_population"] > 0).all()

    def test_sorted_by_population_descending(self, morocco_pop_grid, existing_facilities_gdf):
        result = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=5)
        pops = result["cluster_population"].tolist()
        assert pops == sorted(pops, reverse=True)

    def test_nearest_existing_km_non_negative(self, morocco_pop_grid, existing_facilities_gdf):
        result = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=3)
        assert (result["nearest_existing_km"] >= 0).all()

    def test_reproducible_with_same_seed(self, morocco_pop_grid, existing_facilities_gdf):
        r1 = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=3, random_state=0)
        r2 = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=3, random_state=0)
        assert list(r1["lon"]) == list(r2["lon"])
        assert list(r1["lat"]) == list(r2["lat"])

    def test_different_seeds_may_differ(self, morocco_pop_grid, existing_facilities_gdf):
        r1 = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=5, random_state=0)
        r2 = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=5, random_state=99)
        # Not guaranteed to differ but very likely with different seeds
        lons1, lons2 = set(round(x, 1) for x in r1["lon"]), set(round(x, 1) for x in r2["lon"])
        # Just check both are valid
        assert len(r1) == len(r2) == 5

    def test_invalid_n_sites_raises(self, morocco_pop_grid, existing_facilities_gdf):
        with pytest.raises(ValueError, match="n_sites"):
            find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=0)

    def test_empty_pop_gdf_raises(self, existing_facilities_gdf):
        empty = gpd.GeoDataFrame({"population": []}, geometry=[], crs="EPSG:4326")
        with pytest.raises(ValueError, match="empty"):
            find_candidate_sites(empty, existing_facilities_gdf, n_sites=2)

    def test_empty_facilities_raises(self, morocco_pop_grid):
        empty = gpd.GeoDataFrame({"facility_type": []}, geometry=[], crs="EPSG:4326")
        with pytest.raises(ValueError, match="empty"):
            find_candidate_sites(morocco_pop_grid, empty, n_sites=2)

    def test_crs_is_wgs84(self, morocco_pop_grid, existing_facilities_gdf):
        result = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=2)
        assert result.crs.to_epsg() == 4326

    def test_sites_target_underserved_areas(self, morocco_pop_grid, remote_facilities_gdf):
        """
        With facilities far from population, all candidate sites should be
        closer to population clusters, not near the remote facility.
        """
        result = find_candidate_sites(morocco_pop_grid, remote_facilities_gdf, n_sites=3)
        # Candidate sites should be in northern Morocco where the population is
        assert result["lat"].mean() > 30.0


# ── compute_improvement_metrics ───────────────────────────────────────────────

class TestComputeImprovementMetrics:
    def test_returns_dict(self, morocco_pop_grid, existing_facilities_gdf):
        candidates = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=2)
        result = compute_improvement_metrics(morocco_pop_grid, existing_facilities_gdf, candidates)
        assert isinstance(result, dict)

    def test_required_top_level_keys(self, morocco_pop_grid, existing_facilities_gdf):
        candidates = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=2)
        result = compute_improvement_metrics(morocco_pop_grid, existing_facilities_gdf, candidates)
        for key in ["before", "after", "delta", "n_new_facilities", "population_newly_covered_10km"]:
            assert key in result, f"Missing key: {key}"

    def test_coverage_improves_or_stays_same(self, morocco_pop_grid, existing_facilities_gdf):
        candidates = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=3)
        result = compute_improvement_metrics(morocco_pop_grid, existing_facilities_gdf, candidates)
        for r in [5, 10, 20]:
            k = f"coverage_{r}km"
            assert result["after"][k] >= result["before"][k] - 0.01

    def test_avg_distance_decreases_or_stays_same(self, morocco_pop_grid, existing_facilities_gdf):
        candidates = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=3)
        result = compute_improvement_metrics(morocco_pop_grid, existing_facilities_gdf, candidates)
        assert result["after"]["avg_distance_km"] <= result["before"]["avg_distance_km"] + 0.01

    def test_newly_covered_population_non_negative(self, morocco_pop_grid, existing_facilities_gdf):
        candidates = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=2)
        result = compute_improvement_metrics(morocco_pop_grid, existing_facilities_gdf, candidates)
        assert result["population_newly_covered_10km"] >= 0

    def test_n_new_facilities_matches_input(self, morocco_pop_grid, existing_facilities_gdf):
        for n in [1, 3, 5]:
            candidates = find_candidate_sites(morocco_pop_grid, existing_facilities_gdf, n_sites=n)
            result = compute_improvement_metrics(morocco_pop_grid, existing_facilities_gdf, candidates)
            assert result["n_new_facilities"] == n

    def test_more_facilities_better_coverage(self, morocco_pop_grid, remote_facilities_gdf):
        """More candidate sites should yield >= coverage than fewer."""
        c3 = find_candidate_sites(morocco_pop_grid, remote_facilities_gdf, n_sites=3)
        c6 = find_candidate_sites(morocco_pop_grid, remote_facilities_gdf, n_sites=6)
        r3 = compute_improvement_metrics(morocco_pop_grid, remote_facilities_gdf, c3)
        r6 = compute_improvement_metrics(morocco_pop_grid, remote_facilities_gdf, c6)
        assert r6["after"]["coverage_10km"] >= r3["after"]["coverage_10km"] - 0.01


# ── run_scenario ──────────────────────────────────────────────────────────────

class TestRunScenario:
    def test_returns_dict(self, morocco_pop_grid, existing_facilities_gdf):
        result = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                              new_facilities=2, mobile_units=1, telemedicine_kiosks=1)
        assert isinstance(result, dict)

    def test_required_top_level_keys(self, morocco_pop_grid, existing_facilities_gdf):
        result = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                              new_facilities=1, mobile_units=0, telemedicine_kiosks=0)
        for key in ["scenario_config", "baseline", "after", "delta",
                    "interventions", "combined_results", "cost_analysis"]:
            assert key in result, f"Missing key: {key}"

    def test_scenario_config_matches_input(self, morocco_pop_grid, existing_facilities_gdf):
        result = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                              new_facilities=2, mobile_units=1, telemedicine_kiosks=3)
        assert result["scenario_config"]["new_facilities"] == 2
        assert result["scenario_config"]["mobile_units"] == 1
        assert result["scenario_config"]["telemedicine_kiosks"] == 3

    def test_interventions_keys_present(self, morocco_pop_grid, existing_facilities_gdf):
        result = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                              new_facilities=1, mobile_units=1, telemedicine_kiosks=1)
        assert "new_facilities" in result["interventions"]
        assert "mobile_units" in result["interventions"]
        assert "telemedicine_kiosks" in result["interventions"]

    def test_cost_analysis_total_positive(self, morocco_pop_grid, existing_facilities_gdf):
        result = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                              new_facilities=1, mobile_units=1, telemedicine_kiosks=1)
        assert result["cost_analysis"]["total_cost_mad"] > 0

    def test_combined_results_distance_reduction_non_negative(self, morocco_pop_grid, existing_facilities_gdf):
        result = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                              new_facilities=2, mobile_units=0, telemedicine_kiosks=0)
        assert result["combined_results"]["distance_reduction_km"] >= 0

    def test_zero_interventions_returns_baseline(self, morocco_pop_grid, existing_facilities_gdf):
        result = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                              new_facilities=0, mobile_units=0, telemedicine_kiosks=0)
        assert result["baseline"]["avg_distance_km"] == result["after"]["avg_distance_km"]

    def test_result_is_json_serialisable(self, morocco_pop_grid, existing_facilities_gdf):
        result = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                              new_facilities=1, mobile_units=1, telemedicine_kiosks=1)
        # Should not raise
        serialised = json.dumps(result)
        assert len(serialised) > 100

    def test_reproducible_with_same_random_state(self, morocco_pop_grid, existing_facilities_gdf):
        r1 = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                          new_facilities=2, mobile_units=1, telemedicine_kiosks=1,
                          random_state=7)
        r2 = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                          new_facilities=2, mobile_units=1, telemedicine_kiosks=1,
                          random_state=7)
        assert r1["after"]["avg_distance_km"] == r2["after"]["avg_distance_km"]

    def test_coverage_values_between_0_and_100(self, morocco_pop_grid, existing_facilities_gdf):
        result = run_scenario(morocco_pop_grid, existing_facilities_gdf,
                              new_facilities=2, mobile_units=1, telemedicine_kiosks=1)
        for stage in ["baseline", "after"]:
            for key, val in result[stage].items():
                if "coverage" in key:
                    assert 0.0 <= val <= 100.0, f"{stage}.{key} = {val} out of range"
