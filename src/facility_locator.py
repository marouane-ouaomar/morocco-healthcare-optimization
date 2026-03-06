"""
src/facility_locator.py
========================
Phase 5 — Location-aware nearest facility finder.

Given a (lat, lon) and a triage advice level, returns the N nearest
facilities of the appropriate type from the processed facility GeoDataFrame.

Triage level → facility type priority
--------------------------------------
monitor    → pharmacy           (mild symptoms, OTC products)
see_doctor → clinic, doctor     (needs professional evaluation)
emergency  → hospital           (urgent/emergency care)

All searches use a scipy KDTree on (lon, lat) pairs.
Haversine distance is computed for the final results only — the KDTree
uses Euclidean distance on degrees, which is accurate enough for
short-range nearest-neighbour queries in Morocco's latitude band (~27-36°N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# Facility types served per triage level (ordered by preference)
LEVEL_FACILITY_TYPES: dict[str, list[str]] = {
    "monitor":    ["pharmacy"],
    "see_doctor": ["clinic", "doctor", "hospital"],
    "emergency":  ["hospital", "clinic"],
}

# Display labels per facility type
FACILITY_DISPLAY: dict[str, dict[str, str]] = {
    "hospital": {"icon": "🏥", "label": "Hospital",        "color": "#922B21"},
    "clinic":   {"icon": "🩺", "label": "Clinic",           "color": "#1F4E79"},
    "doctor":   {"icon": "👨‍⚕️", "label": "Doctor / Health Centre", "color": "#2F8F9D"},
    "pharmacy": {"icon": "💊", "label": "Pharmacy",         "color": "#1E8449"},
    "other":    {"icon": "🏩", "label": "Health Facility",  "color": "#5D6D7E"},
}

EARTH_RADIUS_KM = 6371.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NearbyFacility:
    name:          str
    facility_type: str
    lat:           float
    lon:           float
    distance_km:   float
    region:        str = "Unknown"
    sector:        str = "unknown"
    icon:          str = "🏥"
    label:         str = "Health Facility"
    color:         str = "#5D6D7E"
    google_maps_url: str = ""

    def __post_init__(self) -> None:
        meta = FACILITY_DISPLAY.get(self.facility_type, FACILITY_DISPLAY["other"])
        self.icon  = meta["icon"]
        self.label = meta["label"]
        self.color = meta["color"]
        self.google_maps_url = (
            f"https://www.google.com/maps/search/?api=1"
            f"&query={self.lat:.6f},{self.lon:.6f}"
        )


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Exact great-circle distance in kilometres."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


# ── Core search ───────────────────────────────────────────────────────────────

def find_nearest(
    user_lat: float,
    user_lon: float,
    facilities_gdf: gpd.GeoDataFrame,
    advice_level: str,
    n: int = 3,
    max_distance_km: float = 100.0,
) -> list[NearbyFacility]:
    """
    Return the N nearest facilities appropriate for the given triage level.

    Parameters
    ----------
    user_lat, user_lon : float
        User's device location (WGS84).
    facilities_gdf : GeoDataFrame
        Processed facility dataset (from data/processed/facilities.geojson).
    advice_level : str
        One of "emergency", "see_doctor", "monitor".
    n : int
        Number of results to return per facility type.
    max_distance_km : float
        Hard cutoff — facilities beyond this are excluded.

    Returns
    -------
    List of NearbyFacility, sorted by distance ascending.
    Empty list if no suitable facilities found within max_distance_km.
    """
    from scipy.spatial import KDTree

    priority_types = LEVEL_FACILITY_TYPES.get(advice_level, ["clinic", "hospital"])

    # Recompute lon/lat from geometry (never trust stale columns)
    gdf = facilities_gdf.copy()
    gdf["_lon"] = gdf.geometry.x
    gdf["_lat"] = gdf.geometry.y
    gdf = gdf[gdf["_lat"].notna() & gdf["_lon"].notna()]

    results: list[NearbyFacility] = []

    for ftype in priority_types:
        subset = gdf[gdf["facility_type"] == ftype]
        if subset.empty:
            continue

        lons = subset["_lon"].values
        lats = subset["_lat"].values
        tree = KDTree(np.column_stack([lons, lats]))

        k = min(n * 3, len(subset))   # over-fetch, then filter by haversine
        dists, idxs = tree.query([user_lon, user_lat], k=k)

        if k == 1:
            dists, idxs = [dists], [idxs]

        for dist_deg, idx in zip(dists, idxs):
            row = subset.iloc[idx]
            dist_km = haversine_km(user_lat, user_lon, row["_lat"], row["_lon"])
            if dist_km > max_distance_km:
                continue

            name = (
                row.get("name_clean", None)
                or row.get("name", None)
                or f"Unnamed {ftype.capitalize()}"
            )
            results.append(NearbyFacility(
                name=str(name),
                facility_type=ftype,
                lat=float(row["_lat"]),
                lon=float(row["_lon"]),
                distance_km=round(dist_km, 2),
                region=str(row.get("region", "Unknown") or "Unknown"),
                sector=str(row.get("sector", "unknown") or "unknown"),
            ))

        # Keep only top-n per type
        type_results = sorted(
            [r for r in results if r.facility_type == ftype],
            key=lambda r: r.distance_km,
        )[:n]
        results = [r for r in results if r.facility_type != ftype] + type_results

    return sorted(results, key=lambda r: r.distance_km)


def find_nearest_all_types(
    user_lat: float,
    user_lon: float,
    facilities_gdf: gpd.GeoDataFrame,
    n_per_type: int = 2,
) -> dict[str, list[NearbyFacility]]:
    """
    Return nearest facilities for ALL types — used for the full nearby panel.
    """
    result: dict[str, list[NearbyFacility]] = {}
    for ftype in ["hospital", "clinic", "doctor", "pharmacy"]:
        found = find_nearest(user_lat, user_lon, facilities_gdf,
                             advice_level="see_doctor", n=n_per_type)
        result[ftype] = [f for f in found if f.facility_type == ftype]
    return result