"""
scripts/fetch_osm_facilities.py
================================
Fetches healthcare facility data for Morocco from the Overpass API
and saves raw GeoJSON to data/raw/facilities_raw.geojson.

Usage:
    python scripts/fetch_osm_facilities.py
    python scripts/fetch_osm_facilities.py --region "Casablanca-Settat"
    python scripts/fetch_osm_facilities.py --output data/raw/custom_output.geojson
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Morocco bounding box: south, west, north, east
MOROCCO_BBOX = (27.6, -13.2, 35.95, -0.99)

# Western Sahara (administered by Morocco) can be included optionally
MOROCCO_WITH_WS_BBOX = (20.77, -17.1, 35.95, -0.99)

FACILITY_TYPES = ["hospital", "clinic", "doctors", "pharmacy"]

REQUEST_TIMEOUT = 60          # seconds
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 5             # seconds between retries

# ── Overpass query builder ────────────────────────────────────────────────────

def build_overpass_query(bbox: tuple[float, float, float, float]) -> str:
    """
    Build an Overpass QL query to fetch all healthcare facilities
    within a bounding box.

    Args:
        bbox: (south, west, north, east) in decimal degrees.

    Returns:
        Overpass QL query string.
    """
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"

    facility_blocks = "\n  ".join([
        f'node["amenity"="{ft}"]({bbox_str});'
        f'\n  way["amenity"="{ft}"]({bbox_str});'
        f'\n  relation["amenity"="{ft}"]({bbox_str});'
        for ft in FACILITY_TYPES
    ])

    query = f"""
[out:json][timeout:{REQUEST_TIMEOUT}];
(
  {facility_blocks}
);
out center tags;
"""
    return query.strip()


# ── HTTP fetch with retry ─────────────────────────────────────────────────────

def fetch_overpass(query: str) -> dict[str, Any]:
    """
    Send a query to the Overpass API with retry logic.

    Args:
        query: Overpass QL query string.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        logger.info(f"Overpass API request (attempt {attempt}/{RETRY_ATTEMPTS})")
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "morocco-healthcare-optimization/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"Retrieved {len(data.get('elements', []))} elements")
            return data

        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out (attempt {attempt})")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error: {e} (attempt {attempt})")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e} (attempt {attempt})")

        if attempt < RETRY_ATTEMPTS:
            logger.info(f"Retrying in {RETRY_BACKOFF}s...")
            time.sleep(RETRY_BACKOFF)

    raise RuntimeError(
        f"Overpass API unreachable after {RETRY_ATTEMPTS} attempts. "
        "Check your connection or try a mirror: https://overpass.kumi.systems/api/interpreter"
    )


# ── OSM → GeoJSON conversion ──────────────────────────────────────────────────

def osm_element_to_feature(element: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert a single OSM element (node / way / relation) to a GeoJSON Feature.

    Ways and relations are represented by their center point (from `out center`).

    Args:
        element: Raw OSM element dict from Overpass response.

    Returns:
        GeoJSON Feature dict, or None if coordinates cannot be determined.
    """
    tags = element.get("tags", {})
    el_type = element.get("type")

    # Extract coordinates
    if el_type == "node":
        lon = element.get("lon")
        lat = element.get("lat")
    elif el_type in ("way", "relation"):
        center = element.get("center", {})
        lon = center.get("lon")
        lat = center.get("lat")
    else:
        return None

    if lon is None or lat is None:
        logger.debug(f"Skipping element {element.get('id')} — missing coordinates")
        return None

    # Determine facility type from tags
    amenity = tags.get("amenity", "unknown")

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat],
        },
        "properties": {
            "osm_id": element.get("id"),
            "osm_type": el_type,
            "amenity": amenity,
            "name": tags.get("name") or tags.get("name:en") or tags.get("name:fr") or None,
            "name_ar": tags.get("name:ar") or None,
            "name_fr": tags.get("name:fr") or None,
            "operator": tags.get("operator") or None,
            "operator_type": tags.get("operator:type") or None,  # public / private / ngo
            "healthcare": tags.get("healthcare") or None,
            "beds": tags.get("beds") or None,
            "emergency": tags.get("emergency") or None,
            "opening_hours": tags.get("opening_hours") or None,
            "phone": tags.get("phone") or tags.get("contact:phone") or None,
            "website": tags.get("website") or tags.get("contact:website") or None,
            "addr_city": tags.get("addr:city") or None,
            "addr_region": tags.get("addr:region") or None,
        },
    }


def overpass_to_geojson(data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert full Overpass API response to a GeoJSON FeatureCollection.

    Args:
        data: Raw Overpass API JSON response.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    elements = data.get("elements", [])
    features = []
    skipped = 0

    for element in elements:
        feature = osm_element_to_feature(element)
        if feature is not None:
            features.append(feature)
        else:
            skipped += 1

    logger.info(f"Converted {len(features)} features ({skipped} skipped — no coordinates)")

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": "OpenStreetMap via Overpass API",
            "bbox": MOROCCO_BBOX,
            "facility_types": FACILITY_TYPES,
            "total_features": len(features),
        },
    }


# ── Output ────────────────────────────────────────────────────────────────────

def save_geojson(geojson: dict[str, Any], output_path: Path) -> None:
    """
    Save GeoJSON dict to a file.

    Args:
        geojson: GeoJSON FeatureCollection dict.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Saved {len(geojson['features'])} features → {output_path} ({size_kb:.1f} KB)")


# ── Facility type summary ─────────────────────────────────────────────────────

def print_summary(geojson: dict[str, Any]) -> None:
    """Print a breakdown of fetched facilities by type."""
    from collections import Counter
    counts = Counter(
        f["properties"]["amenity"] for f in geojson["features"]
    )
    logger.info("── Facility breakdown ──────────────────")
    for facility_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {facility_type:<20} {count:>5}")
    logger.info(f"  {'TOTAL':<20} {sum(counts.values()):>5}")
    logger.info("────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Morocco healthcare facilities from OpenStreetMap via Overpass API"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/facilities_raw.geojson"),
        help="Output GeoJSON path (default: data/raw/facilities_raw.geojson)",
    )
    parser.add_argument(
        "--include-western-sahara",
        action="store_true",
        default=False,
        help="Extend bounding box to include Western Sahara",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the Overpass query and exit without fetching",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bbox = MOROCCO_WITH_WS_BBOX if args.include_western_sahara else MOROCCO_BBOX
    query = build_overpass_query(bbox)

    if args.dry_run:
        print("── Overpass Query ──────────────────────────────────────")
        print(query)
        print("────────────────────────────────────────────────────────")
        sys.exit(0)

    logger.info("Starting Morocco facility fetch")
    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Facility types: {FACILITY_TYPES}")

    raw_data = fetch_overpass(query)
    geojson = overpass_to_geojson(raw_data)
    print_summary(geojson)
    save_geojson(geojson, args.output)

    logger.info("✅ Fetch complete. Next step: python -c \"from src.data_prep import run_pipeline; run_pipeline()\"")


if __name__ == "__main__":
    main()
