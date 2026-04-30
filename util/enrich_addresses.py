from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import httpx
from sqlalchemy import text
from tqdm import tqdm

from app.db import get_engine

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
CENSUS_BENCHMARK = "Public_AR_Current"
CENSUS_VINTAGE = "Current_Current"
REQUEST_TIMEOUT_SECONDS = 5.0
RATE_LIMIT_SLEEP_SECONDS = 0.2
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000

# Nominatim fallback lives in a future PR once Census gaps are quantified.


def geocode_coords(
    client: httpx.Client, lat: float, lng: float
) -> Optional[dict[str, Optional[str]]]:
    params = {
        "x": lng,
        "y": lat,
        "benchmark": CENSUS_BENCHMARK,
        "vintage": CENSUS_VINTAGE,
        "format": "json",
    }
    try:
        resp = client.get(CENSUS_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    result = payload.get("result") or {}
    geographies = result.get("geographies") or {}

    counties = geographies.get("Counties") or []
    places = geographies.get("Incorporated Places") or geographies.get("Census Designated Places") or []

    city = places[0].get("NAME") if places else None
    county = counties[0].get("NAME") if counties else None
    street = None

    address_matches = result.get("addressMatches") or []
    if address_matches:
        components = address_matches[0].get("addressComponents") or {}
        street_parts = [
            components.get("preQualifier"),
            components.get("preDirection"),
            components.get("preType"),
            components.get("streetName"),
            components.get("suffixType"),
            components.get("suffixDirection"),
        ]
        street = " ".join(p for p in street_parts if p).strip() or None
        full_address = address_matches[0].get("matchedAddress")
    else:
        full_address = None

    if not any([street, city, county, full_address]):
        return None

    return {
        "street": street,
        "city": city,
        "county": county,
        "full_address": full_address,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse-geocode location centroids via the U.S. Census Geocoder.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


SELECT_SQL = """
SELECT id, ST_Y(centroid) AS lat, ST_X(centroid) AS lng
FROM locations
WHERE address_fetched_at IS NULL
ORDER BY id
LIMIT :limit
"""

UPDATE_SQL = """
UPDATE locations
SET street = :street,
    city = :city,
    county = :county,
    full_address = :full_address,
    address_source = 'census',
    address_fetched_at = now()
WHERE id = :id
"""


def main() -> int:
    args = _parse_args()
    limit = max(1, min(args.limit, MAX_LIMIT))

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(SELECT_SQL), {"limit": limit}).mappings().all()

    if not rows:
        print("no locations need geocoding")
        return 0

    updates = 0
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for row in tqdm(rows, desc="geocoding"):
            result = geocode_coords(client, float(row["lat"]), float(row["lng"]))
            if result is None:
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                continue

            if args.dry_run:
                print(f"[dry-run] id={row['id']} -> {result}")
            else:
                with engine.begin() as conn:
                    conn.execute(text(UPDATE_SQL), {"id": row["id"], **result})
                updates += 1

            time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    if args.dry_run:
        print(f"[dry-run] would update {sum(1 for _ in rows)} row(s)")
    else:
        print(f"updated {updates} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
