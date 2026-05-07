from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import text
from tqdm import tqdm

from app.db import get_engine

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
CENSUS_BENCHMARK = "Public_AR_Current"
CENSUS_VINTAGE = "Current_Current"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_USER_AGENT = "UTDisaster-Enrichment/1.0"
REQUEST_TIMEOUT_SECONDS = 5.0
RATE_LIMIT_SLEEP_SECONDS = 0.2
NOMINATIM_RATE_LIMIT_SECONDS = 1.0  # Nominatim requires max 1 req/sec
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000

DEFAULT_MAP_FILENAME = "address_map.json"


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


def reverse_geocode_nominatim(
    client: httpx.Client, lat: float, lng: float
) -> Optional[dict[str, Optional[str]]]:
    """Call Nominatim reverse geocode and extract street/address info."""
    params = {"lat": lat, "lon": lng, "format": "json", "addressdetails": "1"}
    try:
        resp = client.get(NOMINATIM_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    address = payload.get("address") or {}
    road = address.get("road")
    if not road:
        return None

    display_name = payload.get("display_name")
    return {"street": road, "full_address": display_name}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse-geocode location centroids via the U.S. Census Geocoder.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--address-map-path",
        default=None,
        help=(
            "Optional path to JSON map keyed by location_uid. "
            "Defaults to $PARSED_DATA_DIR/address_map.json when present."
        ),
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Use only address map data; do not call Census for misses.",
    )
    parser.add_argument(
        "--nominatim",
        action="store_true",
        help="Use Nominatim to fill missing street data on already-enriched rows.",
    )
    return parser.parse_args()


SELECT_SQL = """
SELECT id, location_uid, ST_Y(centroid) AS lat, ST_X(centroid) AS lng
FROM locations
WHERE address_fetched_at IS NULL
ORDER BY id
LIMIT :limit
"""

SELECT_MISSING_STREET_SQL = """
SELECT id, location_uid, ST_Y(centroid) AS lat, ST_X(centroid) AS lng
FROM locations
WHERE address_fetched_at IS NOT NULL AND street IS NULL
ORDER BY id
LIMIT :limit
"""

UPDATE_SQL_TEMPLATE = """
UPDATE locations
SET street = :street,
    city = :city,
    county = :county,
    full_address = :full_address,
    address_source = :address_source,
    address_fetched_at = now()
WHERE id = :id
"""

UPDATE_STREET_SQL = """
UPDATE locations
SET street = :street,
    full_address = COALESCE(full_address, :full_address),
    address_source = 'nominatim'
WHERE id = :id
"""


def _default_map_path() -> Optional[Path]:
    parsed_data_dir = (os.getenv("PARSED_DATA_DIR", "") or "").strip()
    if not parsed_data_dir:
        return None
    candidate = Path(parsed_data_dir).expanduser() / DEFAULT_MAP_FILENAME
    return candidate


def _normalize_map_value(raw: object) -> Optional[dict[str, Optional[str]]]:
    if not isinstance(raw, dict):
        return None
    street = raw.get("street")
    city = raw.get("city")
    county = raw.get("county")
    full_address = raw.get("full_address")
    normalized = {
        "street": street if isinstance(street, str) and street.strip() else None,
        "city": city if isinstance(city, str) and city.strip() else None,
        "county": county if isinstance(county, str) and county.strip() else None,
        "full_address": (
            full_address
            if isinstance(full_address, str) and full_address.strip()
            else None
        ),
    }
    if not any(normalized.values()):
        return None
    return normalized


def _load_address_map(path: Optional[Path]) -> dict[str, dict[str, Optional[str]]]:
    if path is None or not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}

    if not isinstance(payload, dict):
        return {}

    loaded: dict[str, dict[str, Optional[str]]] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        normalized = _normalize_map_value(value)
        if normalized is None:
            continue
        loaded[key] = normalized
    return loaded


def _run_nominatim_mode(args: argparse.Namespace, limit: int) -> int:
    """Fill missing street data using Nominatim reverse geocoding."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(SELECT_MISSING_STREET_SQL), {"limit": limit}).mappings().all()

    if not rows:
        print("no locations with missing street found")
        return 0

    print(f"found {len(rows)} location(s) missing street data")
    updates = 0
    errors = 0
    headers = {"User-Agent": NOMINATIM_USER_AGENT}

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, headers=headers) as client:
        for row in tqdm(rows, desc="nominatim"):
            result = reverse_geocode_nominatim(client, float(row["lat"]), float(row["lng"]))

            if result is None:
                errors += 1
                time.sleep(NOMINATIM_RATE_LIMIT_SECONDS)
                continue

            if args.dry_run:
                print(f"[dry-run] id={row['id']} -> street={result['street']}")
            else:
                with engine.begin() as conn:
                    conn.execute(text(UPDATE_STREET_SQL), {"id": row["id"], **result})
                updates += 1

            time.sleep(NOMINATIM_RATE_LIMIT_SECONDS)

    if args.dry_run:
        print(f"[dry-run] would update {len(rows) - errors} of {len(rows)} row(s)")
    else:
        print(f"updated {updates} row(s), {errors} error(s)/no-road")
    return 0


def main() -> int:
    args = _parse_args()
    limit = max(1, min(args.limit, MAX_LIMIT))

    if args.nominatim:
        return _run_nominatim_mode(args, limit)

    map_path = (
        Path(args.address_map_path).expanduser()
        if args.address_map_path
        else _default_map_path()
    )
    address_map = _load_address_map(map_path)

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(SELECT_SQL), {"limit": limit}).mappings().all()

    if not rows:
        print("no locations need geocoding")
        return 0

    updates = 0
    map_hits = 0
    map_misses = 0
    no_match_marked = 0
    census_updates = 0
    if map_path is not None and map_path.is_file():
        print(f"loaded address map entries: {len(address_map)} from {map_path}")
    elif args.address_map_path:
        print(f"address map path not found: {map_path}")

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for row in tqdm(rows, desc="geocoding"):
            result = None
            source = None
            location_uid = row["location_uid"]
            if isinstance(location_uid, str):
                result = address_map.get(location_uid)
            if result is not None:
                map_hits += 1
                source = "map"
            else:
                map_misses += 1
                if not args.map_only:
                    result = geocode_coords(client, float(row["lat"]), float(row["lng"]))
                    if result is not None:
                        source = "census"
                        census_updates += 1

            if result is None and args.map_only:
                source = "map_no_match"
                if args.dry_run:
                    print(f"[dry-run] id={row['id']} -> map miss")
                else:
                    with engine.begin() as conn:
                        conn.execute(
                            text(
                                """
                                UPDATE locations
                                SET address_source = :address_source,
                                    address_fetched_at = now()
                                WHERE id = :id
                                """
                            ),
                            {"id": row["id"], "address_source": source},
                        )
                    no_match_marked += 1
                    updates += 1
                continue

            if result is None:
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                continue

            if args.dry_run:
                print(f"[dry-run] id={row['id']} -> source={source} {result}")
            else:
                with engine.begin() as conn:
                    conn.execute(
                        text(UPDATE_SQL_TEMPLATE),
                        {
                            "id": row["id"],
                            "address_source": source,
                            **result,
                        },
                    )
                updates += 1

            if source == "census":
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    if args.dry_run:
        print(f"[dry-run] would update {sum(1 for _ in rows)} row(s)")
    else:
        print(f"updated {updates} row(s)")
        print(
            f"map_hits={map_hits} map_misses={map_misses} "
            f"census_updates={census_updates} map_no_match_marked={no_match_marked}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
