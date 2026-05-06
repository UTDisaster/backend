"""Fill missing street names via Nominatim (OpenStreetMap) reverse geocoding.

Usage:
    python util/enrich_streets_nominatim.py --limit 100
    python util/enrich_streets_nominatim.py --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

sys.path.insert(0, ".")

import httpx
from sqlalchemy import text
from tqdm import tqdm

from app.db import get_engine

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "UTDisaster-Enrichment/1.0"
REQUEST_TIMEOUT_SECONDS = 10.0
RATE_LIMIT_SLEEP_SECONDS = 1.0  # Nominatim requires max 1 req/sec
DEFAULT_LIMIT = 100
MAX_LIMIT = 5000

SELECT_SQL = """
SELECT id, location_uid, ST_Y(centroid) AS lat, ST_X(centroid) AS lng
FROM locations
WHERE address_fetched_at IS NOT NULL AND street IS NULL
ORDER BY id
LIMIT :limit
"""

UPDATE_SQL = """
UPDATE locations
SET street = :street,
    full_address = COALESCE(full_address, :full_address),
    address_source = 'nominatim'
WHERE id = :id
"""


def reverse_geocode_nominatim(
    client: httpx.Client, lat: float, lng: float
) -> Optional[dict[str, Optional[str]]]:
    """Call Nominatim reverse geocode and extract street info."""
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

    city = address.get("city") or address.get("town") or address.get("village")
    display_name = payload.get("display_name")

    return {"street": road, "full_address": display_name}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing street names via Nominatim reverse geocoding."
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to DB.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    limit = max(1, min(args.limit, MAX_LIMIT))

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(SELECT_SQL), {"limit": limit}).mappings().all()

    if not rows:
        print("no locations with missing street found")
        return 0

    print(f"found {len(rows)} location(s) missing street data")

    updates = 0
    errors = 0
    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, headers=headers) as client:
        for row in tqdm(rows, desc="nominatim"):
            result = reverse_geocode_nominatim(client, float(row["lat"]), float(row["lng"]))

            if result is None:
                errors += 1
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                continue

            if args.dry_run:
                print(f"[dry-run] id={row['id']} -> street={result['street']}")
            else:
                with engine.begin() as conn:
                    conn.execute(text(UPDATE_SQL), {"id": row["id"], **result})
                updates += 1

            time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    if args.dry_run:
        print(f"[dry-run] would update {len(rows) - errors} of {len(rows)} row(s)")
    else:
        print(f"updated {updates} row(s), {errors} error(s)/no-road")
    return 0


if __name__ == "__main__":
    sys.exit(main())
