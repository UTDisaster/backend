from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.db import get_engine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export preprocessing-related DB data (disasters/image_pairs/locations/"
            "chat.vlm_assessments) into a JSON snapshot."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write human-readable JSON.",
    )
    return parser.parse_args()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    with engine.connect() as conn:
        disasters = [
            dict(row)
            for row in conn.execute(
                text("SELECT id, type FROM disasters ORDER BY id")
            ).mappings()
        ]
        image_pairs = [
            dict(row)
            for row in conn.execute(
                text(
                    """
                    SELECT
                        id, disaster_id, pair_id, pre_path, post_path,
                        pre_image_id, post_image_id,
                        pre_min_lat, pre_min_lng, pre_max_lat, pre_max_lng,
                        post_min_lat, post_min_lng, post_max_lat, post_max_lng
                    FROM image_pairs
                    ORDER BY id
                    """
                )
            ).mappings()
        ]
        locations = [
            dict(row)
            for row in conn.execute(
                text(
                    """
                    SELECT
                        id, location_uid, image_pair_id, feature_type, classification,
                        street, city, county, full_address, address_source,
                        address_fetched_at,
                        ST_AsText(geom) AS geom_wkt,
                        ST_AsText(centroid) AS centroid_wkt
                    FROM locations
                    ORDER BY id
                    """
                )
            ).mappings()
        ]
        vlm_assessments = [
            dict(row)
            for row in conn.execute(
                text(
                    """
                    SELECT
                        id, location_id, damage_level, confidence, description, created_at
                    FROM chat.vlm_assessments
                    ORDER BY id
                    """
                )
            ).mappings()
        ]

    for row in locations:
        row["address_fetched_at"] = _iso(row.get("address_fetched_at"))
    for row in vlm_assessments:
        row["created_at"] = _iso(row.get("created_at"))

    snapshot = {
        "metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "format_version": 1,
        },
        "disasters": disasters,
        "image_pairs": image_pairs,
        "locations": locations,
        "vlm_assessments": vlm_assessments,
    }

    with output_path.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(snapshot, f, indent=2, sort_keys=True)
            f.write("\n")
        else:
            json.dump(snapshot, f, separators=(",", ":"), sort_keys=True)

    print(
        "exported snapshot: "
        f"disasters={len(disasters)} image_pairs={len(image_pairs)} "
        f"locations={len(locations)} vlm_assessments={len(vlm_assessments)} "
        f"-> {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
