from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.db import get_engine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import preprocessing-related DB snapshot JSON into "
            "disasters/image_pairs/locations/chat.vlm_assessments."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Snapshot JSON path.",
    )
    return parser.parse_args()


def _load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("snapshot must be a JSON object")
    return payload


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"snapshot file not found: {input_path}")

    snapshot = _load_snapshot(input_path)
    disasters = snapshot.get("disasters") or []
    image_pairs = snapshot.get("image_pairs") or []
    locations = snapshot.get("locations") or []
    vlm_assessments = snapshot.get("vlm_assessments") or []

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.execute(
            text(
                """
                TRUNCATE TABLE
                    chat.vlm_assessments,
                    locations,
                    image_pairs,
                    disasters
                RESTART IDENTITY CASCADE
                """
            )
        )

        if disasters:
            conn.execute(
                text("INSERT INTO disasters (id, type) VALUES (:id, :type)"),
                disasters,
            )

        if image_pairs:
            conn.execute(
                text(
                    """
                    INSERT INTO image_pairs (
                        id, disaster_id, pair_id, pre_path, post_path,
                        pre_image_id, post_image_id,
                        pre_min_lat, pre_min_lng, pre_max_lat, pre_max_lng,
                        post_min_lat, post_min_lng, post_max_lat, post_max_lng
                    )
                    VALUES (
                        :id, :disaster_id, :pair_id, :pre_path, :post_path,
                        :pre_image_id, :post_image_id,
                        :pre_min_lat, :pre_min_lng, :pre_max_lat, :pre_max_lng,
                        :post_min_lat, :post_min_lng, :post_max_lat, :post_max_lng
                    )
                    """
                ),
                image_pairs,
            )

        if locations:
            conn.execute(
                text(
                    """
                    INSERT INTO locations (
                        id,
                        location_uid,
                        image_pair_id,
                        feature_type,
                        classification,
                        street,
                        city,
                        county,
                        full_address,
                        address_source,
                        address_fetched_at,
                        geom,
                        centroid
                    )
                    VALUES (
                        :id,
                        :location_uid,
                        :image_pair_id,
                        :feature_type,
                        :classification,
                        :street,
                        :city,
                        :county,
                        :full_address,
                        :address_source,
                        :address_fetched_at,
                        ST_GeomFromText(:geom_wkt, 4326),
                        ST_GeomFromText(:centroid_wkt, 4326)
                    )
                    """
                ),
                locations,
            )
            conn.execute(
                text(
                    """
                    SELECT setval(
                        pg_get_serial_sequence('locations', 'id'),
                        COALESCE((SELECT MAX(id) FROM locations), 1),
                        true
                    )
                    """
                )
            )

        if vlm_assessments:
            conn.execute(
                text(
                    """
                    INSERT INTO chat.vlm_assessments (
                        id,
                        location_id,
                        damage_level,
                        confidence,
                        description,
                        created_at
                    )
                    VALUES (
                        :id,
                        :location_id,
                        :damage_level,
                        :confidence,
                        :description,
                        :created_at
                    )
                    """
                ),
                vlm_assessments,
            )
            conn.execute(
                text(
                    """
                    SELECT setval(
                        pg_get_serial_sequence('chat.vlm_assessments', 'id'),
                        COALESCE((SELECT MAX(id) FROM chat.vlm_assessments), 1),
                        true
                    )
                    """
                )
            )

    print(
        "imported snapshot: "
        f"disasters={len(disasters)} image_pairs={len(image_pairs)} "
        f"locations={len(locations)} vlm_assessments={len(vlm_assessments)} "
        f"from {input_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
