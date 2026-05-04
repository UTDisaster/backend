from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.db import get_engine

DEFAULT_MAP_FILENAME = "address_map.json"

SELECT_SQL = """
SELECT
    location_uid,
    street,
    city,
    county,
    full_address,
    address_source,
    address_fetched_at
FROM locations
WHERE location_uid IS NOT NULL
  AND (
    street IS NOT NULL OR
    city IS NOT NULL OR
    county IS NOT NULL OR
    full_address IS NOT NULL
  )
"""


def _default_output_path() -> Path:
    parsed_data_dir = (os.getenv("PARSED_DATA_DIR", "") or "").strip()
    if parsed_data_dir:
        return Path(parsed_data_dir).expanduser() / DEFAULT_MAP_FILENAME
    return Path(DEFAULT_MAP_FILENAME)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export location_uid keyed address map from DB."
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "$PARSED_DATA_DIR/address_map.json when PARSED_DATA_DIR is set."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write human-readable JSON (indented).",
    )
    parser.add_argument(
        "--where-source",
        default=None,
        help="Optional filter for address_source (e.g. census, map).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = (
        Path(args.output).expanduser() if args.output else _default_output_path()
    )

    sql = SELECT_SQL
    params: dict[str, Any] = {}
    if args.where_source:
        sql += " AND address_source = :address_source"
        params["address_source"] = args.where_source.strip()

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()

    exported: dict[str, dict[str, Any]] = {}
    for row in rows:
        location_uid = row["location_uid"]
        if not isinstance(location_uid, str) or not location_uid.strip():
            continue
        fetched_at = row["address_fetched_at"]
        exported[location_uid] = {
            "street": row["street"],
            "city": row["city"],
            "county": row["county"],
            "full_address": row["full_address"],
            "source": row["address_source"],
            "fetched_at": fetched_at.isoformat() if fetched_at is not None else None,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(exported, f, indent=2, sort_keys=True)
            f.write("\n")
        else:
            json.dump(exported, f, separators=(",", ":"), sort_keys=True)

    print(f"wrote {len(exported)} entries to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
