from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import insert, text

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db import disasters, get_engine, image_pairs, metadata


def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_valid_float(value: Any) -> float | None:
    parsed = _to_float(value)
    if parsed is None or not math.isfinite(parsed):
        return None
    return parsed


def build_polygon_wkt(location: dict[str, Any]) -> str | None:
    points_container = location.get("points", {})
    raw_points = points_container.get("post") or points_container.get("pre") or []

    coords: list[tuple[float, float]] = []
    for point in raw_points:
        lng = _to_float(point.get("long"))
        lat = _to_float(point.get("lat"))

        if lng is None or lat is None:
            continue

        coords.append((lng, lat))

    if len(coords) < 3:
        return None

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    if len(set(coords)) < 3:
        return None

    coord_text = ", ".join(f"{lng} {lat}" for lng, lat in coords)
    return f"POLYGON(({coord_text}))"


def solve_linear_3x3(matrix: list[list[float]], values: list[float]) -> list[float] | None:
    augmented = [row[:] + [values[idx]] for idx, row in enumerate(matrix)]

    for pivot in range(3):
        pivot_row = pivot
        for row in range(pivot + 1, 3):
            if abs(augmented[row][pivot]) > abs(augmented[pivot_row][pivot]):
                pivot_row = row

        if abs(augmented[pivot_row][pivot]) < 1e-12:
            return None

        if pivot_row != pivot:
            augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot]

        pivot_value = augmented[pivot][pivot]
        for col in range(pivot, 4):
            augmented[pivot][col] /= pivot_value

        for row in range(3):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            for col in range(pivot, 4):
                augmented[row][col] -= factor * augmented[pivot][col]

    return [augmented[0][3], augmented[1][3], augmented[2][3]]


def solve_affine_coefficients(
    points: list[tuple[float, float, float, float]], target: str
) -> tuple[float, float, float] | None:
    if len(points) < 3:
        return None

    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    sum_x = 0.0
    sum_y = 0.0
    sum_xt = 0.0
    sum_yt = 0.0
    sum_t = 0.0

    for x, y, lng, lat in points:
        t = lng if target == "lng" else lat
        sum_x2 += x * x
        sum_y2 += y * y
        sum_xy += x * y
        sum_x += x
        sum_y += y
        sum_xt += x * t
        sum_yt += y * t
        sum_t += t

    matrix = [
        [sum_x2, sum_xy, sum_x],
        [sum_xy, sum_y2, sum_y],
        [sum_x, sum_y, float(len(points))],
    ]
    values = [sum_xt, sum_yt, sum_t]
    solved = solve_linear_3x3(matrix, values)
    if not solved:
        return None
    return (solved[0], solved[1], solved[2])


def get_phase_size(image: dict[str, Any], phase: str) -> tuple[float, float]:
    default_size = 1024.0
    size = image.get("size", {}).get(phase, {})
    width = _to_valid_float(size.get("width")) or default_size
    height = _to_valid_float(size.get("height")) or default_size
    return width, height


def collect_phase_control_points(
    image: dict[str, Any], phase: str
) -> list[tuple[float, float, float, float]]:
    points: list[tuple[float, float, float, float]] = []

    for location in image.get("locations", []):
        for point in location.get("points", {}).get(phase, []):
            x = _to_valid_float(point.get("x"))
            y = _to_valid_float(point.get("y"))
            lng = _to_valid_float(point.get("long"))
            lat = _to_valid_float(point.get("lat"))

            if x is None or y is None or lng is None or lat is None:
                continue

            points.append((x, y, lng, lat))

    return points


def compute_phase_bounds(
    image: dict[str, Any], phase: str
) -> tuple[float, float, float, float] | None:
    control_points = collect_phase_control_points(image, phase)
    lng_coeffs = solve_affine_coefficients(control_points, "lng")
    lat_coeffs = solve_affine_coefficients(control_points, "lat")

    if lng_coeffs is None or lat_coeffs is None:
        return None

    width, height = get_phase_size(image, phase)
    corners_xy = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]

    corners_lng_lat: list[tuple[float, float]] = []
    for x, y in corners_xy:
        lng = lng_coeffs[0] * x + lng_coeffs[1] * y + lng_coeffs[2]
        lat = lat_coeffs[0] * x + lat_coeffs[1] * y + lat_coeffs[2]
        if not math.isfinite(lng) or not math.isfinite(lat):
            continue
        corners_lng_lat.append((lng, lat))

    if not corners_lng_lat:
        return None

    lng_values = [c[0] for c in corners_lng_lat]
    lat_values = [c[1] for c in corners_lng_lat]
    return (min(lat_values), min(lng_values), max(lat_values), max(lng_values))


def ensure_schema(conn: Any) -> None:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    metadata.drop_all(conn, checkfirst=True)
    metadata.create_all(conn)


def run_load_step(
    parsed_json_path: str | Path,
    db_url: str | None = None,
) -> dict[str, int]:
    input_path = Path(parsed_json_path).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    engine = get_engine(db_url)

    disaster_rows: list[dict[str, Any]] = []
    image_pair_rows: list[dict[str, Any]] = []
    location_rows: list[dict[str, Any]] = []

    for disaster_id, disaster_payload in parsed.items():
        disaster_rows.append({"id": disaster_id, "type": disaster_payload.get("type")})

        for image in disaster_payload.get("images", []):
            image_pair_id = str(image.get("uid") or f"{disaster_id}-{image.get('pairId')}")
            pre_bounds = compute_phase_bounds(image, "pre")
            post_bounds = compute_phase_bounds(image, "post")
            image_pair_rows.append(
                {
                    "id": image_pair_id,
                    "disaster_id": disaster_id,
                    "pair_id": str(image.get("pairId", "")),
                    "pre_path": str(image.get("path", {}).get("pre", "")),
                    "post_path": str(image.get("path", {}).get("post", "")),
                    "pre_image_id": image.get("id", {}).get("pre"),
                    "post_image_id": image.get("id", {}).get("post"),
                    "pre_min_lat": pre_bounds[0] if pre_bounds else None,
                    "pre_min_lng": pre_bounds[1] if pre_bounds else None,
                    "pre_max_lat": pre_bounds[2] if pre_bounds else None,
                    "pre_max_lng": pre_bounds[3] if pre_bounds else None,
                    "post_min_lat": post_bounds[0] if post_bounds else None,
                    "post_min_lng": post_bounds[1] if post_bounds else None,
                    "post_max_lat": post_bounds[2] if post_bounds else None,
                    "post_max_lng": post_bounds[3] if post_bounds else None,
                }
            )

            for location in image.get("locations", []):
                wkt = build_polygon_wkt(location)
                if not wkt:
                    continue

                location_rows.append(
                    {
                        "location_uid": str(location.get("uid", "")),
                        "image_pair_id": image_pair_id,
                        "feature_type": location.get("type"),
                        "classification": location.get("classification"),
                        "prediction": location.get("prediction"),
                        "wkt": wkt,
                    }
                )

    with engine.begin() as conn:
        ensure_schema(conn)

        conn.execute(
            text("TRUNCATE TABLE locations, image_pairs, disasters RESTART IDENTITY CASCADE")
        )

        if disaster_rows:
            conn.execute(insert(disasters), disaster_rows)

        if image_pair_rows:
            conn.execute(insert(image_pairs), image_pair_rows)

        if location_rows:
            conn.execute(
                text(
                    """
                    INSERT INTO locations (
                        location_uid,
                        image_pair_id,
                        feature_type,
                        classification,
                        prediction,
                        geom,
                        centroid
                    )
                    VALUES (
                        :location_uid,
                        :image_pair_id,
                        :feature_type,
                        :classification,
                        :prediction,
                        ST_GeomFromText(:wkt, 4326),
                        ST_Centroid(ST_GeomFromText(:wkt, 4326))
                    )
                    """
                ),
                location_rows,
            )

    return {
        "disasters": len(disaster_rows),
        "image_pairs": len(image_pair_rows),
        "locations": len(location_rows),
    }
