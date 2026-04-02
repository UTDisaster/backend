from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_prediction(raw_label: str | None) -> str | None:
    if raw_label is None:
        return None

    value = str(raw_label).strip().lower()
    mapping = {
        "no-damage": "none",
        "minor-damage": "minor",
        "major-damage": "severe",
        "destroyed": "destroyed",
        "none": "none",
        "minor": "minor",
        "severe": "severe",
        "unknown": "unknown",
    }

    return mapping.get(value, value.replace("-", "_"))


def _resolve_image_path(data_root: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate

    return (data_root / candidate).resolve()


def _collect_xy_points(location: dict[str, Any]) -> list[tuple[float, float]]:
    points_by_phase = location.get("points", {})
    combined = []

    for phase in ("pre", "post"):
        for point in points_by_phase.get(phase, []):
            x = _to_float(point.get("x"))
            y = _to_float(point.get("y"))
            if x is None or y is None:
                continue
            combined.append((x, y))

    return combined


def _bounds_from_points(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    padding: int = 8,
) -> dict[str, int] | None:
    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(image_width, int(max(xs)) + padding)
    y2 = min(image_height, int(max(ys)) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def run_vlm_step(
    parsed_json_path: str | Path,
    data_root: str | Path | None = None,
    output_json_path: str | Path | None = None,
    model_name: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    input_path = Path(parsed_json_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    # Lazy import keeps parse/load steps runnable when ollama/Pillow are not installed.
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "VLM step requires Pillow. Install dependencies with: pip install Pillow"
        ) from exc

    try:
        from . import generate_vlm
    except ImportError as exc:
        raise RuntimeError(
            "VLM step requires preprocessing.vlm.generate_vlm dependencies. "
            "Install dependencies with: pip install ollama Pillow"
        ) from exc

    if model_name:
        generate_vlm.set_model_name(model_name)

    root_dir = (
        Path(data_root).expanduser().resolve() if data_root else input_path.parent
    )

    predicted_locations = 0
    skipped_locations = 0
    image_pairs = 0

    with tempfile.TemporaryDirectory(prefix="vlm-preprocess-") as temp_dir:
        temp_root = Path(temp_dir)

        for disaster_payload in parsed.values():
            for image in disaster_payload.get("images", []):
                pre_path = _resolve_image_path(
                    root_dir, image.get("path", {}).get("pre")
                )
                post_path = _resolve_image_path(
                    root_dir, image.get("path", {}).get("post")
                )
                if (
                    not pre_path
                    or not post_path
                    or not pre_path.is_file()
                    or not post_path.is_file()
                ):
                    skipped_locations += len(image.get("locations", []))
                    continue

                image_pairs += 1

                with Image.open(pre_path) as pre_img:
                    pre_size = (pre_img.width, pre_img.height)
                with Image.open(post_path) as post_img:
                    post_size = (post_img.width, post_img.height)

                for location in image.get("locations", []):
                    uid = str(location.get("uid", "unknown"))
                    points = _collect_xy_points(location)

                    pre_bounds = _bounds_from_points(points, pre_size[0], pre_size[1])
                    post_bounds = _bounds_from_points(
                        points, post_size[0], post_size[1]
                    )
                    if not pre_bounds or not post_bounds:
                        location["prediction"] = None
                        skipped_locations += 1
                        continue

                    pre_crop = temp_root / f"{uid}_pre_crop.png"
                    post_crop = temp_root / f"{uid}_post_crop.png"
                    pre_norm = temp_root / f"{uid}_pre_norm.png"
                    post_norm = temp_root / f"{uid}_post_norm.png"

                    pre_cropped = generate_vlm.crop_image_to_temp(
                        str(pre_path), pre_bounds, str(pre_crop)
                    )
                    post_cropped = generate_vlm.crop_image_to_temp(
                        str(post_path), post_bounds, str(post_crop)
                    )

                    if not pre_cropped or not post_cropped:
                        location["prediction"] = None
                        skipped_locations += 1
                        continue

                    pre_ready = generate_vlm.preprocess_image(
                        pre_cropped, str(pre_norm)
                    )
                    post_ready = generate_vlm.preprocess_image(
                        post_cropped, str(post_norm)
                    )

                    damage = generate_vlm.compare_images_ollama(
                        pre_ready,
                        post_ready,
                        target_image_path=None,
                        debug=debug,
                    )
                    location["prediction"] = _normalize_prediction(damage)
                    predicted_locations += 1

    output_path = (
        Path(output_json_path).expanduser().resolve()
        if output_json_path
        else input_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    return {
        "output_json": str(output_path),
        "image_pairs": image_pairs,
        "predicted_locations": predicted_locations,
        "skipped_locations": skipped_locations,
    }
