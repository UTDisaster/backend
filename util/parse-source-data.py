#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

STEM_PATTERN = re.compile(
    r"^(?P<disaster>.+)_(?P<pair_id>\d+)_(?P<phase>pre|post)_disaster$"
)
WKT_POINT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a dataset folder containing images/, labels/, and targets/."
    )

    parser.add_argument(
        "data_folder",
        help="Path to folder containing images/, labels/, and targets/ subfolders",
    )

    parser.add_argument(
        "--output",
        default="data",
        help="Output folder for organized images and data.json (default: data/)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all complete pre/post pairs instead of just one pair",
    )

    return parser.parse_args()


def parse_wkt_points(wkt: str) -> list[list[float]]:
    numbers = [float(x) for x in WKT_POINT_PATTERN.findall(wkt)]
    points: list[list[float]] = []

    for i in range(0, len(numbers) - 1, 2):
        points.append([numbers[i], numbers[i + 1]])

    return points


def load_label_entries(labels_dir: Path) -> dict[tuple[str, str], dict[str, Path]]:
    pairs: dict[tuple[str, str], dict[str, Path]] = {}

    for label_path in sorted(labels_dir.glob("*.json")):
        stem_match = STEM_PATTERN.match(label_path.stem)

        if not stem_match:
            continue

        disaster = stem_match.group("disaster")
        pair_id = stem_match.group("pair_id")
        phase = stem_match.group("phase")
        key = (disaster, pair_id)
        pairs.setdefault(key, {})[phase] = label_path

    return pairs


def get_feature_uid(feature: dict[str, Any], fallback_index: int) -> str:
    properties = feature.get("properties", {})
    uid = (
        properties.get("uid")
        or properties.get("object_id")
        or properties.get("id")
        or feature.get("id")
    )

    if uid is None:
        return f"idx-{fallback_index:06d}"

    return str(uid)


def index_phase_locations(label_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    features = label_json.get("features", {}).get("xy", [])

    for idx, feature in enumerate(features):
        properties = feature.get("properties", {})
        uid = get_feature_uid(feature, idx)
        indexed[uid] = {
            "uid": uid,
            "type": properties.get("feature_type"),
            "classification": properties.get("subtype"),
            "points": parse_wkt_points(feature.get("wkt", "")),
        }

    return indexed


def merge_locations(
    pre_label_json: dict[str, Any], post_label_json: dict[str, Any]
) -> list[dict[str, Any]]:
    pre_locations = index_phase_locations(pre_label_json)
    post_locations = index_phase_locations(post_label_json)
    all_uids = sorted(set(pre_locations.keys()) | set(post_locations.keys()))

    merged: list[dict[str, Any]] = []
    for uid in all_uids:
        pre_loc = pre_locations.get(uid, {})
        post_loc = post_locations.get(uid, {})

        merged.append(
            {
                "uid": uid,
                "type": post_loc.get("type") or pre_loc.get("type"),
                "classification": post_loc.get("classification")
                or pre_loc.get("classification"),
                "prediction": None,
                "points": {
                    "pre": pre_loc.get("points", []),
                    "post": post_loc.get("points", []),
                },
            }
        )

    return merged


def build_image_payload(
    pre_label_json: dict[str, Any],
    post_label_json: dict[str, Any],
    pre_image_relative_path: Path,
    post_image_relative_path: Path,
) -> dict[str, Any]:
    pre_metadata = pre_label_json.get("metadata", {})
    post_metadata = post_label_json.get("metadata", {})

    return {
        "path": {
            "pre": str(pre_image_relative_path.as_posix()),
            "post": str(post_image_relative_path.as_posix()),
        },
        "locations": merge_locations(pre_label_json, post_label_json),
        "height": {
            "pre": pre_metadata.get("height"),
            "post": post_metadata.get("height"),
        },
        "width": {
            "pre": pre_metadata.get("width"),
            "post": post_metadata.get("width"),
        },
        "id": {
            "pre": pre_metadata.get("id"),
            "post": post_metadata.get("id"),
        },
    }


def main() -> None:
    args = parse_args()
    data_folder = Path(args.data_folder).expanduser().resolve()
    images_dir = data_folder / "images"
    labels_dir = data_folder / "labels"
    targets_dir = data_folder / "targets"

    for required in (images_dir, labels_dir, targets_dir):
        if not required.is_dir():
            raise FileNotFoundError(f"Missing required subfolder: {required}")

    output_dir = Path(args.output).expanduser().resolve()
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_label_entries(labels_dir)
    if not pairs:
        raise RuntimeError(f"No parseable label pairs found in {labels_dir}")

    processed = 0
    result: dict[str, dict[str, Any]] = {}
    ordered_pair_keys = sorted(pairs.keys())
    max_pairs = len(ordered_pair_keys) if args.all else 1

    for disaster, pair_id in ordered_pair_keys:
        phase_paths = pairs[(disaster, pair_id)]

        if "pre" not in phase_paths or "post" not in phase_paths:
            continue

        with phase_paths["pre"].open("r", encoding="utf-8") as f:
            pre_label_json = json.load(f)
        with phase_paths["post"].open("r", encoding="utf-8") as f:
            post_label_json = json.load(f)

        pre_image_name = pre_label_json.get("metadata", {}).get("img_name")
        post_image_name = post_label_json.get("metadata", {}).get("img_name")

        if not pre_image_name or not post_image_name:
            continue

        pre_image_src = images_dir / pre_image_name
        post_image_src = images_dir / post_image_name

        if not pre_image_src.exists() or not post_image_src.exists():
            continue

        disaster_output_dir = output_images_dir / disaster
        disaster_output_dir.mkdir(parents=True, exist_ok=True)

        pre_image_dst = disaster_output_dir / pre_image_src.name
        post_image_dst = disaster_output_dir / post_image_src.name
        shutil.copy2(pre_image_src, pre_image_dst)
        shutil.copy2(post_image_src, post_image_dst)

        disaster_entry = result.setdefault(
            disaster,
            {
                "disasterId": disaster,
                "images": [],
                "type": pre_label_json.get("metadata", {}).get("disaster_type"),
            },
        )
        disaster_entry["images"].append(
            {
                "pairId": pair_id,
                "uid": f"{disaster}-{pair_id}",
                **build_image_payload(
                    pre_label_json,
                    post_label_json,
                    pre_image_dst.relative_to(output_dir),
                    post_image_dst.relative_to(output_dir),
                ),
            }
        )

        processed += 1
        if processed >= max_pairs:
            break

    if processed == 0:
        raise RuntimeError(
            "No complete pre/post pairs could be processed. "
            "Check that both label JSON and image files exist."
        )

    output_json = output_dir / "parsed_data.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Processed {processed} pre/post pair(s).")
    print(f"Wrote parsed JSON: {output_json}")
    print(f"Wrote organized images under: {output_images_dir}")


if __name__ == "__main__":
    main()
