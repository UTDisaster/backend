#!/usr/bin/env python3
from __future__ import annotations

import argparse

from preprocessing.parse_step import run_parse_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse raw source folder into normalized parsed_data.json + organized image files."
    )
    parser.add_argument(
        "data_folder",
        help="Path to folder containing images/, labels/, and targets/ subfolders",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output folder for organized images and parsed_data.json (default: data)",
    )
    parser.add_argument(
        "--disaster-id",
        default="hurricane-florence",
        help="Disaster id to parse from labels (default: hurricane-florence)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_parse_step(
        data_folder=args.data_folder,
        output=args.output,
        disaster_id=args.disaster_id,
    )

    print(f"Processed {result['processed_pairs']} pre/post pair(s).")
    print(f"Wrote parsed JSON: {result['output_json']}")
    print(f"Wrote organized images under: {result['output_images_dir']}")


if __name__ == "__main__":
    main()
