#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline steps: parse source data, VLM prediction, and load PostGIS."
    )
    parser.add_argument(
        "data_folder",
        nargs="?",
        help="Path to raw source folder containing images/, labels/, targets/ (required when parsing).",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output folder for parsed JSON/images when parse step runs (default: data)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to parsed_data.json for vlm/load steps (defaults to <output>/parsed_data.json)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Optional DATABASE_URL override for load step",
    )
    parser.add_argument(
        "--disaster-id",
        default="hurricane-florence",
        help="Disaster id to parse from labels (default: hurricane-florence)",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Optional Ollama VLM model override for vlm step (defaults to preprocessing.vlm.generate_vlm MODEL_NAME)",
    )
    parser.add_argument(
        "--vlm-debug",
        action="store_true",
        help="Enable debug mode for vlm model responses",
    )
    parser.add_argument(
        "--vlm-print",
        choices=["none", "json", "table"],
        default="none",
        help="Print VLM prediction rows to CLI (default: none)",
    )
    parser.add_argument(
        "--vlm-no-write",
        action="store_true",
        help="Run VLM predictions without writing updated predictions to parsed_data.json",
    )
    parser.add_argument(
        "--vlm-max-locations",
        type=int,
        default=None,
        help="Maximum total locations to predict in VLM step",
    )
    parser.add_argument(
        "--vlm-max-locations-per-image",
        type=int,
        default=None,
        help="Maximum locations to predict per image pair in VLM step",
    )
    parser.add_argument(
        "--vlm-randomize",
        action="store_true",
        help="Randomize disaster/image/location iteration order in VLM step",
    )
    parser.add_argument(
        "--vlm-random-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible randomized VLM selection",
    )
    parser.add_argument(
        "--start-at",
        choices=["parse", "vlm", "load"],
        default="parse",
        help="First step to run (default: parse)",
    )
    parser.add_argument(
        "--stop-after",
        choices=["parse", "vlm", "load"],
        default="load",
        help="Last step to run (default: load)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    order = ["parse", "vlm", "load"]
    start_idx = order.index(args.start_at)
    stop_idx = order.index(args.stop_after)
    if start_idx > stop_idx:
        raise ValueError("Invalid step range: --start-at cannot be after --stop-after")

    run_parse = start_idx <= 0 <= stop_idx
    run_vlm = start_idx <= 1 <= stop_idx
    run_load = start_idx <= 2 <= stop_idx

    if run_parse and not args.data_folder:
        raise ValueError("data_folder is required when parse step is included")

    parsed_json_path = (
        Path(args.input) if args.input else Path(args.output) / "parsed_data.json"
    )

    if run_parse:
        from preprocessing.parse_step import run_parse_step

        parse_result = run_parse_step(
            data_folder=args.data_folder,
            output=args.output,
            disaster_id=args.disaster_id,
        )
        parsed_json_path = Path(parse_result["output_json"])
        print(f"Parsed pairs: {parse_result['processed_pairs']}")
        print(f"Parsed JSON: {parse_result['output_json']}")
        print(f"Images directory: {parse_result['output_images_dir']}")

    if run_vlm:
        from preprocessing.vlm import print_prediction_table, run_vlm_step

        vlm_data_root = Path(args.output) if run_parse else parsed_json_path.parent
        vlm_result = run_vlm_step(
            parsed_json_path=parsed_json_path,
            data_root=vlm_data_root,
            output_json_path=parsed_json_path,
            model_name=args.vlm_model,
            debug=args.vlm_debug,
            max_locations=args.vlm_max_locations,
            max_locations_per_image=args.vlm_max_locations_per_image,
            write_output_json=not args.vlm_no_write,
            randomize=args.vlm_randomize,
            random_seed=args.vlm_random_seed,
        )
        parsed_json_path = Path(vlm_result["output_json"])
        print(f"VLM image pairs: {vlm_result['image_pairs']}")
        print(f"VLM predicted locations: {vlm_result['predicted_locations']}")
        print(f"VLM skipped locations: {vlm_result['skipped_locations']}")
        print(f"VLM wrote JSON: {vlm_result['json_written']}")
        print(f"VLM output JSON: {vlm_result['output_json']}")
        print(f"VLM randomized: {vlm_result['randomized']}")
        if vlm_result["random_seed_used"] is not None:
            print(f"VLM random seed: {vlm_result['random_seed_used']}")

        if args.vlm_print == "json":
            print(json.dumps(vlm_result["predictions"], indent=2))
        elif args.vlm_print == "table":
            print_prediction_table(vlm_result["predictions"])

    if run_load:
        from preprocessing.load_step import run_load_step

        load_result = run_load_step(
            parsed_json_path=parsed_json_path, db_url=args.db_url
        )
        print(f"Loaded disasters: {load_result['disasters']}")
        print(f"Loaded image pairs: {load_result['image_pairs']}")
        print(f"Loaded locations: {load_result['locations']}")


if __name__ == "__main__":
    main()
