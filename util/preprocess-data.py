#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

STEP_ORDER = ["parse", "vlm", "load"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline: parse -> vlm -> load."
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
        help="Path to parsed_data.json for load step (defaults to <output>/parsed_data.json)",
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
        "--start-at",
        choices=STEP_ORDER,
        default="parse",
        help=f"First step to run (default: parse). Steps: {STEP_ORDER}",
    )
    parser.add_argument(
        "--stop-after",
        choices=STEP_ORDER,
        default="load",
        help=f"Last step to run (default: load). Steps: {STEP_ORDER}",
    )
    parser.add_argument(
        "--vlm-limit",
        type=int,
        default=None,
        help="Cap the VLM step to N locations (default: all).",
    )
    parser.add_argument(
        "--vlm-stratified",
        type=int,
        default=None,
        help="Stratified sample of N locations per damage class (random per class).",
    )
    parser.add_argument(
        "--vlm-concurrency",
        type=int,
        default=5,
        help="Parallel workers classifying in parallel (default: 5).",
    )
    parser.add_argument(
        "--vlm-min-crop",
        type=int,
        default=320,
        help="Minimum crop side in pixels (default: 320).",
    )
    parser.add_argument(
        "--vlm-max-crop",
        type=int,
        default=640,
        help="Maximum crop side in pixels (default: 640).",
    )
    parser.add_argument(
        "--vlm-padding",
        type=float,
        default=0.4,
        help="Fractional padding around polygon before clamping (default: 0.4).",
    )
    parser.add_argument(
        "--vlm-rps",
        type=float,
        default=5.0,
        help="Rate limit for Gemini calls in requests/second (default: 5.0).",
    )
    parser.add_argument(
        "--vlm-prompt-version",
        choices=["v1", "v2", "v3", "v4"],
        default="v2",
        help="Prompt template version (default: v2 fewshot). v4=flood-specific + polygon overlay.",
    )
    parser.add_argument(
        "--vlm-model",
        default="gemini-2.5-flash",
        help="Gemini model id (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--vlm-reclassify",
        action="store_true",
        help="Re-classify locations that already have assessments (default: skip).",
    )
    parser.add_argument(
        "--vlm-log",
        default="data/vlm_log.jsonl",
        help="Path to JSONL log of per-call results (default: data/vlm_log.jsonl).",
    )
    parser.add_argument(
        "--vlm-max-errors",
        type=int,
        default=100,
        help="Abort the VLM step after this many consecutive errors (default: 100).",
    )
    parser.add_argument(
        "--vlm-no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_idx = STEP_ORDER.index(args.start_at)
    stop_idx = STEP_ORDER.index(args.stop_after)
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
        from preprocessing.vlm_step import run_vlm_step

        vlm_stats = run_vlm_step(
            disaster_id=args.disaster_id,
            limit=args.vlm_limit,
            stratified_per_class=args.vlm_stratified,
            skip_classified=not args.vlm_reclassify,
            rps=args.vlm_rps,
            concurrency=args.vlm_concurrency,
            prompt_version=args.vlm_prompt_version,
            model=args.vlm_model,
            min_crop_size=args.vlm_min_crop,
            max_crop_size=args.vlm_max_crop,
            padding_fraction=args.vlm_padding,
            max_errors=args.vlm_max_errors,
            log_path=Path(args.vlm_log) if args.vlm_log else None,
            progress=not args.vlm_no_progress,
        )
        print(f"VLM stats: {vlm_stats.to_dict()}")

    if run_load:
        from preprocessing.load_step import run_load_step

        load_result = run_load_step(parsed_json_path=parsed_json_path, db_url=args.db_url)
        print(f"Loaded disasters: {load_result['disasters']}")
        print(f"Loaded image pairs: {load_result['image_pairs']}")
        print(f"Loaded locations: {load_result['locations']}")


if __name__ == "__main__":
    main()
