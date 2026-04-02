from __future__ import annotations

import re
import sys
from pathlib import Path

import ollama
from PIL import Image, ImageOps

FALLBACK_LABEL = "no-damage"
MODEL_NAME = "llava:latest"


def set_model_name(model_name: str) -> None:
    global MODEL_NAME
    MODEL_NAME = model_name


def _build_prompt() -> str:
    return (
        "You are analyzing satellite images of the same location.\n\n"
        "Image order:\n"
        "1) Cropped PRE-disaster image.\n"
        "2) Cropped POST-disaster image.\n"
        "3) Target image (if present) for extra context.\n\n"
        "Compare PRE vs POST and assign a damage score using this rubric:\n"
        "0 = no visible change or only trivial change; structure intact.\n"
        "1 = minor damage; small localized damage, structure intact.\n"
        "2 = major damage; significant roof/wall loss but not total.\n"
        "3 = destroyed; structure largely gone or collapsed.\n\n"
        "Only choose 0 when the post image looks visually unchanged.\n"
        "If unsure between 0 and 1, choose 1.\n\n"
        "Return ONLY a JSON object with a single key in the message content.\n"
        "Do NOT use thinking or extra narration.\n"
        '{"damage_score":0|1|2|3}\n'
        "No extra keys. No prose."
    )


def _strict_prompt() -> str:
    return (
        "Return ONLY a JSON object in the message content like "
        '{"damage_score":0|1|2|3}. '
        "Do NOT use thinking or extra narration."
    )


def _score_to_label(score: int) -> str | None:
    return {
        0: "no-damage",
        1: "minor-damage",
        2: "major-damage",
        3: "destroyed",
    }.get(score)


def _parse_damage_score(text: str) -> int | None:
    match = re.search(r'"damage_score"\s*:\s*([0-3])', text)
    if match:
        return int(match.group(1))

    # Fallback: some models return a bare number.
    match = re.search(r'\b([0-3])\b', text)
    if match:
        return int(match.group(1))

    return None


def _extract_json_object(text: str) -> str:
    match = re.search(r'\{[^{}]*"damage_score"\s*:\s*[0-3][^{}]*\}', text, re.S)
    if match:
        return match.group(0)
    return text


def _invoke_ollama(images: list[str], prompt: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt, "images": images}],
        format="json",
        options={
            "temperature": 0.6,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 192,
        },
        stream=False,
    )

    message = response.get("message", {})
    content = message.get("content", "") or ""
    thinking = message.get("thinking", "") or ""
    return content if content else thinking


def compare_images_ollama(
    pre_image_path: str,
    post_image_path: str,
    target_image_path: str | None = None,
    debug: bool = False,
) -> str:
    images = [pre_image_path, post_image_path]
    if target_image_path and Path(target_image_path).is_file():
        images.append(target_image_path)

    try:
        primary_text = _invoke_ollama(images, _build_prompt())
        if debug:
            print(f"RAW VLM RESPONSE: {primary_text[:1000]}", file=sys.stderr)
        parsed_text = _extract_json_object(primary_text)

        score = _parse_damage_score(parsed_text)
        if score is not None:
            label = _score_to_label(score)
            if label and score != 0:
                return label

        retry_text = _invoke_ollama(images, _strict_prompt())
        if debug:
            print(f"RAW VLM RETRY: {retry_text[:1000]}", file=sys.stderr)
        parsed_retry = _extract_json_object(retry_text)

        retry_score = _parse_damage_score(parsed_retry)
        if retry_score is not None:
            retry_label = _score_to_label(retry_score)
            if retry_label:
                return retry_label

        return FALLBACK_LABEL
    except Exception as exc:
        if debug:
            print(f"VLM error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return FALLBACK_LABEL


def _clamp_crop_bounds(
    bounds: dict[str, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width, int(bounds["x1"])))
    y1 = max(0, min(height, int(bounds["y1"])))
    x2 = max(0, min(width, int(bounds["x2"])))
    y2 = max(0, min(height, int(bounds["y2"])))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def crop_image_to_temp(src_path: str, bounds: dict[str, int], out_path: str) -> str | None:
    with Image.open(src_path) as img:
        crop = _clamp_crop_bounds(bounds, img.width, img.height)
        if not crop:
            return None
        img.crop(crop).save(out_path)
    return out_path


def preprocess_image(src_path: str, out_path: str, max_side: int = 512) -> str:
    with Image.open(src_path) as img:
        img = ImageOps.autocontrast(img)
        width, height = img.size
        scale = min(1.0, max_side / max(width, height))
        if scale != 1.0:
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        img.save(out_path)
    return out_path
