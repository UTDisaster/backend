import json
import os
import re
import sys

import ollama
from PIL import Image, ImageOps

ALLOWED_LABELS = {"no-damage", "minor-damage", "major-damage", "destroyed"}
FALLBACK_LABEL = "no-damage"
MODEL_NAME = "llava:latest"
DATASET_DIR = "/Users/jaimerobles/Desktop/Code/Datasets/hurricane-florence_00000131_post_disaster"
LABELS_PATH = os.path.join(DATASET_DIR, "labels.json")
ASSESSMENTS_DIR = "/Users/jaimerobles/Desktop/Code/Datasets/Assesment's"


def build_prompt():
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
        "Examples (text-only anchors):\n"
        '{"damage_score":0}  # intact roof, no debris, no visible change\n'
        '{"damage_score":1}  # small roof puncture or minor debris nearby\n'
        '{"damage_score":2}  # large roof loss, partial wall collapse\n'
        '{"damage_score":3}  # structure largely gone or flattened\n\n'
        "Only choose 0 when the post image looks visually unchanged.\n"
        "If unsure between 0 and 1, choose 1.\n\n"
        "Return ONLY a JSON object with a single key in the message content.\n"
        "Do NOT use thinking or extra narration.\n"
        '{"damage_score":0|1|2|3}\n'
        "No extra keys. No prose."
    )


def strict_prompt():
    return (
        "Return ONLY a JSON object in the message content like "
        '{"damage_score":0|1|2|3}. '
        "Do NOT use thinking or extra narration."
    )


def score_to_label(score):
    return {
        0: "no-damage",
        1: "minor-damage",
        2: "major-damage",
        3: "destroyed",
    }.get(score)


def parse_damage_score(text):
    if not text:
        return None
    match = re.search(r'"damage_score"\s*:\s*([0-3])', text)
    if match:
        score = int(match.group(1))
        if score in (0, 1, 2, 3):
            return score
    match = re.search(r'\b([0-3])\b', text)
    if match:
        return int(match.group(1))
    return None


def extract_json_object(text):
    if not text:
        return None
    match = re.search(r'\{[^{}]*"damage_score"\s*:\s*[0-3][^{}]*\}', text, re.S)
    if match:
        return match.group(0)
    return None


def parse_damage_assessment(text):
    if not text:
        return None
    match = re.search(r'"damage_assessment"\s*:\s*"([a-z-]+)"', text)
    if match:
        label = match.group(1)
        return label if label in ALLOWED_LABELS else None
    match = re.search(r'(no-damage|minor-damage|major-damage|destroyed)', text)
    if match:
        return match.group(1)
    lowered = text.lower()
    if any(word in lowered for word in ("destroyed", "leveled", "collapsed", "total loss", "total destruction")):
        return "destroyed"
    if any(word in lowered for word in ("major", "severe", "heavy damage", "extensive", "significant")):
        return "major-damage"
    if any(word in lowered for word in ("minor", "light damage", "moderate", "moderate damage")):
        return "minor-damage"
    if any(word in lowered for word in ("no damage", "no-damage", "intact", "undamaged")):
        return "no-damage"
    return None


def compare_images_ollama(pre_image_path, post_image_path, target_image_path=None, debug=False):
    # Always use the pre/post pair (cropped upstream); add target if present.
    images = [pre_image_path, post_image_path]
    if target_image_path and os.path.exists(target_image_path):
        images.append(target_image_path)
    raw_primary = ""
    raw_retry = ""
    error_msg = ""
    used_field = "content"
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(),
                    "images": images,
                }
            ],
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
        used_field = "content" if content else "thinking"
        text = content if content else thinking
        if debug and not text:
            raw_primary = f"EMPTY_MESSAGE response={response!r}"[:1000]
        else:
            raw_primary = text
        if debug:
            print("RAW VLM RESPONSE:", text[:1000], file=sys.stderr)
        text = extract_json_object(text) or text
        score = parse_damage_score(text)
        if score is not None:
            label = score_to_label(score)
            if label and score != 0:
                return (label, raw_primary, raw_retry, error_msg, used_field) if debug else label

        retry = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": strict_prompt(),
                    "images": images,
                }
            ],
            format="json",
            options={
                "temperature": 0.6,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 192,
            },
            stream=False,
        )
        message = retry.get("message", {})
        content = message.get("content", "") or ""
        thinking = message.get("thinking", "") or ""
        used_field_retry = "content" if content else "thinking"
        text = content if content else thinking
        if debug and not text:
            raw_retry = f"EMPTY_MESSAGE response={retry!r}"[:1000]
        else:
            raw_retry = text
        if debug:
            print("RAW VLM RETRY:", text[:1000], file=sys.stderr)
        text = extract_json_object(text) or text
        score = parse_damage_score(text)
        if score is not None:
            label = score_to_label(score)
            if label:
                return (label, raw_primary, raw_retry, error_msg, used_field_retry) if debug else label
        label = parse_damage_assessment(text)
        final = label if label else FALLBACK_LABEL
        return (final, raw_primary, raw_retry, error_msg, used_field_retry) if debug else final
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        return (FALLBACK_LABEL, raw_primary, raw_retry, error_msg, "error") if debug else FALLBACK_LABEL


def load_house_metadata(labels_path):
    with open(labels_path, "r") as f:
        data = json.load(f)
    metadata = {}
    for house_id, house_data in data["houses"].items():
        metadata[house_id] = {
            "uid": house_data["uid"],
            "crop_bounds": house_data.get("crop_bounds"),
        }
    return metadata


def find_assessment_json(assessments_dir, dataset_dir):
    dataset_name = os.path.basename(dataset_dir)
    matches = []
    for name in os.listdir(assessments_dir):
        if dataset_name in name and name.endswith(".json"):
            matches.append(os.path.join(assessments_dir, name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(f"No assessment JSON found for {dataset_name} in {assessments_dir}")
    raise FileExistsError(f"Multiple assessment JSON files found for {dataset_name}: {matches}")


def load_actual_assessments(path):
    with open(path, "r") as f:
        data = json.load(f)
    features = data.get("features", {})
    for section in ("lng_lat", "xy"):
        if section in features:
            items = features[section]
            break
    else:
        items = []
    actuals = {}
    for item in items:
        props = item.get("properties", {})
        uid = props.get("uid")
        subtype = props.get("subtype")
        if uid and subtype:
            actuals[uid] = subtype
    return actuals


def iter_house_dirs(dataset_dir):
    for name in sorted(os.listdir(dataset_dir)):
        if not name.startswith("house_"):
            continue
        full_path = os.path.join(dataset_dir, name)
        if os.path.isdir(full_path):
            yield name, full_path


def clamp_crop_bounds(bounds, width, height):
    x1 = max(0, min(width, int(bounds["x1"])))
    y1 = max(0, min(height, int(bounds["y1"])))
    x2 = max(0, min(width, int(bounds["x2"])))
    y2 = max(0, min(height, int(bounds["y2"])))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def crop_image_to_temp(src_path, bounds, out_path):
    with Image.open(src_path) as img:
        crop = clamp_crop_bounds(bounds, img.width, img.height)
        if not crop:
            return None
        img.crop(crop).save(out_path)
    return out_path


def preprocess_image(src_path, out_path, max_side=512):
    with Image.open(src_path) as img:
        img = ImageOps.autocontrast(img)
        width, height = img.size
        scale = min(1.0, max_side / max(width, height))
        if scale != 1.0:
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        img.save(out_path)
    return out_path


def main(debug=False):
    house_meta = load_house_metadata(LABELS_PATH)
    assessment_path = find_assessment_json(ASSESSMENTS_DIR, DATASET_DIR)
    actuals = load_actual_assessments(assessment_path)
    results = []
    temp_dir = "/tmp/hurricane_flr_crops"
    os.makedirs(temp_dir, exist_ok=True)

    for house_id, house_dir in iter_house_dirs(DATASET_DIR):
        pre_path = os.path.join(house_dir, "pre.png")
        post_path = os.path.join(house_dir, "post.png")
        if not (os.path.exists(pre_path) and os.path.exists(post_path)):
            continue

        meta = house_meta.get(house_id, {})
        bounds = meta.get("crop_bounds")
        if bounds:
            pre_crop = os.path.join(temp_dir, f"{house_id}_pre.png")
            post_crop = os.path.join(temp_dir, f"{house_id}_post.png")
            try:
                pre_cropped = crop_image_to_temp(pre_path, bounds, pre_crop)
                post_cropped = crop_image_to_temp(post_path, bounds, post_crop)
                if pre_cropped and post_cropped:
                    pre_path = pre_cropped
                    post_path = post_cropped
            except Exception:
                pass

        target_path = os.path.join(house_dir, "target.png")
        pre_norm = os.path.join(temp_dir, f"{house_id}_pre_norm.png")
        post_norm = os.path.join(temp_dir, f"{house_id}_post_norm.png")
        target_norm = os.path.join(temp_dir, f"{house_id}_target_norm.png")
        try:
            pre_path = preprocess_image(pre_path, pre_norm)
            post_path = preprocess_image(post_path, post_norm)
            if os.path.exists(target_path):
                target_path = preprocess_image(target_path, target_norm)
        except Exception:
            pass
        raw_primary = None
        raw_retry = None
        error_msg = None
        used_field = None
        damage = compare_images_ollama(pre_path, post_path, target_path, debug=debug)
        if debug and isinstance(damage, tuple):
            damage, raw_primary, raw_retry, error_msg, used_field = damage
        uid = meta.get("uid")
        actual = actuals.get(uid)
        match = actual is not None and str(actual).lower() == str(damage).lower()
        row = {
            "uid": uid,
            "damage_assessment": damage,
            "actual_assessment": actual,
            "match": match,
        }
        if debug:
            row["raw_vlm"] = raw_primary
            row["raw_vlm_retry"] = raw_retry
            row["error"] = error_msg
            row["used_field"] = used_field
        results.append(row)

    return results


def parse_and_validate_json(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON output: {exc}"

    if not isinstance(data, list):
        return None, "Invalid JSON output: expected a list"

    for item in data:
        if not isinstance(item, dict):
            return None, "Invalid JSON output: expected objects in list"
        if "uid" not in item or "damage_assessment" not in item:
            return None, "Invalid JSON output: missing uid or damage_assessment"

    return data, None


def print_table(rows):
    uid_width = max(len("uid"), *(len(str(r["uid"])) for r in rows)) if rows else len("uid")
    dmg_width = max(
        len("damage_assessment"),
        *(len(str(r["damage_assessment"])) for r in rows),
    ) if rows else len("damage_assessment")
    actual_width = max(
        len("actual_assessment"),
        *(len(str(r.get("actual_assessment"))) for r in rows),
    ) if rows else len("actual_assessment")
    match_width = max(
        len("match"),
        *(len(str(r.get("match"))) for r in rows),
    ) if rows else len("match")

    header = (
        f"{'uid'.ljust(uid_width)}  "
        f"{'damage_assessment'.ljust(dmg_width)}  "
        f"{'actual_assessment'.ljust(actual_width)}  "
        f"{'match'.ljust(match_width)}"
    )
    print(header)
    print(
        f"{'-' * uid_width}  "
        f"{'-' * dmg_width}  "
        f"{'-' * actual_width}  "
        f"{'-' * match_width}"
    )
    for row in rows:
        print(
            f"{str(row['uid']).ljust(uid_width)}  "
            f"{str(row['damage_assessment']).ljust(dmg_width)}  "
            f"{str(row.get('actual_assessment')).ljust(actual_width)}  "
            f"{str(row.get('match')).ljust(match_width)}"
        )


if __name__ == "__main__":
    debug = "--debug" in sys.argv
    predictions = main(debug=debug)
    raw_json = json.dumps(predictions, indent=2)
    parsed, error = parse_and_validate_json(raw_json)
    if error:
        print(error, file=sys.stderr)
        sys.exit(1)

    if "--table" in sys.argv:
        print_table(parsed)
    else:
        print(raw_json)
