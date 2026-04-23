from __future__ import annotations

import json
import re

from app.services.vlm.errors import VLMParseError

CANONICAL_LABELS = {
    0: "no-damage",
    1: "minor-damage",
    2: "major-damage",
    3: "destroyed",
}

LABEL_TO_SCORE = {v: k for k, v in CANONICAL_LABELS.items()}

_LABEL_ALIASES = {
    "nodamage": "no-damage",
    "none": "no-damage",
    "no damage": "no-damage",
    "minor": "minor-damage",
    "minordamage": "minor-damage",
    "major": "major-damage",
    "majordamage": "major-damage",
    "severe": "major-damage",
    "destroyed": "destroyed",
    "total": "destroyed",
    "total loss": "destroyed",
}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_label(raw: str) -> str:
    key = raw.strip().lower().replace("_", "-")
    if key in LABEL_TO_SCORE:
        return key
    alt = _LABEL_ALIASES.get(key) or _LABEL_ALIASES.get(key.replace("-", " "))
    if alt:
        return alt
    raise VLMParseError(f"Unknown damage label: {raw!r}")


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if not text:
        raise VLMParseError("Empty model response")
    fenced = _FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(candidate)
        if not match:
            raise VLMParseError(f"No JSON object in response: {text[:200]!r}")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise VLMParseError(f"Malformed JSON: {exc}") from exc


def parse_response(raw_text: str) -> dict:
    """Returns a dict with keys: score (int 0-3), label (canonical), confidence (0-1 float), description (str).

    Raises VLMParseError on any violation.
    """
    data = _extract_json_object(raw_text)

    if "label" in data:
        label = _normalize_label(str(data["label"]))
    elif "score" in data:
        label = CANONICAL_LABELS.get(int(data["score"]))
        if label is None:
            raise VLMParseError(f"Score out of range: {data['score']!r}")
    else:
        raise VLMParseError(f"Response missing label and score: {data!r}")

    score = LABEL_TO_SCORE[label]
    if "score" in data and int(data["score"]) != score:
        score = int(data["score"])
        if score not in CANONICAL_LABELS:
            raise VLMParseError(f"Score out of range: {score}")
        label = CANONICAL_LABELS[score]

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError) as exc:
        raise VLMParseError(f"Bad confidence value: {data.get('confidence')!r}") from exc
    confidence = max(0.0, min(1.0, confidence))

    description = str(data.get("description", "")).strip()[:400]

    return {
        "score": score,
        "label": label,
        "confidence": confidence,
        "description": description,
    }
