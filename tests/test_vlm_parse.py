from __future__ import annotations

import pytest

from app.services.vlm.errors import VLMParseError
from app.services.vlm.parse import parse_response


def test_parse_canonical_json() -> None:
    raw = '{"score": 2, "label": "major-damage", "confidence": 0.85, "description": "roof collapsed"}'
    result = parse_response(raw)
    assert result["score"] == 2
    assert result["label"] == "major-damage"
    assert result["confidence"] == pytest.approx(0.85)
    assert result["description"] == "roof collapsed"


def test_parse_fenced_json() -> None:
    raw = '```json\n{"score": 0, "label": "no-damage", "confidence": 0.9, "description": "unchanged"}\n```'
    result = parse_response(raw)
    assert result["score"] == 0
    assert result["label"] == "no-damage"


def test_parse_json_with_trailing_commentary() -> None:
    raw = 'Here is my analysis: {"score": 3, "label": "destroyed", "confidence": 0.77, "description": "slab only"}. End.'
    result = parse_response(raw)
    assert result["score"] == 3
    assert result["label"] == "destroyed"


def test_parse_label_alias_normalization() -> None:
    raw = '{"score": 2, "label": "severe", "confidence": 0.6, "description": "x"}'
    result = parse_response(raw)
    assert result["label"] == "major-damage"


def test_parse_label_case_insensitive_and_underscore() -> None:
    raw = '{"score": 1, "label": "Minor_Damage", "confidence": 0.5, "description": "x"}'
    result = parse_response(raw)
    assert result["label"] == "minor-damage"


def test_parse_missing_label_derives_from_score() -> None:
    raw = '{"score": 1, "confidence": 0.6, "description": "x"}'
    result = parse_response(raw)
    assert result["score"] == 1
    assert result["label"] == "minor-damage"


def test_parse_score_label_conflict_trusts_score() -> None:
    raw = '{"score": 3, "label": "no-damage", "confidence": 0.9, "description": "x"}'
    result = parse_response(raw)
    assert result["score"] == 3
    assert result["label"] == "destroyed"


def test_parse_confidence_clamped() -> None:
    raw = '{"score": 0, "label": "no-damage", "confidence": 1.6, "description": "x"}'
    result = parse_response(raw)
    assert result["confidence"] == 1.0


def test_parse_description_truncated() -> None:
    long = "x" * 1000
    raw = f'{{"score": 0, "label": "no-damage", "confidence": 0.5, "description": "{long}"}}'
    result = parse_response(raw)
    assert len(result["description"]) <= 400


def test_parse_empty_response_raises() -> None:
    with pytest.raises(VLMParseError):
        parse_response("")


def test_parse_no_json_raises() -> None:
    with pytest.raises(VLMParseError):
        parse_response("I cannot answer")


def test_parse_score_out_of_range_raises() -> None:
    raw = '{"score": 7, "confidence": 0.1, "description": "x"}'
    with pytest.raises(VLMParseError):
        parse_response(raw)


def test_parse_unknown_label_raises() -> None:
    raw = '{"label": "catastrophic", "confidence": 0.5, "description": "x"}'
    with pytest.raises(VLMParseError):
        parse_response(raw)
