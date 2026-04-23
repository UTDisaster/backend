from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.vlm.classifier import (
    GeminiVLMClassifier,
    VLMResult,
    _classify_error,
)
from app.services.vlm.errors import (
    VLMFatalError,
    VLMParseError,
    VLMRateLimitError,
)
from app.services.vlm.rate_limit import BackoffPolicy, TokenBucket


def _mock_response(text: str, tokens_in: int = 100, tokens_out: int = 50) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=tokens_in,
            candidates_token_count=tokens_out,
        ),
    )


def _make_client(*responses: object) -> MagicMock:
    client = MagicMock()
    client.models = MagicMock()
    client.models.generate_content = MagicMock(side_effect=list(responses))
    return client


def test_classify_pair_returns_valid_result() -> None:
    client = _make_client(
        _mock_response('{"score": 2, "label": "major-damage", "confidence": 0.8, "description": "roof damaged"}')
    )
    classifier = GeminiVLMClassifier(
        client=client,
        prompt_version="v2",
        rate=TokenBucket(100.0),
        backoff=BackoffPolicy(max_attempts=1),
    )
    result = classifier.classify_pair(pre=b"PNGBYTES", post=b"PNGBYTES")
    assert isinstance(result, VLMResult)
    assert result.score == 2
    assert result.label == "major-damage"
    assert result.tokens_in == 100
    assert result.tokens_out == 50


def test_classify_pair_rejects_empty_bytes() -> None:
    classifier = GeminiVLMClassifier(client=_make_client(), prompt_version="v2")
    with pytest.raises(VLMFatalError):
        classifier.classify_pair(pre=b"", post=b"PNGBYTES")


def test_classify_pair_retries_on_parse_error_then_succeeds() -> None:
    client = _make_client(
        _mock_response("not valid json"),
        _mock_response('{"score": 0, "label": "no-damage", "confidence": 0.9, "description": "unchanged"}'),
    )
    classifier = GeminiVLMClassifier(
        client=client,
        prompt_version="v2",
        rate=TokenBucket(100.0),
        backoff=BackoffPolicy(max_attempts=2, base_seconds=0.001, jitter_fraction=0.0),
    )
    result = classifier.classify_pair(pre=b"x", post=b"y")
    assert result.score == 0
    assert client.models.generate_content.call_count == 2


def test_classify_pair_raises_after_exhausting_parse_retries() -> None:
    client = _make_client(
        _mock_response("garbage"),
        _mock_response("garbage"),
    )
    classifier = GeminiVLMClassifier(
        client=client,
        prompt_version="v2",
        rate=TokenBucket(100.0),
        backoff=BackoffPolicy(max_attempts=2, base_seconds=0.001, jitter_fraction=0.0),
    )
    with pytest.raises(VLMParseError):
        classifier.classify_pair(pre=b"x", post=b"y")


def test_classify_error_treats_429_as_rate_limit() -> None:
    exc = Exception("429 RESOURCE_EXHAUSTED")
    exc.status_code = 429
    result = _classify_error(exc)
    assert isinstance(result, VLMFatalError)  # not APIError subtype; generic Exception


def test_classify_error_treats_generic_exception_as_fatal() -> None:
    exc = RuntimeError("network broken")
    result = _classify_error(exc)
    assert isinstance(result, VLMFatalError)
