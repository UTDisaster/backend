from __future__ import annotations

import io
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test"
)
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("IMAGE_CONTENT_BASE_URL", "http://test/assets")
os.environ.setdefault("APP_ENV", "dev")

from fastapi.testclient import TestClient

from app import main as app_main
from app.services.vlm.errors import VLMFatalError, VLMParseError, VLMRateLimitError


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_main.app)


def _image_bytes(fmt: str = "PNG", color: tuple[int, int, int] = (50, 120, 180)) -> bytes:
    img = Image.new("RGB", (32, 32), color=color)
    out = io.BytesIO()
    img.save(out, format=fmt)
    return out.getvalue()


def test_vlm_assess_returns_classifier_result(client: TestClient) -> None:
    fake_classifier = MagicMock()
    fake_classifier.classify_pair.return_value = SimpleNamespace(
        score=2,
        label="major-damage",
        confidence=0.84,
        description="Roof section missing in post image.",
        model="gemini-2.5-flash",
        prompt_version="v2",
        latency_ms=321,
    )
    with patch.object(app_main, "GeminiVLMClassifier", return_value=fake_classifier):
        resp = client.post(
            "/vlm/assess",
            files={
                "pre_image": ("pre.jpg", _image_bytes("JPEG"), "image/jpeg"),
                "post_image": ("post.webp", _image_bytes("WEBP"), "image/webp"),
            },
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "score": 2,
        "label": "major-damage",
        "confidence": 0.84,
        "description": "Roof section missing in post image.",
        "model": "gemini-2.5-flash",
        "prompt_version": "v2",
        "latency_ms": 321,
    }
    classify_args, _ = fake_classifier.classify_pair.call_args
    assert classify_args[0].startswith(b"\x89PNG")
    assert classify_args[1].startswith(b"\x89PNG")


def test_vlm_assess_rejects_unsupported_file_type(client: TestClient) -> None:
    resp = client.post(
        "/vlm/assess",
        files={
            "pre_image": ("pre.txt", b"not-an-image", "text/plain"),
            "post_image": ("post.png", _image_bytes("PNG"), "image/png"),
        },
    )

    assert resp.status_code == 400
    assert resp.json() == {"detail": "pre_image must be a PNG, JPEG, or WebP image."}


def test_vlm_assess_surfaces_rate_limit(client: TestClient) -> None:
    fake_classifier = MagicMock()
    fake_classifier.classify_pair.side_effect = VLMRateLimitError(2.2)
    with patch.object(app_main, "GeminiVLMClassifier", return_value=fake_classifier):
        resp = client.post(
            "/vlm/assess",
            files={
                "pre_image": ("pre.png", _image_bytes("PNG"), "image/png"),
                "post_image": ("post.png", _image_bytes("PNG"), "image/png"),
            },
        )

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "3"
    assert resp.json() == {
        "detail": "Gemini VLM is rate limited. Please retry shortly."
    }


def test_vlm_assess_maps_parse_failure_to_502(client: TestClient) -> None:
    fake_classifier = MagicMock()
    fake_classifier.classify_pair.side_effect = VLMParseError("bad output")
    with patch.object(app_main, "GeminiVLMClassifier", return_value=fake_classifier):
        resp = client.post(
            "/vlm/assess",
            files={
                "pre_image": ("pre.png", _image_bytes("PNG"), "image/png"),
                "post_image": ("post.png", _image_bytes("PNG"), "image/png"),
            },
        )

    assert resp.status_code == 502
    assert resp.json() == {
        "detail": "Gemini VLM returned an unparseable assessment."
    }


def test_vlm_assess_maps_fatal_failure_to_503(client: TestClient) -> None:
    fake_classifier = MagicMock()
    fake_classifier.classify_pair.side_effect = VLMFatalError("backend unavailable")
    with patch.object(app_main, "GeminiVLMClassifier", return_value=fake_classifier):
        resp = client.post(
            "/vlm/assess",
            files={
                "pre_image": ("pre.png", _image_bytes("PNG"), "image/png"),
                "post_image": ("post.png", _image_bytes("PNG"), "image/png"),
            },
        )

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Gemini VLM is unavailable."}
