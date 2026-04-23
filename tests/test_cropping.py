from __future__ import annotations

import io

import pytest
from PIL import Image

from app.services.cropping import (
    CropBox,
    compute_crop_box,
    crop_for_location,
    crop_png,
    lnglat_ring_to_xy,
)


def _make_png(width: int, height: int, color: tuple[int, int, int] = (200, 50, 50)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_compute_crop_box_basic() -> None:
    points = [{"x": 100, "y": 200}, {"x": 150, "y": 250}, {"x": 140, "y": 230}]
    box = compute_crop_box(points, image_width=1024, image_height=1024, min_size=128, max_size=512)
    assert box is not None
    assert 0 <= box.left < box.right <= 1024
    assert 0 <= box.top < box.bottom <= 1024


def test_compute_crop_box_clamps_near_edge() -> None:
    points = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]
    box = compute_crop_box(points, image_width=100, image_height=100, min_size=64, max_size=80)
    assert box is not None
    assert box.left == 0 and box.top == 0
    assert box.right <= 100 and box.bottom <= 100


def test_compute_crop_box_returns_none_on_empty_points() -> None:
    assert compute_crop_box([], image_width=1024, image_height=1024) is None


def test_crop_png_returns_expected_dimensions() -> None:
    png = _make_png(1024, 1024)
    box = CropBox(left=100, top=200, right=300, bottom=400)
    cropped = crop_png(png, box)
    with Image.open(io.BytesIO(cropped)) as img:
        assert img.size == (200, 200)


def test_crop_for_location_returns_pair() -> None:
    pre = _make_png(1024, 1024, (10, 10, 10))
    post = _make_png(1024, 1024, (250, 10, 10))
    location = {
        "points": {
            "pre": [{"x": 300, "y": 300}, {"x": 400, "y": 400}],
            "post": [{"x": 305, "y": 305}, {"x": 405, "y": 405}],
        }
    }
    result = crop_for_location(pre, post, location)
    assert result is not None
    pre_crop, post_crop = result
    with Image.open(io.BytesIO(pre_crop)) as img:
        assert min(img.size) >= 128


def test_crop_for_location_missing_points_returns_none() -> None:
    pre = _make_png(256, 256)
    post = _make_png(256, 256)
    assert crop_for_location(pre, post, {"points": {"pre": [], "post": []}}) is None


def test_lnglat_ring_to_xy_corners() -> None:
    ring = [[-77.0, 34.0], [-76.0, 34.0], [-76.0, 33.0], [-77.0, 33.0]]
    points = lnglat_ring_to_xy(
        ring,
        min_lat=33.0,
        min_lng=-77.0,
        max_lat=34.0,
        max_lng=-76.0,
        image_width=1024,
        image_height=1024,
    )
    assert points[0] == pytest.approx({"x": 0.0, "y": 0.0})
    assert points[1] == pytest.approx({"x": 1024.0, "y": 0.0})
    assert points[2] == pytest.approx({"x": 1024.0, "y": 1024.0})
    assert points[3] == pytest.approx({"x": 0.0, "y": 1024.0})
