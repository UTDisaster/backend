from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = 50_000_000


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _iter_xy(points: Iterable[dict]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for pt in points or ():
        x = pt.get("x") if isinstance(pt, dict) else None
        y = pt.get("y") if isinstance(pt, dict) else None
        if x is None or y is None:
            continue
        coords.append((float(x), float(y)))
    return coords


def compute_crop_box(
    points: Iterable[dict],
    image_width: int,
    image_height: int,
    *,
    min_size: int = 224,
    max_size: int = 512,
    padding_fraction: float = 0.25,
) -> CropBox | None:
    xy = _iter_xy(points)
    if not xy:
        return None
    xs = [c[0] for c in xy]
    ys = [c[1] for c in xy]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    poly_w = max(1.0, max_x - min_x)
    poly_h = max(1.0, max_y - min_y)
    side = max(poly_w, poly_h) * (1.0 + padding_fraction)
    side = max(float(min_size), min(float(max_size), side))

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    half = side / 2.0
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = int(round(cx + half))
    bottom = int(round(cy + half))

    if left < 0:
        right += -left
        left = 0
    if top < 0:
        bottom += -top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    if right - left <= 1 or bottom - top <= 1:
        return None
    return CropBox(left=left, top=top, right=right, bottom=bottom)


def crop_png(
    image_bytes: bytes,
    box: CropBox,
    *,
    outline_points: Iterable[dict] | None = None,
    outline_color: tuple[int, int, int] = (255, 0, 0),
    outline_width: int = 3,
) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        cropped = img.crop((box.left, box.top, box.right, box.bottom))
        if cropped.mode not in ("RGB", "RGBA"):
            cropped = cropped.convert("RGB")
        if outline_points:
            crop_space: list[tuple[float, float]] = []
            for pt in outline_points:
                x = pt.get("x") if isinstance(pt, dict) else None
                y = pt.get("y") if isinstance(pt, dict) else None
                if x is None or y is None:
                    continue
                crop_space.append((float(x) - box.left, float(y) - box.top))
            if len(crop_space) >= 3:
                if crop_space[0] != crop_space[-1]:
                    crop_space.append(crop_space[0])
                draw = ImageDraw.Draw(cropped)
                draw.line(crop_space, fill=outline_color, width=outline_width, joint="curve")
        buf = io.BytesIO()
        cropped.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def lnglat_ring_to_xy(
    ring: list[list[float]],
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    image_width: int,
    image_height: int,
) -> list[dict[str, float]]:
    """Linearly map a GeoJSON ring (lng/lat) into pixel xy given axis-aligned image bounds."""
    lat_span = (max_lat - min_lat) or 1e-9
    lng_span = (max_lng - min_lng) or 1e-9
    points: list[dict[str, float]] = []
    for lng, lat in ring:
        x = (lng - min_lng) / lng_span * image_width
        y = (max_lat - lat) / lat_span * image_height
        points.append({"x": x, "y": y})
    return points


def crop_for_location(
    pre_bytes: bytes,
    post_bytes: bytes,
    location: dict,
    *,
    min_size: int = 224,
    max_size: int = 512,
    padding_fraction: float = 0.25,
    draw_outline: bool = False,
) -> tuple[bytes, bytes] | None:
    """Returns (pre_crop, post_crop) PNG bytes, or None if polygon data is missing.

    When draw_outline=True, the building polygon is drawn on each crop (for v4 prompt).
    """
    with Image.open(io.BytesIO(pre_bytes)) as pre_img:
        pre_w, pre_h = pre_img.size
    with Image.open(io.BytesIO(post_bytes)) as post_img:
        post_w, post_h = post_img.size

    points = location.get("points", {}) or {}
    pre_points = points.get("pre") or points.get("post") or []
    post_points = points.get("post") or points.get("pre") or []

    pre_box = compute_crop_box(
        pre_points, pre_w, pre_h,
        min_size=min_size, max_size=max_size, padding_fraction=padding_fraction,
    )
    post_box = compute_crop_box(
        post_points, post_w, post_h,
        min_size=min_size, max_size=max_size, padding_fraction=padding_fraction,
    )
    if pre_box is None or post_box is None:
        return None
    outline_pre = pre_points if draw_outline else None
    outline_post = post_points if draw_outline else None
    pre_crop = crop_png(pre_bytes, pre_box, outline_points=outline_pre)
    post_crop = crop_png(post_bytes, post_box, outline_points=outline_post)
    return pre_crop, post_crop
