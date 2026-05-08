from __future__ import annotations

import io

from fastapi import UploadFile, status
from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 50_000_000

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class UploadedImageError(Exception):
    def __init__(self, detail: str, *, status_code: int = status.HTTP_400_BAD_REQUEST) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


async def normalize_uploaded_image(
    upload: UploadFile,
    *,
    field_name: str,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> bytes:
    content_type = (upload.content_type or "").strip().lower()
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise UploadedImageError(
            f"{field_name} must be a PNG, JPEG, or WebP image."
        )

    raw = await upload.read()
    if not raw:
        raise UploadedImageError(f"{field_name} is empty.")
    if len(raw) > max_bytes:
        raise UploadedImageError(
            f"{field_name} exceeds the 10 MB upload limit.",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    try:
        with Image.open(io.BytesIO(raw)) as img:
            normalized = img.convert("RGB")
            out = io.BytesIO()
            normalized.save(out, format="PNG", optimize=True)
            return out.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UploadedImageError(
            f"{field_name} is not a readable image."
        ) from exc
