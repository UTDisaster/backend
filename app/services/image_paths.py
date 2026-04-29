from __future__ import annotations

from pathlib import Path


def normalize_relative_image_path(raw_path: str) -> str:
    path = (raw_path or "").strip().lstrip("/")
    if not path:
        raise ValueError("image path is empty")

    normalized = Path(path).as_posix()
    if normalized.startswith("../") or "/../" in normalized or normalized == "..":
        raise ValueError("image path escapes base directory")
    return normalized


def build_image_url(base_url: str, relative_path: str) -> str:
    clean_base = (base_url or "").rstrip("/")
    if not clean_base:
        raise ValueError("IMAGE_CONTENT_BASE_URL is not set")
    return f"{clean_base}/{normalize_relative_image_path(relative_path)}"
