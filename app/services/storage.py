from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import httpx

from app.config import get_app_env, get_image_content_base_url
from app.env_loader import load_app_env
from app.services.image_paths import build_image_url, normalize_relative_image_path

load_app_env()


class ImageStore(Protocol):
    def fetch_pair(self, pre_path: str, post_path: str) -> tuple[bytes, bytes]: ...


class _LRU:
    def __init__(self, maxsize: int) -> None:
        self._store: OrderedDict[str, bytes] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)
            return value

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)


class ContentImageStore:
    """Fetches pre/post image bytes from local data dir or image content base URL."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        local_dir: Path | str | None = None,
        cache_size: int = 2048,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = (base_url or get_image_content_base_url()).rstrip("/")
        app_env = get_app_env()
        local_env = os.getenv("VLM_DATASET_DIR")
        parsed_data_dir = os.getenv("PARSED_DATA_DIR")
        self._local_dir = None
        if app_env == "dev":
            self._local_dir = (
                Path(local_dir).expanduser().resolve() if local_dir is not None
                else Path(local_env).expanduser().resolve() if local_env
                else Path(parsed_data_dir).expanduser().resolve() if parsed_data_dir
                else None
            )
        self._cache = _LRU(cache_size)
        self._client = client if client is not None else httpx.Client(
            timeout=30.0, follow_redirects=True
        )

    def fetch_pair(self, pre_path: str, post_path: str) -> tuple[bytes, bytes]:
        return self._fetch_one(pre_path), self._fetch_one(post_path)

    def _fetch_one(self, raw_path: str) -> bytes:
        cached = self._cache.get(raw_path)
        if cached is not None:
            return cached

        local = self._local_path(raw_path)
        if local is not None and local.is_file():
            data = local.read_bytes()
            self._cache.put(raw_path, data)
            return data

        url = self._remote_url(raw_path)
        response = self._client.get(url)
        if response.status_code != 200:
            raise FileNotFoundError(
                f"Image fetch failed [{response.status_code}]: {url}"
            )
        data = response.content
        self._cache.put(raw_path, data)
        return data

    def _remote_url(self, raw_path: str) -> str:
        try:
            return build_image_url(self._base_url, normalize_relative_image_path(raw_path))
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def _local_path(self, raw_path: str) -> Path | None:
        if self._local_dir is None:
            return None
        try:
            normalized = normalize_relative_image_path(raw_path)
        except ValueError:
            return None
        candidate = (self._local_dir / normalized).resolve()
        try:
            candidate.relative_to(self._local_dir)
        except ValueError:
            return None
        return candidate

    def close(self) -> None:
        self._client.close()
