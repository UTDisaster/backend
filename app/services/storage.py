from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import httpx
from dotenv import load_dotenv

load_dotenv()


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


class SupabaseImageStore:
    """Fetches pre/post PNG bytes from a Supabase public Storage bucket.

    Env-driven defaults:
      SUPABASE_URL, SUPABASE_IMAGES_BUCKET, SUPABASE_STRIP_PREFIX,
      SUPABASE_USE_BASENAME, VLM_DATASET_DIR.

    If VLM_DATASET_DIR is set and contains the file, the local copy wins.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        bucket: str | None = None,
        strip_prefix: str | None = None,
        use_basename: bool | None = None,
        local_dir: Path | str | None = None,
        cache_size: int = 2048,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self._bucket = (bucket or os.getenv("SUPABASE_IMAGES_BUCKET", "images")).strip()
        self._strip_prefix = (
            strip_prefix
            if strip_prefix is not None
            else os.getenv("SUPABASE_STRIP_PREFIX", "images/")
        ).strip().lstrip("/")
        self._use_basename = (
            use_basename
            if use_basename is not None
            else os.getenv("SUPABASE_USE_BASENAME", "false").lower() == "true"
        )
        local_env = os.getenv("VLM_DATASET_DIR")
        self._local_dir = (
            Path(local_dir).expanduser().resolve() if local_dir is not None
            else Path(local_env).expanduser().resolve() if local_env
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
                f"Supabase fetch failed [{response.status_code}]: {url}"
            )
        data = response.content
        self._cache.put(raw_path, data)
        return data

    def _normalize(self, raw_path: str) -> str:
        path = raw_path.lstrip("/")
        if self._strip_prefix and path.startswith(self._strip_prefix):
            path = path[len(self._strip_prefix):].lstrip("/")
        if self._use_basename:
            path = Path(path).name
        return path

    def _remote_url(self, raw_path: str) -> str:
        if not self._base_url:
            raise RuntimeError("SUPABASE_URL is not set")
        return f"{self._base_url}/storage/v1/object/public/{self._bucket}/{self._normalize(raw_path)}"

    def _local_path(self, raw_path: str) -> Path | None:
        if self._local_dir is None:
            return None
        candidate = (self._local_dir / self._normalize(raw_path)).resolve()
        try:
            candidate.relative_to(self._local_dir)
        except ValueError:
            return None
        return candidate

    def close(self) -> None:
        self._client.close()
