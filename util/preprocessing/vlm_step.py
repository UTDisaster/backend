from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sqlalchemy as sa

from app.db import get_engine
from app.env_loader import load_app_env
from app.services.cropping import crop_for_location, lnglat_ring_to_xy
from app.services.storage import ContentImageStore, ImageStore
from app.services.vlm import (
    BackoffPolicy,
    GeminiVLMClassifier,
    TokenBucket,
    VLMClassifier,
    VLMClassifyError,
    VLMFatalError,
)

load_app_env()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLMRunStats:
    attempted: int
    classified: int
    skipped_no_crop: int
    skipped_existing: int
    errors: int
    written: int
    started_at: float
    ended_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "classified": self.classified,
            "skipped_no_crop": self.skipped_no_crop,
            "skipped_existing": self.skipped_existing,
            "errors": self.errors,
            "written": self.written,
            "elapsed_seconds": round(self.ended_at - self.started_at, 2),
        }


_BASE_SELECT = """
SELECT
    l.id              AS location_id,
    l.location_uid    AS location_uid,
    l.image_pair_id   AS image_pair_id,
    l.classification  AS classification,
    ST_AsGeoJSON(l.geom)    AS geom_json,
    ip.pre_path       AS pre_path,
    ip.post_path      AS post_path,
    ip.pre_min_lat    AS pre_min_lat,
    ip.pre_min_lng    AS pre_min_lng,
    ip.pre_max_lat    AS pre_max_lat,
    ip.pre_max_lng    AS pre_max_lng,
    ip.post_min_lat   AS post_min_lat,
    ip.post_min_lng   AS post_min_lng,
    ip.post_max_lat   AS post_max_lat,
    ip.post_max_lng   AS post_max_lng,
    existing.id       AS existing_assessment_id
FROM locations AS l
JOIN image_pairs AS ip ON ip.id = l.image_pair_id
LEFT JOIN chat.vlm_assessments AS existing
    ON existing.location_id = l.id
"""


def _build_query(
    *,
    disaster_id: str | None,
    skip_classified: bool,
    stratified_per_class: int | None,
    limit: int | None,
) -> tuple[str, dict[str, Any]]:
    where = ["ip.pre_min_lat IS NOT NULL", "ip.post_min_lat IS NOT NULL"]
    params: dict[str, Any] = {}
    if disaster_id:
        where.append("ip.disaster_id = :disaster_id")
        params["disaster_id"] = disaster_id
    if skip_classified:
        where.append("existing.id IS NULL")
    where_clause = " WHERE " + " AND ".join(where)

    if stratified_per_class:
        params["per_class"] = int(stratified_per_class)
        query = f"""
SELECT * FROM (
    SELECT ranked.*,
        ROW_NUMBER() OVER (PARTITION BY ranked.classification ORDER BY random()) AS rn
    FROM (
        {_BASE_SELECT}
        {where_clause}
    ) AS ranked
) AS stratified
WHERE stratified.rn <= :per_class
ORDER BY stratified.image_pair_id, stratified.location_id
"""
    else:
        query = f"{_BASE_SELECT}{where_clause} ORDER BY l.image_pair_id, l.id"
        if limit:
            query += " LIMIT :limit"
            params["limit"] = int(limit)
    return query, params


def _ring_from_geom(geom_json: str | None) -> list[list[float]] | None:
    if not geom_json:
        return None
    try:
        parsed = json.loads(geom_json)
    except (TypeError, json.JSONDecodeError):
        return None
    coords = parsed.get("coordinates")
    if parsed.get("type") != "Polygon" or not coords:
        return None
    return coords[0]


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    from PIL import Image as PILImage

    with PILImage.open(io.BytesIO(image_bytes)) as img:
        return img.size


def _build_location_points(
    row: dict[str, Any],
    pre_size: tuple[int, int],
    post_size: tuple[int, int],
) -> dict[str, list[dict[str, float]]] | None:
    ring = _ring_from_geom(row.get("geom_json"))
    if not ring:
        return None
    pre_points = lnglat_ring_to_xy(
        ring,
        min_lat=row["pre_min_lat"],
        min_lng=row["pre_min_lng"],
        max_lat=row["pre_max_lat"],
        max_lng=row["pre_max_lng"],
        image_width=pre_size[0],
        image_height=pre_size[1],
    )
    post_points = lnglat_ring_to_xy(
        ring,
        min_lat=row["post_min_lat"],
        min_lng=row["post_min_lng"],
        max_lat=row["post_max_lat"],
        max_lng=row["post_max_lng"],
        image_width=post_size[0],
        image_height=post_size[1],
    )
    return {"pre": pre_points, "post": post_points}


def _upsert_batch(
    conn: sa.engine.Connection,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0
    stmt = sa.text(
        """
        INSERT INTO chat.vlm_assessments (location_id, damage_level, confidence, description)
        VALUES (:location_id, :damage_level, :confidence, :description)
        ON CONFLICT (location_id) DO UPDATE SET
            damage_level = EXCLUDED.damage_level,
            confidence   = EXCLUDED.confidence,
            description  = EXCLUDED.description,
            created_at   = now()
        """
    )
    conn.execute(stmt, rows)
    return len(rows)


class _JsonLogWriter:
    def __init__(self, path: Path | None) -> None:
        self._fp = path.open("a", encoding="utf-8") if path else None
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        if self._fp is None:
            return
        with self._lock:
            self._fp.write(json.dumps(record, default=str) + "\n")
            self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()


def _classify_single(
    row: dict[str, Any],
    *,
    store: ImageStore,
    classifier: VLMClassifier,
    min_crop_size: int,
    max_crop_size: int,
    padding_fraction: float,
) -> dict[str, Any]:
    try:
        pre_bytes, post_bytes = store.fetch_pair(row["pre_path"], row["post_path"])
    except Exception as exc:
        return {
            "status": "image_error",
            "row": row,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        pre_size = _image_size(pre_bytes)
        post_size = _image_size(post_bytes)
        points = _build_location_points(row, pre_size, post_size)
        if not points:
            return {"status": "no_points", "row": row}
        crop = crop_for_location(
            pre_bytes,
            post_bytes,
            {"points": points},
            min_size=min_crop_size,
            max_size=max_crop_size,
            padding_fraction=padding_fraction,
            draw_outline=getattr(classifier, "prompt_version", "") == "v4",
        )
    except Exception as exc:
        return {
            "status": "crop_error",
            "row": row,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not crop:
        return {"status": "no_crop", "row": row}

    pre_crop, post_crop = crop
    try:
        prediction = classifier.classify_pair(
            pre_crop,
            post_crop,
            pair_id=row["image_pair_id"],
            location_uid=row["location_uid"],
        )
    except VLMFatalError as exc:
        return {"status": "fatal", "row": row, "error": str(exc)}
    except VLMClassifyError as exc:
        return {"status": "classify_error", "row": row, "error": str(exc)}

    return {"status": "ok", "row": row, "prediction": prediction}


def run_vlm_step(
    *,
    disaster_id: str | None = None,
    limit: int | None = None,
    stratified_per_class: int | None = None,
    skip_classified: bool = True,
    rps: float = 5.0,
    concurrency: int = 5,
    prompt_version: str = "v2",
    model: str = "gemini-2.5-flash",
    min_crop_size: int = 320,
    max_crop_size: int = 640,
    padding_fraction: float = 0.4,
    checkpoint_every: int = 25,
    max_errors: int = 200,
    log_path: Path | str | None = None,
    store: ImageStore | None = None,
    classifier: VLMClassifier | None = None,
    progress: bool = True,
) -> VLMRunStats:
    engine = get_engine()
    owned_store = store is None
    store = store or ContentImageStore()
    classifier = classifier or GeminiVLMClassifier(
        model=model,
        prompt_version=prompt_version,
        rate=TokenBucket(rps, burst=max(1, int(rps * 2))),
        backoff=BackoffPolicy(max_attempts=5, base_seconds=3.0, max_seconds=30.0),
    )

    query, params = _build_query(
        disaster_id=disaster_id,
        skip_classified=skip_classified,
        stratified_per_class=stratified_per_class,
        limit=limit,
    )

    with engine.connect() as read_conn:
        rows = [dict(r) for r in read_conn.execute(sa.text(query), params).mappings()]

    log_writer = _JsonLogWriter(Path(log_path) if log_path else None)
    attempted = classified = skipped_no_crop = errors = written = 0
    skipped_existing = 0
    buffer: list[dict[str, Any]] = []
    iterator: Any = rows

    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(total=len(rows), unit="loc", desc=f"VLM {prompt_version} c={concurrency}")
        except ImportError:
            iterator = None
    else:
        iterator = None

    started_at = time.monotonic()

    def _flush(rows_to_write: list[dict[str, Any]]) -> int:
        if not rows_to_write:
            return 0
        try:
            with engine.begin() as write_conn:
                return _upsert_batch(write_conn, rows_to_write)
        except Exception as exc:
            logger.error(
                "vlm_step flush failed: rows=%d error=%s",
                len(rows_to_write),
                exc,
            )
            raise

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    _classify_single,
                    row,
                    store=store,
                    classifier=classifier,
                    min_crop_size=min_crop_size,
                    max_crop_size=max_crop_size,
                    padding_fraction=padding_fraction,
                )
                for row in rows
            ]
            for future in as_completed(futures):
                attempted += 1
                try:
                    outcome = future.result()
                except Exception as exc:
                    errors += 1
                    logger.warning(
                        "vlm_step worker crashed: error=%s",
                        exc,
                    )
                    log_writer.write({
                        "event": "worker_crash",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    if iterator is not None:
                        iterator.update(1)
                    if errors >= max_errors:
                        if iterator is not None:
                            iterator.close()
                        raise RuntimeError(
                            f"Aborting: {errors} errors >= max_errors={max_errors}"
                        )
                    continue
                row = outcome["row"]
                status = outcome["status"]
                if status == "ok":
                    prediction = outcome["prediction"]
                    classified += 1
                    buffer.append({
                        "location_id": int(row["location_id"]),
                        "damage_level": prediction.label,
                        "confidence": prediction.confidence,
                        "description": prediction.description,
                    })
                    log_writer.write({
                        "event": "classified",
                        "location_id": int(row["location_id"]),
                        "pair_id": row["image_pair_id"],
                        "truth": row.get("classification"),
                        "pred": prediction.label,
                        "score": prediction.score,
                        "confidence": prediction.confidence,
                        "latency_ms": prediction.latency_ms,
                        "prompt_version": prediction.prompt_version,
                        "model": prediction.model,
                        "tokens_in": prediction.tokens_in,
                        "tokens_out": prediction.tokens_out,
                    })
                elif status in ("no_crop", "no_points"):
                    skipped_no_crop += 1
                    log_writer.write({
                        "event": "skip_no_crop",
                        "location_id": int(row["location_id"]),
                        "reason": status,
                    })
                else:
                    errors += 1
                    error_message = outcome.get("error")
                    logger.warning(
                        "vlm_step error: location_id=%s status=%s error=%s",
                        row["location_id"],
                        status,
                        error_message,
                    )
                    log_writer.write({
                        "event": f"error_{status}",
                        "location_id": int(row["location_id"]),
                        "error": error_message,
                    })

                if iterator is not None:
                    iterator.update(1)

                if len(buffer) >= checkpoint_every:
                    written += _flush(buffer)
                    buffer.clear()

                if errors >= max_errors:
                    if iterator is not None:
                        iterator.close()
                    raise RuntimeError(
                        f"Aborting: {errors} errors >= max_errors={max_errors}"
                    )

            if buffer:
                written += _flush(buffer)
                buffer.clear()
    finally:
        if iterator is not None:
            iterator.close()
        log_writer.close()
        if owned_store:
            try:
                store.close()  # type: ignore[attr-defined]
            except AttributeError:
                pass

    ended_at = time.monotonic()
    stats = VLMRunStats(
        attempted=attempted,
        classified=classified,
        skipped_no_crop=skipped_no_crop,
        skipped_existing=skipped_existing,
        errors=errors,
        written=written,
        started_at=started_at,
        ended_at=ended_at,
    )
    logger.info("vlm_step stats: %s total=%s", stats.to_dict(), len(rows))
    return stats
