from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.config import get_image_content_base_url, validate_env
from app.db import get_engine
from app.routers.chat import router as chat_router
from app.services.image_paths import build_image_url, normalize_relative_image_path

validate_env()


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]


CORS_ALLOW_ORIGINS = _parse_cors_origins()
CORS_ALLOW_ORIGIN_REGEX = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip() or None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
PARSED_DATA_DIR = (
    Path(os.getenv("PARSED_DATA_DIR", "data-example")).expanduser().resolve()
)
app.mount(
    "/assets",
    StaticFiles(directory=str(PARSED_DATA_DIR), check_dir=False),
    name="assets",
)

MOCK_CHATS = {
    "chat_01": {
        "id": "chat_01",
        "title": "Florence Sector 4 Damage",
        "timestamp": "2026-02-26T10:00:00Z",
        "messages": [
            {"role": "user", "content": "How many buildings are un-classified?"},
            {
                "role": "assistant",
                "content": "I found 1 un-classified building in this view.",
            },
        ],
    },
    "chat_02": {
        "id": "chat_02",
        "title": "Evacuation Routes",
        "timestamp": "2026-02-25T14:30:00Z",
        "messages": [
            {"role": "user", "content": "Is the main road clear?"},
            {
                "role": "assistant",
                "content": "Satellite data shows minor debris on Main St.",
            },
        ],
    },
}


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "UTD Disaster Assessment Project"}


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, int | str]:
    return {"status": "OK", "status_code": status.HTTP_200_OK}


def _normalize_bbox(
    min_lng: float, min_lat: float, max_lng: float, max_lat: float
) -> tuple[float, float, float, float]:
    if min_lng > max_lng or min_lat > max_lat:
        raise HTTPException(
            status_code=400,
            detail="Invalid bounding box: min values must be less than or equal to max values.",
        )

    epsilon = 1e-9
    if min_lng == max_lng:
        max_lng += epsilon
    if min_lat == max_lat:
        max_lat += epsilon

    return min_lng, min_lat, max_lng, max_lat


def _build_image_url(request: Request, path: Optional[str]) -> Optional[str]:
    del request
    if not path:
        return None

    base_url = get_image_content_base_url()
    try:
        return build_image_url(base_url, normalize_relative_image_path(path))
    except ValueError:
        return None


@app.get("/locations")
async def get_locations(
    request: Request,
    min_lng: float = Query(...),
    min_lat: float = Query(...),
    max_lng: float = Query(...),
    max_lat: float = Query(...),
    disaster_id: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
    include_address: bool = Query(False),
) -> dict[str, list[dict[str, object]]]:
    min_lng, min_lat, max_lng, max_lat = _normalize_bbox(
        min_lng, min_lat, max_lng, max_lat
    )

    address_select = (
        ",\n            l.street AS street,\n"
        "            l.city AS city,\n"
        "            l.county AS county,\n"
        "            l.full_address AS full_address,\n"
        "            l.address_source AS address_source,\n"
        "            l.address_fetched_at AS address_fetched_at"
        if include_address
        else ""
    )

    # address_select is a hardcoded literal gated by a bool; no user input reaches the f-string.
    query = f"""
        SELECT
            l.id AS location_id,
            l.location_uid,
            l.image_pair_id,
            l.feature_type,
            l.classification,
            CASE a.damage_level
                WHEN 'no-damage' THEN 'No Damage'
                WHEN 'minor-damage' THEN 'Minor Damage'
                WHEN 'major-damage' THEN 'Major Damage'
                WHEN 'destroyed' THEN 'Destroyed'
                WHEN 'unknown' THEN 'Unknown'
                ELSE a.damage_level
            END AS damage_level,
            a.confidence AS vlm_confidence,
            a.description AS vlm_description,
            ip.disaster_id,
            ip.pre_path,
            ip.post_path,
            ST_AsGeoJSON(l.geom) AS geometry,
            ST_X(l.centroid) AS centroid_lng,
            ST_Y(l.centroid) AS centroid_lat{address_select}
        FROM locations AS l
        JOIN image_pairs AS ip ON ip.id = l.image_pair_id
        LEFT JOIN chat.vlm_assessments AS a ON a.location_id = l.id
        WHERE
            l.geom && ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326)
    """

    params: dict[str, object] = {
        "min_lng": min_lng,
        "min_lat": min_lat,
        "max_lng": max_lng,
        "max_lat": max_lat,
        "limit": limit,
    }

    if disaster_id:
        query += " AND ip.disaster_id = :disaster_id"
        params["disaster_id"] = disaster_id

    query += " ORDER BY l.id LIMIT :limit"

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    features = []
    for row in rows:
        pre_path = row["pre_path"]
        post_path = row["post_path"]
        feature: dict[str, object] = {
            "location_id": row["location_id"],
            "location_uid": row["location_uid"],
            "image_pair_id": row["image_pair_id"],
            "disaster_id": row["disaster_id"],
            "classification": row["classification"],
            "damage_level": row["damage_level"],
            "vlm_confidence": (
                float(row["vlm_confidence"])
                if row["vlm_confidence"] is not None
                else None
            ),
            "vlm_description": row["vlm_description"],
            "feature_type": row["feature_type"],
            "pre_path": pre_path,
            "post_path": post_path,
            "pre_url": _build_image_url(request, pre_path),
            "post_url": _build_image_url(request, post_path),
            "geometry": json.loads(row["geometry"]) if row["geometry"] else None,
            "centroid": {
                "lng": row["centroid_lng"],
                "lat": row["centroid_lat"],
            },
        }
        if include_address:
            if row["full_address"] is None and row["street"] is None and row["city"] is None and row["county"] is None:
                feature["address"] = None
            else:
                fetched_at = row["address_fetched_at"]
                # DB columns address_source / address_fetched_at map to JSON keys source / fetched_at.
                feature["address"] = {
                    "street": row["street"],
                    "city": row["city"],
                    "county": row["county"],
                    "full_address": row["full_address"],
                    "source": row["address_source"],
                    "fetched_at": fetched_at.isoformat() if fetched_at is not None else None,
                }
        features.append(feature)

    return {"features": features}


_HOTSPOT_MIN_CLUSTER_SIZE = 10

_NORMALIZED_DAMAGE_SQL = """
        COALESCE(
            CASE a.damage_level
                WHEN 'no-damage' THEN 'No Damage'
                WHEN 'minor-damage' THEN 'Minor Damage'
                WHEN 'major-damage' THEN 'Major Damage'
                WHEN 'destroyed' THEN 'Destroyed'
                WHEN 'unknown' THEN 'Unknown'
                ELSE NULL
            END,
            CASE l.classification
                WHEN 'none' THEN 'No Damage'
                WHEN 'minor' THEN 'Minor Damage'
                WHEN 'severe' THEN 'Major Damage'
                WHEN 'destroyed' THEN 'Destroyed'
                WHEN 'unknown' THEN 'Unknown'
                ELSE NULL
            END,
            'Unknown'
        )
"""


@app.get("/disasters/{disaster_id}/summary")
async def get_disaster_summary(disaster_id: str) -> dict[str, object]:
    query = f"""
        SELECT
            COUNT(*) AS total_locations,
            COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'No Damage') AS none_count,
            COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Minor Damage') AS minor_count,
            COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Major Damage') AS severe_count,
            COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Destroyed') AS destroyed_count,
            COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Unknown') AS unknown_count,
            MIN(ST_Y(l.centroid)) AS min_lat,
            MAX(ST_Y(l.centroid)) AS max_lat,
            MIN(ST_X(l.centroid)) AS min_lng,
            MAX(ST_X(l.centroid)) AS max_lng
        FROM locations AS l
        JOIN image_pairs AS ip ON ip.id = l.image_pair_id
        LEFT JOIN chat.vlm_assessments AS a ON a.location_id = l.id
        WHERE ip.disaster_id = :disaster_id
    """

    engine = get_engine()
    with engine.connect() as conn:
        row = (
            conn.execute(text(query), {"disaster_id": disaster_id}).mappings().first()
        )

    if row is None or not row["total_locations"]:
        raise HTTPException(status_code=404, detail="disaster not found")

    return {
        "disaster_id": disaster_id,
        "total_locations": int(row["total_locations"]),
        "by_classification": {
            "No Damage": int(row["none_count"] or 0),
            "Minor Damage": int(row["minor_count"] or 0),
            "Major Damage": int(row["severe_count"] or 0),
            "Destroyed": int(row["destroyed_count"] or 0),
            "Unknown": int(row["unknown_count"] or 0),
        },
        "bbox": {
            "minLat": row["min_lat"],
            "minLng": row["min_lng"],
            "maxLat": row["max_lat"],
            "maxLng": row["max_lng"],
        },
    }


@app.get("/locations/hotspots")
async def get_location_hotspots(
    disaster_id: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, list[dict[str, object]]]:
    query = f"""
        SELECT
            round(ST_Y(l.centroid)::numeric, 2) AS lat_bin,
            round(ST_X(l.centroid)::numeric, 2) AS lng_bin,
            COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Major Damage') AS severe_count,
            COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Destroyed') AS destroyed_count,
            COUNT(*) AS total_count
        FROM locations AS l
        JOIN image_pairs AS ip ON ip.id = l.image_pair_id
        LEFT JOIN chat.vlm_assessments AS a ON a.location_id = l.id
        WHERE ip.disaster_id = :disaster_id
          AND l.centroid IS NOT NULL
        GROUP BY lat_bin, lng_bin
        HAVING (
            COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Major Damage')
          + COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Destroyed')
        ) > 0
        AND COUNT(*) >= {_HOTSPOT_MIN_CLUSTER_SIZE}
        ORDER BY
            (
                COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Major Damage')
              + COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Destroyed')
            ) DESC,
            COUNT(*) FILTER (WHERE {_NORMALIZED_DAMAGE_SQL} = 'Destroyed') DESC
        LIMIT :limit
    """

    engine = get_engine()
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(query),
                {"disaster_id": disaster_id, "limit": limit},
            )
            .mappings()
            .all()
        )

    hotspots: list[dict[str, object]] = []
    for row in rows:
        severe = int(row["severe_count"] or 0)
        destroyed = int(row["destroyed_count"] or 0)
        hotspots.append(
            {
                "lat": float(row["lat_bin"]),
                "lng": float(row["lng_bin"]),
                "severe": severe,
                "destroyed": destroyed,
                "total": int(row["total_count"] or 0),
            }
        )

    return {"hotspots": hotspots}


@app.get("/image-pairs")
async def get_image_pairs(
    request: Request,
    min_lng: float = Query(...),
    min_lat: float = Query(...),
    max_lng: float = Query(...),
    max_lat: float = Query(...),
    disaster_id: Optional[str] = Query(None),
    limit: int = Query(2000, ge=1, le=10000),
) -> dict[str, list[dict[str, object]]]:
    min_lng, min_lat, max_lng, max_lat = _normalize_bbox(
        min_lng, min_lat, max_lng, max_lat
    )

    query = """
        SELECT
            ip.id AS image_pair_id,
            ip.disaster_id,
            ip.pair_id,
            ip.pre_path,
            ip.post_path,
            ip.pre_image_id,
            ip.post_image_id,
            ip.pre_min_lat,
            ip.pre_min_lng,
            ip.pre_max_lat,
            ip.pre_max_lng,
            ip.post_min_lat,
            ip.post_min_lng,
            ip.post_max_lat,
            ip.post_max_lng
        FROM image_pairs AS ip
        WHERE EXISTS (
            SELECT 1
            FROM locations AS l
            WHERE l.image_pair_id = ip.id
              AND l.geom && ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326)
              AND ST_Intersects(
                l.geom,
                ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326)
            )
        )
    """

    params: dict[str, object] = {
        "min_lng": min_lng,
        "min_lat": min_lat,
        "max_lng": max_lng,
        "max_lat": max_lat,
        "limit": limit,
    }

    if disaster_id:
        query += " AND ip.disaster_id = :disaster_id"
        params["disaster_id"] = disaster_id

    query += " ORDER BY ip.id LIMIT :limit"

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    pairs = []
    for row in rows:
        pre_path = row["pre_path"]
        post_path = row["post_path"]
        pairs.append(
            {
                "image_pair_id": row["image_pair_id"],
                "disaster_id": row["disaster_id"],
                "pair_id": row["pair_id"],
                "pre_path": pre_path,
                "post_path": post_path,
                "pre_image_id": row["pre_image_id"],
                "post_image_id": row["post_image_id"],
                "pre_url": _build_image_url(request, pre_path),
                "post_url": _build_image_url(request, post_path),
                "pre_bounds": (
                    [
                        [row["pre_min_lat"], row["pre_min_lng"]],
                        [row["pre_max_lat"], row["pre_max_lng"]],
                    ]
                    if row["pre_min_lat"] is not None
                    and row["pre_min_lng"] is not None
                    and row["pre_max_lat"] is not None
                    and row["pre_max_lng"] is not None
                    else None
                ),
                "post_bounds": (
                    [
                        [row["post_min_lat"], row["post_min_lng"]],
                        [row["post_max_lat"], row["post_max_lng"]],
                    ]
                    if row["post_min_lat"] is not None
                    and row["post_min_lng"] is not None
                    and row["post_max_lat"] is not None
                    and row["post_max_lng"] is not None
                    else None
                ),
            }
        )

    return {"image_pairs": pairs}


@app.get("/chat/mock-conversations")
async def list_conversations(search: Optional[str] = None) -> list[dict[str, str]]:
    chat_list = list(MOCK_CHATS.values())

    if search:
        chat_list = [c for c in chat_list if search.lower() in c["title"].lower()]

    chat_list.sort(key=lambda x: x["timestamp"], reverse=True)

    return [
        {"id": c["id"], "title": c["title"], "timestamp": c["timestamp"]}
        for c in chat_list
    ]


@app.get("/chat/mock-conversations/{chat_id}")
async def get_chat(chat_id: str) -> dict[str, object]:
    chat = MOCK_CHATS.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat
