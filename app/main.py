from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.db import get_engine

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/locations")
async def get_locations(
    request: Request,
    min_lng: float = Query(...),
    min_lat: float = Query(...),
    max_lng: float = Query(...),
    max_lat: float = Query(...),
    disaster_id: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
) -> dict[str, list[dict[str, object]]]:
    if min_lng >= max_lng or min_lat >= max_lat:
        raise HTTPException(
            status_code=400,
            detail="Invalid bounding box: min values must be less than max values.",
        )

    query = """
        SELECT
            l.location_uid,
            l.image_pair_id,
            l.feature_type,
            l.classification,
            l.prediction,
            ip.disaster_id,
            ip.pre_path,
            ip.post_path,
            ST_AsGeoJSON(l.geom) AS geometry,
            ST_X(l.centroid) AS centroid_lng,
            ST_Y(l.centroid) AS centroid_lat
        FROM locations AS l
        JOIN image_pairs AS ip ON ip.id = l.image_pair_id
        WHERE
            l.geom && ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326)
            AND ST_Intersects(
                l.geom,
                ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326)
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

    query += " ORDER BY l.id LIMIT :limit"

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    features = []
    for row in rows:
        pre_path = row["pre_path"]
        post_path = row["post_path"]
        features.append(
            {
                "location_uid": row["location_uid"],
                "image_pair_id": row["image_pair_id"],
                "disaster_id": row["disaster_id"],
                "classification": row["classification"],
                "prediction": row["prediction"],
                "feature_type": row["feature_type"],
                "pre_path": pre_path,
                "post_path": post_path,
                "pre_url": str(request.url_for("assets", path=pre_path))
                if pre_path
                else None,
                "post_url": str(request.url_for("assets", path=post_path))
                if post_path
                else None,
                "geometry": json.loads(row["geometry"]) if row["geometry"] else None,
                "centroid": {
                    "lng": row["centroid_lng"],
                    "lat": row["centroid_lat"],
                },
            }
        )

    return {"features": features}


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
    if min_lng >= max_lng or min_lat >= max_lat:
        raise HTTPException(
            status_code=400,
            detail="Invalid bounding box: min values must be less than max values.",
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
                "pre_url": str(request.url_for("assets", path=pre_path))
                if pre_path
                else None,
                "post_url": str(request.url_for("assets", path=post_path))
                if post_path
                else None,
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


@app.get("/chat/conversations")
async def list_conversations(search: Optional[str] = None) -> list[dict[str, str]]:
    chat_list = list(MOCK_CHATS.values())

    if search:
        chat_list = [c for c in chat_list if search.lower() in c["title"].lower()]

    chat_list.sort(key=lambda x: x["timestamp"], reverse=True)

    return [
        {"id": c["id"], "title": c["title"], "timestamp": c["timestamp"]}
        for c in chat_list
    ]


@app.get("/chat/conversations/{chat_id}")
async def get_chat(chat_id: str) -> dict[str, object]:
    chat = MOCK_CHATS.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat
