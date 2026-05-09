from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import text

from app.db import get_engine

router = APIRouter(prefix="/data-quality", tags=["data-quality"])


@router.get("/report")
async def get_report(
    disaster_id: str = Query(...),
) -> dict[str, object]:
    main_query = text("""
        SELECT
            COUNT(*) AS total_locations,
            COUNT(*) FILTER (WHERE l.geom IS NOT NULL) AS with_geometry,
            COUNT(*) FILTER (WHERE l.geom IS NULL) AS without_geometry,
            COUNT(*) FILTER (WHERE a.id IS NOT NULL) AS with_vlm_assessment,
            COUNT(DISTINCT ip.id) FILTER (
                WHERE ip.pre_min_lat IS NOT NULL
                  AND ip.pre_min_lng IS NOT NULL
                  AND ip.pre_max_lat IS NOT NULL
                  AND ip.pre_max_lng IS NOT NULL
            ) AS with_valid_pre_bounds,
            COUNT(DISTINCT ip.id) FILTER (
                WHERE ip.post_min_lat IS NOT NULL
                  AND ip.post_min_lng IS NOT NULL
                  AND ip.post_max_lat IS NOT NULL
                  AND ip.post_max_lng IS NOT NULL
            ) AS with_valid_post_bounds,
            COUNT(DISTINCT ip.id) FILTER (
                WHERE ip.pre_path IS NOT NULL AND ip.pre_path != ''
            ) AS with_valid_pre_path,
            COUNT(DISTINCT ip.id) FILTER (
                WHERE ip.post_path IS NOT NULL AND ip.post_path != ''
            ) AS with_valid_post_path,
            COUNT(DISTINCT ip.id) FILTER (
                WHERE (ip.pre_min_lat IS NULL OR ip.pre_min_lng IS NULL
                       OR ip.pre_max_lat IS NULL OR ip.pre_max_lng IS NULL)
                  AND (ip.post_min_lat IS NULL OR ip.post_min_lng IS NULL
                       OR ip.post_max_lat IS NULL OR ip.post_max_lng IS NULL)
            ) AS missing_both_bounds,
            COUNT(DISTINCT ip.id) FILTER (
                WHERE (ip.pre_path IS NULL OR ip.pre_path = '')
                  AND (ip.post_path IS NULL OR ip.post_path = '')
            ) AS missing_both_paths,
            COUNT(DISTINCT ip.id) FILTER (
                WHERE (ip.pre_min_lat IS NOT NULL
                       AND ip.pre_min_lng IS NOT NULL
                       AND ip.pre_max_lat IS NOT NULL
                       AND ip.pre_max_lng IS NOT NULL
                       AND ip.pre_path IS NOT NULL AND ip.pre_path != '')
                   OR (ip.post_min_lat IS NOT NULL
                       AND ip.post_min_lng IS NOT NULL
                       AND ip.post_max_lat IS NOT NULL
                       AND ip.post_max_lng IS NOT NULL
                       AND ip.post_path IS NOT NULL AND ip.post_path != '')
            ) AS renderable_overlays,
            COUNT(*) FILTER (
                WHERE l.geom IS NOT NULL
                  AND NOT (
                      (ip.pre_min_lat IS NOT NULL
                       AND ip.pre_min_lng IS NOT NULL
                       AND ip.pre_max_lat IS NOT NULL
                       AND ip.pre_max_lng IS NOT NULL
                       AND ip.pre_path IS NOT NULL AND ip.pre_path != '')
                      OR
                      (ip.post_min_lat IS NOT NULL
                       AND ip.post_min_lng IS NOT NULL
                       AND ip.post_max_lat IS NOT NULL
                       AND ip.post_max_lng IS NOT NULL
                       AND ip.post_path IS NOT NULL AND ip.post_path != '')
                  )
            ) AS locations_without_overlay
        FROM locations AS l
        JOIN image_pairs AS ip ON ip.id = l.image_pair_id
        LEFT JOIN chat.vlm_assessments AS a ON a.location_id = l.id
        WHERE ip.disaster_id = :disaster_id
    """)

    unreferenced_query = text("""
        SELECT COUNT(*) AS unreferenced
        FROM image_pairs AS ip
        WHERE ip.disaster_id = :disaster_id
          AND NOT EXISTS (
              SELECT 1 FROM locations AS l WHERE l.image_pair_id = ip.id
          )
    """)

    total_ip_query = text("""
        SELECT COUNT(*) AS total
        FROM image_pairs AS ip
        WHERE ip.disaster_id = :disaster_id
    """)

    engine = get_engine()
    with engine.connect() as conn:
        params = {"disaster_id": disaster_id}
        main_row = conn.execute(main_query, params).mappings().first()
        unref_row = conn.execute(unreferenced_query, params).mappings().first()
        total_ip_row = conn.execute(total_ip_query, params).mappings().first()

    if main_row is None:
        main_row = {}

    total_ip = int(total_ip_row["total"]) if total_ip_row else 0
    unreferenced = int(unref_row["unreferenced"]) if unref_row else 0
    referenced = total_ip - unreferenced

    return {
        "disaster_id": disaster_id,
        "locations": {
            "total": int(main_row.get("total_locations", 0) or 0),
            "with_geometry": int(main_row.get("with_geometry", 0) or 0),
            "without_geometry": int(main_row.get("without_geometry", 0) or 0),
            "with_vlm_assessment": int(main_row.get("with_vlm_assessment", 0) or 0),
        },
        "image_pairs": {
            "total": total_ip,
            "with_valid_pre_bounds": int(main_row.get("with_valid_pre_bounds", 0) or 0),
            "with_valid_post_bounds": int(main_row.get("with_valid_post_bounds", 0) or 0),
            "with_valid_pre_path": int(main_row.get("with_valid_pre_path", 0) or 0),
            "with_valid_post_path": int(main_row.get("with_valid_post_path", 0) or 0),
            "missing_both_bounds": int(main_row.get("missing_both_bounds", 0) or 0),
            "missing_both_paths": int(main_row.get("missing_both_paths", 0) or 0),
            "referenced_by_locations": referenced,
            "unreferenced": unreferenced,
        },
        "rendering": {
            "renderable_overlays": int(main_row.get("renderable_overlays", 0) or 0),
            "locations_without_overlay": int(main_row.get("locations_without_overlay", 0) or 0),
        },
    }
