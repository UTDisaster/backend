from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection


@dataclass(frozen=True)
class AddressMatch:
    street: str | None
    city: str | None
    county: str | None
    total: int
    severe: int
    destroyed: int
    lat: float
    lng: float
    score: float


@dataclass(frozen=True)
class DamageAggregate:
    none: int
    minor: int
    severe: int
    destroyed: int
    unknown: int


# Below this length trigram matches return noise; at or above _LONG_QUERY_LEN
# we match the full_address column rather than just street.
_MIN_QUERY_LEN = 4
_LONG_QUERY_LEN = 20
_MAX_QUERY_LEN = 200
_RADIUS_MIN_M, _RADIUS_MAX_M = 20, 5000

# VLM-normalized damage_level with COALESCE fallback to locations.classification.
_EFFECTIVE_LEVEL_SQL = """COALESCE(
    CASE a.damage_level
        WHEN 'no-damage' THEN 'none'
        WHEN 'minor-damage' THEN 'minor'
        WHEN 'major-damage' THEN 'severe'
        WHEN 'destroyed' THEN 'destroyed'
        WHEN 'unknown' THEN 'unknown'
        ELSE a.damage_level
    END,
    l.classification,
    'unknown'
)"""


def lookup_damage_at_address(
    conn: Connection, query: str, *, limit: int = 5
) -> list[AddressMatch]:
    """Fuzzy-match an address/street string; aggregate damage per (street, city, county).

    Short queries (< 4 chars) return []. Queries >= 20 chars match full_address;
    shorter ones match street. Relies on pg_trgm indexes from the stacked migration.
    """
    q = (query or "").strip()[:_MAX_QUERY_LEN]
    if len(q) < _MIN_QUERY_LEN:
        return []
    target = "l.full_address" if len(q) >= _LONG_QUERY_LEN else "l.street"
    sql = text(f"""
        SELECT l.street AS street, l.city AS city, l.county AS county,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE {_EFFECTIVE_LEVEL_SQL} = 'severe') AS severe,
               COUNT(*) FILTER (WHERE {_EFFECTIVE_LEVEL_SQL} = 'destroyed') AS destroyed,
               AVG(ST_Y(l.centroid)) AS lat, AVG(ST_X(l.centroid)) AS lng,
               MAX(similarity({target}, :q)) AS score
        FROM locations l
        LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
        WHERE similarity({target}, :q) > 0.3
        GROUP BY l.street, l.city, l.county
        ORDER BY score DESC, total DESC
        LIMIT :limit
    """)
    rows = conn.execute(sql, {"q": q, "limit": limit}).mappings().all()
    return [
        AddressMatch(
            street=r["street"], city=r["city"], county=r["county"],
            total=int(r["total"]), severe=int(r["severe"]), destroyed=int(r["destroyed"]),
            lat=float(r["lat"]), lng=float(r["lng"]), score=float(r["score"]),
        )
        for r in rows
    ]


def nearby_damage(
    conn: Connection, lat: float, lng: float, *, radius_m: int = 200
) -> DamageAggregate:
    """Aggregate damage counts within radius_m meters of (lat, lng). Radius clamped to [20, 5000]."""
    clamped = max(_RADIUS_MIN_M, min(_RADIUS_MAX_M, int(radius_m)))
    sql = text(f"""
        SELECT {_EFFECTIVE_LEVEL_SQL} AS level, COUNT(*) AS n
        FROM locations l
        LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
        WHERE ST_DWithin(
            l.centroid::geography,
            ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography,
            :radius
        )
        GROUP BY 1
    """)
    rows = conn.execute(sql, {"lat": lat, "lng": lng, "radius": clamped}).mappings().all()
    buckets = {"none": 0, "minor": 0, "severe": 0, "destroyed": 0, "unknown": 0}
    for r in rows:
        key = r["level"] if r["level"] in buckets else "unknown"
        buckets[key] += int(r["n"])
    return DamageAggregate(**buckets)
