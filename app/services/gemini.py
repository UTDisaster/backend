from __future__ import annotations

import os
import json
import logging
import re
import threading

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from sqlalchemy import text

from app.db import get_engine
from app.env_loader import load_app_env
from app.services.location_queries import (
    lookup_damage_at_address,
    nearby_damage,
)

load_app_env()
logger = logging.getLogger(__name__)


class ChatBackendUnavailableError(Exception):
    def __init__(
        self,
        status_code: int = 503,
        retry_after_seconds: int | None = None,
        detail: str = "Chat backend unavailable",
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.detail = detail


def _extract_retry_after_seconds(message: str) -> int | None:
    retry_in = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if retry_in:
        return max(1, int(float(retry_in.group(1))))
    retry_delay = re.search(r"'retryDelay': '([0-9]+)s'", message, re.IGNORECASE)
    if retry_delay:
        return max(1, int(retry_delay.group(1)))
    return None


_client_lock = threading.Lock()
_cached_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _cached_client
    with _client_lock:
        if _cached_client is not None:
            return _cached_client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ChatBackendUnavailableError(
                status_code=503,
                detail="Gemini API key is not configured.",
            )
        _cached_client = genai.Client(api_key=api_key)
        return _cached_client


def _invalidate_client() -> None:
    global _cached_client
    with _client_lock:
        _cached_client = None


def _is_auth_error(exc: ClientError) -> bool:
    status = getattr(exc, "status_code", None)
    return status in (401, 403)


# ── Tool definitions (Gemini decides when to call these) ─────────────

TOOLS = [
    {
        "name": "get_damage_stats",
        "description": (
            "Get damage statistics for a disaster. Returns counts of buildings "
            "by damage classification (No Damage, Minor Damage, Major Damage, Destroyed, Unknown). "
            "Use when the user asks about how many buildings were damaged, destroyed, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "disaster_id": {
                    "type": "string",
                    "description": "The disaster ID e.g. 'hurricane-florence'. Optional — omit to get stats across all disasters."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_locations_by_damage",
        "description": (
            "Get a list of specific buildings/locations filtered by damage level. "
            "Use when the user asks about specific houses or wants to see locations by damage class. "
            "Use get_damage_hotspots instead for worst or most-damaged areas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "damage_level": {
                    "type": "string",
                    "description": "One of: No Damage, Minor Damage, Major Damage, Destroyed, Unknown"
                },
                "disaster_id": {
                    "type": "string",
                    "description": "Filter by disaster ID e.g. 'hurricane-florence'. Optional."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results to return. Default 10."
                }
            },
            "required": ["damage_level"]
        }
    },
    {
        "name": "get_damage_hotspots",
        "description": (
            "Find disaster-scoped hotspot areas ranked by Major Damage plus Destroyed building counts. "
            "Use when the user asks for the worst-hit, most damaged, or hardest-hit area."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "disaster_id": {
                    "type": "string",
                    "description": "Filter by disaster ID e.g. 'hurricane-florence'. Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of hotspot areas to return. Default 5.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_disaster_list",
        "description": (
            "Get a list of all disasters in the database with their types. "
            "Use when the user asks what disasters are available, or wants to compare events."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_assessment_description",
        "description": (
            "Get the VLM damage description and Gemini humanized summary for a specific location. "
            "Use when the user asks what happened at a specific house or location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location_id": {
                    "type": "integer",
                    "description": "The location ID (integer) from the locations table."
                }
            },
            "required": ["location_id"]
        }
    },
    {
        "name": "compare_disasters",
        "description": (
            "Compare damage statistics across multiple disasters side by side. "
            "Use when the user asks to compare two or more disaster events."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "disaster_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of disaster IDs to compare e.g. ['hurricane-florence', 'hurricane-michael']"
                }
            },
            "required": ["disaster_ids"]
        }
    },
    {
        "name": "navigate_map",
        "description": (
            "Navigate the user's map view to a specific location. "
            "Use when the user asks to see an area, go somewhere, or when showing them relevant damage. "
            "For most-damaged areas, call get_damage_hotspots first and navigate to one of its returned coordinates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lng": {"type": "number", "description": "Longitude"},
                "zoom": {
                    "type": "integer",
                    "description": "Map zoom level, 15=neighborhood, 17=building detail, 18=max detail"
                }
            },
            "required": ["lat", "lng"]
        }
    },
    {
        "name": "set_overlay_opacity",
        "description": (
            "Adjust the satellite image overlay transparency. "
            "Use when the user asks to make images more/less transparent, or to see the base map better."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "opacity": {"type": "number", "description": "Opacity from 0.0 (fully transparent) to 1.0 (fully opaque)"}
            },
            "required": ["opacity"]
        }
    },
    {
        "name": "set_overlay_mode",
        "description": (
            "Switch the satellite imagery view. "
            "'pre' shows before the disaster, 'post' shows after, 'none' hides satellite overlays entirely."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["pre", "post", "none"], "description": "Satellite overlay mode"}
            },
            "required": ["mode"]
        }
    },
    {
        "name": "set_classification_filter",
        "description": (
            "Control which building damage classifications are visible on the map. "
            "Set to true to show, false to hide. Only include the classifications you want to change. "
            "Accepts FEMA labels (No Damage, Minor Damage, Major Damage, Destroyed, Unknown) "
            "or short aliases (none, minor, severe, destroyed, unknown)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "destroyed": {"type": "boolean", "description": "Show/hide destroyed buildings"},
                "major-damage": {"type": "boolean", "description": "Show/hide buildings with major damage"},
                "minor-damage": {"type": "boolean", "description": "Show/hide buildings with minor damage"},
                "no-damage": {"type": "boolean", "description": "Show/hide undamaged buildings"},
                "severe": {"type": "boolean", "description": "Alias for major-damage"},
                "minor": {"type": "boolean", "description": "Alias for minor-damage"},
                "none": {"type": "boolean", "description": "Alias for no-damage"},
                "unknown": {"type": "boolean", "description": "Show/hide unknown classification buildings"}
            },
            "required": []
        }
    },
    {
        "name": "get_damage_by_area",
        "description": (
            "Get aggregate damage level counts for a geographic area (city, county, or street). "
            "Use when the user asks about overall/average/total damage in a city, county, town, or street. "
            "Examples: 'how bad is Pender County?', 'damage in Burgaw', 'what's the damage on Main St?'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "area_type": {
                    "type": "string",
                    "enum": ["city", "county", "street"],
                    "description": "The type of geographic area to aggregate by.",
                },
                "area_name": {
                    "type": "string",
                    "description": "The name of the area to search for (e.g. 'Burgaw', 'Pender', 'Main St').",
                },
                "disaster_id": {
                    "type": "string",
                    "description": "Filter by disaster ID e.g. 'hurricane-florence'. Optional.",
                },
            },
            "required": ["area_type", "area_name"],
        },
    },
    {
        "name": "lookup_damage_at_address",
        "description": (
            "Fuzzy-match an address, street name, house, or neighborhood/block string "
            "and return damage aggregates for each matching group. "
            "Use when the user asks about damage at a specific street, address, or neighborhood."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Street name, address, or neighborhood text to match (min 4 chars).",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "nearby_damage",
        "description": (
            "Aggregate damage counts within a radius of a lat/lng point. "
            "Use when the user asks what's damaged near a coordinate, point of interest, or block."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude of the center point"},
                "lng": {"type": "number", "description": "Longitude of the center point"},
                "radius_m": {
                    "type": "integer",
                    "description": "Search radius in meters. Clamped to [20, 5000]. Default 200.",
                },
            },
            "required": ["lat", "lng"],
        },
    },
]


# ── Tool execution (runs actual SQL) ─────────────────────────────────

# Canonical damage classification → short alias used by the frontend filter UI.
_CLASSIFICATION_ALIAS = {
    "no-damage": "No Damage",
    "minor-damage": "Minor Damage",
    "major-damage": "Major Damage",
    "destroyed": "Destroyed",
    "unknown": "Unknown",
    # Short aliases map to FEMA labels so Gemini can use either vocabulary.
    "none": "No Damage",
    "minor": "Minor Damage",
    "severe": "Major Damage",
    # FEMA labels pass through unchanged.
    "No Damage": "No Damage",
    "Minor Damage": "Minor Damage",
    "Major Damage": "Major Damage",
    "Destroyed": "Destroyed",
    "Unknown": "Unknown",
}

_DAMAGE_LABELS = {
    "No Damage": "no visible damage",
    "Minor Damage": "minor damage",
    "Major Damage": "major damage",
    "Destroyed": "destroyed",
    "Unknown": "unknown damage",
}

_DEFAULT_SCOPED_TOOLS = {
    "get_damage_stats",
    "get_locations_by_damage",
    "get_damage_hotspots",
    "get_damage_by_area",
    "lookup_damage_at_address",
    "nearby_damage",
    "navigate_map",
}

_HOTSPOT_MIN_CLUSTER_SIZE = 10

_EFFECTIVE_DAMAGE_SQL = """COALESCE(
    CASE a.damage_level
        WHEN 'no-damage' THEN 'No Damage'
        WHEN 'minor-damage' THEN 'Minor Damage'
        WHEN 'major-damage' THEN 'Major Damage'
        WHEN 'destroyed' THEN 'Destroyed'
        WHEN 'unknown' THEN 'Unknown'
        ELSE a.damage_level
    END,
    CASE l.classification
        WHEN 'none' THEN 'No Damage'
        WHEN 'minor' THEN 'Minor Damage'
        WHEN 'severe' THEN 'Major Damage'
        WHEN 'destroyed' THEN 'Destroyed'
        WHEN 'unknown' THEN 'Unknown'
        ELSE l.classification
    END,
    'Unknown'
)"""


def _normalize_classification_filter(args: dict) -> dict:
    """Accept both canonical and short classification keys, emit FEMA labels."""
    out: dict = {}
    for key, value in args.items():
        alias = _CLASSIFICATION_ALIAS.get(key)
        if alias is not None:
            out[alias] = bool(value)
    return out


def _dominant_match(matches: list[dict]) -> dict | None:
    """Return the top match if it's the only one or clearly dominates the runner-up."""
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    top, runner = matches[0], matches[1]
    top_score = float(top.get("score") or 0.0)
    runner_score = float(runner.get("score") or 0.0)
    # Dominant when the top score is meaningfully larger than the next candidate.
    if top_score - runner_score >= 0.2:
        return top
    return None


def _with_default_disaster(
    tool_name: str, args: dict, default_disaster_id: str | None
) -> dict:
    scoped = dict(args)
    if (
        default_disaster_id
        and tool_name in _DEFAULT_SCOPED_TOOLS
        and not scoped.get("disaster_id")
    ):
        scoped["disaster_id"] = default_disaster_id
    return scoped


def _format_place(match: dict) -> str:
    parts = [
        str(match.get("street") or "").strip(),
        str(match.get("city") or "").strip(),
        str(match.get("county") or "").strip(),
    ]
    label = ", ".join(part for part in parts if part)
    return label or "that area"


def _count_phrase(count: int, label: str) -> str:
    unit = "building" if count == 1 else "buildings"
    return f"{count} {label} {unit}"


def _summarize_damage_counts(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    if total <= 0:
        return "I couldn't find any assessed buildings for that query."
    ordered = ["Destroyed", "Major Damage", "Minor Damage", "No Damage", "Unknown"]
    details = [
        _count_phrase(counts[level], _DAMAGE_LABELS[level])
        for level in ordered
        if counts.get(level, 0) > 0
    ]
    return f"I found {total} assessed buildings: {', '.join(details)}."


def _counts_from_rows(rows: list[dict]) -> dict[str, int]:
    counts = {level: 0 for level in _DAMAGE_LABELS}
    for row in rows:
        level = str(row.get("damage_level") or row.get("level") or "Unknown")
        key = _CLASSIFICATION_ALIAS.get(level, level)
        if key not in counts:
            key = "Unknown"
        counts[key] += int(row.get("count") or row.get("n") or 0)
    return counts


def _synthesize_reply_from_tool_results(
    tool_results: list[dict], actions: list[dict]
) -> str:
    """Build a data-aware fallback when Gemini returns only tool calls."""
    for item in reversed(tool_results):
        name = item["name"]
        data = item["result"]

        if name == "get_damage_stats":
            rows = data.get("result") if isinstance(data, dict) else None
            if isinstance(rows, list):
                return _summarize_damage_counts(_counts_from_rows(rows))

        if name == "get_damage_by_area":
            rows = data.get("result") if isinstance(data, dict) else None
            area_name = data.get("area_name") or "that area"
            area_type = data.get("area_type") or "area"
            if isinstance(rows, list) and rows:
                counts = _counts_from_rows(rows)
                total = sum(counts.values())
                if total <= 0:
                    return f"I couldn't find any assessed buildings in {area_name}."
                ordered = ["Destroyed", "Major Damage", "Minor Damage", "No Damage", "Unknown"]
                details = [
                    _count_phrase(counts[level], _DAMAGE_LABELS[level])
                    for level in ordered
                    if counts.get(level, 0) > 0
                ]
                return f"In {area_name} ({area_type}), I found {total} assessed buildings: {', '.join(details)}."
            return f"I couldn't find any assessed buildings in {area_name}."

        if name == "lookup_damage_at_address":
            matches = data.get("matches") or []
            if not matches:
                return "I couldn't find damage records for that address or street."
            top = matches[0]
            place = _format_place(top)
            counts = {
                "Destroyed": int(top.get("destroyed") or 0),
                "Major Damage": int(top.get("severe") or 0),
            }
            total = int(top.get("total") or 0)
            detail = ", ".join(
                _count_phrase(counts[level], _DAMAGE_LABELS[level])
                for level in ("Destroyed", "Major Damage")
                if counts[level] > 0
            )
            if detail:
                return f"{place} has {total} assessed buildings, including {detail}."
            return f"{place} has {total} assessed buildings with no Major Damage or Destroyed buildings found."

        if name == "nearby_damage":
            counts = {
                "No Damage": int(data.get("none") or 0),
                "Minor Damage": int(data.get("minor") or 0),
                "Major Damage": int(data.get("severe") or 0),
                "Destroyed": int(data.get("destroyed") or 0),
                "Unknown": int(data.get("unknown") or 0),
            }
            return _summarize_damage_counts(counts)

        if name == "get_damage_hotspots":
            hotspots = data.get("hotspots") or []
            if not hotspots:
                return "I couldn't find a damaged-area hotspot for the selected disaster."
            top = hotspots[0]
            severe = int(top.get("severe") or 0)
            destroyed = int(top.get("destroyed") or 0)
            lat = float(top["lat"])
            lng = float(top["lng"])
            return (
                f"The most damaged area is near {lat:.2f}, {lng:.2f}, with "
                f"{severe} major-damage and {destroyed} destroyed buildings."
            )

        if name == "get_locations_by_damage":
            rows = data.get("result") if isinstance(data, dict) else None
            if isinstance(rows, list) and rows:
                first = rows[0]
                level = str(first.get("damage_level") or "matching")
                lat = float(first["lat"])
                lng = float(first["lng"])
                return (
                    f"I found {len(rows)} {level} locations; the first is near "
                    f"{lat:.2f}, {lng:.2f}."
                )
            return "I couldn't find locations matching that damage class."

        if name == "navigate_map" and data.get("status") == "out_of_scope":
            return "I can only move the map to locations inside the selected disaster area."

    return _synthesize_reply_from_actions(actions)


def _synthesize_reply_from_actions(actions: list[dict]) -> str:
    """Build a short natural-language summary when Gemini returns no text."""
    if not actions:
        return "Done."
    summaries: list[str] = []
    for action in actions:
        kind = action.get("type")
        if kind == "flyTo":
            summaries.append("Moving the map to that location.")
        elif kind == "setOpacity":
            pct = int(round(float(action.get("value", 0)) * 100))
            summaries.append(f"Overlay opacity set to {pct}%.")
        elif kind == "setOverlayMode":
            mode = action.get("mode", "")
            label = {"pre": "pre-disaster", "post": "post-disaster", "none": "no"}.get(
                mode, mode
            )
            summaries.append(f"Switched to {label} overlay.")
        elif kind == "setFilters":
            summaries.append("Damage filters updated.")
    return " ".join(summaries) if summaries else "Done."


def _point_within_disaster_bounds(
    conn, lat: float, lng: float, disaster_id: str | None
) -> bool:
    if not disaster_id:
        return True
    row = (
        conn.execute(
            text("""
                SELECT
                    MIN(ST_Y(l.centroid)) AS min_lat,
                    MAX(ST_Y(l.centroid)) AS max_lat,
                    MIN(ST_X(l.centroid)) AS min_lng,
                    MAX(ST_X(l.centroid)) AS max_lng
                FROM locations l
                JOIN image_pairs ip ON ip.id = l.image_pair_id
                WHERE ip.disaster_id = :disaster_id
            """),
            {"disaster_id": disaster_id},
        )
        .mappings()
        .all()
    )
    if not row:
        return False
    bounds = row[0]
    if bounds["min_lat"] is None or bounds["min_lng"] is None:
        return False
    return (
        float(bounds["min_lat"]) <= lat <= float(bounds["max_lat"])
        and float(bounds["min_lng"]) <= lng <= float(bounds["max_lng"])
    )


def _run_tool(
    tool_name: str, args: dict, *, default_disaster_id: str | None = None
) -> str:
    """Executes the requested tool and returns a JSON string result."""
    engine = get_engine()
    args = _with_default_disaster(tool_name, args, default_disaster_id)

    with engine.connect() as conn:

        if tool_name == "get_damage_stats":
            query = f"""
                SELECT
                    {_EFFECTIVE_DAMAGE_SQL} AS damage_level,
                    COUNT(*) AS count
                FROM locations l
                JOIN image_pairs ip ON ip.id = l.image_pair_id
                LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
            """
            params = {}
            if args.get("disaster_id"):
                query += " WHERE ip.disaster_id = :disaster_id"
                params["disaster_id"] = args["disaster_id"]
            query += " GROUP BY 1 ORDER BY count DESC"
            rows = conn.execute(text(query), params).mappings().all()
            return json.dumps([dict(r) for r in rows])

        elif tool_name == "get_damage_hotspots":
            query = f"""
                SELECT
                    round(ST_Y(l.centroid)::numeric, 2) AS lat,
                    round(ST_X(l.centroid)::numeric, 2) AS lng,
                    COUNT(*) FILTER (WHERE {_EFFECTIVE_DAMAGE_SQL} = 'Major Damage') AS severe,
                    COUNT(*) FILTER (WHERE {_EFFECTIVE_DAMAGE_SQL} = 'Destroyed') AS destroyed,
                    COUNT(*) AS total
                FROM locations l
                JOIN image_pairs ip ON ip.id = l.image_pair_id
                LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
                WHERE l.centroid IS NOT NULL
            """
            params: dict = {"limit": min(max(int(args.get("limit", 5)), 1), 20)}
            if args.get("disaster_id"):
                query += " AND ip.disaster_id = :disaster_id"
                params["disaster_id"] = args["disaster_id"]
            query += f"""
                GROUP BY lat, lng
                HAVING (
                    COUNT(*) FILTER (WHERE {_EFFECTIVE_DAMAGE_SQL} = 'Major Damage')
                    + COUNT(*) FILTER (WHERE {_EFFECTIVE_DAMAGE_SQL} = 'Destroyed')
                ) > 0
                AND COUNT(*) >= {_HOTSPOT_MIN_CLUSTER_SIZE}
                ORDER BY
                    (
                        COUNT(*) FILTER (WHERE {_EFFECTIVE_DAMAGE_SQL} = 'Major Damage')
                        + COUNT(*) FILTER (WHERE {_EFFECTIVE_DAMAGE_SQL} = 'Destroyed')
                    ) DESC,
                    COUNT(*) FILTER (WHERE {_EFFECTIVE_DAMAGE_SQL} = 'Destroyed') DESC
                LIMIT :limit
            """
            rows = conn.execute(text(query), params).mappings().all()
            hotspots = [
                {
                    "lat": float(row["lat"]),
                    "lng": float(row["lng"]),
                    "severe": int(row["severe"] or 0),
                    "destroyed": int(row["destroyed"] or 0),
                    "total": int(row["total"] or 0),
                }
                for row in rows
            ]
            return json.dumps({"hotspots": hotspots})

        elif tool_name == "get_locations_by_damage":
            query = f"""
                SELECT
                    l.id AS location_id,
                    l.location_uid,
                    l.image_pair_id,
                    ip.disaster_id,
                    {_EFFECTIVE_DAMAGE_SQL} AS damage_level,
                    a.description,
                    ST_Y(l.centroid) AS lat,
                    ST_X(l.centroid) AS lng
                FROM locations l
                JOIN image_pairs ip ON ip.id = l.image_pair_id
                LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
                WHERE {_EFFECTIVE_DAMAGE_SQL} = :damage_level
            """
            damage_level = _CLASSIFICATION_ALIAS.get(
                str(args["damage_level"]), str(args["damage_level"])
            )
            params: dict = {"damage_level": damage_level}
            if args.get("disaster_id"):
                query += " AND ip.disaster_id = :disaster_id"
                params["disaster_id"] = args["disaster_id"]
            query += " LIMIT :limit"
            params["limit"] = args.get("limit", 10)
            rows = conn.execute(text(query), params).mappings().all()
            return json.dumps([dict(r) for r in rows])

        elif tool_name == "get_disaster_list":
            rows = conn.execute(
                text("SELECT id, type FROM disasters ORDER BY id")
            ).mappings().all()
            return json.dumps([dict(r) for r in rows])

        elif tool_name == "get_assessment_description":
            row = conn.execute(
                text("""
                    SELECT
                        l.id AS location_id,
                        l.location_uid,
                        ip.disaster_id,
                        a.damage_level,
                        a.description AS vlm_description,
                        a.confidence,
                        m.content AS humanized_summary
                    FROM locations l
                    JOIN image_pairs ip ON ip.id = l.image_pair_id
                    LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
                    LEFT JOIN chat.messages m
                        ON m.assessment_id = a.id
                        AND m.role = 'assistant'
                        AND m.is_humanized = true
                    WHERE l.id = :location_id
                    ORDER BY m.turn_index DESC
                    LIMIT 1
                """),
                {"location_id": args["location_id"]}
            ).mappings().one_or_none()
            if not row:
                return json.dumps({"error": "Location not found"})
            return json.dumps(dict(row))

        elif tool_name == "compare_disasters":
            rows = conn.execute(
                text(f"""
                    SELECT
                        ip.disaster_id,
                        {_EFFECTIVE_DAMAGE_SQL} AS damage_level,
                        COUNT(*) AS count
                    FROM locations l
                    JOIN image_pairs ip ON ip.id = l.image_pair_id
                    LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
                    WHERE ip.disaster_id = ANY(:ids)
                    GROUP BY 1, 2
                    ORDER BY 1, count DESC
                """),
                {"ids": args["disaster_ids"]}
            ).mappings().all()
            return json.dumps([dict(r) for r in rows])

        elif tool_name == "navigate_map":
            lat = float(args["lat"])
            lng = float(args["lng"])
            zoom = args.get("zoom", 17)
            disaster_id = args.get("disaster_id")
            if not _point_within_disaster_bounds(conn, lat, lng, disaster_id):
                return json.dumps({
                    "status": "out_of_scope",
                    "lat": lat,
                    "lng": lng,
                    "disaster_id": disaster_id,
                })
            return json.dumps({
                "status": "navigating",
                "lat": lat,
                "lng": lng,
                "zoom": zoom,
            })

        elif tool_name == "set_overlay_opacity":
            opacity = max(0.0, min(1.0, float(args["opacity"])))
            return json.dumps({"status": "ok", "opacity": opacity})

        elif tool_name == "set_overlay_mode":
            mode = args["mode"] if args.get("mode") in ("pre", "post", "none") else "post"
            return json.dumps({"status": "ok", "mode": mode})

        elif tool_name == "set_classification_filter":
            sanitized = _normalize_classification_filter(args)
            return json.dumps({"status": "ok", "filters": sanitized})

        elif tool_name == "get_damage_by_area":
            area_type = str(args.get("area_type") or "city")
            area_name = str(args.get("area_name") or "")
            if not area_name.strip():
                return json.dumps({"error": "area_name is required"})
            column_map = {"city": "l.city", "county": "l.county", "street": "l.street"}
            column = column_map.get(area_type)
            if column is None:
                return json.dumps({"error": f"Invalid area_type: must be one of city, county, street"})
            query = f"""
                SELECT
                    {_EFFECTIVE_DAMAGE_SQL} AS damage_level,
                    COUNT(*) AS count
                FROM locations l
                JOIN image_pairs ip ON ip.id = l.image_pair_id
                LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
                WHERE {column} ILIKE :area_pattern
            """
            params: dict = {"area_pattern": f"%{area_name}%"}
            if args.get("disaster_id"):
                query += " AND ip.disaster_id = :disaster_id"
                params["disaster_id"] = args["disaster_id"]
            query += " GROUP BY 1 ORDER BY count DESC"
            rows = conn.execute(text(query), params).mappings().all()
            return json.dumps({
                "area_type": area_type,
                "area_name": area_name,
                "result": [dict(r) for r in rows],
            })

        elif tool_name == "lookup_damage_at_address":
            query = str(args.get("query") or "")
            matches = lookup_damage_at_address(
                conn, query, disaster_id=args.get("disaster_id")
            )
            return json.dumps(
                {
                    "matches": [
                        {
                            "street": m.street,
                            "city": m.city,
                            "county": m.county,
                            "total": m.total,
                            "severe": m.severe,
                            "destroyed": m.destroyed,
                            "lat": m.lat,
                            "lng": m.lng,
                            "score": m.score,
                        }
                        for m in matches
                    ]
                }
            )

        elif tool_name == "nearby_damage":
            try:
                lat = float(args["lat"])
                lng = float(args["lng"])
            except (KeyError, TypeError, ValueError):
                return json.dumps({"error": "lat and lng are required numeric fields"})
            if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lng <= 180.0):
                return json.dumps({"error": "lat/lng out of valid geographic range"})
            radius_m = int(args.get("radius_m", 200))
            agg = nearby_damage(
                conn, lat, lng, radius_m=radius_m, disaster_id=args.get("disaster_id")
            )
            return json.dumps(
                {
                    "none": agg.none,
                    "minor": agg.minor,
                    "severe": agg.severe,
                    "destroyed": agg.destroyed,
                    "unknown": agg.unknown,
                }
            )

    return json.dumps({"error": "Unknown tool"})


# ── Main chat function ───────────────────────────────────────────────

SYSTEM_PROMPT = """CRITICAL: If the user's message is not about disaster assessment, building damage, disaster information, hurricane facts, map navigation, overlay/filter controls, or damage at a specific address, street, house, neighborhood, or block, respond ONLY with 'I can only help with disaster assessment and map navigation.' Do NOT call any tools for off-topic messages.

You are a disaster assessment tool. You control a map interface showing building damage from satellite imagery.

Rules:
- Be concise. One sentence confirmations for actions. Two to three sentences max for data queries.
- Never editorialize or add emotional commentary.
- Use the tools to fetch data before answering questions about damage.
- When navigating the map, just confirm the action briefly.
- When adjusting overlays or filters, just confirm what changed.
- Only respond to queries about disaster assessment, disaster information, hurricane facts, map navigation, overlays, filters, building damage data, and damage at a specific address, street, house, neighborhood, or block.
- For unrelated questions, say: "I can only help with disaster assessment and map navigation."
- To find or navigate to the most damaged/worst-hit areas, call get_damage_hotspots. If the user asks to go there, call navigate_map with a hotspot coordinate.
- To navigate to a specific damaged building class, first call get_locations_by_damage to get coordinates, then call navigate_map with those lat/lng values.
- Use the selected disaster context when present; do not answer from another disaster unless the user explicitly asks for it.
- For aggregate damage counts by city or county (e.g. 'how bad is Pender County?'), call get_damage_by_area. For street-level or address queries where navigation is useful, prefer lookup_damage_at_address.
- For address/street/house/neighborhood/block queries, call lookup_damage_at_address with the user's query. For "what's damaged near here" or a coordinate, call nearby_damage. If no matches come back, say so briefly.

Knowledge:
- Hurricane Florence was a Category 4 hurricane that weakened to Category 1 at landfall.
- It made landfall near Wrightsville Beach, North Carolina on September 14, 2018.
- Florence caused 53 direct deaths and $24.2 billion in damage (2018 USD).
- Record rainfall of 35.93 inches (91.3 cm) was recorded in Elizabethtown, NC.
- Over 150,000 structures were damaged across North and South Carolina.
- Storm surge flooding reached up to 10 feet in some coastal areas.
- The xBD dataset used in this tool covers the Myrtle Beach, SC area and surrounding regions.
- Damage classifications in this system: No Damage, Minor Damage, Major Damage, Destroyed, Unknown.
"""


def chat(
    user_message: str,
    history: list[dict] | None = None,
    viewport: dict | None = None,
    disaster_id: str | None = None,
    disaster_name: str | None = None,
) -> tuple[str, list[dict], list[dict]]:
    history = history or []
    actions: list[dict] = []
    tool_results: list[dict] = []

    # Prepend UI context so Gemini knows what area/disaster the user means.
    effective_message = user_message
    context_lines: list[str] = []
    if disaster_id:
        context = f"[Selected disaster: {disaster_id}"
        if disaster_name:
            context += f" ({disaster_name})"
        context += "]"
        context_lines.append(context)
    if viewport:
        context_lines.append(
            f"[User is viewing: lat {viewport['minLat']:.2f}-{viewport['maxLat']:.2f}, "
            f"lng {viewport['minLng']:.2f} to {viewport['maxLng']:.2f}]"
        )
    if context_lines:
        effective_message = "\n".join(context_lines) + "\n" + user_message

    # Build contents from history + new message
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=p) for p in msg["parts"]]
        ))
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=effective_message)]
    ))

    # Convert TOOLS list to Gemini SDK tool declarations
    tool_declarations = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
        )
        for t in TOOLS
    ])

    client = _get_client()

    # Tool-calling loop: Gemini may call tools, we execute them and feed results back
    max_rounds = 5
    for _ in range(max_rounds):
        try:
            response = client.models.generate_content(
                # flash-lite handles our tool-calling loop reliably; the full
                # flash model intermittently returns 503 UNAVAILABLE under load.
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[tool_declarations],
                ),
            )
        except ClientError as exc:
            message = str(exc)
            status_code = getattr(exc, "status_code", 503)
            if _is_auth_error(exc):
                _invalidate_client()
                logger.error("Gemini API key rejected (HTTP %s)", status_code)
                raise ChatBackendUnavailableError(
                    status_code=503,
                    detail="The Gemini API key is invalid or expired. Please check your configuration.",
                ) from exc
            if status_code == 429 or "RESOURCE_EXHAUSTED" in message:
                raise ChatBackendUnavailableError(
                    status_code=503,
                    retry_after_seconds=_extract_retry_after_seconds(message),
                ) from exc
            raise ChatBackendUnavailableError(status_code=503) from exc
        except ChatBackendUnavailableError:
            raise
        except Exception as exc:
            raise ChatBackendUnavailableError(status_code=503) from exc

        # Collect any function calls from the response
        function_calls = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            function_calls.append(part.function_call)

        # If no function calls, Gemini is done — extract the text reply
        if not function_calls:
            break

        # Append the model's function-call response to the conversation
        contents.append(response.candidates[0].content)

        # Execute each tool and build function-response parts
        fn_response_parts = []
        for fc in function_calls:
            tool_name = fc.name
            fc_args = dict(fc.args) if fc.args else {}
            scoped_args = _with_default_disaster(tool_name, fc_args, disaster_id)

            result_str = _run_tool(
                tool_name,
                scoped_args,
                default_disaster_id=disaster_id,
            )
            result_data = json.loads(result_str)
            if not isinstance(result_data, dict):
                result_data = {"result": result_data}
            tool_results.append(
                {"name": tool_name, "args": scoped_args, "result": result_data}
            )

            if tool_name == "navigate_map" and result_data.get("status") == "navigating":
                actions.append({
                    "type": "flyTo",
                    "lat": float(result_data["lat"]),
                    "lng": float(result_data["lng"]),
                    "zoom": int(result_data.get("zoom", 17)),
                })
            elif tool_name == "set_overlay_opacity" and result_data.get("status") == "ok":
                actions.append({
                    "type": "setOpacity",
                    "value": float(result_data["opacity"]),
                })
            elif tool_name == "set_overlay_mode" and result_data.get("status") == "ok":
                actions.append({"type": "setOverlayMode", "mode": result_data["mode"]})
            elif tool_name == "set_classification_filter" and result_data.get("status") == "ok":
                filters = result_data.get("filters") or {}
                if filters:
                    actions.append({"type": "setFilters", **filters})

            # Synthetic flyTo when an address lookup resolves to a single or
            # clearly dominant match — let the map follow the answer.
            if tool_name == "lookup_damage_at_address":
                matches = result_data.get("matches") or []
                top = _dominant_match(matches)
                if top is not None and top.get("lat") is not None and top.get("lng") is not None:
                    actions.append({
                        "type": "flyTo",
                        "lat": float(top["lat"]),
                        "lng": float(top["lng"]),
                        "zoom": 17,
                    })
            elif tool_name == "get_damage_hotspots":
                hotspots = result_data.get("hotspots") or []
                if hotspots:
                    top = hotspots[0]
                    actions.append({
                        "type": "flyTo",
                        "lat": float(top["lat"]),
                        "lng": float(top["lng"]),
                        "zoom": 15,
                    })

            fn_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response=result_data,
                )
            )

        # Send tool results back to Gemini
        contents.append(types.Content(
            role="user",
            parts=fn_response_parts,
        ))

    # Extract final text reply
    reply = ""
    if response.candidates:
        for part in (response.candidates[0].content.parts or []):
            if hasattr(part, "text") and part.text:
                reply += part.text
    if not reply or (tool_results and reply.strip().lower() in {"done", "done."}):
        # Gemini sometimes returns tool calls with no text summary.
        # Synthesize a reply from the actions so the user always sees
        # a meaningful response, not a sterile fallback.
        reply = _synthesize_reply_from_tool_results(tool_results, actions)

    updated_history = history + [
        {"role": "user", "parts": [user_message]},
        {"role": "model", "parts": [reply]},
    ]

    return reply, updated_history, actions
