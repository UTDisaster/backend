from __future__ import annotations

import os
import json
import re
from functools import lru_cache
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from sqlalchemy import text

from app.db import get_engine

load_dotenv()


class ChatBackendUnavailableError(Exception):
    def __init__(
        self, status_code: int = 503, retry_after_seconds: int | None = None
    ) -> None:
        super().__init__("Chat backend unavailable")
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def _extract_retry_after_seconds(message: str) -> int | None:
    retry_in = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if retry_in:
        return max(1, int(float(retry_in.group(1))))
    retry_delay = re.search(r"'retryDelay': '([0-9]+)s'", message, re.IGNORECASE)
    if retry_delay:
        return max(1, int(retry_delay.group(1)))
    return None


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ChatBackendUnavailableError(status_code=503)
    return genai.Client(api_key=api_key)
# ── Tool definitions (Gemini decides when to call these) ─────────────

TOOLS = [
    {
        "name": "get_damage_stats",
        "description": (
            "Get damage statistics for a disaster. Returns counts of buildings "
            "by damage classification (no-damage, minor-damage, major-damage, destroyed, unknown). "
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
            "Use when the user asks about specific houses, worst damaged areas, or wants to see locations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "damage_level": {
                    "type": "string",
                    "description": "One of: no-damage, minor-damage, major-damage, destroyed, unknown"
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
            "Use when the user asks to see an area, go somewhere, or when showing them relevant damage."
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
            "Accepts canonical names (no-damage, minor-damage, major-damage, destroyed, unknown) "
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
    }
]


# ── Tool execution (runs actual SQL) ─────────────────────────────────

# Canonical damage classification → short alias used by the frontend filter UI.
_CLASSIFICATION_ALIAS = {
    "no-damage": "none",
    "minor-damage": "minor",
    "major-damage": "severe",
    "destroyed": "destroyed",
    "unknown": "unknown",
    # Short aliases pass through unchanged so Gemini can use either vocabulary.
    "none": "none",
    "minor": "minor",
    "severe": "severe",
}


def _normalize_classification_filter(args: dict) -> dict:
    """Accept both canonical and short classification keys, emit short aliases."""
    out: dict = {}
    for key, value in args.items():
        alias = _CLASSIFICATION_ALIAS.get(key)
        if alias is not None:
            out[alias] = bool(value)
    return out


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


def _run_tool(tool_name: str, args: dict) -> str:
    """Executes the requested tool and returns a JSON string result."""
    engine = get_engine()

    with engine.connect() as conn:

        if tool_name == "get_damage_stats":
            query = """
                SELECT
                    COALESCE(
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
                    ) AS damage_level,
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

        elif tool_name == "get_locations_by_damage":
            query = """
                SELECT
                    l.id AS location_id,
                    l.location_uid,
                    l.image_pair_id,
                    ip.disaster_id,
                    COALESCE(
                        CASE a.damage_level
                            WHEN 'no-damage' THEN 'none'
                            WHEN 'minor-damage' THEN 'minor'
                            WHEN 'major-damage' THEN 'severe'
                            WHEN 'destroyed' THEN 'destroyed'
                            WHEN 'unknown' THEN 'unknown'
                            ELSE a.damage_level
                        END,
                        l.classification
                    ) AS damage_level,
                    a.description,
                    ST_Y(l.centroid) AS lat,
                    ST_X(l.centroid) AS lng
                FROM locations l
                JOIN image_pairs ip ON ip.id = l.image_pair_id
                LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
                WHERE COALESCE(
                    CASE a.damage_level
                        WHEN 'no-damage' THEN 'none'
                        WHEN 'minor-damage' THEN 'minor'
                        WHEN 'major-damage' THEN 'severe'
                        WHEN 'destroyed' THEN 'destroyed'
                        WHEN 'unknown' THEN 'unknown'
                        ELSE a.damage_level
                    END,
                    l.classification
                ) = :damage_level
            """
            params: dict = {"damage_level": args["damage_level"]}
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
                text("""
                    SELECT
                        ip.disaster_id,
                        COALESCE(
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
                    ) AS damage_level,
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
            lat = args["lat"]
            lng = args["lng"]
            zoom = args.get("zoom", 17)
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

    return json.dumps({"error": "Unknown tool"})


# ── Main chat function ───────────────────────────────────────────────

SYSTEM_PROMPT = """CRITICAL: If the user's message is not about disaster assessment, building damage, disaster information, hurricane facts, map navigation, or overlay/filter controls, respond ONLY with 'I can only help with disaster assessment and map navigation.' Do NOT call any tools for off-topic messages.

You are a disaster assessment tool. You control a map interface showing building damage from satellite imagery.

Rules:
- Be concise. One sentence confirmations for actions. Two to three sentences max for data queries.
- Never editorialize or add emotional commentary.
- Use the tools to fetch data before answering questions about damage.
- When navigating the map, just confirm the action briefly.
- When adjusting overlays or filters, just confirm what changed.
- Only respond to queries about disaster assessment, disaster information, hurricane facts, map navigation, overlays, filters, and building damage data.
- For unrelated questions, say: "I can only help with disaster assessment and map navigation."
- To navigate to damaged areas: first call get_locations_by_damage to get coordinates, then call navigate_map with those lat/lng values.

Knowledge:
- Hurricane Florence was a Category 4 hurricane that weakened to Category 1 at landfall.
- It made landfall near Wrightsville Beach, North Carolina on September 14, 2018.
- Florence caused 53 direct deaths and $24.2 billion in damage (2018 USD).
- Record rainfall of 35.93 inches (91.3 cm) was recorded in Elizabethtown, NC.
- Over 150,000 structures were damaged across North and South Carolina.
- Storm surge flooding reached up to 10 feet in some coastal areas.
- The xBD dataset used in this tool covers the Myrtle Beach, SC area and surrounding regions.
- Damage classifications in this system: no-damage, minor-damage, major-damage, destroyed, unknown.
"""


def chat(
    user_message: str,
    history: list[dict] | None = None,
    viewport: dict | None = None,
) -> tuple[str, list[dict], list[dict]]:
    history = history or []
    actions: list[dict] = []

    # Prepend viewport context so Gemini knows what area the user sees
    effective_message = user_message
    if viewport:
        effective_message = (
            f"[User is viewing: lat {viewport['minLat']:.2f}-{viewport['maxLat']:.2f}, "
            f"lng {viewport['minLng']:.2f} to {viewport['maxLng']:.2f}]\n"
            + user_message
        )

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

            # Collect navigate_map calls as frontend actions (no DB needed)
            if tool_name == "navigate_map":
                lat = fc_args.get("lat")
                lng = fc_args.get("lng")
                zoom = fc_args.get("zoom", 17)
                if lat is not None and lng is not None:
                    actions.append({
                        "type": "flyTo",
                        "lat": float(lat),
                        "lng": float(lng),
                        "zoom": int(zoom),
                    })
            elif tool_name == "set_overlay_opacity":
                clamped = max(0.0, min(1.0, float(fc_args["opacity"])))
                actions.append({"type": "setOpacity", "value": clamped})
            elif tool_name == "set_overlay_mode":
                mode = fc_args.get("mode")
                if mode in ("pre", "post", "none"):
                    actions.append({"type": "setOverlayMode", "mode": mode})
            elif tool_name == "set_classification_filter":
                sanitized = _normalize_classification_filter(fc_args)
                if sanitized:
                    actions.append({"type": "setFilters", **sanitized})

            result_str = _run_tool(tool_name, fc_args)
            result_data = json.loads(result_str)
            if not isinstance(result_data, dict):
                result_data = {"result": result_data}
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
    if not reply:
        # Gemini sometimes returns tool calls with no text summary.
        # Synthesize a reply from the actions so the user always sees
        # a meaningful response, not a sterile fallback.
        reply = _synthesize_reply_from_actions(actions)

    updated_history = history + [
        {"role": "user", "parts": [user_message]},
        {"role": "model", "parts": [reply]},
    ]

    return reply, updated_history, actions
