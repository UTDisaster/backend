from __future__ import annotations

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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
    }
]


# ── Tool execution (runs actual SQL) ─────────────────────────────────

def _run_tool(tool_name: str, args: dict) -> str:
    """Executes the requested tool and returns a JSON string result."""
    engine = get_engine()

    with engine.connect() as conn:

        if tool_name == "get_damage_stats":
            query = """
                SELECT
                    COALESCE(a.damage_level, l.classification, 'unknown') AS damage_level,
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
                    COALESCE(a.damage_level, l.classification) AS damage_level,
                    a.description,
                    ST_Y(l.centroid) AS lat,
                    ST_X(l.centroid) AS lng
                FROM locations l
                JOIN image_pairs ip ON ip.id = l.image_pair_id
                LEFT JOIN chat.vlm_assessments a ON a.location_id = l.id
                WHERE COALESCE(a.damage_level, l.classification) = :damage_level
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
                        COALESCE(a.damage_level, l.classification, 'unknown') AS damage_level,
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

    return json.dumps({"error": "Unknown tool"})


# ── Main chat function ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are a disaster assessment assistant with access to a database of 
satellite imagery analysis for natural disasters. You can query real data about building 
damage, disaster events, and specific locations.

When answering questions:
- Use the available tools to fetch real data before responding
- Be specific with numbers and statistics when available
- Be empathetic — this data represents real people's homes
- If asked about a specific location, always fetch its assessment first
- Keep responses clear and concise"""


def chat(
    user_message: str,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    history = history or []

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
        parts=[types.Part(text=user_message)]
    ))

    response = client.models.generate_content(
        model="gemini-2.0-flash",   # ← updated model
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        )
    )

    reply = response.text

    updated_history = history + [
        {"role": "user", "parts": [user_message]},
        {"role": "model", "parts": [reply]}
    ]

    return reply, updated_history