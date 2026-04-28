from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sqlalchemy import text

from app.db import get_engine, news_article_chunks, news_articles

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_CHARS = 4000

SYSTEM_PROMPT = """You are a disaster assessment assistant with access to a database of
satellite imagery analysis for natural disasters. You can query real data about building
damage, disaster events, and specific locations.

When answering questions:
- Use the provided database and imported news context before responding
- Be specific with numbers and statistics when available
- Be empathetic - this data represents real people's homes
- If asked about a specific location, always fetch its assessment first
- If news articles are available, cite them by title and URL and combine them with database facts
- Never claim that news article search is unavailable when imported article context is provided
- If the user explicitly asks for JSON, return strict JSON only with the requested keys
- Otherwise, answer in a natural, friendly style that a real person would write
- Keep responses clear and concise"""


def _trim_text(value: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _trim_history(history: list[dict]) -> list[dict]:
    recent = history[-MAX_HISTORY_MESSAGES:]
    trimmed: list[dict] = []
    for msg in recent:
        trimmed.append(
            {
                "role": msg["role"],
                "parts": [_trim_text(part) for part in msg["parts"]],
            }
        )
    return trimmed


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _wants_json_response(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(
        phrase in lowered
        for phrase in (
            "json",
            "return strict json",
            "only json",
            "structured output",
        )
    )


def _run_tool(tool_name: str, args: dict) -> str:
    """Execute a database-backed tool and return a JSON string."""
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

        if tool_name == "get_locations_by_damage":
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
            params["limit"] = args.get("limit") or 10
            rows = conn.execute(text(query), params).mappings().all()
            return json.dumps([dict(r) for r in rows])

        if tool_name == "get_disaster_list":
            rows = conn.execute(text("SELECT id, type FROM disasters ORDER BY id")).mappings().all()
            return json.dumps([dict(r) for r in rows])

        if tool_name == "get_assessment_description":
            row = conn.execute(
                text(
                    """
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
                    """
                ),
                {"location_id": args["location_id"]},
            ).mappings().one_or_none()
            if not row:
                return json.dumps({"error": "Location not found"})
            return json.dumps(dict(row))

        if tool_name == "compare_disasters":
            rows = conn.execute(
                text(
                    """
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
                    """
                ),
                {"ids": args["disaster_ids"]},
            ).mappings().all()
            return json.dumps([dict(r) for r in rows])

    return json.dumps({"error": "Unknown tool"})


def get_damage_stats(disaster_id: str | None = None) -> str:
    """Get damage counts for a disaster, or across all disasters."""
    return _run_tool("get_damage_stats", {"disaster_id": disaster_id})


def get_locations_by_damage(
    damage_level: str,
    disaster_id: str | None = None,
    limit: int = 10,
) -> str:
    """Get locations filtered by damage level."""
    return _run_tool(
        "get_locations_by_damage",
        {
            "damage_level": damage_level,
            "disaster_id": disaster_id,
            "limit": limit,
        },
    )


def get_disaster_list() -> str:
    """Get the list of disasters in the database."""
    return _run_tool("get_disaster_list", {})


def get_assessment_description(location_id: int) -> str:
    """Get the VLM assessment and humanized summary for a location."""
    return _run_tool("get_assessment_description", {"location_id": location_id})


def compare_disasters(disaster_ids: list[str]) -> str:
    """Compare damage stats across multiple disasters."""
    return _run_tool("compare_disasters", {"disaster_ids": disaster_ids})


def search_news_articles(query: str, disaster_id: str | None = None, limit: int = 5) -> str:
    """Search news articles by query and optionally by disaster."""
    engine = get_engine()
    cleaned_query = query.strip()
    pattern = f"%{cleaned_query}%"

    with engine.connect() as conn:
        sql = """
            SELECT
                a.id,
                a.disaster_id,
                a.source,
                a.title,
                a.url,
                a.published_at,
                COALESCE(a.summary, LEFT(a.content, 300)) AS excerpt
            FROM news_articles a
            LEFT JOIN news_article_chunks c ON c.article_id = a.id
            WHERE 1=1
        """
        params = {"pattern": pattern, "limit": limit}
        if cleaned_query:
            sql += """
                AND (
                    a.title ILIKE :pattern
                    OR COALESCE(a.summary, '') ILIKE :pattern
                    OR a.content ILIKE :pattern
                    OR COALESCE(c.content, '') ILIKE :pattern
                )
            """
        if disaster_id:
            sql += " AND a.disaster_id = :disaster_id"
            params["disaster_id"] = disaster_id
        sql += " GROUP BY a.id, a.disaster_id, a.source, a.title, a.url, a.published_at, a.summary, a.content"
        sql += " ORDER BY a.published_at DESC NULLS LAST, a.id DESC LIMIT :limit"
        rows = conn.execute(text(sql), params).mappings().all()
        return json.dumps([dict(r) for r in rows])


def get_news_article(article_id: int) -> str:
    """Get a full article and its chunks by id."""
    engine = get_engine()

    with engine.connect() as conn:
        article = conn.execute(
            text(
                """
                SELECT id, disaster_id, source, title, url, published_at, summary, content
                FROM news_articles
                WHERE id = :article_id
                """
            ),
            {"article_id": article_id},
        ).mappings().one_or_none()

        if not article:
            return json.dumps({"error": "Article not found"})

        chunks = conn.execute(
            text(
                """
                SELECT chunk_index, content
                FROM news_article_chunks
                WHERE article_id = :article_id
                ORDER BY chunk_index
                """
            ),
            {"article_id": article_id},
        ).mappings().all()

    payload = dict(article)
    payload["chunks"] = [dict(r) for r in chunks]
    return json.dumps(payload)


def _infer_disaster_id(user_message: str) -> str | None:
    lowered = user_message.lower()
    if "florence" in lowered:
        return "hurricane-florence"
    return None


def _build_news_context(user_message: str) -> str | None:
    query = user_message.strip()
    if not query:
        return None

    disaster_id = _infer_disaster_id(query)
    if disaster_id:
        results = json.loads(search_news_articles("", disaster_id=disaster_id, limit=5))
    else:
        results = json.loads(search_news_articles(query, disaster_id=disaster_id, limit=5))
    if not results:
        return None

    lines = [
        "Imported news article context:",
        "Use these sources when answering and cite them by title and URL if relevant.",
    ]
    for article in results:
        lines.append(
            f"- [{article.get('title')}]({article.get('url')})"
            f" | source={article.get('source')}"
            f" | disaster_id={article.get('disaster_id') or ''}"
            f" | excerpt={article.get('excerpt')}"
        )
    return "\n".join(lines)


def _infer_damage_level(user_message: str) -> str | None:
    lowered = user_message.lower()
    for level in ("destroyed", "major-damage", "minor-damage", "no-damage"):
        if level in lowered:
            return level
    return None


def _build_database_context(user_message: str) -> str | None:
    query = user_message.strip()
    if not query:
        return None

    parts: list[str] = []

    if _contains_any(query, ["what type of disasters", "what disasters", "list disasters", "available disasters"]):
        parts.append(f"Disaster list:\n{get_disaster_list()}")

    disaster_id = _infer_disaster_id(query)
    if disaster_id and _contains_any(query, ["damage", "stats", "building", "buildings", "flood", "flooding", "destroyed", "major-damage", "minor-damage"]):
        parts.append(f"Damage stats for {disaster_id}:\n{get_damage_stats(disaster_id)}")

    damage_level = _infer_damage_level(query)
    if disaster_id and damage_level:
        parts.append(
            f"Locations for {disaster_id} with damage level {damage_level}:\n"
            f"{get_locations_by_damage(damage_level=damage_level, disaster_id=disaster_id, limit=10)}"
        )

    if not parts:
        return None

    return "\n\n".join(parts)


def chat(
    user_message: str,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    history = _trim_history(history or [])
    news_context = _build_news_context(user_message)
    database_context = _build_database_context(user_message)
    wants_json = _wants_json_response(user_message)

    contents = []
    response_style = (
        "Return strict JSON only. Do not wrap the JSON in markdown fences. "
        "Use only the keys requested by the user."
        if wants_json
        else (
            "Write a natural, user-friendly answer. Avoid code fences and avoid "
            "turning the response into a list of labels unless the user asked for JSON. "
            "If you cite sources, weave them into the prose or use a short bullet list."
        )
    )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=f"Response style:\n{response_style}")],
        )
    )
    if database_context:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=f"Database context:\n{database_context}")],
            )
        )
    if news_context:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=news_context)],
            )
        )
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=p) for p in msg["parts"]],
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        )
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=512,
        ),
    )

    reply = response.text or ""
    if not reply and news_context:
        response_json = {
            "summary": "Imported articles were found for this question, but Gemini did not return a free-text answer.",
            "key_facts": [],
            "citations": [],
            "news_context": news_context,
        }
        reply = json.dumps(response_json)
    elif not reply:
        raise RuntimeError("Gemini returned an empty response")

    updated_history = history + [
        {"role": "user", "parts": [user_message]},
        {"role": "model", "parts": [reply]},
    ]

    return reply, updated_history
