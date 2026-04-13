from __future__ import annotations

import json
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text

from app.db import get_engine
from app.services.gemini import (
    ChatBackendUnavailableError,
    chat as gemini_chat,
)

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


# ── Pydantic schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:        str
    conversation_id: Optional[int] = None   # None = start new conversation


# ── Helpers ──────────────────────────────────────────────────────────

def _get_history(conn, conversation_id: int) -> list[dict]:
    """Reconstruct Gemini-format history from DB."""
    rows = conn.execute(
        text("""
            SELECT role, content FROM chat.messages
            WHERE conversation_id = :cid
            ORDER BY turn_index
        """),
        {"cid": conversation_id}
    ).mappings().all()

    # Convert DB roles to Gemini roles (assistant → model)
    history = []
    for r in rows:
        role = "model" if r["role"] == "assistant" else r["role"]
        if role in ("user", "model"):
            history.append({"role": role, "parts": [r["content"]]})
    return history


def _save_messages(conn, conversation_id: int, user_msg: str, assistant_reply: str):
    """Save user + assistant turn to DB."""
    # Get next turn index
    result = conn.execute(
        text("SELECT COALESCE(MAX(turn_index), -1) + 1 FROM chat.messages WHERE conversation_id = :cid"),
        {"cid": conversation_id}
    ).scalar()
    next_turn = result or 0

    conn.execute(
        text("""
            INSERT INTO chat.messages
                (conversation_id, turn_index, role, content, is_humanized)
            VALUES
                (:cid, :t1, 'user',      :user_msg,        false),
                (:cid, :t2, 'assistant', :assistant_reply,  true)
        """),
        {
            "cid":              conversation_id,
            "t1":               next_turn,
            "t2":               next_turn + 1,
            "user_msg":         user_msg,
            "assistant_reply":  assistant_reply,
        }
    )


# ── Routes ───────────────────────────────────────────────────────────

@router.post("/message")
def send_message(req: ChatRequest) -> dict:
    """
    Main chat endpoint. 
    - If conversation_id is None → starts a new conversation
    - If conversation_id is provided → continues existing conversation
    """
    engine = get_engine()

    with engine.begin() as conn:

        # Start new conversation if needed
        if req.conversation_id is None:
            result = conn.execute(
                text("""
                    INSERT INTO chat.conversations (title)
                    VALUES (:title)
                    RETURNING id
                """),
                {"title": req.message[:60]}   # first message as title
            )
            conversation_id = result.scalar_one()
        else:
            conversation_id = req.conversation_id
            # Verify it exists
            exists = conn.execute(
                text("SELECT id FROM chat.conversations WHERE id = :cid"),
                {"cid": conversation_id}
            ).scalar_one_or_none()
            if not exists:
                raise HTTPException(status_code=404, detail="Conversation not found")

        # Load history for multi-turn
        history = _get_history(conn, conversation_id)

        # Call Gemini with full history
        try:
            reply, _ = gemini_chat(
                user_message=req.message,
                history=history
            )
        except ChatBackendUnavailableError as exc:
            logger.warning("Chat unavailable: status=%s", exc.status_code)
            headers = {}
            if exc.retry_after_seconds:
                headers["Retry-After"] = str(exc.retry_after_seconds)
            raise HTTPException(
                status_code=503,
                detail="Chat is temporarily unavailable. Please try again later.",
                headers=headers or None,
            ) from exc

        # Save both turns to DB
        _save_messages(conn, conversation_id, req.message, reply)

    return {
        "conversation_id": conversation_id,
        "reply":           reply,
    }


@router.get("/conversations")
def list_conversations(search: Optional[str] = Query(None)) -> list[dict]:
    """List all conversations."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, title, created_at,
                    (SELECT content FROM chat.messages
                     WHERE conversation_id = c.id AND role = 'assistant'
                     ORDER BY turn_index DESC LIMIT 1) AS last_reply
                FROM chat.conversations c
                ORDER BY created_at DESC
            """)
        ).mappings().all()

    results = [dict(r) for r in rows]
    if search:
        results = [r for r in results if search.lower() in (r["title"] or "").lower()]
    return results


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int) -> dict:
    """Get a conversation with its full message history."""
    engine = get_engine()
    with engine.connect() as conn:
        convo = conn.execute(
            text("SELECT * FROM chat.conversations WHERE id = :cid"),
            {"cid": conversation_id}
        ).mappings().one_or_none()

        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = conn.execute(
            text("""
                SELECT turn_index, role, content, created_at
                FROM chat.messages
                WHERE conversation_id = :cid
                ORDER BY turn_index
            """),
            {"cid": conversation_id}
        ).mappings().all()

    return {
        "conversation": dict(convo),
        "messages": [dict(m) for m in messages]
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int) -> dict:
    """Delete a conversation and all its messages."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM chat.conversations WHERE id = :cid"),
            {"cid": conversation_id}
        )
    return {"status": "deleted", "conversation_id": conversation_id}
