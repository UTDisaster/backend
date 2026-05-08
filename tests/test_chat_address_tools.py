from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("IMAGE_CONTENT_BASE_URL", "http://test/assets")
os.environ.setdefault("APP_ENV", "dev")

from app.services import gemini as gemini_mod
from app.routers import chat as chat_router
from app.services.location_queries import (
    AddressMatch,
    DamageAggregate,
    lookup_damage_at_address,
    nearby_damage,
)


# ── Fake SQLAlchemy-ish plumbing ─────────────────────────────────────


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None: self._rows = rows
    def mappings(self) -> "_FakeResult": return self
    def all(self) -> list[dict[str, Any]]: return self._rows
    def one_or_none(self) -> dict[str, Any] | None: return self._rows[0] if self._rows else None
    def scalar(self) -> Any: return self._rows[0]["scalar"] if self._rows else None
    def scalar_one(self) -> Any: return self.scalar()
    def scalar_one_or_none(self) -> Any: return self.scalar()


class _FakeConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows, self.calls = rows, []

    def execute(self, _s: Any, params: dict[str, Any] | None = None) -> _FakeResult:
        self.calls.append(dict(params or {}))
        return _FakeResult(self._rows)


class _FakeEngine:
    def __init__(self, conn: _FakeConn) -> None: self._conn = conn

    @contextmanager
    def connect(self) -> Iterator[_FakeConn]:
        yield self._conn

    @contextmanager
    def begin(self) -> Iterator[_FakeConn]:
        yield self._conn


def _row(**o: Any) -> dict[str, Any]:
    return {
        "street": "Ocean Blvd", "city": "Myrtle Beach", "county": "Horry",
        "total": 12, "severe": 3, "destroyed": 1,
        "lat": 33.69, "lng": -78.89, "score": 0.85, **o,
    }


# ── lookup_damage_at_address ─────────────────────────────────────────


def test_lookup_exact_match_returns_top_row() -> None:
    conn = _FakeConn([_row(score=0.95)])
    out = lookup_damage_at_address(conn, "Ocean Blvd")
    assert out == [AddressMatch(
        street="Ocean Blvd", city="Myrtle Beach", county="Horry",
        total=12, severe=3, destroyed=1, lat=33.69, lng=-78.89, score=0.95,
    )]
    assert conn.calls[0]["q"] == "Ocean Blvd"


def test_lookup_fuzzy_match_keeps_trigram_ordering() -> None:
    # Caller receives rows in the order the DB ranked them by similarity.
    rows = [_row(street="Ocean Blvd", score=0.62), _row(street="Ocala Ave", score=0.31)]
    out = lookup_damage_at_address(_FakeConn(rows), "Ocen Blvd")
    assert [m.street for m in out] == ["Ocean Blvd", "Ocala Ave"]


def test_lookup_short_query_returns_empty_and_skips_db() -> None:
    conn = _FakeConn([])
    assert lookup_damage_at_address(conn, "Oc") == []
    assert lookup_damage_at_address(conn, "   ") == []
    assert conn.calls == []


def test_lookup_can_scope_to_disaster() -> None:
    conn = _FakeConn([_row(score=0.95)])
    lookup_damage_at_address(conn, "Ocean Blvd", disaster_id="hurricane-florence")
    assert conn.calls[0]["disaster_id"] == "hurricane-florence"


# ── nearby_damage ────────────────────────────────────────────────────


def test_nearby_damage_aggregates_buckets() -> None:
    rows = [
        {"level": "none", "n": 2},
        {"level": "minor", "n": 1},
        {"level": "severe", "n": 3},
        {"level": "destroyed", "n": 1},
        {"level": "unknown", "n": 1},
    ]
    conn = _FakeConn(rows)
    agg = nearby_damage(conn, lat=33.69, lng=-78.89, radius_m=200)
    assert agg == DamageAggregate(none=2, minor=1, severe=3, destroyed=1, unknown=1)
    assert conn.calls[0]["radius"] == 200


def test_nearby_damage_unknown_bucket_catches_unmapped_levels() -> None:
    rows = [{"level": "weird", "n": 2}, {"level": "severe", "n": 1}]
    agg = nearby_damage(_FakeConn(rows), lat=0.0, lng=0.0)
    assert (agg.unknown, agg.severe) == (2, 1)


def test_nearby_damage_radius_clamped_up() -> None:
    conn = _FakeConn([])
    nearby_damage(conn, lat=0.0, lng=0.0, radius_m=10)
    assert conn.calls[0]["radius"] == 20


def test_nearby_damage_radius_clamped_down() -> None:
    conn = _FakeConn([])
    nearby_damage(conn, lat=0.0, lng=0.0, radius_m=999_999)
    assert conn.calls[0]["radius"] == 5000


def test_nearby_damage_can_scope_to_disaster() -> None:
    conn = _FakeConn([])
    nearby_damage(conn, lat=33.69, lng=-78.89, disaster_id="hurricane-florence")
    assert conn.calls[0]["disaster_id"] == "hurricane-florence"


# ── gemini._run_tool wiring ──────────────────────────────────────────


def test_run_tool_lookup_damage_at_address_returns_matches() -> None:
    engine = _FakeEngine(_FakeConn([_row(score=0.95)]))
    with patch.object(gemini_mod, "get_engine", return_value=engine):
        payload = json.loads(gemini_mod._run_tool(
            "lookup_damage_at_address", {"query": "Ocean Blvd"}
        ))
    assert payload["matches"][0]["street"] == "Ocean Blvd"
    assert payload["matches"][0]["score"] == 0.95


def test_run_tool_nearby_damage_returns_aggregate() -> None:
    rows = [{"level": "severe", "n": 4}, {"level": "destroyed", "n": 2}]
    engine = _FakeEngine(_FakeConn(rows))
    with patch.object(gemini_mod, "get_engine", return_value=engine):
        payload = json.loads(gemini_mod._run_tool(
            "nearby_damage", {"lat": 33.69, "lng": -78.89, "radius_m": 300}
        ))
    assert payload == {"none": 0, "minor": 0, "severe": 4, "destroyed": 2, "unknown": 0}


def test_run_tool_damage_stats_defaults_to_selected_disaster() -> None:
    conn = _FakeConn([{"damage_level": "destroyed", "count": 81}])
    engine = _FakeEngine(conn)
    with patch.object(gemini_mod, "get_engine", return_value=engine):
        payload = json.loads(gemini_mod._run_tool(
            "get_damage_stats", {}, default_disaster_id="hurricane-florence"
        ))
    assert payload == [{"damage_level": "destroyed", "count": 81}]
    assert conn.calls[0]["disaster_id"] == "hurricane-florence"


def test_run_tool_hotspots_returns_sorted_shape_and_scope() -> None:
    conn = _FakeConn([
        {"lat": 33.62, "lng": -79.05, "severe": 12, "destroyed": 4, "total": 30}
    ])
    engine = _FakeEngine(conn)
    with patch.object(gemini_mod, "get_engine", return_value=engine):
        payload = json.loads(gemini_mod._run_tool(
            "get_damage_hotspots", {}, default_disaster_id="hurricane-florence"
        ))
    assert payload == {
        "hotspots": [
            {"lat": 33.62, "lng": -79.05, "severe": 12, "destroyed": 4, "total": 30}
        ]
    }
    assert conn.calls[0]["disaster_id"] == "hurricane-florence"


# ── _dominant_match helper ───────────────────────────────────────────


def test_dominant_match_single_and_clear_winner_and_ambiguous() -> None:
    assert gemini_mod._dominant_match([]) is None
    single = [{"lat": 1.0, "lng": 2.0, "score": 0.4}]
    assert gemini_mod._dominant_match(single) is single[0]
    clear = [{"lat": 1.0, "lng": 2.0, "score": 0.9}, {"lat": 3.0, "lng": 4.0, "score": 0.5}]
    assert gemini_mod._dominant_match(clear) is clear[0]
    ambiguous = [{"lat": 1.0, "lng": 2.0, "score": 0.55}, {"lat": 3.0, "lng": 4.0, "score": 0.50}]
    assert gemini_mod._dominant_match(ambiguous) is None


# ── End-to-end: chat() emits flyTo for a dominant address match ──────


class _FC:
    def __init__(self, name: str, args: dict[str, Any]) -> None: self.name, self.args = name, args


class _P:
    def __init__(self, *, text: str | None = None, fc: _FC | None = None) -> None:
        self.text, self.function_call = text, fc


class _Resp:
    def __init__(self, parts: list[_P]) -> None:
        self.candidates = [type("C", (), {"content": type("D", (), {"parts": parts})()})()]


class _Client:
    def __init__(self, responses: list[_Resp]) -> None:
        self._r, self.models = list(responses), self

    def generate_content(self, **_kw: Any) -> _Resp: return self._r.pop(0)


def _run_chat(
    tool_args: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    final_text: str | None = "done.",
    tool_name: str = "lookup_damage_at_address",
    disaster_id: str | None = None,
) -> tuple[str, list[dict]]:
    call = _P(fc=_FC(tool_name, tool_args))
    final_parts = [] if final_text is None else [_P(text=final_text)]
    client = _Client([_Resp([call]), _Resp(final_parts)])
    engine = _FakeEngine(_FakeConn(rows))
    with patch.object(gemini_mod, "_get_client", return_value=client), \
         patch.object(gemini_mod, "get_engine", return_value=engine):
        r, _h, actions = gemini_mod.chat("damage?", disaster_id=disaster_id)
    return r, actions


def test_chat_appends_flyto_for_single_address_match() -> None:
    _reply, actions = _run_chat(
        {"query": "Ocean Blvd"},
        [_row(score=0.95, lat=33.69, lng=-78.89)],
    )
    fly = [a for a in actions if a.get("type") == "flyTo"]
    assert len(fly) == 1
    assert fly[0]["zoom"] == 17
    assert fly[0]["lat"] == pytest.approx(33.69)
    assert fly[0]["lng"] == pytest.approx(-78.89)


def test_chat_no_flyto_when_no_matches() -> None:
    _reply, actions = _run_chat({"query": "Nonexistent Rd"}, [])
    assert not any(a.get("type") == "flyTo" for a in actions)


def test_chat_synthesizes_address_summary_when_model_returns_no_text() -> None:
    reply, _actions = _run_chat(
        {"query": "Ocean Blvd"},
        [_row(score=0.95, lat=33.69, lng=-78.89)],
        final_text=None,
    )
    assert reply != "Done."
    assert "Ocean Blvd" in reply
    assert "12 assessed buildings" in reply


def test_chat_synthesizes_no_match_message_when_address_not_found() -> None:
    reply, actions = _run_chat(
        {"query": "Accord Street"},
        [],
        final_text=None,
    )
    assert "couldn't find damage records" in reply
    assert not actions


def test_chat_does_not_fly_to_out_of_scope_navigation() -> None:
    reply, actions = _run_chat(
        {"lat": 0.0, "lng": 0.0, "zoom": 17},
        [{
            "min_lat": 33.0,
            "max_lat": 35.5,
            "min_lng": -80.0,
            "max_lng": -77.0,
        }],
        final_text=None,
        tool_name="navigate_map",
        disaster_id="hurricane-florence",
    )
    assert "selected disaster area" in reply
    assert not any(a.get("type") == "flyTo" for a in actions)


def test_chat_request_forwards_disaster_context_and_history() -> None:
    class _RouterConn:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def execute(self, statement: Any, params: dict[str, Any] | None = None) -> _FakeResult:
            sql = str(statement)
            self.calls.append(dict(params or {}))
            if "INSERT INTO chat.conversations" in sql:
                return _FakeResult([{"scalar": 42}])
            if "SELECT role, content FROM chat.messages" in sql:
                return _FakeResult([{"role": "user", "content": "previous question"}])
            if "SELECT COALESCE(MAX(turn_index)" in sql:
                return _FakeResult([{"scalar": 1}])
            return _FakeResult([])

    conn = _RouterConn()
    engine = _FakeEngine(conn)  # type: ignore[arg-type]
    req = chat_router.ChatRequest(
        message="tell me more",
        disaster_id="hurricane-florence",
        disaster_name="Hurricane Florence",
    )
    with patch.object(chat_router, "get_engine", return_value=engine), \
         patch.object(chat_router, "gemini_chat", return_value=("reply", [], [])) as mocked:
        response = chat_router.send_message(req)

    assert response["conversation_id"] == 42
    kwargs = mocked.call_args.kwargs
    assert kwargs["disaster_id"] == "hurricane-florence"
    assert kwargs["disaster_name"] == "Hurricane Florence"
    assert kwargs["history"] == [{"role": "user", "parts": ["previous question"]}]
