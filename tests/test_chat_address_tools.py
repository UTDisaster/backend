from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from app.services import gemini as gemini_mod
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


def _run_chat(tool_args: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, list[dict]]:
    call = _P(fc=_FC("lookup_damage_at_address", tool_args))
    reply = _P(text="done.")
    client = _Client([_Resp([call]), _Resp([reply])])
    engine = _FakeEngine(_FakeConn(rows))
    with patch.object(gemini_mod, "_get_client", return_value=client), \
         patch.object(gemini_mod, "get_engine", return_value=engine):
        r, _h, actions = gemini_mod.chat("damage?")
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
