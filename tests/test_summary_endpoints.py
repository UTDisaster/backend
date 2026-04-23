from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest

# conftest.py in this repo does not provide a DB fixture, so we mock the
# engine instead of spinning up Postgres for this PR. Follows the same
# _FakeEngine / _FakeConn / _FakeResult pattern used in
# tests/test_locations_address.py.

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test"
)
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from fastapi.testclient import TestClient

from app import main as app_main


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> "_FakeResult":
        return self

    def all(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def first(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def execute(
        self, _stmt: Any, _params: dict[str, Any] | None = None
    ) -> _FakeResult:
        return _FakeResult(self._rows)


class _FakeEngine:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    @contextmanager
    def connect(self) -> Iterator[_FakeConn]:
        yield _FakeConn(self._rows)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_main.app)


def test_disaster_summary_returns_documented_shape(client: TestClient) -> None:
    # The production SQL returns exactly one aggregated row. The fake engine
    # returns that row verbatim to confirm the Python-side shaping.
    summary_row = {
        "total_locations": 11548,
        "none_count": 8125,
        "minor_count": 781,
        "severe_count": 2196,
        "destroyed_count": 410,
        "unknown_count": 36,
        "min_lat": 33.4,
        "max_lat": 34.8,
        "min_lng": -79.3,
        "max_lng": -78.2,
    }
    with patch.object(
        app_main, "get_engine", return_value=_FakeEngine([summary_row])
    ):
        resp = client.get("/disasters/hurricane-florence/summary")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "disaster_id": "hurricane-florence",
        "total_locations": 11548,
        "by_classification": {
            "none": 8125,
            "minor": 781,
            "severe": 2196,
            "destroyed": 410,
            "unknown": 36,
        },
        "bbox": {
            "minLat": 33.4,
            "minLng": -79.3,
            "maxLat": 34.8,
            "maxLng": -78.2,
        },
    }


def test_disaster_summary_missing_returns_404(client: TestClient) -> None:
    # When the disaster_id is not present, the aggregate query still returns
    # one row but with total=0 (and null min/max). The endpoint must 404.
    empty_row = {
        "total_locations": 0,
        "none_count": 0,
        "minor_count": 0,
        "severe_count": 0,
        "destroyed_count": 0,
        "unknown_count": 0,
        "min_lat": None,
        "max_lat": None,
        "min_lng": None,
        "max_lng": None,
    }
    with patch.object(
        app_main, "get_engine", return_value=_FakeEngine([empty_row])
    ):
        resp = client.get("/disasters/does-not-exist/summary")

    assert resp.status_code == 404
    assert resp.json() == {"detail": "disaster not found"}


def test_locations_hotspots_respects_limit_and_order(client: TestClient) -> None:
    # The SQL does the ORDER BY and LIMIT; we pre-sort the fake rows to match
    # what the DB would return. The endpoint then trusts the DB ordering.
    rows = [
        {
            "lat_bin": 33.70,
            "lng_bin": -78.90,
            "severe_count": 42,
            "destroyed_count": 18,
            "total_count": 74,
        },
        {
            "lat_bin": 33.71,
            "lng_bin": -78.91,
            "severe_count": 30,
            "destroyed_count": 12,
            "total_count": 55,
        },
        {
            "lat_bin": 33.72,
            "lng_bin": -78.92,
            "severe_count": 20,
            "destroyed_count": 5,
            "total_count": 40,
        },
    ]
    with patch.object(app_main, "get_engine", return_value=_FakeEngine(rows)):
        resp = client.get(
            "/locations/hotspots",
            params={"disaster_id": "hurricane-florence", "limit": 3},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "hotspots" in body
    hotspots = body["hotspots"]
    assert len(hotspots) <= 3
    # Verify shape of the first bin and the severe+destroyed desc ordering.
    assert hotspots[0] == {
        "lat": 33.70,
        "lng": -78.90,
        "severe": 42,
        "destroyed": 18,
        "total": 74,
    }
    scores = [h["severe"] + h["destroyed"] for h in hotspots]
    assert scores == sorted(scores, reverse=True)
