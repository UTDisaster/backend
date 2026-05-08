from __future__ import annotations

import datetime as dt
import os
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest

# conftest.py in this repo does not provide a DB fixture, so we mock the
# engine instead of spinning up Postgres for this PR. An integration test
# against `make db` can be layered on later without changing this contract.

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("IMAGE_CONTENT_BASE_URL", "http://test/assets")
os.environ.setdefault("APP_ENV", "dev")

from fastapi.testclient import TestClient

from app import main as app_main


BASE_ROW: dict[str, Any] = {
    "location_id": 1,
    "location_uid": "loc-1",
    "image_pair_id": "ip-1",
    "feature_type": "building",
    "classification": "no-damage",
    "damage_level": "none",
    "vlm_confidence": 0.9,
    "vlm_description": "intact",
    "disaster_id": "hurricane-florence",
    "pre_path": "images/pre.png",
    "post_path": "images/post.png",
    "geometry": '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}',
    "centroid_lng": 0.5,
    "centroid_lat": 0.5,
}


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> "_FakeResult":
        return self

    def all(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def execute(self, _stmt: Any, _params: dict[str, Any] | None = None) -> _FakeResult:
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


def _bbox_params() -> dict[str, float]:
    return {"min_lng": -1.0, "min_lat": -1.0, "max_lng": 1.0, "max_lat": 1.0}


def test_locations_default_response_has_no_address_key(client: TestClient) -> None:
    rows = [dict(BASE_ROW)]
    with patch.object(app_main, "get_engine", return_value=_FakeEngine(rows)):
        resp = client.get("/locations", params=_bbox_params())
    assert resp.status_code == 200
    feature = resp.json()["features"][0]
    assert "address" not in feature


def test_locations_include_address_without_geocoded_fields_returns_null(client: TestClient) -> None:
    # Design decision: when include_address=true but the row has never been
    # geocoded, we emit `address: null` rather than an object full of nulls.
    # Callers can distinguish "not yet fetched" from "fetched, no match" by
    # checking `address === null` vs `address.source === 'census'`.
    row = dict(
        BASE_ROW,
        street=None,
        city=None,
        county=None,
        full_address=None,
        address_source=None,
        address_fetched_at=None,
    )
    with patch.object(app_main, "get_engine", return_value=_FakeEngine([row])):
        resp = client.get("/locations", params={**_bbox_params(), "include_address": "true"})
    assert resp.status_code == 200
    feature = resp.json()["features"][0]
    assert feature["address"] is None


def test_locations_include_address_with_geocoded_fields_returns_nested_object(client: TestClient) -> None:
    fetched = dt.datetime(2026, 4, 18, 12, 0, 0, tzinfo=dt.timezone.utc)
    row = dict(
        BASE_ROW,
        street="Main St",
        city="Wilmington",
        county="New Hanover",
        full_address="123 Main St, Wilmington, NC",
        address_source="census",
        address_fetched_at=fetched,
    )
    with patch.object(app_main, "get_engine", return_value=_FakeEngine([row])):
        resp = client.get("/locations", params={**_bbox_params(), "include_address": "true"})
    assert resp.status_code == 200
    feature = resp.json()["features"][0]
    assert feature["address"] == {
        "street": "Main St",
        "city": "Wilmington",
        "county": "New Hanover",
        "full_address": "123 Main St, Wilmington, NC",
        "source": "census",
        "fetched_at": fetched.isoformat(),
    }
