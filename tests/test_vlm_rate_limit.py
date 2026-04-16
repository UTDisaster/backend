from __future__ import annotations

import time

import pytest

from app.services.vlm.rate_limit import BackoffPolicy, TokenBucket


def test_token_bucket_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        TokenBucket(0.0)
    with pytest.raises(ValueError):
        TokenBucket(-1.0)


def test_token_bucket_burst_immediate() -> None:
    bucket = TokenBucket(rate_per_second=100.0, burst=3)
    start = time.monotonic()
    for _ in range(3):
        bucket.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05


def test_token_bucket_enforces_rate_for_sustained_calls() -> None:
    bucket = TokenBucket(rate_per_second=20.0, burst=1)
    start = time.monotonic()
    for _ in range(6):
        bucket.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.2


def test_backoff_uses_suggested_delay() -> None:
    policy = BackoffPolicy(base_seconds=0.1, max_seconds=10.0, jitter_fraction=0.0)
    delay = policy.delay_for(0, suggested=2.0)
    assert delay == pytest.approx(2.0, abs=1e-6)


def test_backoff_respects_max_cap() -> None:
    policy = BackoffPolicy(base_seconds=1.0, max_seconds=4.0, jitter_fraction=0.0)
    delay = policy.delay_for(10)
    assert delay == pytest.approx(4.0, abs=1e-6)


def test_backoff_jitter_keeps_within_band() -> None:
    policy = BackoffPolicy(base_seconds=1.0, max_seconds=10.0, jitter_fraction=0.3)
    samples = [policy.delay_for(1) for _ in range(25)]
    assert all(0.7 * 2.0 <= s <= 1.3 * 2.0 + 0.01 for s in samples)
