from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass


class TokenBucket:
    """Thread-safe token bucket for rate limiting Gemini calls."""

    def __init__(self, rate_per_second: float, burst: int | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._rate = float(rate_per_second)
        self._capacity = float(burst if burst is not None else max(1, int(rate_per_second * 2)))
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._updated_at = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                needed = tokens - self._tokens
                wait_seconds = needed / self._rate
            time.sleep(wait_seconds)


@dataclass(frozen=True)
class BackoffPolicy:
    max_attempts: int = 3
    base_seconds: float = 1.0
    max_seconds: float = 16.0
    jitter_fraction: float = 0.3

    def delay_for(self, attempt: int, suggested: float | None = None) -> float:
        if suggested is not None:
            base = max(self.base_seconds, min(self.max_seconds, suggested))
        else:
            base = min(self.max_seconds, self.base_seconds * (2 ** attempt))
        jitter = base * self.jitter_fraction
        return max(0.0, base + random.uniform(-jitter, jitter))
