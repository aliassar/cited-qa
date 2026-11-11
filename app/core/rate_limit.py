"""
Token-bucket rate limiter.

THE ALGORITHM
-------------
Each key (here: user id, falling back to client IP) owns a bucket holding at
most `capacity` tokens. Tokens refill continuously at `capacity / window`
tokens per second. Each request costs one token; an empty bucket means 429.

Why token bucket rather than a fixed window counter?
  * Fixed windows have a boundary problem: 120 requests at 11:59:59 plus 120 at
    12:00:00 is 240 requests in one second, all "within limits".
  * Token bucket allows a controlled *burst* (up to capacity) and then enforces
    a smooth average rate. That matches how real clients behave.

The refill is computed lazily from a timestamp instead of running a background
ticker - no timers, no extra tasks, O(1) per request.

*** THE PRODUCTION CAVEAT ***
This implementation is IN-PROCESS. With 4 uvicorn workers you get 4 independent
buckets, so the effective limit is 4x what you configured; with N pods it is N x.
That is fine for coarse abuse protection and terrible for billing quotas.
The real answer is shared state: Redis with an atomic Lua script (read tokens,
refill, decrement, write - all in one round trip so two pods cannot interleave),
or an edge/gateway limiter (Envoy, Kong, Cloudflare) before traffic reaches you.

We also keep a bounded dict: without eviction, a stream of unique IPs is an
unbounded memory leak, i.e. a denial-of-service vector against your own service.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class TokenBucketLimiter:
    def __init__(self, capacity: int, window_seconds: int, max_keys: int = 10_000) -> None:
        self.capacity = float(capacity)
        # Tokens added per second so the bucket refills fully in one window.
        self.refill_rate = capacity / window_seconds
        self.max_keys = max_keys
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        """
        Consume one token for `key`.

        Returns (allowed, retry_after_seconds). `retry_after` is how long until
        one token exists again - we hand it to the client in the Retry-After
        header so well-behaved clients stop hammering us.

        NOTE: time.monotonic(), not time.time(). A wall-clock jump (NTP sync,
        DST) would otherwise either freeze the limiter or refill it instantly.
        """
        now = time.monotonic()
        bucket = self._buckets.get(key)

        if bucket is None:
            self._evict_if_needed()
            # New keys start full: a first-time caller is never rate limited.
            bucket = _Bucket(tokens=self.capacity, last_refill=now)
            self._buckets[key] = bucket

        # Lazy refill: add whatever accrued since we last looked, capped.
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_rate)
        bucket.last_refill = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0

        # Seconds until the bucket holds one whole token again.
        deficit = 1.0 - bucket.tokens
        return False, max(1, int(deficit / self.refill_rate) + 1)

    def _evict_if_needed(self) -> None:
        """Crude bounded-memory guard: drop the least recently touched half."""
        if len(self._buckets) < self.max_keys:
            return
        ordered = sorted(self._buckets.items(), key=lambda kv: kv[1].last_refill)
        for key, _ in ordered[: len(ordered) // 2]:
            del self._buckets[key]

    def reset(self) -> None:
        """Test helper - never call this from request code."""
        self._buckets.clear()
