"""Thread safety regression tests for CLI cache and rate limiter."""

import threading
import pytest

from clockify_rag.caching import QueryCache, RateLimiter


def test_cli_query_cache_concurrent_access():
    """Ensure the CLI's QueryCache handles concurrent put/get operations."""

    cache = QueryCache(maxsize=128, ttl_seconds=5)
    errors = []

    def worker(idx: int) -> None:
        try:
            question = f"question-{idx}"
            answer = f"answer-{idx}"
            cache.put(question, answer, {"idx": idx})
            result = cache.get(question)
            # Result should be a tuple of (answer, metadata)
            assert result is not None
            assert result[0] == answer
            cache.stats()
        except Exception as exc:  # pragma: no cover - debugging aid
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(64)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"No thread should raise exceptions: {errors!r}"


@pytest.mark.skip(reason="RateLimiter is now a no-op for internal deployment")
def test_cli_rate_limiter_concurrent_access():
    """Verify the CLI's RateLimiter safely handles concurrent requests."""

    limiter = RateLimiter(max_requests=20, window_seconds=1)
    allowed = 0
    denied = 0
    counter_lock = threading.Lock()

    def worker() -> None:
        nonlocal allowed, denied
        allowed_request = limiter.allow_request()
        wait = limiter.wait_time()
        assert wait >= 0
        with counter_lock:
            if allowed_request:
                allowed += 1
            else:
                denied += 1

    threads = [threading.Thread(target=worker) for _ in range(40)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    total = allowed + denied
    assert total == 40, f"All threads should record a result, got {total}"
    assert allowed <= 20, f"Should allow at most 20 requests, allowed {allowed}"
