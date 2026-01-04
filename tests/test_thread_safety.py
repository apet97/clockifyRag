"""Thread safety tests for concurrent operations."""

import pytest
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clockify_rag.caching import QueryCache, RateLimiter


def test_query_cache_thread_safe():
    """Verify cache works correctly with concurrent access."""
    cache = QueryCache(maxsize=100)
    results = []
    errors = []

    def worker(question, answer):
        try:
            cache.put(question, answer, {})
            result = cache.get(question)
            results.append((question, result))
        except Exception as e:
            errors.append(e)

    # Create 100 threads that simultaneously access the cache
    threads = [threading.Thread(target=worker, args=(f"question_{i}", f"answer_{i}")) for i in range(100)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"No errors should occur: {errors}"
    assert len(results) == 100, f"All threads should complete, got {len(results)}"

    # Verify all queries cached correctly (allowing for some evictions if cache full)
    successful_gets = sum(1 for _, result in results if result is not None)
    assert successful_gets >= 90, f"At least 90% should be cached, got {successful_gets}"

    for _, result in results:
        if result is None:
            continue
        _, metadata = result
        assert "timestamp" in metadata


@pytest.mark.skip(reason="RateLimiter is now a no-op for internal deployment")
def test_rate_limiter_thread_safe():
    """Verify rate limiter works correctly with concurrent access."""
    limiter = RateLimiter(max_requests=50, window_seconds=1)
    allowed_count = [0]
    denied_count = [0]
    lock = threading.Lock()

    def worker():
        if limiter.allow_request():
            with lock:
                allowed_count[0] += 1
        else:
            with lock:
                denied_count[0] += 1

    # Create 100 threads that simultaneously check rate limit
    threads = [threading.Thread(target=worker) for _ in range(100)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = allowed_count[0] + denied_count[0]
    assert total == 100, f"Should process all 100 requests, got {total}"
    assert allowed_count[0] <= 50, f"Should not exceed max_requests (50), got {allowed_count[0]}"
    assert denied_count[0] >= 50, f"Should deny excess requests, denied only {denied_count[0]}"


def test_query_cache_concurrent_put_get():
    """Test concurrent put and get operations."""
    cache = QueryCache(maxsize=50)
    errors = []

    def writer(start_idx):
        try:
            for i in range(start_idx, start_idx + 10):
                cache.put(f"q{i}", f"a{i}", {"idx": i})
        except Exception as e:
            errors.append(("writer", e))

    def reader(start_idx):
        try:
            for i in range(start_idx, start_idx + 10):
                cache.get(f"q{i}")
                # Result might be None if not yet written or evicted
        except Exception as e:
            errors.append(("reader", e))

    # Create interleaved writers and readers
    writers = [threading.Thread(target=writer, args=(i * 10,)) for i in range(5)]
    readers = [threading.Thread(target=reader, args=(i * 10,)) for i in range(5)]

    # Start all threads
    for w, r in zip(writers, readers):
        w.start()
        r.start()

    # Wait for completion
    for w, r in zip(writers, readers):
        w.join()
        r.join()

    assert len(errors) == 0, f"No errors should occur: {errors}"


@pytest.mark.skip(reason="RateLimiter is now a no-op for internal deployment")
def test_rate_limiter_wait_time_thread_safe():
    """Test that wait_time calculation is thread-safe."""
    limiter = RateLimiter(max_requests=10, window_seconds=1)
    wait_times = []
    lock = threading.Lock()

    def worker():
        # Make request
        allowed = limiter.allow_request()
        # Check wait time
        wait = limiter.wait_time()
        with lock:
            wait_times.append((allowed, wait))

    threads = [threading.Thread(target=worker) for _ in range(20)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should complete without errors
    assert len(wait_times) == 20

    # Verify wait times are reasonable (0 for allowed, positive for denied)
    for allowed, wait in wait_times:
        assert wait >= 0, f"Wait time should be non-negative, got {wait}"
        assert wait <= 2.0, f"Wait time should be reasonable (<=2s), got {wait}"


def test_query_cache_stats_thread_safe():
    """Test that cache statistics are accurate under concurrent access."""
    cache = QueryCache(maxsize=100)

    def worker(idx):
        # Put
        cache.put(f"q{idx}", f"a{idx}", {})
        # Get (hit)
        cache.get(f"q{idx}")
        # Get non-existent (miss)
        cache.get(f"nonexistent_{idx}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = cache.stats()
    total = stats["hits"] + stats["misses"]

    # Should have ~50 hits + 50 misses = 100 total gets
    assert total == 100, f"Expected 100 total gets, got {total}"
    assert stats["hits"] > 0, "Should have some hits"
    assert stats["misses"] > 0, "Should have some misses"
    assert stats["size"] <= 100, f"Cache size should not exceed maxsize (100), got {stats['size']}"


@pytest.mark.skip(reason="RateLimiter is now a no-op for internal deployment")
def test_rate_limiter_burst_handling():
    """Test rate limiter handles burst traffic correctly."""
    limiter = RateLimiter(max_requests=20, window_seconds=0.5)
    results = []
    lock = threading.Lock()

    def burst_worker():
        allowed = limiter.allow_request()
        with lock:
            results.append(allowed)

    # Send burst of 50 requests
    threads = [threading.Thread(target=burst_worker) for _ in range(50)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    allowed = sum(1 for r in results if r)
    denied = sum(1 for r in results if not r)

    assert allowed <= 20, f"Should allow at most 20 requests in burst, allowed {allowed}"
    assert denied >= 30, f"Should deny at least 30 requests, denied {denied}"


def test_query_cache_clear_thread_safe():
    """Test that cache clear is thread-safe."""
    cache = QueryCache()

    def writer():
        for i in range(100):
            cache.put(f"q{i}", f"a{i}", {})

    def clearer():
        time.sleep(0.01)  # Let some writes happen
        cache.clear()

    writer_thread = threading.Thread(target=writer)
    clearer_thread = threading.Thread(target=clearer)

    writer_thread.start()
    clearer_thread.start()

    writer_thread.join()
    clearer_thread.join()

    # Should complete without errors
    # Cache may or may not be empty depending on timing
    stats = cache.stats()
    assert stats["size"] >= 0, "Cache size should be non-negative"


def test_query_cache_lru_eviction_thread_safe():
    """Test that LRU eviction works correctly under concurrent access."""
    cache = QueryCache(maxsize=10)
    errors = []

    def worker(start_idx):
        try:
            for i in range(start_idx, start_idx + 20):
                cache.put(f"q{i}", f"a{i}", {})
        except Exception as e:
            errors.append(e)

    # 5 threads each trying to add 20 items to cache with maxsize=10
    threads = [threading.Thread(target=worker, args=(i * 20,)) for i in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"No errors during eviction: {errors}"

    stats = cache.stats()
    assert stats["size"] <= 10, f"Cache should respect maxsize (10), got {stats['size']}"
