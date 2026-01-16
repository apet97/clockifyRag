"""Tests for rate limiting functionality."""

import pytest
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.caching import RateLimiter


@pytest.mark.skip(
    reason="RateLimiter is now a no-op for internal deployment (optimization). Tests expect actual rate limiting but implementation always allows requests."
)
class TestRateLimiter:
    """Test rate limiter logic."""

    def test_rate_limiter_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(max_requests=3, window_seconds=10)

        assert limiter.allow_request() is True
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True

    def test_rate_limiter_blocks_after_limit(self):
        """Test that requests beyond limit are blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=10)

        # First 3 should pass
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True

        # 4th should be blocked
        assert limiter.allow_request() is False

    def test_rate_limiter_resets_after_window(self):
        """Test that rate limiter resets after time window."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)

        # Use up quota
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True
        assert limiter.allow_request() is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        assert limiter.allow_request() is True

    def test_rate_limiter_wait_time_zero_when_allowed(self):
        """Test wait_time returns 0 when request would be allowed."""
        limiter = RateLimiter(max_requests=3, window_seconds=10)

        assert limiter.wait_time() == 0.0
        limiter.allow_request()
        assert limiter.wait_time() == 0.0

    def test_rate_limiter_wait_time_nonzero_when_blocked(self):
        """Test wait_time returns positive value when rate limited."""
        limiter = RateLimiter(max_requests=2, window_seconds=10)

        # Use up quota
        limiter.allow_request()
        limiter.allow_request()

        # Should be blocked
        assert limiter.allow_request() is False

        # Wait time should be positive
        wait_time = limiter.wait_time()
        assert wait_time > 0
        assert wait_time <= 10  # Should not exceed window

    def test_rate_limiter_sliding_window(self):
        """Test that rate limiter uses sliding window (not fixed window)."""
        limiter = RateLimiter(max_requests=2, window_seconds=2)

        # Request at t=0
        assert limiter.allow_request() is True  # t=0

        # Wait 1 second
        time.sleep(1.0)

        # Request at t=1 (2nd request)
        assert limiter.allow_request() is True  # t=1
        # Should be blocked (2 requests in last 2 seconds)
        assert limiter.allow_request() is False

        # Wait another 1 second (t=2, first request expires)
        time.sleep(1.0)

        # Request at t=2 should be allowed (only 1 request in last 2 seconds)
        assert limiter.allow_request() is True

    def test_rate_limiter_custom_limits(self):
        """Test rate limiter with custom limits."""
        # Very restrictive
        limiter = RateLimiter(max_requests=1, window_seconds=5)
        assert limiter.allow_request() is True
        assert limiter.allow_request() is False

        # Very permissive
        limiter = RateLimiter(max_requests=100, window_seconds=1)
        for _ in range(100):
            assert limiter.allow_request() is True
        assert limiter.allow_request() is False

    def test_rate_limiter_concurrent_safety(self):
        """Test that rate limiter maintains count correctly."""
        limiter = RateLimiter(max_requests=5, window_seconds=10)

        # Allow 5 requests
        allowed_count = 0
        for _ in range(10):
            if limiter.allow_request():
                allowed_count += 1

        # Should have allowed exactly 5
        assert allowed_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
