"""Tests for circuit breaker pattern."""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor

from clockify_rag.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    get_circuit_breaker,
    reset_circuit_breaker,
    circuit_breaker,
)


class TestCircuitBreakerBasic:
    """Basic circuit breaker functionality tests."""

    def test_initial_state_closed(self):
        """Circuit should start in closed state."""
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_opens_after_threshold_failures(self):
        """Circuit should open after failure threshold reached."""
        cb = CircuitBreaker("test", failure_threshold=3, reset_timeout=60)

        # Record failures up to threshold
        for i in range(3):
            assert cb.state == CircuitState.CLOSED
            cb.record_failure()

        # Should now be open
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_stays_closed_below_threshold(self):
        """Circuit should stay closed below failure threshold."""
        cb = CircuitBreaker("test", failure_threshold=5)

        for _ in range(4):
            cb.record_failure()

        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_success_reduces_failure_count(self):
        """Successful requests should reduce failure count."""
        cb = CircuitBreaker("test", failure_threshold=5)

        # Record some failures
        for _ in range(3):
            cb.record_failure()

        # Record successes
        for _ in range(3):
            cb.record_success()

        # Should be back to 0 failures
        assert cb._failure_count == 0

    def test_transitions_to_half_open(self):
        """Circuit should transition to half-open after timeout."""
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should be half-open now
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True

    def test_half_open_closes_on_success(self):
        """Circuit should close after success in half-open state."""
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout=0.1)

        # Open and wait for half-open
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN

        # Record success
        cb.record_success()

        # Should be closed
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Circuit should reopen after failure in half-open state."""
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout=0.1)

        # Open and wait for half-open
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN

        # Record failure
        cb.record_failure()

        # Should be open again
        assert cb.state == CircuitState.OPEN

    def test_manual_reset(self):
        """Manual reset should close the circuit."""
        cb = CircuitBreaker("test", failure_threshold=2)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Manual reset
        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_get_stats(self):
        """Stats should reflect current state."""
        cb = CircuitBreaker("test_stats", failure_threshold=5, reset_timeout=30)

        cb.record_failure()
        cb.record_failure()

        stats = cb.get_stats()
        assert stats["name"] == "test_stats"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 2
        assert stats["failure_threshold"] == 5


class TestCircuitBreakerRegistry:
    """Tests for global circuit breaker registry."""

    def test_get_creates_new(self):
        """get_circuit_breaker should create new breakers."""
        # Use unique name to avoid interference
        name = f"test_registry_{time.time()}"
        cb = get_circuit_breaker(name)
        assert cb is not None
        assert cb.name == name

    def test_get_returns_same_instance(self):
        """get_circuit_breaker should return same instance for same name."""
        name = f"test_same_{time.time()}"
        cb1 = get_circuit_breaker(name)
        cb2 = get_circuit_breaker(name)
        assert cb1 is cb2

    def test_reset_specific(self):
        """reset_circuit_breaker should reset specific breaker."""
        name = f"test_reset_{time.time()}"
        cb = get_circuit_breaker(name, failure_threshold=2)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        result = reset_circuit_breaker(name)
        assert result is True
        assert cb.state == CircuitState.CLOSED

    def test_reset_nonexistent(self):
        """reset_circuit_breaker should return False for unknown name."""
        result = reset_circuit_breaker("nonexistent_circuit_12345")
        assert result is False


class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator."""

    def test_decorator_allows_when_closed(self):
        """Decorated function should work when circuit is closed."""
        call_count = 0

        @circuit_breaker(f"test_decorator_{time.time()}")
        def my_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = my_func()
        assert result == "success"
        assert call_count == 1

    def test_decorator_records_failure(self):
        """Decorator should record failures."""
        name = f"test_failure_{time.time()}"

        @circuit_breaker(name, failure_threshold=2)
        def failing_func():
            raise ValueError("test error")

        # First failure
        with pytest.raises(ValueError):
            failing_func()

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            failing_func()

        # Third call - circuit should be open
        with pytest.raises(CircuitOpenError):
            failing_func()

    def test_decorator_with_fallback(self):
        """Decorator should use fallback when circuit is open."""
        name = f"test_fallback_{time.time()}"

        def fallback_func():
            return "fallback"

        @circuit_breaker(name, failure_threshold=2, fallback=fallback_func)
        def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                failing_func()

        # Should use fallback
        result = failing_func()
        assert result == "fallback"


class TestCircuitBreakerConcurrency:
    """Tests for thread safety."""

    def test_concurrent_failures(self):
        """Circuit should handle concurrent failures correctly."""
        cb = CircuitBreaker(f"test_concurrent_{time.time()}", failure_threshold=10)

        def record_failure():
            cb.record_failure()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(record_failure) for _ in range(10)]
            for f in futures:
                f.result()

        assert cb.state == CircuitState.OPEN

    def test_concurrent_mixed_operations(self):
        """Circuit should handle mixed concurrent operations."""
        cb = CircuitBreaker(f"test_mixed_{time.time()}", failure_threshold=100)

        def operation(should_fail):
            if should_fail:
                cb.record_failure()
            else:
                cb.record_success()

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(100):
                futures.append(executor.submit(operation, i % 2 == 0))
            for f in futures:
                f.result()

        # Should still be closed (50 failures < 100 threshold)
        assert cb.state == CircuitState.CLOSED


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_error_message(self):
        """Error should have informative message."""
        error = CircuitOpenError("test", 30.5)
        assert "test" in str(error)
        assert "30.5" in str(error)

    def test_error_attributes(self):
        """Error should have correct attributes."""
        error = CircuitOpenError("myservice", 45.0)
        assert error.name == "myservice"
        assert error.retry_after == 45.0
