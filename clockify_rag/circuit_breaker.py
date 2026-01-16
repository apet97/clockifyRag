"""Circuit breaker pattern for remote service calls.

Prevents cascading failures by temporarily blocking requests to unhealthy
services. After a threshold of failures, the circuit "opens" and fails fast
for a configured timeout period before attempting to recover.

Usage:
    from clockify_rag.circuit_breaker import circuit_breaker, CircuitOpenError

    @circuit_breaker("ollama")
    def call_ollama():
        # ... make remote call ...

    # Or manual usage:
    cb = get_circuit_breaker("ollama")
    if cb.allow_request():
        try:
            result = make_request()
            cb.record_success()
        except Exception as e:
            cb.record_failure()
            raise
"""

import logging
import threading
import time
from enum import Enum
from functools import wraps
from typing import Callable, Dict, Optional, TypeVar

from . import config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing fast, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(Exception):
    """Raised when circuit is open and requests are blocked."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker '{name}' is open. " f"Retry after {retry_after:.1f}s")


class CircuitBreaker:
    """Thread-safe circuit breaker implementation.

    Tracks failures and opens the circuit when threshold is exceeded.
    After a reset timeout, allows a test request (half-open state).
    If the test succeeds, circuit closes; if it fails, circuit reopens.

    Attributes:
        name: Identifier for this circuit breaker
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Seconds to wait before testing recovery
        half_open_max_calls: Max concurrent calls in half-open state
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition on read)."""
        with self._lock:
            self._maybe_transition()
            return self._state

    def _maybe_transition(self) -> None:
        """Check if state should transition based on time."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                logger.info("circuit_breaker: name=%s transition=open->half_open " "elapsed=%.1fs", self.name, elapsed)
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

    def allow_request(self) -> bool:
        """Check if a request should be allowed.

        Returns:
            True if request can proceed, False if circuit is open

        Raises:
            CircuitOpenError: If circuit is open (for explicit error handling)
        """
        with self._lock:
            self._maybe_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            # OPEN state
            return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    logger.info(
                        "circuit_breaker: name=%s transition=half_open->closed " "successes=%d",
                        self.name,
                        self._success_count,
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                # Gradual recovery: reduce failure count on success
                if self._failure_count > 0:
                    self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("circuit_breaker: name=%s transition=half_open->open " "reason=test_failed", self.name)
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                self._success_count = 0

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        "circuit_breaker: name=%s transition=closed->open " "failures=%d threshold=%d",
                        self.name,
                        self._failure_count,
                        self.failure_threshold,
                    )
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            logger.info("circuit_breaker: name=%s manual_reset", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def get_retry_after(self) -> float:
        """Get seconds until circuit might allow requests again."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0
            if self._last_failure_time is None:
                return 0.0
            elapsed = time.time() - self._last_failure_time
            return max(0.0, self.reset_timeout - elapsed)

    def get_stats(self) -> Dict:
        """Get current circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "retry_after": self.get_retry_after(),
            }


# Global registry of circuit breakers
_CIRCUIT_BREAKERS: Dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: Optional[int] = None,
    reset_timeout: Optional[float] = None,
) -> CircuitBreaker:
    """Get or create a named circuit breaker.

    Args:
        name: Unique identifier for the circuit breaker
        failure_threshold: Failures before opening (default from config)
        reset_timeout: Seconds before testing recovery (default from config)

    Returns:
        CircuitBreaker instance (shared across calls with same name)
    """
    with _REGISTRY_LOCK:
        if name not in _CIRCUIT_BREAKERS:
            threshold = (
                failure_threshold if failure_threshold is not None else getattr(config, "CIRCUIT_BREAKER_THRESHOLD", 5)
            )
            timeout = (
                reset_timeout if reset_timeout is not None else getattr(config, "CIRCUIT_BREAKER_RESET_TIMEOUT", 60.0)
            )
            _CIRCUIT_BREAKERS[name] = CircuitBreaker(
                name=name,
                failure_threshold=int(threshold),
                reset_timeout=float(timeout),
            )
            logger.debug("circuit_breaker: created name=%s threshold=%d timeout=%.1fs", name, threshold, timeout)
        return _CIRCUIT_BREAKERS[name]


def reset_circuit_breaker(name: str) -> bool:
    """Reset a specific circuit breaker.

    Args:
        name: Circuit breaker name

    Returns:
        True if circuit was found and reset, False otherwise
    """
    with _REGISTRY_LOCK:
        if name in _CIRCUIT_BREAKERS:
            _CIRCUIT_BREAKERS[name].reset()
            return True
        return False


def reset_all_circuit_breakers() -> int:
    """Reset all circuit breakers.

    Returns:
        Number of circuit breakers reset
    """
    with _REGISTRY_LOCK:
        count = len(_CIRCUIT_BREAKERS)
        for cb in _CIRCUIT_BREAKERS.values():
            cb.reset()
        return count


def circuit_breaker(
    name: str,
    failure_threshold: Optional[int] = None,
    reset_timeout: Optional[float] = None,
    fallback: Optional[Callable[..., T]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to wrap a function with circuit breaker protection.

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        reset_timeout: Seconds before recovery test
        fallback: Optional function to call when circuit is open

    Returns:
        Decorated function with circuit breaker logic

    Example:
        @circuit_breaker("ollama", failure_threshold=3, reset_timeout=30)
        def call_ollama(prompt):
            return ollama_client.generate(prompt)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cb = get_circuit_breaker(name, failure_threshold, reset_timeout)

            if not cb.allow_request():
                retry_after = cb.get_retry_after()
                if fallback is not None:
                    logger.debug("circuit_breaker: name=%s using_fallback retry_after=%.1fs", name, retry_after)
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(name, retry_after)

            try:
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception:
                cb.record_failure()
                raise

        return wrapper

    return decorator


# Convenience circuit breakers for common services
def get_ollama_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Ollama LLM service."""
    return get_circuit_breaker("ollama_llm")


def get_embedding_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for embedding service."""
    return get_circuit_breaker("ollama_embeddings")
