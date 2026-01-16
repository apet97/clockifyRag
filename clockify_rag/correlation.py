"""Correlation ID management for request tracing.

This module provides thread-safe correlation ID management using ContextVar,
allowing correlation IDs to be propagated across async contexts and thread
pool executors without explicit passing.

Usage:
    from clockify_rag.correlation import get_correlation_id, set_correlation_id

    # In middleware/request handler
    set_correlation_id("req-abc123")

    # Anywhere in the request context
    corr_id = get_correlation_id()  # Returns "req-abc123"
"""

import re
import uuid
from contextvars import ContextVar
from typing import Optional

# Pattern for safe correlation IDs: alphanumeric, dash, underscore only
_SAFE_CORRELATION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Thread-safe context variable for correlation ID
# Defaults to None when not set (e.g., outside of request context)
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID.

    Returns:
        A unique correlation ID string (UUID4 hex, 32 chars)
    """
    return uuid.uuid4().hex


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context.

    Returns:
        The current correlation ID, or None if not set
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set the correlation ID in the current context.

    Args:
        correlation_id: The correlation ID to set, or None to clear
    """
    _correlation_id.set(correlation_id)


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    _correlation_id.set(None)


def validate_correlation_id(raw_id: Optional[str], max_length: int = 64) -> Optional[str]:
    """Validate correlation ID to prevent log injection and oversized headers.

    Args:
        raw_id: Raw correlation ID from request header
        max_length: Maximum allowed length (default 64)

    Returns:
        Validated ID if safe, None otherwise
    """
    if not raw_id:
        return None
    if len(raw_id) <= max_length and _SAFE_CORRELATION_ID_PATTERN.match(raw_id):
        return raw_id
    return None


class CorrelationIdFilter:
    """Logging filter that adds correlation_id to log records.

    This filter can be added to any handler to include correlation IDs
    in log output without modifying existing logging calls.

    Example:
        handler.addFilter(CorrelationIdFilter())
    """

    def filter(self, record) -> bool:
        """Add correlation_id to the log record.

        Args:
            record: The log record to modify

        Returns:
            True (always allows the record through)
        """
        record.correlation_id = get_correlation_id() or "-"
        return True
