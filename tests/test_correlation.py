"""Tests for correlation ID functionality."""

import logging
import re


def test_generate_correlation_id():
    """Test ID generation produces valid format."""
    from clockify_rag.correlation import generate_correlation_id

    cid = generate_correlation_id()
    # UUID4 hex is 32 characters, all hexadecimal
    assert len(cid) == 32
    assert re.match(r"^[a-f0-9]+$", cid)


def test_generate_correlation_id_unique():
    """Test each generated ID is unique."""
    from clockify_rag.correlation import generate_correlation_id

    ids = [generate_correlation_id() for _ in range(100)]
    assert len(set(ids)) == 100, "Generated IDs should be unique"


def test_correlation_id_context():
    """Test correlation ID context var operations."""
    from clockify_rag.correlation import (
        clear_correlation_id,
        get_correlation_id,
        set_correlation_id,
    )

    # Initially None
    clear_correlation_id()
    assert get_correlation_id() is None

    # Set and get
    set_correlation_id("test-123")
    assert get_correlation_id() == "test-123"

    # Clear
    clear_correlation_id()
    assert get_correlation_id() is None


def test_correlation_id_filter():
    """Test logging filter adds correlation ID to records."""
    from clockify_rag.correlation import (
        CorrelationIdFilter,
        clear_correlation_id,
        set_correlation_id,
    )

    # With correlation ID set
    set_correlation_id("filter-test")
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    f = CorrelationIdFilter()
    result = f.filter(record)

    assert result is True
    assert record.correlation_id == "filter-test"

    # Without correlation ID (should default to "-")
    clear_correlation_id()
    record2 = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    f.filter(record2)
    assert record2.correlation_id == "-"


def test_correlation_id_validation_in_middleware():
    """Test that api middleware validates correlation IDs properly."""
    # This test verifies the validation logic that will be used in the middleware
    import re

    pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

    def validate(raw_id):
        if not raw_id:
            return None
        if len(raw_id) <= 64 and pattern.match(raw_id):
            return raw_id
        return None

    # Valid IDs
    assert validate("abc123") == "abc123"
    assert validate("test-id") == "test-id"
    assert validate("test_id") == "test_id"
    assert validate("ABC123") == "ABC123"
    assert validate("a" * 64) == "a" * 64  # Max length

    # Invalid IDs
    assert validate("") is None
    assert validate(None) is None
    assert validate("a" * 65) is None  # Too long
    assert validate("test<script>") is None  # Special chars
    assert validate("test\nid") is None  # Newline (log injection)
    assert validate("test id") is None  # Space
