"""Tests for query log rotation."""

import json


def test_query_logger_creates_file(tmp_path):
    """Test RotatingFileHandler creates log file."""
    from clockify_rag.logging_config import get_query_logger, reset_query_logger

    # Reset to ensure fresh logger
    reset_query_logger()

    log_file = tmp_path / "test_queries.jsonl"
    logger = get_query_logger(str(log_file), max_bytes=1024, backup_count=2)
    logger.info('{"test": "entry"}')

    # Flush handlers
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists(), "Log file should be created"


def test_query_logger_writes_json(tmp_path):
    """Test query logger writes valid JSONL."""
    from clockify_rag.logging_config import get_query_logger, reset_query_logger

    reset_query_logger()

    log_file = tmp_path / "test_queries.jsonl"
    logger = get_query_logger(str(log_file), max_bytes=1024, backup_count=2)

    # Write a JSON entry
    entry = {"question": "test", "answer": "response"}
    logger.info(json.dumps(entry))

    # Flush
    for handler in logger.handlers:
        handler.flush()

    # Read and verify
    content = log_file.read_text()
    assert "test" in content
    assert "response" in content


def test_query_logger_path_change(tmp_path):
    """Test logger recreates when path changes."""
    from clockify_rag.logging_config import get_query_logger, reset_query_logger

    reset_query_logger()

    # Create first logger
    log_file1 = tmp_path / "queries1.jsonl"
    logger1 = get_query_logger(str(log_file1))
    logger1.info('{"file": "1"}')
    for h in logger1.handlers:
        h.flush()

    # Create second logger with different path
    log_file2 = tmp_path / "queries2.jsonl"
    logger2 = get_query_logger(str(log_file2))
    logger2.info('{"file": "2"}')
    for h in logger2.handlers:
        h.flush()

    # Both files should exist with their respective content
    assert log_file1.exists()
    assert log_file2.exists()
    assert '"file": "1"' in log_file1.read_text()
    assert '"file": "2"' in log_file2.read_text()


def test_query_logger_singleton_same_path(tmp_path):
    """Test logger returns same instance for same path."""
    from clockify_rag.logging_config import get_query_logger, reset_query_logger

    reset_query_logger()

    log_file = tmp_path / "queries.jsonl"
    logger1 = get_query_logger(str(log_file))
    logger2 = get_query_logger(str(log_file))

    assert logger1 is logger2, "Same path should return same logger instance"
