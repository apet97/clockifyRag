"""
Regression tests for chunk logging toggle (RAG_LOG_INCLUDE_CHUNKS).

Addresses Analysis Report Priority #6: Test that chunk text redaction honors
the LOG_QUERY_INCLUDE_CHUNKS flag independently of LOG_QUERY_INCLUDE_ANSWER.
"""

import json
import os
import tempfile
import pytest
import clockify_rag.config as config_module


def test_chunk_logging_disabled_by_default(monkeypatch):
    """Verify that chunk text is redacted from logs by default."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
        log_file = f.name

    try:
        # Set log file BEFORE reloading config and force logging on for test
        monkeypatch.setenv("RAG_LOG_FILE", log_file)
        monkeypatch.setenv("RAG_LOG_ENABLED", "1")
        monkeypatch.setattr(config_module, "QUERY_LOG_ENABLED", True, raising=False)

        # Reload config to pick up env var
        import importlib
        import clockify_rag.config

        importlib.reload(clockify_rag.config)

        from clockify_rag.caching import log_query
        from clockify_rag.config import LOG_QUERY_INCLUDE_CHUNKS

        # Default should be False (redacted)
        assert not LOG_QUERY_INCLUDE_CHUNKS, "Chunks should be redacted by default for security"

        # Log a query with chunk data
        test_chunks = [
            {
                "id": "test-123",
                "dense": 0.8,
                "bm25": 0.7,
                "hybrid": 0.75,
                "chunk": "Sensitive chunk text that should be redacted",
                "text": "More sensitive text",
            }
        ]

        log_query(
            query="test question",
            answer="test answer",
            retrieved_chunks=test_chunks,
            latency_ms=100.0,
            refused=False,
            metadata={"test": True},
        )

        # Read and verify log entry
        with open(log_file, "r") as f:
            log_entry = json.loads(f.readline())

        # Verify chunks are in log but text is redacted
        assert len(log_entry["retrieved_chunks"]) == 1
        chunk_logged = log_entry["retrieved_chunks"][0]
        assert chunk_logged["id"] == "test-123"
        assert "chunk" not in chunk_logged, "chunk field should be redacted when LOG_QUERY_INCLUDE_CHUNKS=False"
        assert "text" not in chunk_logged, "text field should be redacted when LOG_QUERY_INCLUDE_CHUNKS=False"

    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)


def test_chunk_logging_enabled_when_flag_set(monkeypatch):
    """Verify that chunk text is included when RAG_LOG_INCLUDE_CHUNKS=1."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
        log_file = f.name

    try:
        # Set BOTH env vars BEFORE reloading config
        monkeypatch.setenv("RAG_LOG_INCLUDE_CHUNKS", "1")
        monkeypatch.setenv("RAG_LOG_FILE", log_file)
        monkeypatch.setenv("RAG_LOG_ENABLED", "1")
        monkeypatch.setattr(config_module, "QUERY_LOG_ENABLED", True, raising=False)

        # Reload config to pick up env vars
        import importlib
        import clockify_rag.config

        importlib.reload(clockify_rag.config)

        from clockify_rag.caching import log_query
        from clockify_rag.config import LOG_QUERY_INCLUDE_CHUNKS

        assert LOG_QUERY_INCLUDE_CHUNKS, "Chunks should be included when flag is set"

        # Log a query with chunk data
        test_chunks = [
            {
                "id": "test-456",
                "dense": 0.9,
                "bm25": 0.8,
                "hybrid": 0.85,
                "chunk": "This chunk text should be preserved",
                "text": "This text should also be preserved",
            }
        ]

        log_query(
            query="test question 2",
            answer="test answer 2",
            retrieved_chunks=test_chunks,
            latency_ms=150.0,
            refused=False,
            metadata={"test": True},
        )

        # Read and verify log entry
        with open(log_file, "r") as f:
            log_entry = json.loads(f.readline())

        # Verify chunks and their text are preserved
        assert len(log_entry["retrieved_chunks"]) == 1
        chunk_logged = log_entry["retrieved_chunks"][0]
        assert chunk_logged["id"] == "test-456"
        assert "chunk" in chunk_logged, "chunk field should be present when LOG_QUERY_INCLUDE_CHUNKS=1"
        assert chunk_logged["chunk"] == "This chunk text should be preserved"

    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)
        # Reset env vars
        monkeypatch.delenv("RAG_LOG_INCLUDE_CHUNKS", raising=False)


def test_chunk_logging_independent_of_answer_logging(monkeypatch):
    """Verify chunk logging is independent of answer logging flag."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
        log_file = f.name

    try:
        # Set ALL env vars BEFORE reloading config
        monkeypatch.setenv("RAG_LOG_INCLUDE_ANSWER", "0")
        monkeypatch.setenv("RAG_LOG_INCLUDE_CHUNKS", "1")
        monkeypatch.setenv("RAG_LOG_FILE", log_file)
        monkeypatch.setenv("RAG_LOG_ENABLED", "1")
        monkeypatch.setattr(config_module, "QUERY_LOG_ENABLED", True, raising=False)

        # Reload config to pick up env vars
        import importlib
        import clockify_rag.config

        importlib.reload(clockify_rag.config)

        from clockify_rag.caching import log_query
        from clockify_rag.config import LOG_QUERY_INCLUDE_ANSWER, LOG_QUERY_INCLUDE_CHUNKS

        assert not LOG_QUERY_INCLUDE_ANSWER, "Answer should be redacted"
        assert LOG_QUERY_INCLUDE_CHUNKS, "Chunks should be included"

        # Log a query with chunk data
        test_chunks = [
            {
                "id": "test-789",
                "dense": 0.7,
                "bm25": 0.6,
                "hybrid": 0.65,
                "chunk": "Chunk text should be included despite answer redaction",
            }
        ]

        log_query(
            query="test question 3",
            answer="Sensitive answer that should be redacted",
            retrieved_chunks=test_chunks,
            latency_ms=200.0,
            refused=False,
            metadata={"test": True},
        )

        # Read and verify log entry
        with open(log_file, "r") as f:
            log_entry = json.loads(f.readline())

        # Verify answer is redacted
        assert log_entry["answer"] == "[REDACTED]", "Answer should be redacted when LOG_QUERY_INCLUDE_ANSWER=0"

        # Verify chunks are included (independent of answer flag)
        assert len(log_entry["retrieved_chunks"]) == 1
        chunk_logged = log_entry["retrieved_chunks"][0]
        assert (
            "chunk" in chunk_logged
        ), "chunk field should be present when LOG_QUERY_INCLUDE_CHUNKS=1, regardless of answer flag"
        assert chunk_logged["chunk"] == "Chunk text should be included despite answer redaction"

    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)
        # Reset env vars
        monkeypatch.delenv("RAG_LOG_INCLUDE_ANSWER", raising=False)
        monkeypatch.delenv("RAG_LOG_INCLUDE_CHUNKS", raising=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
