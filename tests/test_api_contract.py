"""API contract tests for Clockify RAG.

These tests verify:
- Request/response schema validation
- HTTP status codes for various scenarios
- Error message formats
- Header handling
- Content-Type enforcement
"""

import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mock index state."""
    from clockify_rag.api import create_app

    app = create_app()
    # Initialize mock index state
    app.state.index_ready = True
    app.state.chunks = [{"id": "test_chunk", "text": "test content"}]
    app.state.vecs_n = MagicMock()
    app.state.bm = MagicMock()
    app.state.hnsw = None

    return TestClient(app)


@pytest.fixture
def client_no_index():
    """Create test client without index ready."""
    from clockify_rag.api import create_app

    app = create_app()
    app.state.index_ready = False
    app.state.chunks = None
    app.state.vecs_n = None
    app.state.bm = None
    app.state.hnsw = None

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health and /v1/health endpoints."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        with patch("clockify_rag.api.check_ollama_connectivity"):
            response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        """Health response should match schema."""
        with patch("clockify_rag.api.check_ollama_connectivity"):
            response = client.get("/health")

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unavailable"]
        assert "timestamp" in data
        assert "version" in data
        assert "platform" in data
        assert "index_ready" in data
        assert "ollama_connected" in data
        assert isinstance(data["index_ready"], bool)
        assert isinstance(data["ollama_connected"], bool)

    def test_health_v1_equivalent(self, client):
        """v1/health should return same structure as /health."""
        with patch("clockify_rag.api.check_ollama_connectivity"):
            response1 = client.get("/health")
            response2 = client.get("/v1/health")

        assert response1.status_code == response2.status_code
        # Both should have same keys
        assert set(response1.json().keys()) == set(response2.json().keys())


class TestConfigEndpoint:
    """Tests for /v1/config endpoint."""

    def test_config_returns_200(self, client):
        """Config endpoint should return 200."""
        response = client.get("/v1/config")
        assert response.status_code == 200

    def test_config_response_schema(self, client):
        """Config response should match schema."""
        response = client.get("/v1/config")
        data = response.json()

        required_fields = [
            "ollama_url",
            "gen_model",
            "emb_model",
            "chunk_size",
            "top_k",
            "pack_top",
            "threshold",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        # Type checks
        assert isinstance(data["ollama_url"], str)
        assert isinstance(data["gen_model"], str)
        assert isinstance(data["emb_model"], str)
        assert isinstance(data["chunk_size"], int)
        assert isinstance(data["top_k"], int)
        assert isinstance(data["pack_top"], int)
        assert isinstance(data["threshold"], float)


class TestQueryEndpoint:
    """Tests for /v1/query endpoint."""

    def test_query_requires_post(self, client):
        """Query endpoint should only accept POST."""
        response = client.get("/v1/query")
        assert response.status_code == 405

    def test_query_requires_body(self, client):
        """Query endpoint should require request body."""
        response = client.post("/v1/query")
        assert response.status_code == 422  # Validation error

    def test_query_requires_question(self, client):
        """Query endpoint should require question field."""
        response = client.post("/v1/query", json={})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_query_rejects_empty_question(self, client):
        """Query should reject empty question."""
        response = client.post("/v1/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_rejects_whitespace_question(self, client):
        """Query should reject whitespace-only question."""
        response = client.post("/v1/query", json={"question": "   "})
        assert response.status_code == 422

    def test_query_rejects_too_long_question(self, client):
        """Query should reject overly long questions."""
        from clockify_rag import config

        long_question = "a" * (config.MAX_QUERY_LENGTH + 1)
        response = client.post("/v1/query", json={"question": long_question})
        assert response.status_code == 422

    def test_query_rejects_xss_script(self, client):
        """Query should reject XSS script tags."""
        response = client.post("/v1/query", json={"question": "<script>alert(1)</script>"})
        assert response.status_code == 422

    def test_query_rejects_javascript_uri(self, client):
        """Query should reject javascript: URIs."""
        response = client.post("/v1/query", json={"question": "javascript:alert(1)"})
        assert response.status_code == 422

    def test_query_503_when_index_not_ready(self, client_no_index):
        """Query should return 503 when index not ready."""
        response = client_no_index.post("/v1/query", json={"question": "test question"})
        assert response.status_code == 503
        assert "index" in response.json()["detail"].lower()

    def test_query_validates_top_k_range(self, client):
        """Query should validate top_k is within range."""
        from clockify_rag import config

        # Too high
        response = client.post("/v1/query", json={"question": "test", "top_k": config.MAX_TOP_K + 1})
        assert response.status_code == 422

        # Too low
        response = client.post("/v1/query", json={"question": "test", "top_k": 0})
        assert response.status_code == 422

    def test_query_validates_threshold_range(self, client):
        """Query should validate threshold is 0-1."""
        # Too high
        response = client.post("/v1/query", json={"question": "test", "threshold": 1.5})
        assert response.status_code == 422

        # Too low
        response = client.post("/v1/query", json={"question": "test", "threshold": -0.1})
        assert response.status_code == 422

    def test_query_success_response_schema(self, client):
        """Successful query should return proper schema."""
        mock_result = {
            "answer": "Test answer",
            "confidence": 85,
            "refused": False,
            "selected_chunks": [0, 1],
            "selected_chunk_ids": ["chunk_1", "chunk_2"],
            "sources_used": ["https://example.com/1"],
            "metadata": {"retrieval_count": 5},
            "routing": {"action": "auto_respond"},
            "timing": {"total_ms": 100, "llm_ms": 50},
        }

        with patch("clockify_rag.api.answer_once", return_value=mock_result):
            response = client.post("/v1/query", json={"question": "How do I track time?"})

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "question" in data
        assert "answer" in data
        assert "timestamp" in data
        assert "processing_time_ms" in data
        assert "sources" in data
        assert "refused" in data

        # Types
        assert isinstance(data["question"], str)
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["processing_time_ms"], (int, float))
        assert isinstance(data["refused"], bool)


class TestIngestEndpoint:
    """Tests for /v1/ingest endpoint."""

    def test_ingest_requires_post(self, client):
        """Ingest endpoint should only accept POST."""
        response = client.get("/v1/ingest")
        assert response.status_code == 405

    def test_ingest_accepts_empty_body(self, client):
        """Ingest should accept empty body (uses defaults)."""
        with patch("clockify_rag.api.resolve_corpus_path") as mock_resolve:
            mock_resolve.return_value = ("knowledge_helpcenter.md", True, ["knowledge_helpcenter.md"])
            response = client.post("/v1/ingest", json={})

        # Should return 200 (processing started) or 404 (file not found)
        assert response.status_code in [200, 404]

    def test_ingest_400_when_input_not_supported(self, client):
        """Ingest should return 400 for unsupported input file."""
        response = client.post("/v1/ingest", json={"input_file": "nonexistent.md"})

        assert response.status_code == 400
        assert "only" in response.json()["detail"].lower()

    def test_ingest_404_when_allowed_file_missing(self, client):
        """Ingest should return 404 when allowed input file is missing."""
        with patch("clockify_rag.api.resolve_corpus_path") as mock_resolve:
            mock_resolve.return_value = ("knowledge_helpcenter.md", False, ["knowledge_helpcenter.md"])
            response = client.post("/v1/ingest", json={"input_file": "knowledge_helpcenter.md"})

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_ingest_response_schema(self, client):
        """Ingest response should match schema."""
        with patch("clockify_rag.api.resolve_corpus_path") as mock_resolve:
            mock_resolve.return_value = ("knowledge_helpcenter.md", True, ["knowledge_helpcenter.md"])
            response = client.post("/v1/ingest", json={})

        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "message" in data
            assert "timestamp" in data
            assert "index_ready" in data


class TestMetricsEndpoint:
    """Tests for /v1/metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """Metrics endpoint should return 200."""
        response = client.get("/v1/metrics")
        assert response.status_code == 200

    def test_metrics_json_format(self, client):
        """Metrics should return JSON by default."""
        response = client.get("/v1/metrics")
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert isinstance(data, dict)

    def test_metrics_prometheus_format(self, client):
        """Metrics should support Prometheus format."""
        response = client.get("/v1/metrics?format=prometheus")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_csv_format(self, client):
        """Metrics should support CSV format."""
        response = client.get("/v1/metrics?format=csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]


class TestErrorResponses:
    """Tests for error response formats."""

    def test_validation_error_format(self, client):
        """Validation errors should have consistent format."""
        response = client.post("/v1/query", json={"question": ""})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_404_error_format(self, client):
        """404 errors should have detail message."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_internal_error_sanitized(self, client):
        """Internal errors should not leak details."""
        with patch("clockify_rag.api.answer_once", side_effect=Exception("Internal DB password: secret123")):
            response = client.post("/v1/query", json={"question": "test question"})

        assert response.status_code == 500
        # Should not contain sensitive info
        detail = response.json().get("detail", "")
        assert "password" not in detail.lower()
        assert "secret" not in detail.lower()


class TestContentTypeHandling:
    """Tests for content-type handling."""

    def test_query_requires_json(self, client):
        """Query should require JSON content-type."""
        response = client.post(
            "/v1/query", content="question=test", headers={"content-type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422

    def test_query_accepts_json(self, client):
        """Query should accept JSON content-type."""
        with patch(
            "clockify_rag.api.answer_once",
            return_value={
                "answer": "test",
                "refused": False,
                "confidence": 80,
                "selected_chunks": [],
                "selected_chunk_ids": [],
                "metadata": {},
                "routing": {},
                "timing": {},
            },
        ):
            response = client.post(
                "/v1/query", json={"question": "test question"}, headers={"content-type": "application/json"}
            )
        assert response.status_code == 200


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_returns_429(self, client):
        """Rate limiting should return 429 status."""
        mock_limiter = MagicMock()
        mock_limiter.allow_request.return_value = False
        mock_limiter.wait_time.return_value = 5.0

        with patch("clockify_rag.api.get_rate_limiter", return_value=mock_limiter):
            response = client.post("/v1/query", json={"question": "test"})

        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()

    def test_rate_limit_includes_retry_after(self, client):
        """Rate limit response should include retry timing."""
        mock_limiter = MagicMock()
        mock_limiter.allow_request.return_value = False
        mock_limiter.wait_time.return_value = 10.5

        with patch("clockify_rag.api.get_rate_limiter", return_value=mock_limiter):
            response = client.post("/v1/query", json={"question": "test"})

        assert response.status_code == 429
        assert "10.50" in response.json()["detail"]


class TestQueryParameterValidation:
    """Tests for query parameter edge cases."""

    def test_accepts_valid_top_k(self, client):
        """Should accept valid top_k values."""
        mock_result = {
            "answer": "Test",
            "refused": False,
            "confidence": 80,
            "selected_chunks": [],
            "selected_chunk_ids": [],
            "metadata": {},
            "routing": {},
            "timing": {},
        }

        with patch("clockify_rag.api.answer_once", return_value=mock_result):
            response = client.post("/v1/query", json={"question": "test", "top_k": 5})
        assert response.status_code == 200

    def test_accepts_valid_threshold(self, client):
        """Should accept valid threshold values."""
        mock_result = {
            "answer": "Test",
            "refused": False,
            "confidence": 80,
            "selected_chunks": [],
            "selected_chunk_ids": [],
            "metadata": {},
            "routing": {},
            "timing": {},
        }

        with patch("clockify_rag.api.answer_once", return_value=mock_result):
            response = client.post("/v1/query", json={"question": "test", "threshold": 0.5})
        assert response.status_code == 200

    def test_accepts_debug_flag(self, client):
        """Should accept debug flag."""
        mock_result = {
            "answer": "Test",
            "refused": False,
            "confidence": 80,
            "selected_chunks": [],
            "selected_chunk_ids": [],
            "metadata": {},
            "routing": {},
            "timing": {},
        }

        with patch("clockify_rag.api.answer_once", return_value=mock_result):
            response = client.post("/v1/query", json={"question": "test", "debug": True})
        assert response.status_code == 200
