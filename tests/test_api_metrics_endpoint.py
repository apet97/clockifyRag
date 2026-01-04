"""Integration tests for the metrics endpoint."""

import json
from typing import Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

import clockify_rag.api as api_module
import clockify_rag.answer as answer_module
from clockify_rag.metrics import MetricsCollector, MetricNames
from clockify_rag.caching import RateLimiter


@pytest.fixture
def reset_metrics(monkeypatch):
    """Provide a fresh metrics collector for each test."""
    import clockify_rag.metrics as metrics_module

    collector = MetricsCollector()
    monkeypatch.setattr(metrics_module, "_global_metrics", collector, raising=False)
    monkeypatch.setattr(metrics_module, "get_metrics", lambda: collector, raising=False)
    monkeypatch.setattr(api_module, "get_metrics", lambda: collector, raising=False)
    return collector


@pytest.fixture
def patched_pipeline(monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
    """Patch answer pipeline and index loading to deterministic fakes."""

    # Ensure API uses a predictable index snapshot
    monkeypatch.setattr(
        api_module,
        "ensure_index_ready",
        lambda retries=2: (sample_chunks, sample_embeddings, sample_bm25, None),
    )

    # Provide a permissive rate limiter instance for API tests
    monkeypatch.setattr(
        api_module,
        "get_rate_limiter",
        lambda: RateLimiter(max_requests=1000, window_seconds=60.0),
    )

    # Patch retrieval pipeline components used by answer_once
    def fake_retrieve(_question, _chunks, _vecs, _bm, **_kwargs):
        dense_scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        scores = {
            "dense": dense_scores,
            "bm25": dense_scores,
            "hybrid": dense_scores,
        }
        return [0, 1, 2], scores

    coverage_calls = {"count": 0}

    def fake_coverage_ok(_selected, _scores, _threshold):
        coverage_calls["count"] += 1
        # First call succeeds, second call forces a refusal
        return coverage_calls["count"] == 1

    def fake_pack_snippets(chunks, selected, pack_top=8, num_ctx=2048):
        packed_ids = [chunks[idx]["id"] for idx in selected[:pack_top]]
        return "context", packed_ids, 42, []

    def fake_rerank(question, chunks, mmr_selected, scores, seed=None, num_ctx=None, num_predict=None, retries=None):
        return mmr_selected, {}, False, "disabled"

    def fake_ask_llm(_question, _context, **_kwargs):
        return json.dumps({"answer": "Use the timer [1]", "confidence": 55})

    monkeypatch.setattr(answer_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(answer_module, "coverage_ok", fake_coverage_ok)
    monkeypatch.setattr(answer_module, "pack_snippets", fake_pack_snippets)
    monkeypatch.setattr(answer_module, "rerank_with_llm", fake_rerank)
    monkeypatch.setattr(answer_module, "ask_llm", fake_ask_llm)

    return {"coverage_calls": coverage_calls}


def _counter_total(counters: Dict[str, float], prefix: str) -> float:
    """Aggregate counters with optional Prometheus-style labels."""
    total = 0.0
    for key, value in counters.items():
        if key == prefix or key.startswith(f"{prefix}{{"):
            total += value
    return total


def test_metrics_endpoint_tracks_pipeline(reset_metrics, patched_pipeline):
    """Metrics endpoint should reflect query volume and refusals."""

    app = api_module.create_app()

    with TestClient(app) as client:
        # Baseline metrics before any query
        baseline = client.get("/v1/metrics").json()
        base_queries = baseline["counters"].get(MetricNames.QUERIES_TOTAL, 0)

        # First query succeeds
        response_ok = client.post("/v1/query", json={"question": "How do I track time?"})
        assert response_ok.status_code == 200

        metrics_after_first = client.get("/v1/metrics").json()
        assert metrics_after_first["counters"].get(MetricNames.QUERIES_TOTAL, 0) == base_queries + 1

        # Second query triggers coverage refusal via patched pipeline
        response_refused = client.post("/v1/query", json={"question": "How do I track time?"})
        assert response_refused.status_code == 200

        metrics_after_second = client.get("/v1/metrics").json()
        assert metrics_after_second["counters"].get(MetricNames.QUERIES_TOTAL, 0) == base_queries + 2
        refusals_total = _counter_total(metrics_after_second["counters"], MetricNames.REFUSALS_TOTAL)
        assert refusals_total >= 1

        # Latency histograms should record observations
        hist_stats = metrics_after_second.get("histogram_stats", {})
        assert MetricNames.QUERY_LATENCY in hist_stats
        assert hist_stats[MetricNames.QUERY_LATENCY]["count"] >= 2
        assert MetricNames.RETRIEVAL_LATENCY in hist_stats

        # Rate limiter metrics should reflect allowed requests
        allowed_total = _counter_total(metrics_after_second["counters"], MetricNames.RATE_LIMIT_ALLOWED)
        assert allowed_total >= 2

        # Prometheus format should be available and include key metrics
        prom_response = client.get("/v1/metrics", params={"format": "prometheus"})
        assert prom_response.status_code == 200
        assert "queries_total" in prom_response.text
        assert prom_response.headers["content-type"].startswith("text/plain")

        # CSV export should also be available
        csv_response = client.get("/v1/metrics", params={"format": "csv"})
        assert csv_response.status_code == 200
        assert "metric_type,metric_name" in csv_response.text
