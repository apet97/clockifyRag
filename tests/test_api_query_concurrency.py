"""Concurrency regression tests for /v1/query."""

import asyncio
import time

import httpx
import pytest
from asgi_lifespan import LifespanManager

import clockify_rag.api as api_module


@pytest.mark.asyncio
async def test_query_concurrency_latency(monkeypatch):
    """Concurrent requests should not serialize when work runs in threadpool."""

    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: ([], [], {}, None))

    delay = 0.05
    request_count = 4

    def slow_answer(*_, **__):
        time.sleep(delay)
        return {
            "answer": "ok",
            "selected_chunks": [1, 2],
            "metadata": {"test": True},
        }

    monkeypatch.setattr(api_module, "answer_once", slow_answer)

    app = api_module.create_app()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Establish sequential baseline (simulates blocking behaviour)
            baseline_start = time.perf_counter()
            for idx in range(request_count):
                response = await client.post("/v1/query", json={"question": f"seq-{idx}"})
                assert response.status_code == 200
            baseline_elapsed = time.perf_counter() - baseline_start

            # Fire concurrent requests and expect combined latency to drop substantially
            concurrent_start = time.perf_counter()
            responses = await asyncio.gather(
                *[client.post("/v1/query", json={"question": f"concurrent-{idx}"}) for idx in range(request_count)]
            )
            concurrent_elapsed = time.perf_counter() - concurrent_start

            for response in responses:
                assert response.status_code == 200

    assert (
        concurrent_elapsed <= baseline_elapsed * 0.6
    ), f"concurrent latency {concurrent_elapsed:.3f}s should be well below baseline {baseline_elapsed:.3f}s"
