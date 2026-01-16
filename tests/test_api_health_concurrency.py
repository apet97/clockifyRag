"""Concurrency regression tests for /health."""

import asyncio
import time

import httpx
import pytest
from asgi_lifespan import LifespanManager

import clockify_rag.api as api_module


@pytest.mark.asyncio
async def test_health_concurrency_latency(monkeypatch, tmp_path):
    """Health checks should execute Ollama probe in threadpool."""

    # Pretend the index is ready so the health endpoint reaches the Ollama probe.
    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: ([], [], {}, None))

    # Satisfy the index file existence check performed by the endpoint.
    monkeypatch.chdir(tmp_path)
    for name in ["chunks.jsonl", "vecs_n.npy", "meta.jsonl", "bm25.json"]:
        (tmp_path / name).write_text("ok")

    delay = 0.05
    request_count = 4

    def slow_probe(*_, **__):
        time.sleep(delay)
        return "http://ollama.test"

    monkeypatch.setattr(api_module, "check_ollama_connectivity", slow_probe)

    app = api_module.create_app()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            baseline_start = time.perf_counter()
            for _ in range(request_count):
                response = await client.get("/health")
                assert response.status_code == 200
            baseline_elapsed = time.perf_counter() - baseline_start

            concurrent_start = time.perf_counter()
            responses = await asyncio.gather(*[client.get("/health") for _ in range(request_count)])
            concurrent_elapsed = time.perf_counter() - concurrent_start

            for response in responses:
                assert response.status_code == 200

    max_factor = 0.8
    assert (
        concurrent_elapsed <= baseline_elapsed * max_factor
    ), f"concurrent latency {concurrent_elapsed:.3f}s should be well below baseline {baseline_elapsed:.3f}s (factor {max_factor}x)"
