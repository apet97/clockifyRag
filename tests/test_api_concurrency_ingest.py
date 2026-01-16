"""Concurrency tests for /v1/ingest endpoint and state management.

Validates thread-safe state updates when queries run concurrently with ingest operations.
Tests the threading.RLock() protection added to api.py for app.state management.
"""

import asyncio
import threading
import time
from typing import List, Dict, Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

import clockify_rag.api as api_module
from clockify_rag.utils import ALLOWED_CORPUS_FILENAME


@pytest.mark.asyncio
async def test_query_during_ingest_sees_consistent_state(monkeypatch, tmp_path):
    """Queries during ingest should see consistent state (either old or new, never partial)."""

    # Create a test knowledge file
    knowledge_file = tmp_path / ALLOWED_CORPUS_FILENAME
    knowledge_file.write_text("# Test\n\nSome content for testing.")

    # Mock index states: "old" and "new"
    old_state = (["chunk1"], [[0.1, 0.2]], {"bm25": "old"}, {"hnsw": "old"})
    new_state = (["chunk1", "chunk2"], [[0.1, 0.2], [0.3, 0.4]], {"bm25": "new"}, {"hnsw": "new"})

    # Track state transitions to detect partial updates
    state_transitions: List[Dict[str, Any]] = []
    state_lock = threading.Lock()

    def mock_build(input_file, retries=2):
        """Mock build that takes time (simulates real index build)."""
        time.sleep(0.1)  # Simulate build work
        return None

    def mock_ensure_index_ready(retries=2):
        """Return new state after build completes."""
        return new_state

    def mock_answer_once(question, chunks, vecs_n, bm, *args, **kwargs):
        """Capture state snapshot during query execution."""
        with state_lock:
            state_transitions.append(
                {
                    "question": question,
                    "chunks_count": len(chunks) if chunks else 0,
                    "vecs_count": len(vecs_n) if vecs_n else 0,
                    "bm_version": bm.get("bm25") if isinstance(bm, dict) else None,
                }
            )

        # Simulate query processing time
        time.sleep(0.02)

        return {
            "answer": f"Answer for: {question}",
            "selected_chunks": [0],
            "metadata": {"chunks": len(chunks) if chunks else 0},
        }

    monkeypatch.setattr(api_module, "build", mock_build)
    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: old_state)
    monkeypatch.setattr(api_module, "answer_once", mock_answer_once)
    monkeypatch.setenv("COVERAGE_MIN_CHUNKS", "0")  # Allow queries with minimal chunks

    app = api_module.create_app()

    async with LifespanManager(app):
        # Manually set initial state (simulates startup)
        app.state.chunks, app.state.vecs_n, app.state.bm, app.state.hnsw = old_state
        app.state.index_ready = True

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=5.0) as client:
            # Start ingest in background (will take ~100ms)
            # Reset ensure_index_ready to return new state after ingest
            monkeypatch.setattr(api_module, "ensure_index_ready", mock_ensure_index_ready)

            ingest_task = asyncio.create_task(client.post("/v1/ingest", json={"input_file": str(knowledge_file)}))

            # Fire multiple concurrent queries during ingest
            # Some should see old state, some should see new state
            # But NONE should see partial/inconsistent state
            await asyncio.sleep(0.01)  # Let ingest start

            query_tasks = [client.post("/v1/query", json={"question": f"query-{i}"}) for i in range(10)]

            query_responses = await asyncio.gather(*query_tasks, return_exceptions=True)
            ingest_response = await ingest_task

            # Validate ingest succeeded
            assert ingest_response.status_code == 200

            # Validate all queries succeeded (no 503 or race condition errors)
            for idx, response in enumerate(query_responses):
                assert not isinstance(response, Exception), f"Query {idx} raised exception: {response}"
                # Queries may return 503 if they ran before index was ready, or 200 if after
                assert response.status_code in (200, 503), f"Query {idx} got unexpected status: {response.status_code}"

            # Validate state consistency: all snapshots should be either old or new, never partial
            with state_lock:
                for transition in state_transitions:
                    chunks_count = transition["chunks_count"]
                    vecs_count = transition["vecs_count"]
                    bm_version = transition["bm_version"]

                    # State should be consistent: either old (1 chunk) or new (2 chunks)
                    assert chunks_count in (1, 2), f"Partial state detected: {transition}"
                    assert vecs_count in (1, 2), f"Partial state detected: {transition}"

                    # If chunks is old, vecs and bm should also be old
                    if chunks_count == 1:
                        assert vecs_count == 1, f"Inconsistent old state: {transition}"
                        assert bm_version == "old", f"Inconsistent old state: {transition}"

                    # If chunks is new, vecs and bm should also be new
                    if chunks_count == 2:
                        assert vecs_count == 2, f"Inconsistent new state: {transition}"
                        assert bm_version == "new", f"Inconsistent new state: {transition}"


@pytest.mark.asyncio
async def test_multiple_concurrent_ingests_serialized(monkeypatch, tmp_path):
    """Multiple concurrent ingests should be serialized (no race on state updates)."""

    knowledge_file = tmp_path / ALLOWED_CORPUS_FILENAME
    knowledge_file.write_text("# Test\n\nContent for testing.")

    build_call_count = 0
    build_call_times: List[float] = []

    def mock_build(input_file, retries=2):
        """Mock build that tracks call timing."""
        nonlocal build_call_count
        build_call_count += 1
        build_call_times.append(time.time())
        time.sleep(0.05)  # Simulate build work
        return None

    def mock_ensure_index_ready(retries=2):
        """Return mock state."""
        return (["chunk"], [[0.1]], {}, None)

    monkeypatch.setattr(api_module, "build", mock_build)
    monkeypatch.setattr(api_module, "ensure_index_ready", mock_ensure_index_ready)

    app = api_module.create_app()

    async with LifespanManager(app):
        app.state.chunks = []
        app.state.index_ready = False

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=10.0) as client:
            # Fire 5 concurrent ingest requests
            ingest_tasks = [client.post("/v1/ingest", json={"input_file": str(knowledge_file)}) for _ in range(5)]

            responses = await asyncio.gather(*ingest_tasks)

            # All should succeed
            for response in responses:
                assert response.status_code == 200

            # Wait for all background tasks to complete
            await asyncio.sleep(0.5)

            # Validate all builds executed
            assert build_call_count == 5, "All 5 ingests should trigger builds"

            # Validate builds didn't run completely in parallel (state lock serializes)
            # If fully parallel, max time would be ~50ms
            # If serialized, max time would be ~250ms (5 * 50ms)
            # We expect somewhere in between due to background task scheduling
            if len(build_call_times) >= 2:
                time_spread = max(build_call_times) - min(build_call_times)
                # Expect at least some serialization (> 100ms spread for 5 builds)
                assert time_spread >= 0.1, f"Builds appear too parallel: {time_spread:.3f}s spread"


@pytest.mark.asyncio
async def test_state_lock_prevents_torn_reads(monkeypatch):
    """Validates that state reads capture atomic snapshots."""

    # Create two distinct states
    state1 = (["chunk1"], [[0.1, 0.2]], {"id": "state1"}, {"hnsw": "state1"})
    state2 = (["chunk2", "chunk3"], [[0.3, 0.4], [0.5, 0.6]], {"id": "state2"}, {"hnsw": "state2"})

    torn_reads_detected = []
    query_count = 0

    def mock_answer_once(question, chunks, vecs_n, bm, *args, hnsw=None, **kwargs):
        """Detect torn reads (mismatched state components)."""
        nonlocal query_count
        query_count += 1

        # Validate consistency
        chunks_id = f"chunk{len(chunks)}" if chunks and len(chunks) == 1 else "chunk2"
        bm_id = bm.get("id") if isinstance(bm, dict) else None
        hnsw_id = hnsw.get("hnsw") if isinstance(hnsw, dict) else None

        # All components should come from same state
        expected_state = "state1" if chunks_id == "chunk1" else "state2"
        if bm_id != expected_state or hnsw_id != expected_state:
            torn_reads_detected.append(
                {
                    "chunks": chunks_id,
                    "bm_id": bm_id,
                    "hnsw_id": hnsw_id,
                }
            )

        return {
            "answer": "ok",
            "selected_chunks": [0],
            "metadata": {},
        }

    # Mock that alternates states rapidly
    state_toggle = [True]

    def mock_ensure_index_ready(retries=2):
        state_toggle[0] = not state_toggle[0]
        return state1 if state_toggle[0] else state2

    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: state1)
    monkeypatch.setattr(api_module, "answer_once", mock_answer_once)

    app = api_module.create_app()

    async with LifespanManager(app):
        app.state.chunks, app.state.vecs_n, app.state.bm, app.state.hnsw = state1
        app.state.index_ready = True

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=5.0) as client:
            # Rapidly alternate state and fire queries
            async def flip_state():
                """Alternate state every 10ms."""
                for _ in range(10):
                    await asyncio.sleep(0.01)
                    app.state.chunks, app.state.vecs_n, app.state.bm, app.state.hnsw = (
                        state2 if app.state.chunks == state1[0] else state1
                    )
                    app.state.index_ready = True

            # Fire queries concurrently with state flipping
            query_tasks = [client.post("/v1/query", json={"question": f"query-{i}"}) for i in range(20)]

            flip_task = asyncio.create_task(flip_state())
            responses = await asyncio.gather(*query_tasks, return_exceptions=True)
            await flip_task

            # All queries should succeed
            for response in responses:
                if not isinstance(response, Exception):
                    assert response.status_code == 200

            # CRITICAL: No torn reads should be detected (lock ensures atomic snapshots)
            assert len(torn_reads_detected) == 0, f"Torn reads detected: {torn_reads_detected}"
            assert query_count == 20, "All queries should have executed"


@pytest.mark.asyncio
async def test_health_check_concurrent_with_ingest(monkeypatch, tmp_path):
    """Health checks during ingest should not raise exceptions or see partial state."""

    knowledge_file = tmp_path / ALLOWED_CORPUS_FILENAME
    knowledge_file.write_text("# Test\n\nContent")

    def mock_build(input_file, retries=2):
        time.sleep(0.1)
        return None

    def mock_ensure_index_ready(retries=2):
        return (["chunk"], [[0.1]], {}, None)

    monkeypatch.setattr(api_module, "build", mock_build)
    monkeypatch.setattr(api_module, "ensure_index_ready", mock_ensure_index_ready)
    monkeypatch.setattr(api_module, "check_ollama_connectivity", lambda *args: None)

    app = api_module.create_app()

    async with LifespanManager(app):
        app.state.chunks = ["old"]
        app.state.index_ready = True

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=5.0) as client:
            # Start ingest
            ingest_task = asyncio.create_task(client.post("/v1/ingest", json={"input_file": str(knowledge_file)}))

            # Hammer health endpoint during ingest
            health_tasks = [client.get("/health") for _ in range(20)]

            health_responses = await asyncio.gather(*health_tasks, return_exceptions=True)
            await ingest_task

            # All health checks should succeed
            for idx, response in enumerate(health_responses):
                assert not isinstance(response, Exception), f"Health check {idx} raised: {response}"
                assert response.status_code == 200, f"Health check {idx} failed: {response.status_code}"

                # Validate response structure
                data = response.json()
                assert "status" in data
                assert "index_ready" in data
                assert isinstance(data["index_ready"], bool)


@pytest.mark.asyncio
async def test_metrics_endpoint_concurrent_with_ingest(monkeypatch, tmp_path):
    """Metrics endpoint should safely read state during concurrent ingest."""

    knowledge_file = tmp_path / ALLOWED_CORPUS_FILENAME
    knowledge_file.write_text("# Test\n\nContent")

    def mock_build(input_file, retries=2):
        time.sleep(0.05)
        return None

    def mock_ensure_index_ready(retries=2):
        return (["chunk1", "chunk2"], [[0.1], [0.2]], {}, None)

    monkeypatch.setattr(api_module, "build", mock_build)
    monkeypatch.setattr(api_module, "ensure_index_ready", mock_ensure_index_ready)

    app = api_module.create_app()

    async with LifespanManager(app):
        app.state.chunks = ["old"]
        app.state.index_ready = True

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=5.0) as client:
            # Start ingest
            ingest_task = asyncio.create_task(client.post("/v1/ingest", json={"input_file": str(knowledge_file)}))

            # Query metrics during ingest
            metrics_tasks = [client.get("/v1/metrics") for _ in range(10)]

            metrics_responses = await asyncio.gather(*metrics_tasks, return_exceptions=True)
            await ingest_task

            # All metrics requests should succeed
            for idx, response in enumerate(metrics_responses):
                assert not isinstance(response, Exception), f"Metrics {idx} raised: {response}"
                assert response.status_code == 200

                data = response.json()
                assert "index_ready" in data
                assert "chunks_loaded" in data

                # chunks_loaded should be consistent: either 1 (old) or 2 (new)
                chunks_loaded = data["chunks_loaded"]
                assert chunks_loaded in (1, 2), f"Partial state in metrics: {chunks_loaded}"
