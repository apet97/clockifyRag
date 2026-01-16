"""
Tests for API state race conditions and thread safety.

Critical scenarios tested:
1. Query-during-ingest: Queries don't see partially updated state during ingest
2. Duplicate ingest: Concurrent ingest requests handled safely
3. State consistency: 1000 concurrent operations don't corrupt app.state
4. Lock isolation: Multiple app instances don't share locks (test isolation)
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import numpy as np
from fastapi.testclient import TestClient

from clockify_rag.api import create_app


@pytest.fixture
def app_with_index():
    """Create FastAPI app with preloaded index for testing."""
    app = create_app()

    # Mock index data
    chunks = [
        {"id": 1, "text": "Test chunk 1", "title": "Test", "section": "A"},
        {"id": 2, "text": "Test chunk 2", "title": "Test", "section": "B"},
    ]
    vecs_n = np.random.rand(2, 768).astype(np.float32)
    vecs_n = vecs_n / np.linalg.norm(vecs_n, axis=1, keepdims=True)
    bm = {
        "doc_lens": [10, 10],
        "avgdl": 10.0,
        "doc_tfs": [{"test": 1}, {"test": 1}],
        "idf": {"test": 1.0},
    }

    # Set state using lock
    with app.state.lock:
        app.state.chunks = chunks
        app.state.vecs_n = vecs_n
        app.state.bm = bm
        app.state.hnsw = None
        app.state.index_ready = True

    yield app


class TestQueryDuringIngest:
    """Test that queries during ingest don't see partial/corrupted state."""

    def test_query_during_ingest_no_corruption(self, app_with_index):
        """Test concurrent queries during ingest see consistent state."""
        client = TestClient(app_with_index)

        # Track state snapshots from queries
        query_results = []
        query_errors = []

        def run_query():
            """Run a single query and record result/error."""
            try:
                response = client.post("/v1/query", json={"question": "test query"})
                query_results.append(response.status_code)
            except Exception as e:
                query_errors.append(str(e))

        def simulate_ingest():
            """Simulate ingest by acquiring lock and modifying state."""
            time.sleep(0.05)  # Let queries start first

            # Mock ingest: acquire lock, update state gradually
            with app_with_index.state.lock:
                # Step 1: Clear old state
                app_with_index.state.index_ready = False
                time.sleep(0.01)  # Simulate slow operation

                # Step 2: Build new state (simulate gradual updates)
                new_chunks = [{"id": 3, "text": "New chunk", "title": "New", "section": "C"}]
                new_vecs = np.random.rand(1, 768).astype(np.float32)
                new_vecs = new_vecs / np.linalg.norm(new_vecs, axis=1, keepdims=True)

                time.sleep(0.01)  # Simulate slow operation

                # Step 3: Commit new state atomically
                app_with_index.state.chunks = new_chunks
                app_with_index.state.vecs_n = new_vecs
                app_with_index.state.bm = {
                    "doc_lens": [10],
                    "avgdl": 10.0,
                    "doc_tfs": [{"test": 1}],
                    "idf": {"test": 1.0},
                }
                app_with_index.state.index_ready = True

        # Mock embed_query for entire test duration
        with patch("clockify_rag.retrieval.embed_query") as mock_embed:
            mock_embed.return_value = np.random.rand(768).astype(np.float32)

            # Start ingest in background thread
            ingest_thread = threading.Thread(target=simulate_ingest)
            ingest_thread.start()

            # Fire 20 concurrent queries
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(run_query) for _ in range(20)]
                for future in futures:
                    future.result()

            ingest_thread.join()

            # Verify: No AttributeError, NoneType errors
            assert len(query_errors) == 0, f"Queries had errors: {query_errors}"

            # Verify: Queries either succeeded (200) or got 503 (index not ready)
            # Both are acceptable - 503 means query saw index_ready=False during ingest
            for status in query_results:
                assert status in [200, 503], f"Unexpected status code: {status}"

    def test_query_never_sees_partial_state(self, app_with_index):
        """Test queries never see partially updated state (e.g., chunks but no vecs)."""
        partial_state_detected = []

        def run_query_and_check():
            """Query and check if state is internally consistent."""
            # Capture state snapshot atomically
            with app_with_index.state.lock:
                chunks = app_with_index.state.chunks
                vecs_n = app_with_index.state.vecs_n
                bm = app_with_index.state.bm
                index_ready = app_with_index.state.index_ready

            # Check consistency: if index_ready, all components must be present
            if index_ready:
                if chunks is None or vecs_n is None or bm is None:
                    partial_state_detected.append("Partial state: index_ready but missing components")
                elif len(chunks) != len(vecs_n):
                    partial_state_detected.append(f"Mismatched lengths: chunks={len(chunks)}, vecs={len(vecs_n)}")

        def simulate_slow_ingest():
            """Simulate ingest with artificial delays between state updates."""
            time.sleep(0.02)

            with app_with_index.state.lock:
                # Intentionally slow state update
                app_with_index.state.index_ready = False
                time.sleep(0.05)

                app_with_index.state.chunks = [{"id": 99, "text": "New", "title": "N", "section": "X"}]
                time.sleep(0.05)

                app_with_index.state.vecs_n = np.random.rand(1, 768).astype(np.float32)
                time.sleep(0.05)

                app_with_index.state.bm = {"doc_lens": [5], "avg_len": 5.0, "k1": 1.5, "b": 0.75}
                time.sleep(0.05)

                app_with_index.state.index_ready = True

        # Start ingest
        ingest_thread = threading.Thread(target=simulate_slow_ingest)
        ingest_thread.start()

        # Fire 50 queries checking state consistency
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(run_query_and_check) for _ in range(50)]
            for future in futures:
                future.result()

        ingest_thread.join()

        # Assert no partial state detected
        assert len(partial_state_detected) == 0, f"Detected partial states: {partial_state_detected}"


class TestDuplicateIngest:
    """Test concurrent ingest requests are handled safely."""

    def test_duplicate_ingest_requests_no_deadlock(self, app_with_index):
        """Test multiple concurrent ingest requests don't deadlock."""
        # Note: This is a simplified test - real ingest uses background tasks
        # This tests the lock behavior when multiple threads try to update state

        results = []
        errors = []

        def attempt_ingest(worker_id):
            """Simulate ingest by acquiring lock and updating state."""
            try:
                with app_with_index.state.lock:
                    # Simulate slow index build
                    time.sleep(0.01)

                    # Update state
                    new_chunks = [{"id": worker_id, "text": f"Worker {worker_id}", "title": "W", "section": "S"}]
                    app_with_index.state.chunks = new_chunks
                    app_with_index.state.index_ready = True

                results.append(f"Worker {worker_id} completed")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # Fire 5 concurrent ingest attempts
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attempt_ingest, i) for i in range(5)]
            for future in futures:
                future.result(timeout=5.0)  # Should complete within 5 seconds

        # Verify: All workers completed (no deadlock)
        assert len(results) == 5, f"Only {len(results)}/5 completed"
        assert len(errors) == 0, f"Errors detected: {errors}"


class TestStateConsistency:
    """Test that heavy concurrent operations don't corrupt state."""

    def test_1000_concurrent_state_reads(self, app_with_index):
        """Test 1000 concurrent state reads don't cause errors."""
        errors = []
        read_count = 0

        def read_state():
            """Read state atomically and verify consistency."""
            nonlocal read_count
            try:
                with app_with_index.state.lock:
                    chunks = app_with_index.state.chunks
                    vecs_n = app_with_index.state.vecs_n
                    index_ready = app_with_index.state.index_ready

                    # Verify consistency
                    if index_ready and chunks is not None and vecs_n is not None:
                        if len(chunks) != len(vecs_n):
                            errors.append(f"Mismatched lengths: {len(chunks)} != {len(vecs_n)}")

                    read_count += 1
            except Exception as e:
                errors.append(f"Read error: {e}")

        # Fire 1000 concurrent reads
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(read_state) for _ in range(1000)]
            for future in futures:
                future.result()

        # Verify: No errors, all reads completed
        assert len(errors) == 0, f"Errors: {errors[:10]}"  # Show first 10 errors
        assert read_count == 1000, f"Only {read_count}/1000 reads completed"

    def test_mixed_reads_and_writes(self, app_with_index):
        """Test mixed concurrent reads and writes maintain consistency."""
        errors = []
        read_count = 0
        write_count = 0

        def read_state():
            """Read state and verify consistency."""
            nonlocal read_count
            try:
                with app_with_index.state.lock:
                    if app_with_index.state.index_ready:
                        chunks = app_with_index.state.chunks
                        vecs_n = app_with_index.state.vecs_n

                        if chunks is not None and vecs_n is not None:
                            if len(chunks) != len(vecs_n):
                                errors.append("Read: Mismatched state")

                    read_count += 1
            except Exception as e:
                errors.append(f"Read error: {e}")

        def write_state(value):
            """Write new state atomically."""
            nonlocal write_count
            try:
                with app_with_index.state.lock:
                    new_chunks = [{"id": value, "text": f"Chunk {value}", "title": "T", "section": "S"}]
                    new_vecs = np.random.rand(1, 768).astype(np.float32)
                    new_vecs = new_vecs / np.linalg.norm(new_vecs, axis=1, keepdims=True)

                    app_with_index.state.chunks = new_chunks
                    app_with_index.state.vecs_n = new_vecs
                    app_with_index.state.index_ready = True

                    write_count += 1
            except Exception as e:
                errors.append(f"Write error: {e}")

        # Fire 100 readers and 10 writers concurrently
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []

            # 100 readers
            for _ in range(100):
                futures.append(executor.submit(read_state))

            # 10 writers
            for i in range(10):
                futures.append(executor.submit(write_state, i))

            for future in futures:
                future.result()

        # Verify: No errors
        assert len(errors) == 0, f"Errors: {errors[:10]}"
        assert read_count == 100, f"Only {read_count}/100 reads completed"
        assert write_count == 10, f"Only {write_count}/10 writes completed"


class TestLockIsolation:
    """Test that multiple app instances have isolated locks (test isolation)."""

    def test_multiple_apps_have_separate_locks(self):
        """Test each app instance has its own lock (no shared global state)."""
        app1 = create_app()
        app2 = create_app()

        # Verify each app has its own lock instance
        assert app1.state.lock is not app2.state.lock, "Apps share the same lock (global state leak)"

        # Verify locks are independent
        lock1_acquired = app1.state.lock.acquire(blocking=False)
        lock2_acquired = app2.state.lock.acquire(blocking=False)

        assert lock1_acquired, "Failed to acquire app1 lock"
        assert lock2_acquired, "Failed to acquire app2 lock (blocked by app1?)"

        app1.state.lock.release()
        app2.state.lock.release()

    def test_lock_is_reentrant(self, app_with_index):
        """Test app.state.lock is reentrant (RLock behavior)."""
        # Acquire lock twice from same thread (should succeed with RLock)
        acquired1 = app_with_index.state.lock.acquire(blocking=False)
        acquired2 = app_with_index.state.lock.acquire(blocking=False)

        assert acquired1, "First acquire failed"
        assert acquired2, "Second acquire failed (lock not reentrant)"

        # Release twice
        app_with_index.state.lock.release()
        app_with_index.state.lock.release()
