"""Load and stress tests for Clockify RAG.

These tests verify system behavior under sustained load and concurrent access.
They help identify:
- Memory leaks
- Performance degradation over time
- Thread safety issues
- Resource exhaustion

Run with: pytest tests/test_load.py -v --timeout=120
"""

import gc
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import numpy as np

# Skip all tests if running in CI without explicit flag
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LOAD_TESTS", "0") != "1", reason="Load tests skipped by default. Set RUN_LOAD_TESTS=1 to run."
)


class TestRetrievalLoad:
    """Load tests for retrieval pipeline."""

    @pytest.fixture
    def mock_index(self):
        """Create a mock index for testing."""
        from clockify_rag.indexing import build_bm25

        # Create sample chunks
        chunks = []
        for i in range(100):
            chunks.append(
                {
                    "id": f"chunk_{i}",
                    "text": f"This is sample content for chunk {i}. It contains information about Clockify features and time tracking. Users can track their time, generate reports, and manage projects.",
                    "title": f"Article {i // 10}",
                    "section": f"Section {i % 5}",
                    "url": f"https://example.com/article-{i // 10}",
                }
            )

        # Create BM25 index
        bm = build_bm25(chunks)

        # Create mock embeddings (384-dim for local backend)
        vecs = np.random.randn(len(chunks), 384).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_n = vecs / norms

        return {
            "chunks": chunks,
            "bm": bm,
            "vecs_n": vecs_n,
        }

    def test_sustained_retrieval_load(self, mock_index):
        """Test retrieval under sustained load."""
        from clockify_rag.retrieval import retrieve

        chunks = mock_index["chunks"]
        bm = mock_index["bm"]
        vecs_n = mock_index["vecs_n"]

        queries = [
            "How do I track time?",
            "Generate reports",
            "Manage projects",
            "Team permissions",
            "Export data",
        ]

        # Run 50 queries
        times = []
        for i in range(50):
            query = queries[i % len(queries)]
            start = time.perf_counter()
            selected, scores = retrieve(query, chunks, vecs_n, bm, top_k=10)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            # Basic validation
            assert len(selected) > 0
            assert "hybrid" in scores

        # Performance assertions
        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Average should be under 500ms (without network)
        assert avg_time < 0.5, f"Average retrieval time {avg_time:.3f}s exceeds 500ms"

        # No single query should take more than 2s
        assert max_time < 2.0, f"Max retrieval time {max_time:.3f}s exceeds 2s"

        # Check for degradation (last 10 should not be 2x slower than first 10)
        first_10_avg = sum(times[:10]) / 10
        last_10_avg = sum(times[-10:]) / 10
        assert last_10_avg < first_10_avg * 2, "Performance degraded over time"

    def test_concurrent_retrieval(self, mock_index):
        """Test concurrent retrieval requests."""
        from clockify_rag.retrieval import retrieve

        chunks = mock_index["chunks"]
        bm = mock_index["bm"]
        vecs_n = mock_index["vecs_n"]

        queries = [
            "time tracking",
            "reports",
            "projects",
            "team",
            "export",
        ]

        errors = []
        results = []

        def run_query(query_idx):
            try:
                query = queries[query_idx % len(queries)]
                selected, scores = retrieve(query, chunks, vecs_n, bm, top_k=10)
                return {
                    "query": query,
                    "count": len(selected),
                    "success": True,
                }
            except Exception as e:
                return {
                    "query": query,
                    "error": str(e),
                    "success": False,
                }

        # Run 20 concurrent queries
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_query, i) for i in range(20)]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if not result["success"]:
                    errors.append(result)

        # All queries should succeed
        assert len(errors) == 0, f"Concurrent queries failed: {errors}"
        assert len(results) == 20


class TestBM25Load:
    """Load tests for BM25 scoring."""

    def test_bm25_memory_stability(self):
        """Test BM25 doesn't leak memory under load."""
        from clockify_rag.indexing import build_bm25, bm25_scores

        # Create chunks
        chunks = [{"id": f"c{i}", "text": f"Sample text {i} about time tracking and reports"} for i in range(100)]
        bm = build_bm25(chunks)

        # Force GC and get baseline memory
        gc.collect()

        # Run many queries
        for i in range(100):
            scores = bm25_scores(f"query term {i % 10}", bm, top_k=20)
            assert len(scores) == len(chunks)

        # Force GC
        gc.collect()

        # Memory check would go here if we had memory profiling
        # For now, just verify no crashes

    def test_bm25_large_query(self):
        """Test BM25 handles large queries efficiently."""
        from clockify_rag.indexing import build_bm25, bm25_scores

        chunks = [{"id": f"c{i}", "text": f"Sample content {i}"} for i in range(100)]
        bm = build_bm25(chunks)

        # Large query (1000 words)
        large_query = " ".join([f"word{i}" for i in range(1000)])

        start = time.perf_counter()
        scores = bm25_scores(large_query, bm, top_k=20)
        elapsed = time.perf_counter() - start

        assert len(scores) == len(chunks)
        assert elapsed < 1.0, f"Large query took {elapsed:.3f}s (>1s)"


class TestNormalizationLoad:
    """Load tests for score normalization."""

    def test_zscore_large_arrays(self):
        """Test z-score normalization with large arrays."""
        from clockify_rag.retrieval import normalize_scores_zscore

        # Large array
        scores = np.random.randn(10000).astype(np.float32)

        start = time.perf_counter()
        result = normalize_scores_zscore(scores)
        elapsed = time.perf_counter() - start

        assert result.shape == scores.shape
        assert elapsed < 0.1, f"Normalization took {elapsed:.3f}s (>100ms)"

        # Verify normalization properties
        assert abs(result.mean()) < 0.01
        assert abs(result.std() - 1.0) < 0.01

    def test_zscore_repeated_calls(self):
        """Test repeated normalization doesn't degrade."""
        from clockify_rag.retrieval import normalize_scores_zscore

        scores = np.random.randn(1000).astype(np.float32)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = normalize_scores_zscore(scores)
            times.append(time.perf_counter() - start)

        # No degradation
        first_10_avg = sum(times[:10]) / 10
        last_10_avg = sum(times[-10:]) / 10
        assert last_10_avg < first_10_avg * 1.5, "Normalization degraded"


class TestChunkingLoad:
    """Load tests for document chunking."""

    def test_large_document_chunking(self):
        """Test chunking a large document."""
        from clockify_rag.chunking import chunk_article

        # Create large article content
        sections = []
        for i in range(50):
            section = f"## Section {i}\n\n"
            for j in range(20):
                section += f"This is paragraph {j} of section {i}. " * 10 + "\n\n"
            sections.append(section)

        large_content = "\n".join(sections)

        start = time.perf_counter()
        chunks = chunk_article(
            article_content=large_content,
            article_title="Test Article",
            article_url="https://example.com/test",
            chunk_chars=512,
            overlap=50,
        )
        elapsed = time.perf_counter() - start

        assert len(chunks) > 0
        assert elapsed < 5.0, f"Chunking large doc took {elapsed:.3f}s (>5s)"

    def test_many_small_articles(self):
        """Test chunking many small articles."""
        from clockify_rag.chunking import chunk_article

        # Chunk 100 small articles
        times = []
        total_chunks = 0

        for i in range(100):
            content = f"# Article {i}\n\nThis is the content of article {i}. " * 20

            start = time.perf_counter()
            chunks = chunk_article(
                article_content=content,
                article_title=f"Article {i}",
                article_url=f"https://example.com/{i}",
            )
            times.append(time.perf_counter() - start)
            total_chunks += len(chunks)

        avg_time = sum(times) / len(times)
        assert avg_time < 0.1, f"Average chunking time {avg_time:.3f}s (>100ms)"
        assert total_chunks > 100  # Should create multiple chunks per article


class TestQueryExpansionLoad:
    """Load tests for query expansion."""

    def test_expansion_many_queries(self):
        """Test query expansion under load."""
        from clockify_rag.retrieval import expand_query

        queries = [
            "time tracking",
            "generate report",
            "team members",
            "project settings",
            "export data",
        ]

        times = []
        for i in range(100):
            query = queries[i % len(queries)]
            start = time.perf_counter()
            expanded = expand_query(query)
            times.append(time.perf_counter() - start)

            assert len(expanded) >= len(query)

        avg_time = sum(times) / len(times)
        assert avg_time < 0.01, f"Average expansion time {avg_time:.6f}s (>10ms)"


class TestConcurrencyStress:
    """Stress tests for concurrent operations."""

    def test_mixed_concurrent_operations(self):
        """Test mixed operations under concurrent load."""
        from clockify_rag.retrieval import (
            normalize_scores_zscore,
            expand_query,
        )
        from clockify_rag.indexing import build_bm25, bm25_scores

        # Create shared resources
        chunks = [{"id": f"c{i}", "text": f"Sample {i}"} for i in range(50)]
        bm = build_bm25(chunks)

        errors = []
        operation_counts = {"normalize": 0, "expand": 0, "bm25": 0}

        def random_operation(op_type):
            try:
                if op_type == "normalize":
                    scores = np.random.randn(100).astype(np.float32)
                    normalize_scores_zscore(scores)
                    operation_counts["normalize"] += 1
                elif op_type == "expand":
                    expand_query("test query")
                    operation_counts["expand"] += 1
                elif op_type == "bm25":
                    bm25_scores("test query", bm)
                    operation_counts["bm25"] += 1
                return True
            except Exception as e:
                errors.append((op_type, str(e)))
                return False

        operations = ["normalize", "expand", "bm25"] * 30  # 90 operations

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(random_operation, op) for op in operations]
            results = [f.result() for f in futures]

        assert all(results), f"Some operations failed: {errors}"
        assert len(errors) == 0


class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    def test_no_thread_leaks(self):
        """Test that operations don't leak threads."""
        from clockify_rag.retrieval import normalize_scores_zscore

        initial_threads = threading.active_count()

        # Run many operations
        for _ in range(50):
            scores = np.random.randn(100).astype(np.float32)
            normalize_scores_zscore(scores)

        # Allow time for cleanup
        time.sleep(0.1)

        final_threads = threading.active_count()

        # Should not have significantly more threads
        thread_diff = final_threads - initial_threads
        assert thread_diff < 5, f"Thread leak: {thread_diff} new threads"
