"""
Integration tests for FAISS index functionality.

Priority #16: FAISS integration test (ROI 6/10)

Tests:
- FAISS index building with different corpus sizes
- Index loading and querying
- Thread-safe index access
- Fallback to linear search when index unavailable
- ARM64 macOS compatibility (IVF training)
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import threading
import time

from clockify_rag.indexing import build_faiss_index, load_faiss_index, save_faiss_index

# Check if FAISS is available (may not be in CI)
try:
    import faiss  # noqa: F401

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Skip all tests in this module if FAISS not available
pytestmark = pytest.mark.skipif(
    not FAISS_AVAILABLE, reason="FAISS not installed (expected in CI, install via conda for M1 compatibility)"
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def small_embeddings():
    """Generate small set of embeddings for testing."""
    np.random.seed(42)
    # 100 vectors, 384 dimensions (matches local embeddings)
    vecs = np.random.randn(100, 384).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)
    return vecs


@pytest.fixture
def medium_embeddings():
    """Generate medium set of embeddings for testing."""
    np.random.seed(42)
    # 1000 vectors, 768 dimensions (matches Ollama embeddings)
    vecs = np.random.randn(1000, 768).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)
    return vecs


@pytest.fixture
def large_embeddings():
    """Generate large set of embeddings for testing."""
    np.random.seed(42)
    # 5000 vectors, 768 dimensions
    vecs = np.random.randn(5000, 768).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)
    return vecs


class TestFAISSIndexBuilding:
    """Test FAISS index building with different configurations."""

    def test_build_small_corpus_linear_fallback(self, temp_dir, small_embeddings):
        """Test that small corpus falls back to linear search (no IVF)."""
        # Build index with small corpus
        index = build_faiss_index(small_embeddings, nlist=64)

        # Should return index (fallback to Flat)
        assert index is not None
        assert index.ntotal == len(small_embeddings)

        # Test search
        query = small_embeddings[0:1]
        distances, indices = index.search(query, k=10)
        assert distances.shape == (1, 10)
        assert indices.shape == (1, 10)
        # First result should be the query itself (index 0)
        assert indices[0, 0] == 0
        assert distances[0, 0] >= 0.99  # Should be ~1.0 (normalized vectors)

    def test_build_medium_corpus_with_ivf(self, temp_dir, medium_embeddings):
        """Test IVF index building with medium corpus."""
        # Build index with medium corpus
        index = build_faiss_index(medium_embeddings, nlist=64)

        assert index is not None
        assert index.ntotal == len(medium_embeddings)

        # Test search with nprobe
        index.nprobe = 16
        query = medium_embeddings[0:1]
        distances, indices = index.search(query, k=20)

        assert distances.shape == (1, 20)
        assert indices.shape == (1, 20)
        # First result should be the query itself
        assert indices[0, 0] == 0

    def test_build_large_corpus_with_ivf(self, temp_dir, large_embeddings):
        """Test IVF index building with large corpus."""
        # Build index with large corpus
        index = build_faiss_index(large_embeddings, nlist=128)

        assert index is not None
        assert index.ntotal == len(large_embeddings)

        # Test search accuracy
        index.nprobe = 32
        query = large_embeddings[100:101]
        distances, indices = index.search(query, k=50)

        # Should find the query vector itself in top results
        assert 100 in indices[0]

    def test_index_persistence(self, temp_dir, medium_embeddings):
        """Test that index can be saved and loaded."""
        index_path = os.path.join(temp_dir, "faiss_persist.index")

        # Build and save index
        index1 = build_faiss_index(medium_embeddings, nlist=64)
        save_faiss_index(index1, index_path)
        assert index1 is not None
        assert os.path.exists(index_path)

        # Load index
        index2 = load_faiss_index(index_path)
        assert index2 is not None
        assert index2.ntotal == index1.ntotal

        # Test that both give same results
        query = medium_embeddings[0:1]
        index1.nprobe = 16
        index2.nprobe = 16

        D1, I1 = index1.search(query, k=10)
        D2, I2 = index2.search(query, k=10)

        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_allclose(D1, D2, rtol=1e-5)

    def test_load_missing_index(self, temp_dir):
        """Test graceful handling of missing index file."""
        missing_path = os.path.join(temp_dir, "nonexistent.index")
        index = load_faiss_index(missing_path)
        assert index is None

    def test_deterministic_training(self, temp_dir, large_embeddings):
        """Test that IVF training is deterministic with same seed."""
        index_path1 = os.path.join(temp_dir, "faiss_det1.index")
        index_path2 = os.path.join(temp_dir, "faiss_det2.index")

        # Build two indexes with same seed
        index1 = build_faiss_index(large_embeddings, nlist=64)
        save_faiss_index(index1, index_path1)
        index2 = build_faiss_index(large_embeddings, nlist=64)
        save_faiss_index(index2, index_path2)

        # Test that both give similar results (may differ slightly due to IVF)
        query = large_embeddings[0:1]
        index1.nprobe = 16
        index2.nprobe = 16

        D1, I1 = index1.search(query, k=20)
        D2, I2 = index2.search(query, k=20)

        # Top-1 should be identical
        assert I1[0, 0] == I2[0, 0]

        # Top-10 should have high overlap (at least 80%)
        overlap = len(set(I1[0, :10]) & set(I2[0, :10]))
        assert overlap >= 8, f"Expected at least 8/10 overlap, got {overlap}"


class TestFAISSThreadSafety:
    """Test thread-safe FAISS index access."""

    def test_concurrent_search(self, temp_dir, medium_embeddings):
        """Test that multiple threads can search index concurrently."""
        # Build index
        index = build_faiss_index(medium_embeddings, nlist=64)
        assert index is not None
        index.nprobe = 16

        results = []
        errors = []

        def search_worker(thread_id):
            try:
                query = medium_embeddings[thread_id : thread_id + 1]
                distances, indices = index.search(query, k=10)
                results.append((thread_id, distances, indices))
            except Exception as e:
                errors.append((thread_id, e))

        # Launch 10 concurrent searches
        threads = []
        for i in range(10):
            t = threading.Thread(target=search_worker, args=(i,))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

        # Verify each thread found its own query vector
        for thread_id, distances, indices in results:
            assert indices.shape == (1, 10)
            # Query vector should be in top results
            assert thread_id in indices[0]

    def test_concurrent_load(self, temp_dir, medium_embeddings):
        """Test that multiple threads can safely load index."""
        index_path = os.path.join(temp_dir, "faiss_load.index")

        # Build index
        index = build_faiss_index(medium_embeddings, nlist=64)
        save_faiss_index(index, index_path)

        loaded_indexes = []
        errors = []

        def load_worker(thread_id):
            try:
                index = load_faiss_index(index_path)
                loaded_indexes.append((thread_id, index))
            except Exception as e:
                errors.append((thread_id, e))

        # Launch 5 concurrent loads
        threads = []
        for i in range(5):
            t = threading.Thread(target=load_worker, args=(i,))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(loaded_indexes) == 5

        # All should have loaded successfully
        for thread_id, index in loaded_indexes:
            assert index is not None
            assert index.ntotal == len(medium_embeddings)


class TestFAISSSearchAccuracy:
    """Test FAISS search quality and accuracy."""

    def test_approximate_vs_exact_search(self, temp_dir, medium_embeddings):
        """Test that FAISS approximate search is close to exact search."""
        # Build index
        index = build_faiss_index(medium_embeddings, nlist=64)
        index.nprobe = 32  # High nprobe for better accuracy

        # Pick a random query
        query_idx = 42
        query = medium_embeddings[query_idx : query_idx + 1]

        # FAISS approximate search
        D_approx, I_approx = index.search(query, k=20)

        # Exact search (brute force)
        similarities = medium_embeddings.dot(query.T).squeeze()
        I_exact = np.argsort(similarities)[::-1][:20]

        # Compare results
        overlap = len(set(I_approx[0]) & set(I_exact))
        recall_at_20 = overlap / 20.0

        # Should have high recall (>= 90%)
        assert recall_at_20 >= 0.9, f"Recall@20: {recall_at_20:.2%} (expected >= 90%)"

        # Top-1 should match
        assert I_approx[0, 0] == I_exact[0]

    def test_nprobe_effect_on_accuracy(self, temp_dir, large_embeddings):
        """Test that higher nprobe improves search accuracy."""
        # Build index
        index = build_faiss_index(large_embeddings, nlist=128)

        query = large_embeddings[100:101]

        # Compute exact top-20
        similarities = large_embeddings.dot(query.T).squeeze()
        I_exact = set(np.argsort(similarities)[::-1][:20])

        # Test with different nprobe values
        recalls = {}
        for nprobe in [1, 4, 16, 64]:
            index.nprobe = nprobe
            distances, indices = index.search(query, k=20)
            overlap = len(set(indices[0]) & I_exact)
            recalls[nprobe] = overlap / 20.0

        # Higher nprobe should give better recall
        assert recalls[16] >= recalls[4]
        assert recalls[64] >= recalls[16]

        # nprobe=64 should achieve high recall
        assert recalls[64] >= 0.85


class TestFAISSARMMacCompatibility:
    """Test ARM64 macOS compatibility (reduced nlist to avoid segfaults)."""

    def test_arm64_safe_nlist(self, temp_dir, medium_embeddings):
        """Test that nlist=64 works on ARM64 (vs. nlist=256 which can segfault)."""
        # This should not crash on ARM64 macOS
        index = build_faiss_index(medium_embeddings, nlist=64)

        assert index is not None
        assert index.ntotal == len(medium_embeddings)

        # Test search
        query = medium_embeddings[0:1]
        index.nprobe = 16
        distances, indices = index.search(query, k=10)

        assert distances.shape == (1, 10)
        assert indices.shape == (1, 10)

    def test_large_corpus_training(self, temp_dir):
        """Test that large corpus IVF training completes without crash."""
        # Generate large corpus
        np.random.seed(42)
        large_vecs = np.random.randn(10000, 768).astype(np.float32)
        norms = np.linalg.norm(large_vecs, axis=1, keepdims=True)
        large_vecs = large_vecs / np.maximum(norms, 1e-9)

        # This should complete without segfault
        start = time.time()
        index = build_faiss_index(large_vecs, nlist=128)
        elapsed = time.time() - start

        assert index is not None
        assert index.ntotal == len(large_vecs)

        # Should complete in reasonable time (<30 seconds)
        assert elapsed < 30, f"Training took {elapsed:.1f}s (expected <30s)"


class TestFAISSFallback:
    """Test fallback behavior when FAISS is unavailable."""

    def test_empty_vectors(self, temp_dir):
        """Test handling of empty vector array."""
        empty_vecs = np.array([], dtype=np.float32).reshape(0, 768)

        index = build_faiss_index(empty_vecs, nlist=64)

        # Should return None or empty index
        if index is not None:
            assert index.ntotal == 0

    def test_invalid_dimensions(self, temp_dir):
        """Test handling of mismatched dimensions."""
        # Build with 384-dim
        vecs_384 = np.random.randn(100, 384).astype(np.float32)
        index = build_faiss_index(vecs_384, nlist=32)

        # Try to search with 768-dim (should raise error)
        query_768 = np.random.randn(1, 768).astype(np.float32)

        with pytest.raises(Exception):
            index.search(query_768, k=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
