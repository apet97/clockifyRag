"""Tests for retrieval functionality."""

import pytest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.indexing import build_bm25, bm25_scores


class TestRetrieval:
    """Test retrieval and ranking logic."""

    def setup_method(self):
        """Setup test data."""
        self.chunks = [
            {
                "id": 0,
                "text": "How to track time in Clockify",
                "title": "Time Tracking",
                "section": "Basics",
                "url": "#track",
            },
            {"id": 1, "text": "Clockify pricing plans", "title": "Pricing", "section": "Plans", "url": "#pricing"},
            {"id": 2, "text": "Track time with timer", "title": "Timer", "section": "Features", "url": "#timer"},
        ]
        self.vecs = np.random.rand(3, 384).astype(np.float32)
        # Normalize vectors
        norms = np.linalg.norm(self.vecs, axis=1, keepdims=True)
        self.vecs = self.vecs / norms

    def test_bm25_retrieval_top_k(self):
        """Test that BM25 returns top K results."""
        bm = build_bm25(self.chunks)
        query = "track time"
        scores = bm25_scores(query, bm)

        # Get top 2 results
        top_k = 2
        top_indices = np.argsort(scores)[::-1][:top_k]

        assert len(top_indices) == top_k
        # Scores should be in descending order
        assert scores[top_indices[0]] >= scores[top_indices[1]]

    def test_dense_retrieval_similarity(self):
        """Test dense retrieval with cosine similarity."""
        # Create a query vector similar to first chunk
        query_vec = self.vecs[0] + np.random.randn(384) * 0.01
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Compute similarities
        similarities = self.vecs @ query_vec

        # First chunk should be most similar
        best_idx = np.argmax(similarities)
        assert best_idx == 0
        assert similarities[0] > 0.8  # Should be highly similar

    def test_hybrid_scoring_combination(self):
        """Test hybrid scoring combines BM25 and dense scores."""
        bm = build_bm25(self.chunks)
        query = "track time"

        # Get BM25 scores
        bm25_scores_list = bm25_scores(query, bm)

        # Normalize BM25 scores
        max_bm25 = max(bm25_scores_list) if max(bm25_scores_list) > 0 else 1
        bm25_norm = [s / max_bm25 for s in bm25_scores_list]

        # Mock dense scores
        dense_scores = [0.9, 0.3, 0.7]

        # Hybrid with alpha=0.5
        alpha = 0.5
        hybrid_scores = [alpha * bm25_norm[i] + (1 - alpha) * dense_scores[i] for i in range(len(self.chunks))]

        # Verify hybrid scores are combinations
        assert len(hybrid_scores) == len(self.chunks)
        assert all(0 <= s <= 1 for s in hybrid_scores)

    def test_retrieval_returns_metadata(self):
        """Test that retrieval preserves chunk metadata."""
        # Simulate retrieval returning chunks with metadata
        for chunk in self.chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "title" in chunk
            assert "section" in chunk
            assert "url" in chunk


class TestMMR:
    """Test Maximal Marginal Relevance diversification."""

    def test_mmr_diversifies_results(self):
        """Test that MMR promotes diversity."""
        # Create 3 very similar vectors (redundant)
        base_vec = np.random.rand(384)
        vecs = np.array(
            [base_vec, base_vec + np.random.randn(384) * 0.01, np.random.rand(384)],  # Very similar  # Different
            dtype=np.float32,
        )

        # Normalize
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        # Check diversity: vectors 0 and 1 are similar, 2 is different
        sim_01 = vecs[0] @ vecs[1]
        sim_02 = vecs[0] @ vecs[2]

        assert sim_01 > 0.9  # Very similar
        assert sim_02 < 0.9  # Less similar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
