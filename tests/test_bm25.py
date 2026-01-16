"""Tests for BM25 scoring functionality."""

import pytest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.indexing import build_bm25, bm25_scores


class TestBM25:
    """Test BM25 indexing and scoring."""

    def setup_method(self):
        """Setup test data."""
        self.chunks = [
            {"id": 0, "text": "How to track time in Clockify"},
            {"id": 1, "text": "Clockify pricing plans and features"},
            {"id": 2, "text": "Track time with timer or manual entry"},
        ]

    def test_build_bm25_index(self):
        """Test that BM25 index builds correctly."""
        bm = build_bm25(self.chunks)

        # Check index structure
        assert "idf" in bm
        assert "avgdl" in bm
        assert "doc_lens" in bm
        assert "doc_tfs" in bm

        # Check that we have the right number of documents
        assert len(bm["doc_lens"]) == len(self.chunks)
        assert len(bm["doc_tfs"]) == len(self.chunks)

    def test_bm25_scores_returns_correct_shape(self):
        """Test that BM25 scoring returns scores for all documents."""
        bm = build_bm25(self.chunks)
        query = "track time"
        scores = bm25_scores(query, bm)

        # Scores can be numpy array or list
        if isinstance(scores, np.ndarray):
            assert scores.shape[0] == len(self.chunks)
        else:
            assert len(scores) == len(self.chunks)

    def test_bm25_scores_relevant_doc_higher(self):
        """Test that relevant document scores higher than irrelevant."""
        bm = build_bm25(self.chunks)
        query = "track time"
        scores = bm25_scores(query, bm)

        # Document 0 and 2 contain "track" and "time"
        # Document 1 doesn't contain "track" or "time"
        # So doc 0 or 2 should score higher than doc 1
        assert max(scores[0], scores[2]) > scores[1]

    def test_bm25_scores_exact_match_high_score(self):
        """Test that exact query match scores high."""
        chunks = [
            {"id": 0, "text": "track time"},
            {"id": 1, "text": "something completely different"},
        ]
        bm = build_bm25(chunks)
        query = "track time"
        scores = bm25_scores(query, bm)

        # Exact match should score higher
        assert scores[0] > scores[1]
        assert scores[0] > 0

    def test_bm25_scores_empty_query(self):
        """Test BM25 with empty query."""
        bm = build_bm25(self.chunks)
        scores = bm25_scores("", bm)

        # Empty query should return zero scores
        assert all(s == 0 for s in scores)

    def test_bm25_scores_no_match(self):
        """Test BM25 when query terms don't appear in corpus."""
        bm = build_bm25(self.chunks)
        query = "xyzabc123"  # Made-up term
        scores = bm25_scores(query, bm)

        # No matches should result in zero scores
        assert all(s == 0 for s in scores)

    def test_bm25_parameters_affect_scores(self):
        """Test that k1 and b parameters affect scoring."""
        bm = build_bm25(self.chunks)
        query = "track time"

        # Score with different parameters
        scores_default = bm25_scores(query, bm, k1=1.0, b=0.65)
        scores_high_k1 = bm25_scores(query, bm, k1=2.0, b=0.65)

        # Different parameters should produce different scores
        # (unless all docs score 0)
        if max(scores_default) > 0:
            # Use numpy comparison for arrays
            if isinstance(scores_default, np.ndarray):
                assert not np.array_equal(scores_default, scores_high_k1)
            else:
                assert scores_default != scores_high_k1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
