"""
Tests for RAG edge cases and robustness.

Critical scenarios:
1. Empty retrieval - No documents above threshold
2. Context window overflow - Token budget exceeded
3. Degenerate embeddings - NaN/inf/zero vectors
4. Single-chunk corpus - Minimal data edge case
5. Unicode/emoji token counting - Special character handling
"""

import numpy as np
from unittest.mock import patch

from clockify_rag.retrieval import retrieve, pack_snippets, coverage_ok
from clockify_rag.answer import answer_once
from clockify_rag.config import REFUSAL_STR
from clockify_rag.exceptions import ValidationError


class TestEmptyRetrieval:
    """Test behavior when retrieval returns no results."""

    def test_answer_once_empty_retrieval_refuses(self, sample_chunks, sample_embeddings, sample_bm25):
        """Test answer_once refuses when no chunks above threshold."""
        question = "Completely irrelevant quantum physics question"

        # Mock retrieve to return empty results
        with patch("clockify_rag.answer.retrieve") as mock_retrieve:
            mock_retrieve.return_value = ([], {"dense": np.array([]), "bm25": np.array([])})

            result = answer_once(question, sample_chunks, sample_embeddings, sample_bm25, threshold=0.5)

            assert result["answer"] == REFUSAL_STR
            assert result["refused"] is True
            assert result["metadata"]["coverage_check"] == "failed"

    def test_coverage_ok_empty_selection(self):
        """Test coverage_ok returns False for empty selection."""
        selected = []
        scores = np.array([])

        result = coverage_ok(selected, scores, threshold=0.25)

        assert result is False

    def test_coverage_ok_all_below_threshold(self):
        """Test coverage_ok returns False when all scores below threshold."""
        selected = [0, 1, 2]
        scores = np.array([0.15, 0.10, 0.05])  # All below 0.25 threshold

        result = coverage_ok(selected, scores, threshold=0.25)

        assert result is False


class TestContextWindowOverflow:
    """Test token budget and context window limits."""

    def test_pack_snippets_respects_budget(self, sample_chunks):
        """Test pack_snippets respects num_ctx token budget."""
        selected = [0, 1, 2, 3, 4]  # 5 chunks
        pack_top = 5
        num_ctx = 100  # Very small budget

        context_block, packed_ids, used_tokens, article_blocks = pack_snippets(
            sample_chunks, selected, pack_top=pack_top, num_ctx=num_ctx
        )

        # Should respect budget and not overflow
        assert used_tokens <= num_ctx
        assert len(packed_ids) > 0  # Should pack at least 1 chunk
        assert len(packed_ids) <= pack_top

    def test_pack_snippets_tiny_budget(self, sample_chunks):
        """Test pack_snippets with extremely small budget."""
        selected = [0]
        pack_top = 1
        num_ctx = 10  # Impossibly small budget

        context_block, packed_ids, used_tokens, article_blocks = pack_snippets(
            sample_chunks, selected, pack_top=pack_top, num_ctx=num_ctx
        )

        # Should handle gracefully - may pack nothing or truncate aggressively
        assert isinstance(context_block, str)
        assert isinstance(packed_ids, list)
        assert used_tokens >= 0

    def test_pack_snippets_single_huge_chunk(self, sample_chunks):
        """Test pack_snippets when single chunk exceeds budget."""
        # Create a huge chunk
        huge_chunk = {
            "id": 999,
            "text": "word " * 1000,  # ~4000 tokens
            "title": "Huge",
            "section": "Test",
        }
        chunks = [huge_chunk]
        selected = [0]
        pack_top = 1
        num_ctx = 500  # Budget smaller than chunk

        context_block, packed_ids, used_tokens, article_blocks = pack_snippets(
            chunks, selected, pack_top=pack_top, num_ctx=num_ctx
        )

        # Should truncate or handle gracefully
        assert isinstance(context_block, str)
        assert used_tokens <= num_ctx + 100  # Allow small overflow for formatting

    def test_pack_snippets_unicode_emoji_tokens(self, sample_chunks):
        """Test pack_snippets handles Unicode and emoji correctly."""
        unicode_chunk = {
            "id": 100,
            "text": "Hello 世界 🌍 😀 测试",  # Mixed Unicode/emoji
            "title": "Unicode",
            "section": "Test",
        }
        chunks = [unicode_chunk]
        selected = [0]
        pack_top = 1
        num_ctx = 2800

        context_block, packed_ids, used_tokens, article_blocks = pack_snippets(
            chunks, selected, pack_top=pack_top, num_ctx=num_ctx
        )

        # Should handle without errors
        assert isinstance(context_block, str)
        assert len(packed_ids) == 1
        assert "世界" in context_block or "Hello" in context_block  # Some content preserved


class TestDegenerateEmbeddings:
    """Test behavior with invalid or degenerate embeddings."""

    def test_retrieve_with_nan_embeddings(self, sample_chunks, sample_bm25):
        """Test retrieve handles NaN embeddings gracefully."""
        # Create embeddings with NaN
        vecs_n = np.random.rand(5, 768).astype(np.float32)
        vecs_n[2] = np.nan  # Inject NaN in one vector

        question = "test question"

        # Mock embed_query to return valid query vector
        with patch("clockify_rag.retrieval.embed_query") as mock_embed:
            mock_embed.return_value = np.random.rand(768).astype(np.float32)

            # Should not crash, but may return unexpected results
            try:
                selected, scores = retrieve(question, sample_chunks, vecs_n, sample_bm25, top_k=5)
                # If successful, verify it returns something
                assert isinstance(selected, list)
                assert isinstance(scores, dict)
            except Exception as e:
                # If it raises, ensure it's a controlled exception
                assert "NaN" in str(e) or "invalid" in str(e).lower()

    def test_retrieve_with_zero_embeddings(self, sample_chunks, sample_bm25):
        """Test retrieve handles all-zero embeddings."""
        # Create embeddings with all zeros
        vecs_n = np.zeros((5, 768), dtype=np.float32)

        question = "test question"

        # Mock embed_query
        with patch("clockify_rag.retrieval.embed_query") as mock_embed:
            mock_embed.return_value = np.random.rand(768).astype(np.float32)

            # Should handle gracefully
            selected, scores = retrieve(question, sample_chunks, vecs_n, sample_bm25, top_k=5)

            # Zero vectors will have zero similarity, but shouldn't crash
            assert isinstance(selected, list)
            assert isinstance(scores, dict)

    def test_retrieve_with_inf_embeddings(self, sample_chunks, sample_bm25):
        """Test retrieve handles infinite embeddings."""
        # Create embeddings with inf
        vecs_n = np.random.rand(5, 768).astype(np.float32)
        vecs_n[1] = np.inf  # Inject inf

        question = "test question"

        # Mock embed_query
        with patch("clockify_rag.retrieval.embed_query") as mock_embed:
            mock_embed.return_value = np.random.rand(768).astype(np.float32)

            # Should not crash
            try:
                selected, scores = retrieve(question, sample_chunks, vecs_n, sample_bm25, top_k=5)
                assert isinstance(selected, list)
            except Exception as e:
                # If it raises, should be controlled
                assert "inf" in str(e).lower() or "invalid" in str(e).lower()


class TestSingleChunkCorpus:
    """Test edge case with minimal corpus (single chunk)."""

    def test_answer_once_single_chunk(self):
        """Test answer_once with corpus of only 1 chunk."""
        from clockify_rag.indexing import build_bm25

        chunks = [{"id": 1, "text": "Single chunk", "title": "Test", "section": "A"}]
        vecs_n = np.random.rand(1, 768).astype(np.float32)
        vecs_n = vecs_n / np.linalg.norm(vecs_n)
        bm = build_bm25(chunks)  # Use proper BM25 builder

        question = "test question"

        # Mock embed_query
        with patch("clockify_rag.retrieval.embed_query") as mock_embed:
            mock_embed.return_value = vecs_n[0]

            result = answer_once(question, chunks, vecs_n, bm, top_k=5, pack_top=3)

            # Should handle gracefully
            assert isinstance(result, dict)
            assert "answer" in result
            # With single chunk, may or may not have enough coverage
            assert isinstance(result["answer"], str)

    def test_retrieve_single_chunk(self):
        """Test retrieve with single chunk corpus."""
        from clockify_rag.indexing import build_bm25

        chunks = [{"id": 1, "text": "Single chunk", "title": "Test", "section": "A"}]
        vecs_n = np.random.rand(1, 768).astype(np.float32)
        vecs_n = vecs_n / np.linalg.norm(vecs_n)
        bm = build_bm25(chunks)  # Use proper BM25 builder

        question = "test question"

        # Mock embed_query
        with patch("clockify_rag.retrieval.embed_query") as mock_embed:
            mock_embed.return_value = vecs_n[0]

            selected, scores = retrieve(question, chunks, vecs_n, bm, top_k=5)

            # Should return the single chunk
            assert len(selected) <= 1
            if len(selected) > 0:
                assert selected[0] == 0  # Index of the only chunk


class TestQueryValidation:
    """Test query input validation edge cases."""

    def test_retrieve_empty_question(self, sample_chunks, sample_embeddings, sample_bm25):
        """Test retrieve with empty question."""
        question = ""

        # Mock embed_query
        with patch("clockify_rag.retrieval.embed_query") as mock_embed:
            mock_embed.return_value = sample_embeddings[0]

            # Should handle gracefully or raise ValidationError
            try:
                selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25)
                # If successful, should return something
                assert isinstance(selected, list)
            except ValidationError:
                # Acceptable behavior - rejecting empty query
                pass

    def test_retrieve_extremely_long_question(self, sample_chunks, sample_embeddings, sample_bm25):
        """Test retrieve with very long question."""
        question = "word " * 5000  # ~20000 chars

        # Mock embed_query
        with patch("clockify_rag.retrieval.embed_query") as mock_embed:
            mock_embed.return_value = sample_embeddings[0]

            # Should truncate or raise ValidationError
            try:
                selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25)
                assert isinstance(selected, list)
            except ValidationError as e:
                # Expected - query too long
                assert "too long" in str(e).lower() or "length" in str(e).lower()

    def test_retrieve_unicode_question(self, sample_chunks, sample_embeddings, sample_bm25):
        """Test retrieve with Unicode/emoji question."""
        question = "How do I track time? 世界 🌍 测试"

        # Mock embed_query
        with patch("clockify_rag.retrieval.embed_query") as mock_embed:
            mock_embed.return_value = sample_embeddings[0]

            # Should handle Unicode without errors
            selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25)

            assert isinstance(selected, list)
            assert isinstance(scores, dict)


class TestMMREdgeCases:
    """Test MMR diversification edge cases."""

    def test_mmr_with_identical_embeddings(self, sample_chunks):
        """Test MMR when all embeddings are identical."""
        from clockify_rag.answer import apply_mmr_diversification

        # All embeddings identical
        vecs_n = np.ones((5, 768), dtype=np.float32)
        vecs_n = vecs_n / np.linalg.norm(vecs_n, axis=1, keepdims=True)

        selected = [0, 1, 2, 3, 4]
        scores = {
            "dense": np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
            "bm25": np.array([0.85, 0.75, 0.70, 0.65, 0.60]),
            "hybrid": np.array([0.875, 0.775, 0.70, 0.625, 0.55]),
        }

        # Should handle gracefully
        result = apply_mmr_diversification(selected, scores, vecs_n, pack_top=3)

        assert isinstance(result, list)
        assert len(result) <= 3
        assert len(result) > 0  # Should return at least top dense

    def test_mmr_single_candidate(self):
        """Test MMR with only one candidate."""
        from clockify_rag.answer import apply_mmr_diversification

        vecs_n = np.random.rand(1, 768).astype(np.float32)
        vecs_n = vecs_n / np.linalg.norm(vecs_n)

        selected = [0]
        scores = {
            "dense": np.array([0.9]),
            "bm25": np.array([0.85]),
            "hybrid": np.array([0.875]),
        }

        result = apply_mmr_diversification(selected, scores, vecs_n, pack_top=5)

        assert result == [0]  # Should return the single candidate
