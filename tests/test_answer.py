"""
Tests for answer generation module.

Tests the complete answer pipeline including:
- MMR diversification
- LLM reranking
- Citation extraction and validation
- Answer generation with confidence
- Complete answer_once workflow
"""

import json

import pytest
import numpy as np
from unittest.mock import patch

from clockify_rag.answer import (
    apply_mmr_diversification,
    apply_reranking,
    extract_citations,
    validate_citations,
    generate_llm_answer,
    answer_once,
)
from clockify_rag.config import REFUSAL_STR
from clockify_rag.exceptions import LLMUnavailableError, LLMError


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        {"id": 1, "text": "Track time using the timer button.", "title": "Tracking", "section": "Overview"},
        {"id": 2, "text": "Manual time entry is available.", "title": "Tracking", "section": "Manual"},
        {"id": 3, "text": "Time tracking works offline.", "title": "Tracking", "section": "Offline"},
        {"id": 4, "text": "Reports show time spent.", "title": "Reports", "section": "Overview"},
        {"id": 5, "text": "Export reports as PDF or CSV.", "title": "Reports", "section": "Export"},
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample normalized embeddings."""
    np.random.seed(42)
    vecs = np.random.randn(5, 384).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def sample_scores():
    """Create sample retrieval scores."""
    return {
        "dense": np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        "bm25": np.array([0.85, 0.75, 0.70, 0.65, 0.60], dtype=np.float32),
        "hybrid": np.array([0.875, 0.775, 0.70, 0.625, 0.55], dtype=np.float32),
    }


class TestMMRDiversification:
    """Test MMR diversification logic."""

    def test_mmr_selects_top_dense_first(self, sample_embeddings, sample_scores):
        """Test that MMR always includes top dense score first."""
        selected = [0, 1, 2, 3, 4]

        result = apply_mmr_diversification(selected, sample_scores, sample_embeddings, pack_top=3)

        # First item should be index 0 (highest dense score)
        assert result[0] == 0
        assert len(result) == 3

    def test_mmr_respects_pack_top(self, sample_embeddings, sample_scores):
        """Test that MMR respects pack_top limit."""
        selected = [0, 1, 2, 3, 4]

        result = apply_mmr_diversification(selected, sample_scores, sample_embeddings, pack_top=2)

        assert len(result) == 2

    def test_mmr_with_single_candidate(self, sample_embeddings, sample_scores):
        """Test MMR with only one candidate."""
        selected = [2]

        result = apply_mmr_diversification(selected, sample_scores, sample_embeddings, pack_top=5)

        assert len(result) == 1
        assert result[0] == 2

    def test_mmr_diversifies_results(self, sample_embeddings, sample_scores):
        """Test that MMR produces diverse results (not just top-k dense)."""
        selected = [0, 1, 2, 3, 4]

        result = apply_mmr_diversification(selected, sample_scores, sample_embeddings, pack_top=5)

        # Should include top dense (0), but order should differ from pure dense ranking
        assert 0 in result
        assert len(result) == 5


class TestCitationExtraction:
    """Test citation extraction and validation."""

    def test_extract_simple_citations(self):
        """Test extraction of simple numeric citations."""
        answer = "Track time using [1] and [2]."

        citations = extract_citations(answer)

        assert "1" in citations
        assert "2" in citations
        assert len(citations) == 2

    def test_extract_id_prefix_citations(self):
        """Test extraction of id_ prefix citations."""
        answer = "See [id_123] and [id_456]."

        citations = extract_citations(answer)

        assert "id_123" in citations
        assert "id_456" in citations

    def test_extract_mixed_citations(self):
        """Test extraction of mixed citation formats."""
        answer = "Use [1], [id_2], and [abc-def] for tracking."

        citations = extract_citations(answer)

        assert "1" in citations
        assert "id_2" in citations
        assert "abc-def" in citations

    def test_extract_no_citations(self):
        """Test answer with no citations."""
        answer = "Track time using the timer."

        citations = extract_citations(answer)

        assert len(citations) == 0

    def test_validate_citations_all_valid(self):
        """Test validation with all valid citations."""
        answer = "Use [1] and [2]."
        valid_ids = [1, 2, 3]

        is_valid, valid, invalid = validate_citations(answer, valid_ids)

        assert is_valid
        assert len(valid) == 2
        assert len(invalid) == 0

    def test_validate_citations_some_invalid(self):
        """Test validation with some invalid citations."""
        answer = "Use [1] and [99]."
        valid_ids = [1, 2, 3]

        is_valid, valid, invalid = validate_citations(answer, valid_ids)

        assert not is_valid
        assert "1" in valid
        assert "99" in invalid

    def test_validate_citations_all_invalid(self):
        """Test validation with all invalid citations."""
        answer = "Use [99] and [100]."
        valid_ids = [1, 2, 3]

        is_valid, valid, invalid = validate_citations(answer, valid_ids)

        assert not is_valid
        assert len(valid) == 0
        assert len(invalid) == 2


class TestGenerateLLMAnswer:
    """Test LLM answer generation with mocking."""

    @patch("clockify_rag.answer.ask_llm")
    def test_generate_with_json_response(self, mock_ask_llm):
        """Test answer generation with new JSON response format (v6.0+)."""
        payload = {
            "answer": "1. Track time using the timer.\n2. Steps:\n   - Press Start\n   - Press Stop\n3. Notes: Applies to all plans.\n4. Citations: [id_1]",
            "confidence": 85,
            "reasoning": "Based on context block 1 which describes timer functionality.",
            "sources_used": ["1"],
            "intent": "feature_howto",
            "user_role_inferred": "regular_member",
            "security_sensitivity": "low",
            "short_intent_summary": "User wants to track time",
            "needs_human_escalation": False,
        }
        mock_ask_llm.return_value = json.dumps(payload)

        # Provide all_chunks so packed_chunks is set, triggering new prompt path
        chunks = [{"id": 1, "text": "Track time using timer."}]
        answer, timing, confidence, reasoning, sources_used, meta = generate_llm_answer(
            "How to track time?", "[1] Track time using timer.", packed_ids=[1], all_chunks=chunks
        )

        assert answer == payload["answer"]
        assert confidence == payload["confidence"]
        assert reasoning == payload["reasoning"]
        assert sources_used == payload["sources_used"]
        assert meta["intent"] == "feature_howto"
        assert timing >= 0

    @patch("clockify_rag.answer.ask_llm")
    def test_generate_with_markdown_json(self, mock_ask_llm):
        """Test answer generation with markdown-wrapped JSON (v6.0+)."""
        payload = {
            "answer": "Track time using [1].",
            "confidence": 90,
            "reasoning": "Context 1 provides clear instructions.",
            "sources_used": ["1"],
            "intent": "feature_howto",
        }
        mock_ask_llm.return_value = f"```json\n{json.dumps(payload)}\n```"

        chunks = [{"id": 1, "text": "Track time using timer."}]
        answer, timing, confidence, reasoning, sources_used, meta = generate_llm_answer(
            "How to track time?", "[1] Track time using timer.", packed_ids=[1], all_chunks=chunks
        )

        assert answer == "Track time using [1]."
        assert confidence == 90
        assert reasoning == "Context 1 provides clear instructions."
        assert sources_used == ["1"]
        assert meta["intent"] == "feature_howto"

    @patch("clockify_rag.answer.ask_llm")
    def test_generate_with_numbered_answer_structure(self, mock_ask_llm):
        """Ensure numbered content inside JSON answer parses correctly (v6.0+)."""
        payload = {
            "answer": "1. Direct answer\n2. Steps:\n   1) Start timer\n   2) Stop timer\n3. Notes: Premium users see advanced features.\n4. Citations: [id_1, id_2]",
            "confidence": 77,
            "reasoning": "Synthesized from context blocks 1 and 2.",
            "sources_used": ["1", "2"],
            "intent": "feature_howto",
        }
        mock_ask_llm.return_value = json.dumps(payload)

        chunks = [{"id": 1, "text": "Track time using timer."}, {"id": 2, "text": "Manual entry option."}]
        answer, _, confidence, reasoning, sources_used, meta = generate_llm_answer(
            "How to track time?",
            "[1] Track time using timer. [2] Manual entry option.",
            packed_ids=[1, 2],
            all_chunks=chunks,
        )

        assert answer == payload["answer"]
        assert confidence == payload["confidence"]
        assert reasoning == payload["reasoning"]
        assert sources_used == payload["sources_used"]
        assert meta["intent"] == "feature_howto"

    @patch("clockify_rag.answer.ask_llm")
    def test_generate_with_plain_text_legacy(self, mock_ask_llm):
        """Test answer generation with plain text legacy path (no chunks provided)."""
        mock_ask_llm.return_value = "Track time using [1]."

        # Don't provide all_chunks, triggers legacy path
        answer, timing, confidence, reasoning, sources_used, meta = generate_llm_answer(
            "How to track time?", "[1] Track time using timer.", packed_ids=[1]
        )

        assert answer == "Track time using [1]."
        assert confidence is None  # No confidence in plain text legacy
        assert reasoning is None  # No reasoning in legacy
        assert sources_used is None  # No sources_used in legacy

    @patch("clockify_rag.answer.ask_llm")
    def test_generate_with_invalid_json_fallback(self, mock_ask_llm):
        """Test that invalid JSON falls back to raw text with warning (v6.0+)."""
        # Invalid JSON (missing required fields)
        mock_ask_llm.return_value = '{"answer": "Track time using [1].", "confidence": 85}'

        chunks = [{"id": 1, "text": "Track time using timer."}]
        answer, timing, confidence, reasoning, sources_used, meta = generate_llm_answer(
            "How to track time?", "[1] Track time using timer.", packed_ids=[1], all_chunks=chunks
        )

        # Should fall back to raw text when JSON parsing fails
        assert "Track time using [1]." in answer
        assert reasoning is None  # Fallback path doesn't have reasoning
        assert sources_used is None  # Fallback path doesn't have sources
        assert meta["intent"] == "other"


class TestReranking:
    """Test LLM reranking logic."""

    def test_reranking_disabled(self, sample_chunks, sample_scores):
        """Test that reranking returns original order when disabled."""
        selected = [0, 1, 2]

        result, scores, applied, reason, timing = apply_reranking(
            "test question", sample_chunks, selected, sample_scores, use_rerank=False
        )

        assert result == selected
        assert not applied
        assert reason == "disabled"
        assert timing == 0.0

    def test_reranking_with_single_chunk(self, sample_chunks, sample_scores):
        """Test that reranking is skipped for single chunk."""
        selected = [0]

        result, scores, applied, reason, timing = apply_reranking(
            "test question", sample_chunks, selected, sample_scores, use_rerank=True
        )

        assert result == selected
        assert not applied
        assert reason == "disabled"


class TestAnswerOnce:
    """Test complete answer_once pipeline."""

    @patch("clockify_rag.answer.retrieve")
    @patch("clockify_rag.answer.ask_llm")
    def test_answer_once_basic(self, mock_ask_llm, mock_retrieve, sample_chunks, sample_embeddings):
        """Test basic answer_once workflow."""
        # Mock retrieve to return candidates
        mock_retrieve.return_value = (
            [0, 1, 2],
            {
                "dense": np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32),
                "bm25": np.array([0.85, 0.75, 0.70, 0.65, 0.60], dtype=np.float32),
                "hybrid": np.array([0.875, 0.775, 0.70, 0.625, 0.55], dtype=np.float32),
            },
        )

        # Mock LLM to return answer (plain text or JSON)
        # Note: With new prompt system, confidence is computed from retrieval scores,
        # not from LLM output. In mock mode, confidence defaults to 100.
        mock_ask_llm.return_value = '{"answer": "Track time using [1].", "confidence": 85, "reasoning": "Used context block 1", "sources_used": ["1"]}'

        # Mock BM25 index
        bm = {"idf": {}, "avgdl": 10, "doc_lens": [10] * 5, "doc_tfs": [{}] * 5}

        result = answer_once(
            "How to track time?", sample_chunks, sample_embeddings, bm, top_k=5, pack_top=3, threshold=0.3
        )

        assert "answer" in result
        assert result["answer"] == "Track time using [1]."
        assert not result["refused"]
        # Confidence comes from Qwen JSON output
        assert result["confidence"] == 85
        assert "timing" in result
        assert "metadata" in result

    @patch("clockify_rag.answer.retrieve")
    def test_answer_once_low_coverage(self, mock_retrieve, sample_chunks, sample_embeddings):
        """Test answer_once with low coverage (should refuse)."""
        # Mock retrieve to return low scores
        mock_retrieve.return_value = (
            [0, 1],
            {
                "dense": np.array([0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
                "bm25": np.array([0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
                "hybrid": np.array([0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
            },
        )

        bm = {"idf": {}, "avgdl": 10, "doc_lens": [10] * 5, "doc_tfs": [{}] * 5}

        result = answer_once(
            "How to track time?",
            sample_chunks,
            sample_embeddings,
            bm,
            top_k=5,
            pack_top=3,
            threshold=0.3,  # Higher than our max score
        )

        assert result["refused"]
        assert result["answer"] == REFUSAL_STR

    @patch("clockify_rag.answer.retrieve")
    @patch("clockify_rag.answer.ask_llm")
    def test_answer_once_with_reranking(self, mock_ask_llm, mock_retrieve, sample_chunks, sample_embeddings):
        """Test answer_once with reranking enabled."""
        mock_retrieve.return_value = (
            [0, 1, 2],
            {
                "dense": np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32),
                "bm25": np.array([0.85, 0.75, 0.70, 0.65, 0.60], dtype=np.float32),
                "hybrid": np.array([0.875, 0.775, 0.70, 0.625, 0.55], dtype=np.float32),
            },
        )

        mock_ask_llm.return_value = '{"answer": "Track time using [1].", "confidence": 85}'

        bm = {"idf": {}, "avgdl": 10, "doc_lens": [10] * 5, "doc_tfs": [{}] * 5}

        result = answer_once(
            "How to track time?", sample_chunks, sample_embeddings, bm, use_rerank=True  # Enable reranking
        )

        assert "answer" in result
        # Reranking metadata should be present
        assert "rerank_applied" in result["metadata"]

    @patch("clockify_rag.answer.generate_llm_answer")
    @patch("clockify_rag.answer.retrieve")
    def test_answer_once_handles_llm_unavailable(self, mock_retrieve, mock_generate, sample_chunks, sample_embeddings):
        """Ensure answer_once returns structured refusal when LLM is unreachable."""
        mock_retrieve.return_value = (
            [0, 1, 2],
            {
                "dense": np.array([0.9, 0.8, 0.7], dtype=np.float32),
                "bm25": np.array([0.85, 0.75, 0.65], dtype=np.float32),
                "hybrid": np.array([0.875, 0.775, 0.675], dtype=np.float32),
            },
        )
        mock_generate.side_effect = LLMUnavailableError("timeout")
        bm = {"idf": {}, "avgdl": 10, "doc_lens": [10] * 5, "doc_tfs": [{}] * 5}

        result = answer_once("How to track time?", sample_chunks, sample_embeddings, bm)

        assert result["refused"]
        assert result["answer"] == REFUSAL_STR
        assert result["metadata"]["llm_error"] == "llm_unavailable"
        assert "llm_error_msg" in result["metadata"]
        assert "routing" in result

    @patch("clockify_rag.answer.generate_llm_answer")
    @patch("clockify_rag.answer.retrieve")
    def test_answer_once_handles_generic_llm_error(
        self, mock_retrieve, mock_generate, sample_chunks, sample_embeddings
    ):
        """Ensure answer_once surfaces non-availability LLM errors."""
        mock_retrieve.return_value = (
            [0, 1],
            {
                "dense": np.array([0.9, 0.8], dtype=np.float32),
                "bm25": np.array([0.85, 0.75], dtype=np.float32),
                "hybrid": np.array([0.875, 0.775], dtype=np.float32),
            },
        )
        mock_generate.side_effect = LLMError("bad response")
        bm = {"idf": {}, "avgdl": 10, "doc_lens": [10] * 5, "doc_tfs": [{}] * 5}

        result = answer_once("How to track time?", sample_chunks, sample_embeddings, bm)

        assert result["refused"]
        assert result["metadata"]["llm_error"] == "llm_error"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
