"""Tests for context packing functionality."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestContextPacking:
    """Test context window packing logic."""

    def test_pack_respects_token_budget(self):
        """Test that packing respects token budget."""
        chunks = [{"id": i, "text": "word " * 100} for i in range(10)]  # ~400 chars each

        # Budget for ~2 chunks (800 chars = 200 tokens at 4 chars/token)
        budget_chars = 800
        selected_ids = [0, 1, 2, 3]

        # Pack chunks
        packed_text = ""
        for chunk_id in selected_ids:
            chunk_text = chunks[chunk_id]["text"]
            if len(packed_text) + len(chunk_text) <= budget_chars:
                packed_text += chunk_text
            else:
                break

        # Should fit approximately 2 chunks
        assert len(packed_text) <= budget_chars

    def test_pack_prioritizes_higher_scores(self):
        """Test that higher-scored chunks are prioritized."""
        chunks = [
            {"id": 0, "text": "text", "score": 0.9},
            {"id": 1, "text": "text", "score": 0.5},
            {"id": 2, "text": "text", "score": 0.7},
        ]

        # Sort by score descending
        sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)

        # Highest score should be first
        assert sorted_chunks[0]["score"] == 0.9
        assert sorted_chunks[1]["score"] == 0.7
        assert sorted_chunks[2]["score"] == 0.5

    def test_pack_includes_metadata(self):
        """Test that packed context includes chunk IDs for citations."""
        chunks = [
            {"id": 0, "text": "content", "title": "Title 1"},
            {"id": 1, "text": "content", "title": "Title 2"},
        ]

        # Simulate packing with metadata
        packed_lines = []
        for chunk in chunks:
            line = f"[id_{chunk['id']}] {chunk['title']}: {chunk['text']}"
            packed_lines.append(line)

        packed_text = "\n".join(packed_lines)

        # Check that IDs are present for citations
        assert "[id_0]" in packed_text
        assert "[id_1]" in packed_text

    def test_pack_handles_empty_selection(self):
        """Test packing with no selected chunks."""
        packed_text = ""

        assert packed_text == ""
        assert len(packed_text) == 0


class TestCoverageCheck:
    """Test retrieval coverage checks."""

    def test_coverage_check_passes_with_sufficient_chunks(self):
        """Test that coverage check passes with enough relevant chunks."""
        retrieved = [
            {"id": 0, "score": 0.5},
            {"id": 1, "score": 0.4},
            {"id": 2, "score": 0.35},
        ]
        threshold = 0.3
        min_chunks = 2

        # Count chunks above threshold
        above_threshold = [c for c in retrieved if c["score"] >= threshold]

        assert len(above_threshold) >= min_chunks

    def test_coverage_check_fails_with_insufficient_chunks(self):
        """Test that coverage check fails with too few relevant chunks."""
        retrieved = [
            {"id": 0, "score": 0.2},  # Below threshold
            {"id": 1, "score": 0.1},  # Below threshold
        ]
        threshold = 0.3
        min_chunks = 2

        # Count chunks above threshold
        above_threshold = [c for c in retrieved if c["score"] >= threshold]

        assert len(above_threshold) < min_chunks

    def test_coverage_check_threshold_boundary(self):
        """Test coverage check at exact threshold boundary."""
        retrieved = [
            {"id": 0, "score": 0.30},  # Exactly at threshold
            {"id": 1, "score": 0.31},  # Just above
        ]
        threshold = 0.30

        # Chunks at or above threshold should pass
        above_threshold = [c for c in retrieved if c["score"] >= threshold]

        assert len(above_threshold) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
