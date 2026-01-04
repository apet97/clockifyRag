"""Tests for chunking functionality."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.chunking import sliding_chunks
from clockify_rag.utils import tokenize


class TestChunker:
    """Test chunking logic."""

    def test_sliding_chunks_small_text(self):
        """Test that small text doesn't get split."""
        text = "This is a short text that shouldn't be split."
        chunks = sliding_chunks(text, maxc=100, overlap=20)
        assert len(chunks) == 1

    def test_sliding_chunks_exact_boundary(self):
        """Test text exactly at maxc boundary."""
        text = "x" * 100
        chunks = sliding_chunks(text, maxc=100, overlap=20)
        assert len(chunks) == 1

    def test_sliding_chunks_needs_split(self):
        """Test that text over maxc gets split with overlap."""
        text = "word " * 100  # 500 characters
        chunks = sliding_chunks(text, maxc=200, overlap=50)
        assert len(chunks) > 1

    def test_sliding_chunks_empty_text(self):
        """Test empty text returns empty or minimal chunks."""
        chunks = sliding_chunks("", maxc=100, overlap=20)
        # Empty text may return empty list or single empty/whitespace chunk
        assert len(chunks) <= 1
        if len(chunks) == 1:
            assert len(chunks[0].strip()) == 0

    def test_sliding_chunks_whitespace_only(self):
        """Test whitespace-only text."""
        chunks = sliding_chunks("   \n\t  ", maxc=100, overlap=20)
        # Should return empty or single whitespace chunk
        assert len(chunks) <= 1


class TestTokenizer:
    """Test tokenization logic."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Hello World"
        tokens = tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_punctuation(self):
        """Test that punctuation is handled."""
        text = "Hello, world! How are you?"
        tokens = tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens

    def test_tokenize_numbers(self):
        """Test number tokenization."""
        text = "There are 123 items"
        tokens = tokenize(text)
        assert "123" in tokens or "items" in tokens

    def test_tokenize_empty(self):
        """Test empty text tokenization."""
        tokens = tokenize("")
        assert len(tokens) == 0

    def test_tokenize_lowercases(self):
        """Test that tokenization lowercases."""
        text = "UPPER Lower MiXeD"
        tokens = tokenize(text)
        # All tokens should be lowercase
        assert all(t.islower() or not t.isalpha() for t in tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
