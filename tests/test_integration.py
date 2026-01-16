"""Integration tests for end-to-end RAG pipeline.

This module tests the complete workflow from knowledge base building
to query processing. Tests use mocked embeddings to avoid external dependencies.
"""

import pytest
import os
import tempfile
import json
import numpy as np
from unittest.mock import patch

# Import from clockify_rag package
from clockify_rag import (
    build,
    load_index,
    build_chunks,
)
from clockify_rag.utils import ALLOWED_CORPUS_FILENAME


@pytest.fixture
def sample_kb_path(tmp_path):
    """Path to the sample knowledge base."""
    sample = tmp_path / "knowledge_helpcenter.md"
    sample.write_text(
        """# [ARTICLE] Track time
https://clockify.me/help/track-time-and-expenses/creating-a-time-entry

## Key points
- Track time with a running timer.

## Body
Use the web app or desktop app to start the timer and record time.

# [ARTICLE] Lock timesheets
https://clockify.me/help/timesheets/lock-timesheets

## Key points
- Lock timesheets from workspace settings.
""",
        encoding="utf-8",
    )
    return str(sample)


@pytest.fixture
def temp_build_dir():
    """Temporary directory for building the index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def built_index(sample_kb_path, temp_build_dir):
    """Build a complete index from the sample KB.

    This fixture creates a full index with chunking, embedding, and indexing.
    It uses mock embeddings to avoid external dependencies.

    Returns (idx, temp_build_dir) tuple.
    """

    # Mock embedding function
    def mock_embed_batch(texts, normalize=False):
        n = len(texts)
        vecs = np.random.randn(n, 384).astype("float32")
        if normalize:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    # Save current directory and change to temp_build_dir
    original_dir = os.getcwd()
    try:
        os.chdir(temp_build_dir)
        # Mock the embed_local_batch function
        with patch("clockify_rag.indexing.embed_local_batch", side_effect=mock_embed_batch):
            # Build the index
            build(sample_kb_path)
        # Load and return the index
        idx = load_index()
    finally:
        os.chdir(original_dir)

    # Verify index structure
    assert "chunks" in idx, "Index should have chunks"
    assert "vecs_n" in idx or "vecs" in idx, "Index should have embeddings"
    assert "bm" in idx or "bm25" in idx, "Index should have BM25 index"

    # Normalize keys for consistency
    if "vecs_n" in idx and "vecs" not in idx:
        idx["vecs"] = idx["vecs_n"]
    if "bm" in idx and "bm25" not in idx:
        idx["bm25"] = idx["bm"]

    return idx, temp_build_dir


class TestBuildPipeline:
    """Test the knowledge base build pipeline."""

    def test_build_creates_all_artifacts(self, sample_kb_path, temp_build_dir):
        """Test that build creates all required index files."""

        # Mock embeddings to avoid external dependencies
        def mock_embed_batch(texts, normalize=False):
            n = len(texts)
            vecs = np.random.randn(n, 384).astype("float32")
            if normalize:
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

        # Save current directory and change to temp_build_dir
        original_dir = os.getcwd()
        try:
            os.chdir(temp_build_dir)
            with patch("clockify_rag.indexing.embed_local_batch", side_effect=mock_embed_batch):
                # Build the index
                build(sample_kb_path)
        finally:
            os.chdir(original_dir)

        # Check all required files exist
        assert os.path.exists(os.path.join(temp_build_dir, "chunks.jsonl"))
        assert os.path.exists(os.path.join(temp_build_dir, "vecs_n.npy"))
        assert os.path.exists(os.path.join(temp_build_dir, "bm25.json"))
        assert os.path.exists(os.path.join(temp_build_dir, "index.meta.json"))

        # Verify metadata
        with open(os.path.join(temp_build_dir, "index.meta.json")) as f:
            meta = json.load(f)
            assert "chunks" in meta
            assert "built_at" in meta
            assert meta["chunks"] > 0

    def test_chunks_are_created(self, sample_kb_path):
        """Test that chunking creates valid chunks from KB."""
        chunks = build_chunks(sample_kb_path)

        # Should create multiple chunks
        assert len(chunks) > 0, "Should create at least one chunk"
        assert len(chunks) > 1, "Sample KB should create multiple chunks"

        # Each chunk should have required fields
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "title" in chunk
            assert chunk["text"], "Chunk text should not be empty"


class TestIndexLoading:
    """Test index loading and structure."""

    def test_index_loads_with_correct_structure(self, built_index):
        """Test that index loads with all required components."""
        idx, _ = built_index

        # Check required keys
        assert "chunks" in idx
        assert "vecs" in idx or "vecs_n" in idx
        assert "bm25" in idx or "bm" in idx
        assert "meta" in idx

        # Check chunks structure
        assert len(idx["chunks"]) > 0
        for chunk in idx["chunks"]:
            assert "id" in chunk
            assert "text" in chunk

        # Check embeddings
        vecs = idx.get("vecs") if "vecs" in idx else idx.get("vecs_n")
        assert vecs is not None, "Index should have vecs or vecs_n"
        assert vecs.shape[0] == len(idx["chunks"])
        assert vecs.shape[1] == 384  # Local embedding dimension

        # Check BM25 structure
        bm25 = idx.get("bm25") or idx.get("bm")
        assert bm25 is not None
        assert "doc_lens" in bm25
        assert "avgdl" in bm25
        assert "idf" in bm25

    def test_metadata_is_valid(self, built_index):
        """Test that index metadata contains required information."""
        idx, _ = built_index

        meta = idx["meta"]
        assert "chunks" in meta
        assert "built_at" in meta
        assert "emb_backend" in meta
        assert meta["chunks"] == len(idx["chunks"])


class TestEdgeCases:
    """Test edge cases in chunking and building."""

    def test_empty_file_handling(self, temp_build_dir):
        """Test handling of empty or minimal knowledge base."""
        # Create a minimal KB file
        minimal_kb = os.path.join(temp_build_dir, ALLOWED_CORPUS_FILENAME)
        with open(minimal_kb, "w") as f:
            f.write("# Minimal\n\nJust a test.")

        def mock_embed_batch(texts, normalize=False):
            n = len(texts)
            vecs = np.random.randn(n, 384).astype("float32")
            if normalize:
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

        original_dir = os.getcwd()
        try:
            os.chdir(temp_build_dir)
            with patch("clockify_rag.indexing.embed_local_batch", side_effect=mock_embed_batch):
                build(minimal_kb)

            # Should still create valid index
            assert os.path.exists(os.path.join(temp_build_dir, "chunks.jsonl"))
            assert os.path.exists(os.path.join(temp_build_dir, "vecs_n.npy"))

            idx = load_index()
            assert len(idx["chunks"]) >= 1
        finally:
            os.chdir(original_dir)


class TestPerformance:
    """Test performance characteristics."""

    def test_build_completes_quickly(self, sample_kb_path, temp_build_dir):
        """Test that build completes in reasonable time."""
        import time

        def mock_embed_batch(texts, normalize=False):
            n = len(texts)
            vecs = np.random.randn(n, 384).astype("float32")
            if normalize:
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

        original_dir = os.getcwd()
        try:
            os.chdir(temp_build_dir)
            start_time = time.time()

            with patch("clockify_rag.indexing.embed_local_batch", side_effect=mock_embed_batch):
                build(sample_kb_path)

            elapsed = time.time() - start_time

            # Build should complete in <10 seconds for small KB with mocked embeddings
            assert elapsed < 10, f"Build took too long: {elapsed:.2f}s"
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
