"""End-to-end tests for the RAG system.

This module tests the complete workflow from ingestion to query processing
using a simple test document and mocked external dependencies.
"""

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from clockify_rag.config import FILES, EMB_DIM
from clockify_rag.indexing import build, load_index
from clockify_rag.utils import ALLOWED_CORPUS_FILENAME


def test_end_to_end_rag_pipeline():
    """Test the complete RAG pipeline: build index → load index → answer query."""
    # Create a simple test knowledge base
    test_kb_content = """# [ARTICLE] Test Documentation
https://example.com/test

## Getting Started

This is a test document for the RAG system. It contains information about how to use the system.

## Advanced Features

The RAG system supports various features including:
- Document ingestion
- Chunking and indexing
- Semantic search
- Answer generation

## Troubleshooting

If you encounter issues, try rebuilding the index or checking your configuration.
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir) / ALLOWED_CORPUS_FILENAME
        kb_path.write_text(test_kb_content, encoding="utf-8")

        def _fake_embed(texts, **_kwargs):
            vectors = []
            for text in texts:
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed)
                vectors.append(rng.standard_normal(EMB_DIM))
            return np.array(vectors, dtype="float32")

        def fake_embed_query(text, **kwargs):
            return _fake_embed([text], **kwargs)[0]

        try:
            with (
                mock.patch("clockify_rag.embedding.embed_local_batch", _fake_embed),
                mock.patch("clockify_rag.embedding.embed_query", fake_embed_query),
                mock.patch("clockify_rag.retrieval._embedding_embed_query", fake_embed_query),
                mock.patch("clockify_rag.indexing.embed_local_batch", _fake_embed),
            ):
                # Ensure previous embedding cache entries don't leak into this test
                Path(FILES["emb_cache"]).unlink(missing_ok=True)

                # Test 1: Build the index
                build(str(kb_path))

                # Verify index files were created
                for file_key in ["chunks", "emb", "meta", "bm25", "index_meta"]:
                    assert Path(FILES[file_key]).exists(), f"Index file {FILES[file_key]} was not created"

                # Test 2: Load the index
                index_data = load_index()
                assert index_data is not None, "Failed to load index"
                assert "chunks" in index_data, "Chunks not found in loaded index"
                assert "vecs_n" in index_data, "Vectors not found in loaded index"
                assert "bm" in index_data, "BM25 index not found in loaded index"

                chunks = index_data["chunks"]
                vecs_n = index_data["vecs_n"]
                bm = index_data["bm"]

                assert len(chunks) > 0, "No chunks were created"
                assert len(vecs_n) == len(chunks), "Vector count doesn't match chunk count"

                # Test 3: Answer a query using the loaded index
                question = "What features does the RAG system support?"

                # Use answer_once to test the full pipeline.
                # For this test, we'll simulate the retrieval part.
                from clockify_rag.retrieval import retrieve, pack_snippets
                from clockify_rag.answer import apply_mmr_diversification

                # Test retrieval
                from clockify_rag.embedding import embed_query

                qv_n = embed_query(question, retries=0)
                assert qv_n.shape[-1] == vecs_n.shape[1], "Query vector dimension mismatch with stored embeddings"

                # Since we can't run a real LLM in this test without external dependencies,
                # we'll test the retrieval components
                selected, scores = retrieve(question, chunks, vecs_n, bm, top_k=5)

                # Test that we got results
                assert len(selected) > 0, "No results retrieved"
                assert len(selected) <= 5, "Too many results returned"

                # Test MMR diversification
                mmr_selected = apply_mmr_diversification(selected, scores, vecs_n, pack_top=3)

                assert len(mmr_selected) <= 3, "MMR diversification returned too many results"
                assert all(i in selected for i in mmr_selected), "MMR selected items not in original results"

                # Test snippet packing
                context_block, packed_ids, used_tokens, article_blocks = pack_snippets(chunks, mmr_selected, pack_top=3)

                assert len(context_block) > 0, "Context block is empty"
                assert len(packed_ids) == len(mmr_selected), "Packed IDs count doesn't match"
                assert used_tokens >= 0, "Used tokens should be non-negative"
                assert isinstance(article_blocks, list), "Article blocks should be a list"

                print("✓ End-to-end test passed!")
                print(f"  - Built index with {len(chunks)} chunks")
                print(f"  - Retrieved {len(selected)} candidates")
                print(f"  - Applied MMR to get {len(mmr_selected)} diversified results")
                print(f"  - Packed {len(packed_ids)} snippets into context")
        finally:
            # Clean up temporary files
            kb_path.unlink(missing_ok=True)

            # Remove index files created during test
            for file_key in ["chunks", "emb", "meta", "bm25", "index_meta", "faiss_index", "emb_cache"]:
                file_path = FILES[file_key]
                Path(file_path).unlink(missing_ok=True)


def test_config_validation():
    """Test that configuration is properly loaded and validated."""
    import clockify_rag.config as config

    # Test that required config values exist (new RAG_* namespace + legacy aliases)
    assert hasattr(config, "RAG_OLLAMA_URL"), "RAG_OLLAMA_URL not found in config"
    assert hasattr(config, "RAG_CHAT_MODEL"), "RAG_CHAT_MODEL not found in config"
    assert hasattr(config, "RAG_EMBED_MODEL"), "RAG_EMBED_MODEL not found in config"
    # Verify legacy aliases still resolve for backwards compatibility
    assert hasattr(config, "OLLAMA_URL") and config.RAG_OLLAMA_URL == config.OLLAMA_URL
    assert hasattr(config, "GEN_MODEL") and config.RAG_CHAT_MODEL == config.GEN_MODEL
    assert hasattr(config, "EMB_MODEL") and config.RAG_EMBED_MODEL == config.EMB_MODEL
    assert hasattr(config, "DEFAULT_TOP_K"), "DEFAULT_TOP_K not found in config"

    # Test that config values have reasonable defaults
    assert isinstance(config.RAG_OLLAMA_URL, str), "RAG_OLLAMA_URL should be a string"
    assert isinstance(config.DEFAULT_TOP_K, int), "DEFAULT_TOP_K should be an integer"
    assert config.DEFAULT_TOP_K > 0, "DEFAULT_TOP_K should be positive"

    print("✓ Configuration validation passed!")
    print(f"  - RAG_OLLAMA_URL: {config.RAG_OLLAMA_URL}")
    print(f"  - RAG_CHAT_MODEL: {config.RAG_CHAT_MODEL}")
    print(f"  - RAG_EMBED_MODEL: {config.RAG_EMBED_MODEL}")


if __name__ == "__main__":
    test_config_validation()
    test_end_to_end_rag_pipeline()
    print("\n✓ All end-to-end tests passed!")
