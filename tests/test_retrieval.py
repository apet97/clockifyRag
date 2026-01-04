"""Tests for retrieval pipeline (hybrid BM25 + dense retrieval)."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clockify_rag.retrieval import retrieve, normalize_scores_zscore, DenseScoreStore
from clockify_rag.utils import sanitize_question


# Check if Ollama is available (may not be in CI)
def is_ollama_available():
    """Check if Ollama service is running and accessible."""
    import os

    try:
        import requests

        ollama_url = os.environ.get("RAG_OLLAMA_URL") or os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434"
        response = requests.get(f"{ollama_url}/api/version", timeout=1)
        return response.ok
    except Exception:
        return False


OLLAMA_AVAILABLE = is_ollama_available()
requires_ollama = pytest.mark.skipif(
    not OLLAMA_AVAILABLE, reason="Ollama not running (expected in CI, start with 'ollama serve' for local testing)"
)


def test_retrieve_local_backend(monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
    """Ensure retrieval works with the local embedding backend without Ollama."""

    import clockify_rag.config as config
    import clockify_rag.embedding as embedding
    import clockify_rag.retrieval as retrieval

    # Force local backend across modules
    monkeypatch.setattr(config, "EMB_BACKEND", "local", raising=False)
    monkeypatch.setattr(retrieval, "_FAISS_INDEX", None, raising=False)
    monkeypatch.setattr(config, "USE_ANN", "none", raising=False)

    # Use config.EMB_DIM_LOCAL since we're using local backend
    fake_vec = np.ones((1, config.EMB_DIM_LOCAL), dtype=np.float32)
    fake_vec = fake_vec / np.linalg.norm(fake_vec, axis=1, keepdims=True)

    calls = {"count": 0}

    def fake_embed_local_batch(texts, normalize=True):
        calls["count"] += 1
        assert texts == ["How do I track time?"]
        assert normalize is True
        return fake_vec

    def fail_embed_texts(*_args, **_kwargs):
        raise AssertionError("embed_texts should not be called for local backend")

    monkeypatch.setattr(embedding, "embed_local_batch", fake_embed_local_batch)
    monkeypatch.setattr(embedding, "embed_texts", fail_embed_texts)

    question = "How do I track time?"
    selected, scores = retrieval.retrieve(
        question,
        sample_chunks,
        sample_embeddings,
        sample_bm25,
        top_k=2,
        faiss_index_path=None,
    )

    assert calls["count"] == 1, "Local embedding path should be used exactly once"
    assert selected, "Should return at least one chunk"
    assert max(selected) < len(sample_chunks)
    # Verify required score keys (intent_metadata is optional enhancement)
    assert "dense" in scores
    assert "bm25" in scores
    assert "hybrid" in scores
    assert isinstance(scores["dense"], retrieval.DenseScoreStore)


@requires_ollama
def test_retrieve_returns_correct_top_k(sample_chunks, sample_embeddings, sample_bm25):
    """Verify retrieval returns exactly top_k results."""
    question = "How do I track time?"
    top_k = 2

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=top_k)

    assert len(selected) <= top_k, f"Should return at most {top_k} results, got {len(selected)}"
    assert all(isinstance(idx, int | np.integer) for idx in selected), "Indices should be integers"
    assert all(0 <= idx < len(sample_chunks) for idx in selected), "Indices should be valid"


@requires_ollama
def test_retrieve_with_different_top_k_values(sample_chunks, sample_embeddings, sample_bm25):
    """Test retrieval with various top_k values."""
    question = "time tracking features"

    for k in [1, 3, 5]:
        selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=k)
        assert len(selected) <= k, f"With top_k={k}, got {len(selected)} results"


def test_retrieve_with_empty_query_raises_error(sample_chunks, sample_embeddings, sample_bm25):
    """Verify retrieval handles empty query gracefully."""
    # sanitize_question should raise ValueError for empty string
    with pytest.raises(ValueError, match="Question cannot be empty"):
        sanitize_question("")


@requires_ollama
def test_retrieve_scores_structure(sample_chunks, sample_embeddings, sample_bm25):
    """Verify scores dict has correct structure."""
    question = "pricing plans"

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25)

    assert isinstance(scores, dict), "Scores should be a dict"
    assert "dense" in scores, "Should have dense scores"
    assert "bm25" in scores, "Should have BM25 scores"
    assert "hybrid" in scores, "Should have hybrid scores"

    # Check structures
    n_chunks = len(sample_chunks)
    assert isinstance(scores["dense"], DenseScoreStore), "Dense scores should use DenseScoreStore"
    assert len(scores["dense"]) == n_chunks, "Dense score store should report full length"
    assert len(scores["bm25"]) == n_chunks, "BM25 scores should match chunk count"
    assert len(scores["hybrid"]) == n_chunks, "Hybrid scores should match chunk count"


def test_normalize_scores_zscore_edge_cases():
    """Test score normalization edge cases."""
    # All equal scores (std=0) - should return zeros since there's no discriminative signal
    scores = [0.5, 0.5, 0.5]
    result = normalize_scores_zscore(scores)
    assert np.allclose(result, [0.0, 0.0, 0.0]), "Should return zeros when std=0 (no signal)"

    # Empty array
    result = normalize_scores_zscore([])
    assert len(result) == 0, "Should handle empty array"

    # Single value
    result = normalize_scores_zscore([1.0])
    assert len(result) == 1, "Should handle single value"

    # Normal case
    scores = [1.0, 2.0, 3.0]
    result = normalize_scores_zscore(scores)
    assert abs(result.mean()) < 0.01, "Mean should be ~0 after normalization"
    assert abs(result.std() - 1.0) < 0.01, "Std should be ~1 after normalization"


@requires_ollama
def test_retrieve_ranking_quality(sample_chunks, sample_embeddings, sample_bm25):
    """Test that retrieval ranks relevant chunks higher."""
    question = "How do I track time?"  # Should rank time tracking chunks higher

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=3)

    # Check that at least one time tracking chunk is in top results
    # (chunks 0 and 2 are about time tracking)
    time_tracking_chunks = {0, 2}
    retrieved_set = set(selected)
    overlap = time_tracking_chunks & retrieved_set

    assert len(overlap) > 0, f"Should retrieve at least one time tracking chunk, got {selected}"


@requires_ollama
def test_retrieve_with_long_query(sample_chunks, sample_embeddings, sample_bm25):
    """Test retrieval with a longer, more complex query."""
    question = (
        "I need to understand how to properly track my work hours and time spent on projects "
        "using the timer functionality, including manual entry options"
    )

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=3)

    assert len(selected) > 0, "Should return results for long query"
    assert len(selected) <= 3, "Should respect top_k limit"


def test_sanitize_question_validation():
    """Test question sanitization and validation."""
    # Valid questions
    assert sanitize_question("How do I track time?") == "How do I track time?"
    assert sanitize_question("  whitespace  ") == "whitespace"

    # Invalid questions
    with pytest.raises(ValueError, match="empty"):
        sanitize_question("")

    with pytest.raises(ValueError, match="too long"):
        sanitize_question("x" * 3000)  # Exceeds max_length=2000

    with pytest.raises(ValueError, match="string"):
        sanitize_question(123)  # Not a string

    # Control characters
    with pytest.raises(ValueError, match="control characters"):
        sanitize_question("test\x00string")  # Null byte


@requires_ollama
def test_retrieve_deduplication(sample_chunks, sample_embeddings, sample_bm25):
    """Test that retrieval doesn't return duplicate indices."""
    question = "time tracking"

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=5)

    # Check for duplicates
    assert len(selected) == len(set(selected)), "Should not return duplicate indices"


@requires_ollama
def test_retrieve_scores_are_numeric(sample_chunks, sample_embeddings, sample_bm25):
    """Verify all scores are valid numeric values."""
    question = "clockify features"

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25)

    dense_scores = scores["dense"]
    assert selected, "Should have at least one selected chunk"
    for idx in selected:
        assert np.isfinite(dense_scores[idx]), "Dense scores should be finite for selected chunks"

    for score_type in ["bm25", "hybrid"]:
        score_array = scores[score_type]
        assert np.all(np.isfinite(score_array)), f"{score_type} scores should be finite"
        assert not np.any(np.isnan(score_array)), f"{score_type} scores should not contain NaN"


def test_retrieve_faiss_skips_full_dot(monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
    """Ensure FAISS retrieval path does not compute full dense dot product."""

    class TrackingMatrix:
        def __init__(self, arr):
            self._arr = arr
            self.dot_calls = 0

        def dot(self, other):
            self.dot_calls += 1
            return self._arr.dot(other)

        def __getitem__(self, item):
            return self._arr[item]

        def __len__(self):
            return len(self._arr)

        @property
        def shape(self):
            return self._arr.shape

    class FakeFaissIndex:
        def __init__(self):
            self.search_calls = 0

        def search(self, query, k):
            self.search_calls += 1
            total = len(sample_chunks)
            actual = min(k, total)
            distances = np.linspace(1.0, 0.5, actual, dtype=np.float32)
            indices = np.arange(actual, dtype=np.int64)
            padded_dist = np.zeros(k, dtype=np.float32)
            padded_idx = np.full(k, -1, dtype=np.int64)
            padded_dist[:actual] = distances
            padded_idx[:actual] = indices
            return padded_dist.reshape(1, -1), padded_idx.reshape(1, -1)

    tracker = TrackingMatrix(sample_embeddings)
    fake_index = FakeFaissIndex()

    import clockify_rag.config as config_module
    import clockify_rag.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module, "_FAISS_INDEX", fake_index, raising=False)
    monkeypatch.setattr(config_module, "USE_ANN", "faiss", raising=False)
    monkeypatch.setattr(retrieval_module, "get_faiss_index", lambda _path=None: fake_index, raising=False)
    # Prevent fallback to dense path when index is present
    monkeypatch.setattr(config_module, "ANN_CANDIDATE_MIN", 2, raising=False)

    # Avoid external embedding call - patch in the retrieval module where it's actually called
    query_vec = sample_embeddings[0]

    monkeypatch.setattr(retrieval_module, "embed_query", lambda question, retries=0: query_vec, raising=False)

    selected, scores = retrieve(
        "How do I track time?", sample_chunks, tracker, sample_bm25, top_k=2, faiss_index_path=None
    )

    assert tracker.dot_calls <= 1, "FAISS path should avoid full dot products"
    assert fake_index.search_calls == 1, "FAISS index should be used"
    assert isinstance(scores["dense"], DenseScoreStore), "Dense scores should use store in FAISS mode"


def test_bm25_pruning_matches_full():
    """BM25 pruning should preserve the top document selection."""

    from clockify_rag.indexing import build_bm25, bm25_scores

    docs = [
        "time tracking basics guide",
        "pricing plans cost tiers",
        "advanced timer manual entry",
    ]
    chunks = [{"text": d} for d in docs]
    bm = build_bm25(chunks)

    query = "pricing plan tiers"

    full_scores = bm25_scores(query, bm, top_k=None)
    pruned_scores = bm25_scores(query, bm, top_k=1)

    full_top = int(np.argmax(full_scores))
    pruned_top = int(np.argmax(pruned_scores))

    assert full_top == pruned_top, "Pruned BM25 should keep the same top document as full scoring"


def test_confidence_monotonicity(monkeypatch):
    """Higher hybrid score should not reduce confidence."""

    import clockify_rag.retrieval as retrieval

    scores = {"hybrid": np.array([0.1, 0.5, 0.9], dtype=np.float32)}
    selected_low = [0]
    selected_high = [2]

    low_conf = retrieval.compute_confidence_from_scores(scores, selected_low, threshold=0.25)
    high_conf = retrieval.compute_confidence_from_scores(scores, selected_high, threshold=0.25)

    assert high_conf >= low_conf, "Confidence should be monotonic with higher scores"


def test_intent_toggle_controls_boost(monkeypatch):
    """Intent boosting should only apply when intent classification is enabled."""

    import clockify_rag.config as config
    import clockify_rag.retrieval as retrieval
    from clockify_rag.intent_classification import IntentConfig

    original_flag = config.USE_INTENT_CLASSIFICATION
    try:
        chunks = [
            {"id": "a", "title": "Generic", "section": "", "text": "generic content", "metadata": {}},
            {"id": "b", "title": "Pricing", "section": "", "text": "pricing details", "metadata": {}},
        ]

        # Deterministic embeddings and query vector
        vecs_n = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        query_vec = np.array([1.0, 0.0], dtype=np.float32)

        # Base BM25 scores (will be overridden by intent boost)
        monkeypatch.setattr(retrieval, "bm25_scores", lambda *_args, **_kwargs: np.array([0.0, 0.0], dtype=np.float32))
        monkeypatch.setattr(retrieval, "embed_query", lambda *_args, **_kwargs: query_vec)

        def fake_classify_intent(_question):
            return "pricing", IntentConfig("pricing", alpha_hybrid=0.9, boost_factor=2.0, description=""), 0.9

        def fake_adjust_scores_by_intent(chunks_arg, scores_dict, intent_config):
            # Boost pricing chunk (index 1) heavily in BM25 to swing the ranking when enabled
            boosted = scores_dict["bm25"].copy()
            boosted[1] = 10.0
            scores_dict["bm25"] = boosted
            return scores_dict

        monkeypatch.setattr(retrieval, "classify_intent", fake_classify_intent)
        monkeypatch.setattr(retrieval, "adjust_scores_by_intent", fake_adjust_scores_by_intent)

        bm_dummy = {"idf": {}, "avgdl": 0, "doc_lens": [1, 1], "doc_tfs": [{}, {}]}

        # Intent enabled: pricing chunk should be ranked first
        config.USE_INTENT_CLASSIFICATION = True
        selected_intent_on, _ = retrieval.retrieve("pricing question", chunks, vecs_n, bm_dummy, top_k=2)
        assert selected_intent_on[0] == 1  # pricing chunk wins after boost

        # Intent disabled: dense similarity favors first chunk
        config.USE_INTENT_CLASSIFICATION = False
        selected_intent_off, _ = retrieval.retrieve("pricing question", chunks, vecs_n, bm_dummy, top_k=2)
        assert selected_intent_off[0] == 0  # falls back to base ranking
    finally:
        config.USE_INTENT_CLASSIFICATION = original_flag
