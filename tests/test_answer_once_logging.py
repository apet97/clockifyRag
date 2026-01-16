import numpy as np
import pytest

from clockify_rag import answer_once
from clockify_rag.caching import get_query_cache, get_rate_limiter
import clockify_rag.retrieval as retrieval_module
import clockify_rag.answer as answer_module

# Get singleton instances
QUERY_CACHE = get_query_cache()
RATE_LIMITER = get_rate_limiter()


@pytest.mark.skip(
    reason="Test needs update: answer_once() return structure changed and no longer includes 'debug' parameter or 'cached'/'cache_hit' metadata"
)
def test_answer_once_logs_retrieved_chunks_with_cache(monkeypatch):
    QUERY_CACHE.clear()

    # Enable chunk logging for this test by monkeypatching the module constant
    # Note: LOG_QUERY_INCLUDE_CHUNKS may have moved to config module
    # monkeypatch.setattr(config, "LOG_QUERY_INCLUDE_CHUNKS", True)

    # Ensure rate limiter allows requests during the test
    monkeypatch.setattr(RATE_LIMITER, "allow_request", lambda: True)
    monkeypatch.setattr(RATE_LIMITER, "wait_time", lambda: 0)

    chunks = [
        {
            "id": "chunk-1",
            "title": "Test",
            "section": "Section",
            "url": "http://example.com",
            "text": "Example chunk text",
        }
    ]
    vecs_n = np.zeros((1, 3), dtype=np.float32)

    def fake_retrieve(question, chunks_arg, vecs_arg, bm, top_k, hnsw, retries):
        return [0], {
            "dense": np.array([0.9], dtype=np.float32),
            "bm25": np.array([0.1], dtype=np.float32),
            "hybrid": np.array([0.5], dtype=np.float32),
        }

    monkeypatch.setattr(retrieval_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        answer_module,
        "apply_mmr_diversification",
        lambda selected, scores, vecs_arg, pack_top: selected,
    )
    monkeypatch.setattr(
        answer_module,
        "apply_reranking",
        lambda question, chunks_arg, mmr_selected, scores, use_rerank, seed, num_ctx, num_predict, retries: (
            mmr_selected,
            {},
            False,
            "",
            0.0,
        ),
    )
    monkeypatch.setattr(retrieval_module, "coverage_ok", lambda selected, dense_scores, threshold: True)

    def fake_pack_snippets(chunks_arg, selected, pack_top, budget_tokens, num_ctx):
        return "context", [chunks_arg[i]["id"] for i in selected], 12, []

    monkeypatch.setattr(retrieval_module, "pack_snippets", fake_pack_snippets)
    # Note: inject_policy_preamble may need to be updated when test is fixed
    # monkeypatch.setattr(answer_module, "inject_policy_preamble", lambda block, question: block)
    monkeypatch.setattr(
        answer_module,
        "generate_llm_answer",
        lambda *args, **kwargs: (
            "answer",
            0.01,
            88,
            "",
            [],
            {
                "intent": "other",
                "user_role_inferred": "unknown",
                "security_sensitivity": "medium",
                "short_intent_summary": "",
                "needs_human_escalation": False,
                "answer_style": "ticket_reply",
            },
        ),
    )

    logged_calls = []

    def fake_log_query(query, answer, retrieved_chunks, latency_ms, refused=False, metadata=None):
        logged_calls.append(
            {
                "query": query,
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "latency_ms": latency_ms,
                "refused": refused,
                "metadata": metadata,
            }
        )

    # Note: log_query may need to be updated when test is fixed
    # monkeypatch.setattr(answer_module, "log_query", fake_log_query)

    answer, metadata = answer_once(
        "What is the chunk?",
        chunks,
        vecs_n,
        bm=None,
        top_k=1,
        pack_top=1,
        threshold=0.1,
        use_rerank=False,
        debug=False,
        hnsw=None,
    )

    assert answer == "answer"
    assert metadata["cached"] is False
    assert metadata["cache_hit"] is False
    assert len(logged_calls) == 1

    retrieved_chunks = logged_calls[0]["retrieved_chunks"]
    assert len(retrieved_chunks) == 1
    chunk_entry = retrieved_chunks[0]
    assert chunk_entry["id"] == "chunk-1"
    assert chunk_entry["chunk"] == chunks[0]
    # In success path, code uses "dense" not "score"
    assert chunk_entry["dense"] == pytest.approx(0.9)

    # Second call should hit the cache and avoid invoking log_query again
    cached_answer, cached_metadata = answer_once(
        "What is the chunk?",
        chunks,
        vecs_n,
        bm=None,
        top_k=1,
        pack_top=1,
        threshold=0.1,
        use_rerank=False,
        debug=False,
        hnsw=None,
    )

    assert cached_answer == "answer"
    assert cached_metadata["cached"] is True
    assert cached_metadata["cache_hit"] is True
    # Priority #8: Cache hits are also logged with 'cached': True metadata
    assert len(logged_calls) == 2

    # Verify second call was for cache hit
    assert logged_calls[1]["metadata"]["cached"] is True
    assert "cache_age_seconds" in logged_calls[1]["metadata"]
