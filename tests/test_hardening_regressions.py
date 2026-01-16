import requests

from clockify_rag import config
from clockify_rag import embeddings_client
from clockify_rag import precomputed_cache
from clockify_rag import answer as answer_mod


def test_embeddings_client_retries_and_timeouts(monkeypatch):
    """Ensure embed_query retries transient failures and respects configured dimension."""

    calls = []

    def fake_post(url, json_payload, retries, timeout, **kwargs):
        calls.append({"retries": retries, "timeout": timeout})
        if len(calls) == 1:
            raise requests.exceptions.Timeout("boom")
        return {"embedding": [1.0] * config.EMB_DIM}

    monkeypatch.setattr(embeddings_client, "http_post_with_retries", fake_post)

    vec = embeddings_client.embed_query("hello", retries=1)

    assert len(calls) == 2  # one retry after timeout
    assert vec.shape[0] == config.EMB_DIM
    # Ensure timeouts propagated to HTTP helper
    assert calls[0]["timeout"] == (config.EMB_CONNECT_T, config.EMB_READ_T)


def test_precomputed_cache_marks_stale_on_signature_mismatch(tmp_path):
    """Cache should be treated as stale when kb_signature mismatches."""

    cache_path = tmp_path / "faq_cache.json"
    cache = precomputed_cache.PrecomputedCache(kb_signature="sig-old")
    cache.put("Question?", {"answer": "A", "confidence": 100, "refused": False})
    cache.save(cache_path)

    fresh = precomputed_cache.PrecomputedCache(kb_signature="sig-new")
    fresh.load(str(cache_path), kb_signature="sig-new")

    assert fresh.is_stale()
    assert fresh.size() == 0


def test_generate_llm_answer_handles_malformed_json(monkeypatch):
    """Malformed LLM output should not crash and should fall back safely."""

    monkeypatch.setattr(answer_mod, "ask_llm", lambda *a, **k: "not-json")

    chunks = [{"id": 1, "text": "body"}]
    scores = {"hybrid": [0.9], "dense": [0.9], "bm25": [0.5]}

    answer, _, confidence, reasoning, sources, meta = answer_mod.generate_llm_answer(
        question="q",
        context_block="ctx",
        packed_ids=[1],
        all_chunks=chunks,
        selected_indices=[0],
        scores_dict=scores,
    )

    assert answer == "not-json"
    assert isinstance(confidence, int)
    assert reasoning is None
    assert sources is None
    assert meta["intent"] == "other"
