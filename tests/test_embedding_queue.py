import numpy as np


def test_embed_texts_does_not_drop_pending(monkeypatch):
    import clockify_rag.config as config
    import clockify_rag.embedding as embedding

    # Patch config values (not embedding module directly)
    monkeypatch.setattr(config, "EMB_MAX_WORKERS", 2)
    monkeypatch.setattr(config, "EMB_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "EMB_BACKEND", "ollama")

    def fake_embed_single_text(index, text, retries, total):
        return index, [float(index)]

    monkeypatch.setattr(embedding, "_embed_single_text", fake_embed_single_text)

    texts = [f"text-{i}" for i in range(2 * 2 + 1)]  # > EMB_MAX_WORKERS * EMB_BATCH_SIZE

    result = embedding.embed_texts(texts)

    assert result.shape[0] == len(texts)
    assert np.array_equal(result.flatten(), np.arange(len(texts), dtype=np.float32))
