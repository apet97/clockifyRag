import numpy as np

from clockify_rag import retrieval
from clockify_rag.indexing import build_bm25


def test_hub_chunks_are_downweighted(monkeypatch):
    """Hub/category pages should not outrank specific articles."""

    chunks = [
        {
            "id": "article",
            "title": "Lock timesheets",
            "section": "How-to",
            "text": "Step-by-step instructions",
            "url": "#article",
            "metadata": {"is_hub": False},
        },
        {
            "id": "hub",
            "title": "Timesheets hub",
            "section": "Overview",
            "text": "Overview of timesheets content",
            "url": "#hub",
            "metadata": {"is_hub": True},
        },
    ]

    # Dense scores: non-hub slightly higher; BM25: hub slightly higher
    vecs_n = np.array([[1.0, 0.0], [0.8, 0.0]], dtype=np.float32)
    vecs_n = vecs_n / np.linalg.norm(vecs_n, axis=1, keepdims=True)
    bm = build_bm25(chunks)

    # Keep raw scores (no z-score) to make weighting deterministic
    monkeypatch.setattr(retrieval, "normalize_scores_zscore", lambda arr: np.asarray(arr, dtype=np.float32))
    monkeypatch.setattr(retrieval, "embed_query", lambda question, retries=0: np.array([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(retrieval, "bm25_scores", lambda q, b, top_k=None: np.array([1.0, 1.4], dtype=np.float32))

    filtered, scores = retrieval.retrieve("timesheets hub", chunks, vecs_n, bm, top_k=2)

    assert chunks[filtered[0]]["metadata"]["is_hub"] is False
    assert scores["hybrid"][0] > scores["hybrid"][1]
