import pytest
from fastapi.testclient import TestClient

import clockify_rag.api as api_module
import clockify_rag.config as config
from clockify_rag.exceptions import ValidationError


def test_api_query_returns_metadata(monkeypatch):
    """API should expose metadata from the new answer_once result schema."""

    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: ([], [], {}, None))

    result_payload = {
        "answer": "Mocked answer",
        "confidence": 0.91,
        "selected_chunks": [3, 4, 5],
        "selected_chunk_ids": ["doc-3", "doc-4", "doc-5"],
        "metadata": {"used_tokens": 128, "retrieval_count": 3},
        "routing": {"action": "self-serve"},
    }

    monkeypatch.setattr(api_module, "answer_once", lambda *_, **__: result_payload)

    app = api_module.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={"question": "How do I track time?"},
        )

        assert response.status_code == 200
        payload = response.json()

        assert payload["answer"] == result_payload["answer"]
        assert payload["sources"] == result_payload["selected_chunk_ids"][:5]
        assert payload["metadata"]["used_tokens"] == result_payload["metadata"]["used_tokens"]
        assert payload["routing"] == result_payload["routing"]


@pytest.mark.parametrize(
    "question,message",
    [
        ("bad-empty", "Query cannot be empty"),
        (
            "bad-long",
            "Query too long (12001 chars). Maximum allowed: 12000 chars. Set MAX_QUERY_LENGTH env var to override.",
        ),
    ],
)
def test_api_query_validation_errors(monkeypatch, question, message):
    """API should convert validation errors into 400 responses."""

    # Set MAX_QUERY_LENGTH to match test expectations (12000 chars)
    monkeypatch.setattr(config, "MAX_QUERY_LENGTH", 12000)

    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: ([], [], {}, None))

    def fake_answer(q, *_args, **_kwargs):
        assert q == question
        raise ValidationError(message)

    monkeypatch.setattr(api_module, "answer_once", fake_answer)

    app = api_module.create_app()

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"question": question})

    assert response.status_code == 400
    assert response.json()["detail"] == message
