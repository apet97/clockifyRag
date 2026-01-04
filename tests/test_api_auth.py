"""Authentication tests for sensitive API endpoints."""

from fastapi.testclient import TestClient

import clockify_rag.api as api_module


def _prepare_app(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "ensure_index_ready",
        lambda retries=2: ([], [], {}, None),
    )

    monkeypatch.setattr(
        api_module,
        "answer_once",
        lambda *_, **__: {
            "answer": "Authorized",
            "confidence": 0.9,
            "selected_chunks": [],
            "metadata": {},
        },
    )

    return api_module.create_app()


def test_query_requires_api_key(monkeypatch):
    monkeypatch.setattr(api_module.config, "API_AUTH_MODE", "api_key")
    monkeypatch.setattr(api_module.config, "API_ALLOWED_KEYS", frozenset({"secret"}))
    monkeypatch.setattr(api_module.config, "API_KEY_HEADER", "x-api-key")

    app = _prepare_app(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={"question": "Will auth reject me?"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API key"


def test_query_rejects_invalid_key(monkeypatch):
    monkeypatch.setattr(api_module.config, "API_AUTH_MODE", "api_key")
    monkeypatch.setattr(api_module.config, "API_ALLOWED_KEYS", frozenset({"secret"}))
    monkeypatch.setattr(api_module.config, "API_KEY_HEADER", "x-api-key")

    app = _prepare_app(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            headers={"x-api-key": "bad"},
            json={"question": "Now?"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid API key"


def test_query_accepts_valid_key(monkeypatch):
    monkeypatch.setattr(api_module.config, "API_AUTH_MODE", "api_key")
    monkeypatch.setattr(api_module.config, "API_ALLOWED_KEYS", frozenset({"secret"}))
    monkeypatch.setattr(api_module.config, "API_KEY_HEADER", "x-api-key")

    app = _prepare_app(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            headers={"x-api-key": "secret"},
            json={"question": "Auth works?"},
        )

        assert response.status_code == 200
        payload = response.json()

    assert payload["answer"] == "Authorized"
