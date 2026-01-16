"""Tests for the shared LLM client abstraction."""

import pytest
import requests

from clockify_rag.api_client import (
    MockLLMClient,
    OllamaAPIClient,
    get_llm_client,
    set_llm_client,
    reset_llm_client,
)
from clockify_rag import config
from clockify_rag.exceptions import EmbeddingError, LLMUnavailableError, LLMBadResponseError


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200, json_exc: Exception | None = None):
        self._payload = payload
        self.status_code = status_code
        self._json_exc = json_exc

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)

    def json(self):
        if self._json_exc:
            raise self._json_exc
        return self._payload


class DummySession:
    def __init__(self, responses: list[DummyResponse], exc: Exception | None = None):
        self._responses = responses
        self._exc = exc
        self.post_calls = []
        self.trust_env = False

    def post(self, url, json=None, timeout=None, allow_redirects=None):
        if self._exc:
            raise self._exc
        self.post_calls.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
                "allow_redirects": allow_redirects,
            }
        )
        if not self._responses:
            raise AssertionError("No dummy responses left for post()")
        return self._responses.pop(0)


def test_mock_llm_client_embeddings_are_deterministic():
    client = MockLLMClient(embed_dim=16)
    vec1 = client.create_embedding("deterministic-text")
    vec2 = client.create_embedding("deterministic-text")
    assert vec1 == vec2
    assert len(vec1) == 16


def test_mock_llm_client_chat_completion_echoes_prompt():
    client = MockLLMClient()
    client.register_chat_response("ping?", "pong!")
    response = client.chat_completion(
        messages=[
            {"role": "user", "content": "ping?"},
        ],
        model=config.RAG_CHAT_MODEL,
    )
    assert response["message"]["content"] == "pong!"


def test_set_llm_client_overrides_global_instance():
    custom = MockLLMClient()
    set_llm_client(custom)
    try:
        assert get_llm_client() is custom
    finally:
        reset_llm_client()


def test_ollama_api_client_validates_chat_response(monkeypatch):
    dummy_session = DummySession(
        [
            DummyResponse(
                {
                    "model": config.RAG_CHAT_MODEL,
                    "message": {"role": "assistant", "content": "pong"},
                    "done": True,
                }
            )
        ]
    )
    monkeypatch.setattr("clockify_rag.api_client.get_session", lambda **kwargs: dummy_session)
    client = OllamaAPIClient(base_url="http://fake-host:11434")

    result = client.chat_completion(messages=[{"role": "user", "content": "ping"}])

    assert result["message"]["content"] == "pong"
    assert dummy_session.post_calls, "Expected chat completion to call POST"


def test_ollama_api_client_errors_on_missing_message(monkeypatch):
    dummy_session = DummySession([DummyResponse({"model": config.RAG_CHAT_MODEL, "not_message": {}})])
    monkeypatch.setattr("clockify_rag.api_client.get_session", lambda **kwargs: dummy_session)
    client = OllamaAPIClient(base_url="http://fake-host:11434")

    with pytest.raises(LLMBadResponseError):
        client.chat_completion(messages=[{"role": "user", "content": "ping"}])


def test_ollama_api_client_embedding_validates_numeric_output(monkeypatch):
    dummy_session = DummySession([DummyResponse({"embedding": ["not", "numbers"]})])
    monkeypatch.setattr("clockify_rag.api_client.get_session", lambda **kwargs: dummy_session)
    monkeypatch.setattr("clockify_rag.api_client.EMB_BACKEND", "ollama")
    client = OllamaAPIClient(base_url="http://fake-host:11434")

    with pytest.raises(EmbeddingError):
        client.create_embedding("text needing embedding")


def test_ollama_api_client_timeout_maps_to_unavailable(monkeypatch):
    dummy_session = DummySession([], exc=requests.exceptions.Timeout("boom"))
    monkeypatch.setattr("clockify_rag.api_client.get_session", lambda **kwargs: dummy_session)
    client = OllamaAPIClient(base_url="http://fake-host:11434")

    with pytest.raises(LLMUnavailableError):
        client.chat_completion(messages=[{"role": "user", "content": "ping"}])


def test_ollama_api_client_invalid_json_raises_bad_response(monkeypatch):
    dummy_session = DummySession([DummyResponse({}, json_exc=ValueError("no json"))])
    monkeypatch.setattr("clockify_rag.api_client.get_session", lambda **kwargs: dummy_session)
    client = OllamaAPIClient(base_url="http://fake-host:11434")

    with pytest.raises(LLMBadResponseError):
        client.chat_completion(messages=[{"role": "user", "content": "ping"}])


def test_embedding_dimension_mismatch_is_rejected(monkeypatch):
    dummy_session = DummySession([DummyResponse({"embedding": [0.1, 0.2, 0.3]})])
    monkeypatch.setattr("clockify_rag.api_client.get_session", lambda **kwargs: dummy_session)
    monkeypatch.setattr("clockify_rag.api_client.EMB_BACKEND", "ollama")
    monkeypatch.setattr("clockify_rag.api_client.EMB_DIM_OLLAMA", 2)
    client = OllamaAPIClient(base_url="http://fake-host:11434")

    with pytest.raises(EmbeddingError):
        client.create_embedding("text needing embedding")


def test_embedding_success_returns_expected_vector(monkeypatch):
    dummy_session = DummySession([DummyResponse({"embedding": [1, 2]})])
    monkeypatch.setattr("clockify_rag.api_client.get_session", lambda **kwargs: dummy_session)
    monkeypatch.setattr("clockify_rag.api_client.EMB_BACKEND", "ollama")
    monkeypatch.setattr("clockify_rag.api_client.EMB_DIM_OLLAMA", 2)
    client = OllamaAPIClient(base_url="http://fake-host:11434")

    vec = client.create_embedding("text needing embedding")
    assert vec == [1.0, 2.0]
