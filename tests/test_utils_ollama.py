import pytest
import requests

from clockify_rag.utils import check_ollama_connectivity


def test_check_ollama_connectivity_success(monkeypatch):
    def fake_get(url, timeout, **kwargs):
        assert url == "http://localhost:11434/api/tags"
        assert timeout == pytest.approx(1.5)
        assert kwargs.get("allow_redirects") is False

        class Response:
            def raise_for_status(self):
                return None

        return Response()

    monkeypatch.setattr("clockify_rag.utils.requests.get", fake_get)

    normalized = check_ollama_connectivity("localhost:11434", timeout=1.5)
    assert normalized == "http://localhost:11434"


def test_check_ollama_connectivity_failure(monkeypatch):
    def fake_get(url, timeout, **kwargs):
        raise requests.exceptions.ConnectTimeout("timed out")

    monkeypatch.setattr("clockify_rag.utils.requests.get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        check_ollama_connectivity("http://localhost:11434", timeout=2)

    assert "Failed to connect to Ollama" in str(excinfo.value)
    assert "timed out" in str(excinfo.value)
