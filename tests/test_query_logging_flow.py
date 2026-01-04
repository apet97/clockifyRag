from fastapi.testclient import TestClient

import clockify_rag.api as api_module
from clockify_rag import config


def test_api_privacy_mode_disables_logging(tmp_path, monkeypatch):
    log_path = tmp_path / "queries.jsonl"
    log_path.write_text("seed")
    monkeypatch.setattr(config, "QUERY_LOG_FILE", str(log_path))
    monkeypatch.setattr(api_module.config, "API_PRIVACY_MODE", True, raising=False)
    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: ([{"id": "chunk-7"}], [], {}, None))
    result_payload = {
        "answer": "Mocked",
        "selected_chunks": [0],
        "metadata": {},
        "timing": {"total_ms": 9},
    }
    monkeypatch.setattr(api_module, "answer_once", lambda *_, **__: result_payload)

    app = api_module.create_app()
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"question": "Hello?"})
        assert response.status_code == 200

    assert log_path.read_text() == "seed"
