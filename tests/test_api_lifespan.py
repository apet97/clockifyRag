from fastapi.testclient import TestClient

import clockify_rag.api as api_module
from clockify_rag.exceptions import IndexLoadError


def test_create_app_lifespan_runs(monkeypatch):
    """create_app should initialize without raising during FastAPI lifespan."""

    def fail_ensure(*_args, **_kwargs):
        raise IndexLoadError("unavailable")

    monkeypatch.setattr(api_module, "ensure_index_ready", fail_ensure)

    app = api_module.create_app()

    # entering the TestClient context runs the app lifespan (startup/shutdown)
    with TestClient(app):
        pass
