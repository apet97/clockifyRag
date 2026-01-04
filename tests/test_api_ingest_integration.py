import asyncio

import pytest
import httpx
from asgi_lifespan import LifespanManager

import clockify_rag.api as api_module
from clockify_rag.utils import ALLOWED_CORPUS_FILENAME


class EnsureStub:
    """Mutable stub to control ensure_index_ready responses across calls."""

    def __init__(self):
        self.calls = []
        self.return_values = []
        self.default_return = None

    def __call__(self, retries=0):
        self.calls.append(retries)
        if self.return_values:
            return self.return_values.pop(0)
        return self.default_return


@pytest.mark.asyncio
async def test_ingest_then_query_succeeds(tmp_path, monkeypatch):
    """Trigger ingest on a temporary KB and verify queries succeed afterwards."""

    kb_path = tmp_path / ALLOWED_CORPUS_FILENAME
    kb_path.write_text("# Title\n\nSome helpful docs.", encoding="utf-8")

    ensure_stub = EnsureStub()
    ensure_stub.return_values.append(None)  # Startup call should report not ready
    ensure_stub.default_return = ([{"id": 1, "text": "chunk"}], ["vec"], {"idf": {}}, "hnsw")
    monkeypatch.setattr(api_module, "ensure_index_ready", ensure_stub)

    build_calls = {}

    def fake_build(path, retries=0):
        build_calls["path"] = path
        build_calls["retries"] = retries

    monkeypatch.setattr(api_module, "build", fake_build)

    recorded_answer_inputs = {}

    def fake_answer(question, chunks, vecs_n, bm, **kwargs):
        recorded_answer_inputs["chunks"] = chunks
        recorded_answer_inputs["vecs_n"] = vecs_n
        recorded_answer_inputs["bm"] = bm
        return {"answer": "ready", "selected_chunks": [1], "metadata": {}}

    monkeypatch.setattr(api_module, "answer_once", fake_answer)

    app = api_module.create_app()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            ingest_response = await client.post("/v1/ingest", json={"input_file": str(kb_path)})

            assert ingest_response.status_code == 200
            assert build_calls == {"path": str(kb_path), "retries": 2}
            assert ensure_stub.calls.count(2) >= 2  # Startup + ingest

            # Wait for the background task to finalize state assignment
            for _ in range(20):
                if app.state.index_ready:
                    break
                await asyncio.sleep(0.01)

            assert app.state.index_ready is True

            query_response = await client.post("/v1/query", json={"question": "How do I use this?"})

            assert query_response.status_code == 200
            assert query_response.json()["answer"] == "ready"
            assert recorded_answer_inputs["chunks"] == ensure_stub.default_return[0]
            assert recorded_answer_inputs["vecs_n"] == ensure_stub.default_return[1]
            assert recorded_answer_inputs["bm"] == ensure_stub.default_return[2]


@pytest.mark.asyncio
async def test_ingest_failure_preserves_cached_state(tmp_path, monkeypatch):
    """Ensure a failed ingest preserves the previously loaded index artifacts."""

    kb_path = tmp_path / ALLOWED_CORPUS_FILENAME
    kb_path.write_text("# Title\n\nSome helpful docs.", encoding="utf-8")

    ensure_stub = EnsureStub()
    ensure_stub.return_values.append(None)
    ensure_stub.default_return = None
    monkeypatch.setattr(api_module, "ensure_index_ready", ensure_stub)

    build_calls = {"count": 0}

    def failing_build(path, retries=0):
        build_calls["count"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module, "build", failing_build)

    app = api_module.create_app()

    async with LifespanManager(app):
        app.state.chunks = ["stale"]
        app.state.vecs_n = ["stale"]
        app.state.bm = {"stale": True}
        app.state.hnsw = "stale"
        app.state.index_ready = True

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            ingest_response = await client.post("/v1/ingest", json={"input_file": str(kb_path)})

            assert ingest_response.status_code == 200
            assert build_calls["count"] == 1

            # Wait for the background task to restore cached state
            for _ in range(20):
                if (
                    app.state.index_ready is True
                    and app.state.chunks == ["stale"]
                    and app.state.vecs_n == ["stale"]
                    and app.state.bm == {"stale": True}
                    and app.state.hnsw == "stale"
                ):
                    break
                await asyncio.sleep(0.01)

            assert app.state.index_ready is True
            assert app.state.chunks == ["stale"]
            assert app.state.vecs_n == ["stale"]
            assert app.state.bm == {"stale": True}
            assert app.state.hnsw == "stale"
