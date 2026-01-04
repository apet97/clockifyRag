import json

import pytest
from typer.testing import CliRunner

import clockify_rag.cli_modern as cli_modern


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_query_command_surfaces_metadata(monkeypatch, cli_runner):
    """Ensure Typer query command consumes new answer_once schema."""

    monkeypatch.setattr(cli_modern, "ensure_index_ready", lambda retries=2: ([], [], {}, None))

    result_payload = {
        "answer": "Mocked answer",
        "confidence": 0.77,
        "selected_chunks": [10, 20],
        "selected_chunk_ids": ["doc-10", "doc-20"],
        "metadata": {"used_tokens": 42},
        "intent": "feature_howto",
        "user_role_inferred": "admin",
        "security_sensitivity": "low",
        "needs_human_escalation": False,
        "sources_used": ["https://example.com/doc-10"],
    }

    monkeypatch.setattr(cli_modern, "answer_once", lambda *_, **__: result_payload)

    response = cli_runner.invoke(
        cli_modern.app,
        [
            "query",
            "How do I track time?",
            "--json",
        ],
    )

    assert response.exit_code == 0
    payload = json.loads("{" + response.stdout.split("{", 1)[1])

    assert payload["answer"] == result_payload["answer"]
    assert payload["sources"] == result_payload["selected_chunk_ids"]
    assert payload["metadata"]["used_tokens"] == result_payload["metadata"]["used_tokens"]


def test_query_default_output_includes_sources(monkeypatch, cli_runner):
    """Default (non-JSON) output should print answer and URLs."""

    chunks = [
        {"id": "doc-1", "url": "https://example.com/a"},
        {"id": "doc-2", "url": "https://example.com/b"},
    ]
    monkeypatch.setattr(cli_modern, "ensure_index_ready", lambda retries=2: (chunks, [], {}, None))

    result_payload = {
        "answer": "Mocked answer",
        "selected_chunk_ids": ["doc-2", "doc-1"],
        "metadata": {"sources_used": ["doc-2", "doc-1"]},
        "intent": "troubleshooting",
        "user_role_inferred": "admin",
        "security_sensitivity": "high",
        "needs_human_escalation": True,
    }

    monkeypatch.setattr(cli_modern, "answer_once", lambda *_, **__: result_payload)

    response = cli_runner.invoke(cli_modern.app, ["query", "How do I track time?"])

    assert response.exit_code == 0
    out = response.stdout
    assert "Intent: troubleshooting" in out
    assert "Role inferred: admin" in out
    assert "Security: high" in out
    assert "Needs escalation: True" in out
    assert "Answer:" in out
    assert "Mocked answer" in out
    assert "Sources:" in out
    # URLs should be deduped and sorted
    assert "- https://example.com/a" in out
    assert "- https://example.com/b" in out


def test_query_default_output_no_sources(monkeypatch, cli_runner):
    """If no URLs can be resolved, show Sources: (none)."""

    chunks = [{"id": "doc-1"}]
    monkeypatch.setattr(cli_modern, "ensure_index_ready", lambda retries=2: (chunks, [], {}, None))

    result_payload = {
        "answer": "No source answer",
        "selected_chunk_ids": ["doc-1"],
        "metadata": {"sources_used": ["doc-1"]},
        "intent": "other",
        "user_role_inferred": "unknown",
        "security_sensitivity": "medium",
        "needs_human_escalation": False,
    }

    monkeypatch.setattr(cli_modern, "answer_once", lambda *_, **__: result_payload)

    response = cli_runner.invoke(cli_modern.app, ["query", "How do I track time?"])

    assert response.exit_code == 0
    out = response.stdout
    assert "Answer:" in out
    assert "No source answer" in out
    assert "Sources:" in out
    assert "(none)" in out
