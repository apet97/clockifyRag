"""Shared pytest fixtures for RAG system tests."""

import json
import os
import platform
import sys
import tempfile
import importlib.util

# Force tests to use local embeddings to avoid remote Ollama dependency
os.environ.setdefault("EMB_BACKEND", "local")

import httpx
import numpy as np
import pytest

# Import from refactored package modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.indexing import build_bm25
from clockify_rag.api_client import MockLLMClient, set_llm_client


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _vpn_reachable() -> bool:
    """Check if the Ollama server is reachable (connectivity check).

    Returns:
        True if Ollama endpoint responds, False otherwise (e.g., VPN down, CI environment)
    """
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("RAG_OLLAMA_URL") or "http://127.0.0.1:11434"
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=1.0)
        return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException, Exception):
        return False


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if VPN/Ollama is unreachable.

    Tests marked with @pytest.mark.integration will be skipped in CI or when VPN is down.
    """
    if _vpn_reachable():
        # VPN is up, run all tests including integration
        return

    skip_integration = pytest.mark.skip(reason="VPN unreachable, skipping integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_configure(config):
    """Validate test environment has required dependencies.

    This hook runs before any tests and ensures all required dependencies
    are installed. It exits with an error if critical dependencies are missing,
    and warns about optional dependencies.
    """
    missing = []

    # Check required dependencies
    if not _has_module("numpy"):
        missing.append("numpy")

    if not _has_module("requests"):
        missing.append("requests")

    if missing:
        pytest.exit(
            f"\nERROR: Missing required test dependencies: {', '.join(missing)}\n"
            f"Install with: pip install -e '.[dev]'\n",
            returncode=1,
        )

    # Check optional dependencies (just warn)
    optional_missing = []
    if not _has_module("faiss"):
        optional_missing.append("faiss-cpu")

    if not _has_module("torch"):
        optional_missing.append("torch")

    if optional_missing and config.option.verbose >= 0:
        faiss_note = ""
        if "faiss-cpu" in optional_missing:
            if platform.system() == "Darwin" and platform.machine().lower() == "arm64":
                faiss_note = (
                    "FAISS wheels are not published for macOS arm64; install via conda:\n"
                    "  conda install -c conda-forge faiss-cpu\n"
                )
                optional_missing.remove("faiss-cpu")
            else:
                faiss_note = "Install faiss-cpu via pip or conda to enable ANN tests.\n"

        if optional_missing:
            print(f"\nWARNING: Optional dependencies not installed: {', '.join(optional_missing)}")
            print("Some tests may be skipped. Install with: pip install -e '.[dev]'\n")
        if faiss_note:
            print(f"\nINFO: {faiss_note}")


@pytest.fixture(autouse=True)
def use_mock_llm_client():
    """Route all LLM calls through the deterministic mock client."""
    client = MockLLMClient()
    set_llm_client(client)
    yield
    set_llm_client(None)


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "id": "1",
            "title": "Time Tracking",
            "section": "Basics",
            "text": "Track time by clicking the timer button in the top right corner",
            "url": "https://clockify.me/help/time-tracking",
        },
        {
            "id": "2",
            "title": "Pricing",
            "section": "Plans",
            "text": "Free plan includes unlimited users and unlimited time tracking",
            "url": "https://clockify.me/pricing",
        },
        {
            "id": "3",
            "title": "Time Tracking",
            "section": "Advanced",
            "text": "Manual time entries can be added by clicking the plus button",
            "url": "https://clockify.me/help/time-tracking-advanced",
        },
        {
            "id": "4",
            "title": "Reports",
            "section": "Overview",
            "text": "Generate detailed reports of your time tracking data",
            "url": "https://clockify.me/help/reports",
        },
        {
            "id": "5",
            "title": "Integrations",
            "section": "Setup",
            "text": "Integrate Clockify with your favorite tools like Jira and Slack",
            "url": "https://clockify.me/help/integrations",
        },
    ]


@pytest.fixture
def sample_embeddings(sample_chunks):
    """Sample embeddings for testing (384-dim for local, 768-dim for Ollama).

    Creates simple embeddings with some semantic similarity between related chunks.
    """
    from clockify_rag.config import EMB_DIM

    n = len(sample_chunks)

    # Create base embeddings
    vecs = np.random.randn(n, EMB_DIM).astype("float32")

    # Add some similarity between time tracking chunks (0 and 2)
    vecs[2] = vecs[0] + np.random.randn(EMB_DIM).astype("float32") * 0.1

    # Normalize all vectors
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def sample_bm25(sample_chunks):
    """Sample BM25 index for testing."""
    return build_bm25(sample_chunks)


@pytest.fixture
def temp_index_dir(sample_chunks, sample_embeddings, sample_bm25):
    """Temporary directory with full index artifacts.

    Creates a temporary directory with:
    - chunks.jsonl
    - vecs_n.npy
    - bm25.json
    - index.meta.json
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write chunks
        chunks_path = os.path.join(tmpdir, "chunks.jsonl")
        with open(chunks_path, "w") as f:
            for c in sample_chunks:
                f.write(json.dumps(c) + "\n")

        # Write embeddings
        vecs_path = os.path.join(tmpdir, "vecs_n.npy")
        np.save(vecs_path, sample_embeddings)

        # Write BM25
        bm25_path = os.path.join(tmpdir, "bm25.json")
        with open(bm25_path, "w") as f:
            json.dump(sample_bm25, f)

        # Write metadata
        meta = {
            "chunks": len(sample_chunks),
            "emb_rows": len(sample_embeddings),
            "bm25_docs": len(sample_bm25["doc_lens"]),
            "kb_sha256": "test_hash",
            "built_at": "2025-11-06T00:00:00Z",
        }
        meta_path = os.path.join(tmpdir, "index.meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        yield tmpdir


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "How do I track time?"


@pytest.fixture
def sample_query_embedding():
    """Sample query embedding for testing."""
    from clockify_rag.config import EMB_DIM

    qv = np.random.randn(EMB_DIM).astype("float32")
    qv = qv / np.linalg.norm(qv)
    return qv


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response for testing."""
    return {"embedding": np.random.randn(768).tolist(), "model": "nomic-embed-text"}  # 768-dim for Ollama


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "message": {"content": "Track time by clicking the timer button in the top right corner."},
        "model": "qwen2.5:32b",
        "done": True,
    }
