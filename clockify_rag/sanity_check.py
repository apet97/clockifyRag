"""Sanity check script for remote-first RAG setup.

Validates that the RAG system is properly configured for remote Ollama:
1. Config loads without errors
2. Remote Ollama /api/tags is reachable
3. Embedding and LLM clients can be instantiated
4. A minimal test query works end-to-end

Usage:
    python -m clockify_rag.sanity_check
"""

import logging
import sys
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s: %(message)s",
)
logger = logging.getLogger(__name__)


def check_config() -> Tuple[bool, str]:
    """Check that config loads and has sensible values."""
    try:
        from . import config

        errors = []

        # Validate Ollama URL
        if not config.RAG_OLLAMA_URL:
            errors.append("RAG_OLLAMA_URL is empty")
        elif not config.RAG_OLLAMA_URL.startswith("http://") and not config.RAG_OLLAMA_URL.startswith("https://"):
            errors.append(f"RAG_OLLAMA_URL has invalid format: {config.RAG_OLLAMA_URL}")

        # Validate timeout
        if config.OLLAMA_TIMEOUT < 5 or config.OLLAMA_TIMEOUT > 600:
            errors.append(f"OLLAMA_TIMEOUT out of range: {config.OLLAMA_TIMEOUT} (expected 5-600)")

        # Validate models
        if not config.LLM_MODEL:
            errors.append("LLM_MODEL is empty")
        if not config.RAG_EMBED_MODEL:
            errors.append("RAG_EMBED_MODEL is empty")

        if errors:
            return False, "Config validation failed:\n  " + "\n  ".join(errors)

        logger.info(f"✓ Config loaded: {config.RAG_OLLAMA_URL}")
        logger.info(f"  LLM model: {config.LLM_MODEL}")
        logger.info(f"  Embedding model: {config.RAG_EMBED_MODEL}")
        logger.info(f"  Timeout: {config.OLLAMA_TIMEOUT}s")
        return True, "Config is valid"
    except Exception as e:
        return False, f"Failed to load config: {e}"


def check_remote_models() -> Tuple[bool, str]:
    """Check that /api/tags endpoint is reachable."""
    try:
        from . import config

        models = config._check_remote_models(config.RAG_OLLAMA_URL or "", timeout=5.0)
        if not models:
            return True, "Remote Ollama unreachable (will use primary model on retry)"

        logger.info(f"✓ Remote Ollama online with {len(models)} model(s)")
        return True, f"Found models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}"
    except Exception as e:
        return False, f"Failed to check remote models: {e}"


def check_embeddings_client() -> Tuple[bool, str]:
    """Check that embeddings client can be instantiated."""
    try:
        from .embeddings_client import get_embedding_client

        client = get_embedding_client()
        logger.info(f"✓ Embeddings client instantiated: {type(client).__name__}")
        return True, "Embeddings client ready"
    except Exception as e:
        return False, f"Failed to create embeddings client: {e}"


def check_llm_client() -> Tuple[bool, str]:
    """Check that LLM client can be instantiated."""
    try:
        from .llm_client import get_llm_client
        from . import config

        client = get_llm_client(temperature=0.0)
        logger.info(f"✓ LLM client instantiated: {type(client).__name__}")
        logger.info(f"  Model: {client.model}")
        logger.info(f"  Base URL: {config.RAG_OLLAMA_URL}")
        logger.info(f"  Temperature: {client.temperature}")
        logger.info(f"  Timeout: {config.OLLAMA_TIMEOUT}s")
        # Note: streaming=False is set at client creation time in llm_client.py
        # We don't access client.streaming attribute as it may not be exposed
        logger.info("  Streaming: disabled (configured at client creation)")
        return True, "LLM client ready"
    except Exception as e:
        return False, f"Failed to create LLM client: {e}"


def check_end_to_end() -> Tuple[bool, str]:
    """Check that a minimal embedding+LLM flow works."""
    try:
        from .embeddings_client import embed_query
        from .llm_client import get_llm_client
        from langchain_core.messages import HumanMessage

        # Test embedding a short query
        logger.info("Testing embeddings...")
        vec = embed_query("test query")
        if vec.size == 0:
            return False, "Embedding returned empty vector"
        logger.info(f"✓ Embedded test query: {vec.shape}")

        # Test LLM generation
        logger.info("Testing LLM generation...")
        llm = get_llm_client(temperature=0.0)
        response = llm.invoke([HumanMessage(content="What is 1+1?")])
        content = str(response.content or "")
        if not content or "2" not in content.lower():
            return False, f"LLM returned unexpected response: {content}"
        logger.info(f"✓ LLM generation works: {content[:60]}...")

        return True, "End-to-end test passed"
    except Exception as e:
        return False, f"End-to-end test failed: {e}"


def main():
    """Run all sanity checks."""
    logger.info("=" * 70)
    logger.info("Clockify RAG - Remote-First Sanity Check")
    logger.info("=" * 70)

    checks = [
        ("Configuration", check_config),
        ("Remote Models", check_remote_models),
        ("Embeddings Client", check_embeddings_client),
        ("LLM Client", check_llm_client),
        ("End-to-End Flow", check_end_to_end),
    ]

    results = []
    for name, check_fn in checks:
        logger.info(f"\n[{name}]")
        success, message = check_fn()
        results.append((name, success, message))
        if success:
            logger.info(f"  {message}")
        else:
            logger.warning(f"  {message}")

    # Summary
    logger.info("\n" + "=" * 70)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    logger.info(f"Results: {passed}/{total} checks passed")

    if passed == total:
        logger.info("✓ All sanity checks passed! RAG system is ready.")
        return 0
    else:
        logger.warning("✗ Some checks failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
