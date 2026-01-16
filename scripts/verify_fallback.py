#!/usr/bin/env python3
"""Fallback model selection verification script.

This script validates that the fallback model selection logic works correctly
in various scenarios (primary available, fallback available, neither available,
network failure, etc.).

Usage:
    python scripts/verify_fallback.py
"""

import logging
import sys
from pathlib import Path
from unittest import mock

# Add parent directory to path so we can import clockify_rag
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s: %(message)s",
)
logger = logging.getLogger(__name__)


def test_scenario_primary_available():
    """Scenario A: Both primary and fallback models available → select primary."""
    logger.info("\n[Scenario A] Both models available (qwen2.5:32b + gpt-oss:20b)")

    from clockify_rag.config import _select_best_model

    with mock.patch("clockify_rag.config._check_remote_models") as mock_check:
        # Mock: Ollama has both models
        mock_check.return_value = ["qwen2.5:32b", "gpt-oss:20b", "llama2:7b"]

        selected = _select_best_model(
            primary="qwen2.5:32b",
            fallback="gpt-oss:20b",
            base_url="http://mock-server:11434",
            timeout=5.0,
        )

        assert selected == "qwen2.5:32b", f"Expected primary model, got {selected}"
        logger.info(f"  ✓ Correctly selected primary: {selected}")
        return True


def test_scenario_fallback_only():
    """Scenario B: Only fallback model available → select fallback (with warning)."""
    logger.info("\n[Scenario B] Only fallback available (gpt-oss:20b, primary qwen2.5:32b missing)")

    from clockify_rag.config import _select_best_model

    with mock.patch("clockify_rag.config._check_remote_models") as mock_check:
        # Mock: Ollama only has fallback
        mock_check.return_value = ["gpt-oss:20b", "llama2:7b"]

        selected = _select_best_model(
            primary="qwen2.5:32b",
            fallback="gpt-oss:20b",
            base_url="http://mock-server:11434",
            timeout=5.0,
        )

        assert selected == "gpt-oss:20b", f"Expected fallback model, got {selected}"
        logger.info(f"  ✓ Correctly selected fallback: {selected}")
        return True


def test_scenario_connection_timeout():
    """Scenario C: Connection timeout → use primary (assume VPN down, will reconnect)."""
    logger.info("\n[Scenario C] Connection timeout (simulating VPN down or Ollama slow)")

    from clockify_rag.config import _select_best_model

    with mock.patch("clockify_rag.config._check_remote_models") as mock_check:
        # Mock: Network timeout returns empty list (VPN down)
        # The real _check_remote_models catches exceptions and returns []
        mock_check.return_value = []

        selected = _select_best_model(
            primary="qwen2.5:32b",
            fallback="gpt-oss:20b",
            base_url="http://mock-server:11434",
            timeout=5.0,
        )

        assert selected == "qwen2.5:32b", f"Expected primary on timeout, got {selected}"
        logger.info(f"  ✓ Correctly used primary on timeout: {selected} (assumes VPN will reconnect)")
        return True


def test_scenario_connection_error():
    """Scenario D: Connection error → use primary (server offline)."""
    logger.info("\n[Scenario D] Connection error (simulating Ollama offline or firewall block)")

    from clockify_rag.config import _select_best_model

    with mock.patch("clockify_rag.config._check_remote_models") as mock_check:
        # Mock: Connection error returns empty list (Ollama offline)
        # The real _check_remote_models catches exceptions and returns []
        mock_check.return_value = []

        selected = _select_best_model(
            primary="qwen2.5:32b",
            fallback="gpt-oss:20b",
            base_url="http://mock-server:11434",
            timeout=5.0,
        )

        assert selected == "qwen2.5:32b", f"Expected primary on error, got {selected}"
        logger.info(f"  ✓ Correctly used primary on connection error: {selected}")
        return True


def test_scenario_neither_available():
    """Scenario E: Neither model available → use primary anyway (best effort)."""
    logger.info("\n[Scenario E] Neither model available (both missing from Ollama)")

    from clockify_rag.config import _select_best_model

    with mock.patch("clockify_rag.config._check_remote_models") as mock_check:
        # Mock: Server online but neither model installed
        mock_check.return_value = ["llama2:7b", "mistral:7b"]

        selected = _select_best_model(
            primary="qwen2.5:32b",
            fallback="gpt-oss:20b",
            base_url="http://mock-server:11434",
            timeout=5.0,
        )

        assert selected == "qwen2.5:32b", f"Expected primary anyway, got {selected}"
        logger.info(f"  ✓ Correctly used primary as last resort: {selected}")
        return True


def test_scenario_empty_list():
    """Scenario F: Empty model list (Ollama online but no models) → use primary."""
    logger.info("\n[Scenario F] Ollama online but no models installed (empty list)")

    from clockify_rag.config import _select_best_model

    with mock.patch("clockify_rag.config._check_remote_models") as mock_check:
        # Mock: Server online but no models
        mock_check.return_value = []

        selected = _select_best_model(
            primary="qwen2.5:32b",
            fallback="gpt-oss:20b",
            base_url="http://mock-server:11434",
            timeout=5.0,
        )

        assert selected == "qwen2.5:32b", f"Expected primary on empty list, got {selected}"
        logger.info(f"  ✓ Correctly used primary when no models available: {selected}")
        return True


def main():
    """Run all fallback scenarios."""
    logger.info("=" * 70)
    logger.info("Fallback Model Selection - Verification Script")
    logger.info("=" * 70)

    scenarios = [
        ("Scenario A: Primary available", test_scenario_primary_available),
        ("Scenario B: Fallback only", test_scenario_fallback_only),
        ("Scenario C: Timeout (VPN down)", test_scenario_connection_timeout),
        ("Scenario D: Connection error (offline)", test_scenario_connection_error),
        ("Scenario E: Neither available", test_scenario_neither_available),
        ("Scenario F: Empty model list", test_scenario_empty_list),
    ]

    results = []
    for name, test_fn in scenarios:
        try:
            success = test_fn()
            results.append((name, success, None))
        except AssertionError as e:
            logger.error(f"  ✗ FAILED: {e}")
            results.append((name, False, str(e)))
        except Exception as e:
            logger.error(f"  ✗ ERROR: {e}")
            results.append((name, False, str(e)))

    # Summary
    logger.info("\n" + "=" * 70)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    logger.info(f"Results: {passed}/{total} scenarios passed")

    if passed == total:
        logger.info("✓ All fallback scenarios passed! Model selection is robust.")
        return 0
    else:
        logger.warning("✗ Some scenarios failed. See above for details.")
        for name, success, error in results:
            if not success:
                logger.warning(f"  {name}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
