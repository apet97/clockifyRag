"""Test import fallback behavior for langchain-ollama vs langchain-community.

These tests verify that:
- In PROD mode, missing langchain-ollama raises ImportError
- In DEV mode, fallback to langchain-community is allowed with warning
- Clear error messages guide users to install the right package
"""

import os
import sys
from unittest.mock import patch
import pytest


def test_prod_mode_fails_without_langchain_ollama():
    """In production, missing langchain-ollama should fail fast."""
    # Set production environment
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        # Mock langchain_ollama as missing
        with patch.dict(sys.modules, {"langchain_ollama": None}):
            # This import should fail in prod mode
            with pytest.raises(ImportError) as exc_info:
                # Force reimport to trigger the import logic
                import importlib
                import clockify_rag.llm_client as llm_module

                importlib.reload(llm_module)

            # Verify error message is helpful
            assert "langchain-ollama is required in production" in str(exc_info.value)
            assert "pip install langchain-ollama" in str(exc_info.value)


def test_dev_mode_allows_fallback():
    """In dev mode, fallback to langchain-community is allowed with warning."""
    # This test assumes langchain-ollama is actually installed in dev environment
    # If it's missing, the fallback should work without raising
    with patch.dict(os.environ, {"ENVIRONMENT": "dev"}):
        try:
            import importlib
            import clockify_rag.llm_client as llm_module

            # Reload to ensure environment is respected
            importlib.reload(llm_module)
            # Should not raise, either using langchain-ollama or falling back
        except ImportError as e:
            # Only acceptable if neither package is available
            assert "Neither langchain-ollama nor langchain-community" in str(e)


def test_ci_mode_is_treated_as_prod():
    """CI environment should be treated as production."""
    with patch.dict(os.environ, {"ENVIRONMENT": "ci"}):
        with patch.dict(sys.modules, {"langchain_ollama": None}):
            with pytest.raises(ImportError) as exc_info:
                import importlib
                import clockify_rag.llm_client as llm_module

                importlib.reload(llm_module)

            assert "langchain-ollama is required in production" in str(exc_info.value)


def test_app_env_fallback():
    """APP_ENV should work as alternative to ENVIRONMENT."""
    with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
        # Clear ENVIRONMENT to test APP_ENV fallback
        if "ENVIRONMENT" in os.environ:
            del os.environ["ENVIRONMENT"]

        with patch.dict(sys.modules, {"langchain_ollama": None}):
            with pytest.raises(ImportError) as exc_info:
                import importlib
                import clockify_rag.llm_client as llm_module

                importlib.reload(llm_module)

            assert "langchain-ollama is required in production" in str(exc_info.value)
