"""Environment checks for Python version and package dependencies.

This module provides centralized functions for verifying:
- Python version compatibility (3.11-3.13 supported, 3.14+ blocked)
- Required and optional package availability

Used by scripts/verify_env.py and tests/test_verify_env.py.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import List, Tuple

# Python version policy
SUPPORTED_MIN = (3, 11)
SUPPORTED_MAX_EXCLUSIVE = (3, 14)  # i.e. up to 3.13
RECOMMENDED = [(3, 11), (3, 12)]

# Test mode hook (set ENV_CHECKS_TEST_MODE environment variable in tests)
TEST_MODE = os.getenv("ENV_CHECKS_TEST_MODE")

# Required packages (must be installed for core RAG functionality)
# The system cannot start without these - all are needed for production deployment.
REQUIRED_PACKAGES = [
    "numpy",  # Core numerical operations (vectors, embeddings, similarity calculations)
    "httpx",  # HTTP client for Ollama API communication (remote-first architecture)
    "langchain_ollama",  # LangChain Ollama integration (required for LLM and embedding calls)
    "rank_bm25",  # BM25 retrieval (required for hybrid search - keyword component)
    "langchain",  # LangChain core (required for document processing and chains)
    "tiktoken",  # Tokenization (required for context budget calculations and chunking)
    "fastapi",  # API server (required - all deployments use the REST API)
    "typer",  # CLI commands (required - management and diagnostics require CLI)
    "pydantic",  # Config validation and API models (required by FastAPI and config system)
]

# Optional packages (enhance functionality but not strictly required)
# The core RAG service works without these in "remote-first" mode.
OPTIONAL_PACKAGES = [
    (
        "faiss",
        "FAISS (vector indexing) - install via conda on M1: conda install -c conda-forge faiss-cpu. Provides faster vector search; falls back to linear search if missing.",
    ),
    (
        "torch",
        "PyTorch (local embeddings) - install with: pip install -e .[embeddings]. Required for local embedding generation; not needed if using remote Ollama.",
    ),
    (
        "sentence_transformers",
        "SentenceTransformers (local embeddings) - install with: pip install -e .[embeddings]. Required for local embedding generation; not needed if using remote Ollama.",
    ),
]


def _try_import(module_name: str) -> bool:
    """Try to import a module and return success status.

    Args:
        module_name: Name of the module to import

    Returns:
        True if import succeeded, False if ImportError raised

    Note:
        Respects ENV_CHECKS_TEST_MODE for testing:
        - "force_missing_optional": Forces optional packages to fail
        - "force_missing_required": Forces required packages to fail
    """
    # Test hook for CLI tests
    if TEST_MODE == "force_missing_optional":
        # Force optional packages to appear missing
        optional_names = [pkg if isinstance(pkg, str) else pkg[0] for pkg in OPTIONAL_PACKAGES]
        if module_name in optional_names:
            return False
    elif TEST_MODE == "force_missing_required":
        # Force required packages to appear missing
        if module_name in REQUIRED_PACKAGES:
            return False

    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def check_python_version() -> Tuple[bool, List[str]]:
    """Check Python version is within supported range.

    Returns:
        Tuple of (success: bool, messages: List[str])
        - success: True if version is 3.11-3.13, False otherwise
        - messages: Human-readable status messages
    """
    version = sys.version_info
    messages = []

    if version.major == 3 and SUPPORTED_MIN[1] <= version.minor < SUPPORTED_MAX_EXCLUSIVE[1]:
        # Within supported range
        if (version.major, version.minor) in RECOMMENDED:
            status = "supported"
        else:
            status = "experimental"
        messages.append(f"Python {version.major}.{version.minor}.{version.micro} ({status})")
        return True, messages

    elif version.major == 3 and version.minor >= SUPPORTED_MAX_EXCLUSIVE[1]:
        # Too new (3.14+)
        messages.append(
            f"Python {version.major}.{version.minor} not supported "
            f"(use {SUPPORTED_MIN[0]}.{SUPPORTED_MIN[1]}-"
            f"{SUPPORTED_MAX_EXCLUSIVE[0]}.{SUPPORTED_MAX_EXCLUSIVE[1] - 1})"
        )
        return False, messages

    else:
        # Too old or wrong major version
        messages.append(
            f"Python {version.major}.{version.minor} "
            f"(need {SUPPORTED_MIN[0]}.{SUPPORTED_MIN[1]}-"
            f"{SUPPORTED_MAX_EXCLUSIVE[0]}.{SUPPORTED_MAX_EXCLUSIVE[1] - 1})"
        )
        return False, messages


def check_packages() -> Tuple[bool, List[str], List[str], List[str]]:
    """Check required and optional Python packages.

    Returns:
        Tuple of (ok: bool, messages: List[str], missing_required: List[str], missing_optional: List[str])
        - ok: True if all required packages present, False otherwise
        - messages: Human-readable status messages for all packages
        - missing_required: List of missing required package names
        - missing_optional: List of missing optional package names
    """
    missing_required = []
    missing_optional = []

    # Check required packages
    for pkg in REQUIRED_PACKAGES:
        if not _try_import(pkg):
            missing_required.append(pkg)

    # Check optional packages (extract package name from tuple)
    for pkg_info in OPTIONAL_PACKAGES:
        pkg_name = pkg_info if isinstance(pkg_info, str) else pkg_info[0]
        if not _try_import(pkg_name):
            missing_optional.append(pkg_name)

    # Build human-readable messages
    messages = []

    if missing_required:
        messages.append(f"❌ Missing REQUIRED: {', '.join(missing_required)}")

    if missing_optional:
        messages.append("ℹ️  Optional dependencies not installed (core RAG works without these):")
        # Add detailed descriptions for optional packages
        for pkg_info in OPTIONAL_PACKAGES:
            pkg_name, pkg_desc = pkg_info if isinstance(pkg_info, tuple) else (pkg_info, pkg_info)
            if pkg_name in missing_optional:
                messages.append(f"  - {pkg_desc}")

    if not missing_required and not missing_optional:
        messages.append("✅ All packages installed (including optional)")
    elif not missing_required:
        messages.append("✅ All required packages installed")

    return len(missing_required) == 0, messages, missing_required, missing_optional
