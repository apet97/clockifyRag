"""Error handling and graceful fallback utilities for the RAG system.

This module provides consistent error handling patterns, clear error messages,
and graceful degradation strategies across all RAG components.
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple
from functools import wraps

from .exceptions import LLMError, EmbeddingError, IndexLoadError, BuildError
from .config import (
    RAG_OLLAMA_URL,
    RAG_CHAT_MODEL,
    RAG_EMBED_MODEL,
    DEFAULT_RAG_OLLAMA_URL,
    DEFAULT_LOCAL_OLLAMA_URL,
)


logger = logging.getLogger(__name__)


def sanitize_for_client(message: str) -> str:
    """Sanitize error messages for client-facing responses.

    Removes:
    - Internal URLs and IPs
    - File paths
    - Environment variable values

    Args:
        message: Error message to sanitize

    Returns:
        Sanitized message safe for client responses
    """
    # Replace URLs (http://10.x.x.x:port, http://localhost:port, etc.)
    message = re.sub(r"http://[\d.:]+", "<endpoint>", message)
    message = re.sub(r"https://[\d.:]+", "<endpoint>", message)

    # Replace file paths (/path/to/file, C:\path\to\file)
    message = re.sub(r"[/\\][\w/\\.-]+\.\w+", "<file>", message)
    message = re.sub(r"[A-Z]:[/\\][\w/\\.-]+", "<file>", message)

    # Replace environment variable values (KEY=value -> KEY=<redacted>)
    message = re.sub(r"(\w+)=([^\s]+)", r"\1=<redacted>", message)

    return message


def format_error_message(error_type: str, message: str, hint: Optional[str] = None) -> str:
    """Format a consistent error message with type, message, and optional hint.

    Args:
        error_type: Type of error (e.g., "LLM_ERROR", "EMBEDDING_ERROR")
        message: Error message
        hint: Optional hint for resolution

    Returns:
        Formatted error message
    """
    formatted = f"[{error_type}] {message}"
    if hint:
        formatted += f" [hint: {hint}]"
    return formatted


def handle_llm_errors(func):
    """Decorator to handle LLM-related errors gracefully."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LLMError as e:
            logger.error(
                format_error_message(
                    "LLM_ERROR", str(e), f"check RAG_OLLAMA_URL={RAG_OLLAMA_URL} or increase CHAT timeouts"
                )
            )
            raise
        except Exception as e:
            logger.error(
                format_error_message("LLM_UNEXPECTED", str(e), f"unexpected error in LLM call: {type(e).__name__}")
            )
            raise LLMError(f"Unexpected LLM error: {e}") from e

    return wrapper


def handle_embedding_errors(func):
    """Decorator to handle embedding-related errors gracefully."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EmbeddingError as e:
            logger.error(
                format_error_message(
                    "EMBEDDING_ERROR", str(e), f"check RAG_OLLAMA_URL={RAG_OLLAMA_URL} or increase EMB timeouts"
                )
            )
            raise
        except Exception as e:
            logger.error(
                format_error_message(
                    "EMBEDDING_UNEXPECTED", str(e), f"unexpected error in embedding: {type(e).__name__}"
                )
            )
            raise EmbeddingError(f"Unexpected embedding error: {e}") from e

    return wrapper


def handle_index_errors(func):
    """Decorator to handle index-related errors gracefully."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IndexLoadError as e:
            logger.error(
                format_error_message(
                    "INDEX_ERROR",
                    str(e),
                    "run 'python -m clockify_rag.cli_modern ingest --input knowledge_helpcenter.md --force' to rebuild",
                )
            )
            raise
        except Exception as e:
            logger.error(
                format_error_message("INDEX_UNEXPECTED", str(e), f"unexpected error in indexing: {type(e).__name__}")
            )
            raise IndexLoadError(f"Unexpected index error: {e}") from e

    return wrapper


def handle_build_errors(func):
    """Decorator to handle build-related errors gracefully."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BuildError as e:
            logger.error(format_error_message("BUILD_ERROR", str(e), "check source file and configuration"))
            raise
        except Exception as e:
            logger.error(
                format_error_message("BUILD_UNEXPECTED", str(e), f"unexpected error in build: {type(e).__name__}")
            )
            raise BuildError(f"Unexpected build error: {e}") from e

    return wrapper


def validate_configuration() -> Tuple[bool, str]:
    """Validate the current configuration and return (is_valid, error_message).

    Returns:
        Tuple of (is_valid, error_message or empty string)
    """
    try:
        # Validate RAG_OLLAMA_URL format
        if not RAG_OLLAMA_URL or not isinstance(RAG_OLLAMA_URL, str):
            return False, format_error_message(
                "CONFIG_ERROR",
                f"RAG_OLLAMA_URL must be a string, got {type(RAG_OLLAMA_URL)}",
                f"set RAG_OLLAMA_URL={DEFAULT_RAG_OLLAMA_URL} (or {DEFAULT_LOCAL_OLLAMA_URL} for local Ollama)",
            )

        # Validate model names
        if not RAG_CHAT_MODEL or not isinstance(RAG_CHAT_MODEL, str):
            return False, format_error_message(
                "CONFIG_ERROR",
                f"RAG_CHAT_MODEL must be a string, got {type(RAG_CHAT_MODEL)}",
                "set RAG_CHAT_MODEL=qwen2.5:32b",
            )

        if not RAG_EMBED_MODEL or not isinstance(RAG_EMBED_MODEL, str):
            return False, format_error_message(
                "CONFIG_ERROR",
                f"RAG_EMBED_MODEL must be a string, got {type(RAG_EMBED_MODEL)}",
                "set RAG_EMBED_MODEL=nomic-embed-text",
            )

        # Additional validation can be added here

        return True, ""

    except Exception as e:
        error_msg = format_error_message(
            "CONFIG_ERROR", f"Configuration validation failed: {str(e)}", "check environment variables"
        )
        logger.error(error_msg)
        return False, error_msg


def handle_api_call_errors(func):
    """Decorator to handle API call errors with detailed logging."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, TimeoutError) as e:
            error_msg = format_error_message(
                "CONNECTION_ERROR", f"API connection failed: {str(e)}", f"check connectivity to {RAG_OLLAMA_URL}"
            )
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = format_error_message(
                "API_ERROR", f"API call failed: {str(e)}", "check API endpoint and credentials"
            )
            logger.error(error_msg)
            raise

    return wrapper


def log_and_raise(exception_class, message: str, hint: Optional[str] = None, log_level: int = logging.ERROR):
    """Log an error message and raise an exception.

    Args:
        exception_class: Exception class to raise
        message: Error message
        hint: Optional hint for resolution
        log_level: Logging level to use

    Raises:
        exception_class: with formatted message
    """
    formatted_msg = format_error_message(exception_class.__name__.upper().replace("ERROR", "_ERROR"), message, hint)
    logger.log(log_level, formatted_msg)
    raise exception_class(formatted_msg)


def graceful_error_handler(error_type: str, default_return: Any = None, log_level: int = logging.ERROR):
    """Decorator to catch errors and return a default value instead of crashing.

    Args:
        error_type: Type of error to catch (or "ALL" for all exceptions)
        default_return: Value to return on error
        log_level: Logging level for the error

    Returns:
        The original function wrapped with error handling
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (SystemExit, KeyboardInterrupt):
                # Never swallow these - always re-raise
                raise
            except Exception as e:
                if error_type == "ALL" or error_type in str(type(e)):
                    error_msg = format_error_message(
                        error_type.upper() if error_type != "ALL" else "GRACEFUL_FALLBACK",
                        f"Error in {func.__name__}: {str(e)}",
                        "using fallback behavior",
                    )
                    logger.log(log_level, error_msg)
                    return default_return
                else:
                    raise  # Re-raise if not the target error type

        return wrapper

    return decorator


def check_endpoint_health() -> Tuple[bool, str, Optional[Dict]]:
    """Check if the configured endpoints are accessible.

    Returns:
        Tuple of (is_healthy, status_message, additional_info)
    """
    from .api_client import get_llm_client

    try:
        client = get_llm_client()
        models = client.list_models()
        if models:
            model_names = [model.get("name") or model.get("model") for model in models]
            return True, f"Healthy - {len(models)} models available", {"models": model_names}
        return True, "Healthy - endpoint accessible", None
    except Exception as e:
        # Sanitize error message for client-facing response
        safe_error = sanitize_for_client(str(e))
        return False, f"Unhealthy - Error checking endpoint: {safe_error}", {"error": safe_error}


def print_system_health():
    """Print detailed system health information to stdout."""
    print("=" * 60)
    print("SYSTEM HEALTH CHECK")
    print("=" * 60)

    # Configuration validation
    is_config_valid, config_error = validate_configuration()
    print(f"✓ Configuration: {'VALID' if is_config_valid else 'INVALID'}")
    if not is_config_valid:
        print(f"  Error: {config_error}")

    # Endpoint health
    is_healthy, status_msg, additional_info = check_endpoint_health()
    print(f"✓ Endpoint Health: {status_msg}")

    if additional_info:
        print(f"  Additional Info: {additional_info}")

    # Check required files
    import os
    from .config import FILES

    missing_files = []
    for name, path in FILES.items():
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        print(f"⚠ Index Files: {len(missing_files)} missing files")
        for f in missing_files[:5]:  # Show first 5 missing files
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    else:
        print("✓ Index Files: All present")

    print("=" * 60)

    if not is_config_valid or not is_healthy:
        print("❌ System has issues that need to be addressed.")
        return False
    else:
        print("✅ System is healthy and ready for use!")
        return True


# Global error handling configuration
def setup_error_handlers():
    """Setup global error handling configuration."""
    # This can be expanded with additional global error handling setup
    pass


# Initialize error handlers
setup_error_handlers()
