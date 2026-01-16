"""Remote embeddings client using LangChain OllamaEmbeddings.

This module provides remote-only embedding generation via the corporate
Ollama instance. Designed for:

- VPN environments (remote-first, no local models)
- Timeout safety (config-driven connect/read + retries)
- Clean abstraction using LangChain's OllamaEmbeddings (when available)
- Explicit HTTP fallback for deterministic timeout/retry handling
"""

import logging
import os
import time
from typing import Any, List, Optional, Sequence

import numpy as np
import requests

from .config import (
    DEFAULT_RETRIES,
    EMB_CONNECT_T,
    EMB_DIM,
    EMB_READ_T,
    RAG_EMBED_MODEL,
    RAG_OLLAMA_URL,
)
from .circuit_breaker import CircuitOpenError, get_embedding_circuit_breaker
from .exceptions import EmbeddingError
from .http_utils import http_post_with_retries

logger = logging.getLogger(__name__)

# Import strategy: Prefer langchain-ollama (newer, better maintained)
# In production, fail fast if not available. In dev, allow fallback with warning.
_ENV = os.getenv("ENVIRONMENT", os.getenv("APP_ENV", "dev")).lower()
_IS_PROD = _ENV in ("prod", "production", "ci")

_HAS_LANGCHAIN = False
try:
    from langchain_ollama import OllamaEmbeddings

    _USING_FALLBACK = False
    _HAS_LANGCHAIN = True
except ImportError as e:
    if _IS_PROD:
        # PRODUCTION: Fail fast with clear error message
        raise ImportError(
            "langchain-ollama is required in production but not installed. "
            "Install with: pip install langchain-ollama\n"
            "Set ENVIRONMENT=dev to allow fallback to langchain-community (not recommended)."
        ) from e
    else:
        # DEVELOPMENT: Allow fallback but log clear warning
        logger.warning(
            "langchain-ollama not found, falling back to langchain-community.embeddings.OllamaEmbeddings. "
            "This is only allowed in DEV mode. "
            "For production, install langchain-ollama: pip install langchain-ollama"
        )
        try:
            from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

            _USING_FALLBACK = True
            _HAS_LANGCHAIN = True
        except ImportError:
            # Dev/test: allow HTTP fallback paths (embed_query/embed_texts) to work without LangChain
            from typing import Any

            OllamaEmbeddings = Any  # type: ignore
            _USING_FALLBACK = True
            _HAS_LANGCHAIN = False

# Global instance (lazy-loaded)
_EMBEDDING_CLIENT = None
_EMBEDDING_DIM: int | None = None
_RETRYABLE_EXC = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.RequestException,
    TimeoutError,
)


def _build_ollama_client_kwargs() -> dict[str, Any]:
    """Return kwargs for OllamaEmbeddings constructor with explicit timeouts where supported."""

    kwargs: dict[str, Any] = {"base_url": RAG_OLLAMA_URL, "model": RAG_EMBED_MODEL}

    # Best-effort: newer langchain-ollama versions accept client/timeout params.
    try:
        import inspect

        sig = inspect.signature(OllamaEmbeddings)
        if "client" in sig.parameters or "client_params" in sig.parameters:
            try:
                from ollama import Client as OllamaHttpClient

                http_client = OllamaHttpClient(host=RAG_OLLAMA_URL, timeout=(EMB_CONNECT_T, EMB_READ_T))
                if "client" in sig.parameters:
                    kwargs["client"] = http_client
                else:
                    kwargs["client_params"] = {"client": http_client}
                logger.debug(
                    "Configured OllamaEmbeddings HTTP client with timeouts connect=%.1fs read=%.1fs",
                    EMB_CONNECT_T,
                    EMB_READ_T,
                )
            except Exception as exc:
                logger.debug("Could not configure ollama.Client timeout: %s", exc)
    except Exception:
        # If inspect fails, fall back to base kwargs.
        pass

    return kwargs


def get_embedding_client() -> OllamaEmbeddings:
    """Get or create the remote Ollama embeddings client (lazy-loaded)."""

    global _EMBEDDING_CLIENT, _EMBEDDING_DIM

    if not _HAS_LANGCHAIN:
        raise ImportError(
            "OllamaEmbeddings not available. Install langchain-ollama or langchain-community to use this helper."
        )

    if _EMBEDDING_CLIENT is None:
        logger.debug(
            "Initializing OllamaEmbeddings: model=%s base_url=%s timeouts=(%.1fs, %.1fs)",
            RAG_EMBED_MODEL,
            RAG_OLLAMA_URL,
            EMB_CONNECT_T,
            EMB_READ_T,
        )
        kwargs = _build_ollama_client_kwargs()
        _EMBEDDING_CLIENT = OllamaEmbeddings(**kwargs)
        # Probe dimension once
        try:
            probe_vec = _EMBEDDING_CLIENT.embed_query("probe dimension")
            _EMBEDDING_DIM = len(probe_vec)
            logger.info("Remote embeddings initialized: %s (%s-dim)", RAG_EMBED_MODEL, _EMBEDDING_DIM)
        except Exception as e:
            logger.warning("Failed to probe embedding dimension: %s; assuming configured dim=%s", e, EMB_DIM)
            _EMBEDDING_DIM = EMB_DIM
    return _EMBEDDING_CLIENT


def _normalize_vectors(embeddings_list: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert embeddings to float32 and L2-normalize rows."""

    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    if embeddings_array.size == 0:
        return np.zeros((len(embeddings_list), EMB_DIM), dtype=np.float32)

    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-9
    return embeddings_array / norms


def _retry_embed(fn_name: str, retries: Optional[int], func):
    """Retry wrapper for embedding calls with small backoff."""

    max_attempts = (DEFAULT_RETRIES if retries is None else retries) + 1
    delay = 0.5
    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except _RETRYABLE_EXC as err:
            last_err = err
            logger.warning(
                "Embedding %s failed (attempt %d/%d): %s",
                fn_name,
                attempt,
                max_attempts,
                err,
            )
        except Exception as err:
            last_err = err
            if "timeout" in str(err).lower() or "connection" in str(err).lower():
                logger.warning(
                    "Embedding %s transient failure (attempt %d/%d): %s",
                    fn_name,
                    attempt,
                    max_attempts,
                    err,
                )
            else:
                raise

        if attempt < max_attempts:
            time.sleep(delay)
            delay = min(delay * 2, 4.0)

    raise EmbeddingError(f"Embedding {fn_name} failed after {max_attempts} attempts: {last_err}") from last_err


def embed_texts(texts: List[str], retries: Optional[int] = None) -> np.ndarray:
    """Embed multiple texts using remote Ollama with L2 normalization.

    Args:
        texts: List of strings to embed
        retries: Optional override for retry attempts (defaults to config.DEFAULT_RETRIES)

    Returns:
        NumPy array of shape (len(texts), embedding_dim) with float32 dtype, L2-normalized

    Raises:
        requests.Timeout: If Ollama is unreachable or too slow
        ValueError: If texts is empty
        CircuitOpenError: If embedding service circuit breaker is open
    """
    if not texts:
        return np.zeros((0, EMB_DIM), dtype=np.float32)

    # Circuit breaker check before attempting embedding
    cb = get_embedding_circuit_breaker()
    if not cb.allow_request():
        raise CircuitOpenError("ollama_embeddings", cb.get_retry_after())

    def _embed_batch() -> np.ndarray:
        # Use the REST API directly to guarantee timeout/retry control
        payloads = []
        for idx, text in enumerate(texts):
            resp = http_post_with_retries(
                f"{RAG_OLLAMA_URL}/api/embeddings",
                {"model": RAG_EMBED_MODEL, "prompt": text},
                retries=0,
                timeout=(EMB_CONNECT_T, EMB_READ_T),
            )
            if not isinstance(resp, dict) or "embedding" not in resp:
                raise EmbeddingError("Embedding response missing 'embedding' field")

            embedding = resp["embedding"]
            # Per-call dimension validation to catch model mismatches early
            actual_dim = len(embedding)
            if actual_dim != EMB_DIM:
                raise EmbeddingError(
                    f"Embedding dimension mismatch at index {idx}: got {actual_dim}, expected {EMB_DIM}. "
                    f"Check that EMB_DIM config matches the model '{RAG_EMBED_MODEL}' output."
                )
            payloads.append(embedding)

        embeddings_array = _normalize_vectors(payloads)
        logger.debug("Successfully embedded and normalized %d texts: shape %s", len(texts), embeddings_array.shape)
        return embeddings_array

    try:
        result = _retry_embed("embed_texts", retries, _embed_batch)
        cb.record_success()
        return result
    except Exception:
        cb.record_failure()
        raise


def embed_query(text: str, retries: Optional[int] = None) -> np.ndarray:
    """Embed a single query text using remote Ollama with L2 normalization.

    Args:
        text: Query string to embed
        retries: Optional override for retry attempts (defaults to config.DEFAULT_RETRIES)

    Returns:
        1D NumPy array of shape (embedding_dim,) with float32 dtype, L2-normalized

    Raises:
        requests.Timeout: If Ollama is unreachable or too slow
        ValueError: If text is empty
        CircuitOpenError: If embedding service circuit breaker is open
    """
    if not text:
        raise ValueError("Cannot embed empty text")

    # Circuit breaker check before attempting embedding
    cb = get_embedding_circuit_breaker()
    if not cb.allow_request():
        raise CircuitOpenError("ollama_embeddings", cb.get_retry_after())

    def _embed_single() -> np.ndarray:
        resp = http_post_with_retries(
            f"{RAG_OLLAMA_URL}/api/embeddings",
            {"model": RAG_EMBED_MODEL, "prompt": text},
            retries=0,
            timeout=(EMB_CONNECT_T, EMB_READ_T),
        )
        embedding_list = resp.get("embedding") if isinstance(resp, dict) else None
        if not isinstance(embedding_list, list):
            raise EmbeddingError("Embedding response missing 'embedding' list")
        embedding_array = np.array(embedding_list, dtype=np.float32)

        # Per-call dimension validation to catch model mismatches early
        actual_dim = len(embedding_array)
        if actual_dim != EMB_DIM:
            raise EmbeddingError(
                f"Embedding dimension mismatch: got {actual_dim}, expected {EMB_DIM}. "
                f"Check that EMB_DIM config matches the model '{RAG_EMBED_MODEL}' output."
            )

        norm = np.linalg.norm(embedding_array) + 1e-9
        embedding_array = embedding_array / norm
        logger.debug("Successfully embedded and normalized query: shape %s", embedding_array.shape)
        return embedding_array

    try:
        result = _retry_embed("embed_query", retries, _embed_single)
        cb.record_success()
        return result
    except Exception:
        cb.record_failure()
        raise


def clear_cache():
    """Clear the cached embedding client instance.

    Useful for testing or switching Ollama endpoints at runtime.
    """
    global _EMBEDDING_CLIENT
    _EMBEDDING_CLIENT = None
    logger.debug("Cleared embedding client cache")
