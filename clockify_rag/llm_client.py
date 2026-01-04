"""Remote-first LLM client factory using LangChain ChatOllama.

This module provides a clean, production-ready interface for LLM calls via
the corporate Ollama instance. Designed for VPN environments with:

- Non-streaming generation (VPN stability)
- Configurable timeouts (default 120s)
- Optional fallback model selection (no remote probe by default)
- VPN-safe error handling (no indefinite hangs)
"""

import asyncio
import logging
import os
import threading
from typing import List, Union

import httpx
from langchain_core.messages import BaseMessage

from . import config
from .circuit_breaker import CircuitOpenError, get_ollama_circuit_breaker

logger = logging.getLogger(__name__)

# Cached HTTP client for connection pool reuse
_HTTP_CLIENT = None
_HTTP_CLIENT_LOCK = threading.Lock()

# Import strategy: Prefer langchain-ollama (newer, better maintained)
# In production, fail fast if not available. In dev, allow fallback with warning.
_ENV = os.getenv("ENVIRONMENT", os.getenv("APP_ENV", "dev")).lower()
_IS_PROD = _ENV in ("prod", "production", "ci")

try:
    from langchain_ollama import ChatOllama

    _USING_FALLBACK = False
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
            "langchain-ollama not found, falling back to langchain-community.chat_models.ChatOllama. "
            "This is only allowed in DEV mode. "
            "For production, install langchain-ollama: pip install langchain-ollama"
        )
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore

            _USING_FALLBACK = True
        except ImportError as e2:
            raise ImportError(
                "Neither langchain-ollama nor langchain-community is available. "
                "Install langchain-ollama: pip install langchain-ollama"
            ) from e2


def _get_http_client() -> httpx.Client:
    """Get or create the cached HTTP client for connection pool reuse.

    Thread-safe lazy initialization ensures:
    - Single httpx.Client instance across all LLM calls
    - Connection pool reuse for better performance
    - Proper timeout configuration

    Returns:
        Cached httpx.Client instance with configured timeout
    """
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            # Double-checked locking pattern
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = httpx.Client(timeout=config.OLLAMA_TIMEOUT, trust_env=config.ALLOW_PROXIES)
                logger.debug(
                    "Created cached HTTP client with timeout=%.1fs trust_env=%s",
                    config.OLLAMA_TIMEOUT,
                    config.ALLOW_PROXIES,
                )
    return _HTTP_CLIENT


def _build_chat_client_kwargs(model_name: str, temperature: float) -> dict:
    """Return kwargs for ChatOllama with best-effort timeout/proxy configuration."""

    kwargs = {
        "base_url": config.RAG_OLLAMA_URL,
        "model": model_name,
        "temperature": temperature,
    }

    try:
        import inspect

        sig = inspect.signature(ChatOllama)

        if "timeout" in sig.parameters:
            kwargs["timeout"] = config.OLLAMA_TIMEOUT
        elif "request_timeout" in sig.parameters:
            kwargs["request_timeout"] = config.OLLAMA_TIMEOUT

        if "http_client" in sig.parameters:
            kwargs["http_client"] = _get_http_client()
        elif "client" in sig.parameters or "client_params" in sig.parameters:
            try:
                from ollama import Client as OllamaHttpClient

                client = OllamaHttpClient(host=config.RAG_OLLAMA_URL, timeout=config.OLLAMA_TIMEOUT)
                if "client" in sig.parameters:
                    kwargs["client"] = client
                else:
                    kwargs["client_params"] = {"client": client}
            except Exception as exc:
                logger.debug("Could not configure ollama.Client timeout: %s", exc)

        if "stream" in sig.parameters:
            kwargs["stream"] = False
    except Exception:
        pass

    return kwargs


def get_llm_client(temperature: float = 0.0) -> ChatOllama:
    """Create and return a ChatOllama client for remote generation.

    This is the single source of truth for all LLM calls in the system.
    Uses remote-first design: connects to corporate Ollama over VPN with
    non-streaming mode for stability and explicit timeout controls.

    Args:
        temperature: Sampling temperature (0.0-1.0; 0.0 = deterministic)

    Returns:
        ChatOllama instance configured for remote generation with:
        - Non-streaming (VPN safe, no infinite hangs)
        - Timeout enforcement (120s default via OLLAMA_TIMEOUT)
        - Selected model (defaults to RAG_CHAT_MODEL; no remote probe by default)
        - Base URL: RAG_OLLAMA_URL (corporate Ollama instance)

    Usage:
        ```python
        from clockify_rag.llm_client import get_llm_client

        llm = get_llm_client(temperature=0.0)
        response = llm.invoke("What is 2+2?")
        print(response.content)
        ```

    Notes:
        - Model selection uses config.get_llm_model() and defaults to RAG_CHAT_MODEL
        - Fallback selection is opt-in (invoke config._select_best_model explicitly)
        - If Ollama is unreachable, this client still uses the configured primary model
        - All calls timeout after OLLAMA_TIMEOUT seconds (default 120s, configurable)
    """
    model_name = config.get_llm_model()
    logger.debug(
        f"Creating ChatOllama client: model={model_name}, "
        f"base_url={config.RAG_OLLAMA_URL}, timeout={config.OLLAMA_TIMEOUT}s, streaming=False"
    )

    kwargs = _build_chat_client_kwargs(model_name, temperature)
    return ChatOllama(**kwargs)


def get_llm_client_async(temperature: float = 0.0) -> ChatOllama:
    """Create an async-capable ChatOllama client.

    LangChain's ChatOllama supports async operations via inherited methods
    from BaseChatModel (ainvoke, abatch, astream). This function returns
    a ChatOllama instance that can be used with both sync and async patterns.

    Args:
        temperature: Sampling temperature (0.0-1.0)

    Returns:
        ChatOllama instance capable of async operations via ainvoke()

    Usage:
        ```python
        import asyncio
        from clockify_rag.llm_client import get_llm_client_async

        async def ask():
            llm = get_llm_client_async()
            response = await llm.ainvoke("What is 2+2?")
            return response.content

        result = asyncio.run(ask())
        ```
    """
    # ChatOllama inherits async support from BaseChatModel (ainvoke, abatch, astream)
    # These methods use asyncio.to_thread() internally for the sync implementation
    return get_llm_client(temperature)


def invoke_llm(
    prompt: Union[str, List[BaseMessage]],
    temperature: float = 0.0,
) -> str:
    """Invoke the LLM with circuit breaker protection.

    This is the recommended way to call the LLM as it provides:
    - Circuit breaker protection to prevent hammering unresponsive services
    - Automatic failure tracking and recovery
    - Clean error handling with CircuitOpenError

    Args:
        prompt: Either a string prompt or a list of LangChain messages
        temperature: Sampling temperature (0.0-1.0; 0.0 = deterministic)

    Returns:
        The LLM response content as a string

    Raises:
        CircuitOpenError: If the LLM service circuit breaker is open
        Exception: Any underlying LLM errors (connection, timeout, etc.)

    Usage:
        ```python
        from clockify_rag.llm_client import invoke_llm

        # Simple string prompt
        response = invoke_llm("What is 2+2?")

        # With messages
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is 2+2?"),
        ]
        response = invoke_llm(messages)
        ```
    """
    cb = get_ollama_circuit_breaker()

    if not cb.allow_request():
        raise CircuitOpenError("ollama_llm", cb.get_retry_after())

    try:
        llm = get_llm_client(temperature)
        response = llm.invoke(prompt)
        cb.record_success()
        return str(getattr(response, "content", response))
    except Exception:
        cb.record_failure()
        raise


async def invoke_llm_async(
    prompt: Union[str, List[BaseMessage]],
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    """Invoke the LLM asynchronously with circuit breaker protection and timeout.

    This is the async equivalent of invoke_llm(), providing:
    - Circuit breaker protection to prevent hammering unresponsive services
    - Automatic failure tracking and recovery
    - Non-blocking LLM calls suitable for async FastAPI endpoints
    - Per-call timeout enforcement

    Args:
        prompt: Either a string prompt or a list of LangChain messages
        temperature: Sampling temperature (0.0-1.0; 0.0 = deterministic)
        timeout: Maximum time in seconds to wait for LLM response (default 120.0)

    Returns:
        The LLM response content as a string

    Raises:
        CircuitOpenError: If the LLM service circuit breaker is open
        asyncio.TimeoutError: If the LLM call exceeds the timeout
        Exception: Any underlying LLM errors (connection, timeout, etc.)

    Usage:
        ```python
        from clockify_rag.llm_client import invoke_llm_async

        # In an async context (FastAPI, asyncio.run, etc.)
        async def handle_query(question: str):
            response = await invoke_llm_async(f"Answer this: {question}")
            return response

        # With messages
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is 2+2?"),
        ]
        response = await invoke_llm_async(messages)

        # With custom timeout
        response = await invoke_llm_async("Quick question", timeout=30.0)
        ```
    """
    cb = get_ollama_circuit_breaker()

    if not cb.allow_request():
        raise CircuitOpenError("ollama_llm", cb.get_retry_after())

    try:
        llm = get_llm_client_async(temperature)
        # Use ainvoke for async operation with timeout enforcement
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout)
        cb.record_success()
        return str(getattr(response, "content", response))
    except asyncio.TimeoutError:
        cb.record_failure()
        raise
    except Exception:
        cb.record_failure()
        raise
