"""Typed API client for Ollama-compatible endpoints.

This module provides a typed, structured client for interacting with
Ollama-style APIs for both chat completion and embedding generation.
"""

import hashlib
import logging
import math
import random
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union, cast
from typing_extensions import TypedDict

import requests

from .config import (
    RAG_OLLAMA_URL,
    RAG_CHAT_MODEL,
    RAG_EMBED_MODEL,
    EMB_BACKEND,
    EMB_DIM_LOCAL,
    EMB_DIM_OLLAMA,
    EMB_CONNECT_T,
    EMB_READ_T,
    CHAT_CONNECT_T,
    CHAT_READ_T,
    RERANK_READ_T,
    DEFAULT_RETRIES,
    DEFAULT_SEED,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    ALLOW_PROXIES,
    get_llm_client_mode,
)
from .circuit_breaker import CircuitOpenError, get_ollama_circuit_breaker
from .exceptions import LLMError, EmbeddingError, LLMUnavailableError, LLMBadResponseError
from .http_utils import get_session


logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    """Typed representation of a chat message."""

    role: str  # "system", "user", or "assistant"
    content: str


class ChatCompletionOptions(TypedDict, total=False):
    """Typed representation of chat completion options."""

    temperature: float
    seed: int
    num_ctx: int
    num_predict: int
    top_p: float
    top_k: int
    repeat_penalty: float


class ChatCompletionRequest(TypedDict, total=False):
    """Typed representation of chat completion request."""

    model: str
    messages: List[ChatMessage]
    options: Optional[ChatCompletionOptions]
    stream: bool
    keep_alive: Union[str, int]


class ChatCompletionResponse(TypedDict):
    """Typed representation of chat completion response."""

    model: str
    created_at: str
    message: Dict[str, str]  # {"role": str, "content": str}
    done: bool
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


class EmbeddingRequest(TypedDict):
    """Typed representation of embedding request."""

    model: str
    prompt: str  # Ollama uses "prompt" for embeddings
    options: Optional[Dict[str, Any]]


class EmbeddingResponse(TypedDict):
    """Typed representation of embedding response."""

    embedding: List[float]


class ModelInfo(TypedDict):
    """Typed representation of model information."""

    model: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any]


class BaseLLMClient:
    """Interface for LLM/embedding clients."""

    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        options: Optional[ChatCompletionOptions] = None,
        stream: bool = False,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> ChatCompletionResponse:
        raise NotImplementedError

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[ChatCompletionOptions] = None,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> str:
        raise NotImplementedError

    def create_embedding(
        self, text: str, model: Optional[str] = None, timeout: Optional[tuple] = None, retries: Optional[int] = None
    ) -> List[float]:
        raise NotImplementedError

    def create_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> List[List[float]]:
        raise NotImplementedError

    def list_models(self) -> List[ModelInfo]:
        return []

    def check_health(self) -> bool:
        return True


class OllamaAPIClient(BaseLLMClient):
    """Type-safe client for Ollama-compatible APIs.

    Provides structured interfaces for chat completion and embedding
    generation with proper error handling and configuration management.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        gen_model: Optional[str] = None,
        emb_model: Optional[str] = None,
        chat_connect_timeout: Optional[float] = None,
        chat_read_timeout: Optional[float] = None,
        emb_connect_timeout: Optional[float] = None,
        emb_read_timeout: Optional[float] = None,
        rerank_read_timeout: Optional[float] = None,
        retries: Optional[int] = None,
    ):
        """Initialize the Ollama API client with configuration.

        Args:
            base_url: Base URL for Ollama API (defaults to config.RAG_OLLAMA_URL)
            gen_model: Generation model name (defaults to config.RAG_CHAT_MODEL)
            emb_model: Embedding model name (defaults to config.RAG_EMBED_MODEL)
            chat_connect_timeout: Chat connection timeout (defaults to config)
            chat_read_timeout: Chat read timeout (defaults to config)
            emb_connect_timeout: Embedding connection timeout (defaults to config)
            emb_read_timeout: Embedding read timeout (defaults to config)
            rerank_read_timeout: Rerank-specific timeout (defaults to config)
            retries: Number of retries for failed requests (defaults to config)
        """
        self.base_url: str = base_url or RAG_OLLAMA_URL or ""
        self.gen_model: str = gen_model or RAG_CHAT_MODEL or ""
        self.emb_model: str = emb_model or RAG_EMBED_MODEL or ""
        self.chat_connect_timeout = chat_connect_timeout or CHAT_CONNECT_T
        self.chat_read_timeout = chat_read_timeout or CHAT_READ_T
        self.emb_connect_timeout = emb_connect_timeout or EMB_CONNECT_T
        self.emb_read_timeout = emb_read_timeout or EMB_READ_T
        self.rerank_read_timeout = rerank_read_timeout or RERANK_READ_T
        self.retries = retries or DEFAULT_RETRIES

        # Set up session with proper proxy configuration
        self.session = get_session(retries=self.retries)
        self.session.trust_env = ALLOW_PROXIES
        base = self.base_url.rstrip("/") if self.base_url else ""
        self._chat_endpoint = f"{base}/api/chat"
        self._emb_endpoint = f"{base}/api/embeddings"

    def _get_session(self, retries: int) -> requests.Session:
        """Return a requests.Session configured for the desired retry count."""
        session = get_session(retries=retries, use_thread_local=True)
        session.trust_env = ALLOW_PROXIES
        return session

    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        options: Optional[ChatCompletionOptions] = None,
        stream: bool = False,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> ChatCompletionResponse:
        """Make a chat completion request to the Ollama API.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to gen_model)
            options: Generation options
            stream: Whether to stream the response
            timeout: (connect, read) timeout tuple
            retries: Number of retries for this request

        Returns:
            Chat completion response

        Raises:
            LLMError: If the request fails after all retries
            CircuitOpenError: If the LLM service circuit breaker is open
        """
        # Circuit breaker check before attempting request
        cb = get_ollama_circuit_breaker()
        if not cb.allow_request():
            raise CircuitOpenError("ollama_llm", cb.get_retry_after())

        model = model or self.gen_model
        options = options or {
            "temperature": 0,
            "seed": DEFAULT_SEED,
            "num_ctx": DEFAULT_NUM_CTX,
            "num_predict": DEFAULT_NUM_PREDICT,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05,
        }

        payload: ChatCompletionRequest = {
            "model": model or self.gen_model,
            "messages": messages,
            "options": options,
            "stream": stream,
        }

        req_timeout = timeout or (self.chat_connect_timeout, self.chat_read_timeout)
        req_retries = retries or self.retries
        session = self._get_session(req_retries)
        start_time = time.time()
        response = None

        try:
            response = session.post(self._chat_endpoint, json=payload, timeout=req_timeout, allow_redirects=False)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout as e:
            cb.record_failure()  # Track failure for circuit breaker
            logger.error(
                "Chat completion timeout (read %.1fs) model=%s host=%s: %s",
                req_timeout[1],
                model,
                self.base_url,
                e,
            )
            raise LLMUnavailableError(f"Chat completion timeout for model {model}") from e
        except requests.exceptions.ConnectionError as e:
            cb.record_failure()  # Track failure for circuit breaker
            logger.error(
                "Chat completion connection error model=%s host=%s: %s",
                model,
                self.base_url,
                e,
            )
            raise LLMUnavailableError(f"Chat completion connection error for model {model}") from e
        except requests.exceptions.HTTPError as e:
            cb.record_failure()  # Track failure for circuit breaker
            status = getattr(e.response, "status_code", getattr(response, "status_code", "unknown"))
            logger.error(
                "Chat completion HTTP error model=%s host=%s status=%s: %s",
                model,
                self.base_url,
                status,
                e,
            )
            raise LLMError(f"Chat completion HTTP error (status {status})") from e
        except ValueError as e:
            cb.record_failure()  # Track failure for circuit breaker
            logger.error(
                "Chat completion invalid JSON model=%s host=%s: %s",
                model,
                self.base_url,
                e,
            )
            raise LLMBadResponseError(f"Chat completion returned invalid JSON for model {model}") from e
        except requests.exceptions.RequestException as e:
            cb.record_failure()  # Track failure for circuit breaker
            logger.error(
                "Chat completion request error model=%s host=%s: %s",
                model,
                self.base_url,
                e,
            )
            raise LLMError(f"Chat completion request error: {e}") from e
        except Exception as e:
            cb.record_failure()  # Track failure for circuit breaker
            logger.error("Chat completion unexpected error model=%s: %s", model, e)
            raise LLMError(f"Chat completion unexpected error: {e}") from e

        validated = self._validate_chat_response(result, model)
        cb.record_success()  # Track success for circuit breaker
        duration = time.time() - start_time
        logger.debug("Chat completion finished in %.2fs model=%s", duration, model)
        return validated

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[ChatCompletionOptions] = None,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> str:
        """Generate text from a simple prompt using the chat endpoint.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use (defaults to gen_model)
            options: Generation options
            timeout: (connect, read) timeout tuple
            retries: Number of retries for this request

        Returns:
            Generated text
        """
        messages: List[ChatMessage] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.chat_completion(
            messages=messages, model=model, options=options, timeout=timeout, retries=retries
        )

        return response["message"]["content"]

    def create_embedding(
        self, text: str, model: Optional[str] = None, timeout: Optional[tuple] = None, retries: Optional[int] = None
    ) -> List[float]:
        """Create an embedding for the given text.

        Args:
            text: Text to embed
            model: Model to use (defaults to emb_model for Ollama, ignored for local)
            timeout: (connect, read) timeout tuple
            retries: Number of retries for this request

        Returns:
            Embedding vector as a list of floats

        Raises:
            EmbeddingError: If embedding generation fails
        """
        # If using local embeddings, return an error since this method is for API calls
        if EMB_BACKEND == "local":
            raise EmbeddingError(
                "create_embedding method is for API embeddings only. " "Use local embedding methods for local backend."
            )

        model = model or self.emb_model
        req_timeout = timeout or (self.emb_connect_timeout, self.emb_read_timeout)
        req_retries = retries or self.retries

        # Get session with appropriate retry settings for this request
        session = self._get_session(req_retries)
        response = None
        payload: EmbeddingRequest = {"model": model, "prompt": text, "options": None}

        logger.debug(f"Creating embedding for text of length {len(text)} with model {model}")
        start_time = time.time()

        try:
            response = session.post(self._emb_endpoint, json=payload, timeout=req_timeout, allow_redirects=False)
            response.raise_for_status()
            result = response.json()
            embedding = self._validate_embedding_response(result)
            duration = time.time() - start_time
            logger.debug("Embedding created in %.2fs for model %s dim=%d", duration, model, len(embedding))
            return embedding

        except requests.exceptions.Timeout as e:
            logger.error(
                "Embedding creation timeout (read %.1fs) model=%s host=%s: %s",
                req_timeout[1],
                model,
                self.base_url,
                e,
            )
            raise EmbeddingError(f"Embedding creation timeout for model {model}") from e
        except requests.exceptions.ConnectionError as e:
            logger.error(
                "Embedding creation connection error model=%s host=%s: %s",
                model,
                self.base_url,
                e,
            )
            raise EmbeddingError(f"Embedding creation connection error for model {model}") from e
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", getattr(response, "status_code", "unknown"))
            logger.error(
                "Embedding creation HTTP error model=%s host=%s status=%s: %s",
                model,
                self.base_url,
                status,
                e,
            )
            raise EmbeddingError(f"Embedding creation HTTP error (status {status})") from e
        except ValueError as e:
            logger.error(
                "Embedding creation invalid JSON model=%s host=%s: %s",
                model,
                self.base_url,
                e,
            )
            raise EmbeddingError(f"Embedding creation returned invalid JSON for model {model}") from e
        except requests.exceptions.RequestException as e:
            logger.error(
                "Embedding creation request error model=%s host=%s: %s",
                model,
                self.base_url,
                e,
            )
            raise EmbeddingError(f"Embedding creation request error: {e}") from e
        except Exception as e:
            logger.error("Embedding creation unexpected error model=%s: %s", model, e)
            raise EmbeddingError(f"Embedding creation unexpected error: {e}") from e

    def create_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> List[List[float]]:
        """Create embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to emb_model)
            timeout: (connect, read) timeout tuple for each request
            retries: Number of retries for each request

        Returns:
            List of embedding vectors
        """
        if EMB_BACKEND == "local":
            raise EmbeddingError(
                "create_embeddings_batch method is for API embeddings only. "
                "Use local embedding methods for local backend."
            )

        embeddings = []
        for text in texts:
            embedding = self.create_embedding(text=text, model=model, timeout=timeout, retries=retries)
            embeddings.append(embedding)

        return embeddings

    @staticmethod
    def _validate_chat_response(result: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Ensure chat responses contain textual assistant content."""
        if not isinstance(result, dict):
            raise LLMBadResponseError(f"Chat completion returned non-object payload for model {model}")

        result_model = result.get("model")
        if not isinstance(result_model, str) or not result_model.strip():
            raise LLMBadResponseError(f"Chat completion missing model name for model {model}")

        message = result.get("message")
        if not isinstance(message, dict):
            raise LLMBadResponseError(f"Chat completion missing 'message' block for model {model}")

        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise LLMBadResponseError(f"Chat completion missing message role for model {model}")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMBadResponseError(f"Chat completion missing textual content for model {model}")

        return cast(ChatCompletionResponse, result)

    @staticmethod
    def _validate_embedding_response(result: Dict[str, Any]) -> List[float]:
        """Ensure embedding responses return numeric vectors of the expected size."""

        if not isinstance(result, dict):
            raise EmbeddingError("Embedding response must be a JSON object.")

        embedding = result.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingError("Embedding response missing 'embedding' list.")

        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding response contained non-numeric values.") from exc

        expected_dim = EMB_DIM_OLLAMA if EMB_BACKEND == "ollama" else EMB_DIM_LOCAL
        if len(vector) != expected_dim:
            raise EmbeddingError(f"Embedding dimension mismatch (expected {expected_dim}, got {len(vector)})")

        return vector

    def list_models(self) -> List[ModelInfo]:
        """List available models on the server.

        Returns:
            List of available models
        """
        try:
            session = get_session(retries=0)  # No retries for listing models
            session.trust_env = ALLOW_PROXIES

            response = session.get(
                f"{self.base_url}/api/tags",
                timeout=(self.chat_connect_timeout, 5.0),  # Short read timeout for listing
                allow_redirects=False,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("models", [])

        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

    def check_health(self) -> bool:
        """Check if the Ollama server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            models = self.list_models()
            return len(models) > 0
        except Exception:
            return False


class MockLLMClient(BaseLLMClient):
    """Deterministic, in-memory LLM client for tests and offline workflows."""

    def __init__(
        self,
        gen_model: Optional[str] = None,
        emb_model: Optional[str] = None,
        embed_dim: Optional[int] = None,
    ) -> None:
        self.gen_model: str = gen_model or RAG_CHAT_MODEL or ""
        self.emb_model: str = emb_model or RAG_EMBED_MODEL or ""
        if embed_dim is not None:
            self.embed_dim = embed_dim
        else:
            self.embed_dim = EMB_DIM_OLLAMA if EMB_BACKEND == "ollama" else EMB_DIM_LOCAL
        self._chat_overrides: Dict[str, str] = {}
        self._embedding_overrides: Dict[str, List[float]] = {}

    def register_chat_response(self, prompt: str, response: str) -> None:
        """Register a deterministic response for a specific prompt."""
        self._chat_overrides[prompt.strip()] = response

    def register_embedding(self, text: str, vector: List[float]) -> None:
        """Register a deterministic embedding override for a specific text."""
        self._embedding_overrides[text.strip()] = vector

    def _deterministic_vector(self, text: str) -> List[float]:
        override = self._embedding_overrides.get(text.strip())
        if override is not None:
            return override

        seed = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.embed_dim)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        options: Optional[ChatCompletionOptions] = None,
        stream: bool = False,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> ChatCompletionResponse:
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        content = self._chat_overrides.get(last_user.strip())
        if content is None:
            snippet = (last_user or "").strip().replace("\n", " ")
            snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
            content = f"[mock-answer] {snippet or 'No question provided.'}"

        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        resolved_model: str = model or self.gen_model
        response: ChatCompletionResponse = {
            "model": resolved_model,
            "created_at": now,
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": len(messages),
            "prompt_eval_duration": 0,
            "eval_count": len(content),
            "eval_duration": 0,
        }
        return response

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[ChatCompletionOptions] = None,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> str:
        messages: List[ChatMessage] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.chat_completion(
            messages=messages,
            model=model,
            options=options,
            stream=False,
            timeout=timeout,
            retries=retries,
        )
        return response.get("message", {}).get("content", "")

    def create_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> List[float]:
        return self._deterministic_vector(text)

    def create_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        timeout: Optional[tuple] = None,
        retries: Optional[int] = None,
    ) -> List[List[float]]:
        return [self.create_embedding(text, model=model) for text in texts]

    def list_models(self) -> List[ModelInfo]:
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return [
            {
                "model": self.gen_model,
                "modified_at": now,
                "size": 0,
                "digest": "mock-gen",
                "details": {"family": "mock"},
            },
            {
                "model": self.emb_model,
                "modified_at": now,
                "size": 0,
                "digest": "mock-emb",
                "details": {"family": "mock"},
            },
        ]

    def check_health(self) -> bool:
        return True


# Global client instance for backward compatibility
_LLM_CLIENT: Optional[BaseLLMClient] = None


def set_llm_client(client: Optional[BaseLLMClient]) -> None:
    """Force the global LLM client (useful for tests)."""
    global _LLM_CLIENT
    _LLM_CLIENT = client


def reset_llm_client() -> None:
    """Reset the cached LLM client (next access will re-instantiate)."""
    set_llm_client(None)


def get_ollama_client() -> BaseLLMClient:
    """Get a global instance of the Ollama API client.

    Returns:
        Ollama API client instance
    """
    return get_llm_client()


def get_llm_client() -> BaseLLMClient:
    """Get the configured LLM client (real or mock).

    NOTE: This returns a BaseLLMClient (api_client abstraction layer).
    For direct LangChain ChatOllama access, use llm_client.get_llm_client() instead.

    The OllamaAPIClient uses requests directly for HTTP calls.
    Future work: consider migrating to llm_client.get_llm_client() for unified client handling.
    """
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT
    client_pref = get_llm_client_mode()
    if client_pref in {"mock", "test"}:
        logger.info("Using MockLLMClient (RAG_LLM_CLIENT=%s)", client_pref or "mock")
        _LLM_CLIENT = MockLLMClient()
    else:
        logger.debug("Initializing OllamaAPIClient (requests-based)")
        _LLM_CLIENT = OllamaAPIClient()
    return _LLM_CLIENT


def chat_completion(
    messages: List[ChatMessage],
    model: Optional[str] = None,
    options: Optional[ChatCompletionOptions] = None,
    stream: bool = False,
    timeout: Optional[tuple] = None,
    retries: Optional[int] = None,
) -> ChatCompletionResponse:
    """Global function to make chat completion requests.

    Args:
        messages: List of chat messages
        model: Model to use (defaults to configured gen_model)
        options: Generation options
        stream: Whether to stream the response
        timeout: (connect, read) timeout tuple
        retries: Number of retries for this request

    Returns:
        Chat completion response
    """
    client = get_llm_client()
    return client.chat_completion(
        messages=messages,
        model=model,
        options=options,
        stream=stream,
        timeout=timeout,
        retries=retries,
    )


def create_embedding(
    text: str,
    model: Optional[str] = None,
    timeout: Optional[tuple] = None,
    retries: Optional[int] = None,
) -> List[float]:
    """Global function to create embeddings.

    Args:
        text: Text to embed
        model: Model to use (defaults to configured emb_model)
        timeout: (connect, read) timeout tuple

    Returns:
        Embedding vector as a list of floats
    """
    client = get_llm_client()
    return client.create_embedding(text=text, model=model, timeout=timeout, retries=retries)


def check_ollama_health() -> bool:
    """Check if the configured Ollama server is accessible.

    Returns:
        True if accessible, False otherwise
    """
    client = get_llm_client()
    return client.check_health()
