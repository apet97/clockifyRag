"""Custom exceptions for Clockify RAG system."""


class EmbeddingError(Exception):
    """Embedding generation failed."""

    pass


class LLMError(Exception):
    """LLM call failed."""

    pass


class LLMUnavailableError(LLMError):
    """LLM endpoint is unreachable or timed out."""

    pass


class LLMBadResponseError(LLMError):
    """LLM returned a malformed or incomplete payload."""

    pass


class IndexLoadError(Exception):
    """Index loading or validation failed."""

    pass


class BuildError(Exception):
    """Knowledge base build failed."""

    pass


class ValidationError(Exception):
    """Input validation failed.

    FIX (Error #5): Added for query length validation to prevent DoS attacks.
    """

    pass
