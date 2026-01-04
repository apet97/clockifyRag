"""Abstract base classes for plugin interfaces.

These interfaces define the contracts that plugins must implement to integrate
with the Clockify RAG system.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class RetrieverPlugin(ABC):
    """Plugin interface for custom retrieval strategies.

    Implement this interface to create custom retrieval methods that can
    replace or augment the default hybrid retrieval.
    """

    @abstractmethod
    def retrieve(self, question: str, top_k: int = 12) -> List[dict]:
        """Retrieve relevant chunks for a question.

        Args:
            question: User query string
            top_k: Number of chunks to retrieve

        Returns:
            List of chunk dicts with 'id', 'text', and 'score' fields
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return plugin name for registration."""
        pass

    def validate(self) -> bool:
        """Validate plugin configuration. Override if needed."""
        return True


class RerankPlugin(ABC):
    """Plugin interface for custom reranking algorithms.

    Implement this interface to create custom reranking strategies that can
    improve retrieval quality by reordering initial results.
    """

    @abstractmethod
    def rerank(self, question: str, chunks: List[dict], scores: List[float]) -> Tuple[List[dict], List[float]]:
        """Rerank retrieved chunks.

        Args:
            question: User query string
            chunks: List of retrieved chunk dicts
            scores: Initial retrieval scores

        Returns:
            Tuple of (reranked_chunks, reranked_scores)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return plugin name for registration."""
        pass

    def validate(self) -> bool:
        """Validate plugin configuration. Override if needed."""
        return True


class EmbeddingPlugin(ABC):
    """Plugin interface for custom embedding models.

    Implement this interface to use custom embedding models beyond the
    default SentenceTransformer or Ollama embeddings.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> Any:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimensionality."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return plugin name for registration."""
        pass

    def validate(self) -> bool:
        """Validate plugin configuration. Override if needed."""
        return True


class IndexPlugin(ABC):
    """Plugin interface for custom index types.

    Implement this interface to use custom index structures beyond the
    default FAISS and BM25 indexes.
    """

    @abstractmethod
    def build(self, vectors: Any, metadata: List[dict]) -> None:
        """Build index from vectors and metadata.

        Args:
            vectors: NumPy array of embeddings
            metadata: List of chunk metadata dicts
        """
        pass

    @abstractmethod
    def search(self, query_vector: Any, top_k: int) -> Tuple[List[int], List[float]]:
        """Search index for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            Tuple of (indices, scores) for top-k results
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return plugin name for registration."""
        pass

    def validate(self) -> bool:
        """Validate plugin configuration. Override if needed."""
        return True
