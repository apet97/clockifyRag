"""Example plugin implementations demonstrating the plugin system.

These examples show how to create custom retrievers, rerankers, embeddings,
and indexes that integrate with the Clockify RAG system.
"""

import logging
from typing import List, Tuple

import numpy as np

from .interfaces import RetrieverPlugin, RerankPlugin, EmbeddingPlugin, IndexPlugin

logger = logging.getLogger(__name__)


class SimpleRetrieverPlugin(RetrieverPlugin):
    """Example retriever plugin using keyword matching.

    This simple plugin demonstrates the RetrieverPlugin interface by
    implementing a basic keyword-based retrieval strategy.
    """

    def __init__(self, chunks_dict: dict):
        """Initialize with chunk dictionary.

        Args:
            chunks_dict: Dict mapping chunk IDs to chunk dicts
        """
        self.chunks_dict = chunks_dict
        self.chunks_list = list(chunks_dict.values())

    def retrieve(self, question: str, top_k: int = 12) -> List[dict]:
        """Retrieve chunks using simple keyword matching.

        Args:
            question: User query
            top_k: Number of results to return

        Returns:
            List of chunk dicts with 'id', 'text', and 'score'
        """
        from ..utils import tokenize

        # Tokenize question
        q_tokens = set(tokenize(question))

        # Score each chunk by keyword overlap
        scores = []
        for chunk in self.chunks_list:
            chunk_tokens = set(tokenize(chunk["text"]))
            overlap = len(q_tokens & chunk_tokens)
            scores.append((chunk, overlap))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for chunk, score in scores[:top_k]:
            results.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "title": chunk.get("title", ""),
                    "section": chunk.get("section", ""),
                    "score": float(score),
                }
            )

        return results

    def get_name(self) -> str:
        """Return plugin name."""
        return "simple_keyword"


class MMRRerankPlugin(RerankPlugin):
    """Example reranker using Maximal Marginal Relevance (MMR).

    This plugin demonstrates reranking to increase diversity in results.
    """

    def __init__(self, lambda_param: float = 0.7):
        """Initialize MMR reranker.

        Args:
            lambda_param: Trade-off between relevance and diversity (0-1)
        """
        self.lambda_param = lambda_param

    def rerank(self, question: str, chunks: List[dict], scores: List[float]) -> Tuple[List[dict], List[float]]:
        """Rerank using MMR to maximize diversity.

        Args:
            question: User query
            chunks: Retrieved chunks
            scores: Initial retrieval scores

        Returns:
            Tuple of (reranked_chunks, reranked_scores)
        """
        if len(chunks) <= 1:
            return chunks, scores

        # Simple MMR: select diverse chunks
        # In practice, this would use actual embeddings
        selected = [0]  # Start with highest scoring
        remaining = list(range(1, len(chunks)))

        while len(selected) < len(chunks) and remaining:
            # Select next chunk that is different from selected ones
            # Simplified: just alternate to create diversity
            if remaining:
                next_idx = remaining.pop(0)
                selected.append(next_idx)

        # Reorder based on selection
        reranked_chunks = [chunks[i] for i in selected]
        reranked_scores = [scores[i] for i in selected]

        return reranked_chunks, reranked_scores

    def get_name(self) -> str:
        """Return plugin name."""
        return "mmr_reranker"


class RandomEmbeddingPlugin(EmbeddingPlugin):
    """Example embedding plugin using random vectors.

    WARNING: This is for demonstration only. Do not use in production!
    Shows how to implement custom embedding models.
    """

    def __init__(self, dimension: int = 384):
        """Initialize random embedding generator.

        Args:
            dimension: Embedding dimensionality
        """
        self.dimension = dimension
        logger.warning("RandomEmbeddingPlugin is for demonstration only!")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate random embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            Random embedding vectors
        """
        # Generate random normalized vectors
        vecs = np.random.randn(len(texts), self.dimension).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vecs / norms

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension

    def get_name(self) -> str:
        """Return plugin name."""
        return "random_embedding"


class LinearScanIndexPlugin(IndexPlugin):
    """Example index plugin using simple linear scan.

    This demonstrates the IndexPlugin interface with a basic
    brute-force similarity search implementation.
    """

    def __init__(self):
        """Initialize empty index."""
        self.vectors = None
        self.metadata = None

    def build(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        """Build index by storing vectors.

        Args:
            vectors: Embedding vectors
            metadata: Chunk metadata
        """
        self.vectors = vectors.astype(np.float32)
        self.metadata = metadata
        logger.info(f"Built linear scan index with {len(vectors)} vectors")

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        """Search using brute-force cosine similarity.

        Args:
            query_vector: Query embedding
            top_k: Number of results

        Returns:
            Tuple of (indices, scores)
        """
        if self.vectors is None:
            return [], []

        # Compute cosine similarities
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        scores = np.dot(self.vectors, query_norm)

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]

        return top_indices.tolist(), top_scores.tolist()

    def save(self, path: str) -> None:
        """Save index to disk."""
        if self.vectors is not None:
            np.save(path + ".vectors.npy", self.vectors)
        logger.info(f"Saved index to {path}")

    def load(self, path: str) -> None:
        """Load index from disk."""
        self.vectors = np.load(path + ".vectors.npy")
        logger.info(f"Loaded index from {path}")

    def get_name(self) -> str:
        """Return plugin name."""
        return "linear_scan"
