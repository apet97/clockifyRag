"""Plugin system for Clockify RAG.

This module provides the plugin architecture for extending the RAG system
with custom retrievers, rerankers, embeddings, and indexes.

Example usage:
    from clockify_rag.plugins import RetrieverPlugin, register_plugin

    class MyRetriever(RetrieverPlugin):
        def retrieve(self, question, top_k):
            # Custom retrieval logic
            return results

    register_plugin('my_retriever', MyRetriever())
"""

from .interfaces import RetrieverPlugin, RerankPlugin, EmbeddingPlugin, IndexPlugin
from .registry import PluginRegistry, register_plugin, get_plugin, list_plugins

__all__ = [
    # Interfaces
    "RetrieverPlugin",
    "RerankPlugin",
    "EmbeddingPlugin",
    "IndexPlugin",
    # Registry
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "list_plugins",
]
