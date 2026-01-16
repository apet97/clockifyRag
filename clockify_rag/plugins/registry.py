"""Plugin registry for managing and discovering plugins.

The registry provides centralized management of all plugins, with validation,
error handling, and discovery mechanisms.
"""

import logging
from typing import Any, Dict, List, Optional

from .interfaces import RetrieverPlugin, RerankPlugin, EmbeddingPlugin, IndexPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Central registry for managing plugins.

    Supports plugin registration, validation, and retrieval with
    proper error handling and logging.
    """

    def __init__(self):
        """Initialize empty plugin registry."""
        self._retrievers: Dict[str, RetrieverPlugin] = {}
        self._rerankers: Dict[str, RerankPlugin] = {}
        self._embeddings: Dict[str, EmbeddingPlugin] = {}
        self._indexes: Dict[str, IndexPlugin] = {}

    def register_retriever(self, plugin: RetrieverPlugin) -> None:
        """Register a retriever plugin.

        Args:
            plugin: RetrieverPlugin instance

        Raises:
            ValueError: If plugin validation fails
        """
        name = plugin.get_name()

        if not plugin.validate():
            raise ValueError(f"Retriever plugin '{name}' failed validation")

        if name in self._retrievers:
            logger.warning(f"Overwriting existing retriever plugin: {name}")

        self._retrievers[name] = plugin
        logger.info(f"Registered retriever plugin: {name}")

    def register_reranker(self, plugin: RerankPlugin) -> None:
        """Register a reranker plugin."""
        name = plugin.get_name()

        if not plugin.validate():
            raise ValueError(f"Reranker plugin '{name}' failed validation")

        if name in self._rerankers:
            logger.warning(f"Overwriting existing reranker plugin: {name}")

        self._rerankers[name] = plugin
        logger.info(f"Registered reranker plugin: {name}")

    def register_embedding(self, plugin: EmbeddingPlugin) -> None:
        """Register an embedding plugin."""
        name = plugin.get_name()

        if not plugin.validate():
            raise ValueError(f"Embedding plugin '{name}' failed validation")

        if name in self._embeddings:
            logger.warning(f"Overwriting existing embedding plugin: {name}")

        self._embeddings[name] = plugin
        logger.info(f"Registered embedding plugin: {name}")

    def register_index(self, plugin: IndexPlugin) -> None:
        """Register an index plugin."""
        name = plugin.get_name()

        if not plugin.validate():
            raise ValueError(f"Index plugin '{name}' failed validation")

        if name in self._indexes:
            logger.warning(f"Overwriting existing index plugin: {name}")

        self._indexes[name] = plugin
        logger.info(f"Registered index plugin: {name}")

    def get_retriever(self, name: str) -> Optional[RetrieverPlugin]:
        """Get retriever plugin by name."""
        return self._retrievers.get(name)

    def get_reranker(self, name: str) -> Optional[RerankPlugin]:
        """Get reranker plugin by name."""
        return self._rerankers.get(name)

    def get_embedding(self, name: str) -> Optional[EmbeddingPlugin]:
        """Get embedding plugin by name."""
        return self._embeddings.get(name)

    def get_index(self, name: str) -> Optional[IndexPlugin]:
        """Get index plugin by name."""
        return self._indexes.get(name)

    def list_plugins(self) -> Dict[str, List[str]]:
        """List all registered plugins by type.

        Returns:
            Dict with plugin types as keys and lists of plugin names as values
        """
        return {
            "retrievers": list(self._retrievers.keys()),
            "rerankers": list(self._rerankers.keys()),
            "embeddings": list(self._embeddings.keys()),
            "indexes": list(self._indexes.keys()),
        }


# Global plugin registry instance
_registry = PluginRegistry()


def register_plugin(plugin: Any) -> None:
    """Register a plugin in the global registry.

    Automatically detects plugin type and registers appropriately.

    Args:
        plugin: Plugin instance implementing one of the plugin interfaces

    Raises:
        TypeError: If plugin doesn't implement any known interface
    """
    if isinstance(plugin, RetrieverPlugin):
        _registry.register_retriever(plugin)
    elif isinstance(plugin, RerankPlugin):
        _registry.register_reranker(plugin)
    elif isinstance(plugin, EmbeddingPlugin):
        _registry.register_embedding(plugin)
    elif isinstance(plugin, IndexPlugin):
        _registry.register_index(plugin)
    else:
        raise TypeError(f"Unknown plugin type: {type(plugin)}")


def get_plugin(plugin_type: str, name: str) -> Optional[Any]:
    """Get a plugin by type and name.

    Args:
        plugin_type: One of 'retriever', 'reranker', 'embedding', 'index'
        name: Plugin name

    Returns:
        Plugin instance or None if not found
    """
    getters = {
        "retriever": _registry.get_retriever,
        "reranker": _registry.get_reranker,
        "embedding": _registry.get_embedding,
        "index": _registry.get_index,
    }

    getter = getters.get(plugin_type)
    if getter is None:
        raise ValueError(f"Unknown plugin type: {plugin_type}")

    return getter(name)


def list_plugins() -> Dict[str, List[str]]:
    """List all registered plugins.

    Returns:
        Dict with plugin types as keys and lists of plugin names as values
    """
    return _registry.list_plugins()
