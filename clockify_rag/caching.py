"""Query caching and rate limiting for RAG system."""

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from typing import Optional

from .metrics import MetricNames, increment_counter, set_gauge

logger = logging.getLogger(__name__)

# FIX (Error #2): Declare globals at module level for safe initialization
_RATE_LIMITER = None
_QUERY_CACHE = None


class RateLimiter:
    """Simple thread-safe sliding-window rate limiter."""

    def __init__(self, max_requests=10, window_seconds=60):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.RLock()

    def allow_request(self) -> bool:
        """Return True if request is allowed under the window."""
        now = time.monotonic()
        with self._lock:
            window_start = now - self.window_seconds
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_requests:
                return False
            self._timestamps.append(now)
            return True

    def wait_time(self) -> float:
        """Return seconds until the next request is allowed (0 if available)."""
        now = time.monotonic()
        with self._lock:
            if len(self._timestamps) < self.max_requests:
                return 0.0
            earliest = self._timestamps[0]
            return max(0.0, (earliest + self.window_seconds) - now)


# Global rate limiter (10 queries per minute by default)
def get_rate_limiter():
    """Get global rate limiter instance.

    FIX (Error #2): Use proper `is None` check instead of fragile globals() check.
    """
    from . import config  # Import here to avoid circular import

    if not getattr(config, "RATE_LIMIT_ENABLED", False):
        return None

    global _RATE_LIMITER
    if _RATE_LIMITER is None:
        _RATE_LIMITER = RateLimiter(max_requests=config.RATE_LIMIT_REQUESTS, window_seconds=config.RATE_LIMIT_WINDOW)
    return _RATE_LIMITER


class QueryCache:
    """TTL-based cache for repeated queries to eliminate redundant computation."""

    def __init__(self, maxsize=100, ttl_seconds=3600):
        """Initialize query cache.

        Args:
            maxsize: Maximum number of cached queries (LRU eviction)
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        # PERF FIX: Use OrderedDict for O(1) LRU operations instead of dict + deque
        # OrderedDict.move_to_end() is O(1) vs deque.remove() which is O(n)
        self._cache: OrderedDict = OrderedDict()  # {question_hash: (answer, metadata_with_timestamp, timestamp)}
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()  # Thread safety lock

    @property
    def cache(self) -> dict:
        """Backwards-compatible access to cache dict."""
        return self._cache

    def _hash_question(self, question: str, params: Optional[dict] = None) -> str:
        """Generate cache key from question and retrieval parameters.

        Args:
            question: User question
            params: Retrieval parameters (top_k, pack_top, use_rerank, threshold)
        """
        if params is None:
            cache_input = question
        else:
            # Sort params for consistent hashing
            sorted_params = sorted(params.items())
            cache_input = question + str(sorted_params)
        return hashlib.md5(cache_input.encode("utf-8")).hexdigest()

    def get(self, question: str, params: Optional[dict] = None):
        """Retrieve cached answer if available and not expired.

        Args:
            question: User question
            params: Retrieval parameters (optional, for cache key)

        Returns:
            (answer, metadata) tuple if cache hit, None if cache miss
        """
        with self._lock:
            key = self._hash_question(question, params)

            if key not in self._cache:
                self.misses += 1
                increment_counter(MetricNames.CACHE_MISSES)
                return None

            answer, metadata, timestamp = self._cache[key]
            # Ensure metadata exposes cache timestamp for downstream logging
            metadata_timestamp = metadata.get("timestamp")
            if metadata_timestamp is None:
                metadata_timestamp = timestamp
                metadata["timestamp"] = metadata_timestamp

            age = time.time() - metadata_timestamp

            # Check if expired
            if age > self.ttl_seconds:
                del self._cache[key]
                self.misses += 1
                increment_counter(MetricNames.CACHE_MISSES)
                return None

            # PERF FIX: O(1) move_to_end() instead of O(n) remove() + append()
            self._cache.move_to_end(key)
            self.hits += 1
            increment_counter(MetricNames.CACHE_HITS)
            logger.debug(f"[cache] HIT question_hash={key[:8]} age={age:.1f}s")
            return answer, metadata

    def put(self, question: str, answer: str, metadata: dict, params: Optional[dict] = None):
        """Store answer in cache.

        Args:
            question: User question
            answer: Generated answer
            metadata: Answer metadata (selected chunks, scores, etc.)
            params: Retrieval parameters (optional, for cache key)
        """
        with self._lock:
            key = self._hash_question(question, params)

            # PERF FIX: O(1) eviction using OrderedDict
            # Evict oldest entry if cache full (oldest is first in OrderedDict)
            if len(self._cache) >= self.maxsize and key not in self._cache:
                oldest, _ = self._cache.popitem(last=False)  # O(1) pop from front
                logger.debug(f"[cache] EVICT question_hash={oldest[:8]} (LRU)")

            # Store entry with timestamp
            # FIX: Deep copy metadata to prevent mutation leaks
            import copy

            timestamp = time.time()
            metadata_copy = copy.deepcopy(metadata) if metadata is not None else {}
            metadata_copy["timestamp"] = timestamp

            # PERF FIX: O(1) update - if key exists, move_to_end; otherwise just add
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (answer, metadata_copy, timestamp)

            logger.debug(f"[cache] PUT question_hash={key[:8]}")
            set_gauge(MetricNames.CACHE_SIZE, len(self._cache))

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("[cache] CLEAR")
            set_gauge(MetricNames.CACHE_SIZE, 0)

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, size, hit_rate
        """
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hit_rate": hit_rate,
            }

    def save(self, path: str = "query_cache.json"):
        """Save cache to disk for persistence across sessions.

        OPTIMIZATION: Enables 100% cache hit rate on repeated queries after restart.

        Args:
            path: File path to save cache (default: query_cache.json)
        """
        import json

        with self._lock:
            try:
                # PERF FIX: OrderedDict maintains order, so we save entries in LRU order
                cache_data = {
                    "version": "1.1",  # Bumped version for new format without access_order
                    "maxsize": self.maxsize,
                    "ttl_seconds": self.ttl_seconds,
                    "entries": [
                        {"key": key, "answer": answer, "metadata": metadata, "timestamp": timestamp}
                        for key, (answer, metadata, timestamp) in self._cache.items()
                    ],
                    "hits": self.hits,
                    "misses": self.misses,
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                logger.info(f"[cache] SAVE {len(self._cache)} entries to {path}")
            except Exception as e:
                logger.warning(f"[cache] Failed to save cache: {e}")

    def load(self, path: str = "query_cache.json"):
        """Load cache from disk to restore across sessions.

        OPTIMIZATION: Restores previous session's cache for instant hits on repeated queries.

        Args:
            path: File path to load cache from (default: query_cache.json)

        Returns:
            Number of entries loaded (0 if file doesn't exist or load fails)
        """
        import json

        with self._lock:
            if not os.path.exists(path):
                logger.debug(f"[cache] No cache file found at {path}")
                return 0

            try:
                with open(path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                # Validate version - support both 1.0 and 1.1 formats
                version = cache_data.get("version", "1.0")
                if version not in ("1.0", "1.1"):
                    logger.warning(f"[cache] Incompatible cache version {version}, skipping load")
                    return 0

                # Restore entries, filtering out expired ones
                # PERF FIX: OrderedDict maintains insertion order for LRU
                now = time.time()
                loaded_count = 0

                # For v1.0, we had access_order; for v1.1, entries are already in LRU order
                entries = cache_data.get("entries", [])
                if version == "1.0":
                    # Reorder entries based on access_order from old format
                    access_order = cache_data.get("access_order", [])
                    entry_map = {e["key"]: e for e in entries}
                    entries = [entry_map[k] for k in access_order if k in entry_map]

                for entry in entries:
                    key = entry["key"]
                    answer = entry["answer"]
                    metadata = entry["metadata"]
                    timestamp = entry["timestamp"]

                    # Skip expired entries
                    age = now - timestamp
                    if age > self.ttl_seconds:
                        continue

                    self._cache[key] = (answer, metadata, timestamp)
                    loaded_count += 1

                # Restore stats (reset to avoid inflated numbers from old sessions)
                # self.hits = cache_data.get("hits", 0)
                # self.misses = cache_data.get("misses", 0)

                logger.info(
                    f"[cache] LOAD {loaded_count} entries from {path} (skipped {len(cache_data.get('entries', [])) - loaded_count} expired)"
                )
                return loaded_count

            except Exception as e:
                logger.warning(f"[cache] Failed to load cache: {e}")
                return 0


# Global query cache (100 entries, 1 hour TTL by default)
def get_query_cache():
    """Get global query cache instance.

    FIX (Error #2): Use proper `is None` check instead of fragile globals() check.
    """
    from . import config  # Import here to avoid circular import

    global _QUERY_CACHE
    if _QUERY_CACHE is None:
        _QUERY_CACHE = QueryCache(maxsize=config.CACHE_MAXSIZE, ttl_seconds=config.CACHE_TTL)
    return _QUERY_CACHE


def log_query(
    query: str,
    answer: str,
    retrieved_chunks: list,
    latency_ms: float,
    refused: bool = False,
    metadata: Optional[dict] = None,
):
    """Log query with structured JSON format for monitoring and analytics.

    Uses rotating file handler to prevent unbounded disk usage.
    FIX (Error #6): Sanitizes user input to prevent log injection attacks.
    """
    from . import config as _config

    if not getattr(_config, "QUERY_LOG_ENABLED", False):
        return

    import json
    from .config import (
        LOG_QUERY_ANSWER_PLACEHOLDER,
        LOG_QUERY_INCLUDE_ANSWER,
        LOG_QUERY_INCLUDE_CHUNKS,
        QUERY_LOG_FILE,
    )
    from .correlation import get_correlation_id
    from .logging_config import get_query_logger
    from .utils import sanitize_for_log

    normalized_chunks = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            normalized = chunk.copy()
            chunk_id = normalized.get("id") or normalized.get("chunk_id")
            normalized["id"] = chunk_id
            normalized["dense"] = float(normalized.get("dense", normalized.get("score", 0.0)) or 0.0)
            normalized["bm25"] = float(normalized.get("bm25", 0.0) or 0.0)
            normalized["hybrid"] = float(normalized.get("hybrid", normalized["dense"]) or 0.0)
            # Redact chunk text for security/privacy unless explicitly enabled
            if not LOG_QUERY_INCLUDE_CHUNKS:
                normalized.pop("chunk", None)  # Remove full chunk text
                normalized.pop("text", None)  # Remove text field if present
        else:
            normalized = {
                "id": chunk,
                "dense": 0.0,
                "bm25": 0.0,
                "hybrid": 0.0,
            }
        normalized_chunks.append(normalized)

    chunk_ids = [c.get("id") for c in normalized_chunks]
    dense_scores = [c.get("dense", 0.0) for c in normalized_chunks]
    bm25_scores = [c.get("bm25", 0.0) for c in normalized_chunks]
    hybrid_scores = [c.get("hybrid", 0.0) for c in normalized_chunks]
    primary_scores = hybrid_scores if hybrid_scores else []
    avg_chunk_score = (sum(primary_scores) / len(primary_scores)) if primary_scores else 0.0
    max_chunk_score = max(primary_scores) if primary_scores else 0.0

    # FIX: Sanitize metadata to prevent chunk text leaks
    # Deep copy and remove any 'text'/'chunk' fields from nested structures
    import copy

    sanitized_metadata = copy.deepcopy(metadata) if metadata else {}
    if not LOG_QUERY_INCLUDE_CHUNKS and isinstance(sanitized_metadata, dict):
        # Remove chunk text from any nested chunk dicts in metadata
        for key in list(sanitized_metadata.keys()):
            val = sanitized_metadata[key]
            if isinstance(val, dict):
                val.pop("text", None)
                val.pop("chunk", None)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        item.pop("text", None)
                        item.pop("chunk", None)

    # FIX (Error #6): Sanitize query and answer to prevent log injection
    log_entry = {
        "correlation_id": get_correlation_id() or "-",
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": sanitize_for_log(query, max_length=2000),
        "refused": refused,
        "latency_ms": latency_ms,
        "num_chunks_retrieved": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "chunk_scores": {
            "dense": dense_scores,
            "bm25": bm25_scores,
            "hybrid": hybrid_scores,
        },
        "retrieved_chunks": normalized_chunks,
        "avg_chunk_score": avg_chunk_score,
        "max_chunk_score": max_chunk_score,
        "metadata": sanitized_metadata,
    }

    if LOG_QUERY_INCLUDE_ANSWER:
        log_entry["answer"] = sanitize_for_log(answer, max_length=5000)
    elif LOG_QUERY_ANSWER_PLACEHOLDER:
        log_entry["answer"] = LOG_QUERY_ANSWER_PLACEHOLDER

    try:
        # PERF FIX: Use rotating logger to prevent unbounded disk usage
        # Default: 10MB max file size, keeps 5 backups (rag_queries.jsonl.1, .2, etc.)
        from .logging_config import flush_query_logger

        # Reset logger if file path changed (e.g., in tests)
        # This ensures we get a fresh logger with the correct path
        query_logger = get_query_logger(
            log_file=QUERY_LOG_FILE,
            max_bytes=10 * 1024 * 1024,  # 10MB
            backup_count=5,
        )
        query_logger.info(json.dumps(log_entry, ensure_ascii=False))
        # Flush immediately to ensure log is written (important for tests)
        flush_query_logger()
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")
