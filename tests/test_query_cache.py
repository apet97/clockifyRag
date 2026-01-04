"""Tests for query caching functionality."""

import pytest
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.caching import QueryCache


class TestQueryCache:
    """Test TTL-based query cache with LRU eviction."""

    def setup_method(self):
        """Setup fresh cache for each test."""
        self.cache = QueryCache(maxsize=3, ttl_seconds=1)  # Small cache for testing

    def test_cache_miss_on_first_access(self):
        """Test that first access is a cache miss."""
        result = self.cache.get("What is Clockify?")
        assert result is None
        assert self.cache.stats()["misses"] == 1
        assert self.cache.stats()["hits"] == 0

    def test_cache_hit_on_second_access(self):
        """Test that second access with same question is a cache hit."""
        question = "What is Clockify?"
        answer = "Clockify is a time tracking tool."
        metadata = {"selected": [1, 2, 3]}

        # First access: miss, then put
        self.cache.get(question)  # miss
        before_put = time.time()
        self.cache.put(question, answer, metadata)

        # Second access: hit
        result = self.cache.get(question)
        assert result is not None
        cached_answer, cached_metadata = result
        assert cached_answer == answer
        assert cached_metadata["selected"] == metadata["selected"]
        assert "timestamp" in cached_metadata
        assert "timestamp" not in metadata  # Original metadata should remain untouched
        after_get = time.time()
        assert before_put <= cached_metadata["timestamp"] <= after_get
        assert self.cache.stats()["hits"] == 1
        assert self.cache.stats()["misses"] == 1

    def test_cache_expires_after_ttl(self):
        """Test that cache entries expire after TTL."""
        question = "What is Clockify?"
        answer = "Clockify is a time tracking tool."
        metadata = {"selected": [1, 2, 3]}

        # Put entry in cache
        self.cache.put(question, answer, metadata)

        # Immediate access: hit
        result = self.cache.get(question)
        assert result is not None

        # Wait for TTL expiration
        time.sleep(1.1)

        # Access after TTL: miss
        result = self.cache.get(question)
        assert result is None
        assert self.cache.stats()["misses"] == 1  # Only the expired miss

    def test_lru_eviction(self):
        """Test that LRU eviction works when cache is full."""
        # Fill cache to capacity (maxsize=3)
        self.cache.put("q1", "a1", {})
        self.cache.put("q2", "a2", {})
        self.cache.put("q3", "a3", {})
        assert self.cache.stats()["size"] == 3

        # Add one more: should evict oldest (q1)
        self.cache.put("q4", "a4", {})
        assert self.cache.stats()["size"] == 3

        # q1 should be evicted
        result = self.cache.get("q1")
        assert result is None

        # q2, q3, q4 should still be in cache
        assert self.cache.get("q2") is not None
        assert self.cache.get("q3") is not None
        assert self.cache.get("q4") is not None

    def test_lru_access_updates_order(self):
        """Test that accessing an entry updates its LRU position."""
        # Fill cache
        self.cache.put("q1", "a1", {})
        self.cache.put("q2", "a2", {})
        self.cache.put("q3", "a3", {})

        # Access q1 to move it to end of LRU
        self.cache.get("q1")

        # Add q4: should evict q2 (oldest unaccessed)
        self.cache.put("q4", "a4", {})

        # q1 should still be in cache (was accessed)
        assert self.cache.get("q1") is not None

        # q2 should be evicted
        assert self.cache.get("q2") is None

    def test_clear_resets_cache(self):
        """Test that clear() removes all entries and resets stats."""
        self.cache.put("q1", "a1", {})
        self.cache.put("q2", "a2", {})
        self.cache.get("q1")  # hit
        self.cache.get("q3")  # miss

        assert self.cache.stats()["size"] == 2
        assert self.cache.stats()["hits"] == 1
        assert self.cache.stats()["misses"] == 1

        self.cache.clear()

        assert self.cache.stats()["size"] == 0
        assert self.cache.stats()["hits"] == 0
        assert self.cache.stats()["misses"] == 0

    def test_stats_hit_rate(self):
        """Test that hit_rate is calculated correctly."""
        self.cache.put("q1", "a1", {})

        # 1 hit, 1 miss -> 50% hit rate
        self.cache.get("q1")  # hit
        self.cache.get("q2")  # miss

        stats = self.cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_stats_hit_rate_no_requests(self):
        """Test hit_rate is 0 when no requests made."""
        stats = self.cache.stats()
        assert stats["hit_rate"] == 0.0

    def test_cache_handles_duplicate_puts(self):
        """Test that putting same question twice updates entry."""
        question = "What is Clockify?"

        self.cache.put(question, "answer1", {"v": 1})
        time.sleep(0.01)
        self.cache.put(question, "answer2", {"v": 2})

        # Should only have 1 entry (updated)
        assert self.cache.stats()["size"] == 1

        result = self.cache.get(question)
        assert result is not None
        answer, metadata = result
        assert answer == "answer2"
        assert metadata["v"] == 2

    def test_cache_hash_consistency(self):
        """Test that same question produces same hash."""
        question = "What is Clockify?"
        self.cache.put(question, "answer", {})

        # Same question should hit cache
        result = self.cache.get(question)
        assert result is not None

    def test_cache_case_sensitive(self):
        """Test that cache is case-sensitive."""
        self.cache.put("What is Clockify?", "answer1", {})
        self.cache.put("what is clockify?", "answer2", {})

        # Should be two separate entries
        assert self.cache.stats()["size"] == 2

        result1 = self.cache.get("What is Clockify?")
        result2 = self.cache.get("what is clockify?")

        assert result1[0] == "answer1"
        assert result2[0] == "answer2"

    def test_metadata_timestamp_age_progresses(self):
        """Metadata should expose timestamp for age calculations."""
        question = "How old is the cache entry?"
        metadata = {"selected": []}
        self.cache.put(question, "age-answer", metadata)

        time.sleep(0.2)

        result = self.cache.get(question)
        assert result is not None
        _, cached_metadata = result
        assert "timestamp" in cached_metadata

        age = time.time() - cached_metadata["timestamp"]
        assert age >= 0.2
        # TTL is 1 second, so the entry should still be valid and age less than TTL
        assert age < 1.0

    def test_cache_respects_retrieval_params(self):
        """Test that cache differentiates queries with different retrieval parameters.

        Regression test for Priority #9: Cache should invalidate when retrieval
        parameters change (top_k, pack_top, threshold, use_rerank).
        Without parameter-aware caching, the same query with different parameters
        would return stale answers from previous parameter settings.
        """
        question = "How do I track time?"

        # Same question, different retrieval parameters
        params1 = {"top_k": 12, "pack_top": 6, "threshold": 0.30, "use_rerank": True}
        params2 = {"top_k": 20, "pack_top": 10, "threshold": 0.40, "use_rerank": False}
        params3 = {"top_k": 12, "pack_top": 6, "threshold": 0.30, "use_rerank": True}  # Same as params1

        # Put with params1
        self.cache.put(question, "answer_with_params1", {"config": "params1"}, params=params1)

        # Get with params1: should hit
        result1 = self.cache.get(question, params=params1)
        assert result1 is not None, "Cache should hit with same params"
        answer1, metadata1 = result1
        assert answer1 == "answer_with_params1"
        assert metadata1["config"] == "params1"

        # Get with params2: should MISS (different params)
        result2 = self.cache.get(question, params=params2)
        assert result2 is None, "Cache should miss when retrieval params differ"

        # Put with params2
        self.cache.put(question, "answer_with_params2", {"config": "params2"}, params=params2)

        # Get with params2: should hit now
        result2 = self.cache.get(question, params=params2)
        assert result2 is not None, "Cache should hit after putting with params2"
        answer2, metadata2 = result2
        assert answer2 == "answer_with_params2"
        assert metadata2["config"] == "params2"

        # Get with params1 again: should still hit (params1 entry still in cache)
        result1_again = self.cache.get(question, params=params1)
        assert result1_again is not None, "Original params1 entry should still be in cache"
        answer1_again, _ = result1_again
        assert answer1_again == "answer_with_params1"

        # Get with params3 (identical to params1): should hit params1 entry
        result3 = self.cache.get(question, params=params3)
        assert result3 is not None, "Identical params should produce same cache key"
        answer3, _ = result3
        assert answer3 == "answer_with_params1"

        # Cache should have 2 distinct entries (params1 and params2)
        assert self.cache.stats()["size"] == 2

    def test_cache_params_order_independence(self):
        """Test that parameter order doesn't affect cache key.

        Cache implementation should sort params for consistent hashing,
        so {"a": 1, "b": 2} and {"b": 2, "a": 1} produce the same cache key.
        """
        question = "What are the pricing plans?"
        params_ordered = {"top_k": 12, "pack_top": 6, "threshold": 0.30}
        params_reordered = {"threshold": 0.30, "top_k": 12, "pack_top": 6}

        # Put with params_ordered
        self.cache.put(question, "answer", {"test": "ordered"}, params=params_ordered)

        # Get with params_reordered: should hit (same params, different order)
        result = self.cache.get(question, params=params_reordered)
        assert result is not None, "Cache should hit regardless of param dict order"
        answer, metadata = result
        assert answer == "answer"
        assert metadata["test"] == "ordered"

        # Should only have 1 entry
        assert self.cache.stats()["size"] == 1

    def test_cache_without_params_isolated(self):
        """Test that queries without params are cached separately from queries with params.

        get(q) and get(q, params={...}) should be separate cache entries.
        """
        question = "Can I track time offline?"
        params = {"top_k": 12, "pack_top": 6}

        # Put without params
        self.cache.put(question, "answer_no_params", {"variant": "no_params"})

        # Put with params
        self.cache.put(question, "answer_with_params", {"variant": "with_params"}, params=params)

        # Should have 2 separate entries
        assert self.cache.stats()["size"] == 2

        # Get without params: should return no_params answer
        result_no_params = self.cache.get(question)
        assert result_no_params is not None
        answer_no_params, metadata_no_params = result_no_params
        assert answer_no_params == "answer_no_params"
        assert metadata_no_params["variant"] == "no_params"

        # Get with params: should return with_params answer
        result_with_params = self.cache.get(question, params=params)
        assert result_with_params is not None
        answer_with_params, metadata_with_params = result_with_params
        assert answer_with_params == "answer_with_params"
        assert metadata_with_params["variant"] == "with_params"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
