"""Tests for metrics collection accuracy and performance."""

import pytest
import time
from clockify_rag.metrics import (
    MetricsCollector,
    get_metrics,
    increment_counter,
    set_gauge,
    observe_histogram,
    time_operation,
)


@pytest.fixture
def fresh_metrics():
    """Create a fresh metrics collector for each test."""
    collector = MetricsCollector()
    # Reset singleton state (if using global)
    import clockify_rag.metrics as metrics_module

    old_collector = metrics_module._METRICS
    metrics_module._METRICS = collector
    yield collector
    metrics_module._METRICS = old_collector


def test_counter_accuracy(fresh_metrics):
    """Verify counters reflect actual increments."""
    collector = fresh_metrics

    # Increment counter multiple times
    collector.increment_counter("test_counter")
    collector.increment_counter("test_counter")
    collector.increment_counter("test_counter", value=3)

    snapshot = collector.get_snapshot()
    assert snapshot.counters.get("test_counter", 0) == 5, "Counter should sum to 5 (1+1+3)"


def test_gauge_accuracy(fresh_metrics):
    """Verify gauges store the latest value."""
    collector = fresh_metrics

    # Set gauge values
    collector.set_gauge("test_gauge", 10)
    collector.set_gauge("test_gauge", 20)  # Overwrite
    collector.set_gauge("test_gauge", 15)  # Overwrite again

    snapshot = collector.get_snapshot()
    assert snapshot.gauges.get("test_gauge") == 15, "Gauge should store latest value (15)"


def test_histogram_accuracy(fresh_metrics):
    """Verify histograms compute correct statistics."""
    collector = fresh_metrics

    # Observe values: 100, 200, 150
    collector.observe_histogram("test_hist", 100)
    collector.observe_histogram("test_hist", 200)
    collector.observe_histogram("test_hist", 150)

    stats = collector.get_histogram_stats("test_hist")

    assert stats["count"] == 3, "Should have 3 observations"
    assert stats["mean"] == 150, f"Mean should be 150, got {stats['mean']}"
    assert stats["min"] == 100, f"Min should be 100, got {stats['min']}"
    assert stats["max"] == 200, f"Max should be 200, got {stats['max']}"


def test_histogram_percentiles(fresh_metrics):
    """Verify histogram percentile calculations."""
    collector = fresh_metrics

    # Add 100 values from 1 to 100
    for i in range(1, 101):
        collector.observe_histogram("percentile_test", i)

    stats = collector.get_histogram_stats("percentile_test")

    # Check percentiles (approximate due to implementation)
    assert 48 <= stats["p50"] <= 52, f"P50 should be ~50, got {stats['p50']}"
    assert 93 <= stats["p95"] <= 97, f"P95 should be ~95, got {stats['p95']}"
    assert 98 <= stats["p99"] <= 100, f"P99 should be ~99, got {stats['p99']}"


def test_multiple_metrics_isolated(fresh_metrics):
    """Verify different metrics don't interfere."""
    collector = fresh_metrics

    # Create multiple metrics
    collector.increment_counter("counter_a")
    collector.increment_counter("counter_b")
    collector.set_gauge("gauge_a", 10)
    collector.set_gauge("gauge_b", 20)
    collector.observe_histogram("hist_a", 100)
    collector.observe_histogram("hist_b", 200)

    snapshot = collector.get_snapshot()

    # Each metric should be independent
    assert snapshot.counters.get("counter_a") == 1
    assert snapshot.counters.get("counter_b") == 1
    assert snapshot.gauges.get("gauge_a") == 10
    assert snapshot.gauges.get("gauge_b") == 20

    hist_a_stats = collector.get_histogram_stats("hist_a")
    hist_b_stats = collector.get_histogram_stats("hist_b")
    assert hist_a_stats["mean"] == 100
    assert hist_b_stats["mean"] == 200


def test_time_operation_decorator(fresh_metrics):
    """Verify time_operation decorator records elapsed time."""
    collector = fresh_metrics

    @time_operation("test_operation", collector)
    def slow_function():
        """Simulate slow operation."""
        time.sleep(0.1)  # 100ms
        return "done"

    result = slow_function()

    assert result == "done", "Function should return correct value"

    stats = collector.get_histogram_stats("test_operation")
    assert stats["count"] == 1, "Should have 1 timing measurement"
    assert stats["mean"] >= 100, f"Mean should be >= 100ms, got {stats['mean']}"
    # CI can be slow; allow a generous buffer over the nominal 100ms sleep.
    assert stats["mean"] < 300, f"Mean should be < 300ms (generous buffer for CI), got {stats['mean']}"


def test_global_metrics_singleton():
    """Verify global get_metrics() returns singleton."""
    collector1 = get_metrics()
    collector2 = get_metrics()

    assert collector1 is collector2, "get_metrics() should return same instance"


def test_metrics_helper_functions():
    """Verify convenience functions work correctly."""
    # Reset metrics
    collector = get_metrics()

    # Use helper functions
    increment_counter("helper_counter")
    increment_counter("helper_counter", value=2)
    set_gauge("helper_gauge", 42)
    observe_histogram("helper_hist", 100)

    snapshot = collector.get_snapshot()

    assert snapshot.counters.get("helper_counter") == 3
    assert snapshot.gauges.get("helper_gauge") == 42
    stats = collector.get_histogram_stats("helper_hist")
    assert stats["count"] == 1
    assert stats["mean"] == 100


def test_metrics_thread_safety_basic():
    """Basic thread safety test for metrics."""
    import threading

    collector = MetricsCollector()

    def increment_many():
        """Increment counter 1000 times."""
        for _ in range(1000):
            collector.increment_counter("thread_test")

    # Run in 4 threads
    threads = [threading.Thread(target=increment_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snapshot = collector.get_snapshot()
    # Should be exactly 4000 if thread-safe
    assert snapshot.counters.get("thread_test") == 4000, "Counter should handle concurrent increments"


def test_aggregated_metrics():
    """Test aggregated metrics snapshot functionality."""
    collector = MetricsCollector()

    # Add some metrics
    collector.increment_counter("requests", value=100)
    collector.increment_counter("errors", value=5)
    collector.set_gauge("active_connections", 42)
    collector.observe_histogram("latency_ms", 150)
    collector.observe_histogram("latency_ms", 200)

    # Get snapshot
    snapshot = collector.get_snapshot()

    # Verify snapshot structure
    assert snapshot.timestamp is not None
    assert isinstance(snapshot.counters, dict)
    assert isinstance(snapshot.gauges, dict)
    assert isinstance(snapshot.histograms, dict)

    # Verify values
    assert snapshot.counters["requests"] == 100
    assert snapshot.counters["errors"] == 5
    assert snapshot.gauges["active_connections"] == 42


def test_metrics_reset():
    """Test that metrics can be reset."""
    collector = MetricsCollector()

    # Add metrics
    collector.increment_counter("test")
    collector.set_gauge("test", 10)
    collector.observe_histogram("test", 100)

    # Reset
    collector.reset()

    # Verify all metrics cleared
    snapshot = collector.get_snapshot()
    assert len(snapshot.counters) == 0
    assert len(snapshot.gauges) == 0
    assert len(snapshot.histograms) == 0


@pytest.mark.parametrize("value", [1, 10, 100, 1000, 10000])
def test_histogram_with_different_scales(fresh_metrics, value):
    """Test histogram handles different value scales correctly."""
    collector = fresh_metrics

    collector.observe_histogram("scale_test", value)
    stats = collector.get_histogram_stats("scale_test")

    assert stats["count"] == 1
    assert stats["mean"] == value
    assert stats["min"] == value
    assert stats["max"] == value


def test_zero_and_negative_values(fresh_metrics):
    """Test metrics handle edge cases correctly."""
    collector = fresh_metrics

    # Zero values
    collector.observe_histogram("zero_test", 0)
    stats = collector.get_histogram_stats("zero_test")
    assert stats["mean"] == 0

    # Negative values (if supported)
    collector.observe_histogram("negative_test", -10)
    stats = collector.get_histogram_stats("negative_test")
    assert stats["mean"] == -10


def test_very_large_numbers(fresh_metrics):
    """Test metrics handle very large numbers."""
    collector = fresh_metrics

    large_value = 10**9  # 1 billion
    collector.increment_counter("large_counter", value=large_value)
    collector.set_gauge("large_gauge", large_value)
    collector.observe_histogram("large_hist", large_value)

    snapshot = collector.get_snapshot()
    assert snapshot.counters["large_counter"] == large_value
    assert snapshot.gauges["large_gauge"] == large_value

    stats = collector.get_histogram_stats("large_hist")
    assert stats["mean"] == large_value
