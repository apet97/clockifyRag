"""
Tests for metrics tracking and export functionality.

Priority #13: Export KPI metrics (ROI 5/10)
"""

import json
import time
import threading
import pytest

from clockify_rag.metrics import (
    MetricsCollector,
    get_metrics,
    increment_counter,
    set_gauge,
    observe_histogram,
    time_operation,
    MetricNames,
)


class TestMetricsCollector:
    """Test metrics collector basic functionality."""

    def test_counter_increment(self):
        """Test counter increments correctly."""
        collector = MetricsCollector()

        collector.increment_counter("test_counter")
        assert collector.get_counter("test_counter") == 1.0

        collector.increment_counter("test_counter", 5.0)
        assert collector.get_counter("test_counter") == 6.0

    def test_counter_with_labels(self):
        """Test counters with labels are tracked separately."""
        collector = MetricsCollector()

        collector.increment_counter("requests", labels={"status": "200"})
        collector.increment_counter("requests", labels={"status": "404"})
        collector.increment_counter("requests", labels={"status": "200"})

        assert collector.get_counter("requests", {"status": "200"}) == 2.0
        assert collector.get_counter("requests", {"status": "404"}) == 1.0

    def test_gauge_set(self):
        """Test gauge values can be set."""
        collector = MetricsCollector()

        collector.set_gauge("temperature", 23.5)
        assert collector.get_gauge("temperature") == 23.5

        collector.set_gauge("temperature", 24.0)
        assert collector.get_gauge("temperature") == 24.0

    def test_gauge_with_labels(self):
        """Test gauges with labels."""
        collector = MetricsCollector()

        collector.set_gauge("cpu_usage", 45.0, {"core": "0"})
        collector.set_gauge("cpu_usage", 55.0, {"core": "1"})

        assert collector.get_gauge("cpu_usage", {"core": "0"}) == 45.0
        assert collector.get_gauge("cpu_usage", {"core": "1"}) == 55.0

    def test_histogram_observe(self):
        """Test histogram records observations."""
        collector = MetricsCollector()

        for i in range(100):
            collector.observe_histogram("latency", float(i))

        stats = collector.get_histogram_stats("latency")
        assert stats is not None
        assert stats.count == 100
        assert stats.min == 0.0
        assert stats.max == 99.0
        assert 45 < stats.mean < 55  # Mean should be around 49.5
        assert 45 < stats.p50 < 55
        assert stats.p95 >= 90
        assert stats.p99 >= 95

    def test_histogram_with_labels(self):
        """Test histograms with labels."""
        collector = MetricsCollector()

        collector.observe_histogram("request_time", 100, {"method": "GET"})
        collector.observe_histogram("request_time", 200, {"method": "POST"})
        collector.observe_histogram("request_time", 150, {"method": "GET"})

        stats_get = collector.get_histogram_stats("request_time", {"method": "GET"})
        stats_post = collector.get_histogram_stats("request_time", {"method": "POST"})

        assert stats_get.count == 2
        assert stats_get.mean == 125.0
        assert stats_post.count == 1
        assert stats_post.mean == 200.0

    def test_timer_context(self):
        """Test timer context manager."""
        collector = MetricsCollector()

        with collector.time_operation("operation"):
            time.sleep(0.01)  # Sleep 10ms

        stats = collector.get_histogram_stats("operation")
        assert stats is not None
        assert stats.count == 1
        assert stats.mean >= 10.0  # Should be at least 10ms

    def test_reset(self):
        """Test reset clears all metrics."""
        collector = MetricsCollector()

        collector.increment_counter("test")
        collector.set_gauge("gauge", 42.0)
        collector.observe_histogram("hist", 100.0)

        collector.reset()

        assert collector.get_counter("test") == 0.0
        assert collector.get_gauge("gauge") is None
        assert collector.get_histogram_stats("hist") is None


class TestMetricsExport:
    """Test metrics export formats."""

    def test_export_json(self):
        """Test JSON export."""
        collector = MetricsCollector()

        collector.increment_counter("queries", 10)
        collector.set_gauge("cache_size", 1000)
        collector.observe_histogram("latency", 100)
        collector.observe_histogram("latency", 200)

        json_output = collector.export_json()
        data = json.loads(json_output)

        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "counters" in data
        assert "gauges" in data
        assert "histogram_stats" in data

        assert data["counters"]["queries"] == 10
        assert data["gauges"]["cache_size"] == 1000
        assert "latency" in data["histogram_stats"]

    def test_export_json_no_histograms(self):
        """Test JSON export without raw histogram data."""
        collector = MetricsCollector()

        for i in range(1000):
            collector.observe_histogram("large_hist", float(i))

        json_output = collector.export_json(include_histograms=False)
        data = json.loads(json_output)

        assert "histogram_raw" not in data
        assert "histogram_stats" in data

    def test_export_prometheus(self):
        """Test Prometheus export format."""
        collector = MetricsCollector()

        collector.increment_counter("http_requests_total", 42, {"status": "200"})
        collector.set_gauge("active_connections", 15)
        collector.observe_histogram("request_duration_ms", 100)
        collector.observe_histogram("request_duration_ms", 200)

        prom_output = collector.export_prometheus()

        assert "# TYPE http_requests_total counter" in prom_output
        assert 'http_requests_total{status="200"} 42' in prom_output
        assert "# TYPE active_connections gauge" in prom_output
        assert "active_connections 15" in prom_output
        assert "# TYPE request_duration_ms summary" in prom_output
        assert "request_duration_ms_count" in prom_output
        assert "request_duration_ms_sum" in prom_output
        assert 'quantile="0.5"' in prom_output
        assert 'quantile="0.95"' in prom_output
        assert 'quantile="0.99"' in prom_output

    def test_export_csv(self):
        """Test CSV export format."""
        collector = MetricsCollector()

        collector.increment_counter("total_queries", 100)
        collector.set_gauge("memory_usage", 512.5)
        collector.observe_histogram("query_time", 50)

        csv_output = collector.export_csv()
        lines = csv_output.split("\n")

        assert lines[0] == "metric_type,metric_name,labels,value"
        assert any("counter,total_queries" in line for line in lines)
        assert any("gauge,memory_usage" in line for line in lines)
        assert any("histogram_mean,query_time" in line for line in lines)

    def test_get_summary(self):
        """Test summary statistics."""
        collector = MetricsCollector()

        collector.increment_counter("queries_total", 1000)
        collector.increment_counter("cache_hits", 800)
        collector.observe_histogram("retrieval_latency_ms", 50)

        summary = collector.get_summary()

        assert "uptime_seconds" in summary
        assert "total_counters" in summary
        assert "total_gauges" in summary
        assert "total_histograms" in summary
        assert "key_metrics" in summary


class TestThreadSafety:
    """Test thread-safe metric updates."""

    def test_concurrent_counter_updates(self):
        """Test concurrent counter increments are safe."""
        collector = MetricsCollector()
        num_threads = 10
        increments_per_thread = 100

        def worker():
            for _ in range(increments_per_thread):
                collector.increment_counter("concurrent_counter")

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        expected = num_threads * increments_per_thread
        assert collector.get_counter("concurrent_counter") == expected

    def test_concurrent_histogram_updates(self):
        """Test concurrent histogram observations are safe."""
        collector = MetricsCollector()
        num_threads = 10
        observations_per_thread = 100

        def worker(thread_id):
            for i in range(observations_per_thread):
                collector.observe_histogram("concurrent_hist", float(thread_id * 100 + i))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        stats = collector.get_histogram_stats("concurrent_hist")
        expected_count = num_threads * observations_per_thread
        assert stats.count == expected_count

    def test_concurrent_timer_contexts(self):
        """Test concurrent timer contexts are safe."""
        collector = MetricsCollector()
        num_threads = 20

        def worker():
            with collector.time_operation("concurrent_timer"):
                time.sleep(0.001)  # 1ms

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        stats = collector.get_histogram_stats("concurrent_timer")
        assert stats.count == num_threads


class TestGlobalMetrics:
    """Test global metrics convenience functions."""

    def test_global_increment_counter(self):
        """Test global counter increment."""
        metrics = get_metrics()
        initial = metrics.get_counter("global_test_counter")

        increment_counter("global_test_counter")
        assert metrics.get_counter("global_test_counter") == initial + 1

    def test_global_set_gauge(self):
        """Test global gauge set."""
        metrics = get_metrics()

        set_gauge("global_test_gauge", 42.0)
        assert metrics.get_gauge("global_test_gauge") == 42.0

    def test_global_observe_histogram(self):
        """Test global histogram observation."""
        metrics = get_metrics()

        observe_histogram("global_test_histogram", 100.0)
        observe_histogram("global_test_histogram", 200.0)

        stats = metrics.get_histogram_stats("global_test_histogram")
        assert stats is not None
        assert stats.count >= 2  # At least our 2 observations

    def test_global_timer(self):
        """Test global timer context."""
        metrics = get_metrics()

        with time_operation("global_test_timer"):
            time.sleep(0.01)

        stats = metrics.get_histogram_stats("global_test_timer")
        assert stats is not None
        assert stats.mean >= 10.0


class TestMetricNames:
    """Test standard metric name constants."""

    def test_standard_names_defined(self):
        """Test all standard metric names are defined."""
        assert hasattr(MetricNames, "QUERIES_TOTAL")
        assert hasattr(MetricNames, "CACHE_HITS")
        assert hasattr(MetricNames, "CACHE_MISSES")
        assert hasattr(MetricNames, "ERRORS_TOTAL")
        assert hasattr(MetricNames, "QUERY_LATENCY")
        assert hasattr(MetricNames, "RETRIEVAL_LATENCY")
        assert hasattr(MetricNames, "LLM_LATENCY")
        assert hasattr(MetricNames, "INGESTIONS_TOTAL")
        assert hasattr(MetricNames, "INGESTION_LATENCY")
        assert hasattr(MetricNames, "CACHE_SIZE")
        assert hasattr(MetricNames, "INDEX_SIZE")

    def test_use_standard_names(self):
        """Test using standard metric names."""
        collector = MetricsCollector()

        collector.increment_counter(MetricNames.QUERIES_TOTAL)
        collector.increment_counter(MetricNames.CACHE_HITS)
        collector.increment_counter(MetricNames.INGESTIONS_TOTAL)
        collector.observe_histogram(MetricNames.QUERY_LATENCY, 150.0)
        collector.observe_histogram(MetricNames.INGESTION_LATENCY, 500.0)

        assert collector.get_counter(MetricNames.QUERIES_TOTAL) == 1.0
        assert collector.get_counter(MetricNames.CACHE_HITS) == 1.0
        assert collector.get_counter(MetricNames.INGESTIONS_TOTAL) == 1.0

        stats = collector.get_histogram_stats(MetricNames.QUERY_LATENCY)
        assert stats is not None
        assert stats.mean == 150.0
        ingest_stats = collector.get_histogram_stats(MetricNames.INGESTION_LATENCY)
        assert ingest_stats is not None
        assert ingest_stats.mean == 500.0


class TestHistogramMaxHistory:
    """Test histogram max history limit."""

    def test_histogram_respects_max_history(self):
        """Test histogram doesn't grow unbounded."""
        collector = MetricsCollector(max_history=100)

        # Add 200 observations
        for i in range(200):
            collector.observe_histogram("limited_hist", float(i))

        stats = collector.get_histogram_stats("limited_hist")
        # Should only keep last 100
        assert stats.count == 100
        # Values should be 100-199
        assert stats.min == 100.0
        assert stats.max == 199.0


class TestLabeledMetricsExport:
    """Test fixes for labeled metrics export issues."""

    def test_prometheus_exports_all_labeled_histogram_series(self):
        """Test that Prometheus export includes all histogram label variants.

        Issue: Prometheus histogram export was skipping every histogram after
        the first one that shared a metric name, dropping labeled series.
        Fix: Track TYPE declarations separately from metric output.
        """
        collector = MetricsCollector()

        # Create histograms with same name but different labels
        collector.observe_histogram("request_duration", 100, {"endpoint": "chat"})
        collector.observe_histogram("request_duration", 150, {"endpoint": "chat"})
        collector.observe_histogram("request_duration", 50, {"endpoint": "search"})
        collector.observe_histogram("request_duration", 75, {"endpoint": "search"})
        collector.observe_histogram("request_duration", 200, {"endpoint": "admin"})

        prom_output = collector.export_prometheus()

        # Should have TYPE declaration exactly once
        assert prom_output.count("# TYPE request_duration summary") == 1

        # Should have all three labeled series
        assert 'request_duration_count{endpoint="chat"}' in prom_output
        assert 'request_duration_count{endpoint="search"}' in prom_output
        assert 'request_duration_count{endpoint="admin"}' in prom_output

        # Verify counts are correct
        assert 'request_duration_count{endpoint="chat"} 2' in prom_output
        assert 'request_duration_count{endpoint="search"} 2' in prom_output
        assert 'request_duration_count{endpoint="admin"} 1' in prom_output

        # Verify quantiles exist for all label sets
        assert 'request_duration{quantile="0.5",endpoint="chat"}' in prom_output
        assert 'request_duration{quantile="0.5",endpoint="search"}' in prom_output
        assert 'request_duration{quantile="0.5",endpoint="admin"}' in prom_output

    def test_get_summary_aggregates_labeled_counters(self):
        """Test that get_summary aggregates counters across all label variants.

        Issue: get_summary only checked for exact key matches, missing labeled metrics.
        Fix: Aggregate all counter values for matching metric names.
        """
        collector = MetricsCollector()

        # Create counters with same name but different labels
        collector.increment_counter("queries_total", 100, {"endpoint": "chat"})
        collector.increment_counter("queries_total", 200, {"endpoint": "search"})
        collector.increment_counter("queries_total", 50, {"endpoint": "admin"})

        summary = collector.get_summary()

        # Should aggregate all variants
        assert "queries_total" in summary["key_metrics"]
        assert summary["key_metrics"]["queries_total"] == 350.0

    def test_get_summary_aggregates_labeled_histograms(self):
        """Test that get_summary aggregates histograms across all label variants.

        Issue: get_summary only checked for exact key matches, missing labeled metrics.
        Fix: Collect all histogram observations for matching metric names.
        """
        collector = MetricsCollector()

        # Create histograms with same name but different labels
        collector.observe_histogram("retrieval_latency_ms", 100, {"method": "bm25"})
        collector.observe_histogram("retrieval_latency_ms", 200, {"method": "bm25"})
        collector.observe_histogram("retrieval_latency_ms", 50, {"method": "faiss"})
        collector.observe_histogram("retrieval_latency_ms", 75, {"method": "faiss"})

        summary = collector.get_summary()

        # Should aggregate all observations
        assert "retrieval_latency_ms" in summary["key_metrics"]
        metric_data = summary["key_metrics"]["retrieval_latency_ms"]

        assert metric_data["count"] == 4
        # Mean should be (100 + 200 + 50 + 75) / 4 = 106.25
        assert abs(metric_data["mean"] - 106.25) < 0.01
        # p95 should be 200 (sorted: [50, 75, 100, 200])
        assert metric_data["p95"] == 200

    def test_get_summary_prioritizes_counters_over_histograms(self):
        """Test that get_summary returns counter if both counter and histogram exist."""
        collector = MetricsCollector()

        # Create both counter and histogram with same name
        collector.increment_counter("queries_total", 500)
        collector.observe_histogram("queries_total", 100)

        summary = collector.get_summary()

        # Should return counter value, not histogram
        assert summary["key_metrics"]["queries_total"] == 500.0

    def test_prometheus_mixed_labeled_and_unlabeled_histograms(self):
        """Test Prometheus export handles mix of labeled and unlabeled histograms."""
        collector = MetricsCollector()

        # Mix of labeled and unlabeled
        collector.observe_histogram("latency", 100)
        collector.observe_histogram("latency", 200)
        collector.observe_histogram("latency", 50, {"region": "us-east"})
        collector.observe_histogram("latency", 75, {"region": "eu-west"})

        prom_output = collector.export_prometheus()

        # Should have TYPE declaration exactly once
        assert prom_output.count("# TYPE latency summary") == 1

        # Should have all three series (unlabeled + 2 labeled)
        assert "latency_count 2" in prom_output  # Unlabeled
        assert 'latency_count{region="us-east"} 1' in prom_output
        assert 'latency_count{region="eu-west"} 1' in prom_output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
