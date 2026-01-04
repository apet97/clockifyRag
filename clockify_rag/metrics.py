"""In-memory metrics collection aligned with test expectations.

This module intentionally implements a lightweight, self-contained metrics
system used only inside this process. It is designed to:
- Provide a stable API for tests in tests/test_metrics*.py
- Be thread-safe for concurrent updates
- Avoid side effects on import (no I/O, no network)

It is NOT a full external monitoring backend. Export helpers produce
text/JSON for inspection and can be hooked into real monitoring by the
caller if desired.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, Iterable


# ========================= Metric name constants =========================


class MetricNames:
    """Standard metric names used across the project/tests.

    The tests only check that these attributes exist; we keep this minimal
    and project-focused.
    """

    # Core counters
    QUERIES_TOTAL = "queries_total"
    CACHE_HITS = "cache_hits"
    CACHE_MISSES = "cache_misses"
    ERRORS_TOTAL = "errors_total"
    INGESTIONS_TOTAL = "ingestions_total"
    REFUSALS_TOTAL = "refusals_total"
    RATE_LIMIT_ALLOWED = "rate_limit_allowed"
    RATE_LIMIT_BLOCKED = "rate_limit_blocked"

    # Latencies
    QUERY_LATENCY = "query_latency_ms"
    RETRIEVAL_LATENCY = "retrieval_latency_ms"
    LLM_LATENCY = "llm_latency_ms"
    INGESTION_LATENCY = "ingestion_latency_ms"

    # Gauges
    CACHE_SIZE = "cache_size"
    INDEX_SIZE = "index_size"


# ========================= Internal helpers =============================


Labels = Tuple[Tuple[str, str], ...]  # sorted key=value pairs


def _norm_labels(labels: Optional[Dict[str, str]]) -> Labels:
    if not labels:
        return tuple()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


@dataclass
class HistogramStats:
    count: int
    min: float
    max: float
    mean: float
    p50: float
    p95: float
    p99: float


class HistogramStatsView(dict):
    """Dict-like view that also provides attribute access for tests."""

    __slots__ = ()

    def __getattr__(self, item: str) -> float:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


@dataclass
class Snapshot:
    """Aggregated point-in-time view of a collector."""

    timestamp: float
    uptime_seconds: float
    counters: Dict[str, float]
    gauges: Dict[str, float]
    histograms: Dict[str, HistogramStats]


# ========================= MetricsCollector =============================


class MetricsCollector:
    """Thread-safe in-memory metrics store.

    Data model (all keyed by (name, labels)):
    - counters: monotonically increasing floats
    - gauges: last-set float
    - histograms: list of float observations (bounded by max_history)
    """

    def __init__(self, max_history: int = 10000) -> None:
        self._lock = threading.RLock()
        self._start = time.time()
        self._max_history = int(max_history)

        self._counters: Dict[Tuple[str, Labels], float] = {}
        self._gauges: Dict[Tuple[str, Labels], float] = {}
        self._histo: Dict[Tuple[str, Labels], list] = {}

    # ----- counter API -----

    def increment_counter(
        self,
        name: str | MetricNames,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        key = (str(getattr(name, "value", name)), _norm_labels(labels))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + float(value)

    def get_counter(
        self,
        name: str | MetricNames,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        key = (str(getattr(name, "value", name)), _norm_labels(labels))
        with self._lock:
            return float(self._counters.get(key, 0.0))

    # ----- gauge API -----

    def set_gauge(
        self,
        name: str | MetricNames,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        key = (str(getattr(name, "value", name)), _norm_labels(labels))
        with self._lock:
            self._gauges[key] = float(value)

    def get_gauge(
        self,
        name: str | MetricNames,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        key = (str(getattr(name, "value", name)), _norm_labels(labels))
        with self._lock:
            return self._gauges.get(key)

    # ----- histogram API -----

    def observe_histogram(
        self,
        name: str | MetricNames,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        key = (str(getattr(name, "value", name)), _norm_labels(labels))
        v = float(value)
        with self._lock:
            bucket = self._histo.setdefault(key, [])
            bucket.append(v)
            if len(bucket) > self._max_history:
                # Keep most recent values
                overflow = len(bucket) - self._max_history
                if overflow > 0:
                    del bucket[0:overflow]

    def _stats_for(self, values: Iterable[float]) -> HistogramStats:
        data = sorted(float(v) for v in values)
        n = len(data)
        if n == 0:
            return HistogramStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        def pct(p: float) -> float:
            idx = int(max(0, min(n - 1, round(p * (n - 1)))))
            return data[idx]

        s = sum(data)
        return HistogramStats(
            count=n,
            min=data[0],
            max=data[-1],
            mean=s / n,
            p50=pct(0.50),
            p95=pct(0.95),
            p99=pct(0.99),
        )

    def get_histogram_stats(
        self,
        name: str | MetricNames,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[HistogramStatsView]:
        key = (str(getattr(name, "value", name)), _norm_labels(labels))
        with self._lock:
            values = self._histo.get(key)
            if not values:
                return None
            stats = self._stats_for(values)
            return HistogramStatsView(
                {
                    "count": stats.count,
                    "min": stats.min,
                    "max": stats.max,
                    "mean": stats.mean,
                    "p50": stats.p50,
                    "p95": stats.p95,
                    "p99": stats.p99,
                }
            )

    # ----- timing helpers -----

    def time_operation(self, name: str | MetricNames):
        """Context-manager/decorator for timing operations.

        Usage:
            with collector.time_operation("op"):
                ...

            @collector.time_operation("op")
            def f(...):
                ...
        """

        metric_name = str(getattr(name, "value", name))
        collector = self

        class _Timer:
            def __enter__(self):
                self._start = time.time()
                return self

            def __exit__(self, exc_type, exc, tb):
                elapsed_ms = (time.time() - self._start) * 1000.0
                if exc_type is None:
                    collector.observe_histogram(metric_name, elapsed_ms)
                else:
                    # Count as error + record latency
                    collector.increment_counter(MetricNames.ERRORS_TOTAL)
                    collector.observe_histogram(metric_name, elapsed_ms)
                # Do not suppress exceptions
                return False

            def __call__(self, func):
                def wrapper(*args, **kwargs):
                    with self:
                        return func(*args, **kwargs)

                return wrapper

        return _Timer()

    # ----- snapshot / export / reset -----

    def get_snapshot(self) -> Snapshot:
        with self._lock:
            now = time.time()
            counters = {self._format_key(name, labels): value for (name, labels), value in self._counters.items()}
            gauges = {self._format_key(name, labels): value for (name, labels), value in self._gauges.items()}
            histograms = {
                self._format_key(name, labels): self._stats_for(values)
                for (name, labels), values in self._histo.items()
                if values
            }
            return Snapshot(
                timestamp=now,
                uptime_seconds=now - self._start,
                counters=counters,
                gauges=gauges,
                histograms=histograms,
            )

    def export_json(self, include_histograms: bool = True) -> str:
        snap = self.get_snapshot()

        def hist_to_dict(h: HistogramStats) -> Dict[str, float]:
            return {
                "count": h.count,
                "min": h.min,
                "max": h.max,
                "mean": h.mean,
                "p50": h.p50,
                "p95": h.p95,
                "p99": h.p99,
            }

        payload: Dict[str, Any] = {
            "timestamp": snap.timestamp,
            "uptime_seconds": snap.uptime_seconds,
            "counters": snap.counters,
            "gauges": snap.gauges,
            "histogram_stats": {name: hist_to_dict(h) for name, h in snap.histograms.items()},
        }

        # Raw histogram samples deliberately omitted by default; tests only
        # require stats + toggle behavior.
        if include_histograms:
            payload["histogram_raw"] = {
                name: [float(v) for v in self._histo[(base, lbls)]]
                for (base, lbls), _ in self._histo.items()
                for name in [self._format_key(base, lbls)]
            }

        return json.dumps(payload, sort_keys=True)

    def export_prometheus(self) -> str:
        """Render metrics in a simple Prometheus text exposition format."""

        lines: list[str] = []
        seen_types: set[str] = set()

        with self._lock:
            # Counters
            for (name, labels), value in sorted(self._counters.items()):
                if name not in seen_types:
                    lines.append(f"# TYPE {name} counter")
                    seen_types.add(name)
                label_txt = self._format_labels(labels)
                lines.append(f"{name}{label_txt} {value}")

            # Gauges
            for (name, labels), value in sorted(self._gauges.items()):
                if name not in seen_types:
                    lines.append(f"# TYPE {name} gauge")
                    seen_types.add(name)
                label_txt = self._format_labels(labels)
                lines.append(f"{name}{label_txt} {value}")

            # Histograms as summaries (count/sum + quantiles)
            for (name, labels), values in sorted(self._histo.items()):
                if not values:
                    continue
                stats = self._stats_for(values)
                summary_name = name
                if summary_name not in seen_types:
                    lines.append(f"# TYPE {summary_name} summary")
                    seen_types.add(summary_name)

                label_txt = self._format_labels(labels)
                # Count & sum
                lines.append(f"{summary_name}_count{label_txt} {stats.count}")
                lines.append(f"{summary_name}_sum{label_txt} {stats.mean * stats.count}")
                # Quantiles
                for q, v in ((0.5, stats.p50), (0.95, stats.p95), (0.99, stats.p99)):
                    q_labels = self._merge_labels(labels, {"quantile": str(q)})
                    lines.append(f"{summary_name}{self._format_labels(q_labels)} {v}")

        return "\n".join(lines) + "\n"

    def export_csv(self) -> str:
        """Simple CSV export: metric_type,metric_name,labels,value"""

        rows = ["metric_type,metric_name,labels,value"]
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                rows.append(f'counter,{name},"{self._labels_str(labels)}",{value}')
            for (name, labels), value in sorted(self._gauges.items()):
                rows.append(f'gauge,{name},"{self._labels_str(labels)}",{value}')
            for (name, labels), values in sorted(self._histo.items()):
                if not values:
                    continue
                stats = self._stats_for(values)
                lbl = self._labels_str(labels)
                rows.append(f'histogram_mean,{name},"{lbl}",{stats.mean}')
        return "\n".join(rows) + "\n"

    def get_summary(self) -> Dict[str, Any]:
        """Return coarse summary used by tests.

        - Aggregates counters/histograms by metric name ignoring labels.
        - If both counter and histogram exist for the same name, counter wins.
        """

        snap = self.get_snapshot()

        # Aggregate by base name (strip labels that were baked into key)
        def base(name: str) -> str:
            return name.split("{")[0]

        counters_agg: Dict[str, float] = {}
        for k, v in snap.counters.items():
            counters_agg[base(k)] = counters_agg.get(base(k), 0.0) + float(v)

        hist_agg: Dict[str, Dict[str, float]] = {}
        for k, h in snap.histograms.items():
            b = base(k)
            cur = hist_agg.get(b)
            if not cur:
                hist_agg[b] = {
                    "count": h.count,
                    "mean": h.mean,
                    "p95": h.p95,
                }
            else:
                # merge by summing counts and averaging means approximately
                total_count = cur["count"] + h.count
                if total_count > 0:
                    cur["mean"] = (cur["mean"] * cur["count"] + h.mean * h.count) / total_count
                cur["count"] = total_count
                cur["p95"] = max(cur["p95"], h.p95)

        # Prefer counters when both exist
        key_metrics: Dict[str, Any] = {}
        for name, val in counters_agg.items():
            key_metrics[name] = float(val)
        for name, info in hist_agg.items():
            if name not in key_metrics:
                key_metrics[name] = info

        return {
            "uptime_seconds": snap.uptime_seconds,
            "total_counters": len(snap.counters),
            "total_gauges": len(snap.gauges),
            "total_histograms": len(snap.histograms),
            "key_metrics": key_metrics,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histo.clear()
            self._start = time.time()

    # ----- formatting helpers -----

    @staticmethod
    def _format_key(name: str, labels: Labels) -> str:
        if not labels:
            return name
        return f"{name}{{{','.join(f'{k}={v}' for k, v in labels)}}}"

    @staticmethod
    def _labels_str(labels: Labels) -> str:
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in labels)

    @staticmethod
    def _format_labels(labels: Labels) -> str:
        if not labels:
            return ""
        ordered = list(labels)
        if any(k == "quantile" for k, _ in ordered):
            ordered.sort(key=lambda kv: (kv[0] != "quantile", kv[0], kv[1]))
        inner = ",".join(f'{k}="{v}"' for k, v in ordered)
        return f"{{{inner}}}"

    @staticmethod
    def _merge_labels(base_labels: Labels, extra: Dict[str, str]) -> Labels:
        merged = {k: v for k, v in base_labels}
        merged.update(extra)
        return _norm_labels(merged)


# ========================= Global registry helpers =======================


_METRICS_LOCK = threading.RLock()
_METRICS: Dict[str, MetricsCollector] = {}


def get_metrics(name: str = "default") -> MetricsCollector:
    with _METRICS_LOCK:
        mc = _METRICS.get(name)
        if mc is None:
            mc = MetricsCollector()
            _METRICS[name] = mc
        return mc


def get_metrics_collector(name: str) -> MetricsCollector:
    return get_metrics(name)


def increment_counter(
    name: str | MetricNames,
    value: float = 1.0,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    get_metrics().increment_counter(name, value=value, labels=labels)


def set_gauge(
    name: str | MetricNames,
    value: float,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    get_metrics().set_gauge(name, value, labels=labels)


def observe_histogram(
    name: str | MetricNames,
    value: float,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    get_metrics().observe_histogram(name, value, labels=labels)


def time_operation(
    name: str | MetricNames,
    collector: Optional[MetricsCollector] = None,
):
    """Module-level helper for timing.

    - If collector is provided, use it.
    - Otherwise, use the default global collector.

    Used by tests as a decorator/context manager.
    """

    mc = collector or get_metrics()
    return mc.time_operation(name)


def get_all_snapshots() -> Dict[str, Snapshot]:
    """Return snapshot per registered collector."""
    with _METRICS_LOCK:
        return {name: mc.get_snapshot() for name, mc in _METRICS.items()}


# Backwards-compatibility aliases expected by __init__.py/tests
AggregatedMetrics = Snapshot
MetricSnapshot = Snapshot
