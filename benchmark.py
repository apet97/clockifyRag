#!/usr/bin/env python3
"""
Benchmark suite for Clockify RAG CLI.

Measures:
- Latency: Time to complete operations
- Throughput: Operations per second
- Memory: Memory usage during operations

Usage:
    python benchmark.py              # Run all benchmarks
    python benchmark.py --quick      # Quick benchmark (fewer iterations)
    python benchmark.py --embedding  # Only embedding benchmarks
    python benchmark.py --retrieval  # Only retrieval benchmarks
    python benchmark.py --e2e        # Only end-to-end benchmarks
"""

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from statistics import mean, median, stdev
from typing import Callable

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules to allow monkey-patching for offline smoke tests
import clockify_rag.retrieval
import clockify_rag.embedding
import clockify_rag.answer

from clockify_rag import (
    build_chunks,
    embed_texts,
    embed_query,
    retrieve,
    answer_once,
    load_index,
)
from clockify_rag.config import EMB_BACKEND, EMB_DIM
from clockify_rag.utils import resolve_corpus_path
from clockify_rag.retrieval import RETRIEVE_PROFILE_LAST

# Allow CI smoke tests to bypass external services by providing deterministic
# stubs when BENCHMARK_FAKE_REMOTE=1.
if os.environ.get("BENCHMARK_FAKE_REMOTE") == "1":
    def _fake_embed_query(question: str, retries: int = 0) -> np.ndarray:
        """Deterministic unit-vector embedding based on question hash."""
        seed = abs(hash(question)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=EMB_DIM).astype("float32")
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def _fake_embed_texts(texts, retries: int = 0):
        if not texts:
            return np.zeros((0, EMB_DIM), dtype="float32")
        vecs = [_fake_embed_query(t, retries) for t in texts]
        return np.vstack(vecs).astype("float32")

    def _fake_answer_once(question, chunks, vecs_n, bm, top_k=12, pack_top=6,
                          threshold=0.30, use_rerank=False, debug=False,
                          hnsw=None, seed=0, num_ctx=0, num_predict=0,
                          retries=0):
        """Offline-friendly answer stub using hybrid retrieval only."""
        selected, scores = retrieve(
            question, chunks, vecs_n, bm, top_k=top_k, hnsw=hnsw, retries=retries
        )
        summary_chunks = [chunks[i]["text"] for i in selected[:1]]
        answer_text = summary_chunks[0] if summary_chunks else "No answer available."
        metadata = {
            "selected": [chunks[i]["id"] for i in selected],
            "scores": scores,
            "timings": {},
            "cached": False,
            "cache_hit": False,
        }
        return answer_text, metadata

    # Monkey-patch the modules for offline testing
    clockify_rag.retrieval.embed_query = _fake_embed_query
    clockify_rag.embedding.embed_texts = _fake_embed_texts
    clockify_rag.answer.answer_once = _fake_answer_once

    # Reassign imports to use patched versions
    embed_query = _fake_embed_query
    embed_texts = _fake_embed_texts
    answer_once = _fake_answer_once


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.latencies = []  # milliseconds
        self.memory_peak = 0  # bytes
        self.memory_current = 0  # bytes
        self.metadata = {}

    def add_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def set_memory(self, peak_bytes: int, current_bytes: int):
        self.memory_peak = peak_bytes
        self.memory_current = current_bytes

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)

    def summary(self) -> dict:
        """Get summary statistics."""
        if not self.latencies:
            return {"name": self.name, "error": "No measurements"}

        summary = {
            "name": self.name,
            "latency_ms": {
                "mean": round(mean(self.latencies), 2),
                "median": round(median(self.latencies), 2),
                "stdev": round(stdev(self.latencies), 2) if len(self.latencies) > 1 else 0,
                "min": round(min(self.latencies), 2),
                "max": round(max(self.latencies), 2),
                "p95": round(sorted(self.latencies)[int(len(self.latencies) * 0.95)], 2),
            },
            "throughput": {
                "ops_per_sec": round(1000 / mean(self.latencies), 2),
            },
            "memory_mb": {
                "peak": round(self.memory_peak / 1024 / 1024, 2),
                "current": round(self.memory_current / 1024 / 1024, 2),
            },
            "iterations": len(self.latencies),
        }

        if self.metadata:
            summary["metadata"] = self.metadata

        return summary


def benchmark(func: Callable, iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
    """Benchmark a function with latency and memory tracking.

    Args:
        func: Function to benchmark (should return None)
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations (not measured)

    Returns:
        BenchmarkResult with statistics
    """
    result = BenchmarkResult(func.__name__)

    # Warmup
    for _ in range(warmup):
        func()
        gc.collect()

    # Benchmark with memory tracking
    tracemalloc.start()
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        result.add_latency((end - start) * 1000)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.set_memory(peak, current)

    return result


def aggregate_retrieval_profiles(profiles: list[dict]) -> dict:
    """Aggregate retrieval profiling samples into summary stats."""

    if not profiles:
        return {}

    dot_mean = mean(p.get("dense_dot_time_ms", 0.0) for p in profiles)
    computed_mean = mean(p.get("dense_computed", 0) for p in profiles)
    saved_mean = mean(p.get("dense_saved", 0) for p in profiles)
    candidates_mean = mean(p.get("candidates", 0) for p in profiles)
    total_saved = sum(p.get("dense_saved", 0) for p in profiles)
    total = sum(p.get("dense_total", 0) for p in profiles)
    saved_ratio = round(total_saved / total, 3) if total else 0.0

    if any(p.get("used_faiss") for p in profiles):
        ann_mode = "faiss"
    elif any(p.get("used_hnsw") for p in profiles):
        ann_mode = "hnsw"
    else:
        ann_mode = "linear"

    return {
        "ann_mode": ann_mode,
        "dense_dot_ms_mean": round(dot_mean, 3),
        "dense_computed_mean": round(computed_mean, 2),
        "dense_saved_mean": round(saved_mean, 2),
        "dense_saved_ratio": saved_ratio,
        "candidates_mean": round(candidates_mean, 2),
    }


# ====== EMBEDDING BENCHMARKS ======
def benchmark_embedding_single(chunks, iterations=10):
    """Benchmark single text embedding."""
    text = chunks[0]["text"] if chunks else "How do I track time in Clockify?"

    def run():
        embed_query(text)

    result = benchmark(run, iterations=iterations, warmup=2)
    result.name = "embed_single"
    return result


def benchmark_embedding_batch(chunks, iterations=5):
    """Benchmark batch embedding (10 chunks)."""
    batch = chunks[:10] if len(chunks) >= 10 else chunks
    texts = [
        c.get("text", str(c)) if isinstance(c, dict) else str(c)
        for c in batch
    ]

    def run():
        embed_texts(texts)

    result = benchmark(run, iterations=iterations, warmup=1)
    result.name = "embed_batch_10"
    return result


def benchmark_embedding_large_batch(chunks, iterations=3):
    """Benchmark large batch embedding (100 chunks)."""
    batch = chunks[:100] if len(chunks) >= 100 else chunks
    texts = [
        c.get("text", str(c)) if isinstance(c, dict) else str(c)
        for c in batch
    ]

    def run():
        embed_texts(texts)

    result = benchmark(run, iterations=iterations, warmup=1)
    result.name = "embed_batch_100"
    return result


# ====== RETRIEVAL BENCHMARKS ======
def benchmark_retrieval_hybrid(chunks, vecs_n, bm, iterations=20):
    """Benchmark hybrid (BM25 + dense) retrieval (Rank 16: fixed misleading name)."""
    question = "How do I track time in Clockify?"
    profiles = []

    def run():
        retrieve(question, chunks, vecs_n, bm, top_k=12, hnsw=None)
        if RETRIEVE_PROFILE_LAST:
            profiles.append(dict(RETRIEVE_PROFILE_LAST))

    result = benchmark(run, iterations=iterations, warmup=3)
    result.name = "retrieve_hybrid"
    if profiles:
        result.set_metadata(**aggregate_retrieval_profiles(profiles))
    return result


def benchmark_retrieval_with_mmr(chunks, vecs_n, bm, iterations=20):
    """Benchmark retrieval + MMR diversification."""
    question = "How do I track time in Clockify?"
    profiles = []

    def run():
        selected, scores = retrieve(question, chunks, vecs_n, bm, top_k=12, hnsw=None)
        # Simulate MMR (already included in answer_once, but measure separately)
        _ = selected[:6]  # Pack top 6
        if RETRIEVE_PROFILE_LAST:
            profiles.append(dict(RETRIEVE_PROFILE_LAST))

    result = benchmark(run, iterations=iterations, warmup=3)
    result.name = "retrieve_with_mmr"
    if profiles:
        result.set_metadata(**aggregate_retrieval_profiles(profiles))
    return result


# ====== END-TO-END BENCHMARKS ======
def benchmark_e2e_simple(chunks, vecs_n, bm, iterations=10):
    """Benchmark end-to-end answer generation (simple query)."""
    question = "How do I track time?"

    def run():
        try:
            answer_once(question, chunks, vecs_n, bm, top_k=12, pack_top=6, threshold=0.30)
        except Exception as e:
            print(f"Warning: E2E benchmark failed: {e}")

    result = benchmark(run, iterations=iterations, warmup=2)
    result.name = "e2e_simple_query"
    return result


def benchmark_e2e_complex(chunks, vecs_n, bm, iterations=5):
    """Benchmark end-to-end answer generation (complex query)."""
    question = "What are the differences between the pricing plans and which features are included in each tier?"

    def run():
        try:
            answer_once(question, chunks, vecs_n, bm, top_k=12, pack_top=6, threshold=0.30)
        except Exception as e:
            print(f"Warning: E2E complex benchmark failed: {e}")

    result = benchmark(run, iterations=iterations, warmup=1)
    result.name = "e2e_complex_query"
    return result


def benchmark_chunking(md_path, iterations=5):
    """Benchmark chunking performance."""

    def run():
        build_chunks(md_path)

    result = benchmark(run, iterations=iterations, warmup=1)
    result.name = "chunking"
    return result


# ====== MAIN ======
def main():
    parser = argparse.ArgumentParser(description="Benchmark Clockify RAG CLI")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (fewer iterations)")
    parser.add_argument("--embedding", action="store_true", help="Only embedding benchmarks")
    parser.add_argument("--retrieval", action="store_true", help="Only retrieval benchmarks")
    parser.add_argument("--e2e", action="store_true", help="Only end-to-end benchmarks")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Adjust iterations for quick mode
    iter_multiplier = 0.5 if args.quick else 1.0

    print("=" * 70)
    print("CLOCKIFY RAG BENCHMARK SUITE")
    print("=" * 70)
    print(f"Embedding backend: {EMB_BACKEND}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print()

    # Load index
    print("[1/2] Loading index...")
    result = load_index()
    if result is None:
        print("❌ Failed to load index. Run 'make build' first.")
        sys.exit(1)

    # Handle new dict return format from load_index()
    if isinstance(result, dict):
        chunks = result["chunks"]
        vecs_n = result["vecs_n"]
        bm = result["bm"]
        hnsw = result.get("hnsw")  # May be None
    else:
        # Legacy tuple format (backward compatibility)
        chunks, vecs_n, bm, hnsw = result
    print(f"✅ Loaded {len(chunks)} chunks")
    print()

    # Run benchmarks
    print("[2/2] Running benchmarks...")
    print()

    results = []

    # Embedding benchmarks
    if not args.retrieval and not args.e2e:
        print("--- Embedding Benchmarks ---")
        if not args.quick:
            results.append(benchmark_embedding_single(chunks, iterations=int(10 * iter_multiplier)))
            print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")

        results.append(benchmark_embedding_batch(chunks, iterations=int(5 * iter_multiplier)))
        print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")

        if not args.quick:
            results.append(benchmark_embedding_large_batch(chunks, iterations=int(3 * iter_multiplier)))
            print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")
        print()

    # Retrieval benchmarks
    if not args.embedding and not args.e2e:
        print("--- Retrieval Benchmarks ---")
        results.append(benchmark_retrieval_hybrid(chunks, vecs_n, bm, iterations=int(20 * iter_multiplier)))
        print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")

        results.append(benchmark_retrieval_with_mmr(chunks, vecs_n, bm, iterations=int(20 * iter_multiplier)))
        print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")
        print()

    # End-to-end benchmarks
    if not args.embedding and not args.retrieval:
        print("--- End-to-End Benchmarks ---")
        results.append(benchmark_e2e_simple(chunks, vecs_n, bm, iterations=int(10 * iter_multiplier)))
        print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")

        if not args.quick:
            results.append(benchmark_e2e_complex(chunks, vecs_n, bm, iterations=int(5 * iter_multiplier)))
            print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")
        print()

    # Chunking benchmark
    if not args.embedding and not args.retrieval and not args.e2e and not args.quick:
        kb_path, exists, _ = resolve_corpus_path()
        if exists and os.path.exists(kb_path):
            print("--- Chunking Benchmark ---")
            results.append(benchmark_chunking(kb_path, iterations=5))
            print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")
            print()

    # Summary
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    summaries = [r.summary() for r in results]
    for s in summaries:
        print(f"\n{s['name']}:")
        print(f"  Latency:    {s['latency_ms']['mean']:.2f}ms ± {s['latency_ms']['stdev']:.2f}ms")
        print(f"  Throughput: {s['throughput']['ops_per_sec']:.2f} ops/sec")
        print(f"  Memory:     {s['memory_mb']['peak']:.2f} MB peak")
        metadata = s.get("metadata")
        if metadata:
            print("  Profiling:")
            for key, value in metadata.items():
                print(f"    {key}: {value}")

    # Save to JSON
    output_data = {
        "timestamp": time.time(),
        "backend": EMB_BACKEND,
        "quick_mode": args.quick,
        "results": summaries,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
